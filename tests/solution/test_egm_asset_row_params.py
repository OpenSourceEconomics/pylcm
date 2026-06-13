"""DC-EGM asset-row savings-stage functions reading model params.

In asset-row mode (any savings-stage function reads the current Euler state),
the per-exogenous-asset-node solve must evaluate the child `resources`
function and the regime-transition probabilities with the regime's model
params, the lifecycle `age`/`period`, and any solve-phase imputed
intermediates — exactly the bindings the brute-force solver supplies. The
asset derivative of a param-dependent, Euler-state-dependent read enters the
published marginal value $dV/dR$ through the continuation's direct Euler-state
channel.

The oracle for the solved properties is a dense-grid brute-force solve of a
mathematically equivalent spec (same state grids, dense consumption grid,
explicit budget constraint).
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, BruteForce
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal regime.
N_PERIODS = 4

# Wealth band over which the savings-stage smoothstep ramps. At many wealth
# cells the band is well resolved at node resolution, so the build-time
# continuity check admits it.
BAND_START = 30.0
BAND_WIDTH = 20.0

# Survival probability below/above the wealth band. Death is absorbing with
# zero value while continued life is worth several utils, so the value gap
# across targets is material and the probability's wealth slope contributes
# a first-order term to the marginal value inside the band.
SURVIVAL_LOW = 0.55
SURVIVAL_HIGH = 0.95

# Rate of return on wealth — the model param the child resources function and
# the means-test intermediate read.
RATE_OF_RETURN = 0.04

# Deterministic labor income added to savings in the wealth law; keeps child
# wealth queries away from the grid's lower edge, where the two solvers clamp
# differently.
LABOR_INCOME = 5.0

# AIME accrual rate mapping the imputed pension wealth into resources.
PENSION_ACCRUAL = 0.3

# 160 wealth nodes: the brute-force oracle interpolates next-period V linearly
# in wealth (a downward bias where V curves), while the DC-EGM carry read uses
# exact slopes; the dense grid keeps that oracle-side bias below the tolerance.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)
AIME_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=7)

# The consumption grid covers the maximum resources so the brute-force oracle
# is not artificially capped at high wealth; 4000 points keep the oracle's own
# resolution error below the comparison tolerance.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=120.0, n_points=4000)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes excluded from the brute-force comparison: there the
# brute solver leans on consumption choices near its grid start and on coarse
# interpolation where log utility curves hardest.
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class AssetRowRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


def smoothstep_in_band(countable: FloatND) -> FloatND:
    """C² quintic smoothstep rising from 0 to 1 across the band."""
    t = jnp.clip((countable - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_dcegm(savings: FloatND) -> ContinuousState:
    return savings + LABOR_INCOME


def next_wealth_brute(
    resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    return resources - consumption + LABOR_INCOME


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        AssetRowRegimeId.dead,
        AssetRowRegimeId.working_life,
    )


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


def _active(age: int) -> bool:
    last_age = 40 + (N_PERIODS - 1) * 10
    return age < last_age


def _params() -> dict:
    return {
        "discount_factor": 0.95,
        "final_age_alive": 40 + (N_PERIODS - 2) * 10,
        "rate_of_return": RATE_OF_RETURN,
    }


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _assert_working_life_V_matches(
    *, dcegm_solution: PeriodToRegimeToVArr, brute_solution: PeriodToRegimeToVArr
) -> None:
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape
        flat_dcegm = dcegm_V.reshape(-1, dcegm_V.shape[-1])
        flat_brute = brute_V.reshape(-1, brute_V.shape[-1])
        np.testing.assert_allclose(
            flat_dcegm[:, N_BRUTE_UNSTABLE_NODES:],
            flat_brute[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


# --- Slice 1: child resources reads a model param --------------------------


def resources_with_rate(wealth: ContinuousState, rate_of_return: float) -> FloatND:
    """Cash-on-hand including a capital-income return on current wealth."""
    return wealth * (1.0 + rate_of_return)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep_in_band(wealth)


def stay_prob_wealth(
    wealth: ContinuousState, age: int, final_age_alive: float
) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob_wealth(
    wealth: ContinuousState, age: int, final_age_alive: float
) -> FloatND:
    return 1.0 - stay_prob_wealth(wealth, age, final_age_alive)


def budget_constraint_rate(
    consumption: ContinuousAction, wealth: ContinuousState, rate_of_return: float
) -> BoolND:
    return consumption <= wealth * (1.0 + rate_of_return)


@functools.cache
def _resources_param_model(solver: str) -> Model:
    """Self-targeting regime whose resources reads `rate_of_return`."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob_wealth),
            "dead": MarkovTransition(death_prob_wealth),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints=({} if is_dcegm else {"budget_constraint": budget_constraint_rate}),
        functions=(
            {
                "utility": utility,
                "resources": resources_with_rate,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources_with_rate}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=AssetRowRegimeId,
    )


def test_child_resources_reading_param_matches_brute_force():
    """A child resources term `wealth * (1 + rate_of_return)` matches brute.

    The regime is solved per exogenous asset node because the survival
    probability reads wealth. At each node the child's resources query depends
    on the model param `rate_of_return`, so the per-node resources evaluation
    must receive it; values agree with the dense-grid brute-force oracle.
    """
    params = _params()
    dcegm_solution = _resources_param_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _resources_param_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


# --- Slice 2: regime-transition prob reads a param-dependent intermediate ---


def capital_income(wealth: ContinuousState, rate_of_return: float) -> FloatND:
    return rate_of_return * wealth


def countable_income(capital_income: FloatND) -> FloatND:
    return capital_income


def share_of_income(countable_income: FloatND) -> FloatND:
    return smoothstep_in_band(countable_income / RATE_OF_RETURN)


def survival_of_share(share_of_income: FloatND) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * share_of_income


def stay_prob_share(
    share_of_income: FloatND, age: int, final_age_alive: float
) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_share(share_of_income))


def death_prob_share(
    share_of_income: FloatND, age: int, final_age_alive: float
) -> FloatND:
    return 1.0 - stay_prob_share(share_of_income, age, final_age_alive)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


@functools.cache
def _smoothstep_intermediate_model(solver: str, *, rate_is_fixed: bool) -> Model:
    """Survival probability reading wealth through a param-dependent chain.

    When `rate_is_fixed`, `rate_of_return` is supplied through `fixed_params`
    (partialled at model build and dropped from the live template) rather than
    as a free solve param, so the prebuilt asset-row kernel must carry the
    partialled qualified param `capital_income__rate_of_return` into the
    per-node regime-transition-probability evaluation.
    """
    is_dcegm = solver == "dcegm"
    intermediates = {
        "capital_income": capital_income,
        "countable_income": countable_income,
        "share_of_income": share_of_income,
    }
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob_share),
            "dead": MarkovTransition(death_prob_share),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions=(
            {
                "utility": utility,
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
                **intermediates,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources, **intermediates}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    fixed_params = (
        {"working_life": {"capital_income": {"rate_of_return": RATE_OF_RETURN}}}
        if rate_is_fixed
        else {}
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=AssetRowRegimeId,
        fixed_params=fixed_params,
    )


@pytest.mark.parametrize("rate_is_fixed", [False, True])
def test_regime_prob_reading_param_intermediate_matches_brute_force(
    rate_is_fixed: bool,  # noqa: FBT001
):
    """A survival probability `share <- countable <- capital_income(w, r)` matches.

    The stay probability reads wealth through a param-dependent intermediate
    chain (the SSI-smoothstep shape), so the regime is solved per exogenous
    asset node. The probability's wealth slope carries the first-order term
    $\\partial P_{stay}/\\partial wealth \\cdot EV_{stay}$ into the marginal
    value, evaluated with the model param — whether `rate_of_return` is a free
    solve param or supplied through `fixed_params` (and thus partialled into
    the prebuilt asset-row kernel). Values agree with the brute oracle.
    """
    params = _params()
    if rate_is_fixed:
        del params["rate_of_return"]
    dcegm_solution = _smoothstep_intermediate_model(
        "dcegm", rate_is_fixed=rate_is_fixed
    ).solve(params=params, log_level="debug")
    brute_solution = _smoothstep_intermediate_model(
        "brute_force", rate_is_fixed=rate_is_fixed
    ).solve(params=params, log_level="debug")
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


# --- Slice 3: Phased-imputed intermediate feeding resources ----------------


PENSION_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=4)


def next_aime(aime: ContinuousState) -> ContinuousState:
    return aime


def pension_wealth_imputed(
    aime: ContinuousState, rate_of_return: float
) -> ContinuousState:
    """Solve-phase pension wealth imputed from AIME and the return param."""
    return aime * (1.0 + rate_of_return)


def next_pension_wealth(pension_wealth: ContinuousState) -> ContinuousState:
    return pension_wealth


def resources_with_pension(
    wealth: ContinuousState, pension_wealth: ContinuousState
) -> FloatND:
    return wealth + PENSION_ACCRUAL * pension_wealth


def resources_with_pension_brute(
    wealth: ContinuousState, aime: ContinuousState, rate_of_return: float
) -> FloatND:
    return wealth + PENSION_ACCRUAL * (aime * (1.0 + rate_of_return))


def budget_constraint_pension(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    aime: ContinuousState,
    rate_of_return: float,
) -> BoolND:
    return consumption <= wealth + PENSION_ACCRUAL * (aime * (1.0 + rate_of_return))


@functools.cache
def _imputed_pension_model(solver: str) -> Model:
    """Resources reads a solve-phase imputed pension wealth (from AIME)."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob_wealth),
            "dead": MarkovTransition(death_prob_wealth),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states=(
            {
                "wealth": WEALTH_GRID,
                "aime": AIME_GRID,
                "pension_wealth": Phased(
                    solve=pension_wealth_imputed, simulate=PENSION_GRID
                ),
            }
            if is_dcegm
            else {"wealth": WEALTH_GRID, "aime": AIME_GRID}
        ),
        state_transitions=(
            {
                "wealth": next_wealth_dcegm,
                "aime": next_aime,
                "pension_wealth": next_pension_wealth,
            }
            if is_dcegm
            else {"wealth": next_wealth_brute, "aime": next_aime}
        ),
        constraints=(
            {} if is_dcegm else {"budget_constraint": budget_constraint_pension}
        ),
        functions=(
            {
                "utility": utility,
                "resources": resources_with_pension,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources_with_pension_brute}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=AssetRowRegimeId,
    )


def test_imputed_pension_wealth_feeding_resources_matches_brute_force():
    """Resources reading a solve-imputed `pension_wealth(aime, r)` matches brute.

    `pension_wealth` is a solve-phase derived function (imputed from the
    passive AIME state and the return param), not a grid axis. The per-node
    resources DAG computes it from AIME and params rather than demanding it as
    a leaf. Values agree with the dense-grid brute-force oracle that inlines
    the same imputation.
    """
    params = _params()
    dcegm_solution = _imputed_pension_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _imputed_pension_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )
