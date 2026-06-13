"""DC-EGM passive continuous state carried through asset-row mode.

The ACA living-regime shape carries two continuous states — the Euler state
`wealth` (assets) and a passive `aime`-like state whose deterministic
transition reads a discrete action and a stochastic process node, independent
of the continuous action and the savings node — while a savings-stage function
(the survival probability) reads the Euler state, so the regime solves per
exogenous asset node *and* carries the passive axis and a process axis
simultaneously. The passive read blends the two neighboring nodes of the
child's passive grid, the process node is integrated with its intrinsic
weights, and each asset row publishes its own node.

The oracle is a dense-grid brute-force solve of a mathematically equivalent
spec (same wealth, AIME, and income grids, dense consumption grid, explicit
budget constraint).
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, BruteForce
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

N_PERIODS = 4
N_INCOME_NODES = 5

# Wealth band over which the survival smoothstep ramps; resolved across many
# wealth cells so the build-time continuity check admits it.
BAND_START = 30.0
BAND_WIDTH = 30.0
SURVIVAL_LOW = 0.6
SURVIVAL_HIGH = 0.95

# AIME accrual per period of work, scaled by the income node; lands off the
# AIME node spacing so the mixed passive read is exercised. Capped at the top
# AIME node.
AIME_GAIN = 1.3
AIME_MAX = 20.0

# Pension annuity drawn from accumulated AIME, added to next wealth.
PENSION_RATE = 0.2

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=120)
AIME_GRID = LinSpacedGrid(start=0.0, stop=AIME_MAX, n_points=6)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=120.0, n_points=600)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 119) ** 3 for i in range(120)))

N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class PassiveAssetRowRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class LaborChoice:
    work: ScalarInt
    rest: ScalarInt


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborChoice.work


def smoothstep_in_band(wealth: ContinuousState) -> FloatND:
    t = jnp.clip((wealth - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep_in_band(wealth)


def stay_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth, age, final_age_alive)


def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - jnp.where(is_working, disutility_of_work, 0.0)


def labor_income(is_working: BoolND, income: ContinuousState, wage: float) -> FloatND:
    return jnp.where(is_working, wage * jnp.exp(income), 0.0)


def pension_income(aime: ContinuousState) -> FloatND:
    return PENSION_RATE * aime


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_aime(
    aime: ContinuousState, is_working: BoolND, income: ContinuousState
) -> ContinuousState:
    """Passive AIME accrual: reads the work choice and the income node only."""
    return jnp.minimum(
        aime + jnp.where(is_working, AIME_GAIN * jnp.exp(income), 0.0), AIME_MAX
    )


def next_wealth_dcegm(
    savings: FloatND, labor_income: FloatND, pension_income: FloatND
) -> ContinuousState:
    return savings + labor_income + pension_income


def next_wealth_brute(
    resources: FloatND,
    consumption: ContinuousAction,
    labor_income: FloatND,
    pension_income: FloatND,
) -> ContinuousState:
    return resources - consumption + labor_income + pension_income


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


def _active(age: int) -> bool:
    return age < 40 + (N_PERIODS - 1) * 10


def _shared_functions() -> dict:
    return {
        "utility": utility,
        "labor_income": labor_income,
        "pension_income": pension_income,
        "is_working": is_working,
    }


@functools.cache
def _model(solver: str) -> Model:
    """Euler `wealth` + passive `aime` + process `income`, asset-row mode."""
    is_dcegm = solver == "dcegm"
    states = {
        "wealth": WEALTH_GRID,
        "aime": AIME_GRID,
        "income": RouwenhorstAR1Process(n_points=N_INCOME_NODES),
    }
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=_active,
        actions={
            "labor_supply": DiscreteGrid(LaborChoice),
            "consumption": CONSUMPTION_GRID,
        },
        states=states,
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "aime": next_aime,
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions=(
            {
                **_shared_functions(),
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {**_shared_functions(), "resources": resources}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=PassiveAssetRowRegimeId,
    )


def _params() -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    return {
        "discount_factor": 0.95,
        "final_age_alive": final_age_alive,
        "working_life": {
            "income": {"mu": 0.0, "sigma": 0.2, "rho": 0.6},
            "labor_income": {"wage": 12.0},
            "utility": {"disutility_of_work": 0.3},
        },
    }


def test_passive_aime_through_asset_row_matches_brute_force():
    """A passive AIME state carried through asset-row mode matches brute force.

    The regime solves per exogenous asset node (the survival probability reads
    wealth) while carrying the passive AIME axis and the income process axis.
    AIME accrues off its node spacing from the work choice and the income node,
    so the mixed passive read is exercised, and the accumulated AIME feeds next
    wealth through the pension annuity. Values agree with the dense-grid
    brute-force oracle across the full wealth-by-AIME-by-income grid.
    """
    params = _params()
    dcegm_solution: PeriodToRegimeToVArr = _model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution: PeriodToRegimeToVArr = _model("brute_force").solve(
        params=params, log_level="debug"
    )
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape
        # V leads with the income node and AIME axes; wealth is the trailing
        # (Euler) axis. Exclude the lowest wealth nodes, where the coarse
        # consumption grid makes the brute oracle itself unreliable.
        flat_dcegm = np.moveaxis(dcegm_V, _euler_axis(dcegm_V), -1).reshape(
            -1, dcegm_V.shape[_euler_axis(dcegm_V)]
        )
        flat_brute = np.moveaxis(brute_V, _euler_axis(brute_V), -1).reshape(
            -1, brute_V.shape[_euler_axis(brute_V)]
        )
        np.testing.assert_allclose(
            flat_dcegm[:, N_BRUTE_UNSTABLE_NODES:],
            flat_brute[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def _euler_axis(value_array: np.ndarray) -> int:
    """Axis of the Euler (wealth) state — the one matching the wealth grid."""
    n_wealth = int(WEALTH_GRID.to_jax().shape[0])
    return next(axis for axis, size in enumerate(value_array.shape) if size == n_wealth)


# --- Regime-transition prob through a param-dependent Euler-state chain ------
#
# The means-tested survival probability reads the Euler state `wealth` through
# a DAG-computed intermediate that itself reads a model param (a capital-income
# return rate), mirroring an SSI/Medicaid eligibility share
# `medicaid_eligibility_share <- countable_income <- capital_income(assets, r)`.
# The regime simultaneously carries a passive AIME axis and an income process
# axis, so the per-asset-node savings-stage probability evaluation must receive
# the qualified param alongside the Euler node.

RATE_OF_RETURN = 0.04


def capital_income(wealth: ContinuousState, rate_of_return: float) -> FloatND:
    """Capital income on current wealth — reads the Euler state and a param."""
    return rate_of_return * wealth


def countable_income(capital_income: FloatND) -> FloatND:
    return capital_income


def medicaid_eligibility_share(countable_income: FloatND) -> FloatND:
    """SSI-style smoothstep eligibility share over the countable income band."""
    return smoothstep_in_band(countable_income / RATE_OF_RETURN)


def survival_of_share(medicaid_eligibility_share: FloatND) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * medicaid_eligibility_share


def stay_prob_share(
    medicaid_eligibility_share: FloatND, age: int, final_age_alive: float
) -> FloatND:
    return jnp.where(
        age >= final_age_alive, 0.0, survival_of_share(medicaid_eligibility_share)
    )


def death_prob_share(
    medicaid_eligibility_share: FloatND, age: int, final_age_alive: float
) -> FloatND:
    return 1.0 - stay_prob_share(medicaid_eligibility_share, age, final_age_alive)


def _means_test_intermediates() -> dict:
    return {
        "capital_income": capital_income,
        "countable_income": countable_income,
        "medicaid_eligibility_share": medicaid_eligibility_share,
    }


@functools.cache
def _means_tested_prob_model(solver: str, *, rate_is_fixed: bool) -> Model:
    """Euler `wealth` + passive `aime` + process `income`; means-tested prob.

    The survival probability reads `wealth` through the param-dependent chain
    `capital_income(wealth, rate_of_return)`, triggering asset-row mode while
    the regime carries the passive AIME and income-process axes. When
    `rate_is_fixed`, the return rate is supplied through `fixed_params` (so it
    is partialled at model build and dropped from the live params template)
    rather than as a free solve param.
    """
    is_dcegm = solver == "dcegm"
    states = {
        "wealth": WEALTH_GRID,
        "aime": AIME_GRID,
        "income": RouwenhorstAR1Process(n_points=N_INCOME_NODES),
    }
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob_share),
            "dead": MarkovTransition(death_prob_share),
        },
        active=_active,
        actions={
            "labor_supply": DiscreteGrid(LaborChoice),
            "consumption": CONSUMPTION_GRID,
        },
        states=states,
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "aime": next_aime,
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions=(
            {
                **_shared_functions(),
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
                **_means_test_intermediates(),
            }
            if is_dcegm
            else {
                **_shared_functions(),
                "resources": resources,
                **_means_test_intermediates(),
            }
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
        regime_id_class=PassiveAssetRowRegimeId,
        fixed_params=fixed_params,
    )


def _means_test_params(*, rate_is_fixed: bool) -> dict:
    params = _params()
    if not rate_is_fixed:
        params["working_life"]["capital_income"] = {"rate_of_return": RATE_OF_RETURN}
    return params


@pytest.mark.parametrize("rate_is_fixed", [False, True])
def test_means_tested_prob_through_param_intermediate_matches_brute_force(
    rate_is_fixed: bool,  # noqa: FBT001
):
    """A means-tested survival prob `share <- capital_income(wealth, r)` matches.

    The stay probability reads the Euler state `wealth` through a param-dependent
    intermediate chain (the SSI/Medicaid eligibility-share shape), so the regime
    solves per exogenous asset node while carrying the passive AIME and income
    process axes. The per-asset-node regime-transition-probability evaluation
    receives the qualified param `capital_income__rate_of_return` — whether the
    return rate is a free solve param or supplied through `fixed_params` (and
    thus partialled into the prebuilt kernel). The probability's wealth slope
    carries the first-order term
    $\\partial P_{stay}/\\partial wealth \\cdot EV_{stay}$ into the marginal
    value. Values agree with the dense-grid brute-force oracle across the full
    wealth-by-AIME-by-income grid.
    """
    params = _means_test_params(rate_is_fixed=rate_is_fixed)
    dcegm_solution: PeriodToRegimeToVArr = _means_tested_prob_model(
        "dcegm", rate_is_fixed=rate_is_fixed
    ).solve(params=params, log_level="debug")
    brute_solution: PeriodToRegimeToVArr = _means_tested_prob_model(
        "brute_force", rate_is_fixed=rate_is_fixed
    ).solve(params=params, log_level="debug")
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape
        flat_dcegm = np.moveaxis(dcegm_V, _euler_axis(dcegm_V), -1).reshape(
            -1, dcegm_V.shape[_euler_axis(dcegm_V)]
        )
        flat_brute = np.moveaxis(brute_V, _euler_axis(brute_V), -1).reshape(
            -1, brute_V.shape[_euler_axis(brute_V)]
        )
        np.testing.assert_allclose(
            flat_dcegm[:, N_BRUTE_UNSTABLE_NODES:],
            flat_brute[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )
