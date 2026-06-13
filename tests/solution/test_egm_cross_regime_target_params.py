"""DC-EGM cross-regime carry: a target's resources reads a model-level param.

A DC-EGM regime may carry into a *different* target regime whose `resources`
function reaches a model-level (shared) parameter that the source regime's own
DAG never touches — for example a pension factor that the source, lacking the
pension function, prunes from its parameter template. Such a parameter is a
genuine model-level value, identical across regimes; the per-exogenous-asset-
node solve must evaluate the target's resources with it, exactly as the
brute-force solver does.

The asset-row solve is active because the source regime's regime-transition
probability reads the Euler state (wealth). The oracle for the solved value
function is a dense-grid brute-force solve of a mathematically equivalent
spec.
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

# Model periods; the source regime is active early, the target regime late,
# so the source carries into a *different* target regime.
N_PERIODS = 4

# Wealth band over which the source's survival smoothstep ramps. Reading
# wealth in the regime-transition probability switches the source kernel into
# the per-exogenous-asset-node (asset-row) solve.
BAND_START = 30.0
BAND_WIDTH = 20.0
SURVIVAL_LOW = 0.55
SURVIVAL_HIGH = 0.95

# Deterministic labor income added to savings in the wealth law; keeps child
# wealth queries away from the grid's lower edge.
LABOR_INCOME = 5.0

# Pension factor scaling the target regime's accrued pension into resources.
# A model-level value supplied through `fixed_params`; the source regime has
# no pension function, so its template never carries this parameter.
PENSION_FACTOR = 0.6

# Accrued pension level the target regime adds to wealth (times the factor).
ACCRUED_PENSION = 12.0

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=120.0, n_points=4000)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes excluded from the comparison: the brute solver leans on
# coarse interpolation and consumption choices near its grid start there.
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class CrossRegimeId:
    young: ScalarInt
    old: ScalarInt
    dead: ScalarInt


def smoothstep_in_band(value: FloatND) -> FloatND:
    """C² quintic smoothstep rising from 0 to 1 across the band."""
    t = jnp.clip((value - BAND_START) / BAND_WIDTH, 0.0, 1.0)
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


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


def _young_active(age: int) -> bool:
    # Young in the first decision period only, so its only DC-EGM carry target
    # is the *old* regime (a different regime) and the terminal `dead` regime.
    return age < 50


def _old_active(age: int) -> bool:
    last_age = 40 + (N_PERIODS - 1) * 10
    return 50 <= age < last_age


# --- Source-regime savings-stage reads (drive the asset-row solve) ----------


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep_in_band(wealth)


def young_stay_prob(wealth: ContinuousState) -> FloatND:
    # Young is active in the first period only; its successor is always `old`,
    # which is active in the next period, so survival never leaks into an
    # inactive regime.
    return survival_of_wealth(wealth)


def young_death_prob(wealth: ContinuousState) -> FloatND:
    return 1.0 - survival_of_wealth(wealth)


# --- Target-regime (old) resources: reads the model-level pension factor ----


def accrued_pension() -> FloatND:
    return jnp.asarray(ACCRUED_PENSION)


def pension_value(accrued_pension: FloatND, pension_factor: float) -> FloatND:
    """Pension income, scaling the accrued pension by the model-level factor."""
    return accrued_pension * pension_factor


def resources_old(wealth: ContinuousState, pension_value: FloatND) -> FloatND:
    return wealth + pension_value


def budget_constraint_old(
    consumption: ContinuousAction, wealth: ContinuousState, pension_value: FloatND
) -> BoolND:
    return consumption <= wealth + pension_value


# --- Source-regime (young) resources: plain wealth --------------------------


def resources_young(wealth: ContinuousState) -> FloatND:
    return wealth


def budget_constraint_young(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def next_old_stay_prob(wealth: ContinuousState, age: int) -> FloatND:
    # At the last decision age `old` must transition into the terminal `dead`
    # regime only, since `old` is inactive in the next period.
    last_age = 40 + (N_PERIODS - 1) * 10
    return jnp.where(age >= last_age - 10, 0.0, survival_of_wealth(wealth))


def next_old_death_prob(wealth: ContinuousState, age: int) -> FloatND:
    return 1.0 - next_old_stay_prob(wealth, age)


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _params(*, factor_is_fixed: bool) -> dict:
    params: dict = {"discount_factor": 0.95}
    if not factor_is_fixed:
        # Free param: supplied at solve time under the target regime's
        # pension function.
        params["old"] = {"pension_value": {"pension_factor": PENSION_FACTOR}}
    return params


@functools.cache
def _cross_regime_model(solver: str, *, factor_is_fixed: bool) -> Model:
    """Young (DC-EGM, asset-row) carries into a different regime `old`.

    `old`'s resources reach the model-level `pension_factor` through the
    pension chain; the source `young` regime has no pension function, so its
    parameter template never carries `pension_factor`. When `factor_is_fixed`
    the factor is supplied through `fixed_params` (partialled at model build
    and dropped from the live template); otherwise it is a free solve param of
    the target regime.
    """
    is_dcegm = solver == "dcegm"
    young = UserRegime(
        transition={
            "old": MarkovTransition(young_stay_prob),
            "dead": MarkovTransition(young_death_prob),
        },
        active=_young_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints=(
            {} if is_dcegm else {"budget_constraint": budget_constraint_young}
        ),
        functions=(
            {
                "utility": utility,
                "resources": resources_young,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources_young}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    pension_funcs = {"accrued_pension": accrued_pension, "pension_value": pension_value}
    old = UserRegime(
        transition={
            "old": MarkovTransition(next_old_stay_prob),
            "dead": MarkovTransition(next_old_death_prob),
        },
        active=_old_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints=({} if is_dcegm else {"budget_constraint": budget_constraint_old}),
        functions=(
            {
                "utility": utility,
                "resources": resources_old,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
                **pension_funcs,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources_old, **pension_funcs}
        ),
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda _age: True,
    )
    fixed_params = (
        {"old": {"pension_value": {"pension_factor": PENSION_FACTOR}}}
        if factor_is_fixed
        else {}
    )
    return Model(
        regimes={"young": young, "old": old, "dead": dead},
        ages=_ages(),
        regime_id_class=CrossRegimeId,
        fixed_params=fixed_params,
    )


def _assert_young_V_matches(
    *, dcegm_solution: PeriodToRegimeToVArr, brute_solution: PeriodToRegimeToVArr
) -> None:
    # The young regime is active in the first period only; compare its V.
    period = min(brute_solution)
    brute_V = np.asarray(brute_solution[period]["young"])
    dcegm_V = np.asarray(dcegm_solution[period]["young"])
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


@pytest.mark.parametrize("factor_is_fixed", [True, False])
def test_cross_regime_target_resources_param_matches_brute_force(
    factor_is_fixed: bool,  # noqa: FBT001
):
    """A target regime's resources reading a model-level param matches brute.

    The source `young` regime carries into the different `old` regime, whose
    resources reach the model-level `pension_factor` through the pension chain
    — a parameter the source regime's template never carries. The asset-row
    solve must evaluate the target's resources with the model-level parameter
    value, whether the factor is a fixed param (partialled into the prebuilt
    kernel) or a free solve param (threaded from the target regime's live
    params). The source regime's value function agrees with the dense-grid
    brute-force oracle.
    """
    params = _params(factor_is_fixed=factor_is_fixed)
    dcegm_solution = _cross_regime_model(
        "dcegm", factor_is_fixed=factor_is_fixed
    ).solve(params=params, log_level="debug")
    brute_solution = _cross_regime_model(
        "brute_force", factor_is_fixed=factor_is_fixed
    ).solve(params=params, log_level="debug")
    _assert_young_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )
