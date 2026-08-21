"""DC-EGM cross-regime carry preserves regime-local parameter identity.

A DC-EGM regime may carry into a *different* target regime whose `resources`
function exposes the same inner qualified parameter name as the source, but a
different regime-local value. The per-exogenous-asset-node solve must evaluate
the target's resources with the target value, exactly as the brute-force solver
does; flattening both namespaces into one source-first mapping is invalid.

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
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

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

# Pension factors scaling the same-named source and target functions. The
# values deliberately differ: both regimes therefore carry the identical inner
# qualified name `pension_value__pension_factor`, and the cross-regime carry
# must retain which regime owns each value.
YOUNG_PENSION_FACTOR = 0.15
OLD_PENSION_FACTOR = 0.6

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


# Source-regime savings-stage reads (drive the asset-row solve)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep_in_band(wealth)


def young_stay_prob(wealth: ContinuousState) -> FloatND:
    # Young is active in the first period only; its successor is always `old`,
    # which is active in the next period, so survival never leaks into an
    # inactive regime.
    return survival_of_wealth(wealth)


def young_death_prob(wealth: ContinuousState) -> FloatND:
    return 1.0 - survival_of_wealth(wealth)


# Target-regime (old) resources: reads the old regime's pension factor


def accrued_pension() -> FloatND:
    return jnp.asarray(ACCRUED_PENSION)


def pension_value(accrued_pension: FloatND, pension_factor: float) -> FloatND:
    """Pension income, scaling accrued pension by its regime-local factor."""
    return accrued_pension * pension_factor


def resources_old(wealth: ContinuousState, pension_value: FloatND) -> FloatND:
    return wealth + pension_value


def budget_constraint_old(
    consumption: ContinuousAction, wealth: ContinuousState, pension_value: FloatND
) -> BoolND:
    return consumption <= wealth + pension_value


# Source-regime (young) resources use the same function-qualified parameter
# name as the target, but a different regime-local value.


def resources_young(wealth: ContinuousState, pension_value: FloatND) -> FloatND:
    return wealth + pension_value


def budget_constraint_young(
    consumption: ContinuousAction, wealth: ContinuousState, pension_value: FloatND
) -> BoolND:
    return consumption <= wealth + pension_value


def next_old_stay_prob(wealth: ContinuousState, age: int) -> FloatND:
    # At the last decision age `old` must transition into the terminal `dead`
    # regime only, since `old` is inactive in the next period.
    last_age = 40 + (N_PERIODS - 1) * 10
    return jnp.where(age >= last_age - 10, 0.0, survival_of_wealth(wealth))


def next_old_death_prob(wealth: ContinuousState, age: int) -> FloatND:
    return 1.0 - next_old_stay_prob(wealth, age)


DCEGM_SOLVER = DCEGM(
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _params(*, factor_is_fixed: bool) -> dict:
    params: dict = {"discount_factor": 0.95}
    if not factor_is_fixed:
        # Free param: supplied at solve time under the target regime's
        # pension function.
        params["young"] = {"pension_value": {"pension_factor": YOUNG_PENSION_FACTOR}}
        params["old"] = {"pension_value": {"pension_factor": OLD_PENSION_FACTOR}}
    return params


@functools.cache
def _cross_regime_model(solver: str, *, factor_is_fixed: bool) -> Model:
    """Young (DC-EGM, asset-row) carries into a different regime `old`.

    `young` and `old` both expose `pension_value__pension_factor`, but each
    owns a different regime-local value. The target resources map must read
    the `old` value rather than the source-first value from `young`. When
    `factor_is_fixed`, both values are supplied through `fixed_params`
    (partialled at model build and dropped from the live template); otherwise
    they are free solve params.
    """
    is_dcegm = solver == "dcegm"
    regime_type = ConsumptionSavingsRegime if is_dcegm else UserRegime
    young = regime_type(
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
                "accrued_pension": accrued_pension,
                "pension_value": pension_value,
            }
            if is_dcegm
            else {
                "utility": utility,
                "resources": resources_young,
                "accrued_pension": accrued_pension,
                "pension_value": pension_value,
            }
        ),
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
        **(
            {
                "liquid": LiquidMargin(
                    state="wealth",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                )
            }
            if is_dcegm
            else {}
        ),
    )
    pension_funcs = {"accrued_pension": accrued_pension, "pension_value": pension_value}
    old = regime_type(
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
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
        **(
            {
                "liquid": LiquidMargin(
                    state="wealth",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                )
            }
            if is_dcegm
            else {}
        ),
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda _age: True,
    )
    fixed_params = (
        {
            "young": {"pension_value": {"pension_factor": YOUNG_PENSION_FACTOR}},
            "old": {"pension_value": {"pension_factor": OLD_PENSION_FACTOR}},
        }
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
    """Same-named source and target parameters remain regime-local.

    The source `young` regime carries into the different `old` regime. Source
    and target resources expose the same inner qualified parameter name but
    receive different regime-local values. The asset-row solve must evaluate
    each resources map with its owning regime's value, whether the values are
    fixed params or free solve params. The source value function agrees with
    the dense-grid brute-force oracle.
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
