"""A discrete action is not silently dropped from a case-piece regime.

The case-piece kernels solve a fixed one-asset consumption--saving problem and
never read the regime's discrete actions: the published value maximizes over
consumption alone, and simulation then draws a discrete policy from a value
function that never saw the choice. Declaring both is rejected at model build.
"""

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    medicaid_eligible,
    subsidy_medicaid,
    subsidy_private,
)

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)


@categorical(ordered=False)
class InsurancePlan:
    basic: ScalarInt
    premium: ScalarInt


def premium_cost(insurance_plan: DiscreteAction, premium_price: float) -> FloatND:
    """Out-of-pocket premium the chosen plan charges."""
    return jnp.where(insurance_plan == InsurancePlan.premium, premium_price, 0.0)


def subsidy(
    subsidy_medicaid: FloatND, subsidy_private: FloatND, medicaid_eligible: BoolND
) -> FloatND:
    """Dense combination of the two subsidy pieces."""
    return jnp.where(medicaid_eligible, subsidy_medicaid, subsidy_private)


def resources(
    liquid: ContinuousState, subsidy: FloatND, premium_cost: FloatND
) -> FloatND:
    """Cash-on-hand: liquid wealth plus the subsidy net of the plan premium."""
    return liquid + subsidy - premium_cost


def resources_without_premium(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the Medicaid-contingent subsidy."""
    return liquid + subsidy


def keep_health(health: DiscreteState) -> DiscreteState:
    """Health is an absorbing ride-along state."""
    return health


def test_a_case_piece_regime_with_a_discrete_action_is_rejected():
    """The case-piece kernels optimize over consumption only, so both is invalid."""
    with pytest.raises(RegimeInitializationError, match="discrete action"):
        make_alive_dead_model(
            n_periods=3,
            n_liquid=20,
            liquid_max=20.0,
            n_consumption=20,
            alive_functions={
                "utility": utility,
                "savings": savings,
                "medicaid_eligible": medicaid_eligible,
                "subsidy_medicaid": subsidy_medicaid,
                "subsidy_private": subsidy_private,
                "subsidy": subsidy,
                "premium_cost": premium_cost,
                "resources": resources,
            },
            liquid_law=next_liquid_from_savings,
            alive_solver=resolve_solver(
                "nbegm",
                savings_grid=SAVINGS_GRID,
                post_decision_function="savings",
            ),
            constraints={"feasible": feasible},
            extra_actions={"insurance_plan": DiscreteGrid(InsurancePlan)},
        )


@categorical(ordered=False)
class Health:
    good: ScalarInt
    bad: ScalarInt


def test_a_case_piece_regime_with_a_second_state_names_that_state() -> None:
    """The single-axis kernels carry the liquid axis alone.

    The canonical variable order leads with discrete states, so resolving the
    liquid axis as the regime's first state would name `health` and reject the
    boundary for comparing the wrong variable. The rejection names the extra
    state instead.
    """
    with pytest.raises(RegimeInitializationError, match=r"also declares.*health"):
        make_alive_dead_model(
            n_periods=3,
            n_liquid=20,
            liquid_max=20.0,
            n_consumption=20,
            alive_functions={
                "utility": utility,
                "savings": savings,
                "medicaid_eligible": medicaid_eligible,
                "subsidy_medicaid": subsidy_medicaid,
                "subsidy_private": subsidy_private,
                "subsidy": subsidy,
                "resources": resources_without_premium,
            },
            liquid_law=next_liquid_from_savings,
            alive_solver=resolve_solver(
                "nbegm",
                savings_grid=SAVINGS_GRID,
                post_decision_function="savings",
            ),
            constraints={"feasible": feasible},
            extra_states={"health": DiscreteGrid(Health)},
            extra_state_transitions={
                "health": {"alive": keep_health, "dead": keep_health},
            },
        )
