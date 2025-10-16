from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm.input_processing import process_regime
from lcm.interfaces import InternalFunctions
from lcm.Q_and_F import (
    _get_feasibility,
    _get_joint_weights_function,
    get_Q_and_F,
)
from lcm.state_action_space import create_state_space_info
from tests.test_models.deterministic import utility
from tests.test_models.utils import get_regime

if TYPE_CHECKING:
    from lcm.typing import BoolND, DiscreteAction, DiscreteState, ParamsDict


@pytest.mark.illustrative
def test_get_Q_and_F_function():
    regime = get_regime("iskhakov_et_al_2017_stripped_down", n_periods=3)
    internal_regime = process_regime(regime)

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    state_space_info = create_state_space_info(
        regime=regime,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        regime=regime,
        internal_functions=internal_regime.internal_functions,
        next_state_space_info=state_space_info,
        period=internal_regime.n_periods - 1,
        is_last_period=True,
    )

    consumption = jnp.array([10, 20, 30])
    retirement = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    Q_arr, F_arr = Q_and_F(
        consumption=consumption,
        retirement=retirement,
        wealth=wealth,
        params=params,
        next_V_arr=None,
    )

    assert_array_equal(
        Q_arr,
        utility(
            consumption=consumption,
            working=1 - retirement,
            disutility_of_work=1.0,
        ),
    )
    assert_array_equal(F_arr, jnp.array([True, True, False]))


@pytest.fixture
def internal_functions_illustrative():
    def age(period: int) -> int:
        return period + 18

    def mandatory_retirement_constraint(
        retirement: DiscreteAction,
        age: int,
        params: ParamsDict,  # noqa: ARG001
    ) -> BoolND:
        # Individuals must be retired from age 65 onwards
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_constraint(
        lagged_retirement: DiscreteState,
        age: int,
        params: ParamsDict,  # noqa: ARG001
    ) -> BoolND:
        # Individuals must have been retired last year from age 66 onwards
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_constraint(
        retirement: DiscreteAction,
        lagged_retirement: DiscreteState,
        params: ParamsDict,  # noqa: ARG001
    ) -> BoolND:
        # If an individual was retired last year, it must be retired this year
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    constraints = {
        "mandatory_retirement_constraint": mandatory_retirement_constraint,
        "mandatory_lagged_retirement_constraint": (
            mandatory_lagged_retirement_constraint
        ),
        "absorbing_retirement_constraint": absorbing_retirement_constraint,
    }

    functions = {"age": age}

    # create an internal regime instance where some attributes are set to None
    # because they are not needed to create the feasibilty mask
    return InternalFunctions(
        utility=lambda: 0,  # type: ignore[arg-type]
        transitions={},
        constraints=constraints,  # type: ignore[arg-type]
        functions=functions,  # type: ignore[arg-type]
    )


@pytest.mark.illustrative
def test_get_combined_constraint_illustrative(internal_functions_illustrative):
    combined_constraint = _get_feasibility(internal_functions_illustrative)

    age, retirement, lagged_retirement = jnp.array(
        [
            # feasible cases
            [60, 0, 0],  # Young, never retired
            [64, 1, 0],  # Near retirement, newly retired
            [70, 1, 1],  # Properly retired with lagged retirement
            # infeasible cases
            [65, 0, 0],  # Must be retired at 65
            [66, 0, 1],  # Must have lagged retirement at 66
            [60, 0, 1],  # Can't be not retired if was retired before
        ]
    ).T

    # combined constraint expects period not age
    period = age - 18

    exp = jnp.array(3 * [True] + 3 * [False])
    got = combined_constraint(
        period=period,
        retirement=retirement,
        lagged_retirement=lagged_retirement,
        params={},
    )
    assert_array_equal(got, exp)


def test_get_multiply_weights():
    multiply_weights = _get_joint_weights_function(
        stochastic_variables=("a", "b"),
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_next_a=a, weight_next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)


def test_get_combined_constraint():
    def f(params):  # noqa: ARG001
        return True

    def g(params):  # noqa: ARG001
        return False

    def h(params):  # noqa: ARG001
        return None

    internal_functions = InternalFunctions(
        utility=lambda: 0,  # type: ignore[arg-type]
        constraints={"f": f, "g": g},  # type: ignore[dict-item]
        transitions={},
        functions={"h": h},  # type: ignore[dict-item]
    )
    combined_constraint = _get_feasibility(internal_functions)
    feasibility: BoolND = combined_constraint(params={})
    assert feasibility.item() is False
