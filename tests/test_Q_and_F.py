from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

import lcm
from lcm.input_processing import process_regimes
from lcm.interfaces import InternalFunctions, PhaseVariantContainer
from lcm.Q_and_F import (
    _get_feasibility,
    _get_joint_weights_function,
    get_Q_and_F_terminal,
)
from tests.test_models.deterministic_regression import (
    LaborStatus,
    RegimeID,
    dead,
    get_params,
    utility,
    working,
)

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        DiscreteAction,
        DiscreteState,
        Int1D,
        ParamsDict,
        Period,
    )


@pytest.mark.illustrative
def test_get_Q_and_F_function():
    internal_regimes = process_regimes(
        [working, dead],
        n_periods=4,
        regime_id_cls=RegimeID,
        enable_jit=True,
    )

    params = get_params(n_periods=4)

    # Test terminal period Q_and_F where Q = U (no continuation value)
    Q_and_F = get_Q_and_F_terminal(
        regime=working,
        internal_functions=internal_regimes[working.name].internal_functions,
        period=3,
    )

    consumption = jnp.array([10, 20, 30])
    labor_choice = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    Q_arr, F_arr = Q_and_F(
        consumption=consumption,
        labor_choice=labor_choice,
        wealth=wealth,
        params=params,
        next_V_arr=jnp.empty(0),  # Terminal period doesn't use continuation value
    )

    assert_array_equal(
        Q_arr,
        utility(
            consumption=consumption,
            is_working=labor_choice == LaborStatus.work,
            disutility_of_work=0.5,  # matches get_params default
        ),
    )
    assert_array_equal(F_arr, jnp.array([True, True, False]))


@pytest.fixture
def internal_functions_illustrative():
    def age(period: Period) -> int | Int1D:
        return period + 18

    def mandatory_retirement_constraint(
        retirement: DiscreteAction,
        age: int | Int1D,
        params: ParamsDict,  # noqa: ARG001
    ) -> BoolND:
        # Individuals must be retired from age 65 onwards
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_constraint(
        lagged_retirement: DiscreteState,
        age: int | Int1D,
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
    mock_transition_solve = lambda *args, params, **kwargs: {"mock": 1.0}  # noqa: E731, ARG005
    mock_transition_simulate = lambda *args, params, **kwargs: {  # noqa: E731, ARG005
        "mock": jnp.array([1.0])
    }
    return InternalFunctions(
        utility=lambda: 0,  # type: ignore[arg-type]
        transitions={},
        constraints=constraints,  # type: ignore[arg-type]
        functions=functions,  # type: ignore[arg-type]
        regime_transition_probs=PhaseVariantContainer(
            solve=mock_transition_solve, simulate=mock_transition_simulate
        ),
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
    @lcm.mark.stochastic
    def next_a():
        return jnp.array([0.1, 0.9])

    @lcm.mark.stochastic
    def next_b():
        return jnp.array([0.2, 0.8])

    transitions = {"next_a": next_a, "next_b": next_b}
    multiply_weights = _get_joint_weights_function(
        regime_name="test",
        transitions=transitions,
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_test__next_a=a, weight_test__next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)


def test_get_combined_constraint():
    def f(params):  # noqa: ARG001
        return True

    def g(params):  # noqa: ARG001
        return False

    def h(params):  # noqa: ARG001
        return None

    mock_transition_solve = lambda *args, params, **kwargs: {"mock": 1.0}  # noqa: E731, ARG005
    mock_transition_simulate = lambda *args, params, **kwargs: {  # noqa: E731, ARG005
        "mock": jnp.array([1.0])
    }
    internal_functions = InternalFunctions(
        utility=lambda: 0,  # type: ignore[arg-type]
        constraints={"f": f, "g": g},  # type: ignore[dict-item]
        transitions={},
        functions={"h": h},  # type: ignore[dict-item]
        regime_transition_probs=PhaseVariantContainer(
            solve=mock_transition_solve, simulate=mock_transition_simulate
        ),
    )
    combined_constraint = _get_feasibility(internal_functions)
    feasibility: BoolND = combined_constraint(params={})
    assert feasibility.item() is False
