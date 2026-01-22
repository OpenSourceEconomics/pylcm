from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

import lcm
from lcm import AgeGrid
from lcm.input_processing import process_regimes
from lcm.interfaces import InternalFunctions, PhaseVariantContainer
from lcm.Q_and_F import (
    _get_feasibility,
    _get_joint_weights_function,
    _get_U_and_F,
    get_Q_and_F_terminal,
)
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    Int1D,
    ParamsDict,
    Period,
)
from tests.test_models.deterministic.regression import (
    LaborSupply,
    dead,
    get_params,
    utility,
    working,
)


@pytest.mark.illustrative
def test_get_Q_and_F_function():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working": working, "dead": dead}
    regime_id = MappingProxyType({name: idx for idx, name in enumerate(regimes.keys())})
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_id,
        enable_jit=True,
    )

    params = get_params(n_periods=4)

    # Test terminal period Q_and_F where Q = U (no continuation value)
    Q_and_F = get_Q_and_F_terminal(
        regime_name="working",
        internal_functions=internal_regimes["working"].internal_functions,
        period=3,
        age=ages.period_to_age(3),
    )

    consumption = jnp.array([10, 20, 30])
    labor_supply = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    Q_arr, F_arr = Q_and_F(
        consumption=consumption,
        labor_supply=labor_supply,
        wealth=wealth,
        params=params,
        next_V_arr=jnp.empty(0),  # Terminal period doesn't use continuation value
    )

    assert_array_equal(
        Q_arr,
        utility(
            consumption=consumption,
            is_working=labor_supply == LaborSupply.work,
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
        utility=lambda: 0,  # ty: ignore[invalid-argument-type]
        transitions=MappingProxyType({}),
        constraints=constraints,  # ty: ignore[invalid-argument-type]
        functions=functions,  # ty: ignore[invalid-argument-type]
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
        utility=lambda: 0,  # ty: ignore[invalid-argument-type]
        constraints={"f": f, "g": g},  # ty: ignore[invalid-argument-type]
        transitions=MappingProxyType({}),
        functions={"h": h},  # ty: ignore[invalid-argument-type]
        regime_transition_probs=PhaseVariantContainer(
            solve=mock_transition_solve, simulate=mock_transition_simulate
        ),
    )
    combined_constraint = _get_feasibility(internal_functions)
    feasibility: BoolND = combined_constraint(params={})
    assert feasibility.item() is False


def test_get_U_and_F_with_annotated_constraints():
    """Test that _get_U_and_F works when constraints and utility have type annotations.

    This test verifies that dags handles the case where:
    1. Constraint functions have type annotations
    2. The utility function has type annotations for the same arguments
    3. The combined feasibility function (created with an aggregator) may have
       "no_annotation_found" for some arguments due to functools.wraps behavior

    With dags < 0.4.3, this would raise AnnotationMismatchError because the
    feasibility function's "no_annotation_found" annotations conflict with the
    proper annotations from the utility and other functions.
    """

    # Constraint with type annotations
    def budget_constraint(
        consumption: float,
        wealth: float,
        params: ParamsDict,  # noqa: ARG001
    ) -> bool:
        return consumption <= wealth

    # Another constraint with type annotations
    def positive_consumption_constraint(
        consumption: float,
        params: ParamsDict,  # noqa: ARG001
    ) -> bool:
        return consumption >= 0

    # Utility function with type annotations for the same arguments
    def utility_func(
        consumption: float,
        params: ParamsDict,  # noqa: ARG001
    ) -> jax.Array:
        return jnp.log(consumption + 1)

    mock_transition_solve = lambda *args, params, **kwargs: {"mock": 1.0}  # noqa: E731, ARG005
    mock_transition_simulate = lambda *args, params, **kwargs: {  # noqa: E731, ARG005
        "mock": jnp.array([1.0])
    }

    internal_functions = InternalFunctions(
        utility=utility_func,  # ty: ignore[invalid-argument-type]
        constraints=MappingProxyType(
            {
                "budget_constraint": budget_constraint,
                "positive_consumption_constraint": positive_consumption_constraint,
            }
        ),
        transitions=MappingProxyType({}),
        functions=MappingProxyType({}),
        regime_transition_probs=PhaseVariantContainer(
            solve=mock_transition_solve, simulate=mock_transition_simulate
        ),
    )

    # This should not raise AnnotationMismatchError
    U_and_F = _get_U_and_F(internal_functions)

    # Verify it works correctly
    U, F = U_and_F(consumption=5.0, wealth=10.0, params={})
    assert jnp.isclose(U, jnp.log(6.0))
    assert F.item() is True

    # Test infeasible case
    U, F = U_and_F(consumption=15.0, wealth=10.0, params={})
    assert F.item() is False
