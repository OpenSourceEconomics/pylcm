from collections.abc import Callable
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm import AgeGrid
from lcm.grids import DiscreteGrid, LinSpacedGrid, categorical
from lcm.model import Model
from lcm.params.processing import (
    create_params_template,
    get_flat_param_names,
    process_params,
)
from lcm.regime import MarkovTransition, Regime
from lcm.regime_building import process_regimes
from lcm.regime_building.Q_and_F import (
    _get_feasibility,
    _get_joint_weights_function,
    _get_U_and_F,
    get_Q_and_F_terminal,
)
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Int1D,
    Period,
    ScalarInt,
)
from tests.test_models.deterministic.regression import (
    LaborSupply,
    dead,
    get_params,
    utility,
    working_life,
)


@pytest.mark.illustrative
def test_get_Q_and_F_function():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: idx for idx, name in enumerate(regimes.keys())}
    )
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
    )

    raw_params = get_params(n_periods=4)

    params_template = create_params_template(internal_regimes)
    internal_params = process_params(params=raw_params, params_template=params_template)

    # Compute flat param names for the working regime's regime_params_template
    flat_param_names = frozenset(
        get_flat_param_names(internal_regimes["working_life"].regime_params_template)
    )

    # Test terminal period Q_and_F where Q = U (no continuation value)
    solve = internal_regimes["working_life"].solve_functions
    Q_and_F = get_Q_and_F_terminal(
        flat_param_names=flat_param_names,
        functions=solve.functions,
        constraints=solve.constraints,
    )

    consumption = jnp.array([10, 20, 30])
    labor_supply = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    Q_arr, F_arr = Q_and_F(
        consumption=consumption,
        labor_supply=labor_supply,
        wealth=wealth,
        **internal_params["working_life"],
        next_regime_to_V_arr=jnp.empty(0),
        period=3,
        age=ages.period_to_age(3),
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
    ) -> BoolND:
        # Individuals must be retired from age 65 onwards
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_constraint(
        lagged_retirement: DiscreteState,
        age: int | Int1D,
    ) -> BoolND:
        # Individuals must have been retired last year from age 66 onwards
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_constraint(
        retirement: DiscreteAction,
        lagged_retirement: DiscreteState,
    ) -> BoolND:
        # If an individual was retired last year, it must be retired this year
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    constraints = MappingProxyType(
        {
            "mandatory_retirement_constraint": mandatory_retirement_constraint,
            "mandatory_lagged_retirement_constraint": (
                mandatory_lagged_retirement_constraint
            ),
            "absorbing_retirement_constraint": absorbing_retirement_constraint,
        }
    )

    functions = MappingProxyType({"utility": lambda: 0, "age": age})

    return {"functions": functions, "constraints": constraints}


@pytest.mark.illustrative
def test_get_combined_constraint_illustrative(internal_functions_illustrative):
    combined_constraint = _get_feasibility(**internal_functions_illustrative)

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
    )
    assert_array_equal(got, exp)


def test_get_multiply_weights():
    def next_a():
        return jnp.array([0.1, 0.9])

    def next_b():
        return jnp.array([0.2, 0.8])

    transitions = MappingProxyType({"next_a": next_a, "next_b": next_b})
    multiply_weights = _get_joint_weights_function(
        regime_name="test",
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        stochastic_transition_names=frozenset({"next_a", "next_b"}),
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_test__next_a=a, weight_test__next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)


def test_get_combined_constraint():
    def f():
        return True

    def g():
        return False

    def h():
        return None

    combined_constraint = _get_feasibility(
        functions=MappingProxyType({"utility": lambda: 0, "h": h}),  # ty: ignore[invalid-argument-type]
        constraints=MappingProxyType({"f": f, "g": g}),  # ty: ignore[invalid-argument-type]
    )
    feasibility: BoolND = combined_constraint()
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
    ) -> bool:
        return consumption <= wealth

    # Another constraint with type annotations
    def positive_consumption_constraint(
        consumption: float,
    ) -> bool:
        return consumption >= 0

    # Utility function with type annotations for the same arguments
    def utility_func(
        consumption: float,
    ) -> jax.Array:
        return jnp.log(consumption + 1)

    # This should not raise AnnotationMismatchError
    U_and_F = _get_U_and_F(
        functions=MappingProxyType({"utility": utility_func}),  # ty: ignore[invalid-argument-type]
        constraints=MappingProxyType(  # ty: ignore[invalid-argument-type]
            {
                "budget_constraint": budget_constraint,
                "positive_consumption_constraint": positive_consumption_constraint,
            }
        ),
    )

    # Verify it works correctly
    U, F = U_and_F(consumption=5.0, wealth=10.0)
    assert jnp.isclose(U, jnp.log(6.0))
    assert F.item() is True

    # Test infeasible case
    U, F = U_and_F(consumption=15.0, wealth=10.0)
    assert F.item() is False


def _health_probs(health: DiscreteState, probs_array: FloatND) -> FloatND:
    return probs_array[health]


@categorical(ordered=True)
class _IncompleteTargetHealth:
    bad: int = 0
    good: int = 1


@categorical(ordered=False)
class _IncompleteTargetRegimeId:
    work: int
    retire: int
    dead: int


def _build_incomplete_target_model(
    *,
    next_regime_func: Callable,
) -> tuple[Model, dict]:
    """Build a model where "retire" is an incomplete target from "work".

    "work" has a per-target MarkovTransition for health that only covers
    "work" (not "retire"), making "retire" incomplete.
    """

    def _utility(
        consumption: float,
        health: DiscreteState,
    ) -> FloatND:
        return jnp.log(consumption)

    def _next_wealth(consumption: float, wealth: float) -> float:
        return wealth - consumption

    work = Regime(
        active=lambda age: age <= 2,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=3),
            "health": DiscreteGrid(_IncompleteTargetHealth),
        },
        state_transitions={
            "wealth": _next_wealth,
            "health": {
                "work": MarkovTransition(_health_probs),
            },
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=3),
        },
        transition=next_regime_func,
        functions={"utility": _utility},
    )
    retire = Regime(
        active=lambda age: age <= 2,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=3),
            "health": DiscreteGrid(_IncompleteTargetHealth),
        },
        state_transitions={
            "wealth": _next_wealth,
            "health": MarkovTransition(_health_probs),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=3),
        },
        transition=next_regime_func,
        functions={"utility": _utility},
    )
    dead_regime = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )

    model = Model(
        regimes={"work": work, "retire": retire, "dead": dead_regime},
        regime_id_class=_IncompleteTargetRegimeId,
        ages=AgeGrid(start=0, stop=3, step="Y"),
    )
    params = {
        "discount_factor": 0.9,
        "probs_array": jnp.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    return model, params


def test_incomplete_target_zero_prob_succeeds():
    """Solve succeeds when incomplete target has zero transition probability."""

    def _next_regime(age: float) -> ScalarInt:
        return jnp.where(
            age >= 2, _IncompleteTargetRegimeId.dead, _IncompleteTargetRegimeId.work
        )

    model, params = _build_incomplete_target_model(next_regime_func=_next_regime)
    model.solve(params=params)
