from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dags import concatenate_functions
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.grids import DiscreteGrid, LinSpacedGrid, categorical
from _lcm.params.processing import (
    create_params_template,
    get_flat_param_names,
    process_params,
)
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.Q_and_F import (
    LAW_SOURCE_ATTR,
    _get_deterministic_transitions,
    _get_feasibility,
    _get_joint_weights_function,
    _get_U_and_F,
    _law_sources_differ,
    get_compute_intermediates,
    get_Q_and_F,
    get_Q_and_F_terminal,
)
from _lcm.regime_building.V import VInterpolationInfo
from lcm import AgeGrid, PowerMean
from lcm.model import Model
from lcm.regime import MarkovTransition
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Int1D,
    Period,
    ScalarInt,
)
from tests.conftest import build_prepared_structure
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
    user_regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(idx) for idx, name in enumerate(user_regimes.keys())}
    )
    finalized_user_regimes = finalize_regimes(
        user_regimes=user_regimes, derived_categoricals={}
    )
    regimes = process_regimes(
        user_regimes=finalized_user_regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
        prepared_structure=build_prepared_structure(
            user_regimes=finalized_user_regimes, ages=ages
        ),
    )

    raw_params = get_params(n_periods=4)

    params_template = create_params_template(regimes)
    flat_params = process_params(params=raw_params, params_template=params_template)

    # Compute flat param names for the working regime's regime_params_template
    flat_param_names = frozenset(
        get_flat_param_names(regimes["working_life"].regime_params_template)
    )

    # Test terminal period Q_and_F where Q = U (no continuation value)
    solve = regimes["working_life"].solution
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
        **flat_params["working_life"],
        next_regime_to_V_arr=MappingProxyType({}),
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


def test_identical_target_specific_deterministic_laws_are_accepted():
    """Identical `next_<state>` laws across targets bind into the decision DAG.

    When every target bundle carries the same `next_durable` function object and
    `utility` reads it, the within-period law is unambiguous, so the merged
    decision DAG builds without error.
    """

    def next_durable(durable: float) -> float:
        return durable

    def utility(consumption: float, next_durable: float) -> FloatND:
        return jnp.log(consumption) + next_durable

    transitions = MappingProxyType(
        {
            "stay": MappingProxyType({"next_durable": next_durable}),
            "leave": MappingProxyType({"next_durable": next_durable}),
        }
    )
    deterministic_transitions, conflicting = _get_deterministic_transitions(
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        stochastic_transition_names=frozenset(),
    )
    assert conflicting == frozenset()
    U_and_F = _get_U_and_F(
        functions=MappingProxyType({"utility": utility}),  # ty: ignore[invalid-argument-type]
        constraints=MappingProxyType({}),
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=conflicting,
    )
    U, _F = U_and_F(consumption=jnp.asarray(2.0), durable=jnp.asarray(3.0))
    assert jnp.isclose(U, jnp.log(2.0) + 3.0)


def test_conflicting_target_specific_deterministic_law_read_by_utility_is_rejected():
    """A `next_<state>` read by `utility` must agree across all targets.

    When two target bundles supply *different* implementations of the same
    `next_durable` law and `utility` reads it, the merged decision DAG would bind
    one target's law while the simulate state-update uses the right one — a silent
    disagreement. The build rejects this, naming the conflicting state.
    """

    def next_durable_stay(durable: float) -> float:
        return durable

    def next_durable_leave(durable: float) -> float:
        return 0.0 * durable

    def utility(consumption: float, next_durable: float) -> FloatND:
        return jnp.log(consumption) + next_durable

    transitions = MappingProxyType(
        {
            "stay": MappingProxyType({"next_durable": next_durable_stay}),
            "leave": MappingProxyType({"next_durable": next_durable_leave}),
        }
    )
    deterministic_transitions, conflicting = _get_deterministic_transitions(
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        stochastic_transition_names=frozenset(),
    )
    assert conflicting == frozenset({"next_durable"})
    with pytest.raises(ValueError, match="next_durable"):
        _get_U_and_F(
            functions=MappingProxyType({"utility": utility}),  # ty: ignore[invalid-argument-type]
            constraints=MappingProxyType({}),
            deterministic_transitions=deterministic_transitions,
            conflicting_deterministic_transition_names=conflicting,
        )


def test_conflicting_deterministic_law_not_read_by_decision_is_accepted():
    """An unread conflicting `next_<state>` law does not block the build.

    When the conflicting `next_durable` is pruned away because neither `utility`
    nor any constraint reads it, the decision DAG never binds it, so the
    disagreement is harmless and the build succeeds.
    """

    def next_durable_stay(durable: float) -> float:
        return durable

    def next_durable_leave(durable: float) -> float:
        return 0.0 * durable

    def utility(consumption: float) -> FloatND:
        return jnp.log(consumption)

    transitions = MappingProxyType(
        {
            "stay": MappingProxyType({"next_durable": next_durable_stay}),
            "leave": MappingProxyType({"next_durable": next_durable_leave}),
        }
    )
    deterministic_transitions, conflicting = _get_deterministic_transitions(
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        stochastic_transition_names=frozenset(),
    )
    assert conflicting == frozenset({"next_durable"})
    U_and_F = _get_U_and_F(
        functions=MappingProxyType({"utility": utility}),  # ty: ignore[invalid-argument-type]
        constraints=MappingProxyType({}),
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=conflicting,
    )
    U, _F = U_and_F(consumption=jnp.asarray(2.0))
    assert jnp.isclose(U, jnp.log(2.0))


class _RaisingEq:
    """A stand-in for an array-backed callable law whose `==`/`!=` is not a plain bool.

    A real array-backed callable object (e.g. one wrapping a jax array) compares by
    value, so `a != b` builds an array and `bool(...)` on it raises. Here `__eq__`
    raises outright, which any value comparison of the provenance token would trigger.
    """

    def __eq__(self, other: object) -> bool:
        raise AssertionError("law base must never be compared by value")

    __hash__ = object.__hash__


def test_law_sources_differ_uses_identity_not_value_equality():
    """The conflict comparison must not invoke a user law's `__eq__` (round-7 F3).

    The base user law is compared by object identity and the parameter location by
    string equality, so an array-backed callable law whose `==` returns a non-bool
    (or raises) never blocks or corrupts the merge.
    """
    base1, base2 = _RaisingEq(), _RaisingEq()

    def a() -> float:
        return 0.0

    def b() -> float:
        return 0.0

    # Distinct base objects, same location -> differ, WITHOUT calling `base.__eq__`.
    setattr(a, LAW_SOURCE_ATTR, (base1, "next_x"))
    setattr(b, LAW_SOURCE_ATTR, (base2, "next_x"))
    assert _law_sources_differ(a, b) is True  # ty: ignore[invalid-argument-type]

    # Same base object, same location -> not differ (identity short-circuit).
    setattr(b, LAW_SOURCE_ATTR, (base1, "next_x"))
    assert _law_sources_differ(a, b) is False  # ty: ignore[invalid-argument-type]

    # Same base object, different (target-qualified) location -> differ, by string.
    setattr(b, LAW_SOURCE_ATTR, (base1, "next_x__retire"))
    assert _law_sources_differ(a, b) is True  # ty: ignore[invalid-argument-type]


def _health_probs(health: DiscreteState, probs_array: FloatND) -> FloatND:
    return probs_array[health]


@categorical(ordered=True)
class _PartialCoverageHealth:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _PartialCoverageRegimeId:
    work: ScalarInt
    retire: ScalarInt
    dead: ScalarInt


def _build_partial_coverage_model(
    *,
    work_transition: dict[str, MarkovTransition],
    next_regime_func: Callable,
) -> tuple[Model, dict]:
    """Build a model whose "work" regime covers `health` only toward "work".

    "retire" also carries `health`; whether the model is valid depends on
    whether "work"'s regime transition declares "retire" reachable.
    """

    def _utility(
        consumption: float,
        health: DiscreteState,
    ) -> FloatND:
        return jnp.log(consumption)

    def _next_wealth(consumption: float, wealth: float) -> float:
        return wealth - consumption

    work = UserRegime(
        active=lambda age: age <= 2,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=3),
            "health": DiscreteGrid(_PartialCoverageHealth),
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
        transition=work_transition,
        functions={"utility": _utility},
    )
    retire = UserRegime(
        active=lambda age: age <= 2,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=3),
            "health": DiscreteGrid(_PartialCoverageHealth),
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
    dead_regime = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )

    model = Model(
        regimes={"work": work, "retire": retire, "dead": dead_regime},
        regime_id_class=_PartialCoverageRegimeId,
        ages=AgeGrid(start=0, stop=3, step="Y"),
    )
    params = {
        "discount_factor": 0.9,
        "probs_array": jnp.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    return model, params


def test_partial_state_laws_solve_with_declared_targets():
    """Per-target state laws covering exactly the declared targets solve.

    "work" declares only itself and "dead" as targets; "retire" carries
    `health` but is structurally unreachable from "work", so no law toward
    it is needed.
    """

    def _next_regime(age: float) -> ScalarInt:
        return jnp.where(
            age >= 2, _PartialCoverageRegimeId.dead, _PartialCoverageRegimeId.work
        )

    work_transition = {
        "work": MarkovTransition(lambda age: jnp.where(age >= 2, 0.0, 1.0)),
        "dead": MarkovTransition(lambda age: jnp.where(age >= 2, 1.0, 0.0)),
    }
    model, params = _build_partial_coverage_model(
        work_transition=work_transition, next_regime_func=_next_regime
    )
    period_to_regime_to_V_arr = model.solve(log_level="debug", params=params)

    # Consumption is unconstrained by wealth in this model, so the optimum is
    # the largest consumption node (c = 2) in every state, giving flow utility
    # log(2) each active period. The value is the discounted sum of log(2) over
    # the remaining active periods (discount_factor 0.9) and is constant across
    # the (health, wealth) grid; the terminal "dead" regime yields exactly 0.
    log_two = float(jnp.log(2.0))
    beta = 0.9
    expected_alive = {
        0: log_two * (1 + beta + beta**2),
        1: log_two * (1 + beta),
        2: log_two,
    }
    for period, expected in expected_alive.items():
        for regime_name in ("work", "retire"):
            V_arr = period_to_regime_to_V_arr[period][regime_name]
            assert V_arr.shape == (2, 3)
            assert_allclose(V_arr, expected, atol=1e-5)
    for regime_to_V_arr in period_to_regime_to_V_arr.values():
        dead_V = regime_to_V_arr["dead"]
        assert dead_V.shape == ()
        assert_allclose(dead_V, 0.0, atol=1e-6)


def _sum_utility(utility_level: FloatND) -> FloatND:
    return utility_level


def _epstein_zin_H(utility: FloatND, E_next_V: FloatND) -> FloatND:
    return utility + E_next_V


def _low_and_high_probs(regime_prob_low: FloatND) -> MappingProxyType[str, FloatND]:
    return MappingProxyType(
        {"low": regime_prob_low, "high": 1.0 - regime_prob_low},
    )


# A target regime without states: its value function array is a scalar, so the
# interpolator is the identity and each target contributes a single lottery node.
_STATELESS_V_INFO = VInterpolationInfo(
    state_names=(),
    discrete_states=MappingProxyType({}),
    continuous_states=MappingProxyType({}),
)


def _build_two_target_closure(builder: Callable, *, certainty_equivalent) -> Callable:
    """Build `Q_and_F` (or the diagnostics twin) over two stateless target regimes."""
    return builder(
        flat_param_names=frozenset({"certainty_equivalent__risk_aversion"}),
        functions=MappingProxyType({"utility": _sum_utility, "H": _epstein_zin_H}),
        constraints=MappingProxyType({}),
        period_targets=("low", "high"),
        transitions=MappingProxyType({}),
        stochastic_transition_names=frozenset(),
        compute_regime_transition_probs=concatenate_functions(
            functions={"regime_transition_probs": _low_and_high_probs},
            targets="regime_transition_probs",
            enforce_signature=False,
            set_annotations=True,
        ),
        regime_to_v_interpolation_info=MappingProxyType(
            {"low": _STATELESS_V_INFO, "high": _STATELESS_V_INFO}
        ),
        certainty_equivalent=certainty_equivalent,
    )


def _two_target_call_kwargs(
    *,
    values: tuple[float, float],
    regime_prob_low: float,
    utility_level: float,
    risk_aversion: float,
    dtype,
) -> dict:
    return {
        "next_regime_to_V_arr": MappingProxyType(
            {
                "low": jnp.asarray(values[0], dtype=dtype),
                "high": jnp.asarray(values[1], dtype=dtype),
            }
        ),
        "utility_level": jnp.asarray(utility_level, dtype=dtype),
        "regime_prob_low": jnp.asarray(regime_prob_low, dtype=dtype),
        "age": jnp.asarray(25),
        "period": jnp.asarray(0),
        "certainty_equivalent__risk_aversion": jnp.asarray(risk_aversion, dtype=dtype),
    }


def test_power_mean_regime_lottery_stays_finite_in_float64(x64_enabled: None):
    """A `(1e-50, 2e-50)` regime lottery at risk aversion 8 keeps its exact value."""
    Q_and_F = _build_two_target_closure(get_Q_and_F, certainty_equivalent=PowerMean())
    got = jax.jit(
        lambda: Q_and_F(
            **_two_target_call_kwargs(
                values=(1e-50, 2e-50),
                regime_prob_low=0.5,
                utility_level=0.0,
                risk_aversion=8.0,
                dtype=jnp.float64,
            )
        )[0]
    )()
    np.testing.assert_allclose(float(got), 1.102862741485982e-50, rtol=5e-5, atol=0.0)


def test_power_mean_regime_lottery_stays_finite_in_float32(x64_disabled: None):
    """A `(1e-8, 2e-8)` regime lottery at risk aversion 8 keeps its exact value."""
    Q_and_F = _build_two_target_closure(get_Q_and_F, certainty_equivalent=PowerMean())
    got = jax.jit(
        lambda: Q_and_F(
            **_two_target_call_kwargs(
                values=(1e-8, 2e-8),
                regime_prob_low=0.5,
                utility_level=0.0,
                risk_aversion=8.0,
                dtype=jnp.float32,
            )
        )[0]
    )()
    np.testing.assert_allclose(float(got), 1.102862741485982e-8, rtol=5e-5, atol=0.0)


# Risk aversion and lottery scale at which the naive
# `inverse(Σ w · transform(v))` route overflows the dtype.
_FLOAT64_ACTION_CASES = [(8.0, 1e-50), (12.0, 1e-30), (20.0, 1e-20), (50.0, 1e-8)]
_FLOAT32_ACTION_CASES = [(8.0, 1e-8), (12.0, 1e-5), (20.0, 1e-3)]


def _assert_the_even_lottery_wins(
    *,
    risk_aversion: float,
    scale: float,
    dtype: Any,
    rtol: float,
) -> None:
    """Assert `Q` ranks two actions over a `(scale, 2 * scale)` regime lottery.

    Both actions face the same two-point lottery under different regime
    probabilities. The even lottery has the higher power mean by more than
    the skewed action's utility advantage, so it is optimal at every scale.
    """
    values = (scale, 2.0 * scale)

    def certainty_equivalent(weights: tuple[float, float]) -> float:
        return float(
            PowerMean().aggregate(
                values=jnp.asarray(values, dtype=dtype),
                weights=jnp.asarray(weights, dtype=dtype),
                params={"risk_aversion": jnp.asarray(risk_aversion, dtype=dtype)},
            )
        )

    skewed = certainty_equivalent((0.9, 0.1))
    even = certainty_equivalent((0.5, 0.5))
    assert even > skewed > 0.0
    utility_advantage = 0.4 * (even - skewed)

    Q_and_F = _build_two_target_closure(get_Q_and_F, certainty_equivalent=PowerMean())
    Q_per_action = jax.jit(
        lambda: jnp.asarray(
            [
                Q_and_F(
                    **_two_target_call_kwargs(
                        values=values,
                        regime_prob_low=prob_low,
                        utility_level=utility,
                        risk_aversion=risk_aversion,
                        dtype=dtype,
                    )
                )[0]
                for prob_low, utility in ((0.9, utility_advantage), (0.5, 0.0))
            ]
        )
    )()
    expected = jnp.asarray([utility_advantage + skewed, even], dtype=dtype)
    np.testing.assert_allclose(
        np.asarray(Q_per_action), np.asarray(expected), rtol=rtol
    )
    assert int(jnp.argmax(Q_per_action)) == 1


@pytest.mark.parametrize(("risk_aversion", "scale"), _FLOAT64_ACTION_CASES)
def test_bellman_prefers_the_higher_certainty_equivalent_at_any_scale(
    x64_enabled: None,
    risk_aversion: float,
    scale: float,
):
    """The action with the larger certainty equivalent wins however small values are."""
    _assert_the_even_lottery_wins(
        risk_aversion=risk_aversion, scale=scale, dtype=jnp.float64, rtol=8e-5
    )


@pytest.mark.parametrize(("risk_aversion", "scale"), _FLOAT32_ACTION_CASES)
def test_bellman_prefers_the_higher_certainty_equivalent_at_any_scale_float32(
    x64_disabled: None,
    risk_aversion: float,
    scale: float,
):
    """The float32 Bellman ranks the same two actions the same way."""
    _assert_the_even_lottery_wins(
        risk_aversion=risk_aversion, scale=scale, dtype=jnp.float32, rtol=5e-4
    )


def _tiny_anchor_action_values(
    *,
    risky_values: tuple[float, float],
    risky_prob_low: float,
    safe_value: float,
    dtype: Any,
) -> tuple[float, float]:
    """Return `Q` for a near-degenerate risky action and a deterministic safe one.

    The risky action puts a near-zero probability on a continuation value far
    below the other branch; the safe action pays `safe_value` for certain.
    """
    Q_and_F = _build_two_target_closure(get_Q_and_F, certainty_equivalent=PowerMean())
    risky = Q_and_F(
        **_two_target_call_kwargs(
            values=risky_values,
            regime_prob_low=risky_prob_low,
            utility_level=0.0,
            risk_aversion=8.0,
            dtype=dtype,
        )
    )[0]
    safe = Q_and_F(
        **_two_target_call_kwargs(
            values=(safe_value, safe_value),
            regime_prob_low=0.5,
            utility_level=0.0,
            risk_aversion=8.0,
            dtype=dtype,
        )
    )[0]
    return float(risky), float(safe)


def test_bellman_keeps_the_safe_action_at_a_tiny_anchor_weight_float64(
    x64_enabled: None,
):
    """A near-zero-probability low branch stays finite and loses to a safe action."""
    risky, safe = _tiny_anchor_action_values(
        risky_values=(1e-50, 1.0),
        risky_prob_low=1e-20,
        safe_value=1e-40,
        dtype=jnp.float64,
    )
    assert safe > risky


def test_bellman_keeps_the_safe_action_at_a_tiny_anchor_weight_float32(
    x64_disabled: None,
):
    """The float32 Bellman comparison keeps the safe action too."""
    risky, safe = _tiny_anchor_action_values(
        risky_values=(1e-8, 1.0),
        risky_prob_low=1e-8,
        safe_value=1e-4,
        dtype=jnp.float32,
    )
    assert safe > risky


def test_bellman_tiny_anchor_weight_matches_the_oracle_float64(x64_enabled: None):
    """The near-degenerate float64 regime lottery evaluates to `7.1969e-48`."""
    risky, _ = _tiny_anchor_action_values(
        risky_values=(1e-50, 1.0),
        risky_prob_low=1e-20,
        safe_value=1e-40,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(risky, 7.196856730011521e-48, rtol=1e-12, atol=0.0)


def test_bellman_tiny_anchor_weight_matches_the_oracle_float32(x64_disabled: None):
    """The near-degenerate float32 regime lottery evaluates to `1.3895e-7`."""
    risky, _ = _tiny_anchor_action_values(
        risky_values=(1e-8, 1.0),
        risky_prob_low=1e-8,
        safe_value=1e-4,
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(risky, 1.3894954943731376e-7, rtol=5e-5, atol=0.0)


def test_diagnostic_intermediates_reproduce_the_Bellman_Q(x64_enabled: None):
    """The NaN diagnostics recompute the same `Q` the backward induction used."""
    call_kwargs = _two_target_call_kwargs(
        values=(1e-50, 2e-50),
        regime_prob_low=0.25,
        utility_level=3e-51,
        risk_aversion=8.0,
        dtype=jnp.float64,
    )
    Q_and_F = _build_two_target_closure(get_Q_and_F, certainty_equivalent=PowerMean())
    compute_intermediates = _build_two_target_closure(
        get_compute_intermediates, certainty_equivalent=PowerMean()
    )
    Q_arr = jax.jit(lambda: Q_and_F(**call_kwargs)[0])()
    diagnostic_Q_arr = jax.jit(lambda: compute_intermediates(**call_kwargs)[3])()
    np.testing.assert_allclose(
        np.asarray(diagnostic_Q_arr), np.asarray(Q_arr), rtol=0.0, atol=0.0
    )
