"""Canonical runtime-bound transport survives actual solver and subject evaluations."""

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Never, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import transition_checks
from _lcm.params.edges import regime_kernel_params
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.regime_building import processing
from _lcm.regime_building.collective import build_pareto_weights
from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.regime_building.Q_and_F import (
    BoundedQAndF,
    get_Q_and_F_terminal,
    get_Q_and_F_terminal_collective,
)
from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.programs import _SubjectTiled
from _lcm.simulation.simulate import _canonical_Q_at_actions, _read_external_replay
from _lcm.simulation.transitions import create_regime_state_action_space
from _lcm.solution.action_streaming import (
    build_partitioned_streaming_max_Q_over_a,
    build_streaming_collective_max_Q_over_a,
    build_streaming_ev1_max_Q_over_a,
    build_streaming_max_Q_over_a,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    CountedGridSearch,
    FeasibilityKernelOutput,
)
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    QAndFFunction,
    QualifiedName,
)
from lcm import (
    AgeGrid,
    ByAge,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    Model,
    PeriodSpecializedFunction,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
    ref,
)
from lcm.collective import CollectiveUtility, ParetoObjective, ValueDependentConstraint
from lcm.exceptions import (
    InvalidValueFunctionError,
    OffGridStateError,
)
from lcm.solver_api import ActionOutput
from lcm.transition import AgeSpecializedFunction, StochasticTransition
from lcm.typing import (
    ActionName,
    BoolND,
    FloatND,
    Intermediate,
    IntND,
    RegimeName,
    ScalarInt,
    StateName,
    UserFunction,
    ValueND,
)
from tests.constraints.test_dynamic_lower_bound import (
    LIQUID,
    RegimeId,
    TemporalRegimeId,
    _model,
    _temporal_model,
    floor_helper,
    lower,
)
from tests.solution.test_dynamic_bound_authority import _bound


class _StateOwnerKwargs(TypedDict, total=False):
    co_map_state_names: tuple[StateName, ...]
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...]


class _TerminalKwargs(TypedDict):
    flat_param_names: frozenset[QualifiedName]
    functions: EconFunctionsMapping
    constraints: ConstraintFunctionsMapping
    bound_specs: tuple[PostDecisionBoundSpec, ...]


@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("kept", [False, True])
def test_public_policy_lookup_keeps_integrity_with_retained_action_parts(
    *,
    budgeted: bool,
    kept: bool,
) -> None:
    """
    Policy and retained action channels accept the same actual floor before
    publication.
    """

    def floor(*, income: FloatND, period: ScalarInt) -> FloatND:
        return jnp.where((period == 1) & (income == 0.5), jnp.nan, 0.0)

    model = _temporal_model(
        lower_function=floor,
        execution_config=ExecutionConfig(
            device_memory_bytes=(1 << 28) if budgeted else None,
            axis_widths={"subject": 1} if budgeted else {},
        ),
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    intermediates: tuple[Intermediate, ...] = ("Q", "F", "U", "CE") if kept else ()
    result = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="saving",
        states={
            "wealth": jnp.asarray([2.0, 2.0, 2.0]),
            "income": jnp.asarray([1.0, 1.0, 1.0]),
        },
        keep_intermediates=intermediates,
    )
    assert result.actions is not None
    np.testing.assert_array_equal(result.actions["consumption"], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(result.value, [2.0, 2.0, 2.0])
    if kept:
        expected = np.broadcast_to(np.arange(0.25, 2.25, 0.25), (3, 8))
        np.testing.assert_array_equal(result.Q, expected)
        np.testing.assert_array_equal(result.U, expected)
        np.testing.assert_array_equal(result.CE, np.zeros((3, 8)))
        np.testing.assert_array_equal(result.F, np.ones((3, 8), dtype=bool))
    else:
        assert result.Q is result.F is result.U is result.CE is None
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=1,
            regime_name="saving",
            states={
                "wealth": jnp.asarray([2.0, 2.0, 2.0]),
                "income": jnp.asarray([1.0, 1.0, 0.5]),
            },
            keep_intermediates=intermediates,
        )


def identity(*, wealth: FloatND) -> FloatND:
    return wealth


def identity_income(*, income: FloatND | IntND) -> FloatND | IntND:
    return income


def _floor_landing_model(
    *,
    handwritten: bool,
    invalid: bool,
    budgeted: bool,
    value_aware: bool,
    gated: bool = False,
    invariant: bool = False,
    raised: bool = True,
) -> Model:
    def floor(*, income: FloatND | IntND) -> FloatND:
        return jnp.where(invalid & (income == 1.0), jnp.nan, 0.0)

    def resources(*, wealth: FloatND) -> FloatND:
        return wealth

    def savings(*, resources: FloatND, consumption: FloatND) -> FloatND:
        return resources - consumption

    def utility(*, consumption: FloatND, income: FloatND | IntND) -> FloatND:
        return consumption + 0.0 * income

    def next_wealth(*, savings: FloatND) -> FloatND:
        return savings + 3.0

    def terminal(*, wealth: FloatND, income: FloatND | IntND) -> FloatND:
        return wealth + 0.0 * income

    def participation(*, Q_alice: FloatND) -> BoolND:
        return Q_alice >= -jnp.inf

    def gate(*, wealth: FloatND) -> BoolND:
        return wealth >= -jnp.inf

    def certain(*, age: FloatND) -> FloatND:
        return jnp.ones_like(age, dtype=float)

    @categorical(ordered=True)
    class Income:
        low: ScalarInt
        high: ScalarInt

    @categorical(ordered=False)
    class GatedRegimeId:
        saving: ScalarInt
        done: ScalarInt
        fallback_alice: ScalarInt
        fallback_bob: ScalarInt

    income_grid = (
        DiscreteGrid(category_class=Income)
        if invariant
        else LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    )
    edge = (
        Transition(
            targets={
                "done": 0,
                **({"fallback_alice": 0, "fallback_bob": 0} if value_aware else {}),
            },
            law=ByAge(cases={0: {"done": StochasticTransition(func=certain)}}),
            gates={
                "done": Gate(
                    predicate=gate,
                    routes={
                        name: StakeholderRoute(
                            target_stakeholder=name if value_aware else None,
                            fallback=ProjectedRegimeValue(
                                regime=f"fallback_{name}" if value_aware else "done",
                                projection={
                                    "wealth": identity,
                                    "income": identity_income,
                                },
                            ),
                        )
                        for name in (("alice", "bob") if value_aware else ("only",))
                    },
                    off_grid="pointwise",
                )
            },
        )
        if gated
        else {"done": 0}
    )
    return Model(
        regimes={
            "saving": ConsumptionSavingsRegime(
                liquid=LIQUID,
                states={
                    "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
                    "income": income_grid,
                },
                actions={"consumption": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
                state_transitions={
                    "wealth": next_wealth,
                    "income": fixed_transition("income"),
                },
                functions={
                    "utility": CollectiveUtility(
                        utilities={"alice": utility, "bob": utility}
                    )
                    if value_aware
                    else utility,
                    "resources": resources,
                    "savings": savings,
                    **({"floor": floor} if handwritten else {}),
                },
                constraints={
                    "borrowing": ref("savings") >= ref("floor")
                    if handwritten
                    else post_decision_lower_bound(margin=LIQUID, lower=floor),
                    **(
                        {
                            "participation": ValueDependentConstraint(
                                predicate=participation
                            )
                        }
                        if value_aware
                        else {}
                    ),
                },
            ),
            "done": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0.0,
                        stop=2.0,
                        n_points=2,
                        off_grid="raise" if raised else "extrapolate",
                    ),
                    "income": income_grid,
                },
                functions={
                    "utility": CollectiveUtility(
                        utilities={"alice": terminal, "bob": terminal}
                    )
                    if value_aware
                    else terminal
                },
            ),
            **(
                {
                    f"fallback_{name}": Regime(
                        states={
                            "wealth": LinSpacedGrid(
                                start=0.0, stop=2.0, n_points=2, off_grid="raise"
                            ),
                            "income": income_grid,
                        },
                        functions={"utility": terminal},
                    )
                    for name in ("alice", "bob")
                }
                if gated and value_aware
                else {}
            ),
        },
        ages=AgeGrid(exact_values=(0, 1)),
        edges={"saving": edge},
        initial_nodes={0: "saving"},
        regime_id_class=GatedRegimeId if gated and value_aware else TemporalRegimeId,
        execution_config=ExecutionConfig(
            device_memory_bytes=(1 << 28) if budgeted else None,
            invariant_block_widths={"income": 1} if invariant else {},
        ),
    )


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("value_aware", [False, True])
def test_floor_integrity_precedes_reached_landing_errors(
    *,
    runtime_checks: bool,
    handwritten: bool,
    invalid: bool,
    budgeted: bool,
    monkeypatch: pytest.MonkeyPatch,
    value_aware: bool,
) -> None:
    """
    Mandatory floor integrity is checked before reached off-grid landings are
    interpreted.
    """
    monkeypatch.setattr(transition_checks, "_OFF_GRID_BLOCK_ROWS", 1)
    monkeypatch.setattr(transition_checks, "_OFF_GRID_MAX_BLOCK_ROWS", 1)
    model = _floor_landing_model(
        handwritten=handwritten,
        invalid=invalid,
        budgeted=budgeted,
        value_aware=value_aware,
    )
    error = InvalidValueFunctionError if invalid else OffGridStateError
    with pytest.raises(error, match=r"floor|state='wealth'"):
        model.solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("value_aware", [False, True])
def test_gated_floor_integrity_precedes_reached_landing_errors(
    *,
    runtime_checks: bool,
    handwritten: bool,
    invalid: bool,
    value_aware: bool,
) -> None:
    """
    Actually gated landings retain own-floor integrity outside gate and
    participation masks.
    """
    model = _floor_landing_model(
        handwritten=handwritten,
        invalid=invalid,
        budgeted=True,
        value_aware=value_aware,
        gated=True,
    )
    error = InvalidValueFunctionError if invalid else OffGridStateError
    with pytest.raises(error, match=r"floor|state='wealth'"):
        model.solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("handwritten", [False, True])
def test_invariant_codes_reduce_floor_integrity_before_landing_interpretation(
    *,
    runtime_checks: bool,
    handwritten: bool,
) -> None:
    """One-code execution keeps invalidity from another code ahead of landing errors."""
    model = _floor_landing_model(
        handwritten=handwritten,
        invalid=True,
        budgeted=True,
        value_aware=False,
        invariant=True,
    )
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_actual_invariant_dispatch_keeps_floor_integrity_across_code_outputs(
    *,
    runtime_checks: bool,
    handwritten: bool,
    invalid: bool,
) -> None:
    """
    Separate compiled code outputs retain mandatory integrity beside unchanged
    values.
    """
    model = _floor_landing_model(
        handwritten=handwritten,
        invalid=invalid,
        budgeted=True,
        value_aware=False,
        invariant=True,
        raised=False,
    )
    if invalid:
        with pytest.raises(InvalidValueFunctionError, match="floor"):
            model.solve(
                params={"discount_factor": 1.0},
                log_level="off",
                runtime_checks=runtime_checks,
            )
    else:
        solution = model.solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )
        np.testing.assert_array_equal(
            solution.values[0]["saving"], [[4.0, 5.0], [4.0, 5.0]]
        )


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_streamed_ev1_owner_keeps_integrity_outside_logsum_mass(
    *,
    invalid: bool,
    width: int,
) -> None:
    """
    EV1 branch streaming retains actual floor validity beside the same logsum
    value.
    """

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=kernel,
        action_names=("consumption",),
        n_discrete_action_axes=1,
        block_width=width,
        scale=0.5,
    )
    result = jax.jit(
        lambda: streamed(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=jnp.asarray(jnp.nan if invalid else -3.0),
            consumption=jnp.asarray([1.0, 2.0]),
        )
    )()
    assert type(result) is BoundedResult
    expected = -np.inf if invalid else 2.0 + 0.5 * np.log1p(np.exp(-2.0))
    np.testing.assert_allclose(result.output.smoothed_value, expected, rtol=1e-6)
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [invalid, False])


@pytest.mark.parametrize("invalid", [False, True])
def test_action_partitions_keep_empty_owners_neutral_to_floor_integrity(
    *,
    invalid: bool,
) -> None:
    """
    Empty action owners are neutral while actual owned invalid floors survive the
    exchange.
    """

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    partitioned = build_partitioned_streaming_max_Q_over_a(
        Q_and_F=kernel,
        action_names=("consumption",),
        block_width=1,
        n_partitions=4,
        axis_name="owners",
    )
    arguments = {
        "next_regime_to_V_arr": MappingProxyType({}),
        "period": jnp.int32(0),
        "age": jnp.asarray(0.0),
        "income": jnp.asarray(jnp.nan if invalid else -3.0),
        "consumption": jnp.asarray([1.0, 2.0]),
    }
    local = jax.jit(jax.vmap(lambda p: partitioned.local(partition=p, **arguments)))(
        jnp.arange(4, dtype=jnp.int32)
    )
    assert type(local) is BoundedResult
    np.testing.assert_array_equal(
        local.bound_status[0].nonfinite_floor, [False, invalid, False, invalid]
    )
    merged = jax.jit(jax.vmap(lambda _: partitioned(**arguments), axis_name="owners"))(
        jnp.arange(4, dtype=jnp.int32)
    )
    assert type(merged) is BoundedResult
    np.testing.assert_array_equal(
        merged.output.best_value, [-np.inf if invalid else 2.0] * 4
    )
    np.testing.assert_array_equal(merged.bound_status[0].nonfinite_floor, [invalid] * 4)
    np.testing.assert_array_equal(
        merged.bound_status[0].invalid_shifted_grid, [False] * 4
    )


@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("clock", ["age", "period"])
@pytest.mark.parametrize("entry", [0, 1])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_decorated_floor_cohorts_never_use_wrong_period_evaluator(
    *, handwritten: bool, clock: str, entry: int, runtime_checks: bool
) -> None:
    """Decorated floors reject nonfinite values at actual subject decision periods."""

    def build(coordinate: float) -> UserFunction:
        def floor(*, income: FloatND) -> FloatND:
            return jnp.where((coordinate == 1) & (income == 0.5), jnp.nan, 0.0)

        return floor

    marker = AgeSpecializedFunction if clock == "age" else PeriodSpecializedFunction
    model = _temporal_model(
        lower_function=marker(build=build, signature=lambda coordinate: coordinate),
        clock=clock,
        handwritten=handwritten,
        initial_entries=(0, 1),
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(solution.values[1]["saving"], [[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.asarray([2.0]),
                "income": jnp.asarray([0.5]),
                "age" if clock == "age" else "period": jnp.asarray([entry]),
                "regime_id": jnp.asarray([TemporalRegimeId.saving], dtype=jnp.int32),
            },
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("streamed", [False, True])
def test_household_action_owner_keeps_integrity_outside_pareto_weights(
    *, streamed: bool
) -> None:
    """Zero household weight cannot hide invalid shared bound arithmetic."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility_a(*, consumption: FloatND) -> FloatND:
        return consumption

    def utility_b(*, consumption: FloatND) -> FloatND:
        return 3.0 * consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal_collective(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility_a": cast("EconFunction", utility_a),
                "utility_b": cast("EconFunction", utility_b),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        stakeholders=("a", "b"),
        bound_specs=(_bound(),),
    )
    weights = build_pareto_weights(
        objective=ParetoObjective(weights={"a": 1.0, "b": 0.0}),
        stakeholders=("a", "b"),
        state_names=frozenset({"income"}),
    )
    builder = get_streaming_max_Q_over_a if streamed else get_max_Q_over_a
    mapped = builder(
        Q_and_F=kernel,
        batch_sizes={"income": 1},
        action_names=("consumption",),
        state_names=("income",),
        stakeholders=("a", "b"),
        pareto_weights=weights,
        return_feasibility=True,
    )
    result = jax.jit(
        lambda: mapped(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=jnp.asarray([-1.0, jnp.nan]),
            consumption=jnp.asarray([1.0, 2.0]),
            **({"_lcm_action_block_width": 1} if streamed else {}),
        )
    )()
    assert type(result) is BoundedResult
    np.testing.assert_array_equal(result.output[0], [[1.0, 3.0], [-np.inf, -np.inf]])
    np.testing.assert_array_equal(result.output[1], [False, True])
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [True, False])


def test_ev1_dense_owner_retains_invalid_infeasible_floor() -> None:
    """The canonical EV1 reduction transports validity independently of logsum mass."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset({"taste_shocks__scale"}),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    mapped = get_max_Q_over_a(
        Q_and_F=kernel,
        batch_sizes={"income": 1},
        action_names=("consumption",),
        state_names=("income",),
        has_taste_shocks=True,
        n_discrete_action_axes=1,
        return_feasibility=True,
    )
    result = jax.jit(
        lambda: mapped(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=jnp.asarray([-1.0, jnp.nan]),
            consumption=jnp.asarray([1.0, 2.0]),
            taste_shocks__scale=jnp.asarray(0.5),
        )
    )()
    assert type(result) is BoundedResult
    np.testing.assert_array_equal(result.output[0], [1.0, -np.inf])
    np.testing.assert_array_equal(result.output[1], [False, True])
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [True, False])


@pytest.mark.parametrize("streamed", [False, True])
def test_action_free_owner_preserves_mandatory_integrity(*, streamed: bool) -> None:
    """
    An action-free decision retains actual bound validity beside its ordinary
    value.
    """

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    mapped = (get_streaming_max_Q_over_a if streamed else get_max_Q_over_a)(
        Q_and_F=kernel,
        batch_sizes={"income": 1},
        action_names=(),
        state_names=("income",),
        return_feasibility=True,
    )
    result = jax.jit(
        lambda: mapped(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=jnp.asarray([-1.0, jnp.nan]),
            consumption=jnp.asarray(1.0),
            **({"_lcm_action_block_width": 1} if streamed else {}),
        )
    )()
    assert type(result) is BoundedResult
    np.testing.assert_array_equal(result.output[0], [1.0, -np.inf])
    np.testing.assert_array_equal(result.output[1], [False, True])
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [True, False])


@pytest.mark.parametrize("handwritten", [False, True])
def test_supplied_actual_actions_cannot_drop_invalid_subject_floor(
    *, handwritten: bool
) -> None:
    """
    Canonical scoring rejects actual invalid inputs before interpreting supplied
    actions.
    """

    def floor(*, income: FloatND) -> FloatND:
        return jnp.where(income == 0.5, jnp.nan, 0.0)

    model = _model(
        declaration=ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=floor),
        helper=None,
        functions={"floor": floor} if handwritten else None,
    )
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        _canonical_Q_at_actions(
            candidate_actions={"consumption": jnp.asarray([1.0, 1.0, 1.0])},
            regime=model._regimes["done"],
            canonical_states={
                "wealth": jnp.asarray([2.0, 2.0, 2.0]),
                "income": jnp.asarray([0.0, 0.5, 1.0]),
            },
            action_names=("consumption",),
            next_regime_to_V_arr=MappingProxyType({}),
            flat_params=regime_kernel_params(
                model._process_params({}, runtime_checks=False),
                regime_name="done",
            ),
            period=0,
            age=jnp.asarray(0.0),
        )


@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_simulation_rejects_invalid_floor_only_at_actual_subject(
    *, handwritten: bool, runtime_checks: bool
) -> None:
    """A valid solve grid cannot authorize a nonfinite floor at an actual subject."""

    def floor(*, income: FloatND) -> FloatND:
        return jnp.where(income == 0.5, jnp.nan, 0.0)

    model = _model(
        declaration=ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=floor),
        helper=None,
        functions={"floor": floor} if handwritten else None,
    )
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_allclose(solution.values[0]["done"], [[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "wealth": jnp.asarray([2.0, 2.0, 2.0]),
                "income": jnp.asarray([0.0, 0.5, 1.0]),
                "age": jnp.asarray([0, 0, 0]),
                "regime_id": jnp.asarray([RegimeId.done] * 3, dtype=jnp.int32),
            },
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_later_subject_decision_rejects_invalid_floor_before_empty_actions(
    *, handwritten: bool, runtime_checks: bool
) -> None:
    """A floor valid at entry remains mandatory at the subject's later decision."""

    def floor(*, income: FloatND, period: ScalarInt) -> FloatND:
        return jnp.where((period == 1) & (income == 0.5), jnp.nan, 0.0)

    model = _temporal_model(lower_function=floor, handwritten=handwritten)
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(solution.values[1]["saving"], [[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.asarray([2.0]),
                "income": jnp.asarray([0.5]),
                "age": jnp.asarray([0]),
                "regime_id": jnp.asarray([TemporalRegimeId.saving], dtype=jnp.int32),
            },
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("width", [1, 2, 8])
def test_subject_tiles_fold_flags_before_concatenating_ordinary_rows(
    *, width: int
) -> None:
    """A partial subject tile keeps ordinary rows and two scalar Boolean flags."""
    assert "n_bound_records" in _SubjectTiled.__dataclass_fields__, (
        "Subject tile owners need explicit mandatory-summary declarations."
    )

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    tiled = _SubjectTiled(
        func=BoundedQAndF(function=kernel),
        subject_arg_names=("income", "consumption"),
        n_bound_records=1,
    )
    result = jax.jit(
        lambda: tiled(
            next_regime_to_V_arr=MappingProxyType({}),
            age=jnp.asarray(0.0),
            period=jnp.int32(0),
            income=jnp.asarray([-1.0, jnp.nan, 0.0]),
            consumption=jnp.asarray([1.0, 1.0, 1.0]),
            **{SUBJECT_WIDTH_KEYWORD: width},
        )
    )()
    np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result[1], [True, False, False])
    assert result[2].shape == result[3].shape == ()
    np.testing.assert_array_equal(jnp.stack(result[2:]), [True, False])


def test_counted_value_owner_preserves_integrity_beside_two_counts() -> None:
    """Mandatory validity is neither an empty-state count nor another infinity count."""

    def values() -> BoundedResult[tuple[FloatND, BoolND]]:
        return BoundedResult(
            output=(
                jnp.asarray([-jnp.inf, -jnp.inf, 2.0]),
                jnp.asarray([False, True, False]),
            ),
            bound_status=(
                BoundInputStatus(
                    nonfinite_floor=jnp.asarray(a=True),
                    invalid_shifted_grid=jnp.asarray(a=False),
                ),
            ),
        )

    counted = CountedGridSearch(function=values, collective=False)
    result = jax.jit(counted)()
    assert type(result) is BoundedResult
    np.testing.assert_array_equal(result.output[0], [-np.inf, -np.inf, 2.0])
    np.testing.assert_array_equal(jnp.stack(result.output[1:]), [1, 1])
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [True, False])


@pytest.mark.parametrize("invalid", [False, True])
def test_kernel_output_accepts_only_scalar_boolean_integrity(*, invalid: bool) -> None:
    """Kernel output summaries are scalar Booleans before their host acceptance seam."""
    assert "bound_status" in FeasibilityKernelOutput.__dataclass_fields__, (
        "The private kernel output must transport mandatory integrity."
    )
    template = BoundInputStatus(
        nonfinite_floor=jnp.asarray(a=False),
        invalid_shifted_grid=jnp.asarray(a=False),
    )
    status = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(template),
        [
            jnp.asarray([True, False]) if invalid else jnp.asarray(a=True),
            jnp.asarray(a=False),
        ],
    )

    def output() -> FeasibilityKernelOutput:
        return FeasibilityKernelOutput(
            value=jnp.asarray([2.0]),
            feasibility_counts=(jnp.int32(0), jnp.int32(0)),
            bound_status=(status,),
        )

    if invalid:
        with pytest.raises(TypeError, match="scalar Boolean"):
            output()
    else:
        np.testing.assert_array_equal(
            jnp.stack(output().bound_status[0]), [True, False]
        )


@pytest.mark.parametrize("streamed", [False, True])
def test_state_axis_owners_keep_integrity_independent_of_quadrature(
    *, streamed: bool
) -> None:
    """Co-maps produce scalar summaries that keep each cell's invalidity."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    builder = get_streaming_max_Q_over_a if streamed else get_max_Q_over_a
    owner_kwargs: _StateOwnerKwargs = {
        "co_map_state_names": ("income",),
        "co_map_v_arr_in_axes": (MappingProxyType({"future": 0}),),
    }
    mapped = builder(
        Q_and_F=kernel,
        batch_sizes={"income": 1},
        action_names=("consumption",),
        state_names=("income",),
        return_feasibility=True,
        **owner_kwargs,
    )
    result = jax.jit(
        lambda: mapped(
            next_regime_to_V_arr=MappingProxyType({"future": jnp.asarray([0.0, 0.0])}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=jnp.asarray([0.0, jnp.nan]),
            consumption=jnp.asarray([0.0, 1.0]),
            **({"_lcm_action_block_width": 1} if streamed else {}),
        )
    )()
    assert type(result) is BoundedResult
    assert all(flag.shape == () for flag in result.bound_status[0]), (
        "State owners must reduce validity without retaining a coordinate bank."
    )
    np.testing.assert_array_equal(result.output[0], [0.0, -np.inf])
    np.testing.assert_array_equal(result.output[1], [False, True])
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [True, False])


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("width", [1, 2, 8])
def test_actual_action_owners_preserve_invalid_infeasible_floor(
    *, streamed: bool, width: int
) -> None:
    """An infeasible action's nonfinite RHS survives action and state reductions."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    builder = get_streaming_max_Q_over_a if streamed else get_max_Q_over_a
    mapped = builder(
        Q_and_F=kernel,
        batch_sizes={"consumption": 1},
        action_names=("income",),
        state_names=("consumption",),
        return_feasibility=True,
    )
    result = jax.jit(
        lambda: mapped(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.asarray(0, dtype=jnp.int32),
            age=jnp.asarray(0.0),
            income=jnp.asarray([-1.0, jnp.nan, 0.0]),
            consumption=jnp.asarray([1.0, 2.0]),
            **({"_lcm_action_block_width": width} if streamed else {}),
        )
    )()
    assert type(result) is BoundedResult, (
        "Action owners must retain mandatory validity."
    )
    np.testing.assert_array_equal(result.output[0], [1.0, -np.inf])
    np.testing.assert_array_equal(result.output[1], [False, True])
    np.testing.assert_array_equal(
        jnp.stack(result.bound_status[0]),
        [True, False],
    )


@pytest.mark.parametrize("spelling", ["sugar", "handwritten", "literal"])
def test_canonical_action_builders_receive_only_runtime_bound_authority(
    *, spelling: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both actual phase builders bind DAG floors while literal models stay ordinary."""
    original = processing.get_Q_and_F_terminal
    requests = []

    def capture(*args: Never, **kwargs: Unpack[_TerminalKwargs]) -> QAndFFunction:
        requests.append(kwargs.get("bound_specs", ()))
        return original(*args, **kwargs)

    monkeypatch.setattr(processing, "get_Q_and_F_terminal", capture)
    declaration = (
        ref("savings") >= ref("floor")
        if spelling == "handwritten"
        else post_decision_lower_bound(
            margin=LIQUID,
            lower=0.0 if spelling == "literal" else lower,
        )
    )
    model = _model(
        declaration=declaration,
        helper=None if spelling == "literal" else floor_helper,
        functions={"floor": lower} if spelling == "handwritten" else None,
    )
    canonical = model._regimes["done"].solution.post_decision_bounds
    expected = tuple(bound for bound in canonical if bound.kind == "dag")
    assert len(requests) == 2
    assert tuple(requests) == (expected, expected)


_MARKER = "stablehlo.round_nearest_even"
_MARKED_CONSUMPTION = (0.3, 0.9, 1.4, 2.2, 2.6, 3.1, 3.7)
_STREAM_ROUTES = ("hard_max", "collective", "partitioned", "ev1")


@pytest.mark.parametrize("route", _STREAM_ROUTES)
@pytest.mark.parametrize("block_width", [1, 3])
def test_bounded_action_streams_lower_q_and_f_once(
    *, route: str, block_width: int
) -> None:
    """A bounded stream over several blocks places one copy of `Q_and_F`."""
    run = _marked_stream(route=route, bounded=True, block_width=block_width)
    lowered = jax.jit(run).lower(jnp.asarray(-3.0)).as_text()
    assert lowered.count(_MARKER) == 1


@pytest.mark.parametrize("route", _STREAM_ROUTES)
@pytest.mark.parametrize("block_width", [7, 14, 3, 2, 1])
@pytest.mark.parametrize("income", [-3.0, np.nan], ids=["valid", "invalid"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_bounded_action_streams_match_the_unbounded_stream(
    *, route: str, block_width: int, income: float, compiled: bool
) -> None:
    """Bounded streams publish the unbounded stream's bits and the floor's validity."""
    bounded, plain = (
        _marked_stream(route=route, bounded=flag, block_width=block_width)
        for flag in (True, False)
    )
    if compiled:
        bounded, plain = jax.jit(bounded), jax.jit(plain)
    result = bounded(jnp.asarray(income))
    expected = plain(jnp.asarray(income))
    assert type(result) is BoundedResult
    actual = [*jax.tree.leaves(result.output), *result.bound_status[0]]
    wanted = [*jax.tree.leaves(expected), np.isnan(income), False]
    assert len(actual) == len(wanted)
    for got, want in zip(actual, wanted, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _marked_stream(*, route: str, bounded: bool, block_width: int):
    """Build one action stream whose utility carries a single rounding marker."""
    kernel = _marked_kernel(route=route, bounded=bounded)
    work = jnp.asarray([0.0, 1.0])
    consumption = jnp.asarray(_MARKED_CONSUMPTION)
    action_names = ("work", "consumption")
    if route == "hard_max":
        stream = build_streaming_max_Q_over_a(
            Q_and_F=kernel, action_names=action_names, block_width=block_width
        )
    elif route == "collective":
        stream = build_streaming_collective_max_Q_over_a(
            Q_and_F=kernel,
            action_names=action_names,
            block_width=block_width,
            stakeholders=("a", "b"),
            weights={"a": 1.0, "b": 0.5},
        )
    elif route == "ev1":
        stream = build_streaming_ev1_max_Q_over_a(
            Q_and_F=kernel,
            action_names=action_names,
            n_discrete_action_axes=1,
            block_width=block_width,
            scale=0.5,
        )
    else:
        partitioned = build_partitioned_streaming_max_Q_over_a(
            Q_and_F=kernel,
            action_names=action_names,
            block_width=block_width,
            n_partitions=2,
            axis_name="owners",
        )

        def run_partitioned(income: FloatND):
            return jax.vmap(
                lambda _: partitioned(
                    next_regime_to_V_arr=MappingProxyType({}),
                    period=jnp.int32(0),
                    age=jnp.asarray(0.0),
                    income=income,
                    work=work,
                    consumption=consumption,
                ),
                axis_name="owners",
            )(jnp.arange(2, dtype=jnp.int32))

        return run_partitioned

    def run(income: FloatND):
        return stream(
            next_regime_to_V_arr=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.asarray(0.0),
            income=income,
            work=work,
            consumption=consumption,
        )

    return run


def _marked_kernel(*, route: str, bounded: bool) -> QAndFFunction:
    """Build a terminal kernel whose utility carries a single rounding marker."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, work: FloatND, consumption: FloatND) -> FloatND:
        rounded = jax.lax.round(
            4.0 * consumption, jax.lax.RoundingMethod.TO_NEAREST_EVEN
        )
        return rounded - 0.1 * consumption * consumption + 0.5 * work

    def utility_b(*, consumption: FloatND) -> FloatND:
        return 2.0 - consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    constraints = MappingProxyType({"borrowing": cast("ConstraintFunction", borrowing)})
    bound_specs = (_bound(),) if bounded else ()
    if route == "collective":
        return get_Q_and_F_terminal_collective(
            flat_param_names=frozenset(),
            functions=MappingProxyType(
                {
                    "utility_a": cast("EconFunction", utility),
                    "utility_b": cast("EconFunction", utility_b),
                    "floor": cast("EconFunction", floor),
                }
            ),
            constraints=constraints,
            stakeholders=("a", "b"),
            bound_specs=bound_specs,
        )
    return get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=constraints,
        bound_specs=bound_specs,
    )


@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("invalid_holder", [False, True])
@pytest.mark.parametrize("fixed_parameter", [False, True])
def test_external_replay_scores_integrity_only_for_actual_regime_holders(
    *, handwritten: bool, invalid_holder: bool, fixed_parameter: bool
) -> None:
    """External actions retain valid holder scores and refuse an actual bad floor."""

    def floor(*, income: FloatND) -> FloatND:
        return jnp.where(income == 0.5, jnp.nan, 0.0)

    model = _model(
        declaration=ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=floor),
        helper=None,
        functions={"floor": floor} if handwritten else None,
    )
    if fixed_parameter:

        def utility(*, consumption: FloatND, multiplier: float) -> FloatND:
            return multiplier * consumption

        declared = model.user_regimes["done"]
        model = Model(
            regimes={
                "done": declared.replace(
                    functions={**declared.functions, "utility": utility}
                )
            },
            edges=model.edges,
            ages=model.ages,
            initial_nodes=model.initial_nodes,
            regime_id_class=RegimeId,
            fixed_params={"done": {"utility": {"multiplier": 1.0}}},
        )
        assert isinstance(model._regimes["done"].simulation.Q_and_F[0], partial)
    regime = model._regimes["done"]
    flat_params = regime_kernel_params(
        model._process_params({}, runtime_checks=False), regime_name="done"
    )
    states = MappingProxyType(
        {
            "wealth": jnp.asarray([2.0, 2.0, 2.0]),
            "income": jnp.asarray([0.0, 0.5, 1.0]),
        }
    )
    holders = jnp.asarray([True, invalid_holder, True])
    space = create_regime_state_action_space(
        regime=regime,
        regime_states=states,
        base=regime.solution.state_action_space(regime_params=flat_params),
    )

    def reader(
        *,
        states: Mapping[StateName, FloatND | IntND],
        fallback_actions: Mapping[ActionName, ValueND],
    ) -> ActionOutput:
        assert tuple(fallback_actions) == ("consumption",)
        return ActionOutput(
            actions={"consumption": jnp.full_like(states["wealth"], 1.0)}
        )

    def evaluate() -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]:
        return _read_external_replay(
            reader=reader,
            regime_name="done",
            regime=regime,
            period=0,
            age=jnp.asarray(0.0),
            states=states,
            state_action_space=space,
            next_regime_to_V_arr=MappingProxyType({}),
            flat_params=flat_params,
            referenced_value_kwargs={},
            subject_ids_in_regime=holders,
        )

    if invalid_holder:
        with pytest.raises(InvalidValueFunctionError, match="floor"):
            evaluate()
        return

    actions, values = evaluate()
    np.testing.assert_array_equal(actions["consumption"], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(values[holders], [1.0, 1.0])
