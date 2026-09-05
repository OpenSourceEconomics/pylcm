"""Tests for planner-owned CoreProgram declarations and resolution."""

import functools
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    CoreProgramGraphAware,
    MaterializedCoreProgram,
    ReductionSemantics,
    ResolvedCoreProgram,
    StreamableProductAxis,
    _TargetValueAccess,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.output_layout import (
    DISSOLUTION_FLAG,
    VALUE,
    resolve_output_layout,
)
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _build_continuation_templates,
    _resolve_value_input_transfer_plan,
)
from lcm.solvers import GridSearch
from lcm.typing import ContinuousState, FloatND
from tests.regime_building.test_collective_feasibility_is_shared import (
    _make_model as _build_collective_model,
)
from tests.test_models import taste_shocks_toy
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    utility,
)

_WIDTH_KEYWORD = "_lcm_action_block_width"

_SCHEDULED_SOURCE = ("source", 0, "main")


def _hard_max_core(
    *,
    first: jnp.ndarray,
    second: jnp.ndarray,
    _lcm_action_block_width: int,
) -> FloatND:
    """Reduce a 2x3 C-order action product using the planner-bound width."""
    first_cells = jnp.repeat(first, second.shape[0])
    second_cells = jnp.tile(second, first.shape[0])
    values = jnp.where(
        ((first_cells == 0) & (second_cells == 30))
        | ((first_cells == 1) & (second_cells == 10)),
        9.0,
        0.0,
    )
    feasible = jnp.ones(values.shape, dtype=bool)
    action_ids = jnp.arange(values.shape[0], dtype=jnp.int32)
    accumulator = HARD_MAX_REDUCTION.initialize(value_template=jnp.asarray(0.0))
    for start in range(0, values.shape[0], _lcm_action_block_width):
        stop = start + _lcm_action_block_width
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=values[start:stop],
            feasible=feasible[start:stop],
            action_ids=action_ids[start:stop],
        )
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator).best_value


def _axis(
    *,
    coordinate_names: tuple[str, ...] = ("first", "second"),
    coordinate_extents: tuple[int, ...] = (2, 3),
    canonical_order: Literal["c"] = "c",
    reduction: ReductionSemantics = HARD_MAX_REDUCTION,
    requested_width: int | None = None,
) -> StreamableProductAxis:
    return StreamableProductAxis(
        name="action",
        coordinate_names=coordinate_names,
        coordinate_extents=coordinate_extents,
        canonical_order=canonical_order,
        reduction=reduction,
        width_keyword=_WIDTH_KEYWORD,
        requested_width=requested_width,
    )


def _program(
    *,
    axis: StreamableProductAxis | None = None,
    arguments: Mapping[str, object] | None = None,
) -> MaterializedCoreProgram:
    if arguments is None:
        arguments = {
            "first": jnp.asarray([0, 1]),
            "second": jnp.asarray([10, 20, 30]),
        }
    return MaterializedCoreProgram(
        name="main",
        function=_hard_max_core,
        arguments=arguments,
        requirements=CoreExecutionRequirements(
            streamable_axes=(_axis() if axis is None else axis,)
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _eval_resolved_shape(resolved: ResolvedCoreProgram) -> object:
    """Evaluate abstract output with the resolver's static choices bound."""
    bound = functools.partial(resolved.function, **resolved.static_kwargs)
    return jax.eval_shape(bound, **resolved.arguments)


def _unused_value_consumer_core(**_arguments: object) -> object:
    """Provide a stable callable for consumer-address planning tests."""
    return jnp.asarray(0.0)


def _value_access(
    *,
    source: tuple[str, int, str],
    target_regime: str = "target",
    channel: ValueInputChannel = ValueInputChannel.NEXT_REGIME_VALUE,
) -> _TargetValueAccess:
    """Build one internally valid target/source value address pair."""
    source_regime, source_period, core_key = source
    target_period = (
        source_period
        if channel is ValueInputChannel.SAME_PERIOD_VALUE
        else source_period + 1
    )
    return _TargetValueAccess(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE,
            period=target_period,
            regime=target_regime,
        ),
        source=ValueConsumerAddress(
            source_regime=source_regime,
            source_period=source_period,
            core_key=core_key,
            channel=channel,
            path=(target_regime,),
        ),
    )


def _value_consumer_program(
    *,
    accesses: tuple[_TargetValueAccess, ...],
    arguments: Mapping[str, object],
) -> MaterializedCoreProgram:
    """Build a synthetic program around exact value-consumer declarations."""
    return MaterializedCoreProgram(
        name="main",
        function=_unused_value_consumer_core,
        arguments=arguments,
        requirements=CoreExecutionRequirements(target_value_accesses=accesses),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


@pytest.mark.parametrize(
    "declared_source",
    [
        pytest.param(("other-source", 0, "main"), id="regime"),
        pytest.param(("source", 1, "main"), id="period"),
        pytest.param(("source", 0, "alternate"), id="core-key"),
    ],
)
def test_value_input_planning_rejects_coordinates_outside_scheduled_core(
    declared_source: tuple[str, int, str],
) -> None:
    """Every declared coordinate must equal the actual compiled core triple."""
    value = jnp.asarray([3.0, 4.0])
    access = _value_access(source=declared_source)
    program = _value_consumer_program(
        accesses=(access,),
        arguments={
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
        },
    )

    with pytest.raises(
        ValueError,
        match="must match the actual compiled core",
    ) as exc_info:
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=value,
            source=_SCHEDULED_SOURCE,
        )

    message = str(exc_info.value)
    assert f"declared={declared_source!r}" in message
    assert f"actual={_SCHEDULED_SOURCE!r}" in message


def test_value_input_planning_rejects_consistently_wrong_consumer_node() -> None:
    """Internal agreement between accesses cannot authenticate a false source node."""
    value = jnp.asarray([3.0, 4.0])
    declared_source = ("other-source", 0, "main")
    next_access = _value_access(
        source=declared_source,
        target_regime="first",
    )
    edge_access = _value_access(
        source=declared_source,
        target_regime="second",
        channel=ValueInputChannel.EDGE_REFERENCE_VALUE,
    )
    program = _value_consumer_program(
        accesses=(next_access, edge_access),
        arguments={
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"first": value},
            ValueInputChannel.EDGE_REFERENCE_VALUE.value: {"second": value},
        },
    )

    with pytest.raises(ValueError, match="must match the actual compiled core"):
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=value,
            source=_SCHEDULED_SOURCE,
        )


@pytest.mark.parametrize(
    ("defect", "error", "message"),
    [
        pytest.param(
            "missing-channel",
            ValueError,
            "input channel.*missing",
            id="missing-channel",
        ),
        pytest.param(
            "missing-path",
            ValueError,
            "argument path.*missing",
            id="missing-path",
        ),
        pytest.param(
            "non-array-leaf",
            TypeError,
            "array-like leaf",
            id="non-array-leaf",
        ),
    ],
)
def test_value_input_planning_independently_resolves_argument_leaf(
    *,
    defect: str,
    error: type[Exception],
    message: str,
) -> None:
    """A matching source triple does not bypass channel/path leaf validation."""
    value = jnp.asarray([3.0, 4.0])
    access = _value_access(source=_SCHEDULED_SOURCE)
    argument_variants: dict[str, Mapping[str, object]] = {
        "missing-channel": {},
        "missing-path": {
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"other": value},
        },
        "non-array-leaf": {
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": object()},
        },
    }
    program = _value_consumer_program(
        accesses=(access,),
        arguments=argument_variants[defect],
    )

    with pytest.raises(error, match=message):
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=value,
            source=_SCHEDULED_SOURCE,
        )


def test_value_input_planning_accepts_exact_scheduled_consumer() -> None:
    """An exact source coordinate and argument leaf produce one transfer."""
    value = jnp.asarray([3.0, 4.0])
    access = _value_access(source=_SCHEDULED_SOURCE)
    program = _value_consumer_program(
        accesses=(access,),
        arguments={
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
        },
    )

    plan = _resolve_value_input_transfer_plan(
        program=program,
        source_value_template=value,
        source=_SCHEDULED_SOURCE,
    )

    assert len(plan) == 1
    assert plan[0].target == access.target
    assert plan[0].source == access.source


def test_value_input_planning_accepts_core_without_value_consumers() -> None:
    """A scheduled core with no declared value reads needs no transfer."""
    value = jnp.asarray([3.0, 4.0])
    program = _value_consumer_program(accesses=(), arguments={})

    assert (
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=value,
            source=_SCHEDULED_SOURCE,
        )
        == ()
    )


@dataclass(frozen=True, kw_only=True)
class _Provider:
    program: CoreProgram

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return MappingProxyType({"main": self.program})


def test_core_program_graph_is_the_native_structural_seam() -> None:
    program = CoreProgram(
        name="main",
        function=_hard_max_core,
        argument_builder=lambda _context: {
            "first": jnp.asarray([0, 1]),
            "second": jnp.asarray([10, 20, 30]),
        },
        requirements=CoreExecutionRequirements(streamable_axes=(_axis(),)),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
    )
    provider = _Provider(program=program)

    assert isinstance(provider, CoreProgramGraphAware)
    assert core_program_graph(kernel=provider)["main"] is program


def test_explicit_core_program_cannot_omit_output_roles_on_unsharded_template() -> None:
    template = jnp.zeros((2,), dtype=jnp.float32)
    program = CoreProgram(
        name="main",
        function=lambda *, value: value,
        argument_builder=lambda _context: {"value": template},
        requirements=CoreExecutionRequirements(),
        output_roles=None,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="test_dense_route",
    )

    with pytest.raises(ValueError, match=r"output.?roles"):
        core_program_graph(kernel=_Provider(program=program))


def _wrong_shape_value_core(*, value: jax.Array) -> jax.Array:
    return jnp.zeros((value.shape[0] + 1,), dtype=value.dtype)


def _wrong_dtype_value_core(*, value: jax.Array) -> jax.Array:
    return jnp.zeros(value.shape, dtype=jnp.int32)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (_wrong_shape_value_core, r"value.*shape"),
        (_wrong_dtype_value_core, r"value.*dtype"),
    ],
    ids=["wrong-shape", "wrong-dtype"],
)
def test_explicit_value_program_rejects_wrong_lowered_metadata_before_compile(
    *, function: Callable[..., object], message: str
) -> None:
    template = jnp.zeros((2,), dtype=jnp.float32)
    program = MaterializedCoreProgram(
        name="main",
        function=function,
        arguments={"value": template},
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        donation_candidates=(),
        disposition_reason="test_dense_route",
    )
    resolved = resolve_core_program(program=program, tile_widths={})
    lowered = jax.jit(resolved.function).lower(**resolved.arguments)

    with pytest.raises(TypeError, match=message):
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=program.output_roles,
            layout=resolve_output_layout(
                core_key="main",
                value_template=template,
                state_order=("x",),
                output_roles=VALUE,
            ),
            label="test core",
        )


def test_ordinary_singleton_grid_search_declares_action_core_program() -> None:
    """An ordinary solve core declares its canonical action product to planning."""

    def resources(*, liquid: ContinuousState) -> FloatND:
        return liquid

    model = make_alive_dead_model(
        n_periods=2,
        n_liquid=3,
        liquid_max=4.0,
        n_consumption=4,
        alive_functions={"utility": utility, "resources": resources},
        liquid_law=next_liquid,
        alive_solver=GridSearch(action_block_width=3),
        constraints={"feasible": feasible},
    )
    flat_params = model._process_params(
        {
            "alive": {
                "utility": {"crra": 2.0},
                "koopmans_aggregator": {"discount_factor": 0.95},
                "alive": {
                    "next_liquid": {"return_liquid": 0.0, "income": 0.0},
                    "next_regime": {"final_age_alive": 1.0},
                },
                "dead": {
                    "next_liquid": {"return_liquid": 0.0, "income": 0.0},
                    "next_regime": {"final_age_alive": 1.0},
                },
            },
            "dead": {"utility": {"crra": 2.0}},
        }
    )
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["alive"]
    kernel = regime.solution.period_kernels[0]
    context = CoreBuildContext(
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["alive"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramGraphAware)
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(program=program, context=context)

    assert program.disposition is CoreExecutionDisposition.PLANNED
    assert program.disposition_reason is None
    assert isinstance(materialized.arguments, MappingProxyType)
    assert tuple(materialized.arguments) == (
        "liquid",
        "consumption",
        "next_regime_to_V_arr",
        *flat_params["alive"],
        "period",
        "age",
    )
    assert program.output_roles is VALUE
    assert program.requirements.streamable_axes == (
        StreamableProductAxis(
            name="action",
            coordinate_names=("consumption",),
            coordinate_extents=(4,),
            canonical_order="c",
            reduction=HARD_MAX_REDUCTION,
            width_keyword=_WIDTH_KEYWORD,
            requested_width=3,
        ),
    )
    with pytest.raises(TypeError):
        cast("dict[str, object]", materialized.arguments)["injected"] = jnp.asarray(0)

    resolved = resolve_core_program(
        program=materialized,
        tile_widths={"action": 3},
        input_transfer_plan=_resolve_value_input_transfer_plan(
            program=materialized,
            source_value_template=next_V["alive"],
            source=("alive", 0, "main"),
        ),
    )
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, jax.ShapeDtypeStruct)
    assert output.shape == next_V["alive"].shape


def test_collective_grid_search_declares_explicit_dense_core_program() -> None:
    """Collective execution stays dense after adverse paired resource evidence."""
    model = _build_collective_model()
    flat_params = model._process_params({"discount_factor": 0.95})
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["couple"]
    kernel = regime.solution.period_kernels[0]
    context = CoreBuildContext(
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["couple"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramGraphAware)
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(program=program, context=context)

    assert program.disposition is CoreExecutionDisposition.DENSE
    assert (
        program.disposition_reason
        == "deliberately_dense:collective_resource_regression"
    )
    assert program.output_roles == (VALUE, DISSOLUTION_FLAG)
    assert program.requirements.streamable_axes == ()

    resolved = resolve_core_program(
        program=materialized,
        tile_widths={},
    )
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, tuple)
    assert isinstance(output[0], jax.ShapeDtypeStruct)
    assert isinstance(output[1], jax.ShapeDtypeStruct)
    assert output[0].shape == next_V["couple"].shape
    assert output[1].shape == next_V["couple"].shape[:-1]
    assert output[1].dtype == jnp.bool_


def test_ev1_grid_search_declares_explicit_dense_core_program() -> None:
    """EV1 execution stays dense after the streamed winner-reversal witness."""
    model = taste_shocks_toy.get_model()
    flat_params = model._process_params(
        taste_shocks_toy.get_params(scale=0.2, discount_factor=0.95)
    )
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["alive"]
    kernel = regime.solution.period_kernels[0]
    context = CoreBuildContext(
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["alive"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramGraphAware)
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(program=program, context=context)

    assert program.disposition is CoreExecutionDisposition.DENSE
    assert (
        program.disposition_reason == "deliberately_dense:ev1_canonical_reduction_order"
    )
    assert program.output_roles is VALUE
    assert program.requirements.streamable_axes == ()

    resolved = resolve_core_program(
        program=materialized,
        tile_widths={},
    )
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, jax.ShapeDtypeStruct)
    assert output.shape == next_V["alive"].shape


def test_resolver_binds_width_without_adding_a_dynamic_argument() -> None:
    program = _program()

    with pytest.raises(TypeError, match=_WIDTH_KEYWORD):
        program.function(**program.arguments)

    resolved_three = resolve_core_program(program=program, tile_widths={"action": 3})
    resolved_four = resolve_core_program(program=program, tile_widths={"action": 4})

    assert isinstance(resolved_three, ResolvedCoreProgram)
    assert _WIDTH_KEYWORD not in program.arguments
    assert _WIDTH_KEYWORD not in resolved_three.arguments
    assert resolved_three.specialization_key != resolved_four.specialization_key
    assert resolved_three.function is resolved_four.function
    assert resolved_three.static_kwargs == {_WIDTH_KEYWORD: 3}
    assert resolved_four.static_kwargs == {_WIDTH_KEYWORD: 4}
    assert resolved_three.tile_widths == {"action": 3}
    assert resolved_four.tile_widths == {"action": 4}

    for resolved in (resolved_three, resolved_four):
        result = resolved.function(**resolved.arguments, **resolved.static_kwargs)
        assert_array_equal(result, 9.0)

    jitted = jax.jit(
        resolved_three.function,
        static_argnames=tuple(resolved_three.static_kwargs),
    )
    compiled = jitted.lower(
        **resolved_three.arguments, **resolved_three.static_kwargs
    ).compile()
    assert_array_equal(compiled(**resolved_three.arguments), 9.0)


def test_resolver_requires_a_planner_width_for_each_streamable_axis() -> None:
    with pytest.raises(ValueError, match=r"[Tt]ile width.*required"):
        resolve_core_program(program=_program())


@pytest.mark.parametrize(
    ("coordinate_extent", "error", "message"),
    [
        (2.5, TypeError, r"extents must be integers"),
        (True, TypeError, r"extents must be integers"),
        (0, ValueError, r"extents must be positive"),
        (-1, ValueError, r"extents must be positive"),
    ],
    ids=["float", "bool", "zero", "negative"],
)
def test_width_candidates_reject_invalid_coordinate_extents_fail_closed(
    *,
    coordinate_extent: object,
    error: type[Exception],
    message: str,
) -> None:
    """Width planning validates product declarations before doing width arithmetic."""
    axis = _axis(coordinate_names=("first",), coordinate_extents=(2,))
    object.__setattr__(axis, "coordinate_extents", (coordinate_extent,))
    program = _program(
        axis=axis,
        arguments={"first": jnp.asarray([0, 1])},
    )

    with pytest.raises(error, match=message):
        workspace_width_candidates(axes=program.requirements.streamable_axes)


def test_unbudgeted_width_candidate_is_the_full_product_without_an_override() -> None:
    candidates = workspace_width_candidates(
        axes=_program().requirements.streamable_axes
    )

    assert candidates == ({"action": 6},)


def test_unbudgeted_width_candidate_is_the_declared_override() -> None:
    candidates = workspace_width_candidates(
        axes=_program(axis=_axis(requested_width=3)).requirements.streamable_axes
    )

    assert candidates == ({"action": 3},)


@pytest.mark.parametrize(
    ("requested_width", "error", "message"),
    [
        (True, TypeError, "requested width.*integer"),
        (1.5, TypeError, "requested width.*integer"),
        (0, ValueError, "requested width.*positive"),
        (-1, ValueError, "requested width.*positive"),
        (7, ValueError, "requested width.*extent"),
    ],
    ids=["bool", "float", "zero", "negative", "beyond-extent"],
)
def test_resolver_rejects_invalid_declared_requested_widths(
    *, requested_width: object, error: type[Exception], message: str
) -> None:
    axis = _axis()
    object.__setattr__(axis, "requested_width", requested_width)

    with pytest.raises(error, match=message):
        resolve_core_program(
            program=_program(axis=axis),
            tile_widths={"action": 1},
        )


def test_requested_width_does_not_duplicate_the_resolved_specialization() -> None:
    inferred = resolve_core_program(program=_program(), tile_widths={"action": 3})
    requested = resolve_core_program(
        program=_program(axis=_axis(requested_width=3)),
        tile_widths={"action": 3},
    )

    assert requested.specialization_key == inferred.specialization_key


def test_program_and_resolution_snapshot_their_input_mappings() -> None:
    raw_arguments: dict[str, object] = {
        "first": jnp.asarray([0, 1]),
        "second": jnp.asarray([10, 20, 30]),
    }
    program = _program(arguments=raw_arguments)
    raw_arguments["injected"] = jnp.asarray(-1)

    requested_widths = {"action": 3}
    resolved = resolve_core_program(
        program=program,
        tile_widths=requested_widths,
    )
    requested_widths["action"] = 4

    assert "injected" not in program.arguments
    assert "injected" not in resolved.arguments
    assert resolved.static_kwargs == {_WIDTH_KEYWORD: 3}
    assert resolved.tile_widths == {"action": 3}
    with pytest.raises(TypeError):
        cast("dict[str, object]", program.arguments)["new"] = jnp.asarray(0)
    with pytest.raises(TypeError):
        cast("dict[str, object]", resolved.arguments)["new"] = jnp.asarray(0)
    with pytest.raises(TypeError):
        cast("dict[str, int]", resolved.static_kwargs)[_WIDTH_KEYWORD] = 4
    with pytest.raises(TypeError):
        cast("dict[str, int]", resolved.tile_widths)["action"] = 4


@pytest.mark.parametrize(
    ("coordinate_names", "coordinate_extents", "message"),
    [
        (("first", "first"), (2, 3), "duplicate"),
        (("first", "second"), (2,), "coordinate.*extent"),
        (("first", "second"), (2, 0), "positive"),
        (("first", "second"), (2, -1), "positive"),
    ],
    ids=["duplicate-name", "extent-count", "zero-extent", "negative-extent"],
)
def test_resolver_rejects_invalid_canonical_product_declarations(
    *,
    coordinate_names: tuple[str, ...],
    coordinate_extents: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_core_program(
            program=_program(
                axis=_axis(
                    coordinate_names=coordinate_names,
                    coordinate_extents=coordinate_extents,
                )
            ),
            tile_widths={"action": 1},
        )


def test_resolver_rejects_a_non_canonical_product_order() -> None:
    with pytest.raises(ValueError, match=r"canonical.*c"):
        resolve_core_program(
            program=_program(
                axis=_axis(canonical_order=cast("Literal['c']", "fortran"))
            ),
            tile_widths={"action": 1},
        )


@pytest.mark.parametrize(
    ("width", "error", "message"),
    [
        (True, TypeError, "width.*integer"),
        (1.5, TypeError, "width.*integer"),
        (0, ValueError, "width.*positive"),
        (-1, ValueError, "width.*positive"),
        (7, ValueError, "width.*extent"),
    ],
    ids=["bool", "float", "zero", "negative", "beyond-extent"],
)
def test_resolver_rejects_invalid_widths(
    *,
    width: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        resolve_core_program(
            program=_program(),
            tile_widths={"action": cast("int", width)},
        )


def test_resolver_rejects_a_width_for_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match=r"unknown.*axis"):
        resolve_core_program(
            program=_program(),
            tile_widths={"other": 2},
        )


def test_resolver_rejects_a_dynamic_argument_colliding_with_the_width() -> None:
    arguments = {
        "first": jnp.asarray([0, 1]),
        "second": jnp.asarray([10, 20, 30]),
        _WIDTH_KEYWORD: 3,
    }

    with pytest.raises(ValueError, match=r"width keyword.*arguments"):
        resolve_core_program(
            program=_program(arguments=arguments),
            tile_widths={"action": 3},
        )


@dataclass(frozen=True, kw_only=True)
class _UnhashableReductionSemantics:
    @property
    def semantic_key(self) -> Hashable:
        return cast("Hashable", [])


@pytest.mark.parametrize(
    ("reduction", "message"),
    [
        (cast("ReductionSemantics", object()), "stable semantic_key"),
        (_UnhashableReductionSemantics(), "semantic_key.*hashable"),
    ],
    ids=["missing-semantic-key", "unhashable-semantic-key"],
)
def test_resolver_rejects_reduction_without_a_stable_semantic_key(
    *, reduction: ReductionSemantics, message: str
) -> None:
    axis = _axis()
    object.__setattr__(axis, "reduction", reduction)
    with pytest.raises(TypeError, match=message):
        resolve_core_program(
            program=_program(axis=axis),
            tile_widths={"action": 1},
        )
