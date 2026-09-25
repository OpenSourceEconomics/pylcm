"""Re-run a single captured regime-period without solving the ones above it.

`_lcm.solution.period_capture` writes the inputs; this module runs them back
through the same funnel the solve loop uses, so a replay repeats every step the
original call took rather than approximating it.

The split is what keeps the import graph acyclic: the capture side is imported
by the backward-induction loop, and the replay side imports that loop.

Two entry points differ in how faithfully they reproduce the captured run, and
each says so in the scope it reports. `replay_period` restores the capture's
logical pytrees and lets the backend place them, so it reports `logical`.
`replay_period_on_recorded_layout` puts every array back on the sharding the
capture recorded, checks the compiled placements and transfer plan against it,
and reports `layout`.
"""

import dataclasses
import time
from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import cloudpickle
import jax
import numpy as np

from _lcm.engine import StateActionSpace, _RegimeSharding
from _lcm.execution.abstract_program_inputs import abstract_program_inputs
from _lcm.execution.compiler_memory import (
    CompilerMemoryBytes,
    compiler_memory_bytes,
)
from _lcm.execution.core_program import (
    CoreBuildContext,
    MaterializedCoreProgram,
    _value_read_argument_leaf,
    core_program_graph,
    materialize_core_program,
    select_programs,
)
from _lcm.execution.internal_outputs import (
    ResolvedProducer,
    assert_width_invariant_internal_outputs,
    consumed_producer_names,
    internal_input_templates,
    resolve_producer,
    topological_program_order,
)
from _lcm.execution.output_layout import PlannedCore, resolve_output_layout
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.execution.workspace_planning import (
    bootstrap_widths,
    compiler_memory_reservation,
)
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _attach_resolved_output_layout,
    _build_base_state_action_spaces,
    _build_continuation_templates,
    _edge_kwargs,
    _resolve_program_for_execution,
    _run_period_kernel,
    _width_key,
)
from _lcm.solution.period_capture import (
    _PAYLOAD_NAME,
    LAYOUTS_KEY,
    CoreLayoutDescriptor,
    LeafLayoutDescriptor,
    PeriodLayouts,
    ShardingDescriptor,
    _describe_transfer,
    describe_core,
    describe_sharding,
    rebuild_sharding,
)
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.solver_api import KernelOutput

# How faithfully a replay reproduced the captured run:
# - `logical` — leaves agree in tree, shape and dtype; placement is the
#   backend's default, whatever the solve used.
# - `layout` — every input leaf was restored on its recorded sharding and every
#   compiled input, output and transfer placement matched the recorded one.
# - `resource` — additionally reinstates residency context and donation
#   ownership, and is admitted under them. No entry point in this module
#   produces it; the label exists so a layout-faithful replay is not read as one.
type PeriodReplayScope = Literal["logical", "layout", "resource"]


@dataclasses.dataclass(frozen=True)
class PeriodReplay:
    """One regime-period re-run from a capture."""

    regime_name: RegimeName
    """Name of the regime the captured kernel belongs to."""

    period: int
    """Index of the captured period in the model's age grid."""

    age: float
    """Age the captured period sits at, for reading against a solve log."""

    output: KernelOutput
    """What the kernel returned: the value array and its artifact channels."""

    scope: PeriodReplayScope = "logical"
    """How faithfully this replay reproduced the captured run."""


@dataclasses.dataclass(frozen=True)
class PeriodCoreMemoryAnalysis:
    """Compiler memory of one captured period's cores at capture-roundtrip placement.

    The capture preserves the period's logical pytrees and array shapes. Its pickle
    round trip does not preserve production sharding, so these byte counts describe
    executables lowered with the restored arrays' default backend placement. They are
    not production-layout or production-memory measurements.
    """

    regime_name: RegimeName
    """Name of the regime the captured kernel belongs to."""

    period: int
    """Index of the captured period in the model's age grid."""

    age: float
    """Age the captured period sits at, for reading against a solve log."""

    preserves_production_sharding: bool
    """Always false: period capture serializes neither sharding nor placement."""

    core_memory_bytes: MappingProxyType[str, CompilerMemoryBytes | None]
    """Byte counts per production core, keyed as the kernel publishes them."""


def replay_period(*, directory: Path) -> PeriodReplay:
    """Re-run the regime-period captured in `directory`.

    The cores are lowered and compiled for this one period only, so the call
    costs one kernel rather than a backward induction. The returned `V_arr` is
    what the original solve produced for that regime-period.

    Args:
        directory: A capture directory written during a solve.

    Returns:
        The captured identity and the kernel's result.

    """
    payload = _load_capture_payload(directory=directory)

    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]
    core_tile_widths = payload["core_tile_widths"]

    output = _run_period_kernel(
        regime=regime,
        capture_target=None,
        compiled_cores=_compile_cores_for_one_period(
            regime=regime,
            period=period,
            kernel_kwargs=kernel_kwargs,
            core_tile_widths=core_tile_widths,
        ),
        **kernel_kwargs,
    )
    return PeriodReplay(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        output=output,
        scope="logical",
    )


def analyze_period_core_memory(*, directory: Path) -> PeriodCoreMemoryAnalysis:
    """Compile, but never execute, one captured period's cores and read their memory.

    The production cores are lowered against the capture's logical pytrees and
    production shapes. The capture round trip loses device sharding, so every executable
    uses the restored arrays' default backend placement. Calling `memory_analysis()`
    on the resulting executables is safe even when their estimated runtime
    allocation exceeds the available device budget because this function
    deliberately has no execution path.

    Args:
        directory: A capture directory written during a solve.

    Returns:
        The captured identity and compiler memory per core.

    """
    payload = _load_capture_payload(directory=directory)
    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]
    core_tile_widths = payload["core_tile_widths"]

    production = _compile_cores_for_one_period(
        regime=regime,
        period=period,
        kernel_kwargs=kernel_kwargs,
        core_tile_widths=core_tile_widths,
    )
    return PeriodCoreMemoryAnalysis(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        preserves_production_sharding=False,
        core_memory_bytes=MappingProxyType(
            {
                key: compiler_memory_bytes(compiled=core.compiled)
                for key, core in production.items()
            }
        ),
    )


def _load_capture_payload(*, directory: Path) -> dict[str, Any]:
    """Load one period's captured regime, inputs, widths and layout block."""
    with (directory / _PAYLOAD_NAME).open("rb") as stream:
        payload = cloudpickle.load(stream)
    if not isinstance(payload, dict):
        msg = "A period capture payload must be a dictionary."
        raise TypeError(msg)
    if "core_tile_widths" not in payload:
        msg = "Period capture is missing required 'core_tile_widths'."
        raise ValueError(msg)
    loaded = dict(payload)
    loaded["core_tile_widths"] = _normalize_core_tile_widths(
        raw=payload["core_tile_widths"]
    )
    return loaded


def _normalize_core_tile_widths(
    *, raw: object
) -> MappingProxyType[str, MappingProxyType[str, int]]:
    """Validate and freeze the portable width map stored in a capture."""
    if not isinstance(raw, Mapping):
        msg = "Period capture 'core_tile_widths' must be a mapping."
        raise TypeError(msg)

    normalized: dict[str, MappingProxyType[str, int]] = {}
    for core_name, raw_widths in raw.items():
        if not isinstance(core_name, str) or not core_name:
            msg = "Captured core names must be non-empty strings."
            raise TypeError(msg)
        if not isinstance(raw_widths, Mapping):
            msg = f"Captured tile widths for core {core_name!r} must be a mapping."
            raise TypeError(msg)

        widths: dict[str, int] = {}
        for axis_name, width in raw_widths.items():
            if not isinstance(axis_name, str) or not axis_name:
                msg = (
                    f"Captured tile-width names for core {core_name!r} must be "
                    "non-empty strings."
                )
                raise TypeError(msg)
            if type(width) is not int:
                msg = (
                    f"Captured tile width for {core_name!r}/{axis_name!r} must be "
                    "an exact integer."
                )
                raise TypeError(msg)
            if width <= 0:
                msg = (
                    f"Captured tile width for {core_name!r}/{axis_name!r} must be "
                    "positive."
                )
                raise ValueError(msg)
            widths[axis_name] = width
        normalized[core_name] = MappingProxyType(widths)
    return MappingProxyType(normalized)


def _require_exact_core_tile_widths(
    *,
    raw: object,
    core_names: tuple[str, ...],
) -> MappingProxyType[str, MappingProxyType[str, int]]:
    """Require one captured width map for every selected replay core."""
    widths = _normalize_core_tile_widths(raw=raw)
    expected = frozenset(core_names)
    actual = frozenset(widths)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg = (
            "Captured core_tile_widths must match the selected replay cores "
            f"exactly; missing={missing!r}, extra={extra!r}."
        )
        raise ValueError(msg)
    return widths


def _project_axis_widths(
    *,
    program: Any,  # noqa: ANN401 - the materialized CoreProgram, circular to import
    axis_widths: Mapping[str, int],
) -> MappingProxyType[str, int]:
    """Bind one program's declared axes from a model-wide width mapping.

    A caller fixes the widths whose effect it is measuring and leaves the rest
    out. An axis the caller names takes that width, rounded onto the set the
    axis admits; an axis the caller omits takes the bootstrap width the
    unbudgeted route would lower it at. A name that binds no axis of the period
    is refused elsewhere, so an omission is never silently a typo.
    """
    axes = program.requirements.axes
    names = {axis.name for axis in axes}
    fixed = {name: int(width) for name, width in axis_widths.items() if name in names}
    return bootstrap_widths(axes=axes, fixed_widths=MappingProxyType(fixed))


def _compile_cores_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
    core_tile_widths: object,
    core_donations: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
    recorded_core_layouts: Mapping[str, CoreLayoutDescriptor] | None = None,
    replay_devices_by_id: Mapping[int, jax.Device] = MappingProxyType({}),
    axis_widths: Mapping[str, int] | None = None,
    core_timings: MutableMapping[str, tuple[float, float]] | None = None,
) -> MappingProxyType[str, PlannedCore]:
    """Lower and compile the cores of a single period.

    The solve loop compiles every regime-period up front and deduplicates
    identical cores across them. A replay wants neither: it needs exactly the
    cores this one period calls.

    A capture pins one width per core, so a producer's record set holds a single
    candidate and its width-invariance check can only pass. It runs anyway, so
    the solve loop and the replay reach a graph's producers through one routine.

    `axis_widths` takes precedence over `core_tile_widths`: when it is given,
    every core's widths are projected from it and the captured `core_tile_widths`
    are ignored entirely. `core_tile_widths` is read only when `axis_widths` is
    `None`, and then it must pin exactly one width mapping per core.

    A core donates exactly the arguments `core_donations` names for it, which is
    empty unless the caller reinstates a captured lowering's donation. The
    placement it lowers against depends on the caller:
    - `replay_period` lowers against the arrays it is handed, at the backend's
      default placement.
    - `replay_period_on_recorded_layout` passes the recorded core layouts. It
      reinstates each recorded value transfer instead of inferring it from the
      stored input alone, and lowers against the same operand descriptors the
      solve does: committed inputs keep their layout, uncommitted ones take the
      source value's mesh, replicated.
    """
    period_kernel = regime.solution.period_kernels[period]
    context = _core_build_context_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    graph = select_programs(
        graph=core_program_graph(kernel=period_kernel),
        retain_replay=kernel_kwargs["retain_replay"],
        selected_artifact_keys=kernel_kwargs["selected_artifact_keys"],
    )
    declared_widths: Mapping[str, int] = (
        MappingProxyType({}) if axis_widths is None else axis_widths
    )
    captured_widths = (
        None
        if axis_widths is not None
        else _require_exact_core_tile_widths(
            raw=core_tile_widths,
            core_names=tuple(graph),
        )
    )
    compiled: dict[str, PlannedCore] = {}
    producers: dict[str, MappingProxyType[Hashable, ResolvedProducer]] = {}
    consumed = consumed_producer_names(graph=graph)
    for core_name in topological_program_order(graph=graph):
        declaration = graph[core_name]
        materialized = materialize_core_program(program=declaration, context=context)
        templates = internal_input_templates(program=materialized, producers=producers)
        widths = (
            captured_widths[core_name]
            if captured_widths is not None
            else _project_axis_widths(program=materialized, axis_widths=declared_widths)
        )
        source_value_template = context.next_regime_to_V_arr[
            kernel_kwargs["regime_name"]
        ]
        input_transfer_plan = (
            _restore_input_transfer_plan(
                program=materialized,
                recorded=recorded_core_layouts[core_name],
                devices_by_id=replay_devices_by_id,
            )
            if recorded_core_layouts is not None
            else None
        )
        if input_transfer_plan is not None:
            # Lower against the operand descriptors the solve lowers against: an
            # uncommitted operand takes the source value's mesh, replicated. A
            # concrete lowering leaves that choice to JAX, which picks among the
            # committed inputs' meshes — the wrong one whenever the inputs span
            # several same-device meshes under different axis names.
            materialized = abstract_program_inputs(
                program=materialized,
                transfers=input_transfer_plan,
                execution_sharding=cast("jax.Array", source_value_template).sharding,
            )
        resolved = _resolve_program_for_execution(
            program=materialized,
            tile_widths=widths,
            source_value_template=source_value_template,
            source=(kernel_kwargs["regime_name"], period, core_name),
            input_transfer_plan=input_transfer_plan,
            abstract_inputs=input_transfer_plan is not None,
        )
        if core_name in consumed:
            records: dict[Hashable, ResolvedProducer] = {
                _width_key(widths=resolved.tile_widths): resolve_producer(
                    program=resolved, templates=templates
                )
            }
            candidates = MappingProxyType(records)
            assert_width_invariant_internal_outputs(candidates=candidates)
            producers[core_name] = candidates
        state_action_space = cast("StateActionSpace", context.state_action_space)
        state_order = tuple(
            name
            for name in state_action_space.states
            if name not in regime.fold_state_names
        )
        layout = resolve_output_layout(
            core_key=core_name,
            value_template=context.next_regime_to_V_arr[kernel_kwargs["regime_name"]],
            state_order=state_order,
            output_roles=resolved.output_roles,
        )
        donated = tuple(core_donations.get(core_name, ()))
        jitted = jax.jit(
            resolved.function,
            static_argnames=tuple(resolved.static_kwargs),
            out_shardings=layout.out_shardings,
            donate_argnames=donated or None,
        )
        lowering_start = time.monotonic()
        lowered = jitted.lower(
            **resolved.arguments, **templates, **resolved.static_kwargs
        )
        lowering_seconds = time.monotonic() - lowering_start
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=resolved.output_roles,
            layout=layout,
            label=(
                f"{kernel_kwargs['regime_name']} {core_name} (replay period {period})"
            ),
        )
        compile_start = time.monotonic()
        executable = lowered.compile()
        if core_timings is not None:
            core_timings[core_name] = (
                lowering_seconds,
                time.monotonic() - compile_start,
            )
        compiled[core_name] = _attach_resolved_output_layout(
            compiled=executable,
            layout=layout,
            tile_widths=resolved.tile_widths,
            input_transfer_plan=resolved.input_transfer_plan,
            internal_input_templates=templates,
            donated_arguments=donated,
            name=core_name,
        )
    return MappingProxyType(compiled)


def _core_build_context_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> CoreBuildContext:
    """Build the immutable program context shared by replay and memory analysis."""
    # A source declaring gated edges reads its targets' folded continuation in
    # place of their raw V, so the kernel is compiled against a pytree the raw
    # mapping does not carry. The projection comes from the solve loop's own
    # builder rather than a second copy of it, which is what keeps the argument
    # tree and logical shapes a replay lowers against match production. Placement
    # is whatever the caller already put on the arrays in `kernel_kwargs`; this
    # builder neither reads nor asserts it.
    edge_kwargs = _edge_kwargs(
        regime=regime,
        regime_name=kernel_kwargs["regime_name"],
        next_edge_to_V_arr=kernel_kwargs["next_edge_to_V_arr"],
    )
    return CoreBuildContext(
        state_action_space=kernel_kwargs["state_action_space"],
        next_regime_to_V_arr=kernel_kwargs["next_regime_to_V_arr"],
        next_regime_to_continuation=kernel_kwargs["next_regime_to_continuation"],
        flat_params=kernel_kwargs["flat_params"],
        period=period,
        ages=kernel_kwargs["ages"],
        edge_regime_to_V_arr=cast(
            "Mapping[str, object] | None",
            edge_kwargs.get("edge_regime_to_V_arr"),
        ),
    )


_SUPPORTED_LAYOUT_ROUTES = frozenset(
    {"_lcm.solution.grid_search._GridSearchPeriodKernel"}
)


def replay_period_on_recorded_layout(
    *, directory: Path, devices: Sequence[jax.Device]
) -> PeriodReplay:
    """Re-run a captured regime-period on the device layout it was solved on.

    Every array input is placed back on its recorded sharding before anything is
    lowered, so the cores are compiled against the production placement rather
    than the backend's default. The restored placements, the compiled
    executables' input and output shardings, and the resolved value-transfer
    plan are each checked against the capture, so the returned `layout` scope is
    established rather than assumed.

    The cores are lowered with the donation the capture recorded, so a core the
    solve dispatched as its donating variant donates here too. Donated inputs are
    consumed by the call and unreadable afterwards: replaying the same directory
    again needs freshly admitted copies, which means loading the capture again
    rather than reusing the arrays this call was handed.

    The scope is never `resource`: residency context and donation ownership are
    not reinstated here, so a run that fits only because the replay owns its
    inputs outright is not evidence about the solve's admission.

    Args:
        directory: A capture directory written during a solve.
        devices: Devices standing in for the recorded ones, in recorded order.
            Only leaves the solve committed are re-placed through them, because
            committing a leaf the solve left uncommitted makes the jit call
            refuse the mix. A substitution that is not the identity therefore
            moves the committed leaves and leaves the rest on the backend's
            default device.

    Returns:
        The captured identity, the kernel's result, and scope `layout`.

    Raises:
        ValueError: The capture records no layouts, was taken on a route this
            entry point does not support, the device count differs from the
            recorded one, or a restored or compiled placement differs from the
            recorded descriptor.

    """
    payload = _load_capture_payload(directory=directory)
    layouts = _require_period_layouts(payload=payload, directory=directory)
    recorded_cores = _require_recorded_cores(layouts=layouts, directory=directory)

    device_by_recorded_id = _device_substitution(
        recorded_ids=layouts.device_ids, devices=devices
    )
    recorded_cores = MappingProxyType(
        {
            name: _substitute_core_layout(
                core=core, device_by_recorded_id=device_by_recorded_id
            )
            for name, core in recorded_cores.items()
        }
    )
    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = _restore_recorded_layout(
        kernel_kwargs=payload["kernel_kwargs"],
        leaves=layouts.leaves,
        device_by_recorded_id=device_by_recorded_id,
    )
    kernel_kwargs = _restore_state_action_space_layout(kernel_kwargs=kernel_kwargs)

    compiled_cores = _compile_cores_for_one_period(
        regime=regime,
        period=period,
        kernel_kwargs=kernel_kwargs,
        core_tile_widths=payload["core_tile_widths"],
        recorded_core_layouts=recorded_cores,
        replay_devices_by_id=MappingProxyType(
            {int(device.id): device for device in devices}
        ),
        core_donations=MappingProxyType(
            {
                core_name: core.donated_arguments
                for core_name, core in recorded_cores.items()
            }
        ),
    )
    _assert_cores_match_recorded_layout(
        compiled_cores=compiled_cores, recorded=recorded_cores
    )

    output = _run_period_kernel(
        regime=regime,
        capture_target=None,
        compiled_cores=compiled_cores,
        **kernel_kwargs,
    )
    return PeriodReplay(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        output=output,
        scope="layout",
    )


def _require_period_layouts(
    *, payload: Mapping[str, Any], directory: Path
) -> PeriodLayouts:
    """Return the capture's layout block, refusing a capture that has none."""
    layouts = payload.get(LAYOUTS_KEY)
    if layouts is None:
        msg = (
            f"The capture in {directory} records no {LAYOUTS_KEY!r} block, so its "
            "device layout cannot be reinstated. Replay it with replay_period, "
            "which reports scope 'logical'."
        )
        raise ValueError(msg)
    if not isinstance(layouts, PeriodLayouts):
        msg = f"A capture's {LAYOUTS_KEY!r} block must be a PeriodLayouts."
        raise TypeError(msg)
    if layouts.route not in _SUPPORTED_LAYOUT_ROUTES:
        msg = (
            f"replay_period_on_recorded_layout supports route(s) "
            f"{sorted(_SUPPORTED_LAYOUT_ROUTES)!r}; the capture in {directory} was "
            f"taken on route {layouts.route!r}."
        )
        raise ValueError(msg)
    return layouts


def _require_recorded_cores(
    *, layouts: PeriodLayouts, directory: Path
) -> MappingProxyType[str, CoreLayoutDescriptor]:
    """Return the recorded core layouts, refusing a capture missing any of them.

    A core whose executable published no input or output shardings — the eager
    route's is one — is recorded as absent, and there is nothing to reinstate
    its placement from.
    """
    absent = sorted(
        core_name for core_name, core in layouts.cores.items() if core is None
    )
    if absent:
        msg = (
            f"The capture in {directory} records no device layout for core(s) "
            f"{absent!r}: the executables that ran published no input or output "
            "shardings, so their placement cannot be reinstated. Replay it with "
            "replay_period, which reports scope 'logical'."
        )
        raise ValueError(msg)
    return MappingProxyType(
        {
            core_name: core
            for core_name, core in layouts.cores.items()
            if core is not None
        }
    )


def _device_substitution(
    *, recorded_ids: tuple[int, ...], devices: Sequence[jax.Device]
) -> MappingProxyType[int, jax.Device]:
    """Pair each recorded device id with the device standing in for it."""
    given = tuple(devices)
    if len(given) != len(recorded_ids):
        msg = (
            "A layout-faithful replay needs one device per recorded device: "
            f"recorded {len(recorded_ids)} ({recorded_ids!r}), given {len(given)} "
            f"({tuple(device.id for device in given)!r})."
        )
        raise ValueError(msg)
    return MappingProxyType(dict(zip(recorded_ids, given, strict=True)))


def _substitute_sharding(
    *,
    descriptor: ShardingDescriptor,
    device_by_recorded_id: Mapping[int, jax.Device],
) -> ShardingDescriptor:
    """Translate physical ids, preserving mesh order, axes and partition semantics."""
    return dataclasses.replace(
        descriptor,
        device_ids=tuple(
            int(device_by_recorded_id[device_id].id)
            for device_id in descriptor.device_ids
        ),
    )


def _substitute_core_layout(
    *,
    core: CoreLayoutDescriptor,
    device_by_recorded_id: Mapping[int, jax.Device],
) -> CoreLayoutDescriptor:
    """Move every expected executable/transfer placement into the replay namespace."""

    def mapped(descriptor: ShardingDescriptor) -> ShardingDescriptor:
        return _substitute_sharding(
            descriptor=descriptor, device_by_recorded_id=device_by_recorded_id
        )

    return dataclasses.replace(
        core,
        lowered_out_shardings=tuple(mapped(s) for s in core.lowered_out_shardings),
        compiled_input_shardings=tuple(
            (name, mapped(s)) for name, s in core.compiled_input_shardings
        ),
        compiled_output_shardings=tuple(
            (name, mapped(s)) for name, s in core.compiled_output_shardings
        ),
        input_transfer_plan=tuple(
            dataclasses.replace(
                transfer,
                stored_sharding=mapped(transfer.stored_sharding),
                source_sharding=mapped(transfer.source_sharding),
            )
            for transfer in core.input_transfer_plan
        ),
    )


def _restore_input_transfer_plan(
    *,
    program: MaterializedCoreProgram,
    recorded: CoreLayoutDescriptor,
    devices_by_id: Mapping[int, jax.Device],
) -> tuple[ResolvedValueTransfer, ...]:
    """Reinstate captured transfer intent using the materialized logical addresses.

    Stored placement does not determine the required consumer placement: a
    continuously sharded reader needs the complete line even on the same mesh.
    Read addresses come from the current declaration, never by parsing repr
    strings in the capture. The rebuilt operator must match every captured field.
    """
    reads = program.requirements.value_reads
    if len(reads) != len(recorded.input_transfer_plan):
        raise ValueError(
            f"Core {recorded.name!r} declares {len(reads)} value reads but the "
            f"capture records {len(recorded.input_transfer_plan)} transfers."
        )
    transfers: list[ResolvedValueTransfer] = []
    for read, expectation in zip(reads, recorded.input_transfer_plan, strict=True):
        if (repr(read.target), repr(read.source)) != (
            expectation.target,
            expectation.source,
        ):
            raise ValueError(
                f"Core {recorded.name!r} declares a different value-read address "
                "than the capture recorded."
            )
        transfer = resolve_value_transfer(
            target=read.target,
            source=read.source,
            kind=ValueTransferKind(expectation.kind),
            stored_template=_value_read_argument_leaf(program=program, read=read),
            source_sharding=rebuild_sharding(
                descriptor=expectation.source_sharding,
                device_by_recorded_id=devices_by_id,
            ),
        )
        observed = _describe_transfer(transfer=transfer)
        if observed != expectation:
            raise ValueError(
                f"Core {recorded.name!r} could not restore a value transfer: "
                f"expected {expectation!r}, got {observed!r}."
            )
        transfers.append(transfer)
    return tuple(transfers)


def _restore_recorded_layout(
    *,
    kernel_kwargs: dict[str, Any],
    leaves: tuple[LeafLayoutDescriptor, ...],
    device_by_recorded_id: Mapping[int, jax.Device],
) -> dict[str, Any]:
    """Place every array leaf back on its recorded sharding and check it landed.

    The placement is checked leaf by leaf rather than trusted: `device_put` onto
    an equivalent-but-different sharding would otherwise pass silently and every
    later check would compare a layout nobody recorded.

    A leaf the solve left uncommitted is left uncommitted here. Committing it
    would pin an argument the production lowering let XLA place with the rest,
    and a jit call mixing a pinned scalar with a sharded array is refused
    outright. Its placement is checked all the same, so a backend whose default
    differs from the recorded one is caught rather than silently accepted.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(kernel_kwargs)
    by_path = {descriptor.tree_path: descriptor for descriptor in leaves}
    restored: list[Any] = []
    for path, leaf in flat:
        key = jax.tree_util.keystr(path)
        descriptor = by_path.get(key)
        if descriptor is None or not isinstance(leaf, jax.Array):
            restored.append(leaf)
            continue
        if descriptor.committed:
            placed = jax.device_put(
                leaf,
                rebuild_sharding(
                    descriptor=descriptor.sharding,
                    device_by_recorded_id=device_by_recorded_id,
                ),
            )
        else:
            placed = leaf
        _assert_recorded_sharding(
            actual=placed.sharding,
            recorded=(
                _substitute_sharding(
                    descriptor=descriptor.sharding,
                    device_by_recorded_id=device_by_recorded_id,
                )
                if descriptor.committed
                else descriptor.sharding
            ),
            label=f"restored input {key}",
        )
        restored.append(placed)
    return cast("dict[str, Any]", jax.tree.unflatten(treedef, restored))


def _restore_state_action_space_layout(
    *, kernel_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Place the state-action space's grids where the solve's builder put them.

    The state-action space is not a pytree, so a capture records no descriptor
    for its grids and they come back from the pickle unplaced. Their placement
    is re-derived by the rules `Regime.state_action_space` applies, from inputs
    whose layout was restored:
    - a grid whose points are a runtime parameter (`<name>__points`) is that
      parameter array, so it takes the restored parameter itself, after a check
      that the values agree;
    - a distributed state is split along its own axis of the regime's mesh,
      which the regime's restored value template carries;
    - every other grid stays where the pickle put it, which the solve's lowering
      treats like any other uncommitted operand.

    The compiled-input check then compares the result with the recorded placement.

    Raises:
        ValueError: A runtime-points parameter disagrees with the captured grid.

    """
    regime_name = kernel_kwargs["regime_name"]
    regime_params = kernel_kwargs["flat_params"][regime_name]
    template_sharding = kernel_kwargs["next_regime_to_V_arr"][regime_name].sharding
    plan = (
        _RegimeSharding(
            mesh=template_sharding.mesh,
            distributed_state_names=tuple(template_sharding.mesh.axis_names),
        )
        if isinstance(template_sharding, jax.NamedSharding)
        else None
    )

    def placed(*, name: str, grid: Any) -> Any:  # noqa: ANN401
        points = regime_params.get(f"{name}__points")
        if points is not None:
            if not np.array_equal(np.asarray(points), np.asarray(grid)):
                msg = (
                    f"The captured grid {name!r} of regime {regime_name!r} differs "
                    f"from its runtime parameter '{name}__points'."
                )
                raise ValueError(msg)
            grid = points
        if plan is not None and name in plan.distributed_state_names:
            return jax.device_put(grid, plan.state_sharding(name))
        return grid

    space: StateActionSpace = kernel_kwargs["state_action_space"]
    return {
        **kernel_kwargs,
        "state_action_space": space.replace(
            states=MappingProxyType(
                {
                    name: placed(name=name, grid=grid)
                    for name, grid in space.states.items()
                }
            ),
            continuous_actions=MappingProxyType(
                {
                    name: placed(name=name, grid=grid)
                    for name, grid in space.continuous_actions.items()
                }
            ),
        ),
    }


def _assert_cores_match_recorded_layout(
    *,
    compiled_cores: Mapping[str, PlannedCore],
    recorded: Mapping[str, CoreLayoutDescriptor],
) -> None:
    """Check every replayed core against the placement the capture recorded.

    Every recorded entry is a descriptor here: a capture with an absent core
    layout block is refused by `_require_recorded_cores` before any core is
    compiled.
    """
    if set(compiled_cores) != set(recorded):
        msg = (
            "The replayed cores do not match the recorded ones: "
            f"replayed={sorted(compiled_cores)!r}, recorded={sorted(recorded)!r}."
        )
        raise ValueError(msg)
    for core_name, core in compiled_cores.items():
        observed = describe_core(name=core_name, core=core)
        expectation = recorded[core_name]
        if observed is None:
            msg = (
                f"The executable replayed for core {core_name!r} publishes no "
                "input or output shardings, so it cannot be checked against the "
                "recorded placement."
            )
            raise ValueError(msg)
        _assert_shardings_match(
            actual=observed.lowered_out_shardings,
            recorded=expectation.lowered_out_shardings,
            label=f"core {core_name!r} lowered out_shardings",
        )
        _assert_named_shardings_match(
            actual=observed.compiled_input_shardings,
            recorded=expectation.compiled_input_shardings,
            label=f"core {core_name!r} compiled input",
        )
        _assert_named_shardings_match(
            actual=observed.compiled_output_shardings,
            recorded=expectation.compiled_output_shardings,
            label=f"core {core_name!r} compiled output",
        )
        if observed.input_transfer_plan != expectation.input_transfer_plan:
            msg = (
                f"core {core_name!r} resolved a different value-transfer plan than "
                f"the capture recorded: expected {expectation.input_transfer_plan!r}, "
                f"got {observed.input_transfer_plan!r}."
            )
            raise ValueError(msg)
        if observed.variant != expectation.variant:
            msg = (
                f"core {core_name!r} was lowered as the {observed.variant!r} variant, "
                f"but the capture recorded {expectation.variant!r} "
                f"(donated {expectation.donated_arguments!r})."
            )
            raise ValueError(msg)


def _assert_shardings_match(
    *,
    actual: tuple[ShardingDescriptor, ...],
    recorded: tuple[ShardingDescriptor, ...],
    label: str,
) -> None:
    """Compare two positional placement lists, naming the first difference."""
    if len(actual) != len(recorded):
        msg = (
            f"{label} has {len(actual)} leaves, but the capture recorded "
            f"{len(recorded)}."
        )
        raise ValueError(msg)
    for index, (observed, expectation) in enumerate(zip(actual, recorded, strict=True)):
        _assert_recorded_sharding(
            actual=observed, recorded=expectation, label=f"{label} leaf {index}"
        )


def _assert_named_shardings_match(
    *,
    actual: tuple[tuple[str, ShardingDescriptor], ...],
    recorded: tuple[tuple[str, ShardingDescriptor], ...],
    label: str,
) -> None:
    """Compare two path-keyed placement lists, naming the first difference."""
    if tuple(name for name, _ in actual) != tuple(name for name, _ in recorded):
        msg = (
            f"{label} shardings name different arguments: "
            f"expected {tuple(name for name, _ in recorded)!r}, "
            f"got {tuple(name for name, _ in actual)!r}."
        )
        raise ValueError(msg)
    for (name, observed), (_, expectation) in zip(actual, recorded, strict=True):
        _assert_recorded_sharding(
            actual=observed, recorded=expectation, label=f"{label} {name}"
        )


def _assert_recorded_sharding(
    *,
    actual: object,
    recorded: ShardingDescriptor,
    label: str,
) -> None:
    """Refuse a placement that differs from the recorded descriptor."""
    observed = (
        actual
        if isinstance(actual, ShardingDescriptor)
        else describe_sharding(sharding=actual)
    )
    if observed != recorded:
        msg = (
            f"{label} does not match the recorded layout: "
            f"expected {recorded!r}, got {observed!r}."
        )
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeclaredCoreCompilation:
    """One core of one declared regime-period, compiled and never dispatched.

    The byte counts are the compiler's own report for this executable at the
    declared widths and placement. They are a statement about the compiled
    program, not about an allocator: nothing here reserves device memory, and
    nothing here is executed.
    """

    regime_name: RegimeName
    """Regime the core belongs to."""

    period: int
    """Index of the declared period in the model's age grid."""

    age: float
    """Age that period sits at, for reading against a solve log."""

    core_name: str
    """Key the period kernel publishes this core under."""

    tile_widths: MappingProxyType[str, int]
    """Width bound to each execution axis the core declares."""

    lowering_seconds: float
    """Host wall of the jaxpr-to-MLIR lowering."""

    compile_seconds: float
    """Host wall of the backend compilation."""

    reservation_bytes: int | None
    """Complete reported allocation the compiler reserves, or `None` when the
    backend reported none."""

    peak_bytes: int | None
    """Raw reported peak of the executable, or `None` when unreported."""

    memory: CompilerMemoryBytes | None
    """The compiler's per-field memory report, or `None` when unavailable."""

    compiled: Any
    """The compiled executable, for a caller that reads its text or its report."""


def compile_declared_period_cores(
    *,
    regimes: MappingProxyType[RegimeName, Any],
    flat_params: FlatParams,
    ages: AgeGrid,
    regime_name: RegimeName,
    period: int,
    axis_widths: Mapping[str, int],
    device_ids: tuple[int, ...] = (),
    process_grid_resolver: ProcessGridResolver | None = None,
) -> tuple[DeclaredCoreCompilation, ...]:
    """Materialize, lower and compile one declared regime-period, without solving.

    The state-action spaces and continuation templates are built by the solve's
    own builders, so the cores are lowered against the argument trees and
    logical shapes production uses. Every period above the declared one is
    skipped: nothing is solved, and no compiled program is called.

    Args:
        regimes: The model's internal regimes.
        flat_params: Regime parameters, as a solve would pass them.
        ages: The model's age grid.
        regime_name: Regime whose period kernel is compiled.
        period: Index of the period in `ages`.
        axis_widths: Width per execution axis, for the model as a whole. Each
            core is handed the axes it declares that this mapping names; an
            axis it declares that the mapping omits takes the bootstrap width
            the unbudgeted route would lower it at, so a caller fixing one axis
            need not enumerate the rest. A name no core of this period declares
            is refused, because a width that binds nothing would read as a
            width that was applied.
        device_ids: Devices the continuation templates are placed over.
        process_grid_resolver: Resolver for runtime process grids, when the
            model needs one.

    Returns:
        One entry per core the period kernel publishes, in compilation order.

    Raises:
        ValueError: When the regime is not declared, when it is inactive in the
            period, or when a declared width binds no axis of this period.

    """
    regime = _require_declared_regime(
        regimes=regimes, regime_name=regime_name, period=period
    )
    base_spaces = _build_base_state_action_spaces(
        regimes=regimes,
        flat_params=flat_params,
        process_grid_resolver=process_grid_resolver,
    )
    next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr = (
        _build_continuation_templates(
            regimes=regimes,
            flat_params=flat_params,
            device_ids=device_ids,
            process_grid_resolver=process_grid_resolver,
        )
    )
    kernel_kwargs = {
        "regime_name": regime_name,
        "period": period,
        "state_action_space": base_spaces[regime_name],
        "flat_params": flat_params,
        "ages": ages,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        "next_regime_to_continuation": next_regime_to_continuation,
        "next_edge_to_V_arr": next_edge_to_V_arr,
        "retain_replay": False,
        "selected_artifact_keys": frozenset(),
    }
    _fail_if_widths_bind_nothing(
        declared_axis_names=_declared_axis_names(
            regime=regime, period=period, kernel_kwargs=kernel_kwargs
        ),
        axis_widths=axis_widths,
    )
    timings: dict[str, tuple[float, float]] = {}
    compiled = _compile_cores_for_one_period(
        regime=regime,
        period=period,
        kernel_kwargs=kernel_kwargs,
        core_tile_widths=MappingProxyType({}),
        axis_widths=axis_widths,
        core_timings=timings,
    )
    return tuple(
        _declared_compilation(
            regime_name=regime_name,
            period=period,
            age=float(ages.values[period]),
            core_name=core_name,
            core=core,
            timing=timings[core_name],
        )
        for core_name, core in compiled.items()
    )


def _require_declared_regime(
    *,
    regimes: Mapping[RegimeName, Any],
    regime_name: RegimeName,
    period: int,
) -> Any:  # noqa: ANN401 - the canonical Regime, circular to import here
    """Return the regime, refusing a name or a period it does not declare."""
    if regime_name not in regimes:
        msg = (
            f"The model declares no regime {regime_name!r}; it declares "
            f"{sorted(regimes)!r}."
        )
        raise ValueError(msg)
    regime = regimes[regime_name]
    if period not in regime.active_periods:
        msg = (
            f"Regime {regime_name!r} is inactive in period {period}, so it "
            "publishes no kernel there."
        )
        raise ValueError(msg)
    return regime


def _declared_axis_names(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> frozenset[str]:
    """Collect the execution axes this period's cores declare.

    Reads the materialized core programs only, so a width that binds nothing is
    refused before anything is lowered or compiled.
    """
    context = _core_build_context_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    graph = select_programs(
        graph=core_program_graph(kernel=regime.solution.period_kernels[period]),
        retain_replay=kernel_kwargs["retain_replay"],
        selected_artifact_keys=kernel_kwargs["selected_artifact_keys"],
    )
    return frozenset(
        axis.name
        for declaration in graph.values()
        for axis in materialize_core_program(
            program=declaration, context=context
        ).requirements.axes
    )


def _fail_if_widths_bind_nothing(
    *,
    declared_axis_names: frozenset[str],
    axis_widths: Mapping[str, int],
) -> None:
    """Raise when a declared width binds no axis of any core of this period."""
    bound = set(declared_axis_names)
    unbound = sorted(set(axis_widths) - bound)
    if unbound:
        msg = (
            f"Declared widths {unbound!r} name no execution axis of any core "
            f"of this period; the cores declare {sorted(bound)!r}."
        )
        raise ValueError(msg)


def _declared_compilation(
    *,
    regime_name: RegimeName,
    period: int,
    age: float,
    core_name: str,
    core: PlannedCore,
    timing: tuple[float, float],
) -> DeclaredCoreCompilation:
    """Read one compiled core's walls and memory report into a record."""
    reservation = compiler_memory_reservation(
        compiled=core.compiled, widths=core.tile_widths
    )
    reservation_bytes = reservation.reservation_bytes
    peak_bytes = reservation.peak_bytes
    lowering_seconds, compile_seconds = timing
    return DeclaredCoreCompilation(
        regime_name=regime_name,
        period=period,
        age=age,
        core_name=core_name,
        tile_widths=MappingProxyType(dict(core.tile_widths)),
        lowering_seconds=lowering_seconds,
        compile_seconds=compile_seconds,
        reservation_bytes=reservation_bytes,
        peak_bytes=peak_bytes,
        memory=compiler_memory_bytes(compiled=core.compiled),
        compiled=core.compiled,
    )
