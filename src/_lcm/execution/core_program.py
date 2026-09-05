"""Solver-declared core programs for engine-owned lowering.

Solvers declare a core's dynamic arguments, output roles, and static execution
requirements. The engine validates the declaration and binds planner-owned choices
before lowering. Every kernel publishes its own graph: the engine plans a program
or runs it deliberately dense, and refuses a kernel that publishes no graph.
"""

from __future__ import annotations

import inspect
import math
import weakref
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueConsumerAddress,
    apply_value_transfer_plan,
)
from _lcm.typing import ActionName
from lcm.solver_api import ArtifactKey

_CORE_PROGRAM_VERSION = 6
_INT32_MAX = 2_147_483_647

if TYPE_CHECKING:
    type _RetainedArtifactKeys = tuple[ArtifactKey, ...]
    type _RetainedArtifactPayloadTypes = Mapping[ArtifactKey, type[object]]
else:
    # Runtime construction deliberately reaches CoreProgram's exact, deterministic
    # validation instead of a decorator-generated annotation error.
    type _RetainedArtifactKeys = object
    type _RetainedArtifactPayloadTypes = object


@dataclass(frozen=True, kw_only=True)
class _TargetValueAccess:
    """One exact target-value read declared by a solver core.

    The target preserves artifact identity for liveness; the source preserves the
    complete dynamic-argument locator. Transfer specialization deliberately lives on
    the resolved adapter so absolute periods do not fragment executable reuse.
    """

    target: ValueArtifactAddress
    source: ValueConsumerAddress

    def __post_init__(self) -> None:
        """Require the shared, already-validated logical address types."""
        if not isinstance(self.target, ValueArtifactAddress):
            msg = "A target-value access target must be a ValueArtifactAddress."
            raise TypeError(msg)
        if not isinstance(self.source, ValueConsumerAddress):
            msg = "A target-value access source must be a ValueConsumerAddress."
            raise TypeError(msg)


@runtime_checkable
class _TransferArgumentLeaf(Protocol):
    """Array-like dynamic leaf validated before transfer planning."""

    shape: tuple[int, ...]
    dtype: object
    sharding: object


@runtime_checkable
class ReductionSemantics(Protocol):
    """Solver-owned reduction semantics used in static program identity."""

    @property
    def semantic_key(self) -> Hashable:
        """Return a stable key for the reduction's numerical contract."""
        ...


@dataclass(frozen=True, kw_only=True)
class StreamableProductAxis:
    """One canonical Cartesian-product axis that the planner may tile."""

    name: str
    coordinate_names: tuple[ActionName, ...]
    coordinate_extents: tuple[int, ...]
    canonical_order: str
    reduction: ReductionSemantics
    width_keyword: str
    requested_width: int | None = None

    def __post_init__(self) -> None:
        """Snapshot caller-owned sequences while leaving validation late."""
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))
        object.__setattr__(self, "coordinate_extents", tuple(self.coordinate_extents))

    @property
    def extent(self) -> int:
        """Return the total number of cells in the canonical product."""
        return math.prod(self.coordinate_extents)


@dataclass(frozen=True, kw_only=True)
class CoreExecutionRequirements:
    """Static requirements that the execution planner must resolve for a core."""

    streamable_axes: tuple[StreamableProductAxis, ...] = ()
    target_value_accesses: tuple[_TargetValueAccess, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot the declared axes and exact target-value reads."""
        object.__setattr__(self, "streamable_axes", tuple(self.streamable_axes))
        object.__setattr__(
            self, "target_value_accesses", tuple(self.target_value_accesses)
        )


class CoreExecutionDisposition(StrEnum):
    """How the engine must execute one declared core."""

    PLANNED = "planned"
    DENSE = "dense"


class ProgramScope(StrEnum):
    """Which result retentions dispatch one declared core.

    - `ANY`: dispatched under every retention.
    - `VALUES_ONLY`: dispatched only when the solve retains no replay artifacts.
    - `REPLAY`: an alternative value-producing variant, dispatched for normal
      replay retention or when one of its exact artifact keys is selected.
    - `ARTIFACT`: an additive program dispatched only when one of its exact
      artifact keys is selected.

    A kernel publishing both scoped variants of one body lets a values-only solve
    skip the replay outputs' assembly instead of computing and discarding them.
    """

    ANY = "any"
    VALUES_ONLY = "values-only"
    REPLAY = "replay"
    ARTIFACT = "artifact"


@dataclass(frozen=True, kw_only=True)
class CoreBuildContext:
    """Immutable inputs from which a core builds its dynamic argument mapping."""

    state_action_space: object
    next_regime_to_V_arr: Mapping[str, object]
    next_regime_to_continuation: Mapping[str, object]
    flat_params: Mapping[str, object]
    period: int
    ages: object
    edge_regime_to_V_arr: Mapping[str, object] | None = None
    same_period_regime_to_V_arr: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Snapshot caller-owned mappings and reject ambiguous period values."""
        if isinstance(self.period, bool) or not isinstance(self.period, int):
            msg = "A CoreBuildContext period must be an integer."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "next_regime_to_V_arr",
            MappingProxyType(dict(self.next_regime_to_V_arr)),
        )
        object.__setattr__(
            self,
            "next_regime_to_continuation",
            MappingProxyType(dict(self.next_regime_to_continuation)),
        )
        object.__setattr__(
            self, "flat_params", MappingProxyType(dict(self.flat_params))
        )
        if self.edge_regime_to_V_arr is not None:
            object.__setattr__(
                self,
                "edge_regime_to_V_arr",
                MappingProxyType(dict(self.edge_regime_to_V_arr)),
            )
        if self.same_period_regime_to_V_arr is not None:
            object.__setattr__(
                self,
                "same_period_regime_to_V_arr",
                MappingProxyType(dict(self.same_period_regime_to_V_arr)),
            )


type CoreArgumentBuilder = Callable[[CoreBuildContext], Mapping[str, object]]


@dataclass(frozen=True, kw_only=True)
class CoreProgram:
    """One authoritative, unmaterialized program in a period kernel's graph."""

    name: str
    function: Callable[..., object]
    argument_builder: CoreArgumentBuilder
    requirements: CoreExecutionRequirements
    output_roles: object
    disposition: CoreExecutionDisposition
    disposition_reason: str | None = None
    donation_candidates: tuple[str, ...] = ()
    scope: ProgramScope = ProgramScope.ANY
    retained_artifact_keys: _RetainedArtifactKeys = ()
    retained_artifact_payload_types: _RetainedArtifactPayloadTypes = MappingProxyType(
        {}
    )
    replaces_program: str | None = None

    def __post_init__(self) -> None:
        """Snapshot caller-owned sequences."""
        if not isinstance(self.retained_artifact_payload_types, Mapping):
            msg = "CoreProgram retained_artifact_payload_types must be a mapping."
            raise TypeError(msg)
        object.__setattr__(self, "donation_candidates", tuple(self.donation_candidates))
        object.__setattr__(
            self, "retained_artifact_keys", tuple(self.retained_artifact_keys)
        )
        object.__setattr__(
            self,
            "retained_artifact_payload_types",
            MappingProxyType(dict(self.retained_artifact_payload_types)),
        )


@dataclass(frozen=True, kw_only=True)
class MaterializedCoreProgram:
    """A declared core paired with exact dynamic arguments for one graph node."""

    name: str
    function: Callable[..., object]
    arguments: Mapping[str, object]
    requirements: CoreExecutionRequirements
    output_roles: object
    disposition: CoreExecutionDisposition
    donation_candidates: tuple[str, ...]
    disposition_reason: str | None = None
    scope: ProgramScope = ProgramScope.ANY
    retained_artifact_keys: tuple[ArtifactKey, ...] = ()
    replaces_program: str | None = None

    def __post_init__(self) -> None:
        """Snapshot the exact dynamic argument tree."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "donation_candidates", tuple(self.donation_candidates))
        object.__setattr__(
            self, "retained_artifact_keys", tuple(self.retained_artifact_keys)
        )


@runtime_checkable
class CoreProgramGraphAware(Protocol):
    """Native kernel interface publishing its complete immutable program graph."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return every named program needed by one period-kernel invocation."""
        ...


def core_program_graph(*, kernel: object) -> MappingProxyType[str, CoreProgram]:
    """Return one kernel's validated native graph.

    This is the only engine seam that reads ``core_programs()``. A kernel without a
    native graph is refused, as is one that also publishes a retired parallel
    declaration; a malformed declaration fails instead of falling back.
    """
    if not isinstance(kernel, CoreProgramGraphAware):
        msg = (
            f"{type(kernel).__name__} publishes no native core-program graph; a "
            "period kernel must implement core_programs()."
        )
        raise TypeError(msg)
    _reject_native_duplicate_authorities(kernel=kernel)
    return _snapshot_and_validate_graph(graph=kernel.core_programs())


def retained_artifact_payload_type(
    *, graph: Mapping[str, CoreProgram], key: ArtifactKey
) -> type[object] | None:
    """Return the unique producer-declared payload type for one retained artifact.

    ``None`` means no program retains ``key``. A retained key without a type is
    refused when a model needs its producer description, as are conflicting type
    declarations across programs in the same graph.
    """
    producers = tuple(
        (name, program)
        for name, program in graph.items()
        if key in program.retained_artifact_keys
    )
    if not producers:
        return None
    missing = tuple(
        name
        for name, program in producers
        if key not in program.retained_artifact_payload_types
    )
    if missing:
        msg = (
            f"CorePrograms {missing!r} retain artifact {key!r} without declaring "
            "its payload type."
        )
        raise ValueError(msg)
    declarations = tuple(
        (name, program.retained_artifact_payload_types[key])
        for name, program in producers
    )
    expected = declarations[0][1]
    if any(payload_type is not expected for _name, payload_type in declarations[1:]):
        msg = (
            f"CorePrograms disagree on the payload type of retained artifact "
            f"{key!r}: {declarations!r}."
        )
        raise ValueError(msg)
    return expected


def _validate_retained_artifact_payload_type_consistency(
    *, graph: Mapping[str, CoreProgram]
) -> None:
    """Require every retained key to have one exact producer payload type."""
    keys = sorted(
        {key for program in graph.values() for key in program.retained_artifact_keys}
    )
    for key in keys:
        retained_artifact_payload_type(graph=graph, key=key)


def select_programs(
    *,
    graph: Mapping[str, CoreProgram],
    retain_replay: bool,
    selected_artifact_keys: frozenset[ArtifactKey] = frozenset(),
) -> MappingProxyType[str, CoreProgram]:
    """Keep the programs one solve dispatches under its result retention.

    Every `ANY` program is kept. Normal replay retention selects every `REPLAY`
    variant. Persistence-oriented retention instead selects only `REPLAY` and
    additive `ARTIFACT` programs whose declared artifact identities intersect the
    exact model-authoritative keys selected for this regime-period cell. If no replay
    variant is selected, `VALUES_ONLY` is the value-producing alternative. A graph
    that leaves nothing to dispatch is refused: a period without a program would
    publish no value.
    """
    if type(retain_replay) is not bool:
        raise TypeError("retain_replay must be an exact bool.")
    if type(selected_artifact_keys) is not frozenset:
        raise TypeError("selected_artifact_keys must be an exact frozenset.")
    selected_replay_names = {
        name
        for name, program in graph.items()
        if program.scope is ProgramScope.REPLAY
        and (
            retain_replay
            or not selected_artifact_keys.isdisjoint(program.retained_artifact_keys)
        )
    }
    replaced_value_names = {
        cast("str", graph[name].replaces_program) for name in selected_replay_names
    }
    selected = {
        name: program
        for name, program in graph.items()
        if program.scope is ProgramScope.ANY
        or name in selected_replay_names
        or (
            program.scope is ProgramScope.VALUES_ONLY
            and name not in replaced_value_names
        )
        or (
            program.scope is ProgramScope.ARTIFACT
            and not selected_artifact_keys.isdisjoint(program.retained_artifact_keys)
        )
    }
    if not selected:
        msg = (
            "No core program is dispatched with "
            f"retain_replay={retain_replay!r} and selected_artifact_keys="
            f"{selected_artifact_keys!r}: the graph declares scopes "
            f"{ {name: program.scope.value for name, program in graph.items()}!r}."
        )
        raise ValueError(msg)
    return MappingProxyType(selected)


def _reject_native_duplicate_authorities(*, kernel: object) -> None:
    """Fail when a native graph publisher retains any parallel declaration seam."""
    duplicate_names = tuple(
        name
        for name in (
            "cores",
            "core",
            "unwrapped_core",
            "streamed_core",
            "build_lower_args",
            "build_core_program",
            "target_value_accesses",
            "output_roles",
            "core_for_output_layout",
        )
        if callable(getattr(kernel, name, None))
    )
    if duplicate_names:
        msg = (
            f"Native kernel {type(kernel).__name__} publishes duplicate execution "
            f"authorities alongside core_programs(): {duplicate_names!r}."
        )
        raise TypeError(msg)


def _snapshot_and_validate_graph(
    *, graph: Mapping[str, CoreProgram]
) -> MappingProxyType[str, CoreProgram]:
    """Snapshot a graph and validate its names and ownership declarations."""
    if not isinstance(graph, Mapping):
        msg = "A core-program graph must be a mapping from names to CoreProgram."
        raise TypeError(msg)
    snapshot = dict(graph)
    if not snapshot:
        msg = "A period kernel must declare at least one core program."
        raise ValueError(msg)
    for name, program in snapshot.items():
        if not isinstance(name, str) or not name:
            msg = f"A core-program graph name must be a non-empty string; got {name!r}."
            raise TypeError(msg)
        if not isinstance(program, CoreProgram):
            msg = f"Core-program graph entry {name!r} is not a CoreProgram."
            raise TypeError(msg)
        if program.name != name:
            msg = (
                "A core-program graph key must equal the program's declared name: "
                f"key={name!r}, program.name={program.name!r}."
            )
            raise ValueError(msg)
        _validate_program_declaration(program=program)
    _validate_replay_replacements(graph=snapshot)
    _validate_retained_artifact_payload_type_consistency(graph=snapshot)
    return MappingProxyType(snapshot)


def _validate_replay_replacements(*, graph: Mapping[str, CoreProgram]) -> None:
    """Require every replay alternative to replace one distinct values program."""
    replacements: dict[str, str] = {}
    for name, program in graph.items():
        if program.scope is not ProgramScope.REPLAY:
            continue
        replaced_name = cast("str", program.replaces_program)
        replaced = graph.get(replaced_name)
        if replaced is None or replaced.scope is not ProgramScope.VALUES_ONLY:
            msg = (
                f"Replay CoreProgram {name!r} declares replacement target "
                f"{replaced_name!r}, which must name a VALUES_ONLY program in the "
                "same graph."
            )
            raise ValueError(msg)
        previous = replacements.get(replaced_name)
        if previous is not None:
            msg = (
                f"Replay CorePrograms {previous!r} and {name!r} share replacement "
                f"target {replaced_name!r}; one values program may have only one "
                "replay alternative."
            )
            raise ValueError(msg)
        replacements[replaced_name] = name


def _validate_program_declaration(*, program: CoreProgram) -> None:
    """Validate declaration facts that do not depend on dynamic arguments."""
    if not callable(program.function):
        msg = f"CoreProgram {program.name!r} function must be callable."
        raise TypeError(msg)
    if not callable(program.argument_builder):
        msg = f"CoreProgram {program.name!r} argument_builder must be callable."
        raise TypeError(msg)
    if not isinstance(program.requirements, CoreExecutionRequirements):
        msg = f"CoreProgram {program.name!r} requirements have the wrong type."
        raise TypeError(msg)
    if not isinstance(program.disposition, CoreExecutionDisposition):
        msg = f"CoreProgram {program.name!r} disposition has the wrong type."
        raise TypeError(msg)
    if not isinstance(program.scope, ProgramScope):
        msg = f"CoreProgram {program.name!r} scope has the wrong type."
        raise TypeError(msg)
    _validate_retention_declaration(program=program)
    if program.output_roles is None:
        msg = f"CoreProgram {program.name!r} must declare output_roles."
        raise ValueError(msg)
    _validate_disposition_reason(program=program)
    donations = program.donation_candidates
    if len(donations) != len(set(donations)) or any(
        not isinstance(name, str) or not name for name in donations
    ):
        msg = (
            f"CoreProgram {program.name!r} donation_candidates must be unique, "
            "non-empty argument names."
        )
        raise ValueError(msg)


def _validate_retention_declaration(
    *, program: CoreProgram | MaterializedCoreProgram
) -> None:
    """Require artifact-scoped programs to name their exact retained outputs."""
    retention_keys = program.retained_artifact_keys
    if any(type(key) is not ArtifactKey for key in retention_keys):
        msg = (
            f"CoreProgram {program.name!r} retained_artifact_keys must contain exact "
            "public ArtifactKey values."
        )
        raise TypeError(msg)
    if len(retention_keys) != len(set(retention_keys)):
        msg = f"CoreProgram {program.name!r} retained_artifact_keys must be unique."
        raise ValueError(msg)
    if isinstance(program, CoreProgram):
        _validate_retained_artifact_payload_types(program=program)
    artifact_scopes = {ProgramScope.REPLAY, ProgramScope.ARTIFACT}
    if (program.scope in artifact_scopes) != bool(retention_keys):
        msg = (
            f"CoreProgram {program.name!r} must declare retained_artifact_keys "
            "exactly when its scope is REPLAY or ARTIFACT."
        )
        raise ValueError(msg)
    if program.scope is ProgramScope.REPLAY:
        if not isinstance(program.replaces_program, str) or not (
            program.replaces_program
        ):
            msg = (
                f"Replay CoreProgram {program.name!r} must name the VALUES_ONLY "
                "program it replaces."
            )
            raise ValueError(msg)
    elif program.replaces_program is not None:
        msg = (
            f"CoreProgram {program.name!r} may declare replaces_program only when "
            "its scope is REPLAY."
        )
        raise ValueError(msg)


def _validate_retained_artifact_payload_types(*, program: CoreProgram) -> None:
    """Require one exact payload type for every artifact this program retains."""
    payload_types = program.retained_artifact_payload_types
    if any(type(key) is not ArtifactKey for key in payload_types):
        msg = (
            f"CoreProgram {program.name!r} retained artifact payload types must "
            "be keyed by exact ArtifactKey values."
        )
        raise TypeError(msg)
    if any(
        not isinstance(payload_type, type) for payload_type in payload_types.values()
    ):
        msg = (
            f"CoreProgram {program.name!r} retained artifact payload values "
            "must be types."
        )
        raise TypeError(msg)
    retention_keys = set(program.retained_artifact_keys)
    unretained = sorted(payload_types.keys() - retention_keys)
    missing = sorted(retention_keys - payload_types.keys())
    if unretained:
        msg = (
            f"CoreProgram {program.name!r} declares payload types for artifacts "
            f"it does not retain: {tuple(unretained)!r}."
        )
        raise ValueError(msg)
    if missing:
        msg = (
            f"CoreProgram {program.name!r} retains artifacts without declaring "
            f"their payload types: {tuple(missing)!r}."
        )
        raise ValueError(msg)


def _validate_disposition_reason(
    *, program: CoreProgram | MaterializedCoreProgram
) -> None:
    """Require an explicit, stable explanation for every dense route."""
    reason = program.disposition_reason
    if program.disposition is CoreExecutionDisposition.PLANNED:
        if reason is not None:
            msg = f"Planned CoreProgram {program.name!r} cannot declare a reason."
            raise ValueError(msg)
        return
    if not isinstance(reason, str) or not reason.strip():
        msg = (
            f"Dense CoreProgram {program.name!r} must declare a non-empty "
            "disposition_reason."
        )
        raise ValueError(msg)


def materialize_core_program(
    *, program: CoreProgram, context: CoreBuildContext
) -> MaterializedCoreProgram:
    """Build and validate one program's exact dynamic argument mapping."""
    arguments = program.argument_builder(context)
    if not isinstance(arguments, Mapping):
        msg = f"CoreProgram {program.name!r} argument_builder must return a mapping."
        raise TypeError(msg)
    materialized = MaterializedCoreProgram(
        name=program.name,
        function=program.function,
        arguments=arguments,
        requirements=program.requirements,
        output_roles=program.output_roles,
        disposition=program.disposition,
        donation_candidates=program.donation_candidates,
        disposition_reason=program.disposition_reason,
        scope=program.scope,
        retained_artifact_keys=program.retained_artifact_keys,
        replaces_program=program.replaces_program,
    )
    missing_donations = set(materialized.donation_candidates) - set(
        materialized.arguments
    )
    if missing_donations:
        msg = (
            f"CoreProgram {program.name!r} donation candidates are absent from its "
            f"arguments: {sorted(missing_donations)!r}."
        )
        raise ValueError(msg)
    return materialized


@dataclass(frozen=True, kw_only=True)
class ResolvedCoreProgram:
    """A core with planner-owned choices bound into its compilation identity."""

    name: str
    function: Callable[..., object]
    arguments: Mapping[str, object]
    static_kwargs: Mapping[str, int]
    requirements: CoreExecutionRequirements
    output_roles: object
    disposition: CoreExecutionDisposition
    donation_candidates: tuple[str, ...]
    tile_widths: Mapping[str, int]
    specialization_key: Hashable
    """Static program fragment composed into the engine's full lowering key."""
    input_transfer_plan: tuple[ResolvedValueTransfer, ...]
    disposition_reason: str | None = None
    scope: ProgramScope = ProgramScope.ANY
    retained_artifact_keys: tuple[ArtifactKey, ...] = ()
    replaces_program: str | None = None

    def __post_init__(self) -> None:
        """Snapshot the materialized argument and planning containers."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(
            self,
            "static_kwargs",
            MappingProxyType(dict(self.static_kwargs)),
        )
        object.__setattr__(
            self,
            "tile_widths",
            MappingProxyType(dict(self.tile_widths)),
        )
        object.__setattr__(
            self, "retained_artifact_keys", tuple(self.retained_artifact_keys)
        )
        object.__setattr__(self, "input_transfer_plan", tuple(self.input_transfer_plan))
        object.__setattr__(self, "donation_candidates", tuple(self.donation_candidates))


def resolve_core_program(
    *,
    program: MaterializedCoreProgram,
    tile_widths: Mapping[str, object] | None = None,
    input_transfer_plan: tuple[ResolvedValueTransfer, ...] = (),
) -> ResolvedCoreProgram:
    """Validate and bind planner-owned tile widths into program.

    Widths are static compilation choices kept separate from the dynamic lowering
    arguments. The engine supplies them as JAX static keyword arguments while retaining
    the raw core's identity, so equivalent solves reuse JAX's trace cache. A streamable
    axis requires an explicit planner choice; silently using its full extent would turn
    a streaming declaration into full materialization.
    """
    _validate_core_program(program=program)
    requested_widths = {} if tile_widths is None else dict(tile_widths)
    if program.disposition is CoreExecutionDisposition.PLANNED:
        (
            resolved_input_transfer_plan,
            input_transfer_specialization_key,
        ) = _resolve_input_transfer_plan(program=program, plan=input_transfer_plan)
    else:
        if input_transfer_plan:
            msg = (
                f"CoreProgram {program.name!r} with disposition "
                f"{program.disposition.value!r} cannot carry a resolved input plan."
            )
            raise ValueError(msg)
        resolved_input_transfer_plan = ()
        input_transfer_specialization_key = ()
    axes = program.requirements.streamable_axes
    axis_names = [axis.name for axis in axes]

    unknown_axes = requested_widths.keys() - set(axis_names)
    if unknown_axes:
        msg = f"Tile widths name an unknown streamable axis: {sorted(unknown_axes)!r}."
        raise ValueError(msg)
    missing_axes = set(axis_names) - requested_widths.keys()
    if missing_axes:
        msg = f"Tile width is required for streamable axes: {sorted(missing_axes)!r}."
        raise ValueError(msg)

    resolved_widths: dict[str, int] = {}
    width_bindings: dict[str, int] = {}
    compilation_axes: list[Hashable] = []
    for axis in axes:
        width = _validate_tile_width(
            axis=axis,
            width=requested_widths[axis.name],
        )
        if axis.requested_width is not None and width != axis.requested_width:
            msg = (
                f"Tile width {width} for axis {axis.name!r} does not match its "
                f"requested width {axis.requested_width}."
            )
            raise ValueError(msg)
        resolved_widths[axis.name] = width
        width_bindings[axis.width_keyword] = width
        compilation_axes.append(
            (
                axis.name,
                axis.coordinate_names,
                axis.coordinate_extents,
                axis.canonical_order,
                axis.reduction.semantic_key,
                axis.width_keyword,
                width,
            )
        )

    return ResolvedCoreProgram(
        name=program.name,
        function=program.function,
        arguments=apply_value_transfer_plan(
            arguments=program.arguments,
            plan=resolved_input_transfer_plan,
        ),
        static_kwargs=width_bindings,
        requirements=program.requirements,
        output_roles=program.output_roles,
        disposition=program.disposition,
        donation_candidates=program.donation_candidates,
        disposition_reason=program.disposition_reason,
        scope=program.scope,
        retained_artifact_keys=program.retained_artifact_keys,
        replaces_program=program.replaces_program,
        tile_widths=resolved_widths,
        input_transfer_plan=resolved_input_transfer_plan,
        specialization_key=(
            "core-program",
            _CORE_PROGRAM_VERSION,
            program.disposition.value,
            program.disposition_reason,
            program.scope.value,
            program.retained_artifact_keys,
            program.replaces_program,
            program.donation_candidates,
            tuple(compilation_axes),
            input_transfer_specialization_key,
        ),
    )


def initial_core_tile_widths(
    *, program: CoreProgram | MaterializedCoreProgram
) -> MappingProxyType[str, int]:
    """Choose each full product width unless its declaration requests an override."""
    if program.disposition is not CoreExecutionDisposition.PLANNED:
        if program.requirements.streamable_axes:
            msg = (
                f"CoreProgram {program.name!r} has disposition "
                f"{program.disposition.value!r} but declares streamable axes."
            )
            raise ValueError(msg)
        return MappingProxyType({})
    result: dict[str, int] = {}
    for axis in program.requirements.streamable_axes:
        _validate_coordinate_declaration(axis=axis)
        if axis.extent <= 1:
            msg = (
                f"Streamable axis {axis.name!r} must have extent greater than one; "
                "the program must declare a dense disposition otherwise."
            )
            raise ValueError(msg)
        requested_width = _validate_requested_width(axis=axis)
        result[axis.name] = axis.extent if requested_width is None else requested_width
    return MappingProxyType(result)


def _validate_core_program(*, program: MaterializedCoreProgram) -> None:
    """Validate a complete declaration before planner choices inspect it."""
    _validate_materialized_declaration(program=program)
    try:
        weakref.ref(program.function)
    except TypeError as error:
        msg = (
            "CoreProgram function must be weak-referenceable so JAX can retain "
            "its raw callable identity."
        )
        raise TypeError(msg) from error

    function_type = type(program.function)
    if (
        function_type.__eq__ is not object.__eq__
        or function_type.__hash__ is not object.__hash__
    ):
        msg = (
            "CoreProgram function must use identity-based equality and hashing "
            "so JAX can use its raw callable as a compilation-cache key."
        )
        raise TypeError(msg)
    _validate_target_value_accesses(program=program)

    axes = program.requirements.streamable_axes
    if program.disposition is not CoreExecutionDisposition.PLANNED and axes:
        msg = (
            f"CoreProgram {program.name!r} has disposition "
            f"{program.disposition.value!r} but declares streamable axes."
        )
        raise ValueError(msg)
    axis_names = [axis.name for axis in axes]
    if len(axis_names) != len(set(axis_names)):
        msg = f"Core program has duplicate streamable axis names: {axis_names!r}."
        raise ValueError(msg)

    width_keywords = [axis.width_keyword for axis in axes]
    if len(width_keywords) != len(set(width_keywords)):
        msg = f"Core program has duplicate planner width keywords: {width_keywords!r}."
        raise ValueError(msg)

    for axis in axes:
        _validate_streamable_axis(axis=axis, arguments=program.arguments)
        _validate_width_keyword(function=program.function, axis=axis)


def _validate_materialized_declaration(*, program: MaterializedCoreProgram) -> None:
    """Validate resolved authority fields independently of planner choices."""
    if not isinstance(program.name, str) or not program.name:
        msg = "A materialized CoreProgram name must be a non-empty string."
        raise TypeError(msg)
    if not callable(program.function):
        msg = f"CoreProgram {program.name!r} function must be callable."
        raise TypeError(msg)
    if not isinstance(program.requirements, CoreExecutionRequirements):
        msg = f"CoreProgram {program.name!r} requirements have the wrong type."
        raise TypeError(msg)
    if not isinstance(program.disposition, CoreExecutionDisposition):
        msg = f"CoreProgram {program.name!r} disposition has the wrong type."
        raise TypeError(msg)
    if not isinstance(program.scope, ProgramScope):
        msg = f"CoreProgram {program.name!r} scope has the wrong type."
        raise TypeError(msg)
    _validate_retention_declaration(program=program)
    if program.output_roles is None:
        msg = f"CoreProgram {program.name!r} must declare output_roles."
        raise ValueError(msg)
    _validate_disposition_reason(program=program)
    donations = program.donation_candidates
    if len(donations) != len(set(donations)) or any(
        not isinstance(name, str) or not name for name in donations
    ):
        msg = (
            f"CoreProgram {program.name!r} donation_candidates must be unique, "
            "non-empty argument names."
        )
        raise ValueError(msg)


def _resolve_input_transfer_plan(
    *,
    program: MaterializedCoreProgram,
    plan: tuple[ResolvedValueTransfer, ...],
) -> tuple[tuple[ResolvedValueTransfer, ...], tuple[Hashable, ...]]:
    """Match resolved transfers to declarations and derive lowering-only identity."""
    transfers = tuple(plan)
    transfer_by_access: dict[
        tuple[ValueArtifactAddress, ValueConsumerAddress], ResolvedValueTransfer
    ] = {}
    for transfer in transfers:
        if not isinstance(transfer, ResolvedValueTransfer):
            msg = (
                "An input transfer plan may contain only ResolvedValueTransfer entries."
            )
            raise TypeError(msg)
        key = (transfer.target, transfer.source)
        if key in transfer_by_access:
            msg = f"Input transfer plan has a duplicate target/source pair: {key!r}."
            raise ValueError(msg)
        transfer_by_access[key] = transfer

    access_keys = tuple(
        (access.target, access.source)
        for access in program.requirements.target_value_accesses
    )
    declared = set(access_keys)
    planned = set(transfer_by_access)
    if declared != planned:
        missing = tuple(key for key in access_keys if key not in planned)
        unexpected = tuple(key for key in transfer_by_access if key not in declared)
        msg = (
            "Input transfer plan must match every declared target-value access "
            f"one-to-one; missing={missing!r}, unexpected={unexpected!r}."
        )
        raise ValueError(msg)

    ordered = tuple(transfer_by_access[key] for key in access_keys)
    specialization_keys: list[Hashable] = []
    for access, transfer in zip(
        program.requirements.target_value_accesses, ordered, strict=True
    ):
        _validate_transfer_argument_metadata(
            program=program, access=access, transfer=transfer
        )
        try:
            hash(transfer.specialization_key)
        except TypeError as error:
            msg = "An input transfer specialization key must be hashable."
            raise TypeError(msg) from error
        specialization_keys.append(transfer.specialization_key)
    return ordered, tuple(specialization_keys)


def _validate_target_value_accesses(*, program: MaterializedCoreProgram) -> None:
    """Validate exact core-input locators without constraining artifact fan-out."""
    locators: set[tuple[object, tuple[str | int, ...]]] = set()
    source_node: tuple[int, str, str] | None = None
    for access in program.requirements.target_value_accesses:
        if not isinstance(access, _TargetValueAccess):
            msg = "Core target_value_accesses must contain _TargetValueAccess entries."
            raise TypeError(msg)

        node = (
            access.source.source_period,
            access.source.source_regime,
            access.source.core_key,
        )
        if source_node is None:
            source_node = node
        elif node != source_node:
            msg = (
                "All target-value accesses in one CoreProgram must name the same "
                f"source period/regime/core; got {source_node!r} and {node!r}."
            )
            raise ValueError(msg)

        locator = (access.source.channel, access.source.path)
        if locator in locators:
            msg = (
                f"Core program has a duplicate target-value argument path: {locator!r}."
            )
            raise ValueError(msg)
        locators.add(locator)
        _target_value_argument_leaf(program=program, access=access)


def _target_value_argument_leaf(
    *, program: MaterializedCoreProgram, access: _TargetValueAccess
) -> _TransferArgumentLeaf:
    """Resolve one declared consumer path to an array-like lowering leaf."""
    channel = access.source.channel.value
    if channel not in program.arguments:
        msg = (
            f"Target-value input channel {channel!r} is missing from program arguments."
        )
        raise ValueError(msg)

    value: object = program.arguments[channel]
    traversed: list[str | int] = []
    for segment in access.source.path:
        traversed.append(segment)
        if isinstance(value, Mapping):
            if segment not in value:
                msg = (
                    f"Target-value argument path {(channel, *traversed)!r} is missing."
                )
                raise ValueError(msg)
            value = value[segment]
            continue
        if isinstance(value, tuple):
            if type(segment) is not int or segment >= len(value):
                msg = (
                    f"Target-value argument path {(channel, *traversed)!r} does not "
                    "select an existing sequence item."
                )
                raise ValueError(msg)
            value = value[segment]
            continue
        msg = (
            f"Target-value argument path {(channel, *traversed)!r} traverses a "
            "non-container value."
        )
        raise ValueError(msg)

    if getattr(value, "shape", None) is None or getattr(value, "dtype", None) is None:
        msg = (
            f"Target-value argument path {(channel, *access.source.path)!r} must "
            "resolve to an array-like leaf with shape and dtype."
        )
        raise TypeError(msg)
    return cast("_TransferArgumentLeaf", value)


def _validate_transfer_argument_metadata(
    *,
    program: MaterializedCoreProgram,
    access: _TargetValueAccess,
    transfer: ResolvedValueTransfer,
) -> None:
    """Reject a correctly addressed transfer resolved from a stale template."""
    leaf = _target_value_argument_leaf(program=program, access=access)
    actual_shape = tuple(leaf.shape)
    if actual_shape != transfer.expected_shape:
        msg = (
            f"Input transfer shape mismatch at {access.source!r}: "
            f"argument has {actual_shape}, plan expects {transfer.expected_shape}."
        )
        raise ValueError(msg)

    actual_dtype = leaf.dtype
    if actual_dtype != transfer.expected_dtype:
        msg = (
            f"Input transfer dtype mismatch at {access.source!r}: "
            f"argument has {actual_dtype}, plan expects {transfer.expected_dtype}."
        )
        raise TypeError(msg)

    actual_sharding = getattr(leaf, "sharding", None)
    if actual_sharding != transfer.stored_sharding:
        msg = (
            f"Input transfer stored-sharding mismatch at {access.source!r}: "
            f"argument has {actual_sharding}, plan expects {transfer.stored_sharding}."
        )
        raise ValueError(msg)


def _validate_streamable_axis(
    *,
    axis: StreamableProductAxis,
    arguments: Mapping[str, object],
) -> None:
    """Fail closed for product declarations outside the supported contract."""
    _validate_coordinate_declaration(axis=axis)
    _validate_requested_width(axis=axis)
    if axis.canonical_order != "c":
        msg = f"Streamable axis {axis.name!r} canonical order must be 'c'."
        raise ValueError(msg)
    _validate_reduction_semantics(axis=axis)
    _validate_axis_width_keyword(axis=axis, arguments=arguments)
    for coordinate_name, coordinate_extent in zip(
        axis.coordinate_names, axis.coordinate_extents, strict=True
    ):
        _validate_coordinate_argument(
            axis_name=axis.name,
            coordinate_name=coordinate_name,
            coordinate_extent=coordinate_extent,
            arguments=arguments,
        )


def _validate_coordinate_declaration(*, axis: StreamableProductAxis) -> None:
    """Validate the names, extents, and global identities of one product."""
    if len(axis.coordinate_names) != len(axis.coordinate_extents):
        msg = (
            f"Streamable axis {axis.name!r} coordinate names and extents must "
            "have the same length."
        )
        raise ValueError(msg)
    if len(axis.coordinate_names) != len(set(axis.coordinate_names)):
        msg = (
            f"Streamable axis {axis.name!r} has duplicate coordinate names: "
            f"{axis.coordinate_names!r}."
        )
        raise ValueError(msg)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int)
        for extent in axis.coordinate_extents
    ):
        msg = f"Streamable axis {axis.name!r} coordinate extents must be integers."
        raise TypeError(msg)
    if any(extent <= 0 for extent in axis.coordinate_extents):
        msg = f"Streamable axis {axis.name!r} coordinate extents must be positive."
        raise ValueError(msg)
    if axis.extent > _INT32_MAX:
        msg = (
            f"Streamable axis {axis.name!r} exceeds the int32 global action "
            f"identity range: {axis.extent}."
        )
        raise ValueError(msg)


def _validate_reduction_semantics(*, axis: StreamableProductAxis) -> None:
    """Require stable, hashable semantics for the axis reduction."""
    if not isinstance(axis.reduction, ReductionSemantics):
        msg = (
            f"Streamable axis {axis.name!r} reduction must expose a stable "
            "semantic_key."
        )
        raise TypeError(msg)
    try:
        hash(axis.reduction.semantic_key)
    except TypeError as exc:
        msg = f"Streamable axis {axis.name!r} reduction semantic_key must be hashable."
        raise TypeError(msg) from exc


def _validate_axis_width_keyword(
    *, axis: StreamableProductAxis, arguments: Mapping[str, object]
) -> None:
    """Keep the planner-owned width distinct from dynamic arguments."""
    if not axis.width_keyword:
        msg = f"Streamable axis {axis.name!r} must declare a width keyword."
        raise ValueError(msg)
    if axis.width_keyword in arguments:
        msg = (
            f"Streamable axis {axis.name!r} width keyword "
            f"{axis.width_keyword!r} is already present in dynamic arguments."
        )
        raise ValueError(msg)


def _validate_requested_width(*, axis: StreamableProductAxis) -> int | None:
    """Validate a solver-requested width against its declared product."""
    width = axis.requested_width
    if width is None:
        return None
    if type(width) is not int:
        msg = f"Streamable axis {axis.name!r} requested width must be an integer."
        raise TypeError(msg)
    if width <= 0:
        msg = f"Streamable axis {axis.name!r} requested width must be positive."
        raise ValueError(msg)
    if width > axis.extent:
        msg = (
            f"Streamable axis {axis.name!r} requested width {width} exceeds its "
            f"product extent {axis.extent}."
        )
        raise ValueError(msg)
    return width


def _validate_tile_width(*, axis: StreamableProductAxis, width: object) -> int:
    """Validate one planner-selected width against its declared product."""
    if isinstance(width, bool) or not isinstance(width, int):
        msg = f"Tile width for axis {axis.name!r} must be an integer."
        raise TypeError(msg)
    if width <= 0:
        msg = f"Tile width for axis {axis.name!r} must be positive."
        raise ValueError(msg)
    if width > axis.extent:
        msg = (
            f"Tile width {width} for axis {axis.name!r} exceeds its product "
            f"extent {axis.extent}."
        )
        raise ValueError(msg)
    return width


def _validate_coordinate_argument(
    *,
    axis_name: str,
    coordinate_name: ActionName,
    coordinate_extent: int,
    arguments: Mapping[str, object],
) -> None:
    """Tie one declared coordinate to the exact dynamic lowering grid."""
    if coordinate_name not in arguments:
        msg = (
            f"Streamable axis {axis_name!r} coordinate {coordinate_name!r} is "
            "missing from the program arguments."
        )
        raise ValueError(msg)
    coordinate = arguments[coordinate_name]
    shape = getattr(coordinate, "shape", None)
    if shape is None or len(shape) != 1:
        msg = (
            f"Streamable axis {axis_name!r} coordinate {coordinate_name!r} must "
            "be one-dimensional."
        )
        raise ValueError(msg)
    if shape[0] == 0:
        msg = (
            f"Streamable axis {axis_name!r} coordinate {coordinate_name!r} must "
            "be non-empty."
        )
        raise ValueError(msg)
    if shape[0] != coordinate_extent:
        msg = (
            f"Streamable axis {axis_name!r} coordinate {coordinate_name!r} has "
            f"extent {shape[0]}, but the declaration says {coordinate_extent}."
        )
        raise ValueError(msg)


def _validate_width_keyword(
    *, function: Callable[..., object], axis: StreamableProductAxis
) -> None:
    """Require the raw core to accept the planner's static width binding."""
    signature = inspect.signature(function)
    parameter = signature.parameters.get(axis.width_keyword)
    accepts_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    if parameter is None and not accepts_kwargs:
        msg = (
            f"Streamable axis {axis.name!r} width keyword "
            f"{axis.width_keyword!r} is not accepted by the core function."
        )
        raise TypeError(msg)
    if parameter is not None and parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        msg = (
            f"Streamable axis {axis.name!r} width keyword "
            f"{axis.width_keyword!r} must be accepted as a keyword by the core "
            "function."
        )
        raise TypeError(msg)
