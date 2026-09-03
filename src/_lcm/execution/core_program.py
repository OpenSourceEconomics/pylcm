"""Solver-declared core programs for engine-owned lowering.

Solvers declare a core's dynamic arguments, output roles, and static execution
requirements. The engine validates the declaration and binds planner-owned choices
before lowering. Native dense routes remain explicit programs; unmigrated kernels
cross one central adapter as ``LEGACY_UNPLANNED`` programs.
"""

import inspect
import math
import weakref
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueConsumerAddress,
    apply_value_transfer_plan,
)
from _lcm.typing import ActionName

_CORE_PROGRAM_VERSION = 4
_INT32_MAX = 2_147_483_647
_INITIAL_TILE_WIDTH_CAP = 64


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
    LEGACY_UNPLANNED = "legacy-unplanned"


class ProgramScope(StrEnum):
    """Which result retentions dispatch one declared core.

    - `ANY`: dispatched under every retention.
    - `VALUES_ONLY`: dispatched only when the solve retains no replay artifacts.
    - `REPLAY`: dispatched only when the solve retains replay artifacts.

    A kernel publishing both scoped variants of one body lets a values-only solve
    skip the replay outputs' assembly instead of computing and discarding them.
    """

    ANY = "any"
    VALUES_ONLY = "values-only"
    REPLAY = "replay"


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
    output_roles: object | None
    disposition: CoreExecutionDisposition
    disposition_reason: str | None = None
    donation_candidates: tuple[str, ...] = ()
    scope: ProgramScope = ProgramScope.ANY

    def __post_init__(self) -> None:
        """Snapshot caller-owned sequences."""
        object.__setattr__(self, "donation_candidates", tuple(self.donation_candidates))


@dataclass(frozen=True, kw_only=True)
class MaterializedCoreProgram:
    """A declared core paired with exact dynamic arguments for one graph node."""

    name: str
    function: Callable[..., object]
    arguments: Mapping[str, object]
    requirements: CoreExecutionRequirements
    output_roles: object | None
    disposition: CoreExecutionDisposition
    donation_candidates: tuple[str, ...]
    disposition_reason: str | None = None
    scope: ProgramScope = ProgramScope.ANY

    def __post_init__(self) -> None:
        """Snapshot the exact dynamic argument tree."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "donation_candidates", tuple(self.donation_candidates))


@runtime_checkable
class CoreProgramGraphAware(Protocol):
    """Native kernel interface publishing its complete immutable program graph."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return every named program needed by one period-kernel invocation."""
        ...


@runtime_checkable
class _LegacyCoreKernel(Protocol):
    """Old core enumeration and argument-building interface, read only here."""

    def cores(self) -> Mapping[str, Callable[..., object]]:
        """Return legacy core callables by name."""
        ...

    def build_lower_args(
        self, *, core_key: str, **kwargs: object
    ) -> Mapping[str, object]:
        """Build one legacy core's arguments."""
        ...


@dataclass(frozen=True, kw_only=True)
class _LegacyArgumentBuilder:
    """Central adapter from a build context to one legacy argument builder."""

    kernel: _LegacyCoreKernel
    core_name: str

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Invoke the old builder without exposing it to an engine caller."""
        optional_edge = (
            {}
            if context.edge_regime_to_V_arr is None
            else {"edge_regime_to_V_arr": context.edge_regime_to_V_arr}
        )
        return self.kernel.build_lower_args(
            core_key=self.core_name,
            state_action_space=context.state_action_space,
            next_regime_to_V_arr=context.next_regime_to_V_arr,
            next_regime_to_continuation=context.next_regime_to_continuation,
            flat_params=context.flat_params,
            period=context.period,
            ages=context.ages,
            **optional_edge,
        )


def core_program_graph(*, kernel: object) -> MappingProxyType[str, CoreProgram]:
    """Return one validated native graph or adapt a legacy kernel exactly once.

    This is the only engine seam permitted to read ``cores()``,
    ``build_lower_args()``, or retired parallel program, target-access, and
    output-layout methods. A malformed native declaration fails instead of falling
    back; only kernels with no native graph cross the legacy adapter.
    """
    if isinstance(kernel, CoreProgramGraphAware):
        _reject_native_duplicate_authorities(kernel=kernel)
        return _snapshot_and_validate_graph(graph=kernel.core_programs(), native=True)
    return _legacy_core_program_graph(kernel=kernel)


def select_programs(
    *, graph: Mapping[str, CoreProgram], retain_replay: bool
) -> MappingProxyType[str, CoreProgram]:
    """Keep the programs one solve dispatches under its result retention.

    Every `ANY` program is kept, plus exactly the programs of the one scope the
    retention selects. A graph that leaves nothing to dispatch is refused: a period
    without a program would publish no value.
    """
    dispatched_scope = (
        ProgramScope.REPLAY if retain_replay else ProgramScope.VALUES_ONLY
    )
    selected = {
        name: program
        for name, program in graph.items()
        if program.scope in (ProgramScope.ANY, dispatched_scope)
    }
    if not selected:
        msg = (
            f"No core program is dispatched with retain_replay={retain_replay!r}: "
            "the graph declares scopes "
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
    *, graph: Mapping[str, CoreProgram], native: bool
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
        _validate_program_declaration(program=program, native=native)
    return MappingProxyType(snapshot)


def _validate_program_declaration(*, program: CoreProgram, native: bool) -> None:
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
    if native and program.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED:
        msg = "A native CoreProgram cannot declare LEGACY_UNPLANNED."
        raise ValueError(msg)
    if (
        not native
        and program.disposition is not CoreExecutionDisposition.LEGACY_UNPLANNED
    ):
        msg = "The legacy core adapter may emit only LEGACY_UNPLANNED programs."
        raise ValueError(msg)
    if native and program.output_roles is None:
        msg = f"Native CoreProgram {program.name!r} must declare output_roles."
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


def _validate_disposition_reason(
    *, program: CoreProgram | MaterializedCoreProgram
) -> None:
    """Require an explicit, stable explanation for every non-planned route."""
    reason = program.disposition_reason
    if program.disposition is CoreExecutionDisposition.PLANNED:
        if reason is not None:
            msg = f"Planned CoreProgram {program.name!r} cannot declare a reason."
            raise ValueError(msg)
        return
    if program.disposition is CoreExecutionDisposition.DENSE:
        if not isinstance(reason, str) or not reason.strip():
            msg = (
                f"Dense CoreProgram {program.name!r} must declare a non-empty "
                "disposition_reason."
            )
            raise ValueError(msg)
        return
    if reason != "legacy_adapter":
        msg = (
            f"Legacy CoreProgram {program.name!r} must use the central "
            "'legacy_adapter' disposition reason."
        )
        raise ValueError(msg)


def _legacy_core_program_graph(*, kernel: object) -> MappingProxyType[str, CoreProgram]:
    """Synthesize explicitly unplanned declarations for one unmigrated kernel."""
    if not isinstance(kernel, _LegacyCoreKernel):
        msg = (
            f"{type(kernel).__name__} publishes neither a native core-program graph "
            "nor the supported legacy core interface."
        )
        raise TypeError(msg)
    if callable(getattr(kernel, "build_core_program", None)):
        msg = (
            f"Legacy kernel {type(kernel).__name__} still publishes the duplicate "
            "build_core_program declaration; migrate it to one native core_programs() "
            "graph before execution."
        )
        raise TypeError(msg)
    if callable(getattr(kernel, "output_roles", None)) or callable(
        getattr(kernel, "core_for_output_layout", None)
    ):
        msg = (
            f"Legacy kernel {type(kernel).__name__} still publishes the duplicate "
            "output-layout declaration; migrate it to one native core_programs() "
            "graph before execution."
        )
        raise TypeError(msg)
    cores = kernel.cores()
    if not isinstance(cores, Mapping):
        msg = "Legacy cores() must return a mapping."
        raise TypeError(msg)
    programs: dict[str, CoreProgram] = {}
    for name, function in cores.items():
        target_value_accesses = getattr(kernel, "target_value_accesses", None)
        accesses = (
            target_value_accesses(core_key=name)
            if callable(target_value_accesses)
            else ()
        )
        programs[name] = CoreProgram(
            name=name,
            function=function,
            argument_builder=_LegacyArgumentBuilder(kernel=kernel, core_name=name),
            requirements=CoreExecutionRequirements(target_value_accesses=accesses),
            output_roles=None,
            disposition=CoreExecutionDisposition.LEGACY_UNPLANNED,
            disposition_reason="legacy_adapter",
        )
    return _snapshot_and_validate_graph(graph=programs, native=False)


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
    output_roles: object | None
    disposition: CoreExecutionDisposition
    donation_candidates: tuple[str, ...]
    tile_widths: Mapping[str, int]
    specialization_key: Hashable
    """Static program fragment composed into the engine's full lowering key."""
    input_transfer_plan: tuple[ResolvedValueTransfer, ...]
    disposition_reason: str | None = None
    scope: ProgramScope = ProgramScope.ANY

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
        tile_widths=resolved_widths,
        input_transfer_plan=resolved_input_transfer_plan,
        specialization_key=(
            "core-program",
            _CORE_PROGRAM_VERSION,
            program.disposition.value,
            program.disposition_reason,
            program.scope.value,
            program.donation_candidates,
            tuple(compilation_axes),
            input_transfer_specialization_key,
        ),
    )


def initial_core_tile_widths(
    *, program: CoreProgram | MaterializedCoreProgram
) -> MappingProxyType[str, int]:
    """Choose the shared bounded bootstrap width for every planned product axis."""
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
        upper_bound = min(_INITIAL_TILE_WIDTH_CAP, axis.extent - 1)
        result[axis.name] = 1 << (upper_bound.bit_length() - 1)
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
    if program.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED:
        if program.output_roles is not None:
            msg = "A legacy-unplanned CoreProgram cannot declare output_roles."
            raise ValueError(msg)
    elif program.output_roles is None:
        msg = f"Native CoreProgram {program.name!r} must declare output_roles."
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
