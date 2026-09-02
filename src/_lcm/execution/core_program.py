"""Solver-declared core programs for engine-owned lowering.

Solvers declare a core's dynamic arguments, output roles, and static execution
requirements. The engine validates the declaration and binds planner-owned choices
before lowering. Kernels without a program for a named core use their dense execution
path for that unsupported route.
"""

import inspect
import math
import weakref
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueConsumerAddress,
    apply_value_transfer_plan,
)
from _lcm.typing import ActionName

_CORE_PROGRAM_VERSION = 2
_INT32_MAX = 2_147_483_647


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
    """Static requirements that the AOT planner must resolve for a core."""

    streamable_axes: tuple[StreamableProductAxis, ...] = ()
    target_value_accesses: tuple[_TargetValueAccess, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot the declared axes and exact target-value reads."""
        object.__setattr__(self, "streamable_axes", tuple(self.streamable_axes))
        object.__setattr__(
            self, "target_value_accesses", tuple(self.target_value_accesses)
        )


@dataclass(frozen=True, kw_only=True)
class CoreProgram:
    """An unlowered core plus its dynamic arguments and static requirements."""

    function: Callable[..., object]
    arguments: Mapping[str, object]
    requirements: CoreExecutionRequirements
    output_roles: object

    def __post_init__(self) -> None:
        """Snapshot the exact dynamic lowering arguments."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@runtime_checkable
class _TargetValueAccessAware(Protocol):  # noqa: PYI046
    """Kernel seam declaring exact value inputs independently of core planning."""

    def target_value_accesses(self, *, core_key: str) -> tuple[_TargetValueAccess, ...]:
        """Return exact logical target reads for one dense or planned core."""
        ...


@runtime_checkable
class CoreProgramAware(Protocol):
    """Protocol implemented by kernels that declare core programs."""

    def build_core_program(
        self,
        *,
        core_key: str,
        arguments: Mapping[str, object],
    ) -> CoreProgram | None:
        """Build the named core program, or opt out for that core."""
        ...


@dataclass(frozen=True, kw_only=True)
class ResolvedCoreProgram:
    """A core with planner-owned choices bound into its compilation identity."""

    function: Callable[..., object]
    arguments: Mapping[str, object]
    static_kwargs: Mapping[str, int]
    output_roles: object
    tile_widths: Mapping[str, int]
    specialization_key: Hashable
    input_transfer_plan: tuple[ResolvedValueTransfer, ...]
    """Static program fragment composed into the engine's full lowering key."""

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


def resolve_core_program(
    *,
    program: CoreProgram,
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
    (
        resolved_input_transfer_plan,
        input_transfer_specialization_key,
    ) = _resolve_input_transfer_plan(program=program, plan=input_transfer_plan)
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
        function=program.function,
        arguments=apply_value_transfer_plan(
            arguments=program.arguments,
            plan=resolved_input_transfer_plan,
        ),
        static_kwargs=width_bindings,
        output_roles=program.output_roles,
        tile_widths=resolved_widths,
        input_transfer_plan=resolved_input_transfer_plan,
        specialization_key=(
            "core-program",
            _CORE_PROGRAM_VERSION,
            tuple(compilation_axes),
            input_transfer_specialization_key,
        ),
    )


def _validate_core_program(*, program: CoreProgram) -> None:
    """Validate a complete declaration before planner choices inspect it."""
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


def _resolve_input_transfer_plan(
    *,
    program: CoreProgram,
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


def _validate_target_value_accesses(*, program: CoreProgram) -> None:
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
    *, program: CoreProgram, access: _TargetValueAccess
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
    program: CoreProgram,
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
