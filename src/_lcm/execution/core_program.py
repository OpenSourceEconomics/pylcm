"""Solver-declared core programs for engine-owned lowering.

Solvers declare a core's dynamic arguments, output roles, and static execution
requirements. The engine validates the declaration and binds planner-owned choices
before lowering. Kernels without a program for a named core use their dense execution
path for that unsupported route.
"""

import functools
import inspect
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from _lcm.typing import ActionName

_CORE_PROGRAM_VERSION = 1
_INT32_MAX = 2_147_483_647


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

    def __post_init__(self) -> None:
        """Snapshot the declared axes."""
        object.__setattr__(self, "streamable_axes", tuple(self.streamable_axes))


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
    output_roles: object
    tile_widths: Mapping[str, int]
    specialization_key: Hashable
    """Static program fragment composed into the engine's full lowering key."""

    def __post_init__(self) -> None:
        """Snapshot the materialized argument and planning containers."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(
            self,
            "tile_widths",
            MappingProxyType(dict(self.tile_widths)),
        )


def resolve_core_program(
    *,
    program: CoreProgram,
    tile_widths: Mapping[str, object] | None = None,
) -> ResolvedCoreProgram:
    """Validate and bind planner-owned tile widths into program.

    Widths are static compilation choices: they are partially applied to the
    core rather than appended to its dynamic lowering arguments. A streamable
    axis requires an explicit planner choice; silently using its full extent
    would turn a streaming declaration into full materialization.
    """
    _validate_core_program(program=program)
    requested_widths = {} if tile_widths is None else dict(tile_widths)
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
        function=functools.partial(program.function, **width_bindings),
        arguments=program.arguments,
        output_roles=program.output_roles,
        tile_widths=resolved_widths,
        specialization_key=(
            "core-program",
            _CORE_PROGRAM_VERSION,
            tuple(compilation_axes),
        ),
    )


def _validate_core_program(*, program: CoreProgram) -> None:
    """Validate a complete declaration before planner choices inspect it."""
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
