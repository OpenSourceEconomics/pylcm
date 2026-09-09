"""Compile-only selection of memory-feasible workspace widths.

The planner owns one narrow seam: callers describe reduced and tiled axes and
provide a compiler for a concrete width mapping.  This module enumerates the static
frontier in rank order, inspects compiler memory reports without executing a
candidate, and returns the first feasible candidate — the already-compiled winner —
for dispatch.
"""

import itertools
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, SupportsIndex, cast

from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from lcm.exceptions import ExecutionPlanningError

_MISSING = object()

# Largest width an unbudgeted streamed axis is lowered at.
BOOTSTRAP_WIDTH_CAP = 64


class _MemoryAnalyzable(Protocol):
    """Compiler result exposing JAX-style memory analysis."""

    def memory_analysis(self) -> object:
        """Return compiler workspace statistics."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilerMemoryRecord:
    """One device's raw peak and represented default-memory allocations."""

    peak_bytes: int
    """Unmodified compiler-reported peak for this device record."""
    argument_bytes: int
    """Storage assigned to the executable's input arguments."""
    output_bytes: int
    """Storage assigned to executable outputs, before subtracting aliases."""
    alias_bytes: int
    """Argument/output overlap to subtract once from represented storage."""
    temporary_bytes: int
    """Compiler-reported preallocated temporary storage."""

    def __post_init__(self) -> None:
        """Require complete counters and an argument/output overlap that can exist."""
        for value in (
            self.peak_bytes,
            self.argument_bytes,
            self.output_bytes,
            self.alias_bytes,
            self.temporary_bytes,
        ):
            _non_negative_bytes(value=value)
        if self.alias_bytes > min(self.argument_bytes, self.output_bytes):
            raise ValueError(
                "Compiler aliases exceed represented arguments or outputs."
            )

    @property
    def allocation_bytes(self) -> int:
        """Count represented storage, removing argument/output overlap once."""
        return (
            self.argument_bytes
            + self.output_bytes
            - self.alias_bytes
            + self.temporary_bytes
        )

    @property
    def reservation_bytes(self) -> int:
        """Enforce both reported requirements within the represented storage scope."""
        return max(self.peak_bytes, self.allocation_bytes)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilerMemoryReservation:
    """Complete per-device records, with raw peak kept separate from reservation.

    Allocation counters represent arguments, outputs, their aliases, and
    preallocated temporaries. This scope excludes generated code, allocator
    overhead, thread stacks, and other runtime storage omitted by the report.
    """

    records: tuple[CompilerMemoryRecord, ...]
    """Nonempty device records whose allocation fields remain paired."""

    def __post_init__(self) -> None:
        """Refuse an empty device report."""
        if not self.records:
            raise ValueError("Compiler memory reservation needs a device record.")

    @property
    def peak_bytes(self) -> int:
        """Return the largest raw compiler peak across devices."""
        return max(record.peak_bytes for record in self.records)

    @property
    def reservation_bytes(self) -> int:
        """Return the largest reservation after accounting within each device."""
        return max(record.reservation_bytes for record in self.records)


@dataclass(frozen=True, slots=True)
class WorkspacePlan[Compiled]:
    """One selected width mapping and its already-compiled executable."""

    widths: Mapping[str, int]
    peak_bytes: int | None
    compiled: Compiled
    reservation_bytes: int | None = None
    """Selected represented compiler requirement, or None without a budget."""

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the planner-selected widths."""
        object.__setattr__(self, "widths", MappingProxyType(dict(self.widths)))


def workspace_width_candidates(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    budget_bytes: int | None = None,
) -> tuple[Mapping[str, int], ...]:
    """Return the candidate sequence in planner rank order without compiling it.

    Without a budget the sequence holds one candidate: each axis at its fixed width
    when `fixed_widths` names it, else at its bootstrap width (see
    `bootstrap_width`).  With a budget it holds the Cartesian product of the
    per-axis frontiers, widest first: descending width product, ties broken toward
    the lexicographically greatest width tuple in axis declaration order.  A fixed
    axis contributes one width.  Every width an axis contributes satisfies the
    width policy it declares: it is the full extent, or a multiple of its
    `alignment` at or above its `minimum_width`.  Names no axis declares are ignored
    here; a name no program of the solve declares is refused before planning starts.
    """
    declared_axes = _validate_axes(axes=axes)
    widths = _validate_fixed_widths(fixed_widths=fixed_widths)
    budget = _validate_budget(budget_bytes=budget_bytes)
    return _workspace_width_candidates(
        axes=declared_axes,
        fixed_widths=widths,
        budget_bytes=budget,
    )


def plan_workspace[Compiled](
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    compile_candidate: Callable[[Mapping[str, int]], Compiled],
    budget_bytes: int | None = None,
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None = None,
    resident_bytes: int = 0,
    resident_bytes_for: Callable[[Compiled], int] | None = None,
) -> WorkspacePlan[Compiled]:
    """Compile the width frontier widest-first and return the first candidate that fits.

    Without a budget, the bootstrap width (or a fixed axis width) is compiled
    exactly once and compiler memory analysis is deliberately not consulted, so
    `resident_bytes` is not consulted either.  With a budget, candidates are
    compiled and analyzed in rank order — descending width product, ties broken
    toward the lexicographically greatest width tuple in axis declaration order —
    and the first whose compiler reservation fits is returned.
    A candidate is feasible when its represented allocation reservation plus the
    bytes the plan keeps resident at the node's scheduled position fits the budget.
    The reservation enforces both raw peak and represented allocation storage;
    storage omitted from those reports remains outside this accounting scope.
    That is the feasible maximum of the whole frontier, reached without compiling any
    candidate narrower than the winner; only a core that fits at no width compiles
    its entire frontier before failing.  A position whose resident bytes already
    reach the budget is refused before any candidate is compiled, since no width
    could serve it.

    ``resident_bytes_for`` may refine that lower bound for each executable, for
    example when compilation removes an argument whose owner stays live. It
    returns total external residency, never a replacement compiler peak. This
    lookup belongs to the current invocation, not to an executable cache.

    The returned executable is the exact object compiled for the selected candidate;
    the planner neither executes it nor recompiles the winner.
    """
    declared_axes = _validate_axes(axes=axes)
    widths_by_axis = _validate_fixed_widths(fixed_widths=fixed_widths)
    budget = _validate_budget(budget_bytes=budget_bytes)
    resident = _validate_resident_bytes(resident_bytes=resident_bytes)
    if not callable(compile_candidate):
        msg = "The workspace candidate compiler must be callable."
        raise TypeError(msg)
    if memory_for is not None and not callable(memory_for):
        msg = "The workspace memory lookup must be callable or None."
        raise TypeError(msg)
    if resident_bytes_for is not None and not callable(resident_bytes_for):
        raise TypeError("The workspace residency lookup must be callable or None.")

    candidates = _workspace_width_candidates(
        axes=declared_axes,
        fixed_widths=widths_by_axis,
        budget_bytes=budget,
    )

    if budget is None:
        widths = candidates[0]
        compiled = compile_candidate(widths)
        return WorkspacePlan(widths=widths, peak_bytes=None, compiled=compiled)

    if resident >= budget:
        msg = (
            f"The plan keeps {resident} bytes resident at the node's position, "
            f"leaving nothing of the {budget}-byte budget for a workspace."
        )
        raise ExecutionPlanningError(msg)

    least_total: int | None = None
    least_peak: int | None = None
    least_reservation: int | None = None
    least_resident = resident
    for widths in candidates:
        compiled = compile_candidate(widths)
        memory = _memory_for_candidate(
            compiled=compiled, widths=widths, memory_for=memory_for
        )
        candidate_resident = _resident_bytes_for_candidate(
            compiled=compiled,
            widths=widths,
            lower_bound=resident,
            resident_bytes_for=resident_bytes_for,
        )
        total = memory.reservation_bytes + candidate_resident
        if least_total is None or total < least_total:
            least_total = total
            least_peak = memory.peak_bytes
            least_reservation = memory.reservation_bytes
            least_resident = candidate_resident
        if total <= budget:
            return WorkspacePlan(
                widths=widths,
                peak_bytes=memory.peak_bytes,
                reservation_bytes=memory.reservation_bytes,
                compiled=compiled,
            )

    if declared_axes and all(axis.name in widths_by_axis for axis in declared_axes):
        msg = (
            "The explicitly requested workspace widths require "
            f"{least_reservation} reservation bytes (raw compiler peak {least_peak}), "
            f"exceeding the {budget}-byte budget "
            f"with {least_resident} resident bytes at the node's position."
        )
    else:
        msg = (
            "No workspace-width candidate fits the "
            f"{budget}-byte budget; the smallest total is {least_total} bytes "
            f"({least_reservation} compiler reservation plus {least_resident} resident "
            f"bytes; raw compiler peak {least_peak}, "
            "at the node's position)."
        )
    raise ExecutionPlanningError(msg)


def _validate_axes(
    *, axes: tuple[ReducedAxis | TiledOutputAxis, ...]
) -> tuple[ReducedAxis | TiledOutputAxis, ...]:
    """Validate planner-local width assumptions and preserve declaration order."""
    declared_axes = tuple(axes)
    for axis in declared_axes:
        if not isinstance(axis, (ReducedAxis, TiledOutputAxis)):
            msg = "Workspace axes must be reduced or tiled axis instances."
            raise TypeError(msg)
        _validate_axis(axis=axis)

    names = tuple(axis.name for axis in declared_axes)
    if len(names) != len(set(names)):
        msg = f"Workspace axes have duplicate names: {names!r}."
        raise ValueError(msg)
    return declared_axes


def _validate_axis(*, axis: ReducedAxis | TiledOutputAxis) -> None:
    """Validate planner-local assumptions about one axis."""
    if not isinstance(axis.name, str) or not axis.name:
        msg = "A workspace axis name must be a non-empty string."
        raise TypeError(msg)
    if isinstance(axis, ReducedAxis):
        _validate_coordinates(axis=axis)
    if axis.extent <= 1:
        msg = f"Workspace axis {axis.name!r} must have product extent greater than one."
        raise ValueError(msg)


def _validate_coordinates(*, axis: ReducedAxis) -> None:
    """Validate the coordinate product one reduced axis enumerates."""
    if len(axis.coordinate_names) != len(axis.coordinate_extents):
        msg = (
            f"Workspace axis {axis.name!r} coordinate names and extents must "
            "have the same length."
        )
        raise ValueError(msg)
    if not axis.coordinate_extents:
        msg = f"Workspace axis {axis.name!r} must declare coordinate extents."
        raise ValueError(msg)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int)
        for extent in axis.coordinate_extents
    ):
        msg = f"Workspace axis {axis.name!r} extents must be integers."
        raise TypeError(msg)
    if any(extent <= 0 for extent in axis.coordinate_extents):
        msg = f"Workspace axis {axis.name!r} extents must be positive."
        raise ValueError(msg)


def _validate_fixed_widths(*, fixed_widths: Mapping[str, int]) -> Mapping[str, int]:
    """Require exact positive widths keyed by non-empty axis names."""
    widths = dict(fixed_widths)
    for name, width in widths.items():
        if type(name) is not str or not name:
            msg = "A fixed workspace width must be keyed by a non-empty axis name."
            raise TypeError(msg)
        if type(width) is not int:
            msg = f"Fixed width for workspace axis {name!r} must be an integer."
            raise TypeError(msg)
        if width <= 0:
            msg = f"Fixed width for workspace axis {name!r} must be positive."
            raise ValueError(msg)
    return MappingProxyType(widths)


def _validate_budget(*, budget_bytes: int | None) -> int | None:
    """Require a positive exact-integer byte budget when one is supplied."""
    if budget_bytes is None:
        return None
    if type(budget_bytes) is not int:
        msg = "The workspace budget must be an integer number of bytes or None."
        raise TypeError(msg)
    if budget_bytes <= 0:
        msg = "The workspace budget must be positive."
        raise ValueError(msg)
    return budget_bytes


def _validate_resident_bytes(*, resident_bytes: int) -> int:
    """Require an exact non-negative count of bytes resident at the node."""
    if type(resident_bytes) is not int:
        msg = "The resident byte count must be an exact int."
        raise TypeError(msg)
    if resident_bytes < 0:
        msg = "The resident byte count cannot be negative."
        raise ValueError(msg)
    return resident_bytes


def bootstrap_width(*, extent: int) -> int:
    """Return the width an axis streams at when no device-memory budget is declared.

    The width is the largest power of two strictly below the extent, capped at
    `BOOTSTRAP_WIDTH_CAP`, so an unbudgeted solve never lowers a whole action
    product and its working set stays bounded on every backend.  The full extent is
    reached only through a budget that shows it fits or through a fixed width.
    """
    if type(extent) is not int or extent <= 1:
        msg = f"An execution axis needs an exact int extent above one, got {extent!r}."
        raise ValueError(msg)
    upper_bound = min(BOOTSTRAP_WIDTH_CAP, extent - 1)
    return 1 << (upper_bound.bit_length() - 1)


def _workspace_width_candidates(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int],
    budget_bytes: int | None,
) -> tuple[MappingProxyType[str, int], ...]:
    """Enumerate one bootstrap width map or the budgeted frontier, widest first."""
    if budget_bytes is None:
        values = tuple(
            _fixed_width(axis=axis, fixed_widths=fixed_widths)
            if axis.name in fixed_widths
            else _admissible_width(axis=axis, width=bootstrap_width(extent=axis.extent))
            for axis in axes
        )
        return (_width_mapping(axes=axes, values=values),)

    frontiers = tuple(
        _axis_frontier(axis=axis, fixed_widths=fixed_widths) for axis in axes
    )
    candidates = (
        _width_mapping(axes=axes, values=values)
        for values in itertools.product(*frontiers)
    )
    return tuple(sorted(candidates, key=_candidate_rank, reverse=True))


def _candidate_rank(widths: Mapping[str, int]) -> tuple[int, tuple[int, ...]]:
    """Rank a candidate by width product, then by its width tuple in axis order."""
    values = tuple(widths.values())
    return (math.prod(values), values)


def _axis_frontier(
    *, axis: ReducedAxis | TiledOutputAxis, fixed_widths: Mapping[str, int]
) -> tuple[int, ...]:
    """Return one fixed width, or the admissible 1/powers-of-two/full ladder."""
    if axis.name in fixed_widths:
        return (_fixed_width(axis=axis, fixed_widths=fixed_widths),)

    widths = [1]
    power = 2
    while power < axis.extent:
        widths.append(power)
        power *= 2
    widths.append(axis.extent)
    admissible = {_admissible_width(axis=axis, width=width) for width in widths}
    return tuple(sorted(admissible))


def _fixed_width(
    *, axis: ReducedAxis | TiledOutputAxis, fixed_widths: Mapping[str, int]
) -> int:
    """Return the fixed width of one axis under the width policy it declares."""
    return _admissible_width(axis=axis, width=min(fixed_widths[axis.name], axis.extent))


def _admissible_width(*, axis: ReducedAxis | TiledOutputAxis, width: int) -> int:
    """Return the width the axis admits nearest the proposal, preferring the shorter.

    An axis admits its full extent, whatever the alignment divides, plus every
    multiple of its alignment lying between its floor and that extent.  A proposal
    is rounded down onto that set; one that falls through it — below the floor, or
    below the alignment and so at zero — is lifted to the smallest width the set
    holds, which is the extent when no multiple of the alignment reaches the floor
    without passing the extent.
    """
    if width >= axis.extent:
        return axis.extent
    aligned = width - width % axis.alignment
    if aligned >= axis.minimum_width:
        return aligned
    return _smallest_admissible_width(axis=axis)


def _smallest_admissible_width(*, axis: ReducedAxis | TiledOutputAxis) -> int:
    """Return the narrowest width the axis admits: an aligned floor, else the extent."""
    lifted = -(-axis.minimum_width // axis.alignment) * axis.alignment
    return min(lifted, axis.extent)


def _width_mapping(
    *, axes: tuple[ReducedAxis | TiledOutputAxis, ...], values: tuple[int, ...]
) -> MappingProxyType[str, int]:
    """Bind a width tuple to axis names without losing declaration order."""
    return MappingProxyType(
        {axis.name: width for axis, width in zip(axes, values, strict=True)}
    )


def _memory_for_candidate[Compiled](
    *,
    compiled: Compiled,
    widths: Mapping[str, int],
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None,
) -> CompilerMemoryReservation:
    """Read complete candidate accounting directly or from its compiler cache."""
    if memory_for is None:
        return compiler_memory_reservation(compiled=compiled, widths=widths)

    try:
        value = memory_for(compiled)
    except Exception as exc:
        msg = f"Compiler memory reservation lookup failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc

    if not isinstance(value, CompilerMemoryReservation):
        msg = (
            "Compiler memory lookup returned no complete reservation for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg)
    return value


def _resident_bytes_for_candidate[Compiled](
    *,
    compiled: Compiled,
    widths: Mapping[str, int],
    lower_bound: int,
    resident_bytes_for: Callable[[Compiled], int] | None,
) -> int:
    """Validate a candidate-specific inventory without changing its raw peak."""
    if resident_bytes_for is None:
        return lower_bound
    try:
        resident = _validate_resident_bytes(resident_bytes=resident_bytes_for(compiled))
    except Exception as error:
        raise ExecutionPlanningError(
            f"Workspace residency lookup failed for widths {dict(widths)!r}."
        ) from error
    if resident < lower_bound:
        raise ExecutionPlanningError(
            "Workspace residency lookup returned fewer bytes than the declared "
            f"lower bound for widths {dict(widths)!r}."
        )
    return resident


def compiler_memory_reservation[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> CompilerMemoryReservation:
    """Reserve complete reported default-memory allocations and the raw peak.

    Accept one scalar record or a nonempty sequence/mapping of device records.
    Every record must supply nonnegative integral peak, argument, output, alias,
    and temporary counters, including the four host allocation counters. Nonzero
    host allocations are refused: the peak report does not reliably separate
    their memory space. Separately compiled CPU assembly remains a CPU profile.
    Generated-code metadata does not establish host allocation residency.

    The reservation enforces represented requirements. It does not establish a
    complete runtime upper bound for storage omitted by the compiler report.
    """
    analysis = _compiler_memory_analysis(compiled=compiled, widths=widths)
    try:
        records = _allocation_records(analysis=analysis)
        return CompilerMemoryReservation(
            records=tuple(_allocation_record(record=record) for record in records)
        )
    except Exception as exc:
        raise ExecutionPlanningError(
            "Compiler memory analysis returned no valid per-device reservation "
            f"for widths {dict(widths)!r}: {exc}"
        ) from exc


def compiler_peak_bytes[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> int:
    """Read and strictly normalize one candidate's compiler-reported peak."""
    analysis = _compiler_memory_analysis(compiled=compiled, widths=widths)
    try:
        return _peak_from_analysis(analysis=analysis)
    except Exception as exc:
        msg = (
            "Compiler memory analysis returned no valid per-device peak for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg) from exc


def _compiler_memory_analysis[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> object:
    """Read an executable report once without numerical dispatch."""
    try:
        analyze = cast("_MemoryAnalyzable", compiled).memory_analysis
    except Exception as exc:
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc
    if not callable(analyze):
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg)
    try:
        return analyze()
    except Exception as exc:
        msg = f"Compiler memory analysis failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc


def _allocation_records(*, analysis: object) -> tuple[object, ...]:
    """Keep fields paired within their record before reducing across devices."""
    if _peak_field(record=analysis) is not _MISSING:
        return (analysis,)
    if isinstance(analysis, Mapping):
        records = tuple(analysis.values())
    elif isinstance(analysis, (list, tuple)):
        records = tuple(analysis)
    else:
        raise TypeError("Memory analysis must expose a complete allocation record.")
    if not records:
        raise ValueError("Per-device memory analysis must not be empty.")
    return records


def _allocation_record(*, record: object) -> CompilerMemoryRecord:
    """Validate a scalar device record and its separately reported host space."""
    names = (
        "peak_memory_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    fields = {
        name: record.get(name, _MISSING)
        if isinstance(record, Mapping)
        else getattr(record, name, _MISSING)
        for name in names
    }
    try:
        values = {
            name: _non_negative_bytes(value=value) for name, value in fields.items()
        }
        _fail_if_host_allocations(values=values)
        return CompilerMemoryRecord(
            peak_bytes=values["peak_memory_in_bytes"],
            argument_bytes=values["argument_size_in_bytes"],
            output_bytes=values["output_size_in_bytes"],
            alias_bytes=values["alias_size_in_bytes"],
            temporary_bytes=values["temp_size_in_bytes"],
        )
    except Exception as exc:
        snapshot = {
            name: "unavailable" if value is _MISSING else value
            for name, value in fields.items()
        }
        raise ValueError(f"{exc} Reported allocation fields: {snapshot!r}") from exc


def _fail_if_host_allocations(*, values: Mapping[str, int]) -> None:
    """Refuse a mixed-space peak whose default-memory share is unavailable."""
    if any(value for name, value in values.items() if name.startswith("host_")):
        raise ValueError("Mixed host/default allocation spaces are unsupported.")


def _peak_from_analysis(*, analysis: object) -> int:
    """Normalize one JAX-style record or nonempty per-device record collection."""
    peak = _peak_field(record=analysis)
    if peak is not _MISSING:
        return _normalize_peak_field(value=peak)

    if isinstance(analysis, Mapping):
        records = tuple(analysis.values())
    elif isinstance(analysis, (list, tuple)):
        records = tuple(analysis)
    else:
        msg = "Memory analysis must expose peak_memory_in_bytes."
        raise TypeError(msg)
    if not records:
        msg = "Per-device memory analysis must not be empty."
        raise ValueError(msg)
    return max(_peak_from_device_record(record=record) for record in records)


def _peak_from_device_record(*, record: object) -> int:
    """Read the required peak field from one per-device analysis record."""
    peak = _peak_field(record=record)
    if peak is _MISSING:
        msg = "Each per-device memory record must expose peak_memory_in_bytes."
        raise TypeError(msg)
    return _normalize_peak_field(value=peak)


def _peak_field(*, record: object) -> object:
    """Read a peak field from an attribute record or a string-keyed mapping."""
    if isinstance(record, Mapping):
        return record.get("peak_memory_in_bytes", _MISSING)
    return getattr(record, "peak_memory_in_bytes", _MISSING)


def _normalize_peak_field(*, value: object) -> int:
    """Normalize one integral peak or nonempty collection of per-device peaks."""
    if isinstance(value, Mapping):
        peaks = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        peaks = tuple(value)
    else:
        return _non_negative_bytes(value=value)
    if not peaks:
        msg = "A per-device peak collection must not be empty."
        raise ValueError(msg)
    return max(_non_negative_bytes(value=peak) for peak in peaks)


def _non_negative_bytes(*, value: object) -> int:
    """Accept integer-like byte counts while rejecting booleans and lossy casts."""
    if isinstance(value, bool):
        msg = "A compiler memory counter must be an integer byte count, not bool."
        raise TypeError(msg)
    try:
        normalized = operator.index(cast("SupportsIndex", value))
    except TypeError as exc:
        msg = "A compiler memory counter must be an integer byte count."
        raise TypeError(msg) from exc
    if normalized < 0:
        msg = "A compiler memory counter must be non-negative."
        raise ValueError(msg)
    return normalized
