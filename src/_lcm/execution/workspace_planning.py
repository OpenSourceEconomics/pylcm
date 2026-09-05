"""Compile-only selection of memory-feasible workspace widths.

The planner owns one narrow seam: callers describe streamable product axes and
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

from _lcm.execution.core_program import StreamableProductAxis
from lcm.exceptions import ExecutionPlanningError

_MISSING = object()


class _MemoryAnalyzable(Protocol):
    """Compiler result exposing JAX-style memory analysis."""

    def memory_analysis(self) -> object:
        """Return compiler workspace statistics."""
        ...


@dataclass(frozen=True, slots=True)
class WorkspacePlan[Compiled]:
    """One selected width mapping and its already-compiled executable."""

    widths: Mapping[str, int]
    peak_bytes: int | None
    compiled: Compiled

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the planner-selected widths."""
        object.__setattr__(self, "widths", MappingProxyType(dict(self.widths)))


def workspace_width_candidates(
    *,
    axes: tuple[StreamableProductAxis, ...],
    budget_bytes: int | None = None,
) -> tuple[Mapping[str, int], ...]:
    """Return the candidate sequence in planner rank order without compiling it.

    Without a budget the sequence holds one candidate: each axis at its full extent
    or its requested width.  With a budget it holds the Cartesian product of the
    per-axis frontiers, widest first: descending width product, ties broken toward
    the lexicographically greatest width tuple in axis declaration order.
    """
    declared_axes = _validate_axes(axes=axes)
    budget = _validate_budget(budget_bytes=budget_bytes)
    return _workspace_width_candidates(
        axes=declared_axes,
        budget_bytes=budget,
    )


def plan_workspace[Compiled](
    *,
    axes: tuple[StreamableProductAxis, ...],
    compile_candidate: Callable[[Mapping[str, int]], Compiled],
    budget_bytes: int | None = None,
    peak_bytes_for: Callable[[Compiled], int] | None = None,
) -> WorkspacePlan[Compiled]:
    """Compile the width frontier widest-first and return the first candidate that fits.

    Without a budget, the full extent (or an axis's requested width) is compiled
    exactly once and compiler memory analysis is deliberately not consulted.  With a
    budget, candidates are compiled and analyzed in rank order — descending width
    product, ties broken toward the lexicographically greatest width tuple in axis
    declaration order — and the first whose compiler-reported peak fits is returned.
    That is the feasible maximum of the whole frontier, reached without compiling any
    candidate narrower than the winner; only a core that fits at no width compiles
    its entire frontier before failing.

    The returned executable is the exact object compiled for the selected candidate;
    the planner neither executes it nor recompiles the winner.
    """
    declared_axes = _validate_axes(axes=axes)
    budget = _validate_budget(budget_bytes=budget_bytes)
    if not callable(compile_candidate):
        msg = "The workspace candidate compiler must be callable."
        raise TypeError(msg)
    if peak_bytes_for is not None and not callable(peak_bytes_for):
        msg = "The workspace peak lookup must be callable or None."
        raise TypeError(msg)

    candidates = _workspace_width_candidates(
        axes=declared_axes,
        budget_bytes=budget,
    )

    if budget is None:
        widths = candidates[0]
        compiled = compile_candidate(widths)
        return WorkspacePlan(widths=widths, peak_bytes=None, compiled=compiled)

    least_peak: int | None = None
    for widths in candidates:
        compiled = compile_candidate(widths)
        peak_bytes = _peak_bytes_for_candidate(
            compiled=compiled, widths=widths, peak_bytes_for=peak_bytes_for
        )
        least_peak = peak_bytes if least_peak is None else min(least_peak, peak_bytes)
        if peak_bytes <= budget:
            return WorkspacePlan(
                widths=widths, peak_bytes=peak_bytes, compiled=compiled
            )

    if declared_axes and all(
        axis.requested_width is not None for axis in declared_axes
    ):
        msg = (
            "The explicitly requested workspace widths require "
            f"{least_peak} peak bytes, exceeding the {budget}-byte budget."
        )
    else:
        msg = (
            "No workspace-width candidate fits the "
            f"{budget}-byte budget; the smallest reported peak is "
            f"{least_peak} bytes."
        )
    raise ExecutionPlanningError(msg)


def _validate_axes(
    *, axes: tuple[StreamableProductAxis, ...]
) -> tuple[StreamableProductAxis, ...]:
    """Validate planner-local width assumptions and preserve declaration order."""
    declared_axes = tuple(axes)
    for axis in declared_axes:
        if not isinstance(axis, StreamableProductAxis):
            msg = "Workspace axes must be StreamableProductAxis instances."
            raise TypeError(msg)
        _validate_axis(axis=axis)

    names = tuple(axis.name for axis in declared_axes)
    if len(names) != len(set(names)):
        msg = f"Workspace axes have duplicate names: {names!r}."
        raise ValueError(msg)
    return declared_axes


def _validate_axis(*, axis: StreamableProductAxis) -> None:
    """Validate planner-local assumptions about one product axis."""
    if not isinstance(axis.name, str) or not axis.name:
        msg = "A workspace axis name must be a non-empty string."
        raise TypeError(msg)
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
    if axis.extent <= 1:
        msg = f"Workspace axis {axis.name!r} must have product extent greater than one."
        raise ValueError(msg)
    if axis.requested_width is not None:
        _validate_width(
            axis_name=axis.name,
            extent=axis.extent,
            width=axis.requested_width,
        )


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


def _validate_width(*, axis_name: str, extent: int, width: object) -> int:
    """Validate an explicit width against one product extent."""
    if type(width) is not int:
        msg = f"Requested width for workspace axis {axis_name!r} must be an integer."
        raise TypeError(msg)
    if width <= 0:
        msg = f"Requested width for workspace axis {axis_name!r} must be positive."
        raise ValueError(msg)
    if width > extent:
        msg = (
            f"Requested width {width} for workspace axis {axis_name!r} exceeds "
            f"its product extent {extent}."
        )
        raise ValueError(msg)
    return width


def _workspace_width_candidates(
    *,
    axes: tuple[StreamableProductAxis, ...],
    budget_bytes: int | None,
) -> tuple[MappingProxyType[str, int], ...]:
    """Enumerate one eager width map or the complete budgeted frontier, widest first."""
    if budget_bytes is None:
        values = tuple(
            axis.extent if axis.requested_width is None else axis.requested_width
            for axis in axes
        )
        return (_width_mapping(axes=axes, values=values),)

    frontiers = tuple(_axis_frontier(axis=axis) for axis in axes)
    candidates = (
        _width_mapping(axes=axes, values=values)
        for values in itertools.product(*frontiers)
    )
    return tuple(sorted(candidates, key=_candidate_rank, reverse=True))


def _candidate_rank(widths: Mapping[str, int]) -> tuple[int, tuple[int, ...]]:
    """Rank a candidate by width product, then by its width tuple in axis order."""
    values = tuple(widths.values())
    return (math.prod(values), values)


def _axis_frontier(*, axis: StreamableProductAxis) -> tuple[int, ...]:
    """Return one requested width, or 1/powers-of-two/full without duplicates."""
    if axis.requested_width is not None:
        return (axis.requested_width,)

    widths = [1]
    power = 2
    while power < axis.extent:
        widths.append(power)
        power *= 2
    if widths[-1] != axis.extent:
        widths.append(axis.extent)
    return tuple(widths)


def _width_mapping(
    *, axes: tuple[StreamableProductAxis, ...], values: tuple[int, ...]
) -> MappingProxyType[str, int]:
    """Bind a width tuple to axis names without losing declaration order."""
    return MappingProxyType(
        {axis.name: width for axis, width in zip(axes, values, strict=True)}
    )


def _peak_bytes_for_candidate[Compiled](
    *,
    compiled: Compiled,
    widths: Mapping[str, int],
    peak_bytes_for: Callable[[Compiled], int] | None,
) -> int:
    """Read one candidate peak directly or through a caller-owned cache."""
    if peak_bytes_for is None:
        return compiler_peak_bytes(compiled=compiled, widths=widths)

    try:
        value = peak_bytes_for(compiled)
    except Exception as exc:
        msg = f"Compiler memory peak lookup failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc

    try:
        return _non_negative_bytes(value=value)
    except Exception as exc:
        msg = (
            "Compiler memory peak lookup returned an invalid byte count for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg) from exc


def compiler_peak_bytes[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> int:
    """Read and strictly normalize one candidate's compiler-reported peak."""
    try:
        analyze = cast("_MemoryAnalyzable", compiled).memory_analysis
    except Exception as exc:
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc
    if not callable(analyze):
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg)
    try:
        analysis = analyze()
    except Exception as exc:
        msg = f"Compiler memory analysis failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc
    try:
        return _peak_from_analysis(analysis=analysis)
    except Exception as exc:
        msg = (
            "Compiler memory analysis returned no valid per-device peak for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg) from exc


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
        msg = "A compiler-reported peak must be an integer byte count, not bool."
        raise TypeError(msg)
    try:
        normalized = operator.index(cast("SupportsIndex", value))
    except TypeError as exc:
        msg = "A compiler-reported peak must be an integer byte count."
        raise TypeError(msg) from exc
    if normalized < 0:
        msg = "A compiler-reported peak must be non-negative."
        raise ValueError(msg)
    return normalized
