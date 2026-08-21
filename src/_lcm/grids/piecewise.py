"""Piecewise continuous grids with explicit interior-boundary ownership."""

import dataclasses
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import jax.numpy as jnp
from beartype import beartype

from _lcm.axis_boundaries import BoundaryOwner, effective_segment_bounds
from _lcm.beartype_conf import GRID_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.base import _fail_if_continuous_grid_distributed
from _lcm.grids.continuous import ContinuousGrid
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, FloatND, Int1D, ScalarFloat, ScalarInt

if TYPE_CHECKING:
    PiecewisePointCounts: TypeAlias = tuple[int | ScalarInt, ...]  # noqa: UP040
else:
    # The constructor's validator owns element errors and maps them to the
    # public GridInitializationError. The runtime alias keeps the package claw
    # from sampling an invalid tuple element before that validator runs.
    PiecewisePointCounts = tuple[Any, ...]


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class GridBreakpoint:
    """An interior grid boundary and the segment that owns equality."""

    value: float
    """Interior boundary value."""

    owner: BoundaryOwner = "right"
    """Segment containing the exact value: `"left"` or `"right"`."""


@dataclass(frozen=True, kw_only=True, init=False)
class _PiecewiseGrid(ContinuousGrid):
    """Common storage and coordinate geometry for piecewise continuous grids."""

    start: ScalarFloat
    """Closed lower endpoint at pylcm's canonical floating dtype."""

    stop: ScalarFloat
    """Closed upper endpoint at pylcm's canonical floating dtype."""

    breakpoints: tuple[GridBreakpoint, ...]
    """Strictly increasing interior boundaries and their equality owners."""

    points_per_segment: tuple[ScalarInt, ...]
    """Output-node count contributed by each nominal segment."""

    _breakpoint_values: Float1D = dataclasses.field(init=False, repr=False)
    _segment_selection_thresholds: Float1D = dataclasses.field(init=False, repr=False)
    _segment_starts: Float1D = dataclasses.field(init=False, repr=False)
    _segment_stops: Float1D = dataclasses.field(init=False, repr=False)
    _segment_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        breakpoints: tuple[GridBreakpoint, ...],
        points_per_segment: PiecewisePointCounts,
        batch_size: int = 0,
        distributed: bool = False,
    ) -> None:
        _init_piecewise_grid(
            self,
            start=start,
            stop=stop,
            breakpoints=breakpoints,
            points_per_segment=points_per_segment,
            batch_size=batch_size,
            distributed=distributed,
            requires_positive_bounds=isinstance(self, PiecewiseLogSpacedGrid),
        )

    @property
    def n_points(self) -> ScalarInt:
        """Return the number of output nodes contributed by all segments."""
        return self._segment_n_points.sum(dtype=jnp.int32)

    def _to_jax(self, *, logarithmic: bool) -> Float1D:
        """Construct and concatenate every segment's output nodes."""
        segments = [
            (
                grid_coordinates.logspace(
                    start=self._segment_starts[i],
                    stop=self._segment_stops[i],
                    n_points=self._segment_n_points[i],
                )
                if logarithmic
                else grid_coordinates.linspace(
                    start=self._segment_starts[i],
                    stop=self._segment_stops[i],
                    n_points=self._segment_n_points[i],
                )
            )
            for i in range(len(self.points_per_segment))
        ]
        return jnp.concatenate(segments)

    def _get_coordinate(self, *, value: FloatND, logarithmic: bool) -> FloatND:
        """Return the ownership-aware generalized coordinate of `value`."""
        segment_idx = jnp.searchsorted(
            self._segment_selection_thresholds,
            value,
            side="right",
        )
        coordinate_function = (
            grid_coordinates.get_logspace_coordinate
            if logarithmic
            else grid_coordinates.get_linspace_coordinate
        )
        local_coordinate = coordinate_function(
            value=value,
            start=self._segment_starts[segment_idx],
            stop=self._segment_stops[segment_idx],
            n_points=self._segment_n_points[segment_idx],
        )
        return self._cumulative_offsets[segment_idx] + local_coordinate


class PiecewiseLinSpacedGrid(_PiecewiseGrid):
    """A linearly spaced grid split at explicitly owned breakpoints.

    Each entry of `points_per_segment` is the number of output nodes contributed
    by that nominal segment. Every breakpoint appears exactly once, in the segment
    selected by its `GridBreakpoint.owner`.
    """

    def to_jax(self) -> Float1D:
        """Return all segment nodes in ascending order."""
        return self._to_jax(logarithmic=False)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the ownership-aware generalized coordinate of a value."""
        return self._get_coordinate(value=value, logarithmic=False)


class PiecewiseLogSpacedGrid(_PiecewiseGrid):
    """A logarithmically spaced grid split at explicitly owned breakpoints.

    The complete domain and every breakpoint must be strictly positive. Ownership
    and contributed-node semantics are identical to `PiecewiseLinSpacedGrid`.
    """

    def to_jax(self) -> Float1D:
        """Return all segment nodes in ascending order."""
        return self._to_jax(logarithmic=True)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the ownership-aware generalized coordinate of a value."""
        return self._get_coordinate(value=value, logarithmic=True)


def _init_piecewise_grid(
    grid: _PiecewiseGrid,
    *,
    start: float | ScalarFloat,
    stop: float | ScalarFloat,
    breakpoints: tuple[GridBreakpoint, ...],
    points_per_segment: PiecewisePointCounts,
    batch_size: int,
    distributed: bool,
    requires_positive_bounds: bool,
) -> None:
    """Cast, validate, and cache one breakpoint-first grid declaration."""
    _fail_if_continuous_grid_distributed(
        grid_kind=type(grid).__name__, distributed=distributed
    )
    dtype = canonical_float_dtype()
    start_jax = jnp.asarray(start, dtype=dtype)
    stop_jax = jnp.asarray(stop, dtype=dtype)
    breakpoint_values = jnp.asarray(
        tuple(declaration.value for declaration in breakpoints),
        dtype=dtype,
    )
    owners = tuple(declaration.owner for declaration in breakpoints)
    integer_counts = _integer_point_counts(points_per_segment=points_per_segment)
    segment_n_points = jnp.asarray(integer_counts, dtype=jnp.int32)
    segment_starts, segment_stops = effective_segment_bounds(
        start=start_jax,
        stop=stop_jax,
        breakpoints=breakpoint_values,
        owners=owners,
    )
    _validate_piecewise_grid(
        start=start_jax,
        stop=stop_jax,
        breakpoint_values=breakpoint_values,
        owners=owners,
        segment_n_points=segment_n_points,
        n_declared_point_counts=len(points_per_segment),
        segment_starts=segment_starts,
        segment_stops=segment_stops,
        requires_positive_bounds=requires_positive_bounds,
    )
    cumulative_offsets = jnp.concatenate(
        (
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.cumsum(segment_n_points[:-1], dtype=jnp.int32),
        )
    )

    object.__setattr__(grid, "start", start_jax)
    object.__setattr__(grid, "stop", stop_jax)
    object.__setattr__(grid, "breakpoints", breakpoints)
    object.__setattr__(
        grid,
        "points_per_segment",
        tuple(jnp.int32(count) for count in integer_counts),
    )
    object.__setattr__(grid, "batch_size", batch_size)
    object.__setattr__(grid, "distributed", distributed)
    object.__setattr__(grid, "_breakpoint_values", breakpoint_values)
    object.__setattr__(
        grid,
        "_segment_selection_thresholds",
        segment_starts[1:],
    )
    object.__setattr__(grid, "_segment_starts", segment_starts)
    object.__setattr__(grid, "_segment_stops", segment_stops)
    object.__setattr__(grid, "_segment_n_points", segment_n_points)
    object.__setattr__(grid, "_cumulative_offsets", cumulative_offsets)


def _integer_point_counts(
    *, points_per_segment: PiecewisePointCounts
) -> tuple[int, ...]:
    """Return exact Python integer counts, refusing lossy numeric coercion."""
    counts: list[int] = []
    errors: list[str] = []
    for index, value in enumerate(points_per_segment):
        if isinstance(value, bool):
            errors.append(
                f"points_per_segment[{index}] must be an integer >= 2, but is bool"
            )
            continue
        try:
            count = operator.index(value)
        except TypeError:
            errors.append(
                f"points_per_segment[{index}] must be an integer >= 2, but is "
                f"{type(value).__name__}"
            )
            continue
        counts.append(count)
    if errors:
        raise GridInitializationError(format_messages(errors))
    return tuple(counts)


_MIN_POINTS_PER_SEGMENT = 2


def _validate_piecewise_grid(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    breakpoint_values: Float1D,
    owners: tuple[BoundaryOwner, ...],
    segment_n_points: Int1D,
    n_declared_point_counts: int,
    segment_starts: Float1D,
    segment_stops: Float1D,
    requires_positive_bounds: bool,
) -> None:
    """Validate a canonical breakpoint-first piecewise-grid declaration."""
    errors = _boundary_validation_errors(
        start=start,
        stop=stop,
        breakpoint_values=breakpoint_values,
        requires_positive_bounds=requires_positive_bounds,
    )
    errors.extend(
        _segment_validation_errors(
            breakpoint_values=breakpoint_values,
            owners=owners,
            segment_n_points=segment_n_points,
            n_declared_point_counts=n_declared_point_counts,
            segment_starts=segment_starts,
            segment_stops=segment_stops,
        )
    )
    if errors:
        raise GridInitializationError(format_messages(errors))


def _boundary_validation_errors(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    breakpoint_values: Float1D,
    requires_positive_bounds: bool,
) -> list[str]:
    """Return endpoint and nominal-breakpoint validation messages."""
    errors: list[str] = []
    finite_outer = bool(jnp.isfinite(start) & jnp.isfinite(stop))
    finite_breakpoints = bool(jnp.all(jnp.isfinite(breakpoint_values)))

    if not finite_outer:
        errors.append("start and stop must be finite")
    elif not bool(start < stop):
        errors.append(f"start < stop is required, but got {start} and {stop}")

    if not finite_breakpoints:
        errors.append("all breakpoint values must be finite")
    elif finite_outer and breakpoint_values.size:
        if not bool(jnp.all((start < breakpoint_values) & (breakpoint_values < stop))):
            errors.append(
                "all breakpoint values must lie strictly between start and stop"
            )
        if not bool(jnp.all(jnp.diff(breakpoint_values) > 0)):
            errors.append("breakpoint values must be strictly increasing")

    if requires_positive_bounds:
        all_bounds = jnp.concatenate((start[None], breakpoint_values, stop[None]))
        if not bool(jnp.all(all_bounds > 0)):
            errors.append("all log-grid boundaries must be strictly positive")

    return errors


def _segment_validation_errors(
    *,
    breakpoint_values: Float1D,
    owners: tuple[BoundaryOwner, ...],
    segment_n_points: Int1D,
    n_declared_point_counts: int,
    segment_starts: Float1D,
    segment_stops: Float1D,
) -> list[str]:
    """Return point-count, ownership, and effective-bound validation messages."""
    errors: list[str] = []
    expected_counts = int(breakpoint_values.size) + 1
    if n_declared_point_counts != expected_counts:
        errors.append(
            "points_per_segment must contain one count per segment: "
            f"expected {expected_counts}, got {n_declared_point_counts}"
        )
    if segment_n_points.size and not bool(
        jnp.all(segment_n_points >= _MIN_POINTS_PER_SEGMENT)
    ):
        errors.append("every points_per_segment count must be an integer >= 2")

    if any(owner not in ("left", "right") for owner in owners):
        errors.append("every breakpoint owner must be exactly 'left' or 'right'")

    finite_bounds = bool(
        jnp.all(jnp.isfinite(segment_starts)) & jnp.all(jnp.isfinite(segment_stops))
    )
    if finite_bounds and not bool(jnp.all(segment_starts < segment_stops)):
        errors.append(
            "every segment must retain distinct representable bounds after "
            "applying breakpoint ownership"
        )
    return errors
