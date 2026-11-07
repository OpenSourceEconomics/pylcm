"""The machinery behind the piecewise continuous grids of `lcm.grids`.

The public grids declare interior breakpoints with explicit equality ownership;
this module validates and caches their segments and builds their nodes and
coordinates.
"""

import operator
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from _lcm.axis_boundaries import BoundaryOwner, effective_segment_bounds
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.continuous import ContinuousGrid
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, FloatND, Int1D, ScalarFloat, ScalarInt

if TYPE_CHECKING:
    type PiecewisePointCounts = tuple[int | ScalarInt, ...]
else:
    # The constructor's validator owns element errors and maps them to the
    # public GridInitializationError. The runtime alias keeps the package claw
    # from sampling an invalid tuple element before that validator runs.
    type PiecewisePointCounts = tuple[Any, ...]


def piecewise_nodes(
    *,
    segment_starts: Float1D,
    segment_stops: Float1D,
    segment_n_points: Int1D,
    logarithmic: bool,
) -> Float1D:
    """Construct and concatenate every segment's output nodes."""
    segments = [
        (
            grid_coordinates.logspace(
                start=segment_starts[i],
                stop=segment_stops[i],
                n_points=segment_n_points[i],
            )
            if logarithmic
            else grid_coordinates.linspace(
                start=segment_starts[i],
                stop=segment_stops[i],
                n_points=segment_n_points[i],
            )
        )
        for i in range(segment_n_points.shape[0])
    ]
    return jnp.concatenate(segments)


def piecewise_coordinate(
    *,
    value: FloatND,
    segment_selection_thresholds: Float1D,
    segment_starts: Float1D,
    segment_stops: Float1D,
    segment_n_points: Int1D,
    cumulative_offsets: Int1D,
    logarithmic: bool,
) -> FloatND:
    """Return the ownership-aware generalized coordinate of `value`."""
    segment_idx = jnp.searchsorted(segment_selection_thresholds, value, side="right")
    coordinate_function = (
        grid_coordinates.get_logspace_coordinate
        if logarithmic
        else grid_coordinates.get_linspace_coordinate
    )
    local_coordinate = coordinate_function(
        value=value,
        start=segment_starts[segment_idx],
        stop=segment_stops[segment_idx],
        n_points=segment_n_points[segment_idx],
    )
    return cumulative_offsets[segment_idx] + local_coordinate


def _init_piecewise_grid(
    *,
    grid: ContinuousGrid,
    start: float,
    stop: float,
    breakpoint_values: tuple[float, ...],
    owners: tuple[BoundaryOwner, ...],
    points_per_segment: PiecewisePointCounts,
    requires_positive_bounds: bool,
) -> None:
    """Cast, validate, and cache one breakpoint-first grid declaration.

    The public piecewise grid hands in its breakpoints split into their values and
    owners and stores the declarations itself.
    """
    dtype = canonical_float_dtype()
    start_jax = jnp.asarray(start, dtype=dtype)
    stop_jax = jnp.asarray(stop, dtype=dtype)
    breakpoint_values_jax = jnp.asarray(breakpoint_values, dtype=dtype)
    integer_counts = _integer_point_counts(points_per_segment=points_per_segment)
    segment_n_points = jnp.asarray(integer_counts, dtype=jnp.int32)
    segment_starts, segment_stops = effective_segment_bounds(
        start=start_jax,
        stop=stop_jax,
        breakpoints=breakpoint_values_jax,
        owners=owners,
    )
    _validate_piecewise_grid(
        start=start_jax,
        stop=stop_jax,
        breakpoint_values=breakpoint_values_jax,
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
    object.__setattr__(
        grid,
        "points_per_segment",
        tuple(jnp.int32(count) for count in integer_counts),
    )
    object.__setattr__(grid, "_breakpoint_values", breakpoint_values_jax)
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
