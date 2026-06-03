import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import portion
from beartype import beartype

from _lcm.beartype_conf import GRID_CONF
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.base import _fail_if_continuous_grid_distributed
from _lcm.grids.continuous import ContinuousGrid
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import GridInitializationError
from lcm.typing import (
    Float1D,
    FloatND,
    Int1D,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True, init=False)
class PiecewiseGridSegment:
    """A segment of a piecewise grid.

    `n_points` is stored as a `jnp.int32` JAX scalar, converted from the
    Python literal supplied at construction.
    """

    interval: str | portion.Interval
    """The interval for this segment.

    Can be a string like "[1, 4)" or a `portion.Interval`.
    """

    n_points: ScalarInt
    """The number of grid points in this segment (`jnp.int32` JAX scalar)."""

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        interval: str | portion.Interval,
        n_points: int | ScalarInt,
    ) -> None:
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "n_points", jnp.int32(n_points))


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class PiecewiseLinSpacedGrid(ContinuousGrid):
    """A piecewise linearly spaced grid with multiple segments.

    This grid type is useful for representing grids that need specific breakpoints,
    such as eligibility thresholds for programs. Each segment has its own linear
    spacing.

    Example:
    --------
    A grid from 1 to 10 with a breakpoint at 4 (e.g., an eligibility threshold):

        PiecewiseLinSpacedGrid(segments=(
            PiecewiseGridSegment(interval="[1, 4)", n_points=30),
            PiecewiseGridSegment(interval="[4, 10]", n_points=60),
        ))

    Notes:
        - Open boundaries (e.g., `4)` in `[1, 4)`) exclude that exact point from
          the grid. The last point will be slightly before the boundary.
        - Segments must be adjacent: the upper bound of each segment must equal the
          lower bound of the next segment, with compatible open/closed boundaries.

    """

    segments: tuple[PiecewiseGridSegment, ...]
    """Tuple of `PiecewiseGridSegment` objects. Segments must be adjacent."""

    # Cached JAX arrays for efficient coordinate computation (set in __post_init__)
    _breakpoints: Float1D = dataclasses.field(init=False, repr=False)
    _segment_starts: Float1D = dataclasses.field(init=False, repr=False)
    _segment_stops: Float1D = dataclasses.field(init=False, repr=False)
    _segment_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _fail_if_continuous_grid_distributed(
            grid_kind="PiecewiseLinSpacedGrid", distributed=self.distributed
        )
        _validate_piecewise_lin_spaced_grid(self.segments)
        _init_piecewise_grid_cache(self)

    @property
    def n_points(self) -> ScalarInt:
        """Return the total number of points in the grid."""
        return self._segment_n_points.sum(dtype=jnp.int32)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        segment_arrays = [
            jnp.linspace(self._segment_starts[i], self._segment_stops[i], s.n_points)  # ty: ignore[no-matching-overload]
            for i, s in enumerate(self.segments)
        ]
        return jnp.concatenate(segment_arrays)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        segment_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_coordinates.get_linspace_coordinate(
            value=value,
            start=self._segment_starts[segment_idx],
            stop=self._segment_stops[segment_idx],
            n_points=self._segment_n_points[segment_idx],
        )
        return self._cumulative_offsets[segment_idx] + local_coord


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class PiecewiseLogSpacedGrid(ContinuousGrid):
    """A piecewise logarithmically spaced grid with multiple segments.

    This grid type is useful for wealth grids where you want more granularity at
    lower values. Each segment has its own logarithmic spacing.

    Example:
    --------
    A wealth grid with denser points at lower values:

        PiecewiseLogSpacedGrid(segments=(
            PiecewiseGridSegment(interval="[0.1, 10)", n_points=50),
            PiecewiseGridSegment(interval="[10, 1000]", n_points=30),
        ))

    Notes:
        - All boundary values must be positive (required for logarithmic spacing).
        - Open boundaries exclude the exact endpoint using nextafter.
        - Segments must be adjacent: the upper bound of each segment must equal the
          lower bound of the next segment, with compatible open/closed boundaries.

    """

    segments: tuple[PiecewiseGridSegment, ...]
    """Tuple of `PiecewiseGridSegment` objects. All boundaries must be positive."""

    # Cached JAX arrays for efficient coordinate computation (set in __post_init__)
    _breakpoints: Float1D = dataclasses.field(init=False, repr=False)
    _segment_starts: Float1D = dataclasses.field(init=False, repr=False)
    _segment_stops: Float1D = dataclasses.field(init=False, repr=False)
    _segment_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _fail_if_continuous_grid_distributed(
            grid_kind="PiecewiseLogSpacedGrid", distributed=self.distributed
        )
        _validate_piecewise_log_spaced_grid(self.segments)
        _init_piecewise_grid_cache(self)

    @property
    def n_points(self) -> ScalarInt:
        """Return the total number of points in the grid."""
        return self._segment_n_points.sum(dtype=jnp.int32)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        segment_arrays = [
            grid_coordinates.logspace(
                start=self._segment_starts[i],
                stop=self._segment_stops[i],
                n_points=s.n_points,
            )
            for i, s in enumerate(self.segments)
        ]
        return jnp.concatenate(segment_arrays)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        segment_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_coordinates.get_logspace_coordinate(
            value=value,
            start=self._segment_starts[segment_idx],
            stop=self._segment_stops[segment_idx],
            n_points=self._segment_n_points[segment_idx],
        )
        return self._cumulative_offsets[segment_idx] + local_coord


def _parse_interval(interval: str | portion.Interval) -> portion.Interval:
    """Parse an interval from a string or return it if already a portion.Interval."""
    if isinstance(interval, str):
        return portion.from_string(interval, conv=float)
    return interval


def _get_effective_bounds(
    interval: portion.Interval,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Return effective bounds for an interval, adjusting for open boundaries.

    Uses jnp.nextafter with the correct dtype based on JAX's x64 setting to compute
    the next representable floating-point value for open boundaries.
    """
    # Use the dtype that matches JAX's current precision setting
    dtype = jnp.result_type(1.0)

    lower = jnp.array(interval.lower, dtype=dtype)
    upper = jnp.array(interval.upper, dtype=dtype)

    effective_lower = (
        lower if interval.left == portion.CLOSED else jnp.nextafter(lower, jnp.inf)
    )
    effective_upper = (
        upper if interval.right == portion.CLOSED else jnp.nextafter(upper, -jnp.inf)
    )
    return effective_lower, effective_upper


def _init_piecewise_grid_cache(
    grid: PiecewiseLinSpacedGrid | PiecewiseLogSpacedGrid,
) -> None:
    """Initialize cached JAX arrays for efficient coordinate computation.

    Precomputes and stores:
    - _breakpoints: effective starts of segments 1..k-1 for searchsorted
    - _segment_starts: effective start for each segment
    - _segment_stops: effective stop for each segment
    - _segment_n_points: n_points for each segment
    - _cumulative_offsets: cumulative sum of n_points

    The breakpoints use effective starts (accounting for open/closed boundaries)
    to ensure correct segment selection for both [a,x)+[x,b] and [a,x]+(x,b] cases.
    """
    parsed = [_parse_interval(s.interval) for s in grid.segments]
    bounds = [_get_effective_bounds(interval) for interval in parsed]

    starts = jnp.array([b[0] for b in bounds])
    stops = jnp.array([b[1] for b in bounds])

    # Breakpoints are the effective starts of segments 1..k-1
    breakpoints = starts[1:] if len(starts) > 1 else jnp.array([])

    n_points = jnp.array([s.n_points for s in grid.segments], dtype=jnp.int32)
    cumulative = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(n_points[:-1])]
    )

    object.__setattr__(grid, "_breakpoints", breakpoints)
    object.__setattr__(grid, "_segment_starts", starts)
    object.__setattr__(grid, "_segment_stops", stops)
    object.__setattr__(grid, "_segment_n_points", n_points)
    object.__setattr__(grid, "_cumulative_offsets", cumulative)


def _validate_piecewise_lin_spaced_grid(  # noqa: C901, PLR0912
    segments: tuple[PiecewiseGridSegment, ...],
) -> None:
    """Validate the piecewise linearly spaced grid parameters.

    Args:
        segments: The segments defining the grid.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    if not isinstance(segments, tuple):
        error_messages.append(
            "segments must be a tuple of PiecewiseGridSegment objects, but is "
            f"{type(segments).__name__}"
        )
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    if len(segments) < 1:
        error_messages.append("segments must have at least 1 element")
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    # Validate each segment
    parsed_intervals: list[portion.Interval] = []
    for i, segment in enumerate(segments):
        if not isinstance(segment, PiecewiseGridSegment):
            error_messages.append(
                f"segments[{i}] must be a PiecewiseGridSegment object, but is "
                f"{type(segment).__name__}"
            )
            continue

        if segment.n_points < 2:  # noqa: PLR2004
            error_messages.append(
                f"segments[{i}].n_points must be an int >= 2, but is {segment.n_points}"
            )

        # Try to parse the interval
        try:
            interval = _parse_interval(segment.interval)
            parsed_intervals.append(interval)

            # Check interval is valid (lower < upper)
            if interval.lower >= interval.upper:
                error_messages.append(
                    f"segments[{i}].interval must have lower < upper, but got "
                    f"{interval}"
                )
        except (ValueError, TypeError) as e:
            error_messages.append(
                f"segments[{i}].interval is invalid: {segment.interval}. Error: {e}"
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    # Check that segments are adjacent (no gaps or overlaps)
    for i in range(len(parsed_intervals) - 1):
        current = parsed_intervals[i]
        next_interval = parsed_intervals[i + 1]

        if not current.adjacent(next_interval):
            # Provide detailed error message about what's wrong
            if current.upper < next_interval.lower:
                error_messages.append(
                    f"Gap between segments[{i}] and segments[{i + 1}]: "
                    f"{current} and {next_interval} are not adjacent. "
                    f"There is a gap between {current.upper} and {next_interval.lower}."
                )
            elif current.upper > next_interval.lower:
                error_messages.append(
                    f"Overlap between segments[{i}] and segments[{i + 1}]: "
                    f"{current} and {next_interval} overlap."
                )
            else:
                # Same boundary value but incompatible open/closed
                error_messages.append(
                    f"segments[{i}] and segments[{i + 1}] are not adjacent: "
                    f"{current} and {next_interval}. "
                    f"The boundary at {current.upper} must be closed on exactly "
                    f"one side (e.g., '[a, x)' followed by '[x, b]')."
                )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def _validate_piecewise_log_spaced_grid(
    segments: tuple[PiecewiseGridSegment, ...],
) -> None:
    """Validate the piecewise logarithmically spaced grid parameters.

    Runs the standard piecewise validation, then additionally checks that all
    boundary values are positive (required for logarithmic spacing).
    """
    _validate_piecewise_lin_spaced_grid(segments)

    error_messages: list[str] = []
    for i, segment in enumerate(segments):
        interval = _parse_interval(segment.interval)
        if interval.lower <= 0:
            error_messages.append(
                f"segments[{i}].interval lower bound must be positive for logspace, "
                f"but got {interval.lower}"
            )
        if interval.upper <= 0:
            error_messages.append(
                f"segments[{i}].interval upper bound must be positive for logspace, "
                f"but got {interval.upper}"
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)
