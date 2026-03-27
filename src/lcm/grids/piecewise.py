import dataclasses
from dataclasses import dataclass
from typing import overload

import jax.numpy as jnp
import portion
from jax import Array

from lcm.exceptions import GridInitializationError, format_messages
from lcm.grids import coordinates as grid_coordinates
from lcm.grids.continuous import ContinuousGrid
from lcm.typing import (
    Float1D,
    Int1D,
    ScalarFloat,
)


@dataclass(frozen=True, kw_only=True)
class Piece:
    """A piece of a piecewise linearly spaced grid."""

    interval: str | portion.Interval
    """The interval for this piece.

    Can be a string like "[1, 4)" or a `portion.Interval`.
    """

    n_points: int
    """The number of grid points in this piece."""


@dataclass(frozen=True, kw_only=True)
class PiecewiseLinSpacedGrid(ContinuousGrid):
    """A piecewise linearly spaced grid with multiple segments.

    This grid type is useful for representing grids that need specific breakpoints,
    such as eligibility thresholds for programs. Each piece has its own linear spacing.

    Example:
    --------
    A grid from 1 to 10 with a breakpoint at 4 (e.g., an eligibility threshold):

        PiecewiseLinSpacedGrid(pieces=(
            Piece("[1, 4)", 30),
            Piece("[4, 10]", 60),
        ))

    Notes:
        - Open boundaries (e.g., `4)` in `[1, 4)`) exclude that exact point from
          the grid. The last point will be slightly before the boundary.
        - Pieces must be adjacent: the upper bound of each piece must equal the
          lower bound of the next piece, with compatible open/closed boundaries.

    """

    pieces: tuple[Piece, ...]
    """Tuple of Piece objects defining each segment. Pieces must be adjacent."""

    # Cached JAX arrays for efficient coordinate computation (set in __post_init__)
    _breakpoints: Float1D = dataclasses.field(init=False, repr=False)
    _piece_starts: Float1D = dataclasses.field(init=False, repr=False)
    _piece_stops: Float1D = dataclasses.field(init=False, repr=False)
    _piece_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_piecewise_lin_spaced_grid(self.pieces)
        _init_piecewise_grid_cache(self)

    @property
    def n_points(self) -> int:
        """Return the total number of points in the grid."""
        return sum(p.n_points for p in self.pieces)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        piece_arrays = [
            jnp.linspace(self._piece_starts[i], self._piece_stops[i], p.n_points)
            for i, p in enumerate(self.pieces)
        ]
        return jnp.concatenate(piece_arrays)

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        piece_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_coordinates.get_linspace_coordinate(
            value=value,
            start=self._piece_starts[piece_idx],
            stop=self._piece_stops[piece_idx],
            n_points=self._piece_n_points[piece_idx],
        )
        return self._cumulative_offsets[piece_idx] + local_coord


@dataclass(frozen=True, kw_only=True)
class PiecewiseLogSpacedGrid(ContinuousGrid):
    """A piecewise logarithmically spaced grid with multiple segments.

    This grid type is useful for wealth grids where you want more granularity at
    lower values. Each piece has its own logarithmic spacing.

    Example:
    --------
    A wealth grid with denser points at lower values:

        PiecewiseLogSpacedGrid(pieces=(
            Piece("[0.1, 10)", 50),   # Dense at low wealth
            Piece("[10, 1000]", 30),  # Sparser at high wealth
        ))

    Notes:
        - All boundary values must be positive (required for logarithmic spacing).
        - Open boundaries exclude the exact endpoint using nextafter.
        - Pieces must be adjacent: the upper bound of each piece must equal the
          lower bound of the next piece, with compatible open/closed boundaries.

    """

    pieces: tuple[Piece, ...]
    """Tuple of Piece objects defining each segment. All boundaries must be positive."""

    # Cached JAX arrays for efficient coordinate computation (set in __post_init__)
    _breakpoints: Float1D = dataclasses.field(init=False, repr=False)
    _piece_starts: Float1D = dataclasses.field(init=False, repr=False)
    _piece_stops: Float1D = dataclasses.field(init=False, repr=False)
    _piece_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_piecewise_log_spaced_grid(self.pieces)
        _init_piecewise_grid_cache(self)

    @property
    def n_points(self) -> int:
        """Return the total number of points in the grid."""
        return sum(p.n_points for p in self.pieces)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        piece_arrays = [
            grid_coordinates.logspace(
                start=self._piece_starts[i],
                stop=self._piece_stops[i],
                n_points=p.n_points,
            )
            for i, p in enumerate(self.pieces)
        ]
        return jnp.concatenate(piece_arrays)

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        piece_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_coordinates.get_logspace_coordinate(
            value=value,
            start=self._piece_starts[piece_idx],
            stop=self._piece_stops[piece_idx],
            n_points=self._piece_n_points[piece_idx],
        )
        return self._cumulative_offsets[piece_idx] + local_coord


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
    - _breakpoints: effective starts of pieces 1..k-1 for searchsorted
    - _piece_starts: effective start for each piece
    - _piece_stops: effective stop for each piece
    - _piece_n_points: n_points for each piece
    - _cumulative_offsets: cumulative sum of n_points

    The breakpoints use effective starts (accounting for open/closed boundaries)
    to ensure correct piece selection for both [a,x)+[x,b] and [a,x]+(x,b] cases.
    """
    parsed = [_parse_interval(p.interval) for p in grid.pieces]
    bounds = [_get_effective_bounds(interval) for interval in parsed]

    starts = jnp.array([b[0] for b in bounds])
    stops = jnp.array([b[1] for b in bounds])

    # Breakpoints are the effective starts of pieces 1..k-1
    breakpoints = starts[1:] if len(starts) > 1 else jnp.array([])

    n_points = jnp.array([p.n_points for p in grid.pieces])
    cumulative = jnp.concatenate([jnp.array([0]), jnp.cumsum(n_points[:-1])])

    object.__setattr__(grid, "_breakpoints", breakpoints)
    object.__setattr__(grid, "_piece_starts", starts)
    object.__setattr__(grid, "_piece_stops", stops)
    object.__setattr__(grid, "_piece_n_points", n_points)
    object.__setattr__(grid, "_cumulative_offsets", cumulative)


def _validate_piecewise_lin_spaced_grid(  # noqa: C901, PLR0912
    pieces: tuple[Piece, ...],
) -> None:
    """Validate the piecewise linearly spaced grid parameters.

    Args:
        pieces: The pieces defining the grid segments.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    if not isinstance(pieces, tuple):
        error_messages.append(
            f"pieces must be a tuple of Piece objects, but is {type(pieces).__name__}"
        )
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    if len(pieces) < 1:
        error_messages.append("pieces must have at least 1 element")
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    # Validate each piece
    parsed_intervals: list[portion.Interval] = []
    for i, piece in enumerate(pieces):
        if not isinstance(piece, Piece):
            error_messages.append(
                f"pieces[{i}] must be a Piece object, but is {type(piece).__name__}"
            )
            continue

        if not isinstance(piece.n_points, int) or piece.n_points < 2:  # noqa: PLR2004
            error_messages.append(
                f"pieces[{i}].n_points must be an int >= 2, but is {piece.n_points}"
            )

        # Try to parse the interval
        try:
            interval = _parse_interval(piece.interval)
            parsed_intervals.append(interval)

            # Check interval is valid (lower < upper)
            if interval.lower >= interval.upper:
                error_messages.append(
                    f"pieces[{i}].interval must have lower < upper, but got {interval}"
                )
        except (ValueError, TypeError) as e:
            error_messages.append(
                f"pieces[{i}].interval is invalid: {piece.interval}. Error: {e}"
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)

    # Check that pieces are adjacent (no gaps or overlaps)
    for i in range(len(parsed_intervals) - 1):
        current = parsed_intervals[i]
        next_interval = parsed_intervals[i + 1]

        if not current.adjacent(next_interval):
            # Provide detailed error message about what's wrong
            if current.upper < next_interval.lower:
                error_messages.append(
                    f"Gap between pieces[{i}] and pieces[{i + 1}]: "
                    f"{current} and {next_interval} are not adjacent. "
                    f"There is a gap between {current.upper} and {next_interval.lower}."
                )
            elif current.upper > next_interval.lower:
                error_messages.append(
                    f"Overlap between pieces[{i}] and pieces[{i + 1}]: "
                    f"{current} and {next_interval} overlap."
                )
            else:
                # Same boundary value but incompatible open/closed
                error_messages.append(
                    f"pieces[{i}] and pieces[{i + 1}] are not adjacent: "
                    f"{current} and {next_interval}. "
                    f"The boundary at {current.upper} must be closed on exactly "
                    f"one side (e.g., '[a, x)' followed by '[x, b]')."
                )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def _validate_piecewise_log_spaced_grid(pieces: tuple[Piece, ...]) -> None:
    """Validate the piecewise logarithmically spaced grid parameters.

    Runs the standard piecewise validation, then additionally checks that all
    boundary values are positive (required for logarithmic spacing).
    """
    _validate_piecewise_lin_spaced_grid(pieces)

    error_messages: list[str] = []
    for i, piece in enumerate(pieces):
        interval = _parse_interval(piece.interval)
        if interval.lower <= 0:
            error_messages.append(
                f"pieces[{i}].interval lower bound must be positive for logspace, "
                f"but got {interval.lower}"
            )
        if interval.upper <= 0:
            error_messages.append(
                f"pieces[{i}].interval upper bound must be positive for logspace, "
                f"but got {interval.upper}"
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)
