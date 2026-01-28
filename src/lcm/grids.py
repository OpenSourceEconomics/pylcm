import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, is_dataclass
from typing import Literal

import jax.numpy as jnp
import portion

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError, format_messages
from lcm.shocks import Shock
from lcm.typing import Float1D, Int1D, MappingProxyType, ParamsDict, ScalarFloat
from lcm.utils import find_duplicates, get_field_names_and_values


def categorical[T](cls: type[T]) -> type[T]:
    """Decorator to create a categorical class with auto-assigned integer values.

    Transforms a class with int annotations into a frozen dataclass where each
    field is assigned a consecutive integer value starting from 0.

    Example:
        @categorical
        class LaborSupply:
            work: int
            retire: int

        # Equivalent to:
        @dataclass(frozen=True)
        class LaborSupply:
            work: int = 0
            retire: int = 1

        # Usage:
        LaborSupply.work   # 0
        LaborSupply.retire # 1

    Args:
        cls: The class to decorate.

    Returns:
        A frozen dataclass with auto-assigned integer values.

    """
    annotations = getattr(cls, "__annotations__", {})

    # Assign sequential integers as defaults
    for i, name in enumerate(annotations):
        setattr(cls, name, i)

    # Apply dataclass decorator
    return dataclass(frozen=True)(cls)


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""


class ContinuousGrid(Grid):
    """Base class for grids representing continuous values with coordinate lookup.

    All subclasses must implement `get_coordinate` for value-to-coordinate mapping
    used in interpolation.

    """

    @abstractmethod
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""


class DiscreteGrid(Grid):
    """A class representing a discrete grid.

    Args:
        category_class (type): The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.

    Attributes:
        categories: The list of category names.
        codes: The list of category codes.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """

    def __init__(self, category_class: type) -> None:
        """Initialize the DiscreteGrid.

        Args:
            category_class (type): The category class representing the grid categories.
                Must be a dataclass with fields that have unique int values.

        """
        _validate_discrete_grid(category_class)

        names_and_values = get_field_names_and_values(category_class)

        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the list of category names."""
        return self.__categories

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the list of category codes."""
        return self.__codes

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)


@dataclass(frozen=True, kw_only=True)
class UniformContinuousGrid(ContinuousGrid, ABC):
    """Grid with start/stop/n_points for linearly or logarithmically spaced values."""

    start: int | float
    stop: int | float
    n_points: int

    def __post_init__(self) -> None:
        _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )

    @abstractmethod
    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""

    @abstractmethod
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""

    def replace(self, **kwargs: float) -> UniformContinuousGrid:
        """Replace the attributes of the grid.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the grid.

        Returns:
            A new grid with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise GridInitializationError(
                f"Failed to replace attributes of the grid. The error was: {e}"
            ) from e


class LinSpacedGrid(UniformContinuousGrid):
    """A linearly spaced grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 50.5, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_helpers.linspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        return grid_helpers.get_linspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


class LogSpacedGrid(UniformContinuousGrid):
    """A logarithmically spaced grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_helpers.logspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        return grid_helpers.get_logspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


@dataclass(frozen=True, kw_only=True)
class IrregSpacedGrid(ContinuousGrid):
    """A grid of continuous values at irregular (user-specified) points.

    This grid type is useful for representing non-uniformly spaced points such as
    Gauss-Hermite quadrature nodes.

    Example:
    --------
    Gauss-Hermite quadrature nodes: `IrregSpacedGrid(points=[-1.73, -0.58, 0.58, 1.73])`

    Attributes:
        points: The grid points. Must be a sequence of floats in ascending order.
            Can be any sequence that is convertible to a JAX array.

    """

    points: Sequence[float] | Float1D

    def __post_init__(self) -> None:
        _validate_irreg_spaced_grid(self.points)

    @property
    def n_points(self) -> int:
        """Return the number of points in the grid."""
        return len(self.points)

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return jnp.asarray(self.points)

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        return grid_helpers.get_irreg_coordinate(value, self.to_jax())


@dataclass(frozen=True, kw_only=True)
class ShockGrid(ContinuousGrid):
    """An empty grid for discretized continuous shocks.

    The actual values will be calculated once the prameters for the shock are
    available during the solution or simulation.

    Attributes:
        distribution_type: Type of the shock.
        n_points: The number of points for the discretization of the shock.
        shock_params: Fixed parameters that are needed for the discretization function
            of the specified shock type.
    """

    distribution_type: Literal["uniform", "normal", "tauchen", "rouwenhorst"]
    n_points: int
    shock_params: ParamsDict = field(default_factory=lambda: MappingProxyType({}))

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        shock = Shock(
            n_points=self.n_points,
            distribution_type=self.distribution_type,
            shock_params=self.shock_params,
        )
        return shock.get_gridpoints()

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        return grid_helpers.get_irreg_coordinate(value, self.to_jax())

    def init_params(self, params: ParamsDict) -> ShockGrid:
        """Augment the grid with fixed params from model initialization."""
        return dataclasses.replace(self, shock_params=params)


@dataclass(frozen=True, kw_only=True)
class Piece:
    """A piece of a piecewise linearly spaced grid.

    Attributes:
        interval: The interval for this piece. Can be a string like "[1, 4)" or
            a portion.Interval object.
        n_points: The number of grid points in this piece.

    """

    interval: str | portion.Interval
    n_points: int


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

    Attributes:
        pieces: A tuple of Piece objects defining each segment. Pieces must be
            adjacent (no gaps or overlaps).

    Notes:
        - Open boundaries (e.g., `4)` in `[1, 4)`) exclude that exact point from
          the grid. The last point will be slightly before the boundary.
        - Pieces must be adjacent: the upper bound of each piece must equal the
          lower bound of the next piece, with compatible open/closed boundaries.

    """

    pieces: tuple[Piece, ...]

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

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        piece_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_helpers.get_linspace_coordinate(
            value,
            self._piece_starts[piece_idx],
            self._piece_stops[piece_idx],
            self._piece_n_points[piece_idx],
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

    Attributes:
        pieces: A tuple of Piece objects defining each segment. Pieces must be
            adjacent (no gaps or overlaps). All boundary values must be positive.

    Notes:
        - All boundary values must be positive (required for logarithmic spacing).
        - Open boundaries exclude the exact endpoint using nextafter.
        - Pieces must be adjacent: the upper bound of each piece must equal the
          lower bound of the next piece, with compatible open/closed boundaries.

    """

    pieces: tuple[Piece, ...]

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
            grid_helpers.logspace(
                self._piece_starts[i], self._piece_stops[i], p.n_points
            )
            for i, p in enumerate(self.pieces)
        ]
        return jnp.concatenate(piece_arrays)

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the generalized coordinate of a value in the grid."""
        piece_idx = jnp.searchsorted(self._breakpoints, value, side="right")
        local_coord = grid_helpers.get_logspace_coordinate(
            value,
            self._piece_starts[piece_idx],
            self._piece_stops[piece_idx],
            self._piece_n_points[piece_idx],
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


# ======================================================================================
# Validate user input
# ======================================================================================


def _validate_discrete_grid(category_class: type) -> None:
    """Validate the field names and values of the category_class passed to DiscreteGrid.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """
    error_messages = validate_category_class(category_class)
    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def validate_category_class(category_class: type) -> list[str]:
    """Validate a category class has proper structure for discrete grids.

    This validates that:
    - The class is a dataclass
    - It has at least one field
    - All field values are int
    - All field values are unique
    - Field values are consecutive integers starting from 0

    Args:
        category_class: The category class to validate. Must be a dataclass with fields
            that have unique int values.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages: list[str] = []

    if not is_dataclass(category_class):
        error_messages.append(
            "category_class must be a dataclass with int fields, "
            f"but is {category_class}."
        )
        return error_messages

    names_and_values = get_field_names_and_values(category_class)

    if not names_and_values:
        error_messages.append("category_class must have at least one field.")

    names_with_non_int_values = [
        name for name, value in names_and_values.items() if not isinstance(value, int)
    ]
    if names_with_non_int_values:
        error_messages.append(
            "Field values of the category_class can only be int values. "
            f"The values to the following fields are not: "
            f"{names_with_non_int_values}"
        )

    values = list(names_and_values.values())

    duplicated_values = find_duplicates(values)
    if duplicated_values:
        error_messages.append(
            "Field values of the category_class must be unique. "
            f"The following values are duplicated: {duplicated_values}"
        )

    if values != list(range(len(values))):
        error_messages.append(
            "Field values of the category_class must be consecutive integers "
            "starting from 0 (e.g., 0, 1, 2, ...)."
        )

    return error_messages


def _validate_continuous_grid(
    start: float,
    stop: float,
    n_points: int,
) -> None:
    """Validate the continuous grid parameters.

    Args:
        start: The start value of the grid.
        stop: The stop value of the grid.
        n_points: The number of points in the grid.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    valid_start_type = isinstance(start, int | float)
    if not valid_start_type:
        error_messages.append("start must be a scalar int or float value")

    valid_stop_type = isinstance(stop, int | float)
    if not valid_stop_type:
        error_messages.append("stop must be a scalar int or float value")

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def _validate_irreg_spaced_grid(points: Sequence[float] | Float1D) -> None:
    """Validate the irregular spaced grid parameters.

    Args:
        points: The grid points.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    if len(points) < 2:  # noqa: PLR2004
        error_messages.append("points must have at least 2 elements")
    else:
        # Check that all elements are numeric
        non_numeric = [
            (i, type(p).__name__)
            for i, p in enumerate(points)
            if not isinstance(p, int | float)
        ]
        if non_numeric:
            error_messages.append(
                f"All elements of points must be int or float. "
                f"Non-numeric elements found at indices: {non_numeric}"
            )
        else:
            # Check that points are in ascending order
            for i in range(len(points) - 1):
                if points[i] >= points[i + 1]:
                    error_messages.append(
                        "Points must be in strictly ascending order. "
                        f"Found points[{i}]={points[i]} >= "
                        f"points[{i + 1}]={points[i + 1]}"
                    )
                    break

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


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
