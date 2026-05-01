import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import jax.numpy as jnp
from jax import Array

from lcm.exceptions import GridInitializationError, format_messages
from lcm.grids import coordinates as grid_coordinates
from lcm.grids.base import Grid
from lcm.typing import (
    Float1D,
    ScalarFloat,
)


@dataclass(frozen=True, kw_only=True)
class ContinuousGrid(Grid):
    """Base class for grids representing continuous values with coordinate lookup.

    All subclasses must implement `get_coordinate` for value-to-coordinate mapping
    used in interpolation.

    """

    batch_size: int = 0
    """Size of the batches that are looped over during the solution."""

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    @abstractmethod
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""


@dataclass(frozen=True, kw_only=True)
class UniformContinuousGrid(ContinuousGrid, ABC):
    """Grid with start/stop/n_points for linearly or logarithmically spaced values."""

    start: int | float
    """The start value of the grid."""

    stop: int | float
    """The stop value of the grid."""

    n_points: int
    """The number of points in the grid."""

    def __post_init__(self) -> None:
        _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )

    @abstractmethod
    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    @abstractmethod
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
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

    """

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_coordinates.linspace(
            start=self.start, stop=self.stop, n_points=self.n_points
        )

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_linspace_coordinate(
            value=value, start=self.start, stop=self.stop, n_points=self.n_points
        )


class LogSpacedGrid(UniformContinuousGrid):
    """A logarithmically spaced grid of continuous values.

    Requires `start > 0`.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    """

    def __post_init__(self) -> None:
        _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
            requires_positive_start=True,
        )

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_coordinates.logspace(
            start=self.start, stop=self.stop, n_points=self.n_points
        )

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_logspace_coordinate(
            value=value, start=self.start, stop=self.stop, n_points=self.n_points
        )


@dataclass(frozen=True, kw_only=True)
class IrregSpacedGrid(ContinuousGrid):
    """A grid of continuous values at irregular (user-specified) points.

    This grid type is useful for representing non-uniformly spaced points such as
    Gauss-Hermite quadrature nodes.

    When `points` is omitted and only `n_points` is given, the `points` must be
    supplied at runtime via the params.

    Example:
    --------
    Fixed grid: `IrregSpacedGrid(points=[-1.73, -0.58, 0.58, 1.73])` Grid that is only
    completed at runtime via params: `IrregSpacedGrid(n_points=4)`

    """

    points: Sequence[float] | Float1D | None = None
    """The grid points in ascending order, or `None` for runtime-supplied points."""

    n_points: int | None = None
    """Number of points. Derived from `len(points)` when points are given."""

    def __post_init__(self) -> None:
        if self.points is not None:
            _validate_irreg_spaced_grid(self.points)
            # Derive n_points from points if not explicitly set
            if self.n_points is None:
                object.__setattr__(self, "n_points", len(self.points))
            elif self.n_points != len(self.points):
                raise GridInitializationError(
                    f"n_points ({self.n_points}) does not match "
                    f"len(points) ({len(self.points)})"
                )
        elif self.n_points is None:
            raise GridInitializationError(
                "Either points or n_points must be specified for IrregSpacedGrid."
            )
        elif self.n_points < 2:  # noqa: PLR2004
            raise GridInitializationError(
                f"n_points must be at least 2, got {self.n_points}"
            )

    @property
    def pass_points_at_runtime(self) -> bool:
        """Whether this grid's points are supplied at runtime via params."""
        return self.points is None

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array.

        Raises `GridInitializationError` for runtime-supplied grids
        (`pass_points_at_runtime=True`). Substitution happens at solve /
        simulate time via `InternalRegime.state_action_space(regime_params=...)`;
        any code path that reads the base grid's points before substitution is
        a bug.
        """
        if self.points is None:
            raise GridInitializationError(
                f"IrregSpacedGrid was declared with n_points={self.n_points} "
                f"and no points; values are supplied at runtime via "
                f"params['<regime>']['<grid_name>']['points']. Reading the grid "
                f"before substitution is a bug — call "
                f"`internal_regime.state_action_space(regime_params=...)` and "
                f"read points from there, or use `.n_points` if only the shape "
                f"is needed."
            )
        return jnp.asarray(self.points)

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: Array) -> Array: ...
    def get_coordinate(self, value: ScalarFloat | Array) -> ScalarFloat | Array:
        """Return the generalized coordinate of a value in the grid."""
        if self.points is None:
            raise GridInitializationError(
                "Cannot compute coordinate without points. Pass points at "
                "initialization or use IrregSpacedGrid(n_points=...) and "
                "supply points at runtime via params."
            )
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.to_jax())


def _validate_continuous_grid(
    *,
    start: float,
    stop: float,
    n_points: int,
    requires_positive_start: bool = False,
) -> None:
    """Validate the continuous grid parameters.

    Args:
        start: The start value of the grid.
        stop: The stop value of the grid.
        n_points: The number of points in the grid.
        requires_positive_start: If True, also require `start > 0` (used by
            log-spaced grids since `log(x)` is undefined for `x <= 0`).

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

    # Reject NaN/inf early — `start >= stop` returns False for NaN, so an
    # un-finite start would otherwise pass silently and produce a broken grid.
    if valid_start_type and not math.isfinite(start):
        error_messages.append(f"start must be finite, got {start}")
        valid_start_type = False
    if valid_stop_type and not math.isfinite(stop):
        error_messages.append(f"stop must be finite, got {stop}")
        valid_stop_type = False

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    if valid_start_type and requires_positive_start and start <= 0:
        error_messages.append(
            f"start must be > 0 for a log-spaced grid (got {start}); "
            f"`log(x)` is undefined for `x <= 0`."
        )

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
            # Reject NaN/inf — comparisons with NaN are False, so the
            # ascending-order check below would silently let them through.
            non_finite = [(i, p) for i, p in enumerate(points) if not math.isfinite(p)]
            if non_finite:
                error_messages.append(
                    f"All elements of points must be finite. "
                    f"Non-finite elements found at: {non_finite}"
                )
            else:
                # Check that points are in strictly ascending order
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
