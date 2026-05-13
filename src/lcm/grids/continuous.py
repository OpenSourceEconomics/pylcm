import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import jax.numpy as jnp

from lcm._beartype_conf import GRID_CONF, beartype_init
from lcm.dtypes import canonical_float_dtype
from lcm.exceptions import GridInitializationError, format_messages
from lcm.grids import coordinates as grid_coordinates
from lcm.grids.base import Grid
from lcm.typing import (
    Float1D,
    FloatND,
    ScalarFloat,
    ScalarInt,
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
    def get_coordinate(self, value: FloatND) -> FloatND: ...
    @abstractmethod
    def get_coordinate(self, value: ScalarFloat | FloatND) -> ScalarFloat | FloatND:
        """Return the generalized coordinate of a value in the grid."""


@dataclass(frozen=True, kw_only=True, init=False)
class UniformContinuousGrid(ContinuousGrid, ABC):
    """Grid with start/stop/n_points for linearly or logarithmically spaced values.

    `start` and `stop` are stored as JAX scalars at `canonical_float_dtype()`,
    `n_points` as a `jnp.int32` JAX scalar — converted from the Python
    literals (or other numeric inputs) supplied at construction.
    """

    start: ScalarFloat
    """The start value of the grid (JAX scalar at `canonical_float_dtype()`)."""

    stop: ScalarFloat
    """The stop value of the grid (JAX scalar at `canonical_float_dtype()`)."""

    n_points: ScalarInt
    """The number of points in the grid (`jnp.int32` JAX scalar)."""

    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        n_points: int | ScalarInt,
        batch_size: int = 0,
    ) -> None:
        _init_uniform_grid(
            self,
            start=start,
            stop=stop,
            n_points=n_points,
            batch_size=batch_size,
            requires_positive_start=False,
        )

    @abstractmethod
    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: FloatND) -> FloatND: ...
    @abstractmethod
    def get_coordinate(self, value: ScalarFloat | FloatND) -> ScalarFloat | FloatND:
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


@beartype_init(GRID_CONF)
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
    def get_coordinate(self, value: FloatND) -> FloatND: ...
    def get_coordinate(self, value: ScalarFloat | FloatND) -> ScalarFloat | FloatND:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_linspace_coordinate(
            value=value,
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )


@beartype_init(GRID_CONF)
class LogSpacedGrid(UniformContinuousGrid):
    """A logarithmically spaced grid of continuous values.

    Requires `start > 0`.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    """

    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        n_points: int | ScalarInt,
        batch_size: int = 0,
    ) -> None:
        _init_uniform_grid(
            self,
            start=start,
            stop=stop,
            n_points=n_points,
            batch_size=batch_size,
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
    def get_coordinate(self, value: FloatND) -> FloatND: ...
    def get_coordinate(self, value: ScalarFloat | FloatND) -> ScalarFloat | FloatND:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_logspace_coordinate(
            value=value,
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )


def _init_uniform_grid(
    grid: UniformContinuousGrid,
    *,
    start: float | ScalarFloat,
    stop: float | ScalarFloat,
    n_points: int | ScalarInt,
    batch_size: int,
    requires_positive_start: bool,
) -> None:
    """Cast `start` / `stop` / `n_points` to canonical JAX scalars, validate, store.

    `jnp.asarray(..., dtype=canonical_float_dtype())` and `jnp.int32(...)` lift
    every numeric input at the boundary: Python literals from user
    construction, JAX scalars from `dataclasses.replace` round-trips, anything
    else raises here. The validator can then assume strict `ScalarFloat` /
    `ScalarInt` types and only check value invariants (finiteness, ordering,
    positivity).
    """
    dtype = canonical_float_dtype()
    start_jax = jnp.asarray(start, dtype=dtype)
    stop_jax = jnp.asarray(stop, dtype=dtype)
    n_points_jax = jnp.int32(n_points)
    _validate_continuous_grid(
        start=start_jax,
        stop=stop_jax,
        n_points=n_points_jax,
        requires_positive_start=requires_positive_start,
    )
    object.__setattr__(grid, "start", start_jax)
    object.__setattr__(grid, "stop", stop_jax)
    object.__setattr__(grid, "n_points", n_points_jax)
    object.__setattr__(grid, "batch_size", batch_size)


@beartype_init(GRID_CONF)
@dataclass(frozen=True, kw_only=True, init=False)
class IrregSpacedGrid(ContinuousGrid):
    """A grid of continuous values at irregular (user-specified) points.

    This grid type is useful for representing non-uniformly spaced points such as
    Gauss-Hermite quadrature nodes.

    `points` is stored as a JAX array at `canonical_float_dtype()`, converted
    from the Python sequence supplied at construction. When `points` is
    omitted and only `n_points` is given, the points must be supplied at
    runtime via the params.

    Example:
    --------
    Fixed grid: `IrregSpacedGrid(points=[-1.73, -0.58, 0.58, 1.73])` Grid that is only
    completed at runtime via params: `IrregSpacedGrid(n_points=4)`

    """

    points: Float1D | None
    """The grid points in ascending order, or `None` for runtime-supplied points."""

    n_points: int
    """Number of points. Derived from `len(points)` when points are given."""

    def __init__(
        self,
        *,
        points: Sequence[float] | Float1D | None = None,
        n_points: int | None = None,
        batch_size: int = 0,
    ) -> None:
        if points is not None:
            _validate_irreg_spaced_grid(points)
            derived_n = len(points)
            if n_points is None:
                n_points = derived_n
            elif n_points != derived_n:
                raise GridInitializationError(
                    f"n_points ({n_points}) does not match len(points) ({derived_n})"
                )
            stored_points: Float1D | None = jnp.asarray(
                points, dtype=canonical_float_dtype()
            )
        elif n_points is None:
            raise GridInitializationError(
                "Either points or n_points must be specified for IrregSpacedGrid."
            )
        elif n_points < 2:  # noqa: PLR2004
            raise GridInitializationError(
                f"n_points must be at least 2, got {n_points}"
            )
        else:
            stored_points = None
        object.__setattr__(self, "points", stored_points)
        object.__setattr__(self, "n_points", n_points)
        object.__setattr__(self, "batch_size", batch_size)

    @property
    def pass_points_at_runtime(self) -> bool:
        """Whether this grid's points are supplied at runtime via params."""
        return self.points is None

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array.

        Raises `GridInitializationError` for runtime-supplied grids
        (`pass_points_at_runtime=True`). To get the substituted points,
        call `internal_regime.state_action_space(regime_params=...)` and
        read from `.states[name]` or `.continuous_actions[name]`.
        """
        if self.points is None:
            raise GridInitializationError(
                f"IrregSpacedGrid declared with n_points={self.n_points} and "
                f"no points; values are supplied at runtime via "
                f"params['<regime>']['<grid_name>']['points']. To get the "
                f"substituted points, call "
                f"`internal_regime.state_action_space(regime_params=...)` and "
                f"read from `.states[name]` or `.continuous_actions[name]`. "
                f"Use `.n_points` if only the shape is needed."
            )
        return self.points

    @overload
    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat: ...
    @overload
    def get_coordinate(self, value: FloatND) -> FloatND: ...
    def get_coordinate(self, value: ScalarFloat | FloatND) -> ScalarFloat | FloatND:
        """Return the generalized coordinate of a value in the grid."""
        if self.points is None:
            raise GridInitializationError(
                "Cannot compute coordinate without points. Pass points at "
                "initialization or use IrregSpacedGrid(n_points=...) and "
                "supply points at runtime via params."
            )
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.points)


def _validate_continuous_grid(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    n_points: ScalarInt,
    requires_positive_start: bool = False,
) -> None:
    """Validate the continuous grid parameters.

    `start` and `stop` are post-cast canonical-dtype JAX scalars (the
    boundary cast in `_init_uniform_grid` already rejects non-numeric
    inputs); the checks here cover only value invariants.

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

    # Reject NaN/inf early — `start >= stop` returns False for NaN, so an
    # un-finite start would otherwise pass silently and produce a broken grid.
    start_finite = bool(jnp.isfinite(start))
    if not start_finite:
        error_messages.append(f"start must be finite, got {start}")
    stop_finite = bool(jnp.isfinite(stop))
    if not stop_finite:
        error_messages.append(f"stop must be finite, got {stop}")

    if n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if start_finite and stop_finite and start >= stop:
        error_messages.append("start must be less than stop")

    if start_finite and requires_positive_start and start <= 0:
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
            non_finite = [(i, p) for i, p in enumerate(points) if not jnp.isfinite(p)]
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
