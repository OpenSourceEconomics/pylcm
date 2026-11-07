"""The abstract continuous grids and the validators behind the public ones.

`ContinuousGrid` is the abstract base; `UniformContinuousGrid` covers the
start/stop/n_points forms `lcm.grids.LinSpacedGrid` and `lcm.grids.LogSpacedGrid`
take. The public classes normalize their input and call the initializers and
validators here.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from _lcm.grids.base import Grid
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import GridInitializationError
from lcm.typing import (
    Float1D,
    FloatND,
    OffGrid,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True)
class ContinuousGrid(Grid):
    """Base class for grids representing continuous values with coordinate lookup.

    All subclasses must implement `get_coordinate` for value-to-coordinate mapping
    used in interpolation.

    """

    @property
    def off_grid(self) -> OffGrid:
        """Default lookup policy; stochastic supports retain their own admission."""
        return "extrapolate"

    @abstractmethod
    def get_coordinate(self, value: FloatND) -> FloatND:
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

    off_grid: OffGrid = dataclasses.field(
        default="extrapolate", metadata={"fingerprint_omit_if_default": True}
    )
    """Extrapolate, clip only lookup, clip the next state, or refuse exterior states."""

    @abstractmethod
    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""

    @abstractmethod
    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""

    def replace(
        self, **kwargs: float | int | ScalarFloat | ScalarInt | OffGrid
    ) -> UniformContinuousGrid:
        """Replace the attributes of the grid.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the grid: a
                float or scalar for `start` and `stop`, an int or scalar for
                `n_points`, and an `OffGrid` policy.

        Returns:
            A new grid with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise GridInitializationError(
                f"Failed to replace attributes of the grid. The error was: {e}"
            ) from e


def _init_uniform_grid(
    *,
    grid: UniformContinuousGrid,
    start: float,
    stop: float,
    n_points: int | ScalarInt,
    requires_positive_start: bool,
    off_grid: OffGrid,
) -> None:
    """Cast `start` / `stop` / `n_points` to canonical JAX scalars, validate, store.

    The public constructor hands `start` and `stop` in as Python floats, whether
    the user wrote literals or `dataclasses.replace` round-trips stored JAX
    scalars. `jnp.asarray(..., dtype=canonical_float_dtype())` and
    `jnp.int32(...)` lift them to the stored form, so the validator can assume
    strict `ScalarFloat` / `ScalarInt` types and only check value invariants
    (finiteness, ordering, positivity).
    """
    dtype = canonical_float_dtype()
    _validate_off_grid(off_grid)
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
    object.__setattr__(grid, "off_grid", off_grid)


def _clip_grid_value(*, value: FloatND, points: Float1D) -> FloatND:
    """Cap finite physical values without concealing an invalid transition."""
    return jnp.where(jnp.isfinite(value), jnp.clip(value, points[0], points[-1]), value)


def _validate_off_grid(off_grid: OffGrid) -> None:
    """Refuse ambiguous clipping and undeclared policies at construction."""
    if off_grid not in ("extrapolate", "clip_lookup", "clip_state", "raise"):
        raise GridInitializationError(
            "off_grid must be 'extrapolate', 'clip_lookup', 'clip_state', or 'raise'."
        )


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


def _validate_irreg_spaced_grid(points: list[int | float]) -> None:
    """Validate the irregular spaced grid parameters.

    Args:
        points: The grid points, as Python scalars of one numeric array.

    Raises:
        GridInitializationError: If the grid parameters are invalid.

    """
    error_messages = []

    if len(points) < 2:  # noqa: PLR2004
        error_messages.append("points must have at least 2 elements")
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
