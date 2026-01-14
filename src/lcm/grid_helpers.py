"""Functions to generate and work with different kinds of grids.

Grid generation functions must have the following signature:

    Signature (start: ScalarFloat, stop: ScalarFloat, n_points: int) -> jax.Array

They take start and end points and create a grid of points between them.


Interpolation info functions must have the following signature:

    Signature (
        value: ScalarFloat,
        start: ScalarFloat,
        stop: ScalarFloat,
        n_points: int
    ) -> ScalarInt

They take the information required to generate a grid, and return an index corresponding
to the value, which is a point in the space but not necessarily a grid point.

Some of the arguments will not be used by all functions but the aligned interface makes
it easy to call functions interchangeably.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lcm.typing import Float1D, ScalarFloat


def linspace(start: ScalarFloat, stop: ScalarFloat, n_points: int) -> Float1D:
    """Wrapper around jnp.linspace.

    Returns a linearly spaced grid between start and stop with n_points, including both
    endpoints.

    """
    return jnp.linspace(start, stop, n_points)


def get_linspace_coordinate(
    value: ScalarFloat,
    start: ScalarFloat,
    stop: ScalarFloat,
    n_points: int,
) -> ScalarFloat:
    """Map a value into the input needed for jax.scipy.ndimage.map_coordinates."""
    step_length = (stop - start) / (n_points - 1)
    return (value - start) / step_length


def logspace(start: ScalarFloat, stop: ScalarFloat, n_points: int) -> Float1D:
    """Wrapper around jnp.logspace.

    Returns a logarithmically spaced grid between start and stop with n_points,
    including both endpoints.

    From the JAX documentation:

        In linear space, the sequence starts at base ** start (base to the power of
        start) and ends with base ** stop [...].

    """
    start_linear = jnp.log(start)
    stop_linear = jnp.log(stop)
    return jnp.logspace(start_linear, stop_linear, n_points, base=jnp.e)


def get_logspace_coordinate(
    value: ScalarFloat,
    start: ScalarFloat,
    stop: ScalarFloat,
    n_points: int,
) -> ScalarFloat:
    """Map a value into the input needed for jax.scipy.ndimage.map_coordinates."""
    # Transform start, stop, and value to linear scale
    start_linear = jnp.log(start)
    stop_linear = jnp.log(stop)
    value_linear = jnp.log(value)

    # Calculate coordinate in linear space
    coordinate_in_linear_space = get_linspace_coordinate(
        value_linear,
        start_linear,
        stop_linear,
        n_points,
    )

    # Calculate rank of lower and upper point in logarithmic space
    rank_lower_gridpoint = jnp.floor(coordinate_in_linear_space)
    rank_upper_gridpoint = rank_lower_gridpoint + 1

    # Calculate lower and upper point in logarithmic space
    step_length_linear = (stop_linear - start_linear) / (n_points - 1)
    lower_gridpoint = jnp.exp(start_linear + step_length_linear * rank_lower_gridpoint)
    upper_gridpoint = jnp.exp(start_linear + step_length_linear * rank_upper_gridpoint)

    # Calculate the decimal part of coordinate
    logarithmic_step_size_at_coordinate = upper_gridpoint - lower_gridpoint
    distance_from_lower_gridpoint = value - lower_gridpoint

    # If the distance from the lower gridpoint is zero, the coordinate corresponds to
    # the rank of the lower gridpoint. The other extreme is when the distance is equal
    # to the logarithmic step size at the coordinate, in which case the coordinate
    # corresponds to the rank of the upper gridpoint. For values in between, the
    # coordinate lies on a linear scale between the ranks of the lower and upper
    # gridpoints.
    decimal_part = distance_from_lower_gridpoint / logarithmic_step_size_at_coordinate
    return rank_lower_gridpoint + decimal_part


def get_irreg_coordinate(
    value: ScalarFloat,
    points: Float1D | Sequence[float],
) -> ScalarFloat:
    """Get the generalized coordinate of a value in an irregularly spaced grid.

    Uses binary search (jnp.searchsorted) to find the position of the value among
    the grid points, then linearly interpolates to get a fractional coordinate.

    Args:
        value: The value to find the coordinate for.
        points: The grid points in ascending order. Can be a JAX array or a sequence
            of floats.

    Returns:
        The generalized coordinate of the value in the grid. For a value equal to
        points[i], returns i. For values between grid points, returns a fractional
        coordinate based on linear interpolation.

    """
    points_arr = jnp.asarray(points)
    n_points = len(points_arr)

    # Find the index of the first point greater than value
    idx_upper = jnp.searchsorted(points_arr, value, side="right")

    # Clamp to valid range for interpolation
    idx_upper = jnp.clip(idx_upper, 1, n_points - 1)
    idx_lower = idx_upper - 1

    # Get the lower and upper grid points
    lower_point = points_arr[idx_lower]
    upper_point = points_arr[idx_upper]

    # Linear interpolation between grid points
    step_size = upper_point - lower_point
    distance_from_lower = value - lower_point
    decimal_part = distance_from_lower / step_size

    return idx_lower + decimal_part
