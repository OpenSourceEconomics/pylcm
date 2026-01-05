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
    from lcm.typing import Float1D, ScalarFloat, ParamsDict


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


def get_shock_coordinate(
    value: ScalarFloat, n_points: int, params: ParamsDict, distribution_type: str
) -> ScalarFloat:
    """Map a value into the input needed for jax.scipy.ndimage.map_coordinates."""
    if distribution_type == "uniform":
        start = params["start"]
        stop = params["stop"]
        return get_linspace_coordinate(
            value=value, start=start, stop=stop, n_points=n_points
        )
    if distribution_type == "normal":
        start = params["mu_eps"] - params["sigma_eps"] * params["n_std"]
        stop = params["mu_eps"] + params["sigma_eps"] * params["n_std"]
        return get_linspace_coordinate(
            value=value, start=start, stop=stop, n_points=n_points
        )

    # Demean values for ar1 processes
    demeaned_sigma_eps = jnp.sqrt(params["sigma_eps"] ** 2 / (1 - params["rho"] ** 2))
    demeaned_mu_eps = params["mu_eps"] / (1 - params["rho"])

    # Rouwenhorst method always uses 2 times std
    n_std = 2 if distribution_type == "rouwenhorst" else params["n_std"]

    start = demeaned_mu_eps - demeaned_sigma_eps * n_std
    stop = demeaned_mu_eps + demeaned_sigma_eps * n_std
    return get_linspace_coordinate(
        value=value, start=start, stop=stop, n_points=n_points
    )