"""Explicit numerical stages of the original DC-EGM declaration validators."""

import jax
import jax.numpy as jnp


def _validation_grid_max(*, points: jax.Array) -> jax.Array:
    """Original grid maximum, before its host float comparison."""
    return jnp.max(points)


def _validation_grid_first(*, points: jax.Array) -> jax.Array:
    """Original first-node extraction for literal lower-bound agreement."""
    return points[0]


def _validation_bound_disagrees(
    *, declared: jax.Array, grid_low: jax.Array
) -> jax.Array:
    """Original exact comparison after conversion to the grid's dtype."""
    return declared != grid_low
