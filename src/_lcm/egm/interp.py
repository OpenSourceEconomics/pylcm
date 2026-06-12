"""Linear interpolation on NaN-padded, weakly ascending EGM grids.

The upper-envelope refinement emits grid rows of static length whose unused
tail slots hold NaN and whose kink abscissae appear twice (left- and
right-extrapolated function values). `interp_on_padded_grid` interpolates on
such rows without ever dividing by a zero-width bracket. `locate_on_grid`
produces edge-clamped bracket indices and weights on ordinary (unpadded)
grids, e.g. the passive-state grids of the mixed carry read.
"""

import jax.numpy as jnp

from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt


def interp_on_padded_grid(*, x_query: FloatND, xp: Float1D, fp: Float1D) -> FloatND:
    """Interpolate linearly on a NaN-padded, weakly ascending grid row.

    The NaN padding must form a contiguous tail of `xp` (matched by NaNs in
    `fp`); it is treated as $+\\infty$ when locating brackets, so padding never
    influences the result. Behavior at the boundaries and at duplicated
    abscissae:

    - Queries outside the non-NaN range are clamped to the boundary values.
    - At a duplicated abscissa (an envelope kink carrying left and right
      values), queries strictly below the duplicate interpolate toward the
      left value; queries at or above it use the right value. The zero-width
      bracket between the duplicates is never used as a divisor.
    - A `-inf` endpoint (an infeasible value) forces the bracket's interior
      to `-inf` instead of NaN; a query exactly on a finite neighbor returns
      that neighbor's value.

    Args:
        x_query: Points at which to evaluate the interpolant; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.

    Returns:
        Interpolated values with the shape of `x_query`.

    """
    xp_filled = jnp.where(jnp.isnan(xp), jnp.inf, xp)
    n_valid = jnp.sum(~jnp.isnan(xp))
    upper = jnp.clip(
        jnp.searchsorted(xp_filled, x_query, side="right"),
        1,
        jnp.maximum(n_valid - 1, 1),
    )
    lower = upper - 1
    xp_lower = xp[lower]
    fp_lower = fp[lower]
    fp_upper = fp[upper]
    bracket_width = xp[upper] - xp_lower
    safe_width = jnp.where(bracket_width == 0.0, 1.0, bracket_width)
    # Zero-width brackets arise only when a duplicated abscissa sits at the end
    # of the non-NaN prefix; queries there are at or above the duplicate, so
    # the right value applies.
    relative_position = jnp.where(
        bracket_width == 0.0,
        1.0,
        jnp.clip((x_query - xp_lower) / safe_width, 0.0, 1.0),
    )
    weight_lower = 1.0 - relative_position
    # Blend on results with zero-weight short-circuits: a `-inf` endpoint
    # yields `-inf` wherever it carries positive weight (instead of the NaN
    # of `fp_lower + rel * (fp_upper - fp_lower)`), and contributes exactly
    # nothing at weight zero.
    return jnp.where(weight_lower > 0.0, weight_lower * fp_lower, 0.0) + jnp.where(
        relative_position > 0.0, relative_position * fp_upper, 0.0
    )


def locate_on_grid(
    *,
    x_query: ScalarFloat,
    grid: Float1D,
) -> tuple[ScalarInt, ScalarInt, ScalarFloat]:
    """Locate the bracketing nodes and upper weight of a query on a sorted grid.

    The bracket is edge-clamped: queries below the first node get upper
    weight `0.0` on the first bracket, queries above the last node get upper
    weight `1.0` on the last bracket, so the linear blend
    `(1 - weight) * f[lower] + weight * f[upper]` never extrapolates. A query
    exactly on a node yields a weight of exactly `0.0` or `1.0`, so on-node
    reads reproduce the node values without interpolation error.

    Args:
        x_query: The query point.
        grid: Strictly ascending grid nodes (at least two).

    Returns:
        Tuple of the lower node index, the upper node index, and the weight
        of the upper node.

    """
    n_nodes = grid.shape[0]
    upper = jnp.clip(
        jnp.searchsorted(grid, x_query, side="right"),
        1,
        n_nodes - 1,
    )
    lower = upper - 1
    bracket_width = grid[upper] - grid[lower]
    safe_width = jnp.where(bracket_width == 0.0, 1.0, bracket_width)
    weight_upper = jnp.where(
        bracket_width == 0.0,
        1.0,
        jnp.clip((x_query - grid[lower]) / safe_width, 0.0, 1.0),
    )
    return lower, upper, weight_upper
