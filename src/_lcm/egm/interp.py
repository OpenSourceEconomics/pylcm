"""Interpolation on NaN-padded, weakly ascending EGM grids.

The upper-envelope refinement emits grid rows of static length whose unused
tail slots hold NaN and whose kink abscissae appear twice (left- and
right-extrapolated function values). `interp_on_padded_grid` interpolates on
such rows without ever dividing by a zero-width bracket; passing the row's
exact slopes upgrades the linear interpolant to a monotone cubic Hermite one.
`locate_on_grid` produces edge-clamped bracket indices and weights on
ordinary (unpadded) grids, e.g. the passive-state grids of the mixed carry
read.
"""

import jax.numpy as jnp

from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt


def interp_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D | None = None,
) -> FloatND:
    """Interpolate on a NaN-padded, weakly ascending grid row.

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

    Without `fp_slopes` the interpolant is piecewise linear. With `fp_slopes`
    — the exact derivatives $f'(x)$ at the `xp` nodes (for an EGM value row,
    the marginal-utility row via the envelope theorem) — each bracket gets a
    cubic Hermite correction instead, with Fritsch-Carlson slope limiting so
    the interpolant stays monotone on monotone data. Linear interpolation of
    a concave value row is biased downward by $O(h^2)$ per read and the bias
    compounds across backward induction; exact slopes remove it at no extra
    data cost. Brackets with a non-finite endpoint or slope fall back to the
    linear rule, so the NaN-padding, kink, and `-inf` contracts above are
    unchanged.

    Args:
        x_query: Points at which to evaluate the interpolant; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.
        fp_slopes: Derivatives of `fp` with respect to `xp` at the `xp`
            nodes, NaN-padded in lockstep; `None` selects linear
            interpolation.

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
    linear = jnp.where(weight_lower > 0.0, weight_lower * fp_lower, 0.0) + jnp.where(
        relative_position > 0.0, relative_position * fp_upper, 0.0
    )
    if fp_slopes is None:
        return linear
    return linear + _hermite_correction(
        relative_position=relative_position,
        bracket_width=bracket_width,
        safe_width=safe_width,
        fp_lower=fp_lower,
        fp_upper=fp_upper,
        slope_lower=fp_slopes[lower],
        slope_upper=fp_slopes[upper],
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


def _hermite_correction(
    *,
    relative_position: FloatND,
    bracket_width: FloatND,
    safe_width: FloatND,
    fp_lower: FloatND,
    fp_upper: FloatND,
    slope_lower: FloatND,
    slope_upper: FloatND,
) -> FloatND:
    """Cubic Hermite correction on top of the linear blend, per bracket.

    With $t$ the relative position, $h$ the bracket width, $\\Delta f$ the
    value difference, and limited node slopes $m_0, m_1$, the cubic Hermite
    interpolant is the linear blend plus
    $t (1-t) \\left[ (1-t)(h m_0 - \\Delta f) + t (\\Delta f - h m_1) \\right]$,
    which vanishes at both endpoints — boundary clamps and zero-width-bracket
    reads are untouched. The slopes are Fritsch-Carlson limited (same sign as
    the secant, magnitude at most three times it), a sufficient condition for
    the interpolant to be monotone on each monotone bracket. The correction
    is applied only on positive-width brackets whose endpoint values and
    slopes are all finite; everywhere else it is zero and the linear rule
    (with its `-inf` and NaN-padding contracts) stands.
    """
    df = fp_upper - fp_lower
    safe_df = jnp.where(jnp.isfinite(df), df, 0.0)
    secant = safe_df / safe_width

    def limit(slope: FloatND) -> FloatND:
        same_sign = slope * secant > 0.0
        limited = jnp.sign(secant) * jnp.minimum(jnp.abs(slope), 3.0 * jnp.abs(secant))
        return jnp.where(same_sign, limited, 0.0)

    coeff_lower = safe_width * limit(slope_lower) - safe_df
    coeff_upper = safe_df - safe_width * limit(slope_upper)
    correction = (
        relative_position
        * (1.0 - relative_position)
        * ((1.0 - relative_position) * coeff_lower + relative_position * coeff_upper)
    )
    applicable = (
        (bracket_width > 0.0)
        & jnp.isfinite(fp_lower)
        & jnp.isfinite(fp_upper)
        & jnp.isfinite(slope_lower)
        & jnp.isfinite(slope_upper)
    )
    return jnp.where(applicable, correction, 0.0)
