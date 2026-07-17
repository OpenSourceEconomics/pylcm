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

from lcm.typing import BoolND, Float1D, FloatND, ScalarFloat, ScalarInt


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
    - A row with a single valid node is a constant clamp: every query returns
      that node's value.
    - An all-NaN row (an already-poisoned carry) returns NaN at every query,
      so the poison surfaces fail-loud instead of reading as a finite value.

    Without `fp_slopes` the interpolant is piecewise linear. With `fp_slopes`
    — the derivatives $f'(x)$ at the `xp` nodes, *exact at the nodes* (for an
    EGM value row, the marginal-utility row via the envelope theorem) — each
    bracket gets a cubic Hermite correction instead, with Fritsch-Carlson slope
    limiting so the interpolant stays monotone on monotone data. The slopes are
    exact node derivatives, but the cubic value interpolant and a *separate*
    linear interpolant of the marginal row (read for the Euler step) are two
    distinct approximants: between nodes the value's derivative and the
    interpolated marginal need not coincide. The discrepancy is $O(h^2)$ and
    enters the value only at second order through the Euler inversion. Linear
    interpolation of a concave value row is biased downward by $O(h^2)$ per read
    and the bias compounds across backward induction; exact slopes remove it at
    no extra data cost. Brackets with a non-finite endpoint or slope fall back
    to the linear rule, so the NaN-padding, kink, and `-inf` contracts above are
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
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def prepare_padded_grid(xp: Float1D) -> tuple[Float1D, ScalarInt]:
    """Build a NaN-padded grid row's search key and valid prefix length.

    Both outputs depend only on the row, never on a query, so a caller that
    reads one row at many queries prepares them once and passes them to
    `interp_on_prepared_grid`. The per-query path then does only a
    scalar-carry `searchsorted` plus gathers — it never recomputes the NaN
    mask or the `+inf`-filled grid, the grid-length intermediates that an
    in-line preamble would materialize and hold for every query lane.

    Args:
        xp: Weakly ascending grid row with NaNs only in the tail.

    Returns:
        Tuple of the search key — `xp` with its NaN tail replaced by `+inf`,
        so padding sorts above every query — and the valid prefix length (the
        count of non-NaN nodes).

    """
    search_grid = jnp.where(jnp.isnan(xp), jnp.inf, xp)
    valid_length = jnp.sum(~jnp.isnan(xp)).astype(jnp.int32)
    return search_grid, valid_length


def interp_on_prepared_grid(
    *,
    x_query: FloatND,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D | None = None,
) -> FloatND:
    """Interpolate on a row whose search key and valid length are prepared.

    The interpolation contract is identical to `interp_on_padded_grid` (linear
    or Hermite, edge-clamped, tie-safe at kinks, `-inf`-propagating); this form
    only takes the row's `search_grid` and `valid_length` precomputed (via
    `prepare_padded_grid`) instead of deriving them from `xp` per call.
    `search_grid` is used solely to locate brackets; the abscissae, values, and
    slopes are gathered from the original NaN-padded `xp` / `fp` / `fp_slopes`,
    so the result matches the NaN-padded path exactly, including on degenerate
    rows whose valid prefix is one node or empty.

    Args:
        x_query: Points at which to evaluate the interpolant; any shape.
        search_grid: The row's `+inf`-padded search key from
            `prepare_padded_grid`.
        valid_length: The row's non-NaN prefix length from
            `prepare_padded_grid`.
        xp: The original NaN-padded grid row (the abscissa gather source).
        fp: Function values on `xp`, NaN-padded in lockstep.
        fp_slopes: Node derivatives, NaN-padded in lockstep; `None` selects
            linear interpolation.

    Returns:
        Interpolated values with the shape of `x_query`.

    """
    # The bracket indices span the full query mesh (the dominant egm_step
    # working buffer at scale), and never exceed the grid length (a few
    # hundred), so int32 holds them with vast headroom. Under x64 `searchsorted`
    # would default to int64 and double these gather-index buffers; the cast
    # halves them with no effect on the gathered values.
    upper = jnp.clip(
        jnp.searchsorted(search_grid, x_query, side="right"),
        1,
        jnp.maximum(valid_length - 1, 1),
    ).astype(jnp.int32)
    lower = upper - 1
    # Degenerate valid prefixes (fewer than two nodes) gather NaN padding into
    # the bracket. `jnp.where` propagates cotangents through BOTH of its
    # branches, so NaN partials in the discarded bracket arithmetic would
    # poison the selected constant's derivative (`0 · NaN = NaN`) — and the
    # readers are differentiated (asset-row mode grads the continuation read).
    # Feed the arithmetic a finite dummy bracket on those rows; the overrides
    # below still publish the contract values.
    degenerate = valid_length < 2  # noqa: PLR2004

    def _sanitized(gathered: FloatND, dummy: float) -> FloatND:
        return jnp.where(degenerate, dummy, gathered)

    result = _interp_between_nodes(
        x_query=x_query,
        xp_lower=_sanitized(xp[lower], 0.0),
        xp_upper=_sanitized(xp[upper], 1.0),
        fp_lower=_sanitized(fp[lower], 0.0),
        fp_upper=_sanitized(fp[upper], 0.0),
        slope_lower=None if fp_slopes is None else _sanitized(fp_slopes[lower], 0.0),
        slope_upper=None if fp_slopes is None else _sanitized(fp_slopes[upper], 0.0),
    )
    # Degenerate-row contract:
    # - one valid node ⇒ the edge clamp on both sides is that node's value
    #   (constant in the query, identity in the node's value — also under
    #   autodiff),
    # - an empty prefix (only from an already-poisoned carry) ⇒ NaN, so the
    #   runtime NaN diagnostics surface the poison instead of a finite constant.
    result = jnp.where(valid_length == 1, fp[0], result)
    result = jnp.where(valid_length == 0, jnp.nan, result)
    # A NaN query marks an upstream failure. Regular rows propagate it through
    # the bracket arithmetic; the degenerate-row constants above would mask it,
    # so re-pin it explicitly — fail-loud on every row shape.
    return jnp.where(jnp.isnan(x_query), jnp.nan, result)


def interp_right_germ_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the right germ of the Hermite value read on a padded row.

    Same row contract as `interp_on_padded_grid`; see
    `interp_right_germ_on_prepared_grid` for the germ semantics.

    Args:
        x_query: Points at which to evaluate the right germ; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep with `xp`.
        fp_slopes: Derivatives of `fp` with respect to `xp` at the `xp` nodes,
            NaN-padded in lockstep.

    Returns:
        Tuple of the right-finiteness flag and the first, second, and third
        right derivatives, each with the shape of `x_query`.

    """
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_right_germ_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def interp_right_germ_on_prepared_grid(
    *,
    x_query: FloatND,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Compute the right germ of the Hermite value read at each query.

    The germ is the complete local description of the *value interpolant* that
    `interp_on_prepared_grid` evaluates with `fp_slopes` immediately to the
    right of the query — not a read of the slope row itself. Each local piece
    is a cubic (or a constant clamp), so the germ is finite-dimensional: a
    right-finiteness flag plus the first, second, and third one-sided
    derivatives determine the read on a right neighborhood exactly. The germ
    differs from the raw slope row exactly where a tie-owner decision needs the
    truth: the Fritsch-Carlson limiter may cap a node's raw slope, the edge
    clamps flatten the read outside the valid range, and a `-inf` bracket
    endpoint kills the read immediately right of a finite node. Semantics:

    - Strictly inside a bracket: the derivatives of that bracket's limited
      cubic Hermite (the secant and zero curvature where the correction is
      inapplicable — the linear fallback).
    - Exactly on a node: the derivatives at the left edge of the node's *right*
      bracket (the bracket search is right-continuous).
    - Strictly below the first node, and at or above the last valid node: the
      read clamps to a constant — right-finite with all derivatives zero.
    - A bracket with a non-finite endpoint value: not right-finite (the read
      is `-inf` on the bracket's interior), derivatives zero.

    Args:
        x_query: Points at which to evaluate the right germ; any shape.
        search_grid: The row's `+inf`-padded search key from
            `prepare_padded_grid`.
        valid_length: The row's non-NaN prefix length from
            `prepare_padded_grid`.
        xp: The original NaN-padded grid row (the abscissa gather source).
        fp: Function values on `xp`, NaN-padded in lockstep.
        fp_slopes: Node derivatives, NaN-padded in lockstep.

    Returns:
        Tuple of the right-finiteness flag and the first, second, and third
        right derivatives, each with the shape of `x_query`.

    """
    # Identical bracket location to `interp_on_prepared_grid`: `side="right"`
    # puts an on-node query into the node's right bracket, which is exactly the
    # bracket whose germ the right-continuous read needs.
    upper = jnp.clip(
        jnp.searchsorted(search_grid, x_query, side="right"),
        1,
        jnp.maximum(valid_length - 1, 1),
    ).astype(jnp.int32)
    lower = upper - 1
    xp_lower = xp[lower]
    xp_upper = xp[upper]
    fp_lower = fp[lower]
    fp_upper = fp[upper]
    slope_lower = fp_slopes[lower]
    slope_upper = fp_slopes[upper]

    bracket_width = xp_upper - xp_lower
    safe_width = jnp.where(bracket_width == 0.0, 1.0, bracket_width)
    relative_position = jnp.where(
        bracket_width == 0.0,
        1.0,
        jnp.clip((x_query - xp_lower) / safe_width, 0.0, 1.0),
    )
    df = fp_upper - fp_lower
    safe_df = jnp.where(jnp.isfinite(df), df, 0.0)
    secant = safe_df / safe_width

    # The same limiter as `_hermite_correction`, so these derivatives are the
    # derivatives of exactly the polynomial the value read evaluates. In the
    # bracket's local coordinate `t` the read is
    # `p(t) = f_l + Δf t + c_l t + (c_u - 2 c_l) t² + (c_l - c_u) t³`.
    def limit(slope: FloatND) -> FloatND:
        same_sign = slope * secant > 0.0
        limited = jnp.sign(secant) * jnp.minimum(jnp.abs(slope), 3.0 * jnp.abs(secant))
        return jnp.where(same_sign, limited, 0.0)

    coeff_lower = safe_width * limit(slope_lower) - safe_df
    coeff_upper = safe_df - safe_width * limit(slope_upper)
    hermite_first = (
        safe_df
        + (1.0 - 2.0 * relative_position)
        * ((1.0 - relative_position) * coeff_lower + relative_position * coeff_upper)
        + relative_position * (1.0 - relative_position) * (coeff_upper - coeff_lower)
    ) / safe_width
    hermite_second = (
        2.0 * (coeff_upper - 2.0 * coeff_lower)
        + 6.0 * (coeff_lower - coeff_upper) * relative_position
    ) / safe_width**2
    hermite_third = 6.0 * (coeff_lower - coeff_upper) / safe_width**3
    applicable = (
        (bracket_width > 0.0)
        & jnp.isfinite(fp_lower)
        & jnp.isfinite(fp_upper)
        & jnp.isfinite(slope_lower)
        & jnp.isfinite(slope_upper)
    )
    first = jnp.where(applicable, hermite_first, secant)
    second = jnp.where(applicable, hermite_second, 0.0)
    third = jnp.where(applicable, hermite_third, 0.0)
    # The read clamps to a constant strictly below the first node and at or
    # above the last valid one; the germ there is right-finite with all
    # derivatives exactly zero. (`search_grid` is `+inf` on the pad, so a
    # poisoned all-NaN row lands on the lower clamp.)
    first_node = search_grid[0]
    last_node = search_grid[jnp.maximum(valid_length - 1, 0)]
    on_clamp_ray = (x_query < first_node) | (x_query >= last_node)
    right_finite = on_clamp_ray | (jnp.isfinite(fp_lower) & jnp.isfinite(fp_upper))
    return (
        right_finite,
        jnp.where(on_clamp_ray, 0.0, first),
        jnp.where(on_clamp_ray, 0.0, second),
        jnp.where(on_clamp_ray, 0.0, third),
    )


def _interp_between_nodes(
    *,
    x_query: FloatND,
    xp_lower: FloatND,
    xp_upper: FloatND,
    fp_lower: FloatND,
    fp_upper: FloatND,
    slope_lower: FloatND | None = None,
    slope_upper: FloatND | None = None,
) -> FloatND:
    """Interpolate a query between its two bracketing grid nodes.

    The pure two-node arithmetic of the padded-grid interpolant, shared by
    `interp_on_prepared_grid` (which gathers the bracket from a full row) and
    the streamed asset-row publish (which captures the bracket directly during
    the upper-envelope scan). Having both paths reduce to this one function
    guarantees the streamed value cannot diverge from the row-then-interpolate
    value: only *which two nodes* differs, not the arithmetic on them.

    The bracket must already be edge-clamped to a real pair of nodes (queries
    below the first node bracket the first pair, queries at or above the last
    bracket the last pair). The interpolant is then edge-safe by construction:

    - At a zero-width bracket (a duplicated kink abscissa) the relative position
      is forced to `1.0`, so the right node's value applies and the zero width
      is never used as a divisor.
    - A `-inf` endpoint yields `-inf` wherever it carries positive weight
      (instead of the NaN of `fp_lower + rel * (fp_upper - fp_lower)`) and
      contributes exactly nothing at weight zero.

    Args:
        x_query: Point(s) at which to evaluate the interpolant.
        xp_lower: Lower bracket node abscissa.
        xp_upper: Upper bracket node abscissa.
        fp_lower: Function value at the lower node.
        fp_upper: Function value at the upper node.
        slope_lower: Node derivative at the lower node; `None` selects linear
            interpolation.
        slope_upper: Node derivative at the upper node; `None` selects linear
            interpolation.

    Returns:
        Interpolated value(s) with the shape of `x_query`.

    """
    bracket_width = xp_upper - xp_lower
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
    if slope_lower is None or slope_upper is None:
        return linear
    return linear + _hermite_correction(
        relative_position=relative_position,
        bracket_width=bracket_width,
        safe_width=safe_width,
        fp_lower=fp_lower,
        fp_upper=fp_upper,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
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
    # int32 bracket indices: the grid has at most a few hundred nodes, so the
    # x64-default int64 only doubles the index buffers for nothing.
    upper = jnp.clip(
        jnp.searchsorted(grid, x_query, side="right"),
        1,
        n_nodes - 1,
    ).astype(jnp.int32)
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
