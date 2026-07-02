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


def interp_across_breakpoints(
    *,
    queries: Float1D,
    grid: Float1D,
    values: Float1D,
    breakpoints: Float1D,
) -> Float1D:
    """Interpolate linearly without averaging across declared value jumps.

    The values are smooth on each side of every breakpoint but may jump across
    it. Each query's stencil is restricted to grid points on the query's own
    side of the breakpoints:

    - both bracketing grid points on the query's side ⇒ ordinary linear
      interpolation between them;
    - the bracketing segment straddles a breakpoint ⇒ one-sided extrapolation
      from the query's own side's boundary segment (the nearest two grid
      points fully inside its interval).

    Each breakpoint interval is assumed to contain at least two grid points;
    a sparser interval falls back to whatever segment the shift reaches.

    Args:
        queries: Points at which to read the values.
        grid: Ascending sample grid.
        values: Values at the grid points, jump-discontinuous only at the
            breakpoints.
        breakpoints: Ascending jump locations partitioning the grid's span.

    Returns:
        The side-faithful linear read at each query.

    """
    query_interval = jnp.searchsorted(breakpoints, queries, side="right")
    grid_interval = jnp.searchsorted(breakpoints, grid, side="right")
    n_points = grid.shape[0]
    hi = jnp.clip(jnp.searchsorted(grid, queries, side="right"), 1, n_points - 1)
    lo = hi - 1
    lo_on_side = grid_interval[lo] == query_interval
    hi_on_side = grid_interval[hi] == query_interval
    # Query below the breakpoint with `hi` beyond it: step the segment left.
    lo_shifted = jnp.clip(jnp.where(hi_on_side, lo, lo - 1), 0, n_points - 1)
    hi_shifted = jnp.where(hi_on_side, hi, lo)
    # Query above the breakpoint with `lo` before it: step the segment right.
    lo_final = jnp.where(lo_on_side, lo_shifted, hi)
    hi_final = jnp.where(lo_on_side, hi_shifted, jnp.clip(hi + 1, 0, n_points - 1))
    # A shifted endpoint can itself sit beyond a *second* breakpoint (a sparse
    # interval with fewer than two grid nodes). Collapse any off-side endpoint
    # onto the own-side one, degrading to a constant own-side read.
    lo_ok = grid_interval[lo_final] == query_interval
    hi_ok = grid_interval[hi_final] == query_interval
    lo_final = jnp.where(lo_ok, lo_final, hi_final)
    hi_final = jnp.where(hi_ok, hi_final, lo_final)
    x0, x1 = grid[lo_final], grid[hi_final]
    y0, y1 = values[lo_final], values[hi_final]
    span = jnp.where(x1 > x0, x1 - x0, 1.0)
    slope = jnp.where(x1 > x0, (y1 - y0) / span, 0.0)
    return y0 + slope * (queries - x0)


def interp_across_breakpoints_on_prepared_grid(
    *,
    x_query: ScalarFloat,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    breakpoints: Float1D,
) -> ScalarFloat:
    """Read one NaN-padded row at a query without averaging across jumps.

    The padded-row counterpart of `interp_across_breakpoints`: the bracket is
    located on the prepared search key and clamped to the row's valid prefix,
    then the stencil is restricted to the query's own side of the breakpoints
    — shifting off the straddling segment and collapsing onto the own-side
    endpoint where the side holds fewer than two nodes.

    Args:
        x_query: The query point.
        search_grid: The row's `+inf`-padded search key from
            `prepare_padded_grid`.
        valid_length: The row's valid prefix length.
        xp: The NaN-padded, weakly ascending abscissae row.
        fp: Values at `xp`, jump-discontinuous only at the breakpoints.
        breakpoints: Ascending jump locations partitioning the row's span.

    Returns:
        The side-faithful linear read at the query.

    """
    last = jnp.maximum(valid_length - 1, 1)
    hi = jnp.clip(jnp.searchsorted(search_grid, x_query, side="right"), 1, last)
    lo = hi - 1
    query_interval = jnp.searchsorted(breakpoints, x_query, side="right")

    def node_interval(index: ScalarInt) -> ScalarInt:
        return jnp.searchsorted(breakpoints, xp[index], side="right")

    lo_on_side = node_interval(lo) == query_interval
    hi_on_side = node_interval(hi) == query_interval
    lo_shifted = jnp.clip(jnp.where(hi_on_side, lo, lo - 1), 0, last)
    hi_shifted = jnp.where(hi_on_side, hi, lo)
    lo_final = jnp.where(lo_on_side, lo_shifted, hi)
    hi_final = jnp.where(lo_on_side, hi_shifted, jnp.clip(hi + 1, 0, last))
    lo_ok = node_interval(lo_final) == query_interval
    hi_ok = node_interval(hi_final) == query_interval
    lo_final = jnp.where(lo_ok, lo_final, hi_final)
    hi_final = jnp.where(hi_ok, hi_final, lo_final)
    x0, x1 = xp[lo_final], xp[hi_final]
    y0, y1 = fp[lo_final], fp[hi_final]
    span = jnp.where(x1 > x0, x1 - x0, 1.0)
    slope = jnp.where(x1 > x0, (y1 - y0) / span, 0.0)
    return y0 + slope * (x_query - x0)


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
    # An empty valid prefix (`valid_length == 0`) can only arise from an
    # already-poisoned carry; the index clamp below keeps the gather in bounds,
    # and the result stays NaN so the runtime NaN diagnostics surface the
    # poisoned carry rather than masking it with an edge-clamped constant.
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
    return _interp_between_nodes(
        x_query=x_query,
        xp_lower=xp[lower],
        xp_upper=xp[upper],
        fp_lower=fp[lower],
        fp_upper=fp[upper],
        slope_lower=None if fp_slopes is None else fp_slopes[lower],
        slope_upper=None if fp_slopes is None else fp_slopes[upper],
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
