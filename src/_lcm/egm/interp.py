"""Interpolation on NaN-padded, weakly ascending EGM grids.

The upper-envelope refinement emits grid rows of static length whose unused
tail slots hold NaN and whose kink abscissae appear twice (left- and
right-extrapolated function values). `interp_on_padded_grid` interpolates on
such rows without ever dividing by a zero-width bracket; passing the row's
exact slopes upgrades the linear interpolant to a monotone cubic Hermite one.
Value jumps ride on rows as duplicated abscissae carrying one-sided limits,
so reads near a jump are one-sided by construction.
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

    - Queries below the first node continue the first bracket's secant
      (linear extrapolation, matching the canonical state-grid read); queries
      at or above the last non-NaN node are clamped to the boundary value,
      because a refined row's last bracket can be a crossing-inserted
      near-duplicate whose secant is no usable extrapolant.
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
    raw_position = (x_query - xp_lower) / safe_width
    # Zero-width brackets arise only when a duplicated abscissa sits at the end
    # of the non-NaN prefix; queries there are at or above the duplicate, so
    # the right value applies.
    relative_position = jnp.where(
        bracket_width == 0.0,
        1.0,
        jnp.clip(raw_position, 0.0, 1.0),
    )
    weight_lower = 1.0 - relative_position
    # Blend on results with zero-weight short-circuits: a `-inf` endpoint
    # yields `-inf` wherever it carries positive weight (instead of the NaN
    # of `fp_lower + rel * (fp_upper - fp_lower)`), and contributes exactly
    # nothing at weight zero.
    linear = jnp.where(weight_lower > 0.0, weight_lower * fp_lower, 0.0) + jnp.where(
        relative_position > 0.0, relative_position * fp_upper, 0.0
    )
    # Out-of-range reads are asymmetric:
    # - Below the first node the query continues the first bracket's secant —
    #   the convention of the canonical state-grid value read
    #   (`map_coordinates` extrapolates linearly) — so a transition landing
    #   under the carry's support (a borrowing corner undershooting the grid
    #   start) is priced on the edge slope, not credited with a boundary value
    #   no action attains.
    # - At or above the last node the clamped boundary value applies: refined
    #   envelope rows can end in a crossing-inserted near-duplicate bracket
    #   whose secant slope is arbitrarily steep, so extending it would let one
    #   degenerate pair poison every above-support read.
    # Brackets with a non-finite endpoint (`-inf` infeasible values) keep the
    # clamped read, and a zero-width kink bracket never extends.
    finite_pair = jnp.isfinite(fp_lower) & jnp.isfinite(fp_upper)
    extension = jnp.where(
        finite_pair & (bracket_width != 0.0) & (raw_position < 0.0),
        (raw_position - relative_position) * (fp_upper - fp_lower),
        0.0,
    )
    linear = linear + extension
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


def interp_and_derivative_on_padded_grid(
    *,
    x_query: FloatND,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D | None = None,
) -> tuple[FloatND, FloatND]:
    """Read a NaN-padded row and the analytic derivative of that exact read.

    The value channel is identical to `interp_on_padded_grid`. The derivative
    channel is the closed-form slope of the *selected piece* of that
    interpolant — never an automatic derivative of the bracket-selection
    program, whose `searchsorted`/`clip` representation returns arbitrary
    subgradients at exact grid nodes. Side conventions at the boundaries of a
    piece:

    - at an ordinary node (including the first) the read selects the right
      bracket (`side="right"`), so the derivative is the right piece's
      left-endpoint slope — the node's limited slope under Hermite, the right
      secant under linear;
    - at the last non-NaN node the read clamps into the last bracket, so the
      derivative is that piece's right-endpoint slope (left-side derivative);
    - at a duplicated jump abscissa the right piece applies, matching the
      value read's one-sided convention;
    - below support the derivative is the first bracket's secant (the slope
      of the value read's linear extension); at or above the last node it is
      zero (the value clamps);
    - zero-width and `-inf` brackets read a zero derivative, keeping the
      (value, derivative) pair consistent with the value conventions.

    Args:
        x_query: Points at which to evaluate; any shape.
        xp: Weakly ascending grid row with NaNs only in the tail.
        fp: Function values on `xp`, NaN-padded in lockstep.
        fp_slopes: Node derivatives, NaN-padded in lockstep; `None` selects
            linear interpolation.

    Returns:
        Tuple of the interpolated values and the read's analytic derivative,
        each with the shape of `x_query`.

    """
    search_grid, valid_length = prepare_padded_grid(xp)
    return interp_and_derivative_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def interp_and_derivative_on_prepared_grid(
    *,
    x_query: FloatND,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D | None = None,
) -> tuple[FloatND, FloatND]:
    """Paired value-and-derivative read on a prepared row.

    The contract is `interp_and_derivative_on_padded_grid`'s; this form takes
    the row's `search_grid` and `valid_length` precomputed (via
    `prepare_padded_grid`), mirroring `interp_on_prepared_grid`. The value
    channel matches `interp_on_prepared_grid` bit for bit — both gather the
    same bracket and share the two-node arithmetic.
    """
    upper = jnp.clip(
        jnp.searchsorted(search_grid, x_query, side="right"),
        1,
        jnp.maximum(valid_length - 1, 1),
    ).astype(jnp.int32)
    lower = upper - 1
    value = _interp_between_nodes(
        x_query=x_query,
        xp_lower=xp[lower],
        xp_upper=xp[upper],
        fp_lower=fp[lower],
        fp_upper=fp[upper],
        slope_lower=None if fp_slopes is None else fp_slopes[lower],
        slope_upper=None if fp_slopes is None else fp_slopes[upper],
    )
    derivative = _derivative_between_nodes(
        x_query=x_query,
        xp_lower=xp[lower],
        xp_upper=xp[upper],
        fp_lower=fp[lower],
        fp_upper=fp[upper],
        slope_lower=None if fp_slopes is None else fp_slopes[lower],
        slope_upper=None if fp_slopes is None else fp_slopes[upper],
    )
    return value, derivative


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


def _derivative_between_nodes(
    *,
    x_query: FloatND,
    xp_lower: FloatND,
    xp_upper: FloatND,
    fp_lower: FloatND,
    fp_upper: FloatND,
    slope_lower: FloatND | None = None,
    slope_upper: FloatND | None = None,
) -> FloatND:
    """Analytic derivative of the two-node read `_interp_between_nodes` makes.

    Closed-form differentiation of the selected piece — the bracket indices
    are data, not differentiated program structure, so exact-node queries get
    the declared side's slope instead of an autodiff subgradient of the
    `clip`/`where` representation. Cases, with $t$ the raw relative position:

    - $t < 0$ (below support): the value read extends the bracket's secant
      linearly, so the derivative is that secant; a non-finite bracket keeps
      the clamped value, so the derivative is zero.
    - $0 \\le t \\le 1$: the secant plus, on Hermite-applicable brackets, the
      correction polynomial's slope $C'(t)/h$ — at $t = 0$ this is exactly the
      lower node's limited slope, at $t = 1$ the upper node's.
    - $t > 1$ (above support): the value clamps, so the derivative is zero.
    - Zero-width brackets and brackets with a `-inf` endpoint read zero,
      matching the value conventions (the right value at a kink duplicate,
      `-inf` in a `-inf` bracket's interior).
    """
    bracket_width = xp_upper - xp_lower
    safe_width = jnp.where(bracket_width == 0.0, 1.0, bracket_width)
    raw_position = (x_query - xp_lower) / safe_width
    finite_pair = jnp.isfinite(fp_lower) & jnp.isfinite(fp_upper)
    secant = (fp_upper - fp_lower) / safe_width

    if slope_lower is None or slope_upper is None:
        in_bracket_slope = secant
    else:
        coeff_lower, coeff_upper, applicable = _hermite_coefficients(
            bracket_width=bracket_width,
            safe_width=safe_width,
            fp_lower=fp_lower,
            fp_upper=fp_upper,
            slope_lower=slope_lower,
            slope_upper=slope_upper,
        )
        # C(t) = t(1-t)[(1-t) a + t b] with a = coeff_lower, b = coeff_upper,
        # so C'(t) = a + 2 (b - 2a) t + 3 (a - b) t^2; at the endpoints
        # C'(0) = a and C'(1) = -b, recovering the limited node slopes.
        position = jnp.clip(raw_position, 0.0, 1.0)
        correction_slope = (
            coeff_lower
            + 2.0 * (coeff_upper - 2.0 * coeff_lower) * position
            + 3.0 * (coeff_lower - coeff_upper) * position**2
        ) / safe_width
        in_bracket_slope = secant + jnp.where(applicable, correction_slope, 0.0)

    derivative = jnp.where(
        raw_position < 0.0,
        jnp.where(finite_pair, secant, 0.0),
        jnp.where(raw_position > 1.0, 0.0, in_bracket_slope),
    )
    derivative = jnp.where(bracket_width == 0.0, 0.0, derivative)
    return jnp.where(jnp.isneginf(fp_lower) | jnp.isneginf(fp_upper), 0.0, derivative)


def _hermite_coefficients(
    *,
    bracket_width: FloatND,
    safe_width: FloatND,
    fp_lower: FloatND,
    fp_upper: FloatND,
    slope_lower: FloatND,
    slope_upper: FloatND,
) -> tuple[FloatND, FloatND, BoolND]:
    """Cubic-correction coefficients and applicability mask, per bracket.

    With $h$ the bracket width, $\\Delta f$ the value difference, and
    Fritsch-Carlson limited node slopes $m_0, m_1$ (same sign as the secant,
    magnitude at most three times it — a sufficient condition for the cubic
    to be monotone on each monotone bracket), the coefficients are
    $a = h m_0 - \\Delta f$ and $b = \\Delta f - h m_1$. The correction is
    applicable only on positive-width brackets whose endpoint values and
    slopes are all finite; everywhere else the linear rule (with its `-inf`
    and NaN-padding contracts) stands.
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
    applicable = (
        (bracket_width > 0.0)
        & jnp.isfinite(fp_lower)
        & jnp.isfinite(fp_upper)
        & jnp.isfinite(slope_lower)
        & jnp.isfinite(slope_upper)
    )
    return coeff_lower, coeff_upper, applicable


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

    With $t$ the relative position and $a, b$ the coefficients from
    `_hermite_coefficients`, the cubic Hermite interpolant is the linear
    blend plus $t (1-t) \\left[ (1-t) a + t b \\right]$, which vanishes at
    both endpoints — boundary clamps and zero-width-bracket reads are
    untouched.
    """
    coeff_lower, coeff_upper, applicable = _hermite_coefficients(
        bracket_width=bracket_width,
        safe_width=safe_width,
        fp_lower=fp_lower,
        fp_upper=fp_upper,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
    )
    correction = (
        relative_position
        * (1.0 - relative_position)
        * ((1.0 - relative_position) * coeff_lower + relative_position * coeff_upper)
    )
    return jnp.where(applicable, correction, 0.0)
