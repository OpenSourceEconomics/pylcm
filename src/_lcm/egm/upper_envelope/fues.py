"""Fast Upper-Envelope Scan (FUES) over EGM candidate solutions.

Implements the upper-envelope refinement of Dobrescu, L. I., & Shanker, A.
(2022). Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming.
SSRN 4181302.

Inverting the Euler equation in models with discrete choices yields a value
*correspondence*: in non-concave regions, several candidate points share a
neighborhood of the endogenous grid, each lying on a different choice-specific
value segment. `refine_envelope` scans the candidates once, in ascending grid
order, and keeps only the points on the upper envelope:

- Candidates on a different segment than the last kept point are detected via
  the implied savings $A = R - c$: the segments differ iff
  $|\\Delta A / \\Delta R|$ exceeds `jump_thresh`.
- Dominated candidates are dropped (value decreases, savings non-monotone with
  a falling value gradient, or the candidate lies below the continuation of
  the current segment found by a bounded forward scan).
- Where two kept segments cross, the linear intersection of the segments is
  inserted twice — once with the left- and once with the right-extrapolated
  policy — so the refined arrays remain weakly ascending and policy
  discontinuities at kinks are preserved exactly.

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, ScalarBool, ScalarFloat, ScalarInt


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    jump_thresh: float = 2.0,
    n_points_to_scan: int = 10,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its upper envelope.

    The candidates may arrive in any order; they are sorted (NaN-stable)
    ascending in `endog_grid` first. The refined arrays have static length
    `n_refined`, hold the envelope points in weakly ascending grid order, and
    are NaN-padded in the tail. Segment crossings contribute their linear
    intersection point twice — same abscissa, left- and right-extrapolated
    policy values.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ of the implied
            savings $A = R - c$ above which two points are treated as lying on
            different value-function segments.
        n_points_to_scan: Number of subsequent (or preceding) candidates the
            bounded scans inspect when searching for the next point on a given
            segment.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow; the arrays then hold
        a valid truncated prefix of the envelope. Callers must check the
        counter rather than publish the truncated arrays silently — the EGM
        step NaN-poisons its published rows on overflow so the solve loop's
        NaN diagnostics name the offending (regime, period).

    """
    order = jnp.argsort(endog_grid)
    grid_sorted = endog_grid[order]
    policy_sorted = policy[order]
    value_sorted = value[order]
    n_input = grid_sorted.shape[0]

    first_point = jnp.stack([grid_sorted[0], policy_sorted[0], value_sorted[0]])
    # Carry layout: the two most recent envelope points, k then j, each as
    # (grid, policy, value). Initially both are the first sorted candidate.
    carry_init = jnp.concatenate([first_point, first_point])

    def step(
        carry: Float1D, idx: ScalarInt
    ) -> tuple[Float1D, tuple[Float1D, Float1D, Float1D, ScalarInt]]:
        """Delegate to `_inspect_candidate`; positional per `jax.lax.scan`."""
        return _inspect_candidate(
            carry=carry,
            idx=idx,
            grid_sorted=grid_sorted,
            policy_sorted=policy_sorted,
            value_sorted=value_sorted,
            jump_thresh=jump_thresh,
            n_points_to_scan=n_points_to_scan,
        )

    indices = jnp.arange(1, n_input, dtype=jnp.int32)
    carry_final, (block_grid, block_policy, block_value, block_count) = jax.lax.scan(
        step, carry_init, indices
    )

    # Compact the per-step blocks: route each valid block row to its position
    # in the refined arrays, NaN rows and overflow positions are dropped.
    offsets = jnp.cumsum(block_count) - block_count
    total = jnp.sum(block_count, dtype=jnp.int32)
    slot = jnp.arange(3, dtype=jnp.int32)
    positions = jnp.where(
        slot[None, :] < block_count[:, None],
        offsets[:, None] + slot[None, :],
        n_refined,
    ).ravel()

    refined_grid = jnp.full(n_refined, jnp.nan, dtype=grid_sorted.dtype)
    refined_policy = jnp.full(n_refined, jnp.nan, dtype=policy_sorted.dtype)
    refined_value = jnp.full(n_refined, jnp.nan, dtype=value_sorted.dtype)
    refined_grid = refined_grid.at[positions].set(block_grid.ravel(), mode="drop")
    refined_policy = refined_policy.at[positions].set(block_policy.ravel(), mode="drop")
    refined_value = refined_value.at[positions].set(block_value.ravel(), mode="drop")

    # The last accepted point is still pending in the carry; emit it.
    final_grid, final_policy, final_value = (
        carry_final[3],
        carry_final[4],
        carry_final[5],
    )
    final_valid = ~jnp.isnan(final_grid) & ~jnp.isnan(final_value)
    final_pos = jnp.where(final_valid, total, n_refined)
    refined_grid = refined_grid.at[final_pos].set(final_grid, mode="drop")
    refined_policy = refined_policy.at[final_pos].set(final_policy, mode="drop")
    refined_value = refined_value.at[final_pos].set(final_value, mode="drop")

    n_kept = (total + final_valid).astype(jnp.int32)
    return refined_grid, refined_policy, refined_value, n_kept


def _inspect_candidate(
    *,
    carry: Float1D,
    idx: ScalarInt,
    grid_sorted: Float1D,
    policy_sorted: Float1D,
    value_sorted: Float1D,
    jump_thresh: float,
    n_points_to_scan: int,
) -> tuple[Float1D, tuple[Float1D, Float1D, Float1D, ScalarInt]]:
    """Inspect candidate `idx` and emit the envelope points it finalizes.

    The last accepted point `j` stays in the carry until its successor is
    decided, so a point revealed as dominated by a later candidate is never
    emitted. Each step emits up to three points (in ascending grid order): the
    finalized `j`, and the duplicated intersection of two crossing segments.

    Args:
        carry: The two most recent envelope points, `k` then `j`, each as
            (grid, policy, value).
        idx: Index of the candidate to inspect.
        grid_sorted: Sorted candidate endogenous grid points.
        policy_sorted: Candidate policy values at `grid_sorted`.
        value_sorted: Candidate value-correspondence points at `grid_sorted`.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ above which two
            points lie on different segments.
        n_points_to_scan: Number of candidates the bounded scans inspect.

    Returns:
        Tuple of the updated carry and the per-step output block: three
        emission rows for grid, policy, and value, plus the number of valid
        rows.

    """
    grid_k, policy_k, value_k, grid_j, policy_j, value_j = carry
    grid_i = grid_sorted[idx]
    policy_i = policy_sorted[idx]
    value_i = value_sorted[idx]

    candidate_valid = ~jnp.isnan(grid_i) & ~jnp.isnan(value_i)
    switches = _has_policy_jump(
        grid_a=grid_j,
        policy_a=policy_j,
        grid_b=grid_i,
        policy_b=policy_i,
        jump_thresh=jump_thresh,
    )
    secant = _slope(x_a=grid_j, y_a=value_j, x_b=grid_i, y_b=value_i)
    grad_before = _slope(x_a=grid_k, y_a=value_k, x_b=grid_j, y_b=value_j)

    # Next candidate on j's segment after i: defines the continuation of the
    # current segment for both the suboptimality test and the lower line of a
    # crossing between j and i.
    j_seg_found, j_seg_grid, j_seg_policy, j_seg_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        anchor_grid=grid_j,
        anchor_policy=policy_j,
        idx=idx,
        direction=1,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    secant_i_to_j_seg = _slope(x_a=grid_i, y_a=value_i, x_b=j_seg_grid, y_b=j_seg_value)
    below_j_segment = switches & j_seg_found & (secant < secant_i_to_j_seg)

    dropped = (
        ~candidate_valid
        | (value_i < value_j)
        | (((grid_i - policy_i) < (grid_j - policy_j)) & (secant < grad_before))
        | below_j_segment
    )

    # A same-segment partner of i defines i's segment line (forward preferred:
    # after a crossing, i's segment continues to the right).
    fwd_found, fwd_grid, fwd_policy, fwd_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        anchor_grid=grid_i,
        anchor_policy=policy_i,
        idx=idx,
        direction=1,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    bwd_found, bwd_grid, bwd_policy, bwd_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        anchor_grid=grid_i,
        anchor_policy=policy_i,
        idx=idx,
        direction=-1,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    partner_found = fwd_found | bwd_found
    partner_grid = jnp.where(fwd_found, fwd_grid, bwd_grid)
    partner_policy = jnp.where(fwd_found, fwd_policy, bwd_policy)
    partner_value = jnp.where(fwd_found, fwd_value, bwd_value)
    i_seg_slope = _slope(x_a=grid_i, y_a=value_i, x_b=partner_grid, y_b=partner_value)
    i_seg_policy_slope = _slope(
        x_a=grid_i, y_a=policy_i, x_b=partner_grid, y_b=partner_policy
    )

    # j below i's segment line means the crossing happened before j: j is
    # dominated and is replaced by the intersection of the line through (k, j)
    # with i's segment line.
    j_dominated = ~dropped & switches & partner_found & (secant > i_seg_slope)
    policy_slope_kj = _slope(x_a=grid_k, y_a=policy_k, x_b=grid_j, y_b=policy_j)
    kink_grid_6, kink_value_6 = _intersect_lines(
        x_a=grid_j,
        y_a=value_j,
        slope_a=grad_before,
        x_b=grid_i,
        y_b=value_i,
        slope_b=i_seg_slope,
    )
    kink_policy_left_6 = policy_j + policy_slope_kj * (kink_grid_6 - grid_j)
    kink_policy_right_6 = policy_i + i_seg_policy_slope * (kink_grid_6 - grid_i)
    kink_6 = j_dominated & (kink_grid_6 > grid_k) & (kink_grid_6 < grid_i)

    # Crossing strictly between j and i: both stay on the envelope and the
    # intersection of their segment lines is inserted between them.
    j_seg_slope = jnp.where(
        j_seg_found,
        _slope(x_a=grid_j, y_a=value_j, x_b=j_seg_grid, y_b=j_seg_value),
        grad_before,
    )
    j_seg_policy_slope = jnp.where(
        j_seg_found,
        _slope(x_a=grid_j, y_a=policy_j, x_b=j_seg_grid, y_b=j_seg_policy),
        policy_slope_kj,
    )
    kink_grid_5, kink_value_5 = _intersect_lines(
        x_a=grid_j,
        y_a=value_j,
        slope_a=j_seg_slope,
        x_b=grid_i,
        y_b=value_i,
        slope_b=i_seg_slope,
    )
    kink_policy_left_5 = policy_j + j_seg_policy_slope * (kink_grid_5 - grid_j)
    kink_policy_right_5 = policy_i + i_seg_policy_slope * (kink_grid_5 - grid_i)
    kink_5 = (
        ~dropped
        & ~j_dominated
        & switches
        & partner_found
        & (kink_grid_5 > grid_j)
        & (kink_grid_5 < grid_i)
    )

    plain = ~dropped & ~j_dominated & ~kink_5
    nan_scalar = jnp.full((), jnp.nan, dtype=grid_sorted.dtype)

    emits_j = plain | kink_5
    row_grid = jnp.stack(
        [
            jnp.where(kink_6, kink_grid_6, jnp.where(emits_j, grid_j, nan_scalar)),
            jnp.where(kink_6, kink_grid_6, jnp.where(kink_5, kink_grid_5, nan_scalar)),
            jnp.where(kink_5, kink_grid_5, nan_scalar),
        ]
    )
    row_policy = jnp.stack(
        [
            jnp.where(
                kink_6,
                kink_policy_left_6,
                jnp.where(emits_j, policy_j, nan_scalar),
            ),
            jnp.where(
                kink_6,
                kink_policy_right_6,
                jnp.where(kink_5, kink_policy_left_5, nan_scalar),
            ),
            jnp.where(kink_5, kink_policy_right_5, nan_scalar),
        ]
    )
    row_value = jnp.stack(
        [
            jnp.where(kink_6, kink_value_6, jnp.where(emits_j, value_j, nan_scalar)),
            jnp.where(
                kink_6, kink_value_6, jnp.where(kink_5, kink_value_5, nan_scalar)
            ),
            jnp.where(kink_5, kink_value_5, nan_scalar),
        ]
    )
    count = jnp.where(kink_5, 3, jnp.where(kink_6, 2, jnp.where(plain, 1, 0))).astype(
        jnp.int32
    )

    # New k is the point emitted last this step: the right-policy copy of an
    # inserted kink, the finalized j on a plain accept, or unchanged when
    # nothing was emitted (drop, or j dominated without a valid kink). New j
    # is the accepted candidate i.
    keeps_k = dropped | (j_dominated & ~kink_6)
    new_k_grid = jnp.where(
        keeps_k,
        grid_k,
        jnp.where(kink_5, kink_grid_5, jnp.where(kink_6, kink_grid_6, grid_j)),
    )
    new_k_policy = jnp.where(
        keeps_k,
        policy_k,
        jnp.where(
            kink_5,
            kink_policy_right_5,
            jnp.where(kink_6, kink_policy_right_6, policy_j),
        ),
    )
    new_k_value = jnp.where(
        keeps_k,
        value_k,
        jnp.where(kink_5, kink_value_5, jnp.where(kink_6, kink_value_6, value_j)),
    )
    carry_accepted = jnp.stack(
        [new_k_grid, new_k_policy, new_k_value, grid_i, policy_i, value_i]
    )
    carry_new = jnp.where(dropped, carry, carry_accepted)

    return carry_new, (row_grid, row_policy, row_value, count)


def _find_same_segment_point(
    *,
    grid: Float1D,
    policy: Float1D,
    value: Float1D,
    anchor_grid: ScalarFloat,
    anchor_policy: ScalarFloat,
    idx: ScalarInt,
    direction: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> tuple[ScalarBool, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Find the candidate nearest to `idx` on the anchor's value segment.

    Inspects up to `n_points_to_scan` candidates after (`direction=1`) or
    before (`direction=-1`) index `idx` and returns the first whose implied
    savings, relative to the anchor point, do not jump.

    Args:
        grid: Sorted candidate endogenous grid points.
        policy: Candidate policy values at `grid`.
        value: Candidate value-correspondence points at `grid`.
        anchor_grid: Endogenous grid point of the anchor.
        anchor_policy: Policy value of the anchor.
        idx: Index the scan starts from (exclusive).
        direction: Scan direction, `1` (forward) or `-1` (backward).
        n_points_to_scan: Number of candidates to inspect.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ above which two
            points lie on different segments.

    Returns:
        Tuple of a found-indicator and the grid, policy, and value of the
        first same-segment candidate (arbitrary when not found).

    """
    n_input = grid.shape[0]
    window = idx + direction * (1 + jnp.arange(n_points_to_scan, dtype=jnp.int32))
    in_bounds = (window >= 0) & (window < n_input)
    clipped = jnp.clip(window, 0, n_input - 1)
    window_grid = grid[clipped]
    window_policy = policy[clipped]
    window_value = value[clipped]
    same_segment = (
        in_bounds
        & ~jnp.isnan(window_grid)
        & ~jnp.isnan(window_value)
        & ~_has_policy_jump(
            grid_a=anchor_grid,
            policy_a=anchor_policy,
            grid_b=window_grid,
            policy_b=window_policy,
            jump_thresh=jump_thresh,
        )
    )
    found = jnp.any(same_segment)
    pos = jnp.argmax(same_segment)
    return found, window_grid[pos], window_policy[pos], window_value[pos]


def _has_policy_jump(
    *,
    grid_a: FloatND,
    policy_a: FloatND,
    grid_b: FloatND,
    policy_b: FloatND,
    jump_thresh: float,
) -> BoolND:
    """Indicate whether two points lie on different value-function segments.

    Points lie on different segments iff the gradient of the implied savings
    $A = R - c$ between them exceeds `jump_thresh` in absolute value.

    Args:
        grid_a: Endogenous grid point(s) of the first point.
        policy_a: Policy value(s) of the first point.
        grid_b: Endogenous grid point(s) of the second point.
        policy_b: Policy value(s) of the second point.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$.

    Returns:
        Boolean indicator(s), broadcast over the inputs.

    """
    savings_slope = _slope(
        x_a=grid_a,
        y_a=grid_a - policy_a,
        x_b=grid_b,
        y_b=grid_b - policy_b,
    )
    return jnp.abs(savings_slope) > jump_thresh


def _slope(*, x_a: FloatND, y_a: FloatND, x_b: FloatND, y_b: FloatND) -> FloatND:
    """Compute the slope between two points, with `0.0` for coincident abscissae.

    Args:
        x_a: Abscissa(e) of the first point.
        y_a: Ordinate(s) of the first point.
        x_b: Abscissa(e) of the second point.
        y_b: Ordinate(s) of the second point.

    Returns:
        Slope(s) $\\Delta y / \\Delta x$, broadcast over the inputs.

    """
    delta_x = x_b - x_a
    return jnp.where(
        delta_x == 0.0, 0.0, (y_b - y_a) / jnp.where(delta_x == 0.0, 1.0, delta_x)
    )


def _intersect_lines(
    *,
    x_a: ScalarFloat,
    y_a: ScalarFloat,
    slope_a: ScalarFloat,
    x_b: ScalarFloat,
    y_b: ScalarFloat,
    slope_b: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Intersect two lines given in point-slope form.

    Args:
        x_a: Abscissa of a point on the first line.
        y_a: Ordinate of a point on the first line.
        slope_a: Slope of the first line.
        x_b: Abscissa of a point on the second line.
        y_b: Ordinate of a point on the second line.
        slope_b: Slope of the second line.

    Returns:
        Tuple of the intersection's abscissa and ordinate; NaN for parallel
        lines.

    """
    denominator = slope_a - slope_b
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    x = jnp.where(
        denominator == 0.0,
        jnp.nan,
        (y_b - y_a + slope_a * x_a - slope_b * x_b) / safe_denominator,
    )
    y = y_a + slope_a * (x - x_a)
    return x, y
