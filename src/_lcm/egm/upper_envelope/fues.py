"""Fast Upper-Envelope Scan (FUES) over EGM candidate solutions.

Implements the upper-envelope refinement of Dobrescu, L. I., & Shanker, A.
(2022). Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming.
SSRN 4181302.

Adapted from the `OpenSourceEconomics/upper-envelope` package (Apache-2.0,
© The Upper-Envelope Authors) and substantially modified for pylcm's JAX kernel.

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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, ScalarBool, ScalarFloat, ScalarInt


class QueryBracket(NamedTuple):
    """The two envelope nodes bracketing one query, plus the publish statics.

    The streamed counterpart of a full refined envelope row read at a single
    query: instead of materializing the NaN-padded `n_pad` rows and locating
    the bracket with `searchsorted(side="right")`, `refine_to_bracket` captures
    only the bracketing pair directly during the scan. The arithmetic on the
    pair is then identical to the row-then-interpolate path, so the published
    value cannot diverge — only the bracket-finding differs.

    All fields are arrays so the struct threads through `jax.vmap`/`jax.lax.map`.
    The `lower_*`/`upper_*` pair is already edge-clamped to a real node pair
    (matching the reference's `clip(searchsorted, 1, max(n_kept - 1, 1))`): a
    query below the first node brackets the first pair (weight 0 ⇒ lower value),
    a query at or above the last node brackets the last pair (weight 1 ⇒ upper
    value), and a query exactly on a duplicated kink abscissa brackets the right
    copy as its lower node (the `side="right"` tie-break).
    """

    lower_grid: ScalarFloat
    """Abscissa of the bracket's lower node."""
    upper_grid: ScalarFloat
    """Abscissa of the bracket's upper node."""
    lower_policy: ScalarFloat
    """Policy at the bracket's lower node."""
    upper_policy: ScalarFloat
    """Policy at the bracket's upper node."""
    lower_value: ScalarFloat
    """Value at the bracket's lower node."""
    upper_value: ScalarFloat
    """Value at the bracket's upper node."""
    first_grid: ScalarFloat
    """Abscissa of the lowest envelope node (the constrained-floor edge test)."""
    n_kept: ScalarInt
    """Number of envelope points kept; `> n_pad` signals overflow."""


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    jump_thresh: float = 2.0,
    n_points_to_scan: int = 10,
    segment_id: Float1D | None = None,
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
            different value-function segments. This is a heuristic, *not* a
            theorem-level segment detector: a sound value must exceed the
            within-segment savings-slope (Lipschitz) bound and lie below the
            separated jump signal on the chosen grid, so it is model- and
            grid-dependent. The default `2.0` suits the smooth-policy,
            discrete-choice-kink models pylcm targets; a model with steep
            continuous policies or genuine notches should set it deliberately
            (or pass `segment_id`).
        n_points_to_scan: Number of subsequent (or preceding) candidates the
            bounded scans inspect when searching for the next point on a given
            segment.
        segment_id: Optional per-candidate segment/bracket label, aligned with
            `endog_grid`. When supplied, two points lie on different segments
            iff their policy jumps (above) *or* their labels differ — the
            label-driven switch the FUES bracket indicator uses for genuine
            discontinuities (tax notches) whose policy is locally flat. `None`
            (the default) uses the policy-jump test alone. No pylcm model wires
            a label yet: the constrained/interior join is a *continuous* concave
            kink, not a value-segment switch, so labelling it would insert a
            spurious crossing; this parameter is the hook for a future
            notch/bracket model.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow; the arrays then hold
        a valid truncated prefix of the envelope. Callers must check the
        counter rather than publish the truncated arrays silently — the EGM
        step NaN-poisons its published rows on overflow so the solve loop's
        NaN diagnostics name the offending (regime, period).

    """
    # Sort ascending in the endogenous grid, breaking ties by descending value
    # so the maximal-value candidate leads each run of coincident abscissae;
    # NaN grid points sort to the tail. Coincident abscissae with *unequal*
    # values are not envelope kinks (those are inserted later, with equal
    # value): the lower copies are dominated and collapsed away here — NaN-ed
    # in place — before the scan, so a dominated duplicate can never reach the
    # output and corrupt the index-keyed interpolation of a kink.
    grid_key = jnp.where(jnp.isnan(endog_grid), jnp.inf, endog_grid)
    value_key = jnp.where(jnp.isnan(value), -jnp.inf, value)
    order = jnp.lexsort(jnp.stack([-value_key, grid_key]))
    grid_sorted = endog_grid[order]
    policy_sorted = policy[order]
    value_sorted = value[order]
    n_input = grid_sorted.shape[0]

    duplicate = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=bool),
            (grid_sorted[1:] == grid_sorted[:-1]) & ~jnp.isnan(grid_sorted[1:]),
        ]
    )
    grid_sorted = jnp.where(duplicate, jnp.nan, grid_sorted)
    policy_sorted = jnp.where(duplicate, jnp.nan, policy_sorted)
    value_sorted = jnp.where(duplicate, jnp.nan, value_sorted)

    # All-zero labels reduce the segment test to a no-op (`seg != seg` is always
    # false, `seg == seg` always true), so the `segment_id is None` path is
    # bit-identical to the policy-jump-only scan.
    segment_sorted = (
        jnp.zeros_like(grid_sorted)
        if segment_id is None
        else segment_id[order].astype(grid_sorted.dtype)
    )

    first_point = jnp.stack([grid_sorted[0], policy_sorted[0], value_sorted[0]])
    first_segment = segment_sorted[0]
    # Carry layout: the two most recent envelope points, k then j, each as
    # (grid, policy, value), then their segment labels (seg_k, seg_j). Initially
    # both points are the first sorted candidate.
    carry_init = jnp.concatenate(
        [first_point, first_point, jnp.stack([first_segment, first_segment])]
    )

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
            segment_sorted=segment_sorted,
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


def refine_to_bracket(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    x_query: ScalarFloat,
    jump_thresh: float = 2.0,
    n_points_to_scan: int = 10,
    segment_id: Float1D | None = None,
) -> QueryBracket:
    """Refine to the two envelope nodes bracketing a single query, streaming.

    Geometry-only counterpart of `refine_envelope` for the asset-row publish,
    where the refined envelope is intra-node scratch read at exactly one query
    (`resources_at_node`). Runs the same FUES scan (reusing `_inspect_candidate`
    verbatim — identical drop/dominance/kink decisions), but folds each step's
    up-to-three emitted points into an O(1) bracket-capture carry instead of
    scattering them into NaN-padded `n_pad` rows. The `[n_input, 3]` per-step
    stack and the `[n_pad]` envelope never materialize, so the per-(combo, node)
    envelope working set is O(1) rather than O(n_pad).

    The captured bracket reproduces `searchsorted(search_grid, x_query,
    side="right")` clamped to `[1, max(n_kept - 1, 1)]` on the full refined
    row, node-for-node:

    - The bracket's lower node is the latest emitted point with grid
      $\\le$ `x_query`; the upper node is the first emitted point with grid
      $>$ `x_query`. At a duplicated kink abscissa (left then right copy, same
      abscissa) both copies have grid $\\le$ `x_query` when the query sits on
      the kink, so the later-emitted right copy wins the lower slot — the
      `side="right"` tie-break.
    - A query below the first node brackets the first pair (first, second);
      below-first weight 0 then publishes the first node's value.
    - A query at or above the last node brackets the last pair (second-last,
      last); above-last weight 1 then publishes the last node's value.

    This computes envelope geometry only — no utility, borrowing limit, or
    constrained floor; `publish_node_from_bracket` owns that EGM economics.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        x_query: The single point at which the envelope is read.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ above which two
            points lie on different value-function segments (see
            `refine_envelope`).
        n_points_to_scan: Number of candidates the bounded scans inspect.
        segment_id: Optional per-candidate segment labels (see
            `refine_envelope`).

    Returns:
        The query bracket and the kept-point count.

    """
    grid_key = jnp.where(jnp.isnan(endog_grid), jnp.inf, endog_grid)
    value_key = jnp.where(jnp.isnan(value), -jnp.inf, value)
    order = jnp.lexsort(jnp.stack([-value_key, grid_key]))
    grid_sorted = endog_grid[order]
    policy_sorted = policy[order]
    value_sorted = value[order]
    n_input = grid_sorted.shape[0]

    duplicate = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=bool),
            (grid_sorted[1:] == grid_sorted[:-1]) & ~jnp.isnan(grid_sorted[1:]),
        ]
    )
    grid_sorted = jnp.where(duplicate, jnp.nan, grid_sorted)
    policy_sorted = jnp.where(duplicate, jnp.nan, policy_sorted)
    value_sorted = jnp.where(duplicate, jnp.nan, value_sorted)

    segment_sorted = (
        jnp.zeros_like(grid_sorted)
        if segment_id is None
        else segment_id[order].astype(grid_sorted.dtype)
    )

    first_point = jnp.stack([grid_sorted[0], policy_sorted[0], value_sorted[0]])
    first_segment = segment_sorted[0]
    fues_carry_init = jnp.concatenate(
        [first_point, first_point, jnp.stack([first_segment, first_segment])]
    )
    bracket_carry_init = _empty_bracket_carry(dtype=grid_sorted.dtype)

    def step(
        carry: tuple[Float1D, Float1D], idx: ScalarInt
    ) -> tuple[tuple[Float1D, Float1D], None]:
        """Inspect candidate `idx`, then fold its emissions into the bracket."""
        fues_carry, bracket_carry = carry
        fues_carry_new, (row_grid, row_policy, row_value, count) = _inspect_candidate(
            carry=fues_carry,
            idx=idx,
            grid_sorted=grid_sorted,
            policy_sorted=policy_sorted,
            value_sorted=value_sorted,
            segment_sorted=segment_sorted,
            jump_thresh=jump_thresh,
            n_points_to_scan=n_points_to_scan,
        )
        bracket_carry_new = _fold_emissions_into_bracket(
            bracket_carry=bracket_carry,
            row_grid=row_grid,
            row_policy=row_policy,
            row_value=row_value,
            count=count,
            x_query=x_query,
        )
        return (fues_carry_new, bracket_carry_new), None

    indices = jnp.arange(1, n_input, dtype=jnp.int32)
    (fues_carry_final, bracket_carry), _ = jax.lax.scan(
        step, (fues_carry_init, bracket_carry_init), indices
    )

    # The last accepted point is still pending in the FUES carry; fold it as the
    # scan's final single emission (mirrors `refine_envelope`'s post-scan emit).
    final_grid, final_policy, final_value = (
        fues_carry_final[3],
        fues_carry_final[4],
        fues_carry_final[5],
    )
    final_valid = ~jnp.isnan(final_grid) & ~jnp.isnan(final_value)
    nan_scalar = jnp.full((), jnp.nan, dtype=grid_sorted.dtype)
    bracket_carry = _fold_emissions_into_bracket(
        bracket_carry=bracket_carry,
        row_grid=jnp.stack([final_grid, nan_scalar, nan_scalar]),
        row_policy=jnp.stack([final_policy, nan_scalar, nan_scalar]),
        row_value=jnp.stack([final_value, nan_scalar, nan_scalar]),
        count=final_valid.astype(jnp.int32),
        x_query=x_query,
    )

    return _assemble_bracket(bracket_carry=bracket_carry, dtype=grid_sorted.dtype)


# Bracket-capture carry layout (flat Float1D so it threads through `lax.scan`):
# six (grid, policy, value) triples, then three counters.
#   first  (0:3)   — lowest emitted node (constrained-floor edge test, below clamp)
#   second (3:6)   — second emitted node (below-first clamp upper)
#   prev   (6:9)   — second-most-recent emitted node (above-last clamp lower)
#   last   (9:12)  — most-recent emitted node (above-last clamp upper)
#   lo     (12:15) — latest emitted node with grid <= x_query (interior lower)
#   hi     (15:18) — first emitted node with grid > x_query (interior upper)
#   seen   (18)    — running count of emitted nodes (for first/second/n_kept)
#   below  (19)    — running count of emitted nodes with grid <= x_query (`s`)
#   hi_set (20)    — 1.0 once `hi` has been frozen, else 0.0
_BRACKET_CARRY_LEN = 21


def _empty_bracket_carry(*, dtype: jnp.dtype) -> Float1D:
    """Initialize the bracket-capture carry to all-NaN nodes and zero counters."""
    carry = jnp.full((_BRACKET_CARRY_LEN,), jnp.nan, dtype=dtype)
    return carry.at[18:21].set(jnp.zeros((3,), dtype=dtype))


def _fold_emissions_into_bracket(
    *,
    bracket_carry: Float1D,
    row_grid: Float1D,
    row_policy: Float1D,
    row_value: Float1D,
    count: ScalarInt,
    x_query: ScalarFloat,
) -> Float1D:
    """Fold a step's up-to-three emitted nodes into the bracket-capture carry.

    Emissions arrive in ascending grid order; only the first `count` of the
    three slots are valid. Each valid node updates, in order:

    - `seen` and the `first`/`second` registers (the first two ever emitted),
    - the rolling `prev`/`last` pair (the two most recent),
    - on grid $\\le$ `x_query`: the `below` count and the `lo` register
      (latest-wins, so a duplicated kink's right copy wins),
    - on grid $>$ `x_query`: the `hi` register, frozen on first occurrence.
    """
    for slot in range(3):
        valid = slot < count
        bracket_carry = _fold_one_node(
            bracket_carry=bracket_carry,
            grid=row_grid[slot],
            policy=row_policy[slot],
            value=row_value[slot],
            valid=valid,
            x_query=x_query,
        )
    return bracket_carry


def _fold_one_node(
    *,
    bracket_carry: Float1D,
    grid: ScalarFloat,
    policy: ScalarFloat,
    value: ScalarFloat,
    valid: ScalarBool,
    x_query: ScalarFloat,
) -> Float1D:
    """Update the bracket-capture carry with one emitted node when `valid`."""
    node = jnp.stack([grid, policy, value])
    seen = bracket_carry[18]
    below = bracket_carry[19]
    hi_set = bracket_carry[20]

    is_first = valid & (seen == 0.0)
    is_second = valid & (seen == 1.0)
    at_or_below = valid & (grid <= x_query)
    above = valid & (grid > x_query)
    freezes_hi = above & (hi_set == 0.0)

    new = bracket_carry
    new = new.at[0:3].set(jnp.where(is_first, node, bracket_carry[0:3]))
    new = new.at[3:6].set(jnp.where(is_second, node, bracket_carry[3:6]))
    # Roll the most-recent pair: `prev` takes the old `last`, `last` takes the
    # node — only on a valid emission, so dropped/NaN slots leave them intact.
    new = new.at[6:9].set(jnp.where(valid, bracket_carry[9:12], bracket_carry[6:9]))
    new = new.at[9:12].set(jnp.where(valid, node, bracket_carry[9:12]))
    new = new.at[12:15].set(jnp.where(at_or_below, node, bracket_carry[12:15]))
    new = new.at[15:18].set(jnp.where(freezes_hi, node, bracket_carry[15:18]))
    new = new.at[18].set(seen + jnp.where(valid, 1.0, 0.0))
    new = new.at[19].set(below + jnp.where(at_or_below, 1.0, 0.0))
    return new.at[20].set(jnp.where(freezes_hi, 1.0, hi_set))


def _assemble_bracket(*, bracket_carry: Float1D, dtype: jnp.dtype) -> QueryBracket:
    """Assemble the edge-clamped query bracket from the capture carry.

    Reproduces `clip(searchsorted(side="right"), 1, max(n_kept - 1, 1))` on the
    full refined row by selecting among the captured registers on `s` (count of
    emitted nodes with grid $\\le$ `x_query`) and `n_kept`:

    - `s == 0` (query below the first node): bracket (first, second).
    - `s == n_kept` (query at or above the last node): bracket (prev, last) when
      `n_kept >= 2`; with a single live node the reference clamps the upper
      index to the NaN-padded slot, so the bracket is (first, second=NaN) there.
    - otherwise: bracket (lo, hi) — the searchsorted pair node-for-node.
    """
    first = bracket_carry[0:3]
    second = bracket_carry[3:6]
    prev = bracket_carry[6:9]
    last = bracket_carry[9:12]
    lo = bracket_carry[12:15]
    hi = bracket_carry[15:18]
    n_kept = bracket_carry[18].astype(jnp.int32)
    below = bracket_carry[19].astype(jnp.int32)

    below_first = below == 0
    at_or_above_last = below == n_kept
    # The reference clamps the upper index to `max(n_kept - 1, 1)`, so the
    # above-last bracket is (second-last, last) once there are at least two
    # nodes (`max(n_kept - 1, 1) == n_kept - 1`); with a single node the upper
    # index stays at the NaN-padded slot, so the bracket is the (first,
    # NaN-padded-second) pair that `first`/`second` already hold (`second`
    # stayed NaN, never emitted).
    above_clamp_is_last = (n_kept - 1) >= jnp.maximum(n_kept - 1, 1)
    above_lower = jnp.where(above_clamp_is_last, prev, first)
    above_upper = jnp.where(above_clamp_is_last, last, second)

    lower = jnp.where(below_first, first, jnp.where(at_or_above_last, above_lower, lo))
    upper = jnp.where(below_first, second, jnp.where(at_or_above_last, above_upper, hi))
    return QueryBracket(
        lower_grid=lower[0].astype(dtype),
        upper_grid=upper[0].astype(dtype),
        lower_policy=lower[1].astype(dtype),
        upper_policy=upper[1].astype(dtype),
        lower_value=lower[2].astype(dtype),
        upper_value=upper[2].astype(dtype),
        first_grid=first[0].astype(dtype),
        n_kept=n_kept,
    )


def _inspect_candidate(
    *,
    carry: Float1D,
    idx: ScalarInt,
    grid_sorted: Float1D,
    policy_sorted: Float1D,
    value_sorted: Float1D,
    segment_sorted: Float1D,
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
    grid_k, policy_k, value_k, grid_j, policy_j, value_j, seg_k, seg_j = carry
    grid_i = grid_sorted[idx]
    policy_i = policy_sorted[idx]
    value_i = value_sorted[idx]
    seg_i = segment_sorted[idx]

    candidate_valid = ~jnp.isnan(grid_i) & ~jnp.isnan(value_i)
    # Two points lie on different segments if the implied-savings policy jumps
    # or — when labels are supplied — their segment labels differ.
    switches = _has_policy_jump(
        grid_a=grid_j,
        policy_a=policy_j,
        grid_b=grid_i,
        policy_b=policy_i,
        jump_thresh=jump_thresh,
    ) | (seg_j != seg_i)
    secant = _slope(x_a=grid_j, y_a=value_j, x_b=grid_i, y_b=value_i)
    grad_before = _slope(x_a=grid_k, y_a=value_k, x_b=grid_j, y_b=value_j)

    # Next candidate on j's segment after i: defines the continuation of the
    # current segment for both the suboptimality test and the lower line of a
    # crossing between j and i.
    j_seg_found, j_seg_grid, j_seg_policy, j_seg_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        segment=segment_sorted,
        anchor_grid=grid_j,
        anchor_policy=policy_j,
        anchor_segment=seg_j,
        idx=idx,
        direction=1,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    secant_i_to_j_seg = _slope(x_a=grid_i, y_a=value_i, x_b=j_seg_grid, y_b=j_seg_value)
    below_j_segment = switches & j_seg_found & (secant < secant_i_to_j_seg)

    # A value drop marks a dominated candidate only *within* a segment, where the
    # envelope value is monotone in the grid. Across a segment switch the value may
    # legitimately fall (the winning segment changes at a crossing), so a switch is
    # judged geometrically by `below_j_segment`, not by the raw value comparison —
    # otherwise the genuine crossing point of two branches is dropped as dominated.
    dropped = (
        ~candidate_valid
        | ((value_i < value_j) & ~switches)
        | (((grid_i - policy_i) < (grid_j - policy_j)) & (secant < grad_before))
        | below_j_segment
    )

    # A same-segment partner of i defines i's segment line (forward preferred:
    # after a crossing, i's segment continues to the right).
    fwd_found, fwd_grid, fwd_policy, fwd_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        segment=segment_sorted,
        anchor_grid=grid_i,
        anchor_policy=policy_i,
        anchor_segment=seg_i,
        idx=idx,
        direction=1,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )
    bwd_found, bwd_grid, bwd_policy, bwd_value = _find_same_segment_point(
        grid=grid_sorted,
        policy=policy_sorted,
        value=value_sorted,
        segment=segment_sorted,
        anchor_grid=grid_i,
        anchor_policy=policy_i,
        anchor_segment=seg_i,
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
    # The j-dominated crossing happened before j, so the inserted kink lies in
    # `(grid_k, grid_j)`. The `< grid_j` bound is implied by the `secant >
    # i_seg_slope` precondition in exact arithmetic; it is kept explicit so a
    # near-parallel-slope rounding error cannot place the kink past j.
    kink_6 = (
        j_dominated
        & (kink_grid_6 > grid_k)
        & (kink_grid_6 < grid_j)
        & (kink_grid_6 < grid_i)
    )

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
    # New k inherits the segment of the point it represents: an inserted kink
    # carries `i`'s segment (the crossing continues to the right on i's
    # segment), a plain accept carries the finalized `j`'s segment, and a
    # retained k keeps its own. New j is the accepted candidate `i`.
    new_k_seg = jnp.where(keeps_k, seg_k, jnp.where(kink_5 | kink_6, seg_i, seg_j))
    carry_accepted = jnp.stack(
        [
            new_k_grid,
            new_k_policy,
            new_k_value,
            grid_i,
            policy_i,
            value_i,
            new_k_seg,
            seg_i,
        ]
    )
    carry_new = jnp.where(dropped, carry, carry_accepted)

    return carry_new, (row_grid, row_policy, row_value, count)


def _find_same_segment_point(
    *,
    grid: Float1D,
    policy: Float1D,
    value: Float1D,
    segment: Float1D,
    anchor_grid: ScalarFloat,
    anchor_policy: ScalarFloat,
    anchor_segment: ScalarFloat,
    idx: ScalarInt,
    direction: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> tuple[ScalarBool, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Find the candidate nearest to `idx` on the anchor's value segment.

    Inspects up to `n_points_to_scan` candidates after (`direction=1`) or
    before (`direction=-1`) index `idx` and returns the first whose implied
    savings, relative to the anchor point, do not jump and whose segment label
    matches the anchor's.

    Args:
        grid: Sorted candidate endogenous grid points.
        policy: Candidate policy values at `grid`.
        value: Candidate value-correspondence points at `grid`.
        segment: Candidate segment labels at `grid` (all-zero when unused).
        anchor_grid: Endogenous grid point of the anchor.
        anchor_policy: Policy value of the anchor.
        anchor_segment: Segment label of the anchor.
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
    window_segment = segment[clipped]
    same_segment = (
        in_bounds
        & ~jnp.isnan(window_grid)
        & ~jnp.isnan(window_value)
        & (window_segment == anchor_segment)
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
    # Evaluate the ordinate from a finite abscissa so the parallel-lines branch
    # carries a finite (dead) value: NaN here would poison reverse-mode gradients
    # through the `jnp.where` even though the forward result discards it.
    safe_x = jnp.where(denominator == 0.0, x_a, x)
    y = y_a + slope_a * (safe_x - x_a)
    return x, y
