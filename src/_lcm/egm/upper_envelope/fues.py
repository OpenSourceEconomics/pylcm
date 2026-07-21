"""Fast Upper-Envelope Scan (FUES) over EGM candidate solutions.

Implements the upper-envelope refinement of Dobrescu, L. I., & Shanker, A.
(2022). Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming.
SSRN 4181302.

Adapted from the `OpenSourceEconomics/upper-envelope` package (Apache-2.0,
© The Upper-Envelope Authors) and substantially modified for pylcm's JAX kernel.

Inverting the Euler equation in models with discrete choices yields a value
*correspondence*: in non-concave regions, several candidate points share a
neighborhood of the endogenous grid, each lying on a different choice-specific
value segment. `refine_envelope` sorts the candidates ascending in grid, reduces
each coincident-abscissa group to its envelope-relevant points, then scans once
and keeps only the points on the upper envelope:

- Coincident-abscissa groups are reduced first (`_reduce_coincident_groups`):
  within a run of equal grid values only the group's value-maximizers survive
  (exact max-equality, so translation-invariant), and same-source maximizers
  collapse to one copy. A node-aligned crossing — two maximizers at one abscissa
  with equal value but different branch — therefore keeps both branch policies.
- Candidates on a different segment than the last kept point are detected via
  the implied savings $A = R - c$: the segments differ iff
  $|\\Delta A / \\Delta R|$ exceeds `jump_thresh`, or (at a coincident abscissa,
  an infinite gradient) whenever the implied savings differ.
- Dominated candidates are dropped (value decreases within a segment, savings
  non-monotone with a falling value gradient, or the candidate lies below the
  continuation of the current segment found by a bounded forward scan). A
  node-aligned crossing is exempt — it is an envelope kink, never interior-
  dominated.
- Where two kept segments cross, the intersection is emitted inline during the
  scan as two points at one abscissa — the left- and right-extrapolated policy —
  so the refined arrays stay weakly ascending and the policy discontinuity at the
  kink is preserved exactly.

`refine_to_bracket` builds the same refined row and slices the two nodes
bracketing a single query, so the streamed asset-row read agrees with the full
row by construction (no separate streaming geometry to keep in sync).

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


class QueryBracket(NamedTuple):
    """The two envelope nodes bracketing one query, plus the publish statics.

    A single refined envelope row read at one query: `refine_to_bracket` builds
    the full refined row (`refine_envelope`) and locates the bracket with
    `searchsorted(side="right")`, returning only the bracketing pair. The
    arithmetic on the pair is identical to the row-then-interpolate path — it
    reads the same row — so the published value agrees by construction.

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


def _resolve_n_points_to_scan(n_points_to_scan: int | None, *, n_input: int) -> int:
    """Resolve the exhaustive-scan sentinel to a concrete window width.

    `None` requests an exhaustive scan — every other candidate. That is the only
    width proven correct when more than the window's worth of off-segment
    candidates interleave between two points of one segment: a bounded window
    then never reaches the segment's continuation and silently accepts the
    interlopers. A finite value keeps the cheaper $O(\\text{width})$ scan at the
    cost of that guarantee. `n_input` is a static shape, so the result stays a
    Python int and the downstream `jnp.arange(n_points_to_scan)` stays
    static-shape.
    """
    if n_points_to_scan is None:
        return max(n_input - 1, 1)
    return n_points_to_scan


def _init_fues_carry(
    *, first_point: Float1D, first_segment: ScalarFloat, first_savings: ScalarFloat
) -> Float1D:
    """Seed the FUES scan carry: k and j both the first sorted candidate.

    Carry layout (flat `Float1D` so it threads through `jax.lax.scan`): the two
    most recent envelope points, k then j, each as (grid, policy, value); their
    segment labels (seg_k, seg_j); and the exogenous source savings of j
    (`savings_j`, ignored on the noise-floor path). Initially k and j are the
    first sorted candidate.
    """
    return jnp.concatenate(
        [
            first_point,
            first_point,
            jnp.stack([first_segment, first_segment, first_savings]),
        ]
    )


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    jump_thresh: float = 2.0,
    n_points_to_scan: int | None = None,
    segment_id: Float1D | None = None,
    savings: Float1D | None = None,
    scan_unroll: int = 1,
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
            segment. `None` (the default) scans exhaustively — every other
            candidate — which is the only width proven correct when more than
            the window's worth of off-segment candidates interleave between two
            points of one segment (a bounded window silently accepts the
            interlopers). A finite value keeps the cheaper $O(\\text{width})$
            scan at the cost of that guarantee.
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
        savings: Optional per-candidate exogenous source savings, aligned with
            `endog_grid` — the exogenous savings grid point that generated each
            candidate (equal to the implied savings `R - c` in exact arithmetic).
            When supplied, the savings-monotonicity dominance clause compares
            these true sources directly (`s_i < s_j` is a genuine decrease; equal
            sources are ties, never dropped), which is exact and backend-stable.
            `None` (the default) falls back to the magnitude-scaled noise floor on
            the implied difference, which protects same-source ties against
            rounding but can mask a real cross-source decrease when the resources
            dwarf the savings span.
        scan_unroll: Loop-unroll factor for the sequential `jax.lax.scan` over
            candidates. Unrolling trades compile time for fewer loop-carry round
            trips on accelerators; the refined output is identical across values.

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
    # so the maximal-value candidate leads each run of coincident abscissae, and
    # finally by descending policy so a *node crossing* (two branches meeting at
    # one abscissa with equal value) is ordered left-owner-then-right-owner
    # independently of input order: under the EGM envelope condition the value
    # slope is `u'(c)`, decreasing in consumption, so the higher-policy branch has
    # the shallower slope and owns the interval just left of the node. NaN grid
    # points sort to the tail. Coincident abscissae with *unequal* values are not
    # envelope kinks (those are inserted later, with equal value): the lower
    # copies are dominated and collapsed away here — NaN-ed in place — before the
    # scan, so a dominated duplicate can never reach the output and corrupt the
    # index-keyed interpolation of a kink.
    grid_key = jnp.where(jnp.isnan(endog_grid), jnp.inf, endog_grid)
    value_key = jnp.where(jnp.isnan(value), -jnp.inf, value)
    policy_key = jnp.where(jnp.isnan(policy), -jnp.inf, policy)
    order = jnp.lexsort(jnp.stack([-policy_key, -value_key, grid_key]))
    grid_sorted = endog_grid[order]
    policy_sorted = policy[order]
    value_sorted = value[order]
    n_input = grid_sorted.shape[0]
    n_points_to_scan = _resolve_n_points_to_scan(n_points_to_scan, n_input=n_input)

    # All-zero labels reduce the segment test to a no-op (`seg != seg` is always
    # false, `seg == seg` always true), so the `segment_id is None` path is
    # bit-identical to the policy-jump-only scan.
    segment_sorted = (
        jnp.zeros_like(grid_sorted)
        if segment_id is None
        else segment_id[order].astype(grid_sorted.dtype)
    )
    use_savings = savings is not None
    savings_sorted = (
        savings[order].astype(grid_sorted.dtype)
        if use_savings
        else jnp.zeros_like(grid_sorted)
    )

    grid_sorted, policy_sorted, value_sorted, savings_sorted = (
        _reduce_coincident_groups(
            grid_sorted=grid_sorted,
            policy_sorted=policy_sorted,
            value_sorted=value_sorted,
            savings_sorted=savings_sorted,
            use_savings=use_savings,
        )
    )

    first_point = jnp.stack([grid_sorted[0], policy_sorted[0], value_sorted[0]])
    first_segment = segment_sorted[0]
    carry_init = _init_fues_carry(
        first_point=first_point,
        first_segment=first_segment,
        first_savings=savings_sorted[0],
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
            savings_sorted=savings_sorted,
            use_savings=use_savings,
            jump_thresh=jump_thresh,
            n_points_to_scan=n_points_to_scan,
        )

    indices = jnp.arange(1, n_input, dtype=jnp.int32)
    carry_final, (block_grid, block_policy, block_value, block_count) = jax.lax.scan(
        step, carry_init, indices, unroll=scan_unroll
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


def _reduce_coincident_groups(
    *,
    grid_sorted: Float1D,
    policy_sorted: Float1D,
    value_sorted: Float1D,
    savings_sorted: Float1D,
    use_savings: bool,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Reduce each coincident-abscissa group to its envelope-relevant maximizers.

    Replaces the destructive coincident-node dedup. Within each maximal run of
    equal finite abscissae (the arrays arrive sorted ascending in grid):

    - keep only members whose value equals the group maximum. The test is exact
      equality against the segment maximum, so it is invariant to a common value
      shift `value -> value + K` (a relative `rtol * |value|` band would widen at
      large value level and manufacture kinks). A dominated member — value below
      the group max by any representable margin — is NaN-ed. Two branches that
      cross exactly on the node share one value and both survive.
    - collapse maximizers that share an exogenous savings source (or, without
      provenance, an equal policy) to a single copy, so a true duplicate is one
      point while a genuine branch switch keeps both branch policies. The
      collapse is group-wide — a maximizer is dropped when its source already
      appears on any earlier maximizer of the group — so same-source copies split
      by a different source (an A, B, A run) still reduce to A, B.

    Points equal in exact arithmetic but split by a single ULP of construction
    rounding are treated as one maximizer and its off-by-one-ULP shadow; the
    shadow is dropped, publishing one branch's policy across the node — a benign
    ULP-level degradation, and translation-invariant.

    Known limitation (pointwise vs interval dominance). Group membership is
    decided by the value *at the node*, not by segment ownership on the adjacent
    interval. When two branches are both sampled at the *same* pair of abscissae
    and swap ownership strictly between them, the branch that is pointwise-lower
    at each shared node is dropped here, losing its slope anchor, so the scan
    bridges the surviving node maxima instead of reconstructing the true crossing.
    A one-sided interval-ownership reducer would be needed to close this. The
    trigger requires exact endogenous-grid coincidence across branches, which the
    production models do not exhibit (their oracle gates hold); see
    `test_pointwise_lower_branch_that_owns_an_interval_is_retained` (xfail).
    """
    n = grid_sorted.shape[0]
    finite = ~jnp.isnan(grid_sorted)
    new_group = jnp.concatenate(
        [
            jnp.ones((1,), dtype=bool),
            (grid_sorted[1:] != grid_sorted[:-1]) & finite[1:],
        ]
    )
    group_id = jnp.cumsum(new_group) - 1

    value_or_neg = jnp.where(finite, value_sorted, -jnp.inf)
    group_max = jax.ops.segment_max(value_or_neg, group_id, num_segments=n)[group_id]
    is_max = finite & (value_sorted == group_max)

    # Collapse a maximizer whose exogenous source already appears on an earlier
    # maximizer of the same group. The test is group-wide, not only against the
    # immediate predecessor, so an A, B, A run — two same-source copies split by a
    # different source — keeps one A and the B rather than all three.
    source = savings_sorted if use_savings else policy_sorted
    idx = jnp.arange(n)
    earlier_same_source_max = (
        (group_id[:, None] == group_id[None, :])
        & (source[:, None] == source[None, :])
        & (idx[None, :] < idx[:, None])
        & is_max[None, :]
    )
    collapse = is_max & jnp.any(earlier_same_source_max, axis=1)

    keep = finite & is_max & ~collapse
    grid_out = jnp.where(keep, grid_sorted, jnp.nan)
    policy_out = jnp.where(keep, policy_sorted, jnp.nan)
    value_out = jnp.where(keep, value_sorted, jnp.nan)
    savings_out = jnp.where(keep, savings_sorted, jnp.nan)
    return grid_out, policy_out, value_out, savings_out


def refine_to_bracket(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    x_query: ScalarFloat,
    n_refined: int,
    jump_thresh: float = 2.0,
    n_points_to_scan: int | None = None,
    segment_id: Float1D | None = None,
    savings: Float1D | None = None,
    scan_unroll: int = 1,
) -> QueryBracket:
    """Refine to the two envelope nodes bracketing a single query.

    Counterpart of `refine_envelope` for the asset-row publish, where the
    refined envelope is intra-node scratch read at exactly one query
    (`resources_at_node`). Builds the full refined row via `refine_envelope`
    (identical drop/dominance/kink decisions and node-aligned crossings) and
    slices the bracketing node pair, so the streamed publish agrees with the
    row-then-interpolate publish by construction — there is no separate
    streaming geometry to keep in sync with the full row.

    The bracket is `searchsorted(search_grid, x_query, side="right")` clamped to
    `[1, max(n_kept - 1, 1)]` on the full refined row:

    - The bracket's lower node is the latest row node with grid $\\le$ `x_query`;
      the upper node is the first row node with grid $>$ `x_query`. At a
      duplicated kink abscissa (left then right copy, same abscissa) the query on
      the kink brackets the right copy as its lower node — the `side="right"`
      tie-break.
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
        n_refined: Static length of the full refined row that is built and
            sliced (see `refine_envelope`).
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ above which two
            points lie on different value-function segments (see
            `refine_envelope`).
        n_points_to_scan: Number of candidates the bounded scans inspect; `None`
            (the default) scans exhaustively (see `refine_envelope`).
        segment_id: Optional per-candidate segment labels, forwarded to
            `refine_envelope`. No production caller passes labels.
        savings: Optional per-candidate exogenous source savings (see
            `refine_envelope`); the savings-monotonicity clause compares true
            sources when supplied, else the noise floor.
        scan_unroll: Loop-unroll factor for the sequential `jax.lax.scan` over
            candidates (see `refine_envelope`); the bracket is identical across
            values.

    Returns:
        The query bracket and the kept-point count.

    """
    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        jump_thresh=jump_thresh,
        n_points_to_scan=n_points_to_scan,
        segment_id=segment_id,
        savings=savings,
        scan_unroll=scan_unroll,
    )

    # Locate the query bracket exactly as the dense read does:
    # `searchsorted(side="right")` clamped to `[1, max(n_kept - 1, 1)]`. NaN-padded
    # tail entries become `+inf` so they never bracket a finite query, and a query
    # on a duplicated kink abscissa lands the right copy in the lower slot.
    search_grid = jnp.where(jnp.isnan(refined_grid), jnp.inf, refined_grid)
    n_valid = jnp.minimum(n_kept, n_refined)
    upper_idx = jnp.searchsorted(search_grid, x_query, side="right")
    upper_idx = jnp.clip(upper_idx, 1, jnp.maximum(n_valid - 1, 1))
    lower_idx = upper_idx - 1
    return QueryBracket(
        lower_grid=refined_grid[lower_idx],
        upper_grid=refined_grid[upper_idx],
        lower_policy=refined_policy[lower_idx],
        upper_policy=refined_policy[upper_idx],
        lower_value=refined_value[lower_idx],
        upper_value=refined_value[upper_idx],
        first_grid=refined_grid[0],
        n_kept=n_kept,
    )


def _judge_savings_decrease(
    *,
    use_savings: bool,
    savings_i: ScalarFloat,
    savings_j: ScalarFloat,
    grid_i: ScalarFloat,
    policy_i: ScalarFloat,
    grid_j: ScalarFloat,
    policy_j: ScalarFloat,
) -> ScalarBool:
    """Judge a savings decrease between two candidates.

    With exogenous source savings the decrease is exact and backend-stable
    (`s_i < s_j`; equal sources are ties, never dropped). Without them, the
    noise floor on the implied difference `R - c` protects same-source ties from
    rounding at the cost of masking a real cross-source decrease when resources
    dwarf the savings span.
    """
    if use_savings:
        return savings_i < savings_j
    return _savings_decrease_past_noise(
        grid_i=grid_i, policy_i=policy_i, grid_j=grid_j, policy_j=policy_j
    )


def _emission_block(
    *,
    plain: ScalarBool,
    kink_5: ScalarBool,
    kink_6: ScalarBool,
    kink_on_j: ScalarBool,
    kink_on_i: ScalarBool,
    grid_j: ScalarFloat,
    grid_i: ScalarFloat,
    policy_j: ScalarFloat,
    value_j: ScalarFloat,
    kink_grid_5: ScalarFloat,
    kink_policy_left_5: ScalarFloat,
    kink_policy_right_5: ScalarFloat,
    kink_value_5: ScalarFloat,
    kink_grid_6: ScalarFloat,
    kink_policy_left_6: ScalarFloat,
    kink_policy_right_6: ScalarFloat,
    kink_value_6: ScalarFloat,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Assemble a scan step's up-to-three emission rows in ascending grid order.

    A step emits, at most:

    - a `j`-dominated crossing kink (`kink_6`, two copies — left then right policy
      at one abscissa; `j` itself is dominated, so not emitted),
    - the finalized `j` (`plain`, or when a crossing also fires),
    - a strictly-between crossing kink (`kink_5`, two more copies at the crossing),
    - a crossing snapped to an endpoint: on `grid_j` (`kink_on_j`, one right copy
      at `grid_j` — `j` is the left copy) or on `grid_i` (`kink_on_i`, one left
      copy at `grid_i` — the accepted `i` is the right copy, emitted next step).

    The flags are mutually exclusive by construction, so the three slots pack the
    valid emissions densely and `count` is the number of filled slots.
    """
    nan_scalar = jnp.full((), jnp.nan, dtype=grid_j.dtype)
    emits_j = plain | kink_5 | kink_on_j | kink_on_i
    crossing_value = kink_5 | kink_on_j | kink_on_i
    row_grid = jnp.stack(
        [
            jnp.where(kink_6, kink_grid_6, jnp.where(emits_j, grid_j, nan_scalar)),
            jnp.where(
                kink_6,
                kink_grid_6,
                jnp.where(
                    kink_5,
                    kink_grid_5,
                    jnp.where(
                        kink_on_j, grid_j, jnp.where(kink_on_i, grid_i, nan_scalar)
                    ),
                ),
            ),
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
                jnp.where(
                    kink_5 | kink_on_i,
                    kink_policy_left_5,
                    jnp.where(kink_on_j, kink_policy_right_5, nan_scalar),
                ),
            ),
            jnp.where(kink_5, kink_policy_right_5, nan_scalar),
        ]
    )
    row_value = jnp.stack(
        [
            jnp.where(kink_6, kink_value_6, jnp.where(emits_j, value_j, nan_scalar)),
            jnp.where(
                kink_6,
                kink_value_6,
                jnp.where(crossing_value, kink_value_5, nan_scalar),
            ),
            jnp.where(kink_5, kink_value_5, nan_scalar),
        ]
    )
    count = jnp.where(
        kink_5, 3, jnp.where(kink_6 | kink_on_j | kink_on_i, 2, jnp.where(plain, 1, 0))
    ).astype(jnp.int32)
    return row_grid, row_policy, row_value, count


def _inspect_candidate(
    *,
    carry: Float1D,
    idx: ScalarInt,
    grid_sorted: Float1D,
    policy_sorted: Float1D,
    value_sorted: Float1D,
    segment_sorted: Float1D,
    savings_sorted: Float1D,
    use_savings: bool,
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
        segment_sorted: Sorted candidate segment labels.
        savings_sorted: Sorted candidate exogenous source savings (dummy zeros
            when `use_savings` is false).
        use_savings: When true, the savings-monotonicity clause compares
            exogenous sources; otherwise it uses the noise-floor heuristic.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$ above which two
            points lie on different segments.
        n_points_to_scan: Number of candidates the bounded scans inspect.

    Returns:
        Tuple of the updated carry and the per-step output block: three
        emission rows for grid, policy, and value, plus the number of valid
        rows.

    """
    grid_k, policy_k, value_k, grid_j, policy_j, value_j, seg_k, seg_j, savings_j = (
        carry
    )
    grid_i = grid_sorted[idx]
    policy_i = policy_sorted[idx]
    value_i = value_sorted[idx]
    seg_i = segment_sorted[idx]
    savings_i = savings_sorted[idx]

    candidate_valid = ~jnp.isnan(grid_i) & ~jnp.isnan(value_i)
    # Two points lie on different segments if the implied-savings policy jumps
    # or — when labels are supplied — their segment labels differ.
    switches = _has_policy_jump(
        grid_a=grid_j,
        policy_a=policy_j,
        grid_b=grid_i,
        policy_b=policy_i,
        jump_thresh=jump_thresh,
        savings_a=savings_j if use_savings else None,
        savings_b=savings_i if use_savings else None,
    ) | (seg_j != seg_i)
    # A node-aligned crossing is a genuine switch landing on a grid node: two
    # maximizers at one abscissa with equal value and different branch. It is an
    # envelope kink, never interior-dominated, so it is exempt from the
    # domination drops below and both copies are emitted (left then right).
    node_crossing = (
        candidate_valid & switches & (grid_i == grid_j) & (value_i == value_j)
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
    # `i` is below `j`'s segment continuation when it lies under that line. If the
    # continuation lands on `i`'s own abscissa (a node-aligned crossing, so the
    # secant slope is degenerate), "below" is the direct value comparison: equal
    # value means the branches meet there and `i` is kept, not dominated.
    below_j_segment = (
        switches
        & j_seg_found
        & jnp.where(
            grid_i == j_seg_grid,
            value_i < j_seg_value,
            secant < secant_i_to_j_seg,
        )
    )

    # A value drop marks a dominated candidate only *within* a segment, where the
    # envelope value is monotone in the grid. Across a segment switch the value may
    # legitimately fall (the winning segment changes at a crossing), so a switch is
    # judged geometrically by `below_j_segment`, not by the raw value comparison —
    # otherwise the genuine crossing point of two branches is dropped as dominated.
    #
    savings_decrease = _judge_savings_decrease(
        use_savings=use_savings,
        savings_i=savings_i,
        savings_j=savings_j,
        grid_i=grid_i,
        policy_i=policy_i,
        grid_j=grid_j,
        policy_j=policy_j,
    )
    dropped = (
        ~candidate_valid
        | ((value_i < value_j) & ~switches)
        | (savings_decrease & (secant < grad_before))
        | below_j_segment
    ) & ~node_crossing

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
    j_dominated = (
        ~dropped & switches & partner_found & (secant > i_seg_slope) & ~node_crossing
    )
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
    # The crossing of j's and i's segment lines can land strictly between them (an
    # interior kink: both nodes kept, the intersection inserted twice), or on an
    # existing endpoint `grid_j` / `grid_i` — a node-aligned crossing where one
    # branch spans the node without a coincident candidate there. An endpoint
    # landing (within rounding) is snapped to that node and emitted as a canonical
    # two-copy kink; the redundant endpoint copy is suppressed so `n_kept` is not
    # inflated. `near_tol` is a few ULP of the node scale — "certified within
    # rounding error", not a tolerance that would swallow a genuine interior kink.
    # A coincident node crossing (`grid_i == grid_j`, handled by `node_crossing`
    # with both copies kept) is excluded: its segment-line intersection is not a
    # separate inserted kink.
    crossing = ~dropped & ~j_dominated & ~node_crossing & switches & partner_found
    near_tol = (
        16.0
        * jnp.finfo(grid_sorted.dtype).eps
        * jnp.maximum(jnp.maximum(jnp.abs(grid_j), jnp.abs(grid_i)), 1.0)
    )
    kink_on_j = crossing & (jnp.abs(kink_grid_5 - grid_j) <= near_tol)
    kink_on_i = crossing & ~kink_on_j & (jnp.abs(kink_grid_5 - grid_i) <= near_tol)
    kink_5 = (
        crossing
        & ~kink_on_j
        & ~kink_on_i
        & (kink_grid_5 > grid_j)
        & (kink_grid_5 < grid_i)
    )

    plain = ~dropped & ~j_dominated & ~kink_5 & ~kink_on_j & ~kink_on_i
    row_grid, row_policy, row_value, count = _emission_block(
        plain=plain,
        kink_5=kink_5,
        kink_6=kink_6,
        kink_on_j=kink_on_j,
        kink_on_i=kink_on_i,
        grid_j=grid_j,
        grid_i=grid_i,
        policy_j=policy_j,
        value_j=value_j,
        kink_grid_5=kink_grid_5,
        kink_policy_left_5=kink_policy_left_5,
        kink_policy_right_5=kink_policy_right_5,
        kink_value_5=kink_value_5,
        kink_grid_6=kink_grid_6,
        kink_policy_left_6=kink_policy_left_6,
        kink_policy_right_6=kink_policy_right_6,
        kink_value_6=kink_value_6,
    )

    # New k is the point emitted last this step: the right-policy copy of an
    # inserted kink, the finalized j on a plain accept, or unchanged when
    # nothing was emitted (drop, or j dominated without a valid kink). For a
    # crossing on `grid_j` the last emitted node is the right copy at `grid_j`;
    # for a crossing on `grid_i` the left copy sits at `grid_i` (coincident with
    # the accepted `i`), so k carries the plain-accept `j` and the left copy is an
    # inserted node only. New j is the accepted candidate i.
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
            jnp.where(
                kink_6,
                kink_policy_right_6,
                jnp.where(kink_on_j, kink_policy_right_5, policy_j),
            ),
        ),
    )
    new_k_value = jnp.where(
        keeps_k,
        value_k,
        jnp.where(
            kink_5,
            kink_value_5,
            jnp.where(
                kink_6,
                kink_value_6,
                jnp.where(kink_on_j, kink_value_5, value_j),
            ),
        ),
    )
    # New k inherits the segment of the point it represents: an inserted kink
    # (interior, or the right copy of a crossing on `grid_j`) carries `i`'s
    # segment as the crossing continues to the right; a plain accept and the
    # `grid_i` crossing carry the finalized `j`'s segment; a retained k keeps its
    # own. New j is the accepted candidate `i`.
    new_k_seg = jnp.where(
        keeps_k, seg_k, jnp.where(kink_5 | kink_6 | kink_on_j, seg_i, seg_j)
    )
    # New j is the accepted candidate `i`, so its source savings become
    # `savings_i`. Only `savings_j` is carried (the clause never reads a `k`
    # source), so no `k` savings slot is threaded.
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
            savings_i,
        ]
    )
    carry_new = jnp.where(dropped, carry, carry_accepted)

    return carry_new, (row_grid, row_policy, row_value, count)


def _savings_decrease_past_noise(
    *,
    grid_i: ScalarFloat,
    policy_i: ScalarFloat,
    grid_j: ScalarFloat,
    policy_j: ScalarFloat,
) -> ScalarBool:
    """Indicate a genuine decrease in implied savings between two candidates.

    Judges a decrease only past a noise floor: each implied saving $A = R - c$
    is a difference of like-magnitude grid and policy values, so its rounding
    error scales with those magnitudes, not with the saving itself. Candidates
    whose savings are tied in exact arithmetic (one exogenous savings point
    feeding consecutive candidates) would otherwise be kept or dropped by the
    sign of pure rounding noise — which varies with the backend's reduction
    order and makes the kept set platform-dependent.

    The floor `16 * eps * max(|R|, |c|)` masks a genuine savings decrease only
    when it is smaller than that band. The band is a small multiple of the
    representable spacing at the operands' magnitude, so a grid fine enough to
    place two distinct savings within it would need on the order of
    `1 / (16 * eps)` points across the same magnitude (hundreds of thousands in
    float32, astronomically more in float64) — far beyond any solved grid. A
    decrease inside the dead zone is therefore below the grid's own resolution,
    where the two candidates are numerically indistinguishable and dropping
    either is within interpolation error; the band buys cross-backend
    determinism at no resolvable accuracy cost.

    Args:
        grid_i: Endogenous grid point of the later candidate.
        policy_i: Policy value of the later candidate.
        grid_j: Endogenous grid point of the earlier candidate.
        policy_j: Policy value of the earlier candidate.

    Returns:
        Boolean indicator, true iff the later candidate's implied savings lie
        below the earlier candidate's by more than the noise floor.

    """
    savings_scale = jnp.maximum(
        jnp.maximum(jnp.abs(grid_i), jnp.abs(policy_i)),
        jnp.maximum(jnp.abs(grid_j), jnp.abs(policy_j)),
    )
    noise_floor = 16.0 * jnp.finfo(grid_i.dtype).eps * savings_scale
    return (grid_i - policy_i) < (grid_j - policy_j) - noise_floor


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
    savings_a: FloatND | None = None,
    savings_b: FloatND | None = None,
) -> BoolND:
    """Indicate whether two points lie on different value-function segments.

    Points lie on different segments iff the gradient of the implied savings
    $A = R - c$ between them exceeds `jump_thresh` in absolute value. At a
    coincident abscissa the gradient is undefined — `_slope` returns 0 there —
    so a branch switch landing exactly on a grid node would be invisible to the
    threshold test. A differing saving at a coincident abscissa is an infinite
    savings gradient, i.e. a genuine discontinuity, and counts as a jump
    directly; an equal saving (a true duplicate point) does not.

    At a coincident abscissa the comparison uses the pristine exogenous
    `savings` source when supplied — two candidates from one savings node are a
    duplicate, not a switch, even when their implied `R - c` rounds differently
    — and falls back to the implied saving otherwise.

    Args:
        grid_a: Endogenous grid point(s) of the first point.
        policy_a: Policy value(s) of the first point.
        grid_b: Endogenous grid point(s) of the second point.
        policy_b: Policy value(s) of the second point.
        jump_thresh: Threshold on $|\\Delta A / \\Delta R|$.
        savings_a: Exogenous source saving(s) of the first point, or `None`.
        savings_b: Exogenous source saving(s) of the second point, or `None`.

    Returns:
        Boolean indicator(s), broadcast over the inputs.

    """
    implied_a = grid_a - policy_a
    implied_b = grid_b - policy_b
    savings_slope = _slope(x_a=grid_a, y_a=implied_a, x_b=grid_b, y_b=implied_b)
    if savings_a is not None and savings_b is not None:
        coincident_jump = savings_a != savings_b
    else:
        coincident_jump = implied_a != implied_b
    return jnp.where(
        grid_a == grid_b,
        coincident_jump,
        jnp.abs(savings_slope) > jump_thresh,
    )


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
