r"""MSS upper-envelope refinement of EGM candidates (HARK's EGM upper envelope).

Implements the upper-envelope method of HARK (Carroll et al. 2018), referenced
in Dobrescu & Shanker (2026) as the `MSS` method column. Inverting the Euler
equation in models with discrete choices yields a value *correspondence*: the
candidates form a chain of linear segments between consecutive nodes that, in
non-concave regions, overlap in the endogenous grid. The refinement emits a
weakly ascending row that represents the correspondence's exact upper envelope
under linear interpolation: every envelope breakpoint is present as a node.

Between adjacent candidate abscissae no live segment starts or ends, so the
bracketing segments are full lines there and the envelope's interior
breakpoints are enumerated exactly: starting from the interval's left-endpoint
winner, the earliest abscissa at which another bracketing line overtakes the
current winner is the next breakpoint, and iterating from it walks the whole
switch sequence. The enumeration budget per interval is
`_MAX_CROSSINGS_PER_INTERVAL`; an interval needing more switches overflows
loudly through the `n_kept > n_refined` contract instead of truncating the
sequence.

Discontinuities and switches are represented by duplicated abscissae — same
grid value, left record then right record — covering three cases:

- an interior crossing: both branch policies at the exact intersection, the
  common crossing value;
- a switch exactly at a candidate abscissa: the two one-sided winners' records
  (equal values, distinct policies);
- an envelope value jump at a candidate abscissa (a winning branch ending
  above, or a new branch starting above, every other line): both one-sided
  values and policies, preserving the discontinuity instead of smearing it.

Interior coverage gaps — abscissa ranges no live segment spans — split the
chain in two ways:

- NaN-dead candidates (infeasible cases) break the segment chain outright;
- a finite value decrease between consecutive candidates is treated as a
  branch boundary rather than a segment, leaving the open interval between
  the two abscissae uncovered.

Gapped rows are compacted (the flanking live nodes become adjacent output
rows), so a linear read would bridge the gap with fabricated values. The
`read_supported` verdict from `refine_envelope_with_support` fail-closes such
rows for off-grid policy reads; the crossing-completeness guarantee applies on
the live-covered domain.

Crossing arithmetic runs in interval-local coordinates (offsets from the
interval's left endpoint), so no absolute-coordinate product enters the
enumeration. Tolerances are few-ulp windows of the compared quantity itself:
crossing ties coalesce only offsets that are numerically the same point (a
pencil of lines through one intersection), and value/policy record ties
merge only gaps at the storage-indistinguishability scale of the value
magnitude — so neither a translation of the resource grid nor a common shift
of all values can merge distinct representable crossings or records.

A live candidate *point* whose value exceeds both one-sided winners at its
abscissa beyond representational rounding (a zero-width dominating link that
no interval read can represent) triggers the loud `n_kept > n_refined`
overflow rather than being dropped.

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, Int1D, ScalarBool, ScalarInt

# Per-interval envelope-switch enumeration budget. The upper envelope of the
# lines bracketing one candidate interval can switch winners several times;
# each switch costs one enumeration step. Real DC-EGM correspondences rarely
# switch more than twice between adjacent candidates; an interval exceeding
# the budget overflows loudly via `n_kept > n_refined`.
_MAX_CROSSINGS_PER_INTERVAL = 8


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its exact upper envelope.

    Thin wrapper over `refine_envelope_with_support` that drops the
    read-support verdict — the historical four-field kernel contract for
    consumers that only interpolate the solve-side rows (which keep the
    compaction convention across coverage gaps).
    """
    out_grid, out_policy, out_value, n_kept, _ = refine_envelope_with_support(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=n_refined,
        segment_id=segment_id,
    )
    return out_grid, out_policy, out_value, n_kept


def refine_envelope_with_support(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]:
    """Refine to the exact upper envelope and report off-grid read support.

    The candidates arrive as a chain of consecutive linear segments (one segment
    per consecutive input pair), as the Euler inversion produces them: the
    constrained run followed by the interior run, each ascending along its own
    margin but jointly non-monotone in the endogenous grid. The output row
    carries, in weakly ascending grid order:

    - one node per distinct live candidate abscissa (two where the one-sided
      winners differ — a switch or value jump at the node);
    - every interior envelope crossing, inserted twice (same abscissa, left-
      then right-branch policy), enumerated to completeness within the
      per-interval budget.

    Args:
        endog_grid: Candidate endogenous grid points (resources). Consecutive
            entries form the linear segments scanned for the envelope.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        segment_id: Optional per-candidate branch label, aligned with
            `endog_grid`. When supplied, a consecutive-pair link is a real value
            segment iff both endpoints carry the same label, so unrelated
            branches are never bridged — the explicit-topology path that replaces
            the `decreases` heuristic. `None` (the default) infers segments from a
            grid or value decrease, HARK's monotone split.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), the number of envelope points
        `n_kept`, and the read-support verdict. `n_kept > n_refined` signals
        overflow — more envelope rows than `n_refined` slots, an interval
        whose switch sequence exceeds the enumeration budget, or a live
        candidate point that strictly dominates both one-sided link records
        at its abscissa (a single-abscissa spike no linearly-read row can
        carry). Callers must check the counter rather than publish the
        truncated arrays silently — the EGM step NaN-poisons its published
        rows on overflow so the solve loop's NaN diagnostics name the
        offending (regime, period). The read-support verdict is `False` when
        any interval between adjacent emitted nodes has no covering live link
        (a coverage gap — from NaN-dead candidates or an inferred
        finite-value-decrease split): the compacted row bridges such an
        interval linearly, so its off-grid read fabricates values and the
        publication layer must withhold the simulation rows.

    """
    # A dead candidate arrives NaN-filled (the EGM step poisons `-inf`-valued
    # corners to NaN before refinement). A segment touching a dead endpoint is
    # excluded from the scan, and a dead abscissa sorts to the NaN tail, so it
    # is neither queried nor evaluated.
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)

    # Sort the candidate abscissae ascending so the emission order is
    # left-to-right and the NaN tail is contiguous; dead nodes sort last.
    grid_key = jnp.where(dead, jnp.inf, endog_grid)
    order = jnp.argsort(grid_key)
    query_grid = jnp.where(dead, jnp.nan, endog_grid)[order]
    query_dead = dead[order]

    # Segment endpoints: candidate `k` to candidate `k+1`, consecutive in the
    # (unsorted) input order — the EGM cloud's natural segment chain. A segment
    # with a dead endpoint is excluded from every evaluation.
    left_grid = endog_grid[:-1]
    right_grid = endog_grid[1:]
    left_policy = policy[:-1]
    left_value = value[:-1]

    # Per-link branch id and live mask. With explicit topology a link is a real
    # value segment iff both endpoints carry the same branch label, so unrelated
    # branches are never bridged. Without it, fall back to HARK's monotone split:
    # a new segment starts wherever the grid or value *decreases* between
    # consecutive candidates, and a link spanning such a decrease is a
    # non-monotone bridge excluded from the scan. Either way the winner stays
    # constant across one branch, so only a genuine branch switch is a kink.
    if segment_id is None:
        decreases = (right_grid < left_grid) | (value[1:] < left_value)
        link_segment = jnp.cumsum(decreases.astype(jnp.int32))
        segment_live = ~dead[:-1] & ~dead[1:] & ~decreases
    else:
        same_segment = segment_id[:-1] == segment_id[1:]
        link_segment = segment_id[:-1].astype(jnp.int32)
        segment_live = ~dead[:-1] & ~dead[1:] & same_segment

    links = _LinkLines(
        lower=jnp.minimum(left_grid, right_grid),
        upper=jnp.maximum(left_grid, right_grid),
        anchor_grid=left_grid,
        anchor_value=left_value,
        anchor_policy=left_policy,
        value_slope=_slope(
            x_a=left_grid, y_a=left_value, x_b=right_grid, y_b=value[1:]
        ),
        policy_slope=_slope(
            x_a=left_grid, y_a=left_policy, x_b=right_grid, y_b=policy[1:]
        ),
        segment=link_segment,
        live=segment_live,
    )

    left_side, right_side = _node_side_winners(query_grid=query_grid, links=links)

    # One node per distinct live abscissa; duplicated abscissae in the input
    # collapse onto the first occurrence (all copies see the same winners).
    prev_grid = jnp.concatenate(
        [jnp.full((1,), -jnp.inf, dtype=query_grid.dtype), query_grid[:-1]]
    )
    is_new = query_grid > prev_grid
    # Record equality is a few-ulp window: two computations of the same
    # quantity agree to within a handful of rounding steps of the stored
    # magnitude, and gaps below that scale are indistinguishable in storage.
    # A wider cushion would swallow genuinely representable gaps whenever a
    # common cardinal shift raises the value level — the winner would then
    # depend on an arbitrary value normalization. A missed tie merely
    # duplicates the node, which the read resolves by its side convention.
    tolerance = 8.0 * float(jnp.finfo(query_grid.dtype).eps)
    same_record = (
        (left_side.branch == right_side.branch)
        & jnp.isclose(left_side.value, right_side.value, rtol=tolerance, atol=0.0)
        & jnp.isclose(left_side.policy, right_side.policy, rtol=tolerance, atol=0.0)
    )
    node_left_valid = left_side.exists & is_new
    node_right_valid = right_side.exists & is_new & ~(left_side.exists & same_record)

    unrepresented_point, read_supported = _unsupported_read_verdicts(
        query_grid=query_grid,
        query_dead=query_dead,
        point_value=jnp.where(query_dead, jnp.nan, value[order]),
        left_side=left_side,
        right_side=right_side,
        links=links,
        tolerance=tolerance,
    )

    crossings, interval_overflow = _enumerate_interval_crossings(
        query_grid=query_grid,
        prev_grid=prev_grid,
        is_new=is_new,
        init_winner=jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.int32), right_side.link[:-1]]
        ),
        init_exists=jnp.concatenate(
            [jnp.zeros((1,), dtype=bool), right_side.exists[:-1]]
        ),
        links=links,
    )

    # Per-query emission block, ascending: the interval's crossings (each two
    # rows — same abscissa, left- then right-branch policy) precede the node's
    # left and right records. All crossings of the interval ending at node `i`
    # lie strictly below `grid_i`, so the block order is the output order.
    nan_scalar = jnp.full((), jnp.nan, dtype=query_grid.dtype)
    cross_grid_rows = jnp.repeat(crossings.grid, 2, axis=1)
    cross_value_rows = jnp.repeat(crossings.value, 2, axis=1)
    cross_policy_rows = jnp.stack(
        [crossings.policy_left, crossings.policy_right], axis=2
    ).reshape(crossings.grid.shape[0], -1)
    cross_valid_rows = jnp.repeat(crossings.valid, 2, axis=1)

    row_grid = jnp.concatenate(
        [cross_grid_rows, query_grid[:, None], query_grid[:, None]], axis=1
    ).ravel()
    row_value = jnp.concatenate(
        [cross_value_rows, left_side.value[:, None], right_side.value[:, None]], axis=1
    ).ravel()
    row_policy = jnp.concatenate(
        [cross_policy_rows, left_side.policy[:, None], right_side.policy[:, None]],
        axis=1,
    ).ravel()
    row_valid = jnp.concatenate(
        [cross_valid_rows, node_left_valid[:, None], node_right_valid[:, None]], axis=1
    ).ravel()

    # Compact the valid rows into the NaN-padded prefix, preserving order.
    position = jnp.cumsum(row_valid.astype(jnp.int32)) - 1
    slot = jnp.where(row_valid, position, n_refined)
    out_grid = jnp.full(n_refined, jnp.nan, dtype=endog_grid.dtype)
    out_policy = jnp.full(n_refined, jnp.nan, dtype=policy.dtype)
    out_value = jnp.full(n_refined, jnp.nan, dtype=value.dtype)
    out_grid = out_grid.at[slot].set(
        jnp.where(row_valid, row_grid, nan_scalar), mode="drop"
    )
    out_policy = out_policy.at[slot].set(
        jnp.where(row_valid, row_policy, nan_scalar), mode="drop"
    )
    out_value = out_value.at[slot].set(
        jnp.where(row_valid, row_value, nan_scalar), mode="drop"
    )

    n_kept = jnp.sum(row_valid, dtype=jnp.int32)
    n_kept = jnp.where(
        interval_overflow | unrepresented_point,
        jnp.maximum(n_kept, n_refined + 1),
        n_kept,
    )
    return out_grid, out_policy, out_value, n_kept, read_supported


def _unsupported_read_verdicts(
    *,
    query_grid: Float1D,
    query_dead: BoolND,
    point_value: Float1D,
    left_side: _SideWinner,
    right_side: _SideWinner,
    links: _LinkLines,
    tolerance: float,
) -> tuple[ScalarBool, ScalarBool]:
    """Detect envelope features a linearly-read compacted row cannot carry.

    Returns:
        Tuple of two verdicts:

        - `unrepresented_point`: a live candidate point strictly dominates
          both one-sided link records at its abscissa — a single-abscissa
          spike (e.g. a zero-width terminal link). The duplicated-node
          convention publishes one-sided *limits*, and a spike is neither, so
          dropping it would understate the envelope at an actual candidate
          node; the caller routes it into the loud overflow contract. A
          duplicate candidate matching a covering link's record has zero
          excess and stays quiet.
        - `read_supported`: between adjacent *emitted* abscissae the row is
          read as a linear span, which is genuine only where some live link
          covers the interval. An uncovered interval — the chain split by
          NaN-dead candidates or by the inferred finite-value-decrease
          segment break — is a coverage gap the compaction bridges; `False`
          lets the publication layer withhold the simulation rows
          (fail-closed) while the solve rows keep the compaction convention.

    """
    best_side_value = jnp.maximum(
        jnp.where(left_side.exists, left_side.value, -jnp.inf),
        jnp.where(right_side.exists, right_side.value, -jnp.inf),
    )
    dominance_margin = tolerance * jnp.maximum(
        jnp.abs(point_value),
        jnp.where(jnp.isfinite(best_side_value), jnp.abs(best_side_value), 0.0),
    )
    unrepresented_point = jnp.any(
        ~query_dead & (point_value - best_side_value > dominance_margin)
    )

    emits = (left_side.exists | right_side.exists) & ~query_dead
    previous_emitted_grid = jnp.concatenate(
        [
            jnp.full((1,), -jnp.inf, dtype=query_grid.dtype),
            jax.lax.cummax(jnp.where(emits, query_grid, -jnp.inf))[:-1],
        ]
    )
    interval_covered = jnp.any(
        links.live[None, :]
        & (links.lower[None, :] <= previous_emitted_grid[:, None])
        & (links.upper[None, :] >= query_grid[:, None]),
        axis=1,
    )
    coverage_gap = jnp.any(
        emits
        & jnp.isfinite(previous_emitted_grid)
        & (query_grid > previous_emitted_grid)
        & ~interval_covered
    )
    return unrepresented_point, ~coverage_gap


class _LinkLines:
    """Per-link line geometry of the candidate segment chain.

    Each consecutive candidate pair defines one link; within any interval
    between adjacent candidate abscissae a live link is a full line, described
    by its anchor point and slopes.
    """

    def __init__(
        self,
        *,
        lower: Float1D,
        upper: Float1D,
        anchor_grid: Float1D,
        anchor_value: Float1D,
        anchor_policy: Float1D,
        value_slope: Float1D,
        policy_slope: Float1D,
        segment: Int1D,
        live: BoolND,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.anchor_grid = anchor_grid
        self.anchor_value = anchor_value
        self.anchor_policy = anchor_policy
        self.value_slope = value_slope
        self.policy_slope = policy_slope
        self.segment = segment
        self.live = live

    def value_at(self, x: FloatND) -> FloatND:
        """Evaluate every link's value line at `x` (broadcast on the last axis)."""
        return self.anchor_value + self.value_slope * (x - self.anchor_grid)

    def policy_at(self, x: FloatND) -> FloatND:
        """Evaluate every link's policy line at `x` (broadcast on the last axis)."""
        return self.anchor_policy + self.policy_slope * (x - self.anchor_grid)


class _SideWinner:
    """One-sided envelope winner per query node.

    The left winner is the envelope's limit from below the abscissa (among
    links covering it from the left, the highest value, ties to the flattest
    line — the one that was higher just before); the right winner mirrors it
    (ties to the steepest line — the one higher just after).
    """

    def __init__(
        self,
        *,
        exists: BoolND,
        link: Int1D,
        branch: Int1D,
        value: Float1D,
        policy: Float1D,
    ) -> None:
        self.exists = exists
        self.link = link
        self.branch = branch
        self.value = value
        self.policy = policy


def _node_side_winners(
    *, query_grid: Float1D, links: _LinkLines
) -> tuple[_SideWinner, _SideWinner]:
    """Compute the left- and right-side envelope winners at every query node.

    A link owns a node's left side iff the node lies strictly inside or at the
    link's upper endpoint (`lower < x <= upper`), and the right side iff it
    lies at the lower endpoint or strictly inside (`lower <= x < upper`).
    Where the two sides' records differ — branch switch or value jump exactly
    at the node — the caller emits both, duplicating the abscissa.
    """
    query = query_grid[:, None]
    value_at = links.value_at(query)
    policy_at = links.policy_at(query)
    covers_left = (
        links.live[None, :]
        & (links.lower[None, :] < query)
        & (query <= links.upper[None, :])
    )
    covers_right = (
        links.live[None, :]
        & (links.lower[None, :] <= query)
        & (query < links.upper[None, :])
    )
    left = _lexicographic_winner(
        covers=covers_left,
        value_at=value_at,
        policy_at=policy_at,
        links=links,
        slope_sign=-1.0,
    )
    right = _lexicographic_winner(
        covers=covers_right,
        value_at=value_at,
        policy_at=policy_at,
        links=links,
        slope_sign=1.0,
    )
    return left, right


def _lexicographic_winner(
    *,
    covers: BoolND,
    value_at: FloatND,
    policy_at: FloatND,
    links: _LinkLines,
    slope_sign: float,
) -> _SideWinner:
    """Pick the covering link with the highest value, ties by signed slope.

    `slope_sign = -1` prefers the flattest line among value ties (the left-side
    winner: highest just before the node); `slope_sign = +1` prefers the
    steepest (the right-side winner: highest just after).
    """
    masked_value = jnp.where(covers, value_at, -jnp.inf)
    best_value = jnp.max(masked_value, axis=1)
    exists = jnp.isfinite(best_value)
    # Value ties are a few-ulp window of the stored magnitude — the scale at
    # which two computations of the same envelope value are indistinguishable.
    # A wider cushion would let a common cardinal value shift merge branches
    # whose gap is genuinely representable. A missed tie merely skips the
    # slope tie-break and duplicates the node downstream — never merges
    # records.
    tolerance = 8.0 * jnp.finfo(value_at.dtype).eps
    tie = covers & jnp.isclose(
        masked_value, best_value[:, None], rtol=tolerance, atol=0.0
    )
    slope_score = jnp.where(tie, slope_sign * links.value_slope[None, :], -jnp.inf)
    link = jnp.argmax(slope_score, axis=1).astype(jnp.int32)
    return _SideWinner(
        exists=exists,
        link=link,
        branch=jnp.where(exists, links.segment[link], -1).astype(jnp.int32),
        value=jnp.take_along_axis(value_at, link[:, None], axis=1)[:, 0],
        policy=jnp.take_along_axis(policy_at, link[:, None], axis=1)[:, 0],
    )


class _IntervalCrossings:
    """Enumerated envelope crossings per interval, aligned with the query axis.

    Entry `(i, s)` is the interval's `s`-th switch strictly inside
    `(grid_{i-1}, grid_i)`, in ascending order; `valid` masks the enumerated
    prefix. The two policy fields carry the outgoing and incoming branch's
    policy line at the crossing.
    """

    def __init__(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        policy_left: FloatND,
        policy_right: FloatND,
        valid: BoolND,
    ) -> None:
        self.grid = grid
        self.value = value
        self.policy_left = policy_left
        self.policy_right = policy_right
        self.valid = valid


def _enumerate_interval_crossings(
    *,
    query_grid: Float1D,
    prev_grid: Float1D,
    is_new: BoolND,
    init_winner: Int1D,
    init_exists: BoolND,
    links: _LinkLines,
) -> tuple[_IntervalCrossings, BoolND]:
    """Enumerate every envelope switch strictly inside each candidate interval.

    Between adjacent candidate abscissae no live link starts or ends, so every
    link bracketing the interval is a full line there. Starting from the
    interval's left-endpoint winner (the previous node's right-side winner),
    the earliest abscissa at which a bracketing line with a strictly greater
    slope overtakes the current winner is the next envelope breakpoint;
    iterating from it enumerates the complete ordered switch sequence. The
    iteration runs `_MAX_CROSSINGS_PER_INTERVAL` steps; a scalar overflow flag
    reports any interval whose sequence is longer, so the caller can refuse the
    truncated row via the `n_kept` contract.

    Returns:
        Tuple of the enumerated crossings and the scalar overflow flag.

    """
    covers_interval = (
        links.live[None, :]
        & (links.lower[None, :] <= prev_grid[:, None])
        & (links.upper[None, :] >= query_grid[:, None])
    )
    interval_active = is_new & init_exists & jnp.isfinite(prev_grid)
    link_index = jnp.arange(links.lower.shape[0], dtype=jnp.int32)
    # The crossing-coincidence window scales with the computed offset itself
    # (a few ulp of the crossing's own local position), so only crossings
    # that are numerically the same point coalesce (a pencil of lines through
    # one intersection). Scaling by the whole interval width instead would
    # merge crossings that are far apart at ulp scale inside a wide interval
    # and silently skip the branch that wins between them.
    crossing_ulp = 16.0 * jnp.finfo(query_grid.dtype).eps

    def overtaking(winner: Int1D, x_current: Float1D) -> tuple[FloatND, BoolND]:
        """Overtake offsets from `x_current` for each bracketing line.

        All quantities are interval-local differences: every line is evaluated
        at `x_current` (an offset from its own anchor, bounded by the link
        span plus one interval), and the crossing offset is the value gap
        divided by the slope gap — no absolute-coordinate products enter, so
        precision does not degrade with the resource origin.
        """
        winner_slope = links.value_slope[winner]
        value_at_current = links.value_at(x_current[:, None])
        winner_value = jnp.take_along_axis(value_at_current, winner[:, None], axis=1)
        slope_gap = links.value_slope[None, :] - winner_slope[:, None]
        safe_slope_gap = jnp.where(slope_gap <= 0.0, 1.0, slope_gap)
        offset = jnp.where(
            slope_gap <= 0.0,
            jnp.inf,
            (winner_value - value_at_current) / safe_slope_gap,
        )
        valid = (
            covers_interval
            & (link_index[None, :] != winner[:, None])
            & (slope_gap > 0.0)
            & (offset > 0.0)
            & (offset < (query_grid - x_current)[:, None])
        )
        return offset, valid

    def step(
        carry: tuple[Int1D, Float1D, BoolND], _: ScalarInt
    ) -> tuple[tuple[Int1D, Float1D, BoolND], _IntervalCrossings]:
        winner, x_current, active = carry
        offset, valid = overtaking(winner, x_current)
        valid = valid & active[:, None]
        offset_masked = jnp.where(valid, offset, jnp.inf)
        offset_next = jnp.min(offset_masked, axis=1)
        found = jnp.isfinite(offset_next)
        offset_tolerance = crossing_ulp * jnp.where(found, offset_next, 0.0)
        tie = valid & (offset <= (offset_next + offset_tolerance)[:, None])
        incoming = jnp.argmax(
            jnp.where(tie, links.value_slope[None, :], -jnp.inf), axis=1
        ).astype(jnp.int32)

        x_next = x_current + offset_next
        winner_slope = links.value_slope[winner]
        crossing_value = (
            links.anchor_value[winner]
            + winner_slope * (x_current - links.anchor_grid[winner])
            + winner_slope * offset_next
        )
        emitted = _IntervalCrossings(
            grid=jnp.where(found, x_next, jnp.nan),
            value=crossing_value,
            policy_left=links.anchor_policy[winner]
            + links.policy_slope[winner] * (x_next - links.anchor_grid[winner]),
            policy_right=links.anchor_policy[incoming]
            + links.policy_slope[incoming] * (x_next - links.anchor_grid[incoming]),
            valid=found,
        )
        new_winner = jnp.where(found, incoming, winner).astype(jnp.int32)
        new_x = jnp.where(found, x_next, x_current)
        return (new_winner, new_x, active & found), emitted

    carry_init = (init_winner, prev_grid, interval_active)
    carry_final, rows = jax.lax.scan(
        step,
        carry_init,
        jnp.arange(_MAX_CROSSINGS_PER_INTERVAL, dtype=jnp.int32),
    )
    winner, x_current, active = carry_final
    _, leftover_valid = overtaking(winner, x_current)
    interval_overflow = jnp.any(leftover_valid & active[:, None])

    stack_to_query_axis = lambda leaf: jnp.moveaxis(leaf, 0, 1)  # noqa: E731
    return (
        _IntervalCrossings(
            grid=stack_to_query_axis(rows.grid),
            value=stack_to_query_axis(rows.value),
            policy_left=stack_to_query_axis(rows.policy_left),
            policy_right=stack_to_query_axis(rows.policy_right),
            valid=stack_to_query_axis(rows.valid),
        ),
        interval_overflow,
    )


jax.tree_util.register_pytree_node(
    _IntervalCrossings,
    lambda r: (
        (r.grid, r.value, r.policy_left, r.policy_right, r.valid),
        None,
    ),
    lambda _aux, children: _IntervalCrossings(
        grid=children[0],
        value=children[1],
        policy_left=children[2],
        policy_right=children[3],
        valid=children[4],
    ),
)


def _slope(*, x_a: FloatND, y_a: FloatND, x_b: FloatND, y_b: FloatND) -> FloatND:
    r"""Compute the slope between two points, with `0.0` for coincident abscissae.

    Args:
        x_a: Abscissa(e) of the first point.
        y_a: Ordinate(s) of the first point.
        x_b: Abscissa(e) of the second point.
        y_b: Ordinate(s) of the second point.

    Returns:
        Slope(s) $\Delta y / \Delta x$, broadcast over the inputs.

    """
    delta_x = x_b - x_a
    return jnp.where(
        delta_x == 0.0, 0.0, (y_b - y_a) / jnp.where(delta_x == 0.0, 1.0, delta_x)
    )
