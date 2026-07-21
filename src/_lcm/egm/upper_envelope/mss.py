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

    left_is_lower = left_grid <= right_grid
    links = _LinkLines(
        lower=jnp.minimum(left_grid, right_grid),
        upper=jnp.maximum(left_grid, right_grid),
        lower_value=jnp.where(left_is_lower, left_value, value[1:]),
        upper_value=jnp.where(left_is_lower, value[1:], left_value),
        lower_policy=jnp.where(left_is_lower, left_policy, policy[1:]),
        upper_policy=jnp.where(left_is_lower, policy[1:], left_policy),
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

    point_value = jnp.where(query_dead, jnp.nan, value[order])
    certificate = _certify_chord_point_relations(
        query_grid=query_grid, point_value=point_value, links=links
    )
    left_side, right_side = _node_side_winners(
        query_grid=query_grid,
        point_value=point_value,
        links=links,
        certificate=certificate,
    )

    # One node per distinct live abscissa; duplicated abscissae in the input
    # collapse onto the first occurrence (all copies see the same winners).
    prev_grid = jnp.concatenate(
        [jnp.full((1,), -jnp.inf, dtype=query_grid.dtype), query_grid[:-1]]
    )
    is_new = query_grid > prev_grid
    # Record equality is exact: when both side winners are the same branch,
    # its value and policy at the node come from the same link evaluated at
    # the same query — identical arithmetic, so genuinely one record is
    # bitwise equal. Any tolerance window here scales with the absolute value
    # level and would merge representable records under a common cardinal
    # shift. A missed merge merely duplicates the node, which the read
    # resolves by its side convention.
    same_record = (
        (left_side.branch == right_side.branch)
        & (left_side.value == right_side.value)
        & (left_side.policy == right_side.policy)
    )
    node_left_valid = left_side.exists & is_new
    node_right_valid = right_side.exists & is_new & ~(left_side.exists & same_record)

    unrepresented_point, read_supported = _unsupported_read_verdicts(
        query_grid=query_grid,
        query_dead=query_dead,
        left_side=left_side,
        right_side=right_side,
        links=links,
        certificate=certificate,
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

    missed_switch = _missed_switch_overflow(
        is_new=is_new,
        left_side=left_side,
        right_side=right_side,
        crossings=crossings,
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
        interval_overflow | unrepresented_point | missed_switch,
        jnp.maximum(n_kept, n_refined + 1),
        n_kept,
    )
    return out_grid, out_policy, out_value, n_kept, read_supported


def _unsupported_read_verdicts(
    *,
    query_grid: Float1D,
    query_dead: BoolND,
    left_side: _SideWinner,
    right_side: _SideWinner,
    links: _LinkLines,
    certificate: _ChordPointCertificate,
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
    # A point is representable in the linearly read row only if some live
    # covering link certifiably reaches it: an endpoint match compares the
    # stored records exactly, and an interior read is decided by the
    # compensated cross-multiplied chord gap — a single rounded affine
    # evaluation of a spanning link's chord can land on either side of a
    # stored point a few ulp away, so it can neither absorb nor flag a point
    # on its own. Any magnitude-relative tolerance on the stored-record path
    # would scale with the absolute value level and silently absorb
    # representable strict maxima a few stored ulp above the side record.
    covered = jnp.any(certificate.dominates(), axis=1)
    # A covering interior chord whose ordering is unresolved may be the true
    # envelope above the point; an endpoint record that reaches the point must
    # not mask it. Unless some chord certifiably dominates the point (it is
    # then below the envelope and safely dropped), an unresolved interior chord
    # leaves the point unrepresentable, routed to loud overflow.
    dominated = jnp.any(certificate.covers & certificate.above, axis=1)
    unresolved = jnp.any(certificate.interior_unresolved(), axis=1)
    supported = covered & ~(unresolved & ~dominated)
    unrepresented_point = jnp.any(~query_dead & ~supported)

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
    by its anchor point and slopes. The stored endpoint records are kept
    alongside the line: an exact endpoint query returns the stored record, so
    a candidate node is never reconstructed through the affine product (whose
    cancellation error scales with the span times the slope and can lose the
    stored value entirely on a long link).
    """

    def __init__(
        self,
        *,
        lower: Float1D,
        upper: Float1D,
        lower_value: Float1D,
        upper_value: Float1D,
        lower_policy: Float1D,
        upper_policy: Float1D,
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
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.lower_policy = lower_policy
        self.upper_policy = upper_policy
        self.anchor_grid = anchor_grid
        self.anchor_value = anchor_value
        self.anchor_policy = anchor_policy
        self.value_slope = value_slope
        self.policy_slope = policy_slope
        self.segment = segment
        self.live = live

    def value_at(self, x: FloatND) -> FloatND:
        """Evaluate every link's value at `x`, snapping exact endpoint queries.

        Broadcasts on the last axis. An exact endpoint query returns the
        stored endpoint record; an interior query evaluates the line from the
        *nearer* stored endpoint. Anchoring at the nearer endpoint keeps the
        slope-times-offset term small relative to the far-anchored form, whose
        cancellation against the anchor value grows with the span times the
        slope and can round a covering chord clear past a stored point
        candidate near the far end of a long link.
        """
        return self._at(
            x=x,
            lower_record=self.lower_value,
            upper_record=self.upper_value,
            slope=self.value_slope,
        )

    def policy_at(self, x: FloatND) -> FloatND:
        """Evaluate every link's policy at `x`, snapping exact endpoint queries.

        Broadcasts on the last axis. Same evaluation contract as `value_at`:
        stored records at exact endpoints, nearer-endpoint anchoring for
        interior queries.
        """
        return self._at(
            x=x,
            lower_record=self.lower_policy,
            upper_record=self.upper_policy,
            slope=self.policy_slope,
        )

    def _at(
        self,
        *,
        x: FloatND,
        lower_record: Float1D,
        upper_record: Float1D,
        slope: Float1D,
    ) -> FloatND:
        from_lower = lower_record + slope * (x - self.lower)
        from_upper = upper_record + slope * (x - self.upper)
        nearer_is_lower = (x - self.lower) <= (self.upper - x)
        line = jnp.where(nearer_is_lower, from_lower, from_upper)
        return jnp.where(
            x == self.lower,
            lower_record,
            jnp.where(x == self.upper, upper_record, line),
        )


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
    *,
    query_grid: Float1D,
    point_value: Float1D,
    links: _LinkLines,
    certificate: _ChordPointCertificate,
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
        point_value=point_value,
        certificate=certificate,
    )
    right = _lexicographic_winner(
        covers=covers_right,
        value_at=value_at,
        policy_at=policy_at,
        links=links,
        slope_sign=1.0,
        point_value=point_value,
        certificate=certificate,
    )
    return left, right


def _lexicographic_winner(
    *,
    covers: BoolND,
    value_at: FloatND,
    policy_at: FloatND,
    links: _LinkLines,
    slope_sign: float,
    point_value: Float1D,
    certificate: _ChordPointCertificate,
) -> _SideWinner:
    """Pick the covering link with the highest value, ties by signed slope.

    `slope_sign = -1` prefers the flattest line among value ties (the left-side
    winner: highest just before the node); `slope_sign = +1` prefers the
    steepest (the right-side winner: highest just after).

    The node's own stored point value anchors the comparison wherever it can:
    when no covering chord is *certified* strictly above the point and some
    covering link is certified tied with it (a stored-record match or an
    interior chord inside the certification margin), the published value is
    the stored point itself and the tie set is the certified-tied links —
    never a single rounded interior read, which can land a few ulp on either
    side of an exact tie and hand the node to the wrong branch with a value
    above every underlying candidate. Only when some chord is certified
    strictly above the point (the routine dominated-node case) does the
    rounded maximum decide, with exact-equality ties: any tolerance window
    there would scale with the absolute value level, and a common cardinal
    shift would pull genuinely representable gaps inside it.
    """
    masked_value = jnp.where(covers, value_at, -jnp.inf)
    rounded_best = jnp.max(masked_value, axis=1)
    rounded_tie = covers & (masked_value == rounded_best[:, None])

    side_tied = covers & certificate.ties_point()
    point_anchored = (
        jnp.isfinite(point_value)
        & ~jnp.any(covers & certificate.above, axis=1)
        & ~jnp.any(covers & certificate.interior_unresolved(), axis=1)
        & jnp.any(side_tied, axis=1)
    )
    best_value = jnp.where(point_anchored, point_value, rounded_best)
    tie = jnp.where(point_anchored[:, None], side_tied, rounded_tie)

    exists = jnp.isfinite(best_value)
    slope_score = jnp.where(tie, slope_sign * links.value_slope[None, :], -jnp.inf)
    link = jnp.argmax(slope_score, axis=1).astype(jnp.int32)
    # The published value is the maximum (or the certified point), not the tie
    # winner's own read: the winner owns the policy and the slope convention,
    # but a numerical tie rule must never replace the hard maximum of the
    # stored records by a near-maximal competitor's lower value.
    return _SideWinner(
        exists=exists,
        link=link,
        branch=jnp.where(exists, links.segment[link], -1).astype(jnp.int32),
        value=best_value,
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


def _two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Float addition with residual: `a + b = total + error`, both floats.

    Knuth's branch-free TwoSum. The returned pair represents the exact real
    sum — `total` is the rounded sum and `error` its exact rounding residual —
    provided the addition itself neither overflows nor produces a nonfinite
    intermediate; at overflow scales the residual is not meaningful.
    """
    total = a + b
    b_virtual = total - a
    a_virtual = total - b_virtual
    return total, (a - a_virtual) + (b - b_virtual)


def _two_product(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Float multiplication with residual: `a * b = product + error`.

    Dekker's split-based TwoProd (no FMA required): each factor is split into
    high and low halves whose pairwise products are exact, so the residual of
    the rounded product is recovered exactly in working precision — for
    operands whose split products stay finite and normal. Near the overflow
    boundary the split factor itself can overflow (a nonfinite residual), and
    subnormal split halves can lose low bits; the callers' certification
    margins and the loud-overflow failure direction absorb those extremes
    rather than the residual arithmetic.
    """
    mantissa_digits = jnp.finfo(a.dtype).nmant + 1
    split_factor = jnp.asarray(2.0 ** ((mantissa_digits + 1) // 2) + 1.0, dtype=a.dtype)

    def split(x: FloatND) -> tuple[FloatND, FloatND]:
        scaled = split_factor * x
        high = scaled - (scaled - x)
        return high, x - high

    product = a * b
    a_high, a_low = split(a)
    b_high, b_low = split(b)
    error = (
        ((a_high * b_high - product) + a_high * b_low) + a_low * b_high
    ) + a_low * b_low
    return product, error


def _compensated_line_gap(
    *,
    x: FloatND,
    anchor_value: FloatND,
    anchor_grid: FloatND,
    slope: FloatND,
    winner_anchor_value: FloatND,
    winner_anchor_grid: FloatND,
    winner_slope: FloatND,
) -> FloatND:
    """Line-difference `(a + s (x - t)) - (a_w + s_w (x - t_w))`, compensated.

    Double-float evaluation: every subtraction runs through TwoSum and every
    slope-offset product through TwoProd, the high parts accumulate through a
    TwoSum chain, and the collected residuals fold in at the end. The result
    carries a relative error of order machine epsilon in the *gap itself* —
    not in the cancelling terms — so a genuine sub-unit gap between lines with
    large local terms survives the evaluation instead of being rounded away.
    """
    value_gap, value_gap_err = _two_sum(anchor_value, -winner_anchor_value)
    offset, offset_err = _two_sum(x, -anchor_grid)
    winner_offset, winner_offset_err = _two_sum(x, -winner_anchor_grid)
    term, term_err = _two_product(slope, offset)
    winner_term, winner_term_err = _two_product(winner_slope, winner_offset)
    partial, partial_err = _two_sum(value_gap, term)
    high, high_err = _two_sum(partial, -winner_term)
    residual = (
        value_gap_err
        + partial_err
        + high_err
        + term_err
        + slope * offset_err
        - winner_term_err
        - winner_slope * winner_offset_err
    )
    return high + residual


class _ChordPointCertificate:
    """Certified per-(query, link) relations of each link's chord to the point.

    Each query abscissa carries its own stored point value; each live link's
    exact chord relates to that point in one of the certified ways below. At
    an endpoint abscissa the relation is the exact stored-record comparison.
    In a link's interior the decision is the sign of the cross-multiplied
    chord gap

        D = (v_l - p) (x_u - x_l) + (v_u - v_l) (q - x_l),

    which shares the sign of `chord(q) - p` (the abscissae are sorted, so the
    span is positive). `D` is evaluated in compensated double-float
    arithmetic — exact difference pairs via TwoSum, exact main products via
    TwoProd, the epsilon-scale cross terms carried explicitly — so the computed
    gap is accurate to a tight *noise floor* (a small multiple of the
    second-order-in-epsilon term error). Two thresholds classify the gap:

    - `|D| <= noise` — certified equal: the chord passes through the point to
      the arithmetic's own resolution (an exactly-tied chord computes `D = 0`).
    - `D > margin` — certified strictly above (`margin` a conservative multiple
      of `noise`, so the certified-above verdict never fires on rounding noise).
    - `noise < D <= margin` — unresolved above: the compensated gap sees a real
      excess the conservative margin cannot certify, so the chord may be the
      true envelope above the point and the node cannot be anchored on it.

    Every certificate term is a difference, so the relations are exactly
    invariant to common translations of the value level and resource origin.
    """

    def __init__(
        self,
        *,
        covers: BoolND,
        at_endpoint: BoolND,
        stored_ge: BoolND,
        stored_eq: BoolND,
        above: BoolND,
        tied: BoolND,
        unresolved_above: BoolND,
    ) -> None:
        self.covers = covers
        """Live link covering the query from at least one side (zero-width
        links cover neither side, so a single-abscissa spike never absorbs
        itself through its own duplicated record)."""
        self.at_endpoint = at_endpoint
        """The query is one of the link's stored endpoint abscissae."""
        self.stored_ge = stored_ge
        """Endpoint case: the stored record at the query is `>=` the point."""
        self.stored_eq = stored_eq
        """Endpoint case: the stored record at the query equals the point."""
        self.above = above
        """Interior case: the chord certifiably strictly exceeds the point
        (compensated gap above the conservative margin)."""
        self.tied = tied
        """Interior case: the chord equals the point to the arithmetic noise
        floor (`|D| <= noise`) — a certified tie, not merely margin-close."""
        self.unresolved_above = unresolved_above
        """Interior case: the compensated gap sees a real excess above the
        noise floor but below the certified-above margin — the chord may be the
        true envelope above the point, so it cannot be masked by an endpoint
        tie."""

    def dominates(self) -> BoolND:
        """Link certifiably reaches the point (absorbs it quietly).

        An unresolved interior ordering does NOT dominate: over-reporting an
        unrepresented point is the safe direction — the routine
        dominated-candidate absorptions clear the margin by orders of
        magnitude, and an eps-close point-versus-chord ordering is exactly
        where a quiet answer could hide a genuine spike.
        """
        return self.covers & jnp.where(self.at_endpoint, self.stored_ge, self.above)

    def ties_point(self) -> BoolND:
        """Link's value at the query equals the point at working precision."""
        return self.covers & jnp.where(self.at_endpoint, self.stored_eq, self.tied)

    def interior_unresolved(self) -> BoolND:
        """Covering interior chord that may be the true envelope above the point.

        The compensated gap resolves a real excess above the noise floor that
        the conservative margin cannot certify as strictly above. A separate
        stored endpoint tie must not mask it: a node carrying such a chord
        cannot be anchored on the point and is routed to loud overflow. An
        exactly-tied chord (gap within the noise floor) is not unresolved, so a
        genuine node-coincident tie still anchors and publishes the point.
        """
        return self.covers & ~self.at_endpoint & self.unresolved_above


def _certify_chord_point_relations(
    *, query_grid: Float1D, point_value: Float1D, links: _LinkLines
) -> _ChordPointCertificate:
    """Build the certified chord-to-point relations for every (query, link)."""
    query = query_grid[:, None]
    point = point_value[:, None]
    covers = links.live[None, :] & (
        ((links.lower[None, :] < query) & (query <= links.upper[None, :]))
        | ((links.lower[None, :] <= query) & (query < links.upper[None, :]))
    )
    at_lower = query == links.lower[None, :]
    at_upper = query == links.upper[None, :]
    stored_record = jnp.where(
        at_lower, links.lower_value[None, :], links.upper_value[None, :]
    )
    stored_ge = stored_record >= point
    stored_eq = stored_record == point

    value_gap, value_gap_err = _two_sum(links.lower_value[None, :], -point)
    span, span_err = _two_sum(links.upper[None, :], -links.lower[None, :])
    rise, rise_err = _two_sum(links.upper_value[None, :], -links.lower_value[None, :])
    offset, offset_err = _two_sum(query, -links.lower[None, :])
    left_term, left_term_err = _two_product(value_gap, span)
    right_term, right_term_err = _two_product(rise, offset)
    cross_terms = (
        value_gap * span_err
        + value_gap_err * span
        + value_gap_err * span_err
        + rise * offset_err
        + rise_err * offset
        + rise_err * offset_err
    )
    high, high_err = _two_sum(left_term, right_term)
    chord_gap = high + (high_err + left_term_err + right_term_err + cross_terms)
    eps = jnp.finfo(query_grid.dtype).eps
    term_scale = jnp.abs(left_term) + jnp.abs(right_term)
    # The noise floor bounds the compensated evaluation's own residual (a small
    # multiple of the second-order-in-epsilon term error); the certified-above
    # margin sits a further order above it so a certified-above verdict never
    # fires on rounding noise. A gap between the two is resolved as a real
    # excess the margin cannot certify — the unresolved-above band.
    noise = 4.0 * eps * eps * term_scale
    margin = 32.0 * eps * eps * term_scale

    return _ChordPointCertificate(
        covers=covers,
        at_endpoint=at_lower | at_upper,
        stored_ge=stored_ge,
        stored_eq=stored_eq,
        above=chord_gap > margin,
        tied=jnp.abs(chord_gap) <= noise,
        unresolved_above=(chord_gap > noise) & (chord_gap <= margin),
    )


def _branch_envelope_value(
    *, x: Float1D, winner: Int1D, incoming: Int1D, links: _LinkLines
) -> Float1D:
    """Envelope value of the two switching branches at `x`.

    The larger of the outgoing and incoming branch values, each read from its
    nearer stored endpoint. At a crossing the two agree; where the emitted
    abscissa rounds off the intersection the maximum stays at or above the
    envelope, never below both branches.
    """
    reads = links.value_at(x[:, None])
    winner_value = jnp.take_along_axis(reads, winner[:, None], axis=1)[:, 0]
    incoming_value = jnp.take_along_axis(reads, incoming[:, None], axis=1)[:, 0]
    return jnp.maximum(winner_value, incoming_value)


def _crossing_off_intersection(
    *, x: Float1D, winner: Int1D, incoming: Int1D, links: _LinkLines
) -> BoolND:
    """Flag a crossing whose branches do not meet at the rounded abscissa `x`.

    The two switching branches must cross at the emitted abscissa. Their
    compensated line gap above a second-order-in-epsilon margin means the
    rounded abscissa is not the intersection, so the crossing value cannot be a
    certified common value and the interval must fail loudly.
    """
    gap = _compensated_line_gap(
        x=x,
        anchor_value=links.anchor_value[incoming],
        anchor_grid=links.anchor_grid[incoming],
        slope=links.value_slope[incoming],
        winner_anchor_value=links.anchor_value[winner],
        winner_anchor_grid=links.anchor_grid[winner],
        winner_slope=links.value_slope[winner],
    )
    eps = jnp.finfo(x.dtype).eps
    margin = (
        32.0
        * eps
        * eps
        * (
            jnp.abs(links.anchor_value[incoming])
            + jnp.abs(links.value_slope[incoming])
            * jnp.abs(x - links.anchor_grid[incoming])
            + jnp.abs(links.anchor_value[winner])
            + jnp.abs(links.value_slope[winner])
            * jnp.abs(x - links.anchor_grid[winner])
        )
    )
    return jnp.abs(gap) > margin


def _missed_switch_overflow(
    *,
    is_new: BoolND,
    left_side: _SideWinner,
    right_side: _SideWinner,
    crossings: _IntervalCrossings,
) -> ScalarBool:
    """Flag an interval whose owner switched but emitted no breaking crossing.

    The interval bridges its start-of-interval winner (the previous node's right
    side) to its end-of-interval winner (this node's left side) as one linear
    span. When the two differ the envelope owner switched inside the interval,
    so at least one crossing must break the span; if the earliest-overtake scan
    emitted none — a switch near a coincident endpoint whose offset rounds past
    the interval width — the linear read overshoots a representable query, so
    the interval must fail loudly.
    """
    start_link = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), right_side.link[:-1]]
    )
    start_exists = jnp.concatenate(
        [jnp.zeros((1,), dtype=bool), right_side.exists[:-1]]
    )
    interval_active = is_new & left_side.exists & start_exists
    endpoint_owner_switch = start_link != left_side.link
    has_emitted_crossing = jnp.any(crossings.valid, axis=1)
    return jnp.any(interval_active & endpoint_owner_switch & ~has_emitted_crossing)


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
    ) -> tuple[tuple[Int1D, Float1D, BoolND], tuple[_IntervalCrossings, ScalarBool]]:
        winner, x_current, active = carry
        offset, valid = overtaking(winner, x_current)
        valid = valid & active[:, None]
        offset_masked = jnp.where(valid, offset, jnp.inf)
        offset_next = jnp.min(offset_masked, axis=1)
        found = jnp.isfinite(offset_next)
        offset_tolerance = crossing_ulp * jnp.where(found, offset_next, 0.0)
        near_minimal = valid & (offset <= (offset_next + offset_tolerance)[:, None])
        # Offset proximity alone cannot certify a simultaneous crossing: the
        # window scales with the offset itself, so inside it two crossings can
        # sit many representable positions apart, and handing the group to the
        # steepest line would skip the branch that wins between them. A line
        # joins the tie only if it also passes through the same numerical
        # point. Two demands make that test sound:
        # - the line-difference at the emitted crossing is evaluated in
        #   compensated (double-float) arithmetic, so its own rounding cannot
        #   absorb a genuine gap the way the plainly evaluated anchor products
        #   can — their cancellation error scales with the local terms and
        #   once certified a distinct overtake as simultaneous;
        # - the window is the gap a one-ulp abscissa displacement of the
        #   crossing induces, `|slope gap| * ulp(x)`: crossings closer than
        #   one representable abscissa step admit no query between them, so
        #   merging is harmless there, while any wider separation leaves a
        #   representable query the middle branch must win. A false negative
        #   over-emits a duplicated crossing, which the side convention reads
        #   correctly (and capacity overflow is loud); a false positive skips
        #   the branch that wins between two representable crossings.
        x_next_safe = x_current + jnp.where(found, offset_next, 0.0)
        slope_gap_at_next = (
            links.value_slope[None, :] - links.value_slope[winner][:, None]
        )
        gap_at_next = _compensated_line_gap(
            x=x_next_safe[:, None],
            anchor_value=links.anchor_value[None, :],
            anchor_grid=links.anchor_grid[None, :],
            slope=links.value_slope[None, :],
            winner_anchor_value=links.anchor_value[winner][:, None],
            winner_anchor_grid=links.anchor_grid[winner][:, None],
            winner_slope=links.value_slope[winner][:, None],
        )
        abscissa_ulp = jnp.finfo(query_grid.dtype).eps * jnp.abs(x_next_safe[:, None])
        same_point = jnp.abs(gap_at_next) <= (jnp.abs(slope_gap_at_next) * abscissa_ulp)
        # The exact-minimum line(s) define the crossing being emitted, so they
        # belong to the tie unconditionally — the residual test only decides
        # which *other* near-minimal lines genuinely pass through the point.
        is_earliest = valid & (offset <= offset_next[:, None])
        tie = (near_minimal & same_point) | is_earliest
        incoming = jnp.argmax(
            jnp.where(tie, links.value_slope[None, :], -jnp.inf), axis=1
        ).astype(jnp.int32)

        x_next = x_current + offset_next
        # A crossing whose emitted abscissa rounds onto a candidate node (the
        # interval's right endpoint) or onto the current position is a node
        # event, not an interior switch: the node's one-sided records own that
        # abscissa, and the reconstructed crossing value there is a rounded
        # affine product that can exceed both stored branches — an emitted
        # third record would overstate the envelope at an actual node. Two
        # cases, neither emitting a record:
        # - landing on the right node exhausts the interval (any later switch
        #   would lie beyond it), so the scan ends there;
        # - landing on the current position (the positive offset rounds to no
        #   representable advance) hands the running winner to the incoming
        #   line and keeps scanning — no representable query separates the
        #   true crossing from the position, so the ownership handoff is the
        #   only observable effect.
        lands_on_node = found & (x_next >= query_grid)
        lands_in_place = found & ~lands_on_node & (x_next <= x_current)
        emit = found & ~lands_on_node & ~lands_in_place
        # The published crossing value is the envelope at the emitted abscissa —
        # the larger of the two switching branches read from their nearer stored
        # endpoints — never the outgoing line's own far-anchored product, which
        # can round below both branches when the anchor is distant.
        crossing_value = _branch_envelope_value(
            x=x_next, winner=winner, incoming=incoming, links=links
        )
        emitted = _IntervalCrossings(
            grid=jnp.where(emit, x_next, jnp.nan),
            value=crossing_value,
            policy_left=links.anchor_policy[winner]
            + links.policy_slope[winner] * (x_next - links.anchor_grid[winner]),
            policy_right=links.anchor_policy[incoming]
            + links.policy_slope[incoming] * (x_next - links.anchor_grid[incoming]),
            valid=emit,
        )
        # An emitted crossing whose rounded abscissa is not the true
        # intersection cannot be published as a certified envelope row: the two
        # switching branches must meet there. A compensated line gap above a
        # second-order-in-epsilon margin flags a rounded abscissa off the
        # intersection, routing the interval to loud overflow rather than a
        # silently wrong value payload.
        uncertifiable = jnp.any(
            emit
            & _crossing_off_intersection(
                x=x_next, winner=winner, incoming=incoming, links=links
            )
        )

        new_winner = jnp.where(found & ~lands_on_node, incoming, winner).astype(
            jnp.int32
        )
        new_x = jnp.where(emit, x_next, x_current)
        return (new_winner, new_x, active & found & ~lands_on_node), (
            emitted,
            uncertifiable,
        )

    carry_init = (init_winner, prev_grid, interval_active)
    carry_final, (rows, uncertifiable) = jax.lax.scan(
        step,
        carry_init,
        jnp.arange(_MAX_CROSSINGS_PER_INTERVAL, dtype=jnp.int32),
    )
    winner, x_current, active = carry_final
    _, leftover_valid = overtaking(winner, x_current)
    interval_overflow = jnp.any(leftover_valid & active[:, None]) | jnp.any(
        uncertifiable
    )

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
