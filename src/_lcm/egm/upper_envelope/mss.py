r"""MSS upper-envelope refinement of EGM candidates (HARK's EGM upper envelope).

Implements the upper-envelope method of HARK (Carroll et al. 2018), referenced
in Dobrescu & Shanker (2024) as the `MSS` method column. Inverting the Euler
equation in models with discrete choices yields a value *correspondence*: the
candidates form a chain of linear segments between consecutive nodes that, in
non-concave regions, overlap in the endogenous grid. MSS sweeps the common grid
left-to-right and, at each output abscissa, evaluates every currently
overlapping segment, keeps the max-value branch, and — where the winning branch
switches between two adjacent abscissae — inserts the exact segment-crossing
point (the kink).

Inserting the crossing is what separates MSS from LTM: both evaluate the
envelope at the candidate abscissae, but MSS adds the intersection abscissa as
its own node, so the kink is placed exactly rather than smeared across the local
grid spacing. The refined arrays therefore track the FUES envelope tightly. The
sweep is a single left-to-right scan over the sorted abscissae — `O(N)` emitting
steps, each evaluating the segment set — in contrast to RFC's per-pair dominance
test, and unlike FUES it consumes no `jump_thresh` heuristic: the winner switch
is read directly off the evaluated values.

A crossing abscissa is inserted twice — same abscissa, left- and
right-extrapolated policy — so the refined arrays stay weakly ascending and the
policy discontinuity at a discrete-choice switch is preserved exactly, the same
convention FUES uses.

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, Int1D, ScalarInt

# Band within which the dense envelope evaluated at a crossing abscissa and the
# crossing's own value count as the same number. They are one mathematical
# quantity reached by two routes -- a maximum over segment lines, and a two-line
# intersection solve whose division amplifies the inputs' rounding -- so the band
# is set by how far those routes can drift, not by the format's resolution.
_ON_ENVELOPE_RTOL = 1e-5
_ON_ENVELOPE_ATOL = 1e-7
# ...but it can never be *narrower* than the format can resolve. The relative
# term vanishes near zero, leaving the absolute term to decide there, and a
# fixed `1e-7` sits below float32's epsilon (`1.19e-7`): two readings agreeing
# to the last bit the format has would still fall outside it, so whether a
# crossing is emitted would stop being a property of the geometry.
_ON_ENVELOPE_EPS_MULTIPLE = 8.0


def _on_envelope_atol(reference: FloatND) -> FloatND:
    """Return the absolute band for "this crossing lies on the envelope".

    The declared tolerance, floored at a small multiple of the working dtype's
    epsilon. At double precision the floor never binds and the band is the
    declared constant; at single precision the floor is what makes the
    comparison decidable at all.
    """
    return jnp.maximum(
        _ON_ENVELOPE_ATOL, _ON_ENVELOPE_EPS_MULTIPLE * jnp.finfo(reference.dtype).eps
    )


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its upper envelope.

    The candidates arrive as a chain of consecutive linear segments (one segment
    per consecutive input pair), as the Euler inversion produces them: the
    constrained run followed by the interior run, each ascending along its own
    margin but jointly non-monotone in the endogenous grid. The abscissae are
    sorted ascending and swept left-to-right; at each abscissa the highest
    bracketing segment is kept, and where the winning segment switches between
    two adjacent abscissae the exact crossing point is inserted (twice — left and
    right policy). The refined arrays have static length `n_refined`, hold the
    envelope points in weakly ascending grid order, and are NaN-padded in the
    tail.

    Args:
        endog_grid: Candidate endogenous grid points (resources). Consecutive
            entries form the linear segments scanned for the envelope.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        segment_id: Optional per-candidate branch label, aligned with
            `endog_grid`. When supplied, a consecutive-pair link is a real value
            segment iff both endpoints carry the same label, so unrelated branches
            are never bridged, and the topology is declared rather than inferred.
            `None` (the default) infers it from a grid or value decrease past a
            noise floor — HARK's monotone split.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow; the arrays then hold a
        valid truncated prefix of the envelope. Callers must check the counter
        rather than publish the truncated arrays silently — the EGM step
        NaN-poisons its published rows on overflow so the solve loop's NaN
        diagnostics name the offending (regime, period).

    """
    # A dead candidate arrives NaN-filled (the EGM step poisons `-inf`-valued
    # corners to NaN before refinement). A segment touching a dead endpoint is
    # excluded from the scan, and a dead abscissa sorts to the NaN tail, so it
    # is neither queried nor evaluated.
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)

    # Sort the candidate abscissae ascending so the sweep is left-to-right and
    # the NaN tail is contiguous; dead nodes sort last.
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
    right_policy = policy[1:]
    left_value = value[:-1]
    right_value = value[1:]

    # Per-link branch id and live mask. With explicit topology a link is a real
    # value segment iff both endpoints carry the same branch label, so unrelated
    # branches are never bridged. Without it, fall back to HARK's monotone split:
    # a new segment starts wherever the grid decreases, or the value decreases past
    # the noise floor, between consecutive candidates, and a link spanning such a
    # decrease is a non-monotone bridge excluded from the scan. Either way the
    # winner stays constant across one branch, so only a genuine branch switch is a
    # kink.
    if segment_id is None:
        decreases = (right_grid < left_grid) | _value_decrease_past_noise(
            left_value=left_value, right_value=right_value
        )
        link_segment = jnp.cumsum(decreases.astype(jnp.int32))
        segment_live = ~dead[:-1] & ~dead[1:] & ~decreases
    else:
        same_segment = segment_id[:-1] == segment_id[1:]
        link_segment = segment_id[:-1].astype(jnp.int32)
        segment_live = ~dead[:-1] & ~dead[1:] & same_segment

    envelope_value, envelope_policy, winner_link, winner_segment = _evaluate_envelope(
        query_grid=query_grid,
        left_grid=left_grid,
        right_grid=right_grid,
        left_policy=left_policy,
        right_policy=right_policy,
        left_value=left_value,
        right_value=right_value,
        segment_id=link_segment,
        segment_live=segment_live,
    )

    # A query no segment brackets (e.g. the lone dead-padded tail) yields no
    # envelope value: poison the whole triple to NaN so it joins the tail.
    no_segment = jnp.isneginf(envelope_value)
    query_drop = query_dead | no_segment
    node_grid = jnp.where(query_drop, jnp.nan, query_grid)
    node_policy = jnp.where(query_drop, jnp.nan, envelope_policy)
    node_value = jnp.where(query_drop, jnp.nan, envelope_value)

    # Sweep left-to-right: each step emits its query node, then — if the winning
    # *branch* (segment id) differs from the previous live query's winner — the
    # exact crossing of the two winning links (twice, left and right policy). The
    # first live query has no predecessor, so it emits only its node.
    crossing = _crossing_blocks(
        query_grid=query_grid,
        query_drop=query_drop,
        winner_link=winner_link,
        winner_segment=winner_segment,
        left_grid=left_grid,
        right_grid=right_grid,
        left_policy=left_policy,
        right_policy=right_policy,
        left_value=left_value,
        right_value=right_value,
    )

    # A crossing is a genuine envelope kink only if both branches are on the
    # envelope there: evaluate the dense envelope at each crossing abscissa and
    # keep the crossing only where that value is finite and equals the crossing
    # value. A switch across a *gap* in the candidate cloud intersects two lines
    # in an interval no segment covers — the dense envelope is `-inf` there, so
    # the test rejects it; colinear adjacent links of one monotone branch pass,
    # since the branch's own line is on the envelope at the crossing.
    crossing_envelope, _, _, _ = _evaluate_envelope(
        query_grid=crossing.grid,
        left_grid=left_grid,
        right_grid=right_grid,
        left_policy=left_policy,
        right_policy=right_policy,
        left_value=left_value,
        right_value=right_value,
        segment_id=link_segment,
        segment_live=segment_live,
    )
    on_envelope = jnp.isfinite(crossing_envelope) & jnp.isclose(
        crossing_envelope,
        crossing.value,
        atol=_on_envelope_atol(crossing_envelope),
        rtol=_ON_ENVELOPE_RTOL,
    )
    crossing_valid = crossing.valid & on_envelope

    # Per-query output block: up to three rows in ascending grid order — the two
    # crossing copies (same abscissa, left then right policy) followed by the
    # query node. The crossing of step `i` lies in `(grid_{i-1}, grid_i)`, so it
    # precedes the node `i` in the weakly-ascending output.
    nan_scalar = jnp.full((), jnp.nan, dtype=query_grid.dtype)
    node_valid = ~query_drop
    # Per-query emission rows in ascending grid order: the crossing inserted
    # before the node (left then right policy copy, same abscissa), then the node
    # itself. The crossing of step `i` lies in `(grid_{i-1}, grid_i)`, so it
    # precedes node `i`. Each row carries its own validity flag.
    row_valid = jnp.stack([crossing_valid, crossing_valid, node_valid], axis=1).ravel()
    row_grid = jnp.stack([crossing.grid, crossing.grid, node_grid], axis=1).ravel()
    row_policy = jnp.stack(
        [crossing.policy_left, crossing.policy_right, node_policy], axis=1
    ).ravel()
    row_value = jnp.stack([crossing.value, crossing.value, node_value], axis=1).ravel()

    # Compact the valid rows into the NaN-padded prefix, preserving sweep order.
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
    return out_grid, out_policy, out_value, n_kept


@dataclass(frozen=True, kw_only=True)
class _CrossingBlocks:
    """Per-query crossing candidate: abscissa, value, both policy copies, flag.

    The fields are aligned with the query sweep: entry `i` is the crossing
    inserted *before* query node `i`, present (`valid`) only when query `i`'s
    live winning segment differs from the previous live query's winner and the
    intersection abscissa falls strictly between the two queries.
    """

    grid: Float1D
    """Crossing abscissa per query."""
    value: Float1D
    """Envelope value at the crossing."""
    policy_left: Float1D
    """Policy of the outgoing owner at the crossing."""
    policy_right: Float1D
    """Policy of the incoming owner at the crossing."""
    valid: BoolND
    """Whether this query contributes a crossing at all."""


def _crossing_blocks(
    *,
    query_grid: Float1D,
    query_drop: BoolND,
    winner_link: Int1D,
    winner_segment: Int1D,
    left_grid: Float1D,
    right_grid: Float1D,
    left_policy: Float1D,
    right_policy: Float1D,
    left_value: Float1D,
    right_value: Float1D,
) -> _CrossingBlocks:
    """Compute, per query, the crossing of its winner with the previous winner.

    Sweeps the live queries left-to-right (carrying the previous live query's
    winning link and branch id) and, whenever the winning *branch* (segment id)
    switches, intersects the two winning links' value lines. The crossing is
    kept only when its abscissa falls strictly between the two adjacent live
    query abscissae — the interval the switch happened in. A move from one link
    to the next within one monotone branch keeps the segment id, so it is not a
    switch and inserts nothing; only a genuine branch change is a kink.
    """
    n_query = query_grid.shape[0]
    live = ~query_drop

    def step(
        carry: tuple[ScalarInt, ScalarInt, FloatND], idx: ScalarInt
    ) -> tuple[tuple[ScalarInt, ScalarInt, FloatND], _CrossingRow]:
        prev_link, prev_segment, prev_grid = carry
        is_live = live[idx]
        this_link = winner_link[idx]
        this_segment = winner_segment[idx]
        this_grid = query_grid[idx]

        switches = is_live & (prev_segment >= 0) & (this_segment != prev_segment)
        row = _intersect_winners(
            seg_a=prev_link,
            seg_b=this_link,
            left_grid=left_grid,
            right_grid=right_grid,
            left_policy=left_policy,
            right_policy=right_policy,
            left_value=left_value,
            right_value=right_value,
        )
        # The crossing must fall strictly inside the switch interval; whether it
        # is a genuine envelope kink (rather than a phantom intersection across a
        # gap) is settled by the dense on-envelope test the caller applies.
        within = switches & (row.grid > prev_grid) & (row.grid < this_grid)
        emitted = _CrossingRow(
            grid=row.grid,
            value=row.value,
            policy_left=row.policy_a,
            policy_right=row.policy_b,
            valid=within,
        )

        # Advance the previous-live-query carry only on a live query; a dropped
        # query leaves the comparison anchored at the last live winner/abscissa.
        new_link = jnp.where(is_live, this_link, prev_link).astype(jnp.int32)
        new_segment = jnp.where(is_live, this_segment, prev_segment).astype(jnp.int32)
        new_grid = jnp.where(is_live, this_grid, prev_grid)
        return (new_link, new_segment, new_grid), emitted

    carry_init = (
        jnp.int32(0),
        jnp.int32(-1),
        jnp.asarray(-jnp.inf, dtype=query_grid.dtype),
    )
    _, rows = jax.lax.scan(step, carry_init, jnp.arange(n_query, dtype=jnp.int32))
    return _CrossingBlocks(
        grid=rows.grid,
        value=rows.value,
        policy_left=rows.policy_left,
        policy_right=rows.policy_right,
        valid=rows.valid,
    )


@dataclass(frozen=True, kw_only=True)
class _CrossingRow:
    """One step's crossing emission, stacked by the scan into row arrays."""

    grid: FloatND
    """Crossing abscissa emitted at this step."""
    value: FloatND
    """Envelope value at that abscissa."""
    policy_left: FloatND
    """Policy of the outgoing owner."""
    policy_right: FloatND
    """Policy of the incoming owner."""
    valid: BoolND
    """Whether this step emits a crossing."""


_CROSSING_ROW_FIELDS = ("grid", "value", "policy_left", "policy_right", "valid")


def _flatten_crossing_row(row: _CrossingRow) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(row, name) for name in _CROSSING_ROW_FIELDS), None


def _unflatten_crossing_row(_aux: None, children: Sequence[Any]) -> _CrossingRow:
    row = object.__new__(_CrossingRow)
    for name, child in zip(_CROSSING_ROW_FIELDS, children, strict=True):
        object.__setattr__(row, name, child)
    return row


# The scan stacks this row's leaves, so it must be a pytree. A frozen dataclass
# is not one by construction; registering it is the same shape `EGMCarry` and
# the params leaves use.
jax.tree_util.register_pytree_node(
    _CrossingRow, _flatten_crossing_row, _unflatten_crossing_row
)


@dataclass(frozen=True, kw_only=True)
class _SegmentIntersection:
    """Intersection of two winning segments' value lines, with both policies."""

    grid: FloatND
    """Abscissa where the two value lines meet."""
    value: FloatND
    """Common value there."""
    policy_a: FloatND
    """Policy of the first segment at the intersection."""
    policy_b: FloatND
    """Policy of the second segment at the intersection."""


def _intersect_winners(
    *,
    seg_a: ScalarInt,
    seg_b: ScalarInt,
    left_grid: Float1D,
    right_grid: Float1D,
    left_policy: Float1D,
    right_policy: Float1D,
    left_value: Float1D,
    right_value: Float1D,
) -> _SegmentIntersection:
    """Intersect the value lines of segments `seg_a` and `seg_b`.

    Each segment is the line through its two endpoints; the intersection
    abscissa solves `v_a(x) = v_b(x)`. The policy is read off each segment's own
    line at that abscissa, so `policy_a` is the left-branch (segment `a`) policy
    and `policy_b` the right-branch (segment `b`) policy at the kink.
    """
    a_x0, a_x1 = left_grid[seg_a], right_grid[seg_a]
    a_v0, a_v1 = left_value[seg_a], right_value[seg_a]
    a_p0, a_p1 = left_policy[seg_a], right_policy[seg_a]
    b_x0, b_x1 = left_grid[seg_b], right_grid[seg_b]
    b_v0, b_v1 = left_value[seg_b], right_value[seg_b]
    b_p0, b_p1 = left_policy[seg_b], right_policy[seg_b]

    slope_a = _slope(x_a=a_x0, y_a=a_v0, x_b=a_x1, y_b=a_v1)
    slope_b = _slope(x_a=b_x0, y_a=b_v0, x_b=b_x1, y_b=b_v1)
    grid, value = _intersect_lines(
        x_a=a_x0, y_a=a_v0, slope_a=slope_a, x_b=b_x0, y_b=b_v0, slope_b=slope_b
    )

    policy_slope_a = _slope(x_a=a_x0, y_a=a_p0, x_b=a_x1, y_b=a_p1)
    policy_slope_b = _slope(x_a=b_x0, y_a=b_p0, x_b=b_x1, y_b=b_p1)
    policy_a = a_p0 + policy_slope_a * (grid - a_x0)
    policy_b = b_p0 + policy_slope_b * (grid - b_x0)
    return _SegmentIntersection(
        grid=grid, value=value, policy_a=policy_a, policy_b=policy_b
    )


def _evaluate_envelope(
    *,
    query_grid: Float1D,
    left_grid: Float1D,
    right_grid: Float1D,
    left_policy: Float1D,
    right_policy: Float1D,
    left_value: Float1D,
    right_value: Float1D,
    segment_id: Int1D,
    segment_live: BoolND,
) -> tuple[Float1D, Float1D, Int1D, Int1D]:
    """Evaluate the upper envelope and its winning link/branch at every query.

    Builds the dense `(N_query, N_segments)` bracket-and-interpolate matrix: each
    query `m_j` is tested against every link `(k, k+1)`. A link brackets the
    query iff `m_j` lies in its abscissa range; the value and policy are then
    linearly interpolated along the link. The envelope value is the maximum over
    bracketing links, the policy is the winner's, the winning link index is
    reported (so its value line can be intersected), and the winner's branch id
    `segment_id` is reported (so the sweep can detect a branch switch). A query
    no link brackets reports `-inf` value (the absent-envelope sentinel), winning
    link `0`, and branch `-1`.

    Args:
        query_grid: Abscissae at which to evaluate the envelope; NaN tail.
        left_grid: Lower endpoint abscissa of each link.
        right_grid: Upper endpoint abscissa of each link.
        left_policy: Policy at each link's lower endpoint.
        right_policy: Policy at each link's upper endpoint.
        left_value: Value at each link's lower endpoint.
        right_value: Value at each link's upper endpoint.
        segment_id: Per-link monotone-branch id; equal across one branch.
        segment_live: Per-link live indicator; a dead-endpoint or non-monotone
            bridge link is excluded from the scan.

    Returns:
        Tuple of the envelope value, the envelope policy, the winning link index
        (`0` where no link brackets), and the winning branch id (`-1` where no
        link brackets) at each query.

    """
    query = query_grid[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = segment_live[None, :] & (query >= lower) & (query <= upper)

    # Linear position of the query along each segment; a zero-width segment takes
    # weight 0, so its left endpoint applies.
    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (query - left_grid[None, :]) / safe_width)

    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )

    # A link whose interpolated value is not finite is excluded rather than let
    # into the reduction: `jnp.max` propagates a NaN across the whole row, which
    # would publish NaN as live envelope data at every node the link covers. Losing
    # only itself leaves the node to any other link that covers it, and a node no
    # finite link covers reports `-inf` and joins the dropped tail.
    masked_value = jnp.where(
        brackets & jnp.isfinite(value_interp), value_interp, -jnp.inf
    )
    envelope_value = jnp.max(masked_value, axis=1)
    # Select the winning segment deterministically: among segments whose
    # interpolated value sits within a scale-aware band of the max, take the
    # lowest index. A raw first-index `argmax` over independently interpolated
    # values flips on a 1-ulp tie (two segments crossing at a node, with the
    # bracket endpoints carrying upstream reduction noise), which would make the
    # kept segment — hence `winner_segment`, the crossing decision, and `n_kept`
    # — backend-dependent. The band makes the near-tie choice reduction-order
    # independent.
    #
    # Link order is therefore the last tiebreak, and it is the *caller's* order:
    # two colinear value lines from different discrete choices are separated by
    # nothing else here, so the one concatenated first wins. That is arbitrary but
    # reproducible — `jnp.argsort` above is stable, so the same inputs always
    # produce the same order. Any rule would be arbitrary; a policy-based one would
    # impose an ordering on a quantity with no envelope meaning.
    tie_floor = (
        16.0
        * jnp.finfo(value_interp.dtype).eps
        * jnp.maximum(1.0, jnp.abs(envelope_value))
    )
    in_tie = brackets & (masked_value >= envelope_value[:, None] - tie_floor[:, None])
    best_link = jnp.argmax(in_tie, axis=1).astype(jnp.int32)
    any_bracket = jnp.any(brackets, axis=1)
    envelope_policy = jnp.take_along_axis(policy_interp, best_link[:, None], axis=1)[
        :, 0
    ]
    winner_link = jnp.where(any_bracket, best_link, 0).astype(jnp.int32)
    winner_segment = jnp.where(any_bracket, segment_id[best_link], -1).astype(jnp.int32)
    return envelope_value, envelope_policy, winner_link, winner_segment


def _value_decrease_past_noise(*, left_value: Float1D, right_value: Float1D) -> BoolND:
    """Report whether the value genuinely falls from one candidate to the next.

    Without an explicit `segment_id` the branch boundaries have to be inferred, and
    a value decrease is the signal. Along a near-linear tail, though, the sign of
    the difference between two consecutive values is set by rounding rather than by
    economics — the same shape the FUES savings-monotonicity test guards against —
    and reading such a decrease as a boundary drops the whole run above it from the
    envelope. The floor `16 * eps * max(|left|, |right|)` scales with the values in
    play, so it masks a genuine decrease only when the two values are
    indistinguishable at the working precision.

    Returns:
        Per-link mask, true where the right value falls below the left one by more
        than the noise floor.

    """
    scale = jnp.maximum(jnp.abs(left_value), jnp.abs(right_value))
    noise_floor = 16.0 * jnp.finfo(left_value.dtype).eps * scale
    return right_value < left_value - noise_floor


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


def _intersect_lines(
    *,
    x_a: FloatND,
    y_a: FloatND,
    slope_a: FloatND,
    x_b: FloatND,
    y_b: FloatND,
    slope_b: FloatND,
) -> tuple[FloatND, FloatND]:
    """Intersect two lines given in point-slope form.

    Args:
        x_a: Abscissa of a point on the first line.
        y_a: Ordinate of a point on the first line.
        slope_a: Slope of the first line.
        x_b: Abscissa of a point on the second line.
        y_b: Ordinate of a point on the second line.
        slope_b: Slope of the second line.

    Returns:
        Tuple of the intersection's abscissa and ordinate; NaN abscissa for
        parallel lines.

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
