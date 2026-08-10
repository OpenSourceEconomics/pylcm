"""Exact query-side upper envelope of an EGM candidate correspondence.

The query-side counterpart of the full-row refiners (`fues`, `rfc`, `ltm`,
`mss`). Those materialise the whole refined envelope row and the caller then
reads it at a query; this evaluates the envelope *directly* at a set of query
abscissae without ever building the row.

For one query the value is the maximum, over every live branch segment that
brackets it, of the segment's linear value; the policy and marginal are the
winning segment's. A folded branch contributes several bracketing segments, so
the maximum is exact for the piecewise-linear correspondence. Topology is
explicit: a segment is the link between two consecutive candidates carrying the
same `segment_id`, so unrelated branches are never bridged — the contract the
host oracle enforces.

By default the evaluation is a fixed-shape `(n_query, n_segment)`
bracket-and-reduce: no sequential scan, no NaN-padded refined row,
branch-parallel and reduction-heavy, which is the shape an accelerator runs
fastest. This is the backend asset-row mode wants — one query per Euler node, no
full envelope to refine. For a large `(n_query, n_segment)` that dense matrix is
itself the memory wall; `segment_block_size` swaps it for a two-pass blocked
scan over segment blocks (running max, then max-slope-among-near-max against the
fixed envelope value), which peaks at `(n_query, block)` instead of
`(n_query, n_segment)` and returns the identical result.
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import (
    BELOW_RESOLUTION_SIGN,
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from lcm.typing import BoolND, Float1D, FloatND, IntND

# Right-continuous tie tolerance (relative). This settles *only* the case the
# certified predicate has already declared an exact tie: among segments certified
# equal to the owner at the query, the larger value-slope wins, because it is the
# one that is higher just to the right. It never decides which segment is higher —
# that is `certified_margin_sign`'s job, and a magnitude-scaled band cannot do it,
# being not invariant to a common additive value level.
_VALUE_TIE_ATOL = 1e-12

# How many times a provisional owner may be replaced before the query is declared
# unresolved. Each round validates the standing owner against every bracketing
# contender, so one round suffices whenever the plain-float maximum is already the
# certified owner — the ordinary case. The bound exists so a cyclic or
# non-transitive comparison surfaces as a loud NaN rather than spinning.
_CERTIFIED_PROMOTION_ROUNDS = 3


def _value_tie_band(reference: FloatND) -> FloatND:
    """Return the scale-aware absolute tie band around a reference envelope value."""
    return _VALUE_TIE_ATOL * jnp.maximum(1.0, jnp.abs(reference))


class _ComparableLines(NamedTuple):
    """Per-segment affine lines in the form `certified_margin_sign` accepts.

    The predicate compares two lines through their endpoints and requires a
    strictly positive width from each. A zero-width self-bracket is a point, so it
    is presented as the constant line through its own value: equal endpoint values
    over a positive synthetic width, which takes that value at every abscissa and
    so agrees with the point exactly where the point is defined.
    """

    x0: FloatND
    x1: FloatND
    v0: FloatND
    v1: FloatND


def _as_comparable_lines(links: _SegmentLinks) -> _ComparableLines:
    """Return each link as a line the certified predicate can read."""
    lower = jnp.minimum(links.left_grid, links.right_grid)
    upper = jnp.maximum(links.left_grid, links.right_grid)
    degenerate = upper <= lower
    # Any positive width represents the same constant line, so the scale is chosen
    # only to stay representable beside the abscissae it sits among.
    synthetic = jnp.maximum(jnp.abs(lower), 1.0)
    low_value = jnp.where(
        links.left_grid <= links.right_grid, links.left_value, links.right_value
    )
    high_value = jnp.where(
        links.left_grid <= links.right_grid, links.right_value, links.left_value
    )
    return _ComparableLines(
        x0=lower,
        x1=jnp.where(degenerate, lower + synthetic, upper),
        v0=low_value,
        v1=jnp.where(degenerate, low_value, high_value),
    )


def _owner_against_contenders(
    *, lines: _ComparableLines, owner: _ComparableLines, brackets: BoolND, flat: Float1D
) -> IntND:
    """Return the certified sign of `owner - contender` for every bracketing pair.

    Shaped `(n_query, n_segment)`. Non-bracketing entries are reported `+1`: they
    are not contenders, so they can neither unseat the owner nor make it unresolved.
    """
    sign = certified_margin_sign(
        a_x0=owner.x0[:, None],
        a_x1=owner.x1[:, None],
        a_v0=owner.v0[:, None],
        a_v1=owner.v1[:, None],
        b_x0=lines.x0[None, :],
        b_x1=lines.x1[None, :],
        b_v0=lines.v0[None, :],
        b_v1=lines.v1[None, :],
        x_query=flat[:, None],
    )
    return jnp.where(brackets, sign, jnp.ones_like(sign))


def _certified_owner(
    *,
    lines: _ComparableLines,
    brackets: BoolND,
    flat: Float1D,
    provisional: IntND,
    rank: FloatND,
) -> tuple[IntND, BoolND]:
    """Return the certified owning segment per query, and where none is certain.

    The plain-float maximum supplies a provisional owner, which is then validated
    against *every* bracketing contender: if one is certified above it, that
    contender takes over and the validation repeats. Ordinarily the provisional
    owner is already the certified one and the first round confirms it.

    A query is unresolved where the standing owner is still beaten after the last
    round — a comparison set that is cyclic or non-transitive — or where any
    bracketing contender's comparison could not be certified. Its channels are
    poisoned rather than guessed.

    Among contenders certified *exactly equal* to the owner, and only among those,
    the documented right-continuous rule picks: a segment extending strictly right
    of the query over one ending at it, then the larger value-slope.
    """
    owner = provisional
    for _ in range(_CERTIFIED_PROMOTION_ROUNDS):
        sign = _owner_against_contenders(
            lines=lines, owner=_take_lines(lines, owner), brackets=brackets, flat=flat
        )
        beaten = sign == -1
        challenger = jnp.argmax(beaten, axis=1).astype(jnp.int32)
        owner = jnp.where(jnp.any(beaten, axis=1), challenger, owner)

    sign = _owner_against_contenders(
        lines=lines, owner=_take_lines(lines, owner), brackets=brackets, flat=flat
    )
    uncertain = (sign == UNRESOLVED_SIGN) | (sign == BELOW_RESOLUTION_SIGN)
    unresolved = jnp.any(sign == -1, axis=1) | jnp.any(uncertain, axis=1)

    index = jnp.arange(lines.x0.shape[0])[None, :]
    tied = (sign == 0) | (index == owner[:, None])
    best = jnp.argmax(jnp.where(tied, rank, -jnp.inf), axis=1).astype(jnp.int32)
    return best, unresolved


def _take_lines(lines: _ComparableLines, index: IntND) -> _ComparableLines:
    """Return the line each query's index selects."""
    return _ComparableLines(*(jnp.take(column, index) for column in lines))


class _SegmentLinks(NamedTuple):
    """Per-link endpoints of the candidate correspondence (length `n - 1`).

    A link is the consecutive-candidate pair `(i, i+1)`; it is a real envelope
    segment only where `live` (both endpoints finite and sharing a branch label).
    """

    left_grid: Float1D
    right_grid: Float1D
    left_value: Float1D
    right_value: Float1D
    left_policy: Float1D
    right_policy: Float1D
    left_marginal: Float1D
    right_marginal: Float1D
    live: BoolND


def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    x_query: FloatND,
    segment_block_size: int = 0,
    arithmetic: Literal["certified", "ordinary"] = "certified",
) -> tuple[FloatND, FloatND, FloatND]:
    """Evaluate the branch-aware upper envelope at each query abscissa.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order
            within a branch; a NaN entry is a dead/padding candidate.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        marginal: Candidate marginal values (the supgradient) at `endog_grid`.
        segment_id: Per-candidate branch label. A segment is a consecutive-pair
            link whose endpoints share a label, so unrelated branches never join.
        x_query: Abscissae at which to evaluate the envelope.
        segment_block_size: When `0` (or at least the number of segments), the
            dense `(n_query, n_segment)` reduction. A positive value below the
            segment count instead runs the two-pass blocked scan, peaking at
            `(n_query, segment_block_size)`; the result is identical.
        arithmetic: How ownership is decided between bracketing segments.
            - `"certified"` settles it on the exact sign of the difference, so a
              strictly lower branch can never supply a channel and a comparison
              that cannot be certified yields NaN rather than a guess.
            - `"ordinary"` settles it on the plain-float maximum with a
              magnitude-scaled tie band. Cheaper, and wrong exactly where the
              band is wider than a real difference, which is what the certified
              route exists to prevent. Only available for the dense reduction.
            Chosen at trace time, so `"ordinary"` emits none of the error-free
            transforms rather than computing both and selecting.

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`. A
        query no live segment brackets yields NaN in all three, as does one whose
        ownership `"certified"` could not settle.

    Raises:
        ValueError: If `arithmetic="ordinary"` is combined with the blocked scan,
            which carries the certified comparison only. Serving the certified
            cost under the ordinary label would misreport what was paid for.
    """
    if arithmetic not in {"certified", "ordinary"}:
        msg = f"arithmetic must be 'certified' or 'ordinary', not {arithmetic!r}"
        raise ValueError(msg)
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)
    # A link is a real segment only within one branch: both endpoints live and
    # carrying the same label.
    consecutive = _SegmentLinks(
        left_grid=endog_grid[:-1],
        right_grid=endog_grid[1:],
        left_value=value[:-1],
        right_value=value[1:],
        left_policy=policy[:-1],
        right_policy=policy[1:],
        left_marginal=marginal[:-1],
        right_marginal=marginal[1:],
        live=~dead[:-1] & ~dead[1:] & (segment_id[:-1] == segment_id[1:]),
    )
    # Every live candidate is also a zero-width self-bracket at its own abscissa,
    # so a lone point — a folded-out or boundary-collapsed candidate with no
    # consecutive same-segment neighbour — stays visible where a query lands on
    # it, instead of collapsing to a lower multi-point branch. A right-extending
    # consecutive link outranks a zero-width self-bracket in the right-continuous
    # tie-break, so multi-point chains and their interpolation are unchanged; a
    # self-bracket wins only where nothing brackets the query from the right.
    self_bracket = _SegmentLinks(
        left_grid=endog_grid,
        right_grid=endog_grid,
        left_value=value,
        right_value=value,
        left_policy=policy,
        right_policy=policy,
        left_marginal=marginal,
        right_marginal=marginal,
        live=~dead,
    )
    links = _SegmentLinks(
        *(
            jnp.concatenate([pair, point])
            for pair, point in zip(consecutive, self_bracket, strict=True)
        )
    )

    query = jnp.asarray(x_query)
    n_segment = links.left_grid.shape[0]
    if 0 < segment_block_size < n_segment:
        if arithmetic == "ordinary":
            msg = (
                "the blocked scan carries the certified comparison only; "
                "request arithmetic='certified' or drop segment_block_size"
            )
            raise ValueError(msg)
        return _envelope_at_query_blocked(
            links=links, query=query, block_size=segment_block_size
        )

    flat = query.reshape(-1)[:, None]
    left_grid, right_grid = links.left_grid, links.right_grid
    left_value, right_value = links.left_value, links.right_value
    left_policy, right_policy = links.left_policy, links.right_policy
    left_marginal, right_marginal = links.left_marginal, links.right_marginal
    segment_live = links.live
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = segment_live[None, :] & (flat >= lower) & (flat <= upper)

    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (flat - left_grid[None, :]) / safe_width)
    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )
    marginal_interp = (
        left_marginal[None, :] + relative * (right_marginal - left_marginal)[None, :]
    )

    masked_value = jnp.where(brackets, value_interp, -jnp.inf)
    any_bracket = jnp.any(brackets, axis=1)
    # The plain-float maximum is only a provisional owner. It is the right starting
    # point — it is the certified owner almost always — but it cannot settle a pair
    # whose difference is smaller than the level they sit on, which is exactly where
    # a strictly lower branch would otherwise be published.
    provisional = jnp.argmax(masked_value, axis=1).astype(jnp.int32)
    slope = (right_value - left_value)[None, :] / safe_width
    rank = _right_continuous_rank(right_available=flat < upper, slope=slope)
    if arithmetic == "certified":
        best, unresolved = _certified_owner(
            lines=_as_comparable_lines(links),
            brackets=brackets,
            flat=flat.reshape(-1),
            provisional=provisional,
            rank=rank,
        )
    else:
        # The plain-float route: the maximum decides, and a magnitude-scaled band
        # around it decides the rest. Nothing here can be unresolved, because
        # nothing here is certified.
        max_value = jnp.max(masked_value, axis=1, keepdims=True)
        near_max = brackets & (masked_value >= max_value - _value_tie_band(max_value))
        best = jnp.argmax(jnp.where(near_max, rank, -jnp.inf), axis=1).astype(jnp.int32)
        unresolved = jnp.zeros(best.shape, dtype=bool)

    published = any_bracket & ~unresolved
    env_value = jnp.where(
        published,
        jnp.take_along_axis(value_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    env_policy = jnp.where(
        published,
        jnp.take_along_axis(policy_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    env_marginal = jnp.where(
        published,
        jnp.take_along_axis(marginal_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )


def _right_continuous_rank(*, right_available: BoolND, slope: FloatND) -> FloatND:
    """Return one comparable scalar per segment for the right-continuous tie rule.

    Ranks a right-extending segment above one that ends at the query, and among
    equally-eligible segments the larger value-slope. `arctan` bounds the slope into
    `(-pi/2, pi/2)`, so the integer right-extends bit dominates it. The caller masks
    this key to the set the certified predicate has declared tied, so `argmax` over
    it reproduces "prefer a right-extending tied segment, else the largest tied
    slope" — and the dense path and the blocked scan, sharing the key, select the
    same winner.
    """
    bounded_slope = jnp.arctan(slope) / jnp.pi + 0.5
    return right_available.astype(bounded_slope.dtype) + bounded_slope


def _block_query_terms(
    *, block: FloatND, live: BoolND, flat: Float1D
) -> tuple[BoolND, FloatND, FloatND, FloatND, FloatND, FloatND]:
    """Bracket-and-interpolate one segment block against every query.

    `block` is one `(block_size, 8)` slice of the stacked link endpoint columns
    and `live` its `(block_size,)` live-flag slice. Returns the
    `(n_query, block_size)` bracket mask; the value, policy, marginal, and
    value-slope interpolated at each query for each link in the block; and the
    link's upper endpoint (for the right-continuous tie-break) — the same
    quantities the dense path forms over all segments at once, but only for this
    block, so the peak working set is `(n_query, block_size)`.
    """
    left_grid, right_grid = block[:, 0], block[:, 1]
    left_value, right_value = block[:, 2], block[:, 3]
    left_policy, right_policy = block[:, 4], block[:, 5]
    left_marginal, right_marginal = block[:, 6], block[:, 7]

    q = flat[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = live[None, :] & (q >= lower) & (q <= upper)

    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (q - left_grid[None, :]) / safe_width)
    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )
    marginal_interp = (
        left_marginal[None, :] + relative * (right_marginal - left_marginal)[None, :]
    )
    slope = (right_value - left_value)[None, :] / safe_width
    return brackets, value_interp, policy_interp, marginal_interp, slope, upper


def _blocked_columns(
    *, links: _SegmentLinks, block_size: int
) -> tuple[_ComparableLines, FloatND, BoolND, IntND]:
    """Return the link table reshaped into segment blocks, with its line view.

    The eight link columns and the four comparable-line columns travel together
    so that one scan carry reaches both the interpolation and the certified
    comparison. Padding to a multiple of `block_size` fills with dead segments,
    which never bracket and so never win, tie, or poison; `offsets` recovers each
    block's global segment index.
    """
    lines = _as_comparable_lines(links)
    pad = (-links.live.shape[0]) % block_size

    def _padded(column: FloatND) -> FloatND:
        if pad == 0:
            return column
        return jnp.concatenate([column, jnp.zeros((pad,), dtype=column.dtype)])

    padded_lines = _ComparableLines(*(_padded(column) for column in lines))
    columns = jnp.stack(
        [
            _padded(links.left_grid),
            _padded(links.right_grid),
            _padded(links.left_value),
            _padded(links.right_value),
            _padded(links.left_policy),
            _padded(links.right_policy),
            _padded(links.left_marginal),
            _padded(links.right_marginal),
            *padded_lines,
        ],
        axis=1,
    )
    live = (
        links.live
        if pad == 0
        else jnp.concatenate([links.live, jnp.zeros((pad,), dtype=bool)])
    )
    blocks = columns.reshape(-1, block_size, columns.shape[1])
    offsets = jnp.arange(blocks.shape[0], dtype=jnp.int32) * block_size
    return padded_lines, blocks, live.reshape(-1, block_size), offsets


def _envelope_at_query_blocked(
    *, links: _SegmentLinks, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Evaluate the dense reduction in blocked passes, selecting the same owner.

    The dense path holds `(n_query, n_segment)` at once; here each pass folds over
    segment blocks and peaks at `(n_query, block_size)` instead. What is folded is
    the *same* certified decision, not a cheaper stand-in — the owner's identity and
    the certified status travel in the scan carry, so neither path reconstructs a
    winner from rounded levels:

    - Pass 1 takes the running plain-float maximum and, with it, the provisional
      owner's global index.
    - Each promotion round validates that owner against every block, adopting the
      first contender certified above it.
    - The final pass collects, against the settled owner, the certified-tie set (to
      apply the right-continuous rule) and whether any comparison went uncertified.

    The links are padded to a multiple of `block_size` with dead segments, which
    never bracket and so never win, tie, or poison.
    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    n_segment = links.live.shape[0]
    padded_lines, blocks, live_blocks, offsets = _blocked_columns(
        links=links, block_size=block_size
    )
    dtype = links.left_grid.dtype

    def _block_sign(
        *, block: FloatND, block_live: BoolND, owner: _ComparableLines
    ) -> IntND:
        """Return the certified sign of `owner - link` over one block."""
        brackets, *_ = _block_query_terms(block=block, live=block_live, flat=flat)
        contender = _ComparableLines(
            x0=block[:, 8], x1=block[:, 9], v0=block[:, 10], v1=block[:, 11]
        )
        sign = certified_margin_sign(
            a_x0=owner.x0[:, None],
            a_x1=owner.x1[:, None],
            a_v0=owner.v0[:, None],
            a_v1=owner.v1[:, None],
            b_x0=contender.x0[None, :],
            b_x1=contender.x1[None, :],
            b_v0=contender.v0[None, :],
            b_v1=contender.v1[None, :],
            x_query=flat[:, None],
        )
        return jnp.where(brackets, sign, jnp.ones_like(sign))

    def provisional_step(
        carry: tuple[FloatND, IntND, BoolND],
        block_and_live: tuple[FloatND, BoolND, IntND],
    ) -> tuple[tuple[FloatND, IntND, BoolND], None]:
        best_value, best_index, any_bracket = carry
        block, block_live, offset = block_and_live
        brackets, value_interp, *_ = _block_query_terms(
            block=block, live=block_live, flat=flat
        )
        masked = jnp.where(brackets, value_interp, -jnp.inf)
        within = jnp.argmax(masked, axis=1)
        block_value = jnp.take_along_axis(masked, within[:, None], axis=1)[:, 0]
        take = block_value > best_value
        return (
            jnp.where(take, block_value, best_value),
            jnp.where(take, within.astype(jnp.int32) + offset, best_index),
            any_bracket | jnp.any(brackets, axis=1),
        ), None

    (_, owner_index, any_bracket), _ = jax.lax.scan(
        provisional_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.zeros((n_query,), dtype=jnp.int32),
            jnp.zeros((n_query,), dtype=bool),
        ),
        (blocks, live_blocks, offsets),
    )

    for _ in range(_CERTIFIED_PROMOTION_ROUNDS):
        owner = _take_lines(padded_lines, owner_index)

        def beaten_step(
            carry: IntND,
            block_and_live: tuple[FloatND, BoolND, IntND],
            owner: _ComparableLines = owner,
        ) -> tuple[IntND, None]:
            best_challenger = carry
            block, block_live, offset = block_and_live
            sign = _block_sign(block=block, block_live=block_live, owner=owner)
            beaten = sign == -1
            within = jnp.argmax(beaten, axis=1).astype(jnp.int32) + offset
            challenger = jnp.where(jnp.any(beaten, axis=1), within, n_segment)
            return jnp.minimum(best_challenger, challenger), None

        challenger, _ = jax.lax.scan(
            beaten_step,
            jnp.full((n_query,), n_segment, dtype=jnp.int32),
            (blocks, live_blocks, offsets),
        )
        owner_index = jnp.where(challenger < n_segment, challenger, owner_index)

    owner = _take_lines(padded_lines, owner_index)

    def settle_step(
        carry: tuple[FloatND, IntND, BoolND],
        block_and_live: tuple[FloatND, BoolND, IntND],
    ) -> tuple[tuple[FloatND, IntND, BoolND], None]:
        best_rank, best_index, uncertain = carry
        block, block_live, offset = block_and_live
        sign = _block_sign(block=block, block_live=block_live, owner=owner)
        *_, slope, upper = _block_query_terms(block=block, live=block_live, flat=flat)
        within_index = offset + jnp.arange(block_size, dtype=jnp.int32)[None, :]
        tied = (sign == 0) | (within_index == owner_index[:, None])
        rank = jnp.where(
            tied,
            _right_continuous_rank(right_available=flat[:, None] < upper, slope=slope),
            -jnp.inf,
        )
        best_within = jnp.argmax(rank, axis=1)
        block_rank = jnp.take_along_axis(rank, best_within[:, None], axis=1)[:, 0]
        take = block_rank > best_rank
        block_uncertain = jnp.any(
            (sign == -1) | (sign == UNRESOLVED_SIGN) | (sign == BELOW_RESOLUTION_SIGN),
            axis=1,
        )
        return (
            jnp.where(take, block_rank, best_rank),
            jnp.where(take, best_within.astype(jnp.int32) + offset, best_index),
            uncertain | block_uncertain,
        ), None

    (_, best_index, unresolved), _ = jax.lax.scan(
        settle_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.zeros((n_query,), dtype=jnp.int32),
            jnp.zeros((n_query,), dtype=bool),
        ),
        (blocks, live_blocks, offsets),
    )

    published = any_bracket & ~unresolved
    value, policy, marginal = _interpolate_one(
        links=links, index=jnp.minimum(best_index, n_segment - 1), flat=flat
    )
    return (
        jnp.where(published, value, jnp.nan).reshape(query.shape),
        jnp.where(published, policy, jnp.nan).reshape(query.shape),
        jnp.where(published, marginal, jnp.nan).reshape(query.shape),
    )


def _interpolate_one(
    *, links: _SegmentLinks, index: IntND, flat: Float1D
) -> tuple[FloatND, FloatND, FloatND]:
    """Return value, policy and marginal of one chosen segment per query."""
    left_grid = jnp.take(links.left_grid, index)
    right_grid = jnp.take(links.right_grid, index)
    width = right_grid - left_grid
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (flat - left_grid) / safe_width)

    def at(left: Float1D, right: Float1D) -> FloatND:
        low = jnp.take(left, index)
        return low + relative * (jnp.take(right, index) - low)

    return (
        at(links.left_value, links.right_value),
        at(links.left_policy, links.right_policy),
        at(links.left_marginal, links.right_marginal),
    )
