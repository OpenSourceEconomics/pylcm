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

Which segment wins is a decision, not a reported quantity, so it is not settled
by comparing rounded reads. Each segment's value at the query is read in the
double-double arithmetic of `double_double`, which returns a measured bound on
its own error, and the winner is taken between those intervals. A segment stays
in contention only while its read could still be the highest; where several
could, the documented right-continuous tie-break chooses among them
deterministically, and value, policy, and marginal are all published from the
one segment it names.

By default the evaluation is a fixed-shape `(n_query, n_segment)`
bracket-and-reduce: no sequential scan, no NaN-padded refined row,
branch-parallel and reduction-heavy, which is the shape an accelerator runs
fastest. This is the backend asset-row mode wants — one query per Euler node, no
full envelope to refine. For a large `(n_query, n_segment)` that dense matrix is
itself the memory wall; `segment_block_size` swaps it for a two-pass blocked
scan over segment blocks (running certified lower bound, then the winner among
the segments that could still reach it), which peaks at `(n_query, block)`
instead of `(n_query, n_segment)` and returns the identical result.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import affine_numerator
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_from_difference,
    dd_quotient,
    dd_quotient_bounded,
    two_sum,
)
from lcm.typing import BoolND, Float1D, FloatND


def _along_link(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> FloatND:
    """Read a channel along a link at `query`.

    Whether the query *is* an endpoint is decided against the stored abscissae. A
    normalized coordinate cannot answer that: `(query - left_grid) / width` is a
    rounded quotient that reaches exactly `1.0` for queries strictly inside the
    link, and a read that trusted it would publish the wrong endpoint's value and
    policy for a point that is not an endpoint at all.

    Strictly inside, the channel is `v0*(x1 - x) + v1*(x - x0)` over the width,
    evaluated in the double-double arithmetic of `double_double` — the same form
    `certified_sign` compares two links by. Working at roughly twice the format's
    precision is what a link whose endpoint values nearly cancel requires: there
    the value at the query is smaller than either endpoint by many orders of
    magnitude, so rounding the slope first spends the whole result's significance
    on the cancellation, and what survives is not enough to report, let alone to
    decide on.
    """
    numerator, divisor, endpoint, at_endpoint = _link_terms(
        left=left,
        right=right,
        query=query,
        left_grid=left_grid,
        right_grid=right_grid,
    )
    high, low = dd_quotient(numerator, divisor)
    return jnp.where(at_endpoint, endpoint, high + low)


def _along_link_bounded(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[FloatND, FloatND]:
    """Read a channel along a link at `query`, and bound how far off the read is.

    The bound is what turns a read into something a decision can rest on, so it is
    measured rather than assumed: `dd_quotient_bounded` multiplies the quotient
    back through the denominator and reports what fails to reproduce the
    numerator. It is exactly zero for a read that was exact — an endpoint, a
    degenerate self-bracket, or an interior query whose affine value the format
    happens to hold — so a caller can tell a value it may act on from one it may
    only choose deterministically among.
    """
    numerator, divisor, endpoint, at_endpoint = _link_terms(
        left=left,
        right=right,
        query=query,
        left_grid=left_grid,
        right_grid=right_grid,
    )
    high, low, unreproduced = dd_quotient_bounded(numerator, divisor)
    # Collapsing the pair to one float is itself a rounding, and what it discards
    # is available exactly, so it is added rather than charged as a blanket ULP.
    # A flat charge would be the width of the very gaps this bound has to resolve,
    # and would refuse exactly the reads that are easiest to be sure about.
    interior, discarded = two_sum(high, low)
    value = jnp.where(at_endpoint, endpoint, interior)
    bound = jnp.where(at_endpoint, 0.0, unreproduced + jnp.abs(discarded))
    return value, bound


def _link_terms(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[DoubleDouble, DoubleDouble, FloatND, BoolND]:
    """Split a link read into its exact endpoint case and its affine quotient."""
    at_left = query == left_grid
    at_right = query == right_grid
    # A zero-width self-bracket carries no affine line. Its own abscissa is the
    # only query it brackets, so the quotient is never read there and has only to
    # stay finite; displacing the divisor keeps it so.
    degenerate = right_grid == left_grid
    divisor_grid = jnp.where(degenerate, left_grid + 1.0, right_grid)
    numerator = affine_numerator(
        x0=left_grid, x1=divisor_grid, v0=left, v1=right, x_query=query
    )
    endpoint = jnp.where(at_left | degenerate, left, right)
    return (
        numerator,
        dd_from_difference(divisor_grid, left_grid),
        endpoint,
        at_left | at_right | degenerate,
    )


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

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`. A
        query no live segment brackets yields NaN in all three.
    """
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
    abscissae = {
        "query": flat,
        "left_grid": left_grid[None, :],
        "right_grid": right_grid[None, :],
    }
    value_interp, value_bound = _along_link_bounded(
        left=left_value[None, :], right=right_value[None, :], **abscissae
    )
    policy_interp = _along_link(
        left=left_policy[None, :], right=right_policy[None, :], **abscissae
    )
    marginal_interp = _along_link(
        left=left_marginal[None, :], right=right_marginal[None, :], **abscissae
    )

    any_bracket = jnp.any(brackets, axis=1)
    # Ownership is settled between intervals, not between rounded values. A
    # candidate stays in contention while its own read could still be the highest
    # — its upper bound reaches the best lower bound any candidate certifies — so
    # a link is never dropped for an error its own arithmetic already declared.
    certain_lower = jnp.max(
        jnp.where(brackets, value_interp - value_bound, -jnp.inf),
        axis=1,
        keepdims=True,
    )
    near_max = brackets & (value_interp + value_bound >= certain_lower)
    # Among candidates no arithmetic separates, break the tie right-continuously,
    # matching the kernel's `side="right"` read: prefer one that extends strictly
    # to the right of the query (so "larger value-slope is higher just to the
    # right" is meaningful), and among those the larger slope. Only at the global
    # upper endpoint, where nothing continues right, fall back to the largest
    # near-max slope. `_right_continuous_rank` folds both keys into one comparable
    # scalar so this dense reduction and the blocked scan select the same winner.
    slope = (right_value - left_value)[None, :] / safe_width
    right_available = flat < upper
    best = jnp.argmax(
        _right_continuous_rank(
            near_max=near_max, right_available=right_available, slope=slope
        ),
        axis=1,
    )[:, None]

    def _from_winner(channel: FloatND) -> FloatND:
        """Publish one channel of the winning candidate, or NaN where none brackets."""
        taken = jnp.take_along_axis(channel, best, axis=1)[:, 0]
        return jnp.where(any_bracket, taken, jnp.nan)

    return (
        _from_winner(value_interp).reshape(query.shape),
        _from_winner(policy_interp).reshape(query.shape),
        _from_winner(marginal_interp).reshape(query.shape),
    )


def _right_continuous_rank(
    *, near_max: BoolND, right_available: BoolND, slope: FloatND
) -> FloatND:
    """One comparable scalar per segment for the right-continuous tie-break.

    Ranks a right-extending near-max segment above one that ends at the query, and
    among equally-eligible segments the larger value-slope. `arctan` bounds the slope
    into `(-pi/2, pi/2)`, so the integer right-extends bit dominates it; non-near-max
    segments get `-inf`. `argmax` over this key reproduces "prefer a right-extending
    near-max segment, else the largest near-max slope" with no global reduction, so
    the dense path and the blocked scan (which compares this scalar across blocks)
    select the same winner.
    """
    bounded_slope = jnp.arctan(slope) / jnp.pi + 0.5
    rank = right_available.astype(bounded_slope.dtype) + bounded_slope
    return jnp.where(near_max, rank, -jnp.inf)


class _BlockTerms(NamedTuple):
    """What one segment block contributes at every query, all `(n_query, block)`."""

    brackets: BoolND
    """Whether the link is live and brackets the query."""
    value: FloatND
    """Value read along the link at the query."""
    value_bound: FloatND
    """Bound on how far that read can be from the exact affine value."""
    policy: FloatND
    """Policy read along the link at the query."""
    marginal: FloatND
    """Marginal read along the link at the query."""
    slope: FloatND
    """Value-slope of the link, for the right-continuous tie-break."""
    upper: FloatND
    """Upper endpoint of the link, for the same tie-break."""


def _block_query_terms(*, block: FloatND, live: BoolND, flat: Float1D) -> _BlockTerms:
    """Bracket-and-read one segment block against every query.

    `block` is one `(block_size, 8)` slice of the stacked link endpoint columns
    and `live` its `(block_size,)` live-flag slice — the same quantities the dense
    path forms over all segments at once, but only for this block, so the peak
    working set is `(n_query, block_size)`.
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
    abscissae = {
        "query": q,
        "left_grid": left_grid[None, :],
        "right_grid": right_grid[None, :],
    }
    value_interp, value_bound = _along_link_bounded(
        left=left_value[None, :], right=right_value[None, :], **abscissae
    )
    policy_interp = _along_link(
        left=left_policy[None, :], right=right_policy[None, :], **abscissae
    )
    marginal_interp = _along_link(
        left=left_marginal[None, :], right=right_marginal[None, :], **abscissae
    )
    slope = (right_value - left_value)[None, :] / safe_width
    return _BlockTerms(
        brackets=brackets,
        value=value_interp,
        value_bound=value_bound,
        policy=policy_interp,
        marginal=marginal_interp,
        slope=slope,
        upper=upper,
    )


def _envelope_at_query_blocked(
    *, links: _SegmentLinks, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Two-pass blocked equivalent of the dense `(n_query, n_segment)` reduction.

    Both passes are exact associative folds against a fixed target, so the result
    matches the dense path (up to floating-point reassociation between the two
    XLA lowerings):

    - Pass 1 accumulates the running per-query maximum of the *certified lower
      bound* over segment blocks — the highest value any candidate can be shown to
      reach — with a running `any_bracket` flag.
    - Pass 2 re-scans the blocks and, among segments whose read could still reach
      that bound, keeps the winner of the right-continuous rank
      (`_right_continuous_rank`: a right-extending near-max segment over one
      ending at the query, then larger value-slope) — the dense path's tie-break.
      The strict cross-block `>` keeps the earliest such winner, matching the
      dense `argmax`, and value, policy, and marginal are all published from it.

    The links are padded to a multiple of `block_size` with dead segments (which
    never bracket) and reshaped to `(n_block, block_size)`; the scan peaks at
    `(n_query, block_size)` working memory.
    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    n_segment = links.live.shape[0]
    pad = (-n_segment) % block_size

    def _padded(column: FloatND, fill: float) -> FloatND:
        if pad == 0:
            return column
        return jnp.concatenate([column, jnp.full((pad,), fill, dtype=column.dtype)])

    columns = jnp.stack(
        [
            _padded(links.left_grid, 0.0),
            _padded(links.right_grid, 0.0),
            _padded(links.left_value, 0.0),
            _padded(links.right_value, 0.0),
            _padded(links.left_policy, 0.0),
            _padded(links.right_policy, 0.0),
            _padded(links.left_marginal, 0.0),
            _padded(links.right_marginal, 0.0),
        ],
        axis=1,
    )

    live = (
        links.live
        if pad == 0
        else jnp.concatenate([links.live, jnp.zeros((pad,), dtype=bool)])
    )
    blocks = columns.reshape(-1, block_size, columns.shape[1])
    live_blocks = live.reshape(-1, block_size)
    dtype = links.left_grid.dtype

    def lower_bound_step(
        carry: tuple[FloatND, BoolND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, BoolND], None]:
        running_lower, any_bracket = carry
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        block_lower = jnp.max(
            jnp.where(terms.brackets, terms.value - terms.value_bound, -jnp.inf), axis=1
        )
        return (
            jnp.maximum(running_lower, block_lower),
            any_bracket | jnp.any(terms.brackets, axis=1),
        ), None

    (certain_lower, any_bracket), _ = jax.lax.scan(
        lower_bound_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.zeros((n_query,), dtype=bool),
        ),
        (blocks, live_blocks),
    )

    def winner_step(
        carry: tuple[FloatND, FloatND, FloatND, FloatND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, FloatND, FloatND, FloatND], None]:
        best_rank, best_value, best_policy, best_marginal = carry
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        near_max = terms.brackets & (
            terms.value + terms.value_bound >= certain_lower[:, None]
        )
        rank = _right_continuous_rank(
            near_max=near_max,
            right_available=flat[:, None] < terms.upper,
            slope=terms.slope,
        )
        winner = jnp.argmax(rank, axis=1)[:, None]

        def _take(channel: FloatND) -> FloatND:
            return jnp.take_along_axis(channel, winner, axis=1)[:, 0]

        block_rank = _take(rank)
        take = block_rank > best_rank
        return (
            jnp.where(take, block_rank, best_rank),
            jnp.where(take, _take(terms.value), best_value),
            jnp.where(take, _take(terms.policy), best_policy),
            jnp.where(take, _take(terms.marginal), best_marginal),
        ), None

    empty = jnp.full((n_query,), jnp.nan, dtype=dtype)
    (_, env_value, env_policy, env_marginal), _ = jax.lax.scan(
        winner_step,
        (jnp.full((n_query,), -jnp.inf, dtype=dtype), empty, empty, empty),
        (blocks, live_blocks),
    )

    def _published(channel: FloatND) -> FloatND:
        """Shape one channel like the query, NaN where nothing brackets it."""
        return jnp.where(any_bracket, channel, jnp.nan).reshape(query.shape)

    return _published(env_value), _published(env_policy), _published(env_marginal)
