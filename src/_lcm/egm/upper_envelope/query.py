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
between separate reads. However well each segment's own value is read, the error
that read carries is proportional to its magnitude, and a decision needs a bound
proportional to the *gap* — otherwise a common value level, which moves no
crossing and reverses no ordering, decides the ownership.

So ownership is settled on differences, and by the *sign* of one — never by how
near two reads are. One bracketing segment is carried as the reference line, and
every segment is compared with it by `certified_sign`, which cross-multiplies
before dividing so the common level cancels in arithmetic that loses nothing. A
strict sign orders the two outright: a segment certified above the reference
replaces it, and a bounded number of such promotions is followed by a validation
round that no remaining bracketing segment beats the one about to be published.
Each promotion strictly raises the true value at the query, so the sequence
cannot cycle.

An interval, by contrast, cannot order anything it overlaps. Deciding among
candidates whose error bounds overlap would promote "not separated" to "equal",
and the tie-break — which prefers the larger value-slope — would then hand the
query to whichever strict loser happens to be steeper. So overlap is not a tie:
only a sign certified *exactly zero*, or a difference certified below the
arithmetic's own resolution, reaches the documented right-continuous tie-break,
which chooses among genuine ties deterministically. Value, policy, and marginal
are all published from the one segment it names.

Failing to *separate* two segments is therefore an ordinary outcome with a
defined answer. Failing to *compute* the comparison at all is not: where a
product leaves the range in which the error-free transforms are exact, nothing
follows about the geometry and the segments may be far apart. Only that second
case abstains, publishing NaN in all three channels rather than a guess,
identically in both backends below.

By default the evaluation is a fixed-shape `(n_query, n_segment)`
bracket-and-reduce: no sequential scan, no NaN-padded refined row,
branch-parallel and reduction-heavy, which is the shape an accelerator runs
fastest. This is the backend asset-row mode wants — one query per Euler node, no
full envelope to refine. For a large `(n_query, n_segment)` that dense matrix is
itself the memory wall; `segment_block_size` swaps it for a blocked scan over
segment blocks (the reference line, then the promotions, then the winner among
the segments certified level with it), which peaks at `(n_query, block)` instead
of `(n_query, n_segment)` and returns the identical result.
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import (
    BELOW_RESOLUTION_SIGN,
    UNRESOLVED_SIGN,
    QuotientMargin,
    affine_numerator,
    certified_margin_sign,
    certified_quotient_margin,
)
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul,
    dd_quotient,
)
from lcm.typing import BoolND, Float1D, FloatND, IntND

# Which arithmetic compares two candidates; see `envelope_at_query`. It names what
# varies between the two backends — the comparison — rather than the arithmetic
# itself, which `double_double` already is.
type ComparisonArithmetic = Literal["certified", "ordinary"]

# How many times a query's reference line may be replaced by a candidate certified
# above it before the comparison that publishes. Each promotion strictly raises the
# true value at the query, so the sequence cannot cycle; the round after the last
# one certifies that nothing remains above. A query needing more promotions than
# this publishes NaN rather than a candidate that was never validated.
_PROMOTION_ROUNDS = 2


def _along_link(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    arithmetic: ComparisonArithmetic = "certified",
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
    if arithmetic == "ordinary":
        at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
            query=query,
            left_grid=left_grid,
            right_grid=right_grid,
            left=left,
            right=right,
        )
        inside = ((divisor_grid - query) * left + (query - left_grid) * right) / (
            divisor_grid - left_grid
        )
        return jnp.where(at_endpoint, endpoint, inside)
    numerator, divisor, endpoint, at_endpoint = _link_terms(
        left=left,
        right=right,
        query=query,
        left_grid=left_grid,
        right_grid=right_grid,
    )
    high, low = dd_quotient(numerator, divisor)
    return jnp.where(at_endpoint, endpoint, high + low)


def _value_quotient(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    level: FloatND,
) -> tuple[DoubleDouble, DoubleDouble]:
    """Return numerator and divisor whose quotient is the link's value above `level`.

    Ownership is decided from this pair rather than from the quotient it stands
    for, because division is the one operation here with no error-free transform:
    dividing each candidate before comparing spends the certificate that comparing
    first would have kept.

    `level` is subtracted from each endpoint value before anything is multiplied,
    and `two_sum` makes that subtraction exact, so no information is traded for it.
    What it buys is the scale everything downstream runs at: the endpoint values of
    an EGM candidate can sit on a common level many orders of magnitude above the
    distances between candidates, and every product formed from them — along with
    the tail each one discards — inherits that level. Removing it first leaves an
    arithmetic whose magnitudes are those of the differences being decided, so one
    common level shared by every candidate cannot reach the decision at all.
    Passing the same `level` for every candidate at a query is what makes the
    subtraction cancel out of their margins and so leave the ordering alone.

    Where the query is one of the stored abscissae — or the link is a zero-width
    self-bracket, whose own abscissa is the only query it brackets — the value is
    that endpoint's exactly, and the pair says so with a unit divisor. Everywhere
    else it is the affine form over the link's width.
    """
    at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
        query=query, left_grid=left_grid, right_grid=right_grid, left=left, right=right
    )
    numerator = dd_add(
        dd_mul(
            dd_from_difference(divisor_grid, query), dd_from_difference(left, level)
        ),
        dd_mul(dd_from_difference(query, left_grid), dd_from_difference(right, level)),
    )
    divisor = dd_from_difference(divisor_grid, left_grid)
    zero = jnp.zeros_like(endpoint)
    unit = (jnp.ones_like(endpoint), zero, zero)
    return (
        _select_double_double(
            at_endpoint, dd_from_difference(endpoint, level), numerator
        ),
        _select_double_double(at_endpoint, unit, divisor),
    )


def _select_double_double(
    condition: BoolND, when: DoubleDouble, otherwise: DoubleDouble
) -> DoubleDouble:
    """Choose elementwise between two double-doubles, word by word."""
    return (
        jnp.where(condition, when[0], otherwise[0]),
        jnp.where(condition, when[1], otherwise[1]),
        jnp.where(condition, when[2], otherwise[2]),
    )


def _take_column(value: DoubleDouble, index: IntND) -> DoubleDouble:
    """Gather one segment's double-double per query row, word by word."""
    return (
        jnp.take_along_axis(value[0], index, axis=1),
        jnp.take_along_axis(value[1], index, axis=1),
        jnp.take_along_axis(value[2], index, axis=1),
    )


def _pivot_index(*, brackets: BoolND, value: FloatND) -> IntND:
    """Pick the candidate every other one is first compared against.

    Any bracketing candidate serves as the opening reference: the promotions that
    follow replace it with anything certified above it, so the choice decides how
    many promotions are needed and nothing else. The plain read's maximum is the
    cheapest good guess at the one already on top.
    """
    index = jnp.argmax(jnp.where(brackets, value, -jnp.inf), axis=1)
    return index[:, None].astype(jnp.int32)


class _ComparableLines(NamedTuple):
    """Each link as an ascending, positive-width affine line.

    `certified_margin_sign` compares lines, and states plainly what it will not
    invent: a link of zero width has no affine value line, and a non-finite input
    is unresolved rather than false. Both are shaped here instead, once, so every
    comparison downstream is handed operands it can actually decide on.
    """

    x0: FloatND
    """Lower abscissa."""
    x1: FloatND
    """Upper abscissa, strictly above `x0`."""
    v0: FloatND
    """Value at `x0`."""
    v1: FloatND
    """Value at `x1`."""


def _comparable_lines(
    *,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: BoolND,
) -> _ComparableLines:
    """Shape each link into a line the certified comparison can decide on.

    Three cases, and only the first carries information:

    - a link stored right-to-left is swapped. An affine line is the same line
      read either way, so this is exact and changes no answer.
    - a *live* lone candidate is a zero-width self-bracket. The flat line through
      its own point reads exactly its own value at the one abscissa it brackets,
      and `nextafter` is the narrowest positive width the format has, so no other
      query can reach it.
    - a dead or non-finite entry is replaced by a fixed placeholder line. This is
      the case that would otherwise do damage: a NaN abscissa makes every
      comparison against it `UNRESOLVED`, which is the loud failure signal, so a
      padded row would abstain on a query it should simply have won. The
      placeholder is never read — a dead entry brackets nothing — it only keeps
      the arithmetic finite.
    """
    usable = live & jnp.isfinite(left_grid) & jnp.isfinite(right_grid)
    floor = jnp.zeros_like(left_grid)
    ceiling = jnp.ones_like(left_grid)
    lower = jnp.where(usable, left_grid, floor)
    upper = jnp.where(usable, right_grid, ceiling)
    at_lower = jnp.where(usable, left_value, floor)
    at_upper = jnp.where(usable, right_value, floor)

    stored_descending = upper < lower
    x0 = jnp.where(stored_descending, upper, lower)
    x1 = jnp.where(stored_descending, lower, upper)
    v0 = jnp.where(stored_descending, at_upper, at_lower)
    v1 = jnp.where(stored_descending, at_lower, at_upper)

    degenerate = x1 <= x0
    return _ComparableLines(
        x0=x0,
        x1=jnp.where(degenerate, jnp.nextafter(x0, jnp.inf), x1),
        v0=v0,
        v1=jnp.where(degenerate, v0, v1),
    )


def _sign_against_reference(
    *, lines: _ComparableLines, reference: _ComparableLines, query: FloatND
) -> IntND:
    """Certified sign of each candidate's value at `query` less the reference's."""
    return certified_margin_sign(
        a_x0=lines.x0,
        a_x1=lines.x1,
        a_v0=lines.v0,
        a_v1=lines.v1,
        b_x0=reference.x0,
        b_x1=reference.x1,
        b_v0=reference.v0,
        b_v1=reference.v1,
        x_query=query,
    )


class _CertifiedOwner(NamedTuple):
    """Which candidate owns each query, and whether that could be settled."""

    index: IntND
    """Column of the winning candidate, per query."""
    settled: BoolND
    """Whether every bracketing comparison was computable and none beat the winner."""


def _certain_lower_bound(*, brackets: BoolND, margin: QuotientMargin) -> FloatND:
    """The largest margin over the pivot anything bracketing can be shown to reach."""
    return jnp.max(
        jnp.where(brackets & margin.trustworthy, margin.value - margin.bound, -jnp.inf),
        axis=1,
        keepdims=True,
    )


def _contending_against(
    *, brackets: BoolND, margin: QuotientMargin, certain_lower: FloatND
) -> BoolND:
    """Which bracketing candidates the margins over the pivot leave in contention.

    Each candidate's margin over the common pivot comes with an error bound, so a
    candidate is certainly beaten when its own upper bound falls short of the best
    lower bound anyone reaches. That is a *certified exclusion* and it is the one
    thing bounds can decide: they measure every candidate against the same pivot,
    after subtracting its value level exactly, so they stay sharp where an
    ordinary comparison of two large near-equal values would not.

    What bounds cannot do is order the candidates they fail to exclude — overlap
    is not equality. Those go on to the pairwise comparison, which is sharp in the
    opposite regime. A candidate whose own bound is untrustworthy excludes
    nothing and is excluded by nothing; it is simply left for the sharper test.
    """
    return brackets & (
        ~margin.trustworthy | (margin.value + margin.bound >= certain_lower)
    )


def _certified_owner(
    *,
    lines: _ComparableLines,
    reference: _ComparableLines,
    contending: BoolND,
    query: FloatND,
    value: FloatND,
    right_available: BoolND,
    slope: FloatND,
) -> _CertifiedOwner:
    """Settle ownership among the contenders by certified sign.

    A candidate certified strictly above the reference replaces it, up to
    `_PROMOTION_ROUNDS` times; the round after that is a validation the winner has
    to survive against every remaining contender. Candidates certified level with
    the winner — exactly equal, or apart by less than the arithmetic can resolve —
    are the ones the right-continuous tie-break chooses among. The winner is
    always among them, provided the reference is itself a contender: a line
    compared with the candidate it was taken from yields its own sign, which is
    never strict, so no separate record of which candidate it was needs carrying.
    """
    lines = _ComparableLines(
        *(jnp.broadcast_to(field, contending.shape) for field in lines)
    )
    for _ in range(_PROMOTION_ROUNDS):
        sign = _sign_against_reference(lines=lines, reference=reference, query=query)
        beats = contending & (sign == 1)
        challenger = jnp.argmax(jnp.where(beats, value, -jnp.inf), axis=1)[:, None]
        promoted = jnp.any(beats, axis=1, keepdims=True)
        reference = _ComparableLines(
            *(
                jnp.where(
                    promoted, jnp.take_along_axis(field, challenger, axis=1), held
                )
                for field, held in zip(lines, reference, strict=True)
            )
        )

    sign = _sign_against_reference(lines=lines, reference=reference, query=query)
    level_with = contending & ((sign == 0) | (sign == BELOW_RESOLUTION_SIGN))
    return _CertifiedOwner(
        index=jnp.argmax(
            _right_continuous_rank(
                near_max=level_with, right_available=right_available, slope=slope
            ),
            axis=1,
        )[:, None].astype(jnp.int32),
        settled=~jnp.any(contending & (sign == 1), axis=1)
        & ~jnp.any(contending & (sign == UNRESOLVED_SIGN), axis=1),
    )


def _link_geometry(
    *,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    left: FloatND,
    right: FloatND,
) -> tuple[BoolND, FloatND, FloatND, FloatND]:
    """Whether the query is an endpoint, which one, and a divisor-safe right node."""
    at_left = query == left_grid
    at_right = query == right_grid
    # A zero-width self-bracket carries no affine line. Its own abscissa is the
    # only query it brackets, so the quotient is never read there and has only to
    # stay finite; displacing the divisor keeps it so.
    degenerate = right_grid == left_grid
    return (
        at_left | at_right | degenerate,
        jnp.where(at_left | degenerate, left, right),
        left_grid,
        jnp.where(degenerate, left_grid + 1.0, right_grid),
    )


def _link_terms(
    *,
    left: FloatND,
    right: FloatND,
    query: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[DoubleDouble, DoubleDouble, FloatND, BoolND]:
    """Split a link read into its exact endpoint case and its affine quotient."""
    at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
        query=query, left_grid=left_grid, right_grid=right_grid, left=left, right=right
    )
    numerator = affine_numerator(
        x0=left_grid, x1=divisor_grid, v0=left, v1=right, x_query=query
    )
    return (
        numerator,
        dd_from_difference(divisor_grid, left_grid),
        endpoint,
        at_endpoint,
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
    arithmetic: ComparisonArithmetic = "certified",
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
        arithmetic: Which arithmetic decides ownership.
            - `"certified"` compares candidates in double-double precision and
              publishes NaN wherever no candidate is separated, so a reported
              winner is one the arithmetic could prove. Ordering survives the
              cancellation between endpoint values that a nearly-tied crossing
              produces.
            - `"ordinary"` reads each link in the working format and takes the
              largest. It decides every bracketed query — there is no certificate
              to abstain on — and costs roughly an order of magnitude less per
              read. Adequate where candidate values are separated by much more
              than the format's resolution at their own magnitude.
            The choice is made when the function is traced, so `"ordinary"`
            emits none of the error-free transforms rather than masking them.
            Implemented for the dense reduction only.

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
        _fail_if_blocked_scan_cannot_serve(arithmetic=arithmetic)
        return _envelope_blocked(
            links=links, query=query, block_size=segment_block_size
        ).published

    return _envelope_dense(links=links, query=query, arithmetic=arithmetic).published


def _fail_if_blocked_scan_cannot_serve(*, arithmetic: ComparisonArithmetic) -> None:
    """The blocked scan carries only the certified arithmetic.

    Refusing is the point: silently serving the certified result for an
    `"ordinary"` request would report the arithmetic's cost as if the choice had
    been honoured, which is exactly the measurement the setting exists to make.
    """
    if arithmetic != "certified":
        msg = (
            f"envelope arithmetic {arithmetic!r} is implemented for the dense "
            "reduction only, but a positive `segment_block_size` below the "
            "segment count selected the blocked scan. Set "
            "`segment_block_size=0` to use the dense reduction, or leave the "
            "arithmetic at 'certified'."
        )
        raise ValueError(msg)


class _EnvelopeReduction(NamedTuple):
    """The envelope at every query."""

    published: tuple[FloatND, FloatND, FloatND]
    """Value, policy, and marginal of the winning candidate at each query."""


def _envelope_dense(
    *,
    links: _SegmentLinks,
    query: FloatND,
    arithmetic: ComparisonArithmetic = "certified",
) -> _EnvelopeReduction:
    """Evaluate the envelope at every query as one `(n_query, n_segment)` reduction."""
    flat = query.reshape(-1)[:, None]
    left_grid, right_grid = links.left_grid, links.right_grid
    left_value, right_value = links.left_value, links.right_value
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
    value_interp = _along_link(
        left=left_value[None, :],
        right=right_value[None, :],
        arithmetic=arithmetic,
        **abscissae,
    )
    policy_interp = _along_link(
        left=links.left_policy[None, :],
        right=links.right_policy[None, :],
        arithmetic=arithmetic,
        **abscissae,
    )
    marginal_interp = _along_link(
        left=links.left_marginal[None, :],
        right=links.right_marginal[None, :],
        arithmetic=arithmetic,
        **abscissae,
    )

    any_bracket = jnp.any(brackets, axis=1)
    # Among candidates the arithmetic certifies level with one another, break the
    # tie right-continuously, matching the kernel's `side="right"` read: prefer one
    # that extends strictly to the right of the query (so "larger value-slope is
    # higher just to the right" is meaningful), and among those the larger slope.
    # Only at the global upper endpoint, where nothing continues right, fall back
    # to the largest slope. `_right_continuous_rank` folds both keys into one
    # comparable scalar so this dense reduction and the blocked scan select the
    # same winner.
    slope = (right_value - left_value)[None, :] / safe_width
    right_available = flat < upper
    if arithmetic == "ordinary":
        # No certificate, so nothing to order by but the reads themselves: the
        # largest wins outright and a query anything brackets is decided. Ties go
        # to the same right-continuous rank as the certified branch.
        best_read = jnp.max(
            jnp.where(brackets, value_interp, -jnp.inf), axis=1, keepdims=True
        )
        best = jnp.argmax(
            _right_continuous_rank(
                near_max=brackets & (value_interp >= best_read),
                right_available=right_available,
                slope=slope,
            ),
            axis=1,
        )[:, None]
        decided = any_bracket
    else:
        pivot = _pivot_index(brackets=brackets, value=value_interp)
        numerator, divisor = _value_quotient(
            left=left_value[None, :],
            right=right_value[None, :],
            level=jnp.take_along_axis(value_interp, pivot, axis=1),
            **abscissae,
        )
        margin = certified_quotient_margin(
            left_numerator=numerator,
            left_divisor=divisor,
            right_numerator=_take_column(numerator, pivot),
            right_divisor=_take_column(divisor, pivot),
        )
        contending = _contending_against(
            brackets=brackets,
            margin=margin,
            certain_lower=_certain_lower_bound(brackets=brackets, margin=margin),
        )
        lines = _comparable_lines(
            left_grid=left_grid[None, :],
            right_grid=right_grid[None, :],
            left_value=left_value[None, :],
            right_value=right_value[None, :],
            live=segment_live[None, :],
        )
        # The reference has to be a contender, or the winner could fall outside
        # the set the tie-break chooses from. The highest-reading contender is
        # both cheap and already close to the answer.
        opening = _pivot_index(brackets=contending, value=value_interp)
        owner = _certified_owner(
            lines=lines,
            reference=_ComparableLines(
                *(
                    jnp.take_along_axis(
                        jnp.broadcast_to(field, brackets.shape), opening, axis=1
                    )
                    for field in lines
                )
            ),
            contending=contending,
            query=flat,
            value=value_interp,
            right_available=right_available,
            slope=slope,
        )
        best = owner.index
        decided = any_bracket & owner.settled

    def _from_winner(channel: FloatND) -> FloatND:
        """Publish one channel of the winner, or NaN where no winner was decided."""
        taken = jnp.take_along_axis(channel, best, axis=1)[:, 0]
        return jnp.where(decided, taken, jnp.nan).reshape(query.shape)

    return _EnvelopeReduction(
        published=(
            _from_winner(value_interp),
            _from_winner(policy_interp),
            _from_winner(marginal_interp),
        )
    )


def _right_continuous_rank(
    *, near_max: BoolND, right_available: BoolND, slope: FloatND
) -> FloatND:
    """Return one comparable scalar per segment for the right-continuous tie-break.

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
    spread_left_grid = left_grid[None, :]
    spread_right_grid = right_grid[None, :]
    value_interp = _along_link(
        left=left_value[None, :],
        right=right_value[None, :],
        query=q,
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
    )
    policy_interp = _along_link(
        left=left_policy[None, :],
        right=right_policy[None, :],
        query=q,
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
    )
    marginal_interp = _along_link(
        left=left_marginal[None, :],
        right=right_marginal[None, :],
        query=q,
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
    )
    slope = (right_value - left_value)[None, :] / safe_width
    return _BlockTerms(
        brackets=brackets,
        value=value_interp,
        policy=policy_interp,
        marginal=marginal_interp,
        slope=slope,
        upper=upper,
    )


def _block_lines(*, block: FloatND, live: BoolND) -> _ComparableLines:
    """One block's links as lines the certified comparison can decide on."""
    return _comparable_lines(
        left_grid=block[:, 0][None, :],
        right_grid=block[:, 1][None, :],
        left_value=block[:, 2][None, :],
        right_value=block[:, 3][None, :],
        live=live[None, :],
    )


def _highest_reading_line_over_blocks(
    *,
    blocks: FloatND,
    live_blocks: BoolND,
    flat: Float1D,
    dtype: jnp.dtype,
    held: _ComparableLines | None,
    eligible: Callable[[_BlockTerms, _ComparableLines], BoolND],
) -> tuple[_ComparableLines, FloatND]:
    """Keep the highest-reading eligible link across all blocks, as a line.

    The same fold serves the opening reference (every bracketing link is
    eligible) and a promotion round (only links certified above the line being
    held). `held` is what a query keeps when no block offers an eligible link;
    a query nothing brackets keeps a placeholder whose comparisons nothing reads,
    since such a query publishes NaN on `any_bracket` regardless.
    """
    n_query = flat.shape[0]

    def step(
        carry: tuple[FloatND, ...], block_and_live: tuple[FloatND, BoolND]
    ) -> tuple[tuple[FloatND, ...], None]:
        best_read, *kept = carry
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        lines = _block_lines(block=block, live=block_live)
        candidate = jnp.where(eligible(terms, lines), terms.value, -jnp.inf)
        index = jnp.argmax(candidate, axis=1)[:, None]
        take = jnp.take_along_axis(candidate, index, axis=1)[:, 0] > best_read
        taken = (
            jnp.take_along_axis(
                jnp.broadcast_to(field, candidate.shape), index, axis=1
            )[:, 0]
            for field in lines
        )
        return (
            jnp.where(
                take, jnp.take_along_axis(candidate, index, axis=1)[:, 0], best_read
            ),
            *(
                jnp.where(take, offered, standing)
                for offered, standing in zip(taken, kept, strict=True)
            ),
        ), None

    zeros = jnp.zeros((n_query,), dtype=dtype)
    placeholder = _ComparableLines(
        x0=zeros, x1=jnp.ones((n_query,), dtype=dtype), v0=zeros, v1=zeros
    )
    standing = (
        placeholder
        if held is None
        else _ComparableLines(*(field[:, 0] for field in held))
    )
    carry, _ = jax.lax.scan(
        step,
        (jnp.full((n_query,), -jnp.inf, dtype=dtype), *standing),
        (blocks, live_blocks),
    )
    return _ComparableLines(*(field[:, None] for field in carry[1:])), carry[0][:, None]


def _envelope_blocked(
    *, links: _SegmentLinks, query: FloatND, block_size: int
) -> _EnvelopeReduction:
    """Evaluate the dense `(n_query, n_segment)` reduction as blocked passes.

    Every pass is an exact associative fold against a fixed target, so the result
    matches the dense path (up to floating-point reassociation between the two XLA
    lowerings):

    - Pass 1 accumulates the running per-query maximum of the plain read and the
      line of the link that attained it: the reference each query opens against.
      Any bracketing link would serve, so only its own block-order matters, and
      that is fixed.
    - One pass per promotion round re-scans the blocks and replaces that line with
      the highest-reading link certified strictly above it. Whether a link is
      above the reference is a property of the two alone, so a block can answer it
      without seeing any other block — which is why the reference travels as a
      *line* rather than as a segment index the blocks would have to resolve.
    - The final pass re-scans once more and, among segments certified level with
      the reference, keeps the winner of the right-continuous rank
      (`_right_continuous_rank`: a right-extending segment over one ending at the
      query, then larger value-slope) — the dense path's tie-break. The strict
      cross-block `>` keeps the earliest such winner, matching the dense `argmax`,
      and value, policy, and marginal are all published from it. The same pass
      accumulates whether any bracketing link still beats the reference, and
      whether any bracketing comparison could not be computed at all; either
      leaves the query undecided.

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

    pivot_lines, best_read = _highest_reading_line_over_blocks(
        blocks=blocks,
        live_blocks=live_blocks,
        flat=flat,
        dtype=dtype,
        held=None,
        eligible=lambda terms, _lines: terms.brackets,
    )
    level = jnp.where(jnp.isfinite(best_read), best_read, 0.0)
    pivot_numerator, pivot_divisor = _value_quotient(
        left=pivot_lines.v0,
        right=pivot_lines.v1,
        query=flat[:, None],
        left_grid=pivot_lines.x0,
        right_grid=pivot_lines.x1,
        level=level,
    )

    def _block_margin(lines: _ComparableLines) -> QuotientMargin:
        """Certify every link of one block against the query's pivot."""
        numerator, divisor = _value_quotient(
            left=lines.v0,
            right=lines.v1,
            query=flat[:, None],
            left_grid=lines.x0,
            right_grid=lines.x1,
            level=level,
        )
        return certified_quotient_margin(
            left_numerator=numerator,
            left_divisor=divisor,
            right_numerator=pivot_numerator,
            right_divisor=pivot_divisor,
        )

    def bounds_step(
        carry: FloatND, block_and_live: tuple[FloatND, BoolND]
    ) -> tuple[FloatND, None]:
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        margin = _block_margin(_block_lines(block=block, live=block_live))
        return jnp.maximum(
            carry,
            jnp.max(
                jnp.where(
                    terms.brackets & margin.trustworthy,
                    margin.value - margin.bound,
                    -jnp.inf,
                ),
                axis=1,
            ),
        ), None

    running, _ = jax.lax.scan(
        bounds_step,
        jnp.full((n_query,), -jnp.inf, dtype=dtype),
        (blocks, live_blocks),
    )
    certain_lower = running[:, None]

    def block_contending(terms: _BlockTerms, lines: _ComparableLines) -> BoolND:
        """Which of the block's bracketing links the margins leave in contention."""
        return _contending_against(
            brackets=terms.brackets,
            margin=_block_margin(lines),
            certain_lower=certain_lower,
        )

    reference, _ = _highest_reading_line_over_blocks(
        blocks=blocks,
        live_blocks=live_blocks,
        flat=flat,
        dtype=dtype,
        held=None,
        eligible=block_contending,
    )
    for _ in range(_PROMOTION_ROUNDS):
        standing = reference
        reference, _ = _highest_reading_line_over_blocks(
            blocks=blocks,
            live_blocks=live_blocks,
            flat=flat,
            dtype=dtype,
            held=standing,
            eligible=lambda terms, lines, standing=standing: (
                block_contending(terms, lines)
                & (
                    _sign_against_reference(
                        lines=lines, reference=standing, query=flat[:, None]
                    )
                    == 1
                )
            ),
        )

    def winner_step(
        carry: tuple[FloatND, FloatND, FloatND, FloatND, BoolND, BoolND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, FloatND, FloatND, FloatND, BoolND, BoolND], None]:
        best_rank, best_value, best_policy, best_marginal, any_bracket, settled = carry
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        lines = _block_lines(block=block, live=block_live)
        contending = block_contending(terms, lines)
        sign = _sign_against_reference(
            lines=lines, reference=reference, query=flat[:, None]
        )
        rank = _right_continuous_rank(
            near_max=contending & ((sign == 0) | (sign == BELOW_RESOLUTION_SIGN)),
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
            any_bracket | jnp.any(terms.brackets, axis=1),
            settled
            & ~jnp.any(contending & (sign == 1), axis=1)
            & ~jnp.any(contending & (sign == UNRESOLVED_SIGN), axis=1),
        ), None

    empty = jnp.full((n_query,), jnp.nan, dtype=dtype)
    (_, env_value, env_policy, env_marginal, any_bracket, settled), _ = jax.lax.scan(
        winner_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            empty,
            empty,
            empty,
            jnp.zeros((n_query,), dtype=bool),
            jnp.ones((n_query,), dtype=bool),
        ),
        (blocks, live_blocks),
    )

    decided = any_bracket & settled

    def _published(channel: FloatND) -> FloatND:
        """Shape one channel like the query, NaN where no winner was decided."""
        return jnp.where(decided, channel, jnp.nan).reshape(query.shape)

    return _EnvelopeReduction(
        published=(
            _published(env_value),
            _published(env_policy),
            _published(env_marginal),
        )
    )
