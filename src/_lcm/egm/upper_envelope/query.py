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
only a sign certified *exactly zero* reaches the documented right-continuous
tie-break, which chooses among genuine ties deterministically. Value, policy,
and marginal are all published from the one segment it names.

Two segments the arithmetic cannot separate are therefore not level, and the
query abstains. That case and the one where no comparison could be computed at
all — a product leaving the range in which the error-free transforms are exact,
so that nothing follows about the geometry and the segments may be far apart —
both publish NaN in all three channels rather than a guess, identically in both
backends below. Between "these two are equally good" and "which of these two is
better is beyond this arithmetic" only the first has an answer to publish, and
a query is answered only when every segment bracketing it was decided.

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

import functools
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import (
    BELOW_RESOLUTION_SIGN,
    UNRESOLVED_SIGN,
    QuotientMargin,
    affine_numerator,
    backend_flushes_subnormals,
    certified_margin_sign,
    certified_quotient_margin,
    is_subnormal,
)
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul,
    dd_quotient,
    scale_by_power_of_two,
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
    left_grid, divisor_grid, query = _on_the_links_own_scale(
        left_grid=left_grid, divisor_grid=divisor_grid, query=query
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


def _on_the_links_own_scale(
    *, left_grid: FloatND, divisor_grid: FloatND, query: FloatND
) -> tuple[FloatND, FloatND, FloatND]:
    """Return the link's three abscissae scaled to where its differences are readable.

    A link read is a ratio of quantities built from the differences between these
    three, and it is homogeneous of degree one in them: scaling all three by one
    power of two multiplies numerator and divisor alike and leaves the ratio
    identical. Scaling by a power of two is itself exact, so nothing is traded
    for it.

    What it buys is an arithmetic that can represent what it is handed. A link at
    the bottom of the normal range has differences among the subnormals, where a
    product rounds to a whole ULP — the endpoint values it was carrying are gone,
    and the ratio reads one for a link of any value. Measuring the link on its own
    scale first puts those differences back among ordinary numbers.

    The scale is anchored on the *smallest* of the three, not the largest. What
    ruins a read is a difference falling under the smallest normal, and it is the
    smallest abscissa that a difference sits closest to; normalizing the largest
    instead would push the small end down and manufacture the same loss on a link
    whose endpoints span a wide range. The anchor is then backed off far enough to
    keep the largest of the three finite, since a link spanning the whole range
    cannot have both.
    """
    terms = (left_grid, divisor_grid, query)
    _mantissa, largest = jnp.frexp(
        jnp.maximum(
            jnp.maximum(jnp.abs(terms[0]), jnp.abs(terms[1])), jnp.abs(terms[2])
        )
    )
    smallest = _smallest_finite_exponent(*terms)
    # Scaling the smallest term to the binade around one costs the largest term
    # `smallest` binades of headroom; back the exponent off by whatever that
    # overruns the top of the range.
    headroom = jnp.asarray(jnp.finfo(left_grid.dtype).maxexp - 1, dtype=largest.dtype)
    exponent = jnp.maximum(smallest, largest - headroom)
    scaled = tuple(scale_by_power_of_two(term, -exponent) for term in terms)
    return scaled[0], scaled[1], scaled[2]


def _smallest_finite_exponent(*terms: FloatND) -> IntND:
    """Return the `frexp` exponent of the smallest finite nonzero term.

    Zero and non-finite terms carry no scale of their own and are ignored; a group
    holding nothing else scales by `2**0`, which leaves it alone.
    """
    magnitude = jnp.full_like(terms[0], jnp.inf)
    for term in terms:
        usable = jnp.isfinite(term) & (term != 0.0)
        magnitude = jnp.minimum(magnitude, jnp.where(usable, jnp.abs(term), jnp.inf))
    _mantissa, exponent = jnp.frexp(
        jnp.where(jnp.isfinite(magnitude), magnitude, 1.0),
    )
    return exponent


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
    - a *live* lone candidate is a zero-width self-bracket, and so is a link
      between two coincident nodes. Both are given a flat line one representable
      step wide. A flat line takes its endpoint value at every abscissa, so that
      width changes no reading, and it cannot widen the link's reach either:
      which queries a link brackets is settled from the stored abscissae, never
      from this line, so a self-bracket brackets its own abscissa and nothing
      else. At abscissa zero, though, one step is the smallest subnormal, and a
      subnormal operand is refused outright — which would abstain on a query
      whose candidates are separated by orders of magnitude. So a line that is
      flat *as a matter of fact*, both endpoints carrying one stored value, is
      given unit width there instead. The reading is unchanged and the decision
      does not depend on which width it was: rivals spanning nine orders of
      magnitude either side of it are decided alike.
    - where two coincident nodes disagree, the flat line is imposed rather than
      factual — it reports the lower-stored endpoint and drops the other. That
      is left exactly as it is, because nothing rests on it: each of the two
      nodes also reaches the envelope as its own self-bracket, and the higher of
      them wins there, so the imposed line settles nothing it should not.

    One boundary is known and deliberately not repaired: a flat line in the
    format's *topmost* binade is refused, where an ordinary two-node link at the
    same magnitude is decided. Every binade below it is decided. Resource grids
    are hundreds of orders of magnitude away, so no model reaches it — but the
    suite exercises the format's edge on purpose, so a witness built one binade
    higher than the existing ones would meet it. It is this boundary rather than
    a new defect.
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
    step = jnp.nextafter(x0, jnp.inf)
    # One representable step is a readable width at every abscissa but zero,
    # where it is the smallest subnormal — and a subnormal operand is refused
    # outright, so the comparison would abstain on a query whose candidates are
    # separated by orders of magnitude. A flat line takes its endpoint value at
    # every abscissa, so replacing that width changes no reading; it is only
    # widened where the line is flat as a matter of fact rather than by
    # imposition, which is what `v1 == v0` asks.
    flat_at_zero = degenerate & (v1 == v0) & (x0 == 0.0)
    return _ComparableLines(
        x0=x0,
        x1=jnp.where(flat_at_zero, jnp.ones_like(x0), jnp.where(degenerate, step, x1)),
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
    bracketing: BoolND,
    query: FloatND,
    value: FloatND,
    right_available: BoolND,
    slope_high: FloatND,
    slope_low: FloatND,
) -> _CertifiedOwner:
    """Settle ownership among the contenders by certified sign.

    A candidate certified strictly above the reference replaces it, up to
    `_PROMOTION_ROUNDS` times; the round after that is a validation the winner has
    to survive against every remaining contender. Only candidates certified
    *exactly* level with the winner reach the right-continuous tie-break. The
    winner is always among them, provided the reference is itself a contender: a
    line compared with the candidate it was taken from yields its own sign, which
    is exactly zero, so no separate record of which candidate it was needs
    carrying.

    A difference certified below the arithmetic's own resolution is not one of
    them. That outcome says the comparison was not settled, not that the two
    lines are level, and the distinction is the whole content of the certificate:
    a strictly better candidate whose margin the format cannot resolve is still
    strictly better, and handing it to a tie-break lets a deterministic
    preference answer a question the geometry has already answered. So it leaves
    the query unsettled, and the query publishes NaN.

    The winner is validated against every candidate that *brackets* the query,
    not only those still contending. Both tests are sound, but they are complete
    in different regimes, so a candidate dismissed by one may be undecidable by
    the other — and which of the two dismissed it depends on the pivot the bounds
    were taken around, which is chosen by a reduction over reads that a genuine
    near-tie leaves equal. Validating against the wider set is what keeps the
    published owner a property of the lines rather than of the order they arrived
    in.
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
    level_with = contending & (sign == 0)
    return _CertifiedOwner(
        index=_lexicographic_argmax(
            _tie_break_key(
                level_with=level_with,
                right_available=right_available,
                slope_high=slope_high,
                slope_low=slope_low,
            )
        ),
        settled=~jnp.any(bracketing & (sign == 1), axis=1)
        & ~jnp.any(bracketing & (sign == UNRESOLVED_SIGN), axis=1)
        & ~jnp.any(bracketing & (sign == BELOW_RESOLUTION_SIGN), axis=1),
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
    """Split a link read into its exact endpoint case and its affine quotient.

    The quotient is homogeneous of degree one in the link's three abscissae, so
    the link is measured on its own scale first; see `_value_quotient`, which
    carries the same argument for the margin.
    """
    at_endpoint, endpoint, left_grid, divisor_grid = _link_geometry(
        query=query, left_grid=left_grid, right_grid=right_grid, left=left, right=right
    )
    left_grid, divisor_grid, query = _on_the_links_own_scale(
        left_grid=left_grid, divisor_grid=divisor_grid, query=query
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
        published = _envelope_blocked(
            links=links, query=query, block_size=segment_block_size
        ).published
    else:
        published = _envelope_dense(
            links=links, query=query, arithmetic=arithmetic
        ).published

    unreadable = _subnormal_operand_present(
        row=(endog_grid, value, policy, marginal), query=query
    ) | _derived_subnormal_possible(links=links, query=query)
    readable_value, readable_policy, readable_marginal = (
        jnp.where(unreadable, jnp.nan, channel) for channel in published
    )
    return readable_value, readable_policy, readable_marginal


def _subnormal_operand_present(*, row: tuple[Float1D, ...], query: FloatND) -> BoolND:
    """Report, per query, whether an operand the backend cannot read is in play.

    Whether a subnormal operand is readable belongs to the backend, not to the
    format, so the predicate asks before it refuses. On a backend that reads the
    whole band every stored operand arrives intact, nothing is lost on the way
    into the read, and refusing would withhold a verdict the arithmetic reached
    correctly — so the refusal leaves the compiled program entirely.

    Where the backend flushes, a subnormal operand compares equal to zero and
    every arithmetic operation on it yields zero. Its magnitude is then not
    merely rounded on the way through the affine read — it is gone before the
    read begins, and no rearrangement of the arithmetic downstream can recover
    it. A link from `0` to the smallest normal, read at a subnormal query, has an
    exact affine value of `2**-23` at float32 and `2**-52` at float64; the
    compiled program publishes zero, and a rival at half the exact value then
    wins a comparison it loses.

    Only the bit pattern survives a flush, so that is what is inspected. A query
    carrying one is refused on its own; one anywhere in the row refuses every
    query, because which segment a query ends up owned by is not known here and
    the affected link may be any of them.

    Refusing is not the repair. The repair is an arithmetic that reads operands
    through their significands and exponents rather than as floats, which is a
    representation change this predicate does not attempt. What it does is make
    the failure loud, so a wrong number is never published in place of one the
    format holds perfectly well.
    """
    if not backend_flushes_subnormals(row[0].dtype):
        return jnp.zeros_like(jnp.asarray(query), dtype=bool)
    in_row = jnp.any(jnp.stack([jnp.any(is_subnormal(term)) for term in row]))
    return in_row | is_subnormal(query)


def _derived_subnormal_possible(*, links: _SegmentLinks, query: FloatND) -> BoolND:
    """Report, per query, where the read's own result lands below the range.

    Every stored operand can be normal and the exact result still be a positive
    subnormal: a link from `0` to a value near the top of the range, read close
    to its zero endpoint, weights the far value by a ratio small enough to carry
    the product out of the normal range. A backend that flushes hands back a
    finite zero — neither the represented value nor an abstention, which is the
    one outcome the contract forbids in all three channels at once.

    The predicate is on the **result**, not on the terms that build it. A term of
    the read is routinely subnormal while the result is an ordinary number the
    format holds exactly: reading a link whose two values agree weights one of
    them by a vanishing ratio, and the other by a ratio of essentially one, so
    the sum is that second value. Refusing there would withhold an answer the
    arithmetic reached correctly, and no bound over the terms can tell the two
    situations apart, because in both of them a term is subnormal.

    Two facts make the result's magnitude cheap to decide exactly:

    - The two ratios sum to one, so the endpoint nearer the query carries a ratio
      of at least a half and contributes at least half its own value. The result
      can therefore be subnormal only where that near value is exactly zero, and
      then the result *is* the far term.
    - `_subnormal_operand_present` has already refused every row carrying a
      subnormal stored operand, so each endpoint value reaching here is zero or
      normal — there is no third case for "exactly zero" to hide.

    So the question reduces to whether `|far| * |query - near| / |span|` is
    subnormal, which is settled from the three exponents rather than by forming
    the quantity: on a backend that flushes, forming it is precisely what
    destroys the answer being asked for.

    Evaluated per link, over every live link that brackets the query. Ownership
    is not consulted and does not need to be: a link whose near value is nonzero
    publishes a normal result whether or not it wins.

    A link whose span, distance or far value is not finite is left alone. The
    result there is not a flushed magnitude but an ordinary infinity or NaN,
    which the caller can already see.
    """
    dtype = links.left_grid.dtype
    query_arr = jnp.asarray(query)
    if not backend_flushes_subnormals(dtype):
        return jnp.zeros_like(query_arr, dtype=bool)

    flat = query_arr.reshape(-1, 1)
    left_grid = links.left_grid.reshape(1, -1)
    right_grid = links.right_grid.reshape(1, -1)
    span = right_grid - left_grid

    to_left = jnp.abs(flat - left_grid)
    to_right = jnp.abs(right_grid - flat)
    left_is_near = to_left <= to_right
    distance_to_near = jnp.minimum(to_left, to_right)

    brackets = (
        links.live.reshape(1, -1)
        & (span != 0.0)
        & (jnp.minimum(left_grid, right_grid) <= flat)
        & (flat <= jnp.maximum(left_grid, right_grid))
    )
    readable = (
        jnp.isfinite(span) & jnp.isfinite(distance_to_near) & (distance_to_near > 0.0)
    )

    channels = (
        (links.left_value, links.right_value),
        (links.left_policy, links.right_policy),
        (links.left_marginal, links.right_marginal),
    )
    refused = jnp.zeros_like(brackets)
    for left_channel, right_channel in channels:
        left_c = left_channel.reshape(1, -1)
        right_c = right_channel.reshape(1, -1)
        near = jnp.where(left_is_near, left_c, right_c)
        far = jnp.where(left_is_near, right_c, left_c)
        refused = refused | (
            brackets
            & readable
            & (near == 0.0)
            & (far != 0.0)
            & jnp.isfinite(far)
            & _quotient_is_subnormal(factors=(far, distance_to_near), divisor=span)
        )
    return jnp.any(refused, axis=-1).reshape(query_arr.shape)


def _quotient_is_subnormal(
    *, factors: tuple[FloatND, FloatND], divisor: FloatND
) -> BoolND:
    """Report where `|f0| * |f1| / |divisor|` is a nonzero subnormal.

    Decided from the exponents, because the product and the quotient are exactly
    the intermediates a flushing backend destroys — forming either of them would
    answer with the zero the caller is trying to detect.

    `frexp` writes each magnitude as `mantissa * 2**exponent` with the mantissa in
    `[0.5, 1)`, so the quotient's mantissa lies in `(0.25, 2)` and one halving or
    one doubling returns it to `[0.5, 1)`. Both mantissa and exponent are then
    directly comparable with the same decomposition of the smallest normal, and
    neither carries a magnitude that can leave the range on the way.
    """
    mantissa = jnp.ones_like(divisor)
    exponent = jnp.zeros_like(divisor, dtype=jnp.int32)
    for factor in factors:
        factor_mantissa, factor_exponent = jnp.frexp(jnp.abs(factor))
        mantissa = mantissa * factor_mantissa
        exponent = exponent + factor_exponent
    divisor_mantissa, divisor_exponent = jnp.frexp(jnp.abs(divisor))
    mantissa = mantissa / divisor_mantissa
    exponent = exponent - divisor_exponent

    # `frexp` writes a magnitude as `mantissa * 2**exponent` with the mantissa in
    # `[mantissa_floor, 1)`. The quotient of three such mantissas lies in
    # `(mantissa_floor / 2, 2)`, so one halving or one doubling restores the
    # normalization and makes exponent and mantissa comparable term by term.
    mantissa_floor = 0.5
    mantissa, exponent = (
        jnp.where(mantissa >= 1.0, mantissa * mantissa_floor, mantissa),
        jnp.where(mantissa >= 1.0, exponent + 1, exponent),
    )
    mantissa, exponent = (
        jnp.where(mantissa < mantissa_floor, mantissa / mantissa_floor, mantissa),
        jnp.where(mantissa < mantissa_floor, exponent - 1, exponent),
    )

    tiny_mantissa, tiny_exponent = jnp.frexp(
        jnp.asarray(jnp.finfo(divisor.dtype).tiny, dtype=divisor.dtype)
    )
    return (exponent < tiny_exponent) | (
        (exponent == tiny_exponent) & (mantissa < tiny_mantissa)
    )


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
    # to the largest slope. The fields stay separate and are compared in order, so
    # this dense reduction and the blocked scan select the same winner.
    slope_high, slope_low = _slope_words(
        left_value=left_value[None, :],
        right_value=right_value[None, :],
        left_grid=left_grid[None, :],
        right_grid=right_grid[None, :],
    )
    right_available = flat < upper
    if arithmetic == "ordinary":
        # No certificate, so nothing to order by but the reads themselves: the
        # largest wins outright and a query anything brackets is decided. Ties go
        # to the same right-continuous rank as the certified branch.
        best_read = jnp.max(
            jnp.where(brackets, value_interp, -jnp.inf), axis=1, keepdims=True
        )
        best = _lexicographic_argmax(
            _tie_break_key(
                level_with=brackets & (value_interp >= best_read),
                right_available=right_available,
                slope_high=slope_high,
                slope_low=slope_low,
            )
        )
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
            bracketing=brackets,
            query=flat,
            value=value_interp,
            right_available=right_available,
            slope_high=slope_high,
            slope_low=slope_low,
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


class _TieBreakKey(NamedTuple):
    """The right-continuous tie-break's fields, most significant first.

    Kept apart rather than folded into one number. Any single scalar has to
    bound the slope to leave room for the right-extension bit above it, and a
    bounded reparametrisation flattens as it saturates: slopes hundreds of
    representable steps apart map onto one key exactly where EGM candidates are
    steepest, and the tie-break is then handed a collision it can only settle by
    array position. Comparing the fields in order costs one more reduction and
    settles the same question without ever inventing a tie.
    """

    right_available: FloatND
    """`1` where the segment extends strictly right of the query, else `0`."""
    slope_high: FloatND
    """Leading word of the value-slope."""
    slope_low: FloatND
    """Trailing word of the value-slope, which orders slopes the leading word ties."""


def _slope_words(
    *,
    left_value: FloatND,
    right_value: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
) -> tuple[FloatND, FloatND]:
    """Return the value-slope as two words, from exact endpoint differences.

    Both differences are formed by `two_sum` and so are exact; only the division
    rounds, and it rounds into a pair rather than a single float. Two slopes are
    then ordered whenever the working precision separates them twice over, where
    one rounded float stops separating them once the slope is steep.

    A zero-width link has no slope. Its divisor is displaced to keep the
    quotient finite; the tie-break only ever reads it against another
    zero-width link, which the right-extension field has already ordered.
    """
    width = dd_from_difference(right_grid, left_grid)
    safe_width = (
        jnp.where(width[0] == 0.0, jnp.ones_like(width[0]), width[0]),
        jnp.where(width[0] == 0.0, jnp.zeros_like(width[1]), width[1]),
        width[2],
    )
    return dd_quotient(dd_from_difference(right_value, left_value), safe_width)


def _tie_break_key(
    *,
    level_with: BoolND,
    right_available: BoolND,
    slope_high: FloatND,
    slope_low: FloatND,
) -> _TieBreakKey:
    """Return the ordered fields the tie-break compares, per segment.

    A segment not level with the winner is given `-inf` in every field, so it
    loses on the first comparison and never reaches the later ones.
    """
    excluded = jnp.full_like(slope_high, -jnp.inf)
    return _TieBreakKey(
        right_available=jnp.where(
            level_with, right_available.astype(slope_high.dtype), excluded
        ),
        slope_high=jnp.where(level_with, slope_high, excluded),
        slope_low=jnp.where(level_with, slope_low, excluded),
    )


def _lexicographic_argmax(key: _TieBreakKey) -> IntND:
    """Return the column winning the ordered comparison, per query.

    Each field narrows the field of candidates still tied on every field before
    it. Segments excluded by `_tie_break_key` carry `-inf` throughout, so they
    survive only where nothing else does at all.
    """
    still_tied = jnp.ones_like(key.right_available, dtype=bool)
    for field in key:
        best = jnp.max(jnp.where(still_tied, field, -jnp.inf), axis=1, keepdims=True)
        still_tied = still_tied & (field == best)
    return jnp.argmax(still_tied, axis=1)[:, None].astype(jnp.int32)


def _outranks(*, challenger: _TieBreakKey, held: _TieBreakKey) -> BoolND:
    """Whether one segment's fields beat another's in order.

    The blocked scan carries the standing winner's fields across blocks, so the
    same ordering has to be expressible between two single segments rather than
    across a row. Comparing in order is the same rule the dense reduction
    applies, so both paths name the same winner.
    """
    decided = jnp.zeros_like(challenger[0], dtype=bool)
    beaten = jnp.zeros_like(challenger[0], dtype=bool)
    for challenging, standing in zip(challenger, held, strict=True):
        beaten = beaten | (~decided & (challenging > standing))
        decided = decided | (challenging != standing)
    return beaten


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
    slope_high: FloatND
    """Leading word of the link's value-slope, for the right-continuous tie-break."""
    slope_low: FloatND
    """Trailing word of the same slope, which orders what the leading word ties."""
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
    slope_high, slope_low = _slope_words(
        left_value=left_value[None, :],
        right_value=right_value[None, :],
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
    )
    return _BlockTerms(
        brackets=brackets,
        value=value_interp,
        policy=policy_interp,
        marginal=marginal_interp,
        slope_high=slope_high,
        slope_low=slope_low,
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
    - The final pass re-scans once more and, among segments certified exactly
      level with the reference, keeps the winner of the right-continuous
      tie-break (`_tie_break_key`: a right-extending segment over one ending at
      the query, then larger value-slope) — the dense path's tie-break. Comparing
      the fields in order across blocks keeps the earliest such winner, matching
      the dense selection, and value, policy, and marginal are all published from
      it. The same pass
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
        carry: tuple[_TieBreakKey, FloatND, FloatND, FloatND, BoolND, BoolND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[_TieBreakKey, FloatND, FloatND, FloatND, BoolND, BoolND], None]:
        best_key, best_value, best_policy, best_marginal, any_bracket, settled = carry
        block, block_live = block_and_live
        terms = _block_query_terms(block=block, live=block_live, flat=flat)
        lines = _block_lines(block=block, live=block_live)
        contending = block_contending(terms, lines)
        sign = _sign_against_reference(
            lines=lines, reference=reference, query=flat[:, None]
        )
        key = _tie_break_key(
            level_with=contending & (sign == 0),
            right_available=flat[:, None] < terms.upper,
            slope_high=terms.slope_high,
            slope_low=terms.slope_low,
        )
        winner = _lexicographic_argmax(key)

        def _take(channel: FloatND) -> FloatND:
            return jnp.take_along_axis(channel, winner, axis=1)[:, 0]

        block_key = _TieBreakKey(*(_take(field) for field in key))
        take = _outranks(challenger=block_key, held=best_key)
        return (
            _TieBreakKey(
                *(
                    jnp.where(take, challenging, standing)
                    for challenging, standing in zip(block_key, best_key, strict=True)
                )
            ),
            jnp.where(take, _take(terms.value), best_value),
            jnp.where(take, _take(terms.policy), best_policy),
            jnp.where(take, _take(terms.marginal), best_marginal),
            any_bracket | jnp.any(terms.brackets, axis=1),
            settled
            & ~jnp.any(terms.brackets & (sign == 1), axis=1)
            & ~jnp.any(terms.brackets & (sign == UNRESOLVED_SIGN), axis=1)
            & ~jnp.any(terms.brackets & (sign == BELOW_RESOLUTION_SIGN), axis=1),
        ), None

    empty = jnp.full((n_query,), jnp.nan, dtype=dtype)
    nothing_yet = jnp.full((n_query,), -jnp.inf, dtype=dtype)
    (_, env_value, env_policy, env_marginal, any_bracket, settled), _ = jax.lax.scan(
        winner_step,
        (
            _TieBreakKey(nothing_yet, nothing_yet, nothing_yet),
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


# ---------------------------------------------------------------------------
# PARKED -- this branch's envelope-query reduction, deliberately off the code path.
#
# `feat/nb-egm` and `feat/continuous-outer` independently rewrote
# `envelope_at_query`: upstream toward `_envelope_dense` / `_envelope_blocked`
# (double-double escalation, `1ce705ba`), this branch toward an exactly-certified
# winner (`_candidate_terms` / `_exactly_maximal`, contouter rounds 5-13). The two
# share no helper, so a merge can run one or the other, never both.
#
# HMvG's call (2026-08-06): run upstream's path, park this branch's, revisit once
# upstream is finalized. `_envelope_at_query_407` below is this branch's reduction
# verbatim, intentionally unreferenced. Every exact-arithmetic primitive it and
# `nbegm_step.py` depend on stays defined, so nothing importing them breaks:
# reviving it is a one-line rename back.
# ---------------------------------------------------------------------------


def _dekker_split_factor(dtype: jnp.dtype) -> float:
    """Dekker splitting constant `2**ceil(p/2) + 1` for a p-bit significand."""
    return float(2 ** ((jnp.finfo(dtype).nmant + 2) // 2) + 1)


def _two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """`a + b` as `(fl(a + b), exact residual)` — Knuth's TwoSum."""
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


def _fast_two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """TwoSum for pre-ordered operands (`|a| >= |b|` or `a == 0`)."""
    s = a + b
    return s, b - (s - a)


def _two_diff(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """`a - b` as `(fl(a - b), exact residual)`."""
    d = a - b
    t = d - a
    return d, (a - (d - t)) - (b + t)


def _two_prod(a: FloatND, b: FloatND, split: float) -> tuple[FloatND, FloatND]:
    """`a * b` as `(fl(a * b), exact residual)` via Dekker splitting."""
    p = a * b
    ca = split * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = split * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    return p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def _dd_add_fp(ah: FloatND, al: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Double-double plus float, renormalised."""
    sh, sl = _two_sum(ah, b)
    return _fast_two_sum(sh, sl + al)


def _dd_add(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND
) -> tuple[FloatND, FloatND]:
    """Double-double plus double-double (accurate variant)."""
    sh, sl = _two_sum(ah, bh)
    th, tl = _two_sum(al, bl)
    sh, sl = _fast_two_sum(sh, sl + th)
    return _fast_two_sum(sh, sl + tl)


def _dd_mul_fp(
    ah: FloatND, al: FloatND, b: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double times float."""
    ph, pl = _two_prod(ah, b, split)
    return _fast_two_sum(ph, pl + al * b)


def _dd_mul(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double times double-double."""
    ph, pl = _two_prod(ah, bh, split)
    return _fast_two_sum(ph, pl + (ah * bl + al * bh))


def _dd_div(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double division: long division with two residual corrections."""
    q1 = ah / bh
    th, tl = _dd_mul_fp(bh, bl, q1, split)
    rh, rl = _dd_add(ah, al, -th, -tl)
    q2 = rh / bh
    th, tl = _dd_mul_fp(bh, bl, q2, split)
    rh, rl = _dd_add(rh, rl, -th, -tl)
    q3 = rh / bh
    qh, ql = _fast_two_sum(q1, q2)
    return _dd_add_fp(qh, ql, q3)


_INTERIOR_RADIUS_ULPS2 = 16.0


_SCREEN_SLACK = 4.0


_EXACT_TERMS = 64


_SLOPE_SCREEN_ULPS = 8.0


def _one_hot(index: jax.Array, width: int) -> BoolND:
    """Row-wise indicator of `index` over `width` columns."""
    return jnp.arange(width) == index[:, None]


def _node_selection(
    *,
    q: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
) -> tuple[BoolND, BoolND]:
    """Node-event flag and which stored endpoint the candidate publishes.

    Shared by the double-double evaluation and the exact comparison so the two
    can never disagree about which lane is a node or which stored float that
    node publishes.
    """
    at_left = q == left_grid
    at_right = q == right_grid
    # A zero-width segment sets both flags; publish its higher end (left on a
    # value tie, matching the oracle's vertical-edge rule).
    use_right = (at_right & ~at_left) | (
        at_left & at_right & (right_value > left_value)
    )
    return at_left | at_right, use_right


class _Dyadic(NamedTuple):
    """Exactly-represented reals `mantissa * 2**exponent`, one per trailing slot.

    Every quantity in the ordering kernel travels this way: a finite float
    mantissa and a SEPARATE `int32` exponent. That split is the whole design.
    An exponent held as an integer never rounds, never overflows and never
    flushes, so a term's magnitude may range over the entire real line while its
    mantissa stays inside one binade — the binade where every error-free
    transform below is exact.

    Nine consecutive repairs to this kernel (rounds 6 to 12) each moved the same
    defect one layer down: a value comparison, a slope tie-break, an exponent, a
    normalization's scope, then its shape, then the last normalization inside
    the "exact" fallback. All nine shared one premise — that a rescaling can
    make a rounded evaluation safe. It cannot: scaling operands DOWN merges
    distinct floats and scaling them UP eventually overflows, and no single
    binade holds a problem whose terms span more binades than the format has.
    Carrying the exponent outside the float removes the premise instead of
    patching its latest consequence.
    """

    mantissa: FloatND
    exponent: jax.Array


class _ExactRatio(NamedTuple):
    """A candidate's value at the query as an exact rational `num / den`.

    `numerator` and `denominator` are unevaluated sums of dyadic terms — error
    free transforms throughout, so no information has been discarded and no
    term's magnitude has been constrained. The denominator is canonically
    positive, which lets a cross-multiplied comparison read its sign straight
    off the numerator difference.
    """

    numerator: _Dyadic
    denominator: _Dyadic


def _dyadic_parts(x: FloatND) -> tuple[FloatND, jax.Array]:
    """`x` as `(m, e)` with `m` in `[0.5, 1)` and `x == m * 2**e` EXACTLY.

    Zero and non-finite inputs carry exponent zero and pass through unchanged,
    so a NaN stays a NaN and surfaces rather than being silently ordered.
    """
    mantissa, exponent = jnp.frexp(x)
    usable = jnp.isfinite(x) & (x != 0)
    return mantissa, jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _binade_exponent(magnitude: FloatND) -> jax.Array:
    """Exponent `e` with `magnitude` in `[2**(e-1), 2**e)`; `0` where there is none.

    Returned as an exponent rather than a factor so callers scale with `ldexp`.
    Materializing `2**-e` as a float would itself be lossy at the extremes: for
    a float32 magnitude near the top of the range `2**-e` is SUBNORMAL, so the
    factor carries fewer bits than the scaling it is meant to perform exactly.
    `ldexp` adjusts the exponent field directly and has no such failure mode.

    Used only by the double-double SCREEN, which may round; the exact ordering
    kernel carries its exponents in `_Dyadic` and normalizes nothing.
    """
    _, exponent = jnp.frexp(magnitude)
    usable = (magnitude > 0) & jnp.isfinite(magnitude)
    return jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _as_dyadic(x: FloatND) -> _Dyadic:
    """A raw float as a one-term dyadic list; its own exponent is implicit."""
    return _Dyadic(
        mantissa=x[..., None],
        exponent=jnp.zeros((*x.shape, 1), dtype=jnp.int32),
    )


def _dyadic_join(*parts: _Dyadic) -> _Dyadic:
    """Concatenate dyadic term lists — the exact sum of the parts."""
    return _Dyadic(
        mantissa=jnp.concatenate([part.mantissa for part in parts], axis=-1),
        exponent=jnp.concatenate([part.exponent for part in parts], axis=-1),
    )


def _dyadic_negate(terms: _Dyadic, flip: BoolND) -> _Dyadic:
    """Negate every term where `flip`; exact, and it leaves the exponents alone."""
    return _Dyadic(
        mantissa=jnp.where(flip[..., None], -terms.mantissa, terms.mantissa),
        exponent=terms.exponent,
    )


def _exact_difference(a: FloatND, b: FloatND) -> _Dyadic:
    """`a - b` as an UNEVALUATED two-term dyadic sum. No subtraction is performed.

    Round 13 formed the difference with `_two_diff` on operands LIFTED into
    mid-range, and justified it with the hinge claim that "subtraction of two
    finite floats cannot overflow". That claim is false, and opposite-signed
    top-binade operands are the counterexample: for `H = 2**127` in float32 or
    `2**1023` in float64 — both finite normals — the lift is zero because the
    operands already exceed the target, and `_two_diff(H, -H)` returns
    `(inf, nan)`. A finite exact envelope was then published as three NaNs
    (round-13 audit F2).

    Rounds 8 to 12 guarded the same operation against UNDERFLOW and round 13
    against nothing else; each repair fixed the direction its witness pointed at
    and left the mirror image open. The fix is to stop performing the operation.

    `a = m_a * 2**e_a` and `b = m_b * 2**e_b` exactly, from `frexp`, so
    `a - b` is exactly the two-term sum `[(m_a, e_a), (-m_b, e_b)]`. Each
    mantissa is already in `[0.5, 1)`, each exponent is an int32, and NOTHING is
    added, subtracted, scaled or rounded to build it. There is no direction left
    in which to be wrong: the construction is total over every pair of finite
    floats, at every relative scale, and `_exact_sign_of_sum` consumes exactly
    such term lists already.

    The two terms are also strictly TIGHTER than the lifted pair they replace —
    they live in the operands' own binades rather than a head one binade above
    and a residual `precision` binades below — so the bound in
    `_accumulator_layout` still holds with room to spare.
    """
    mantissa_a, exponent_a = _dyadic_parts(a)
    mantissa_b, exponent_b = _dyadic_parts(b)
    return _Dyadic(
        mantissa=jnp.stack([mantissa_a, -mantissa_b], axis=-1),
        exponent=jnp.stack([exponent_a, exponent_b], axis=-1),
    )


def _framed_difference(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND, jax.Array]:
    """`(head, tail, exponent)` with `a - b == (head + tail) * 2**exponent`.

    The double-double SCREEN's counterpart to `_exact_difference`, and the answer
    to the same round-13 finding on the public path: `_candidate_terms` formed
    its grid and value differences in the working dtype after a LIFT-ONLY shift,
    so two opposite-signed top-binade operands overflowed there exactly as they
    did in the exact kernel, and the published triple was `(NaN, NaN, NaN)`.

    Both operands are first divided by `2**max(e_a, e_b)` — which is `ldexp` on
    the `frexp` mantissa, hence exact — landing them in `(-1, 1]`. `_two_diff` on
    such a pair has `|head| <= 2` and a residual at most `precision` binades
    below, so it can neither overflow nor lose its tail, whatever the operands'
    own magnitudes. The frame rides along as an INTEGER exponent.

    Where the operands are more than `precision + |minexp|` binades apart the
    smaller flushes to zero in the frame, and the head is then the larger operand
    to within `2**-precision-|minexp|` relative — some 100 binades below the
    certified radius, so the screen stays honest. That is the ONLY approximation
    here, it is bounded, and the exact predicate does not use this function.
    """
    mantissa_a, exponent_a = _dyadic_parts(a)
    mantissa_b, exponent_b = _dyadic_parts(b)
    exponent = jnp.maximum(exponent_a, exponent_b)
    head, tail = _two_diff(
        jnp.ldexp(mantissa_a, exponent_a - exponent),
        jnp.ldexp(mantissa_b, exponent_b - exponent),
    )
    return head, tail, exponent


class _FramedAffine(NamedTuple):
    """`left + r*(right - left)` as a double-double in an integer frame.

    `hi`/`lo` are in ORIGINAL units. `framed_left` and `framed_product` are the
    two addends inside the frame, kept so a caller can size a certified radius
    against them without leaving the frame.
    """

    hi: FloatND
    lo: FloatND
    framed_left: FloatND
    framed_product: FloatND
    frame: jax.Array


def _framed_affine(
    *,
    left: FloatND,
    right: FloatND,
    ratio_hi: FloatND,
    ratio_lo: FloatND,
    ratio_exp: jax.Array,
    split: float,
) -> _FramedAffine:
    """Interpolate ONE affine channel exponent-preservingly, rounding once.

    Round 14 gave the VALUE channel this treatment and left `policy` and
    `marginal` on `left + fraction*(right - left)` in the working dtype, where
    `fraction = ldexp(rh, t_exp - w_exp)`. Both halves of that expression lose
    finite results (round-14 audit F2):

    - `fraction` is materialized BEFORE it is multiplied, so it can flush to
      zero even when `fraction * (right - left)` is a finite normal. With
      `x0 = 1`, `q = nextafter(1)`, `x1 = 2**(maxexp-2)` and outputs
      `[0, x1]` the exact result is `ulp(1)` and production published `0`.
    - `right - left` is a raw subtraction, so opposite-signed top-binade
      outputs overflow: `[H, -H]` at `q = 1/2` has exact result `0` and
      production published `-inf`.

    The ratio therefore never becomes a float. It stays a significand pair with
    a separate integer exponent, the endpoint difference is framed, the product
    is formed from significands alone, and the sum with `left` happens in a
    frame that holds BOTH addends — the single rounding is at publication.

    The winner's value, policy and marginal all go through this one function, so
    a channel cannot be repaired while its siblings are forgotten. That is how
    this defect survived round 14: the value was certified, and the outputs
    gathered from the very same certified winner were not.
    """
    difference_hi, difference_lo, difference_frame = _framed_difference(right, left)
    difference_exp = _binade_exponent(jnp.abs(difference_hi)) + difference_frame
    difference_shift = difference_frame - difference_exp

    product_hi, product_lo = _dd_mul(
        ratio_hi,
        ratio_lo,
        jnp.ldexp(difference_hi, difference_shift),
        jnp.ldexp(difference_lo, difference_shift),
        split,
    )
    product_exp = ratio_exp + difference_exp

    # A frame that holds both addends. Anchoring on the product alone overflows
    # whenever `r*(right-left)` does, even where the sum does not.
    _, left_exp = _dyadic_parts(left)
    frame = jnp.maximum(product_exp, left_exp)
    framed_product = jnp.ldexp(product_hi, product_exp - frame)
    framed_product_lo = jnp.ldexp(product_lo, product_exp - frame)
    framed_left = jnp.ldexp(left, -frame)
    hi, lo = _dd_add_fp(framed_product, framed_product_lo, framed_left)
    return _FramedAffine(
        hi=jnp.ldexp(hi, frame),
        lo=jnp.ldexp(lo, frame),
        framed_left=framed_left,
        framed_product=framed_product,
        frame=frame,
    )


def _dyadic_product(a: _Dyadic, b: _Dyadic, split: float) -> _Dyadic:
    """Exact outer product of two dyadic term lists: `2 * k_a * k_b` terms.

    Each pair of mantissas is normalized into `[0.5, 1)` BEFORE multiplying, so
    `_two_prod` always runs on operands whose product lands in `[0.25, 1)`: it
    can neither overflow to infinity nor lose its residual to underflow, at any
    input scale whatsoever. Dekker's split constant is likewise safe there,
    which retires the caveat at the top of this module for the exact path.

    The magnitudes ride along in the integer exponents, where nothing rounds.
    This is what the round-12 finding asked for: the decisive product of a
    half-minimum-normal numerator term with a half-minimum-normal denominator
    term used to flush to zero before the exact summation ever saw it.
    """
    b_parts = [_dyadic_parts(b.mantissa[..., j]) for j in range(b.mantissa.shape[-1])]
    mantissas: list[FloatND] = []
    exponents: list[jax.Array] = []
    for i in range(a.mantissa.shape[-1]):
        a_mantissa, a_exponent = _dyadic_parts(a.mantissa[..., i])
        a_exponent = a_exponent + a.exponent[..., i]
        for j, (b_mantissa, b_exponent) in enumerate(b_parts):
            head, tail = _two_prod(a_mantissa, b_mantissa, split)
            exponent = a_exponent + b_exponent + b.exponent[..., j]
            mantissas += [head, tail]
            exponents += [exponent, exponent]
    return _Dyadic(
        mantissa=jnp.stack(mantissas, axis=-1),
        exponent=jnp.stack(exponents, axis=-1),
    )


class _AccumulatorLayout(NamedTuple):
    """Geometry of the exact fixed-point accumulator, derived from the dtype."""

    limb_bits: int
    n_limb: int
    n_digit: int


def _accumulator_layout(dtype: jnp.dtype) -> _AccumulatorLayout:
    """Limb width, limb count and digits per term of the exact accumulator.

    Derived from the format rather than tabulated, so float32 and float64 cannot
    drift apart and a wrong constant cannot hide behind a passing float64 test.

    A limb holds a signed INTEGER in a float of the working dtype. It receives at
    most one digit per term plus the carry from the limb below, so it needs
    `log2(_EXACT_TERMS)` guard bits above `limb_bits` to stay exact.

    The limb count covers the full exponent span a cross-product term can reach.
    That span is what the previous nine repairs kept trying to compress into one
    binade: a value and a grid difference each range over the whole format, and
    their products then range over twice it. Nothing here compresses it — the
    accumulator is simply wide enough to hold it, which is why no input scale can
    make a term vanish before the sign is read.
    """
    info = jnp.finfo(dtype)
    precision = int(info.nmant) + 1
    # `frexp` exponents of the smallest and largest finite magnitudes.
    min_exponent = int(info.minexp) + 1
    max_exponent = int(info.maxexp)

    limb_bits = precision - (_EXACT_TERMS.bit_length() + 2)
    n_digit = -(-precision // limb_bits) + 1

    # A difference contributes its two OPERANDS as terms (`_exact_difference`
    # performs no subtraction), so each sits in the operand's own binade — inside
    # `[min_exponent, max_exponent]`. The wider window kept here is the round-13
    # bound for a rounded head-plus-residual pair; it strictly CONTAINS the
    # current one, so it stays valid, and holding it fixed keeps the layout the
    # one the round-13 review verified directly at 2,000/2,000 sign matches. A
    # product of two normalized mantissas puts its own residual at most
    # `2 * precision` below its head.
    difference_hi = max_exponent + 1
    difference_lo = min_exponent - precision - 1
    numerator_hi = max_exponent + difference_hi
    numerator_lo = min_exponent + difference_lo - 2 * precision
    cross_hi = numerator_hi + difference_hi
    cross_lo = numerator_lo + difference_lo - 2 * precision

    span = cross_hi - cross_lo
    n_limb = -(-(span + n_digit * limb_bits) // limb_bits) + 2
    return _AccumulatorLayout(limb_bits=limb_bits, n_limb=n_limb, n_digit=n_digit)


def _exact_sign_of_sum(terms: _Dyadic) -> FloatND:
    """Exact sign of `sum(mantissa * 2**exponent)`, with no rounding anywhere.

    Every term is deposited into a fixed-point accumulator of `n_limb` limbs,
    each limb an exact integer weighted by `2**(-limb_bits)` relative to the one
    above it and anchored at the largest term exponent present. A term's mantissa
    is sliced into `n_digit` such digits by repeated `trunc`, which is exact, and
    the digits are scattered into consecutive limbs. Nothing is rounded, nothing
    is compared against a tolerance, and no term is ever too small to contribute:
    a term 3000 binades below the leading one simply lands 3000 binades further
    down the accumulator.

    That is the difference from the expansion this replaces. Shewchuk's
    GROW-EXPANSION is exact for the terms it RECEIVES, and the loss was always
    upstream — in forming the terms at all. A fixed-point accumulator has no
    upstream: the exponent selects a limb, and selecting a limb cannot round.

    Carries are then propagated once from the least significant limb upward,
    leaving every limb in `[0, 2**limb_bits)` and the sum equal to
    `carry * 2**(n_limb * limb_bits) + (non-negative remainder)`. So the carry
    out decides the sign, and only an all-zero accumulator with zero carry is a
    TRUE tie — the one thing a rounded evaluation can never certify.
    """
    dtype = terms.mantissa.dtype
    limb_bits, n_limb, n_digit = _accumulator_layout(dtype)
    scale = float(2**limb_bits)

    mantissa, exponent = _dyadic_parts(terms.mantissa)
    exponent = exponent + terms.exponent
    active = mantissa != 0
    finite = jnp.all(jnp.isfinite(terms.mantissa), axis=-1)

    # A sentinel at or below every exponent present, so an all-zero lane picks a
    # harmless anchor instead of an out-of-range one.
    absent = jnp.min(exponent) - 1
    top = jnp.max(jnp.where(active, exponent, absent), axis=-1, keepdims=True)
    shift = jnp.where(active, top - exponent, jnp.zeros_like(exponent))
    limb = shift // limb_bits
    # `|mantissa| < 1` and the intra-limb offset is below `limb_bits`, so the
    # carrier stays inside one limb's range and its lowest bit is `precision`
    # bits below it — which is what `n_digit` is sized to drain exactly.
    carrier = jnp.ldexp(
        jnp.where(active, mantissa, jnp.zeros_like(mantissa)),
        limb_bits - (shift - limb * limb_bits),
    )

    indices: list[jax.Array] = []
    digits: list[FloatND] = []
    residue = carrier
    for offset in range(n_digit):
        digit = jnp.trunc(residue)
        indices.append(limb + offset)
        digits.append(digit)
        residue = jnp.ldexp(residue - digit, limb_bits)

    leading = terms.mantissa.shape[:-1]
    n_lane = 1
    for size in leading:
        n_lane *= size
    flat_index = jnp.concatenate(indices, axis=-1).reshape(n_lane, -1)
    flat_digit = jnp.concatenate(digits, axis=-1).reshape(n_lane, -1)
    accumulator = (
        jnp.zeros((n_lane, n_limb), dtype)
        .at[jnp.arange(n_lane)[:, None], flat_index]
        .add(flat_digit, mode="drop")
    )

    def propagate(carry: FloatND, limb_value: FloatND) -> tuple[FloatND, FloatND]:
        total = limb_value + carry
        quotient = jnp.floor(total * (1.0 / scale))
        return quotient, total - quotient * scale

    carry, residues = jax.lax.scan(
        propagate,
        jnp.zeros((n_lane,), dtype),
        jnp.moveaxis(accumulator, -1, 0)[::-1],
    )
    sign = jnp.where(
        carry < 0,
        -jnp.ones_like(carry),
        jnp.where(
            (carry > 0) | jnp.any(residues != 0, axis=0),
            jnp.ones_like(carry),
            jnp.zeros_like(carry),
        ),
    ).reshape(leading)

    # `_accumulator_layout` proves this cannot fire; it is kept because a silent
    # `mode="drop"` is exactly the failure this rewrite exists to remove, and a
    # loud NaN is recoverable evidence where a dropped digit is not.
    dropped = jnp.any(active & (limb + n_digit > n_limb), axis=-1)
    return jnp.where(finite & ~dropped, sign, jnp.full_like(sign, jnp.nan))


def _exact_cross_sign(*, a: _ExactRatio, b: _ExactRatio, dtype: jnp.dtype) -> FloatND:
    """Exact sign of `a - b` for two rationals with positive denominators.

    `a - b = (N_a*D_b - N_b*D_a) / (D_a*D_b)`, so with both denominators
    canonically positive the sign is that of the cross-multiplied numerator
    difference. Every cross product is formed on normalized mantissas with its
    exponent carried alongside, so the terms sum to that difference with no error
    at ANY model scale — not merely at the scale the tests happened to use.

    This is the single ordering kernel: value selection and the right-continuous
    slope tie-break share it, so the two can never disagree about what an exact
    tie is.
    """
    split = _dekker_split_factor(dtype)
    forward = _dyadic_product(a.numerator, b.denominator, split)
    reverse = _dyadic_product(b.numerator, a.denominator, split)
    return _exact_sign_of_sum(
        _dyadic_join(forward, _Dyadic(-reverse.mantissa, reverse.exponent))
    )


def _exact_ratio(*, cols: FloatND, q: FloatND) -> _ExactRatio:
    """Exact rational value of one candidate per query, from its raw columns.

    `V = (v0*(x1 - q) + v1*(q - x0)) / (x1 - x0)` is the same interpolant
    `_candidate_terms` evaluates, but every factor is kept as a dyadic term list
    instead of being rounded, so the pair `(numerator, denominator)` represents
    `V` with no error at all. A node event publishes a STORED float, so its exact
    value is that float over one — exact by construction, and expressible here
    with a unit denominator and no arithmetic at all.

    **No scale is shared and none is passed in.** Earlier rounds took a value
    exponent from the caller so both candidates' values could be compared in one
    frame; that frame is what underflowed. With the exponent carried per term,
    `V_a` and `V_b` are comparable because they are EXACT, not because they were
    manoeuvred into common units — and the grid differences, which used to need a
    group scale of their own, need none either.
    """
    dtype = cols.dtype
    split = _dekker_split_factor(dtype)
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]

    left_weight = _exact_difference(right_grid, q)
    right_weight = _exact_difference(q, left_grid)
    width = _exact_difference(right_grid, left_grid)

    numerator = _dyadic_join(
        _dyadic_product(_as_dyadic(left_value), left_weight, split),
        _dyadic_product(_as_dyadic(right_value), right_weight, split),
    )

    # Canonical orientation. Endpoints may be stored in either order within a
    # branch; negating both numerator and denominator leaves `V` unchanged.
    #
    # Read off the STORED endpoints, not off a leading term. Under the round-13
    # representation the width's first term was the rounded difference, whose
    # sign was the width's sign; it is now the frexp mantissa of `right_grid`,
    # whose sign is that endpoint's. A comparison of two finite floats is exact
    # and needs no difference to be formed at all.
    flip = right_grid < left_grid
    numerator = _dyadic_negate(numerator, flip)
    denominator = _dyadic_negate(width, flip)

    node, use_right = _node_selection(
        q=q,
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
    )
    node_value = jnp.where(use_right, right_value, left_value)
    zeros = jnp.zeros_like(node_value)
    is_node = node[..., None]
    node_numerator = jnp.stack(
        [node_value, *([zeros] * (numerator.mantissa.shape[-1] - 1))], axis=-1
    )
    node_denominator = jnp.stack([jnp.ones_like(node_value), zeros], axis=-1)
    return _ExactRatio(
        numerator=_Dyadic(
            mantissa=jnp.where(is_node, node_numerator, numerator.mantissa),
            exponent=jnp.where(
                is_node, jnp.zeros_like(numerator.exponent), numerator.exponent
            ),
        ),
        denominator=_Dyadic(
            mantissa=jnp.where(is_node, node_denominator, denominator.mantissa),
            exponent=jnp.where(
                is_node, jnp.zeros_like(denominator.exponent), denominator.exponent
            ),
        ),
    )


def _exact_compare(*, cols_a: FloatND, cols_b: FloatND, q: FloatND) -> FloatND:
    """Exact sign of `V_a(q) - V_b(q)`: `+1`, `-1`, or `0` for a TRUE tie.

    `V_a - V_b = (N_a*D_b - N_b*D_a) / (D_a*D_b)` with both denominators
    positive, so the sign is that of the cross-multiplied numerator difference.
    This is the only test that can tell a genuine tie from a strict gap finer
    than the working precision — double-double values cannot, because
    algebraically different segment parameterizations of the SAME exact value
    need not produce the same low word (round-6 audit F2).

    The columns are handed over RAW. Every difference is formed on them, every
    product carries its own exponent, and the sum is accumulated in fixed point:
    there is no scale to choose, hence no scale that can be chosen wrongly.
    """
    return _exact_cross_sign(
        a=_exact_ratio(cols=cols_a, q=q),
        b=_exact_ratio(cols=cols_b, q=q),
        dtype=cols_a.dtype,
    )


def _exact_slope_ratio(*, cols: FloatND) -> _ExactRatio:
    """A candidate's value slope as an exact rational `(v1-v0)/(x1-x0)`.

    Both differences are dyadic pairs, so the ratio carries no rounding at all.
    The denominator is canonically positive so a cross-multiplied comparison
    reads its sign off the numerator difference.

    A slope's numerator and denominator scales do NOT cancel within one candidate
    — they change the ratio — which used to force BOTH on the caller as shared
    exponents. Carrying the exponent per term removes that coupling: neither
    scale is chosen at all, so neither can be chosen wrongly.
    """
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]

    numerator = _exact_difference(right_value, left_value)
    width = _exact_difference(right_grid, left_grid)

    # A zero-width segment has no grid span; the established convention publishes
    # its raw value jump as the slope, i.e. a unit denominator. Under exponent
    # tagging that unit is literally `1 * 2**0`, so the convention is exact by
    # construction rather than by surviving a scale choice.
    zero_width = (right_grid == left_grid)[..., None]
    ones = jnp.ones_like(width.mantissa[..., :1])
    unit = jnp.concatenate([ones, jnp.zeros_like(width.mantissa[..., 1:])], axis=-1)
    width = _Dyadic(
        mantissa=jnp.where(zero_width, unit, width.mantissa),
        exponent=jnp.where(zero_width, jnp.zeros_like(width.exponent), width.exponent),
    )

    # As in `_exact_ratio`: the orientation is a property of the STORED
    # endpoints, read by an exact float comparison rather than off a term.
    flip = right_grid < left_grid
    return _ExactRatio(
        numerator=_dyadic_negate(numerator, flip),
        denominator=_dyadic_negate(width, flip),
    )


def _exact_slope_compare(*, cols_a: FloatND, cols_b: FloatND) -> FloatND:
    """Exact sign of `slope_a - slope_b`: `+1`, `-1`, or `0` for a TRUE tie.

    Same cross-multiplied construction as `_exact_compare`, on the slope ratio
    instead of the value.

    This exists because an exact VALUE tie is only half the rule. The
    right-continuous break then orders slopes, and ordering them on
    `fl((v1-v0)/(x1-x0))` re-introduces the very defect the exact value predicate
    removed one operation earlier: two strictly ordered exact slopes can share a
    single float key, and `argmax` then silently falls back to candidate order,
    so a pure branch permutation flips the published policy and marginal
    (round-7 audit F2).
    """
    return _exact_cross_sign(
        a=_exact_slope_ratio(cols=cols_a),
        b=_exact_slope_ratio(cols=cols_b),
        dtype=cols_a.dtype,
    )


class _SlopeState(NamedTuple):
    """Running state of the exact slope resolution loop, one entry per query."""

    lead_cols: FloatND
    lead_index: jax.Array
    remaining: BoolND


class _ResolveState(NamedTuple):
    """Running state of the exact resolution loop, one entry per query."""

    lead_cols: FloatND
    tied: BoolND
    remaining: BoolND


def _exactly_maximal(
    *,
    terms: _CandidateTerms,
    gather: Callable[[jax.Array], FloatND],
    q: FloatND,
) -> BoolND:
    """Mask of the exactly-maximal bracketing candidates, per query.

    Two stages, and the split is the point of the design:

    1. **Certified screen.** The double-double leader's interval is compared
       against every other candidate's. Anything whose certified interval lies
       strictly below the leader's is certified to lose and is discarded with
       no further work — this is where the radius, previously computed and then
       ignored, actually enters selection. The screen is deliberately generous:
       it must be a superset, and a superset only costs comparisons.
    2. **Exact resolution.** Whatever the screen could not separate is resolved
       by `_exact_compare`, which is exact and therefore recognises a true tie
       as a true tie. Candidates that are *certifiably* tied already — node
       events, which publish stored data and so are exact BY CONSTRUCTION, with
       bitwise-equal `(hi, lo)` pairs — skip it, since exactness has nothing to
       add there. That certificate is structural on purpose: see `certified_tie`.

    Consequently the loop body runs zero times on the overwhelmingly common
    query: at a node every survivor is a certified exact tie, and in the
    interior the runner-up is certified strictly below. `while_loop` under
    `vmap` executes only as many iterations as some lane still needs, so an
    empty screen costs nothing at all.
    """
    n_segment = terms.brackets.shape[1]
    masked_hi = jnp.where(terms.brackets, terms.value_hi, -jnp.inf)
    max_hi = jnp.max(masked_hi, axis=1, keepdims=True)
    hi_tied = terms.brackets & (masked_hi == max_hi)
    masked_lo = jnp.where(hi_tied, terms.value_lo, -jnp.inf)
    max_lo = jnp.max(masked_lo, axis=1, keepdims=True)
    dd_tied = hi_tied & (masked_lo == max_lo)

    lead = jnp.argmax(dd_tied, axis=1)
    lead_hot = _one_hot(lead, n_segment)
    lead_hi = jnp.take_along_axis(terms.value_hi, lead[:, None], axis=1)
    lead_lo = jnp.take_along_axis(terms.value_lo, lead[:, None], axis=1)
    lead_radius = jnp.take_along_axis(terms.radius, lead[:, None], axis=1)

    # Certified screen: keep candidate i when `V_i + r_i >= V_lead - r_lead`,
    # i.e. when its certified interval still reaches the leader's. Sharing a
    # high word is kept unconditionally — that is exactly the class where the
    # low word is rounding residue rather than order.
    gap_hi, gap_lo = _dd_add(terms.value_hi, terms.value_lo, -lead_hi, -lead_lo)
    reach = _SCREEN_SLACK * (terms.radius + lead_radius)
    contends = terms.brackets & (
        ((gap_hi + gap_lo) >= -reach) | (terms.value_hi == lead_hi)
    )
    # Skipping the exact comparison needs a STRUCTURAL certificate of exactness,
    # never a numerical one. `terms.exact` is set by how the value was produced —
    # a node event publishes stored data and performs no arithmetic — so a
    # bitwise-equal `(hi, lo)` pair between two such candidates IS an exact tie.
    #
    # The previous rule read `radius == 0` as that certificate, and a radius is a
    # float like any other: near the bottom of the range `eps**2 * |v|` underflows
    # to zero for interior lanes whose value pair also collapsed, so two candidates
    # that are NOT tied presented as certifiably tied and `_exact_compare` — which
    # returns the correct strict sign on exactly that input — was never consulted
    # (round-11 audit F2/RT11). "The radius came out zero" is a statement about
    # the arithmetic's dynamic range, not about the candidate's value.
    #
    # The consequence is the invariant the whole selection now rests on: the
    # approximate layer may RESOLVE an ordering, never CERTIFY an equality. It
    # resolves soundly because `value_hi` is correctly rounded and rounding to
    # nearest is monotone, so `value_hi_a > value_hi_b` implies the exact values
    # are strictly ordered the same way. Everything it cannot separate — every
    # tie, however certain it looks — is decided by `_exact_compare`.
    lead_exact = jnp.take_along_axis(terms.exact, lead[:, None], axis=1)
    certified_tie = dd_tied & terms.exact & lead_exact

    def unresolved(state: _ResolveState) -> BoolND:
        return jnp.any(state.remaining)

    def resolve_one(state: _ResolveState) -> _ResolveState:
        active = jnp.any(state.remaining, axis=1)
        index = jnp.argmax(state.remaining, axis=1)
        hot = _one_hot(index, n_segment) & state.remaining
        cols = gather(index)
        sign = _exact_compare(cols_a=cols, cols_b=state.lead_cols, q=q)
        # A NaN sign (non-finite candidate arithmetic) is neither greater nor
        # equal, so it never takes the lead; `poisoned` NaNs the query anyway.
        greater = active & (sign > 0)
        equal = active & (sign == 0)
        return _ResolveState(
            lead_cols=jnp.where(greater[:, None], cols, state.lead_cols),
            # A promotion invalidates every earlier tie: those candidates tied
            # with a leader now known to be strictly smaller.
            tied=jnp.where(
                greater[:, None],
                hot,
                state.tied | (hot & equal[:, None]),
            ),
            remaining=state.remaining & ~hot,
        )

    final = jax.lax.while_loop(
        unresolved,
        resolve_one,
        _ResolveState(
            lead_cols=gather(lead),
            tied=certified_tie | lead_hot,
            remaining=contends & ~certified_tie & ~lead_hot,
        ),
    )
    return final.tied


class _CandidateTerms(NamedTuple):
    """Per-(query, segment) candidate quantities, each `(n_query, n_block)`.

    `value_hi/value_lo` is the certified double-double candidate value (stored
    floats with `value_lo == 0` at node events, compensated interpolation in the
    interior) and `radius` its certified residual rounding radius (zero at node
    events, O(eps^2) interior). `exact` records STRUCTURALLY — by how the value
    was produced, not by inspecting it — whether the pair is the exact candidate
    value; see `_exactly_maximal`. `policy`/`marginal` are the candidate's outputs
    at the query, `slope` its value-slope, and `right_available` whether it
    extends strictly right of the query — the right-continuous tie-break keys.
    """

    brackets: BoolND
    value_hi: FloatND
    value_lo: FloatND
    radius: FloatND
    exact: BoolND
    policy: FloatND
    marginal: FloatND
    slope: FloatND
    right_available: BoolND


def _candidate_terms(*, block: FloatND, live: BoolND, flat: Float1D) -> _CandidateTerms:
    """Evaluate one block of segments against every query, with certification.

    `block` is a `(n_block, 8)` slice of the stacked link endpoint columns and
    `live` its `(n_block,)` live-flag slice. Both the dense reduction (which
    passes the whole link set as one block) and the blocked scan call this, so
    every per-lane quantity is computed by the very same expressions and the
    two paths select from identical candidates.

    Node events (query equal to a stored endpoint, or a zero-width segment) are
    published as the stored floats with zero certified radius — requirement (1)
    of the selection architecture. Interior lanes are evaluated compensated in
    double-double with an eps^2-level certified radius — requirement (2). A
    zero-width segment carrying a value jump publishes its higher end (its
    lower end still competes through that endpoint's zero-width self-bracket),
    matching the host oracle's vertical-edge rule.
    """
    left_grid, right_grid = block[:, 0][None, :], block[:, 1][None, :]
    left_value, right_value = block[:, 2][None, :], block[:, 3][None, :]
    left_policy, right_policy = block[:, 4][None, :], block[:, 5][None, :]
    left_marginal, right_marginal = block[:, 6][None, :], block[:, 7][None, :]
    q = flat[:, None]

    lower = jnp.minimum(left_grid, right_grid)
    upper = jnp.maximum(left_grid, right_grid)
    brackets = live[None, :] & (q >= lower) & (q <= upper)

    # Node events: the candidate value IS stored data; no arithmetic, radius 0.
    node, use_right = _node_selection(
        q=q,
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
    )

    # Interior: compensated interpolation `left + (t/w)*d` in double-double.
    # TwoDiff makes t, w, d exact dd representations; division and product are
    # dd-accurate, so (hi, lo) carries the interpolant to O(eps^2) relative.
    #
    # SCALE AFTER DIFFERENCING, AND CARRY THE EXPONENT AS AN INTEGER.
    #
    # Dekker's TwoProd splits an operand by multiplying it by `2**s + 1`, so it
    # needs `|a| < 2**(emax - s)` — `2**115` in float32, `2**996` in float64 —
    # and past that the interior pair goes non-finite (round-8 audit F2). Rounds
    # 9 and 10 bought that headroom by scaling the OPERANDS first: round 9 with
    # one exponent for the whole array, round 10 with one per candidate. Both are
    # lossy for the same reason — scaling `q` and `x0` before subtracting them
    # can map two distinct represented grid points onto the SAME float, and then
    # `t = q - x0` is zero and no downstream exactness can recover it. Round 10's
    # own witness: `x0 = 1`, `q = nextafter(1)`, `x1 = 2**126`, where the
    # candidate exponent 127 makes both `x0` and `q` the same subnormal and a
    # strictly lower constant competitor takes the value, policy and marginal
    # (round-10 audit F2).
    #
    # So the differences are formed FIRST, on the raw stored operands, where
    # `_two_diff` is exact and subtraction of finite floats cannot overflow.
    # Only then is each difference scaled by its OWN binade, and the scale is
    # kept as an integer exponent rather than being applied to the operands:
    #
    #     r = t/w = (t * 2**-et) / (w * 2**-ew) * 2**(et - ew)
    #     p = r*d = [(t*2**-et)/(w*2**-ew) * (d*2**-ed)] * 2**(et - ew + ed)
    #
    # Every significand handed to Dekker is O(1), so splitting cannot overflow at
    # any model scale, and the exponent is applied ONCE, exactly, at the end.
    #
    # No value scale is needed to keep the intermediates BOUNDED. A bracketing
    # candidate has `q` in `[x0, x1]`, hence `|t| <= |w|` and `r` in `[0, 1]`, so
    # `|p| <= |d|` and `v = v0 + p` is bounded by the endpoint values themselves.
    # The unbounded intermediates only ever came from mixing units — a value
    # times a grid difference — which this form never constructs.
    #
    # Boundedness is not the whole requirement, though, and reading it as if it
    # were is what left the value axis unscaled through round 11 (round-11 audit
    # F2). The value axis needs a scale for the OPPOSITE reason to the grid axis:
    # not because its intermediates can grow, but because they can vanish. See
    # the value lift below.
    zero_width = right_grid == left_grid
    split = _dekker_split_factor(block.dtype)

    # ... and the ONE thing a shift of the OPERANDS can never deliver is safety
    # in both directions at once, which is the round-13 finding. A difference of
    # finite floats can UNDERFLOW — near the bottom of the range the gap between
    # two distinct normals is subnormal and XLA flushes it, so `q - x0` came back
    # exactly `0.0` and the interpolant collapsed onto its left value (round-9
    # audit MT6). Rounds 9 to 13 answered that by LIFTING, never lowering, and
    # asserted that subtraction of finite floats cannot overflow. It can: for
    # `H = 2**127` in float32 the operands already exceed any target, the lift is
    # zero, and `H - (-H)` is `+inf`. A finite exact envelope was published as
    # three NaNs (round-13 audit F2, MT10: 288 of 288 generated cells).
    #
    # No choice of shift closes both ends, because the spread the operands can
    # span is the whole format and no single frame holds it. So the difference is
    # not formed in the working dtype at all: `_framed_difference` divides each
    # PAIR by its own `2**max(e_a, e_b)` — exact, by `ldexp` on the `frexp`
    # mantissa — and returns the pair's exponent alongside the double-double
    # head and tail. Both operands are then in `(-1, 1]`, where `_two_diff`
    # cannot overflow and cannot lose its residual, at any model scale.
    #
    # Each difference carries its OWN frame, so the grid axis needs no common
    # shift for `q` and the two nodes: `r = t/w` recombines them through the
    # integer exponents, where the round-10 hazard of merging two distinct grid
    # points onto one float cannot arise because no operand is ever lowered
    # relative to another it is compared against.
    th, tl, t_frame = _framed_difference(q, left_grid)
    wh, wl, w_frame = _framed_difference(right_grid, left_grid)
    wh = jnp.where(zero_width, jnp.ones_like(wh), wh)
    wl = jnp.where(zero_width, jnp.zeros_like(wl), wl)
    w_frame = jnp.where(zero_width, jnp.zeros_like(w_frame), w_frame)

    # The value axis takes the same treatment, and for BOTH reasons. Downward:
    # with `v0 = tiny` and `v1 = nextafter(tiny)` the endpoint gap `d` used to
    # flush and a strictly lower branch took the policy and the marginal
    # (round-11 audit F2, MT8). Upward: with `v0 = -H` and `v1 = H` the gap `2H`
    # is not representable at all, though every endpoint and the interpolated
    # value are. In the framed form `d` is `(1.0, 0.0)` at exponent `maxexp`, and
    # nothing overflows.
    #
    # Only the HEAD is needed here: the interpolated value itself is built by
    # `_framed_affine`, which forms this difference again for the channel it is
    # evaluating. What remains is the cross-candidate SLOPE screen key.
    dh, _, d_frame = _framed_difference(right_value, left_value)

    # Renormalize each head into `[0.5, 1)` before Dekker sees it, and fold the
    # shift into the integer exponent. This is the round-8 requirement — the
    # split constant `2**s + 1` needs `|a| < 2**(emax - s)` — now applied to a
    # quantity that is already O(1), so it is conditioning rather than rescue.
    t_exp = _binade_exponent(jnp.abs(th)) + t_frame
    w_exp = _binade_exponent(jnp.abs(wh)) + w_frame
    d_exp = _binade_exponent(jnp.abs(dh)) + d_frame
    t_shift = t_frame - t_exp
    w_shift = w_frame - w_exp
    d_shift = d_frame - d_exp

    rh, rl = _dd_div(
        jnp.ldexp(th, t_shift),
        jnp.ldexp(tl, t_shift),
        jnp.ldexp(wh, w_shift),
        jnp.ldexp(wl, w_shift),
        split,
    )
    # EVERY published affine channel goes through the SAME evaluator. The ratio
    # is handed over as a significand pair plus a separate integer exponent and
    # is never materialized as a float.
    #
    # Round 14 gave this treatment to the value alone and left `policy` and
    # `marginal` on `left + fraction*(right - left)` in the working dtype, so a
    # certified winner published uncertified outputs: a `fraction` that flushed
    # to zero before multiplication, and a `right - left` that overflowed on
    # opposite-signed top-binade endpoints (round-14 audit F2, RT16 24/24 and
    # MT11 66/66). Repairing the channel a witness names and not its siblings is
    # exactly how the previous nine rounds each ended.
    #
    # For a BRACKETING candidate `|t| <= |w|`, so `r` is in `[0, 1]` and each
    # result is bounded by its own endpoints: scaling back out of the frame is
    # exact and lands inside the binade the endpoints came from, so every
    # channel is finite whenever its endpoints are, and rounded exactly ONCE.
    #
    # `value_lo` and the radius are scaled back by the same shift and may flush
    # to zero there. That costs resolution, never correctness: a flushed residual
    # says the remainder sits below the representable grid, and under the
    # structural-exactness rule in `_exactly_maximal` a `(hi, lo)` tie is never a
    # certificate — it routes to `_exact_compare` — while a strict difference in
    # a correctly rounded `value_hi` certifies a strict difference in the exact
    # values, because rounding to nearest is monotone.
    ratio_exp = t_exp - w_exp
    affine = functools.partial(
        _framed_affine, ratio_hi=rh, ratio_lo=rl, ratio_exp=ratio_exp, split=split
    )
    value_affine = affine(left=left_value, right=right_value)
    policy_affine = affine(left=left_policy, right=right_policy)
    marginal_affine = affine(left=left_marginal, right=right_marginal)

    vh, vl = value_affine.hi, value_affine.lo

    eps = jnp.finfo(block.dtype).eps
    interior_radius = jnp.ldexp(
        _INTERIOR_RADIUS_ULPS2
        * eps
        * eps
        * (jnp.abs(value_affine.framed_left) + jnp.abs(value_affine.framed_product)),
        value_affine.frame,
    )

    zero = jnp.zeros_like(vh)
    return _CandidateTerms(
        brackets=brackets,
        value_hi=jnp.where(node, jnp.where(use_right, right_value, left_value), vh),
        value_lo=jnp.where(node, zero, vl),
        radius=jnp.where(node, zero, interior_radius),
        # A node event's value IS stored data: no arithmetic was performed, so
        # the pair is exact by construction. Nothing else here is.
        exact=node,
        policy=jnp.where(
            node,
            jnp.where(use_right, right_policy, left_policy),
            policy_affine.hi,
        ),
        marginal=jnp.where(
            node,
            jnp.where(use_right, right_marginal, left_marginal),
            marginal_affine.hi,
        ),
        # In ORIGINAL units, not the per-candidate binade: this is a screen key
        # compared ACROSS candidates, so every candidate must express it in the
        # same units.
        #
        # Built from the framed differences rather than from raw subtractions.
        # "One subtraction over another carries no overflow risk of its own" was
        # the same false hinge as in `_exact_difference`: with `v0 = -H`,
        # `v1 = H` the numerator alone is `+inf`, and against an equally
        # overflowing width it becomes NaN — a key the round-13 sentinel repair
        # in `_right_continuous_winner` then has to demote, losing the ordering
        # rather than getting it right. Here the mantissa ratio is O(1) and the
        # magnitude rides in the integer exponent, so the key is finite whenever
        # the mathematical slope is, and infinite only when it genuinely is.
        slope=jnp.ldexp(jnp.ldexp(dh, d_shift) / jnp.ldexp(wh, w_shift), d_exp - w_exp),
        right_available=q < upper,
    )


def _right_continuous_winner(
    *,
    tied: BoolND,
    terms: _CandidateTerms,
    gather: Callable[[jax.Array], FloatND],
) -> tuple[BoolND, jax.Array]:
    """Per-query right-continuous winner among the EXACTLY value-tied candidates.

    Implements "prefer a segment extending strictly right of the query, then
    the larger value slope, then the earliest candidate" — and implements the
    slope half EXACTLY.

    The rounded key `fl((v1-v0)/(x1-x0))` is used only as a screen. A slope gap
    wider than a few ULP is certified by it; anything closer is settled by
    `_exact_slope_compare`. Ordering on the rounded key alone is unsound: two
    strictly ordered exact slopes can collapse onto one float key, and `argmax`
    then resolves by candidate order, so permuting the branches flips the
    published policy and marginal (round-7 audit F2 — 16 of 16 generated
    collision classes were order-dependent).

    Keeping the two keys separate rather than folding them into one scalar is
    still required: an `arctan(slope)/pi + right_available` rank loses slope
    bits for near-equal small slopes in float32 (round-4 audit F2).

    Returns the per-query right-extension flag (so the blocked scan can
    reconcile that priority across blocks) and ONE winner index.
    """
    n_segment = tied.shape[1]
    eligible = tied & terms.right_available
    any_eligible = jnp.any(eligible, axis=1, keepdims=True)
    compete = jnp.where(any_eligible, eligible, tied)

    # `-inf` is the MASK sentinel here, and it is also a REACHABLE rounded key:
    # `slope` is `fl(v1-v0) / fl(x1-x0)` in original units, so a large value gap
    # over a small grid width overflows to `-inf` with every stored input still
    # finite normal. A plain `argmax` over `key` then cannot tell the only
    # competitor from the candidates the mask excluded, and returns index 0 —
    # a candidate that is not competing at all, which the loop below seeds as
    # the lead and never revisits. The published policy and marginal are then
    # whichever branch happens to sit first, so a pure branch permutation flips
    # them while the value stays right: the round-7 F2 signature, reached
    # through the sentinel rather than through a shared float key.
    #
    # Selecting the first COMPETING candidate that attains the maximum keeps the
    # documented earliest-on-tie rule and makes the sentinel unreachable as an
    # answer. NaN keys are folded onto the sentinel for the same reason — they
    # must not be allowed to win a comparison the screen cannot make — and
    # `contends` below then hands every one of them to the exact predicate.
    key = jnp.where(compete, terms.slope, -jnp.inf)
    rankable = jnp.where(jnp.isnan(key), -jnp.inf, key)
    lead = jnp.argmax(
        compete & (rankable == jnp.max(rankable, axis=1, keepdims=True)), axis=1
    )
    lead_slope = jnp.take_along_axis(terms.slope, lead[:, None], axis=1)
    eps = jnp.finfo(terms.slope.dtype).eps
    reach = _SLOPE_SCREEN_ULPS * eps * (jnp.abs(terms.slope) + jnp.abs(lead_slope))
    # A screen that cannot discriminate must DEFER to the exact comparison, not
    # decide. `slope` is a rounded key in ORIGINAL units — deliberately, since
    # it is compared across candidates — so it overflows to an infinity when a
    # large value gap meets a small grid width, with every stored input still
    # finite normal. `reach` is then infinite, `lead_slope - reach` is NaN,
    # every `>=` is False, and `contends` would be EMPTY: the exact loop never
    # runs and the winner is whatever `argmax` over the rounded key returned,
    # i.e. candidate order. That is precisely the round-7 F2 failure — level
    # tied either way, only the published policy and marginal wrong — and it is
    # reachable at `|Δvalue| / Δgrid > max_float` (probe: float32 values at
    # `2**100` over a width of `2**-60` flip policy 1.0 <-> 2.0 under a branch
    # permutation). Where the screen's own arithmetic is not finite it admits
    # every competitor and lets `_exact_slope_compare` settle the order.
    screen_usable = jnp.isfinite(reach) & jnp.isfinite(lead_slope)
    contends = compete & (~screen_usable | (terms.slope >= lead_slope - reach))
    lead_hot = _one_hot(lead, n_segment)

    def unresolved(state: _SlopeState) -> BoolND:
        return jnp.any(state.remaining)

    def resolve_one(state: _SlopeState) -> _SlopeState:
        active = jnp.any(state.remaining, axis=1)
        index = jnp.argmax(state.remaining, axis=1)
        hot = _one_hot(index, n_segment) & state.remaining
        cols = gather(index)
        sign = _exact_slope_compare(cols_a=cols, cols_b=state.lead_cols)
        # Candidates are visited in increasing index and a STRICTLY greater
        # exact slope takes the lead, so the winner is the earliest index
        # attaining the exact maximum — the documented earliest-on-tie rule,
        # conditioned on exact rather than rounded equality.
        #
        # The `sign == 0` clause is what makes that unconditional. The loop is
        # seeded from `argmax` over the ROUNDED key, and that key is a doubly
        # rounded quantity: `fl(fl(v1-v0) / fl(x1-x0))`, whereas the exact ratio
        # uses the error-free `(hi, lo)` differences. When a stored difference
        # is not representable the two disagree, so two candidates with EQUAL
        # exact slopes can carry different rounded keys and `argmax` can seed
        # the lead at the later of them. Without this clause the last-resort
        # fallback would then publish the later candidate — the rounded key
        # deciding an order it has no standing to decide, which is the very
        # defect this function exists to remove. Taking the lead on an exact tie
        # from any strictly earlier index closes that without relying on the
        # rounded key being monotone.
        earlier_on_exact_tie = (sign == 0) & (index < state.lead_index)
        greater = active & ((sign > 0) | earlier_on_exact_tie)
        return _SlopeState(
            lead_cols=jnp.where(greater[:, None], cols, state.lead_cols),
            lead_index=jnp.where(greater, index, state.lead_index),
            remaining=state.remaining & ~hot,
        )

    final = jax.lax.while_loop(
        unresolved,
        resolve_one,
        _SlopeState(
            lead_cols=gather(lead),
            lead_index=lead,
            remaining=contends & ~lead_hot,
        ),
    )
    return any_eligible[:, 0], final.lead_index


class _BlockedCarry(NamedTuple):
    """Running winner of the blocked scan, one entry per query.

    The full selection rule is one lexicographic maximum over the candidate key
    `(value_hi, value_lo, right_available, slope, earliest)`, so a single scan
    can carry the current winner's key components together with its gathered
    value/policy/marginal. Every candidate is evaluated exactly once, inside
    one compiled scan body; no quantity is ever recomputed in a second program
    and compared for equality (XLA is free to fuse each lowering differently at
    the bit level, so a cross-program exact-equality rendezvous would be
    unsound — the round-5 rewrite's first blocked draft failed exactly there).
    """

    any_bracket: BoolND
    poisoned: BoolND
    has_winner: BoolND
    hi: FloatND
    lo: FloatND
    radius: FloatND
    # Whether the carried winner's value pair is exact by construction, carried
    # for the same reason the radius is: `_combine_blocked` runs the shared
    # selection rule, and that rule may only skip the exact comparison on a
    # structural certificate. A winner interpolated in a block's interior is not
    # exact, and the fold must not treat it as such just because two blocks
    # produced the same rounded pair.
    exact: BoolND
    right_extending: BoolND
    slope: FloatND
    value: FloatND
    policy: FloatND
    marginal: FloatND
    cols: FloatND


def _combine_blocked(
    *, carry: _BlockedCarry, block: _BlockedCarry, q: FloatND
) -> _BlockedCarry:
    """Fold a block's winner into the running winner, under the shared rule.

    The cross-block comparison is the SAME certified-screen-then-exact
    resolution the within-block reduction runs, applied to a two-candidate set.
    Routing both through one implementation is what makes dense and blocked
    provably agree: a bespoke lexicographic update here would re-introduce
    exactly the low-word ordering the exact predicate exists to prevent, and
    only on the block boundary, where no within-block test would see it.
    """
    stacked = _CandidateTerms(
        brackets=jnp.stack([carry.has_winner, block.has_winner], axis=1),
        value_hi=jnp.stack([carry.hi, block.hi], axis=1),
        value_lo=jnp.stack([carry.lo, block.lo], axis=1),
        radius=jnp.stack([carry.radius, block.radius], axis=1),
        exact=jnp.stack([carry.exact, block.exact], axis=1),
        policy=jnp.stack([carry.policy, block.policy], axis=1),
        marginal=jnp.stack([carry.marginal, block.marginal], axis=1),
        slope=jnp.stack([carry.slope, block.slope], axis=1),
        right_available=jnp.stack(
            [carry.right_extending, block.right_extending], axis=1
        ),
    )
    pair_cols = jnp.stack([carry.cols, block.cols], axis=1)
    tied = _exactly_maximal(
        terms=stacked,
        gather=lambda index: jnp.take_along_axis(
            pair_cols, index[:, None, None], axis=1
        )[:, 0],
        q=q,
    )
    _, winner = _right_continuous_winner(
        tied=tied,
        terms=stacked,
        gather=lambda index: jnp.take_along_axis(
            pair_cols, index[:, None, None], axis=1
        )[:, 0],
    )
    # Column 0 is the carry, so keeping the earlier column on an exact tie is
    # precisely the dense reduction's earliest-candidate rule.
    take = winner == 1
    return _BlockedCarry(
        any_bracket=carry.any_bracket | block.any_bracket,
        poisoned=carry.poisoned | block.poisoned,
        has_winner=carry.has_winner | block.has_winner,
        hi=jnp.where(take, block.hi, carry.hi),
        lo=jnp.where(take, block.lo, carry.lo),
        radius=jnp.where(take, block.radius, carry.radius),
        exact=jnp.where(take, block.exact, carry.exact),
        right_extending=jnp.where(take, block.right_extending, carry.right_extending),
        slope=jnp.where(take, block.slope, carry.slope),
        value=jnp.where(take, block.value, carry.value),
        policy=jnp.where(take, block.policy, carry.policy),
        marginal=jnp.where(take, block.marginal, carry.marginal),
        cols=jnp.where(take[:, None], block.cols, carry.cols),
    )


def _envelope_at_query_blocked(
    *, columns: FloatND, live: BoolND, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Single-scan blocked equivalent of the dense `(n_query, n_segment)` path.

    Evaluates every candidate through the shared `_candidate_terms` and reduces
    with the same lexicographic rule as the dense path — certified `(hi, lo)`
    value pairs first, then (only on an exact tie) right-extension and value
    slope. Within a block the winner is found exactly as in the dense reduction
    (`_right_continuous_winner` restricted to the block's exactly-tied candidates)
    and its value/policy/marginal are gathered from the SAME within-block
    index. Across blocks the carried winner is updated by strict lexicographic
    comparison of the carried keys, which keeps the earliest winner and so
    matches the dense `argmax`. Lexicographic maximum is associative, so the
    scan order cannot change the result.

    The links are padded to a multiple of `block_size` with dead segments
    (which never bracket); the scan peaks at `(n_query, block_size)` working
    memory. A bracketed query with a non-finite candidate value pair (infinite
    endpoint arithmetic) is `poisoned` and fails loud with NaN in all three
    outputs, matching the dense path's empty-exact-tie rule.
    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    n_segment = live.shape[0]
    pad = (-n_segment) % block_size
    if pad:
        columns = jnp.concatenate(
            [columns, jnp.zeros((pad, columns.shape[1]), dtype=columns.dtype)]
        )
        live = jnp.concatenate([live, jnp.zeros((pad,), dtype=bool)])
    blocks = columns.reshape(-1, block_size, columns.shape[1])
    live_blocks = live.reshape(-1, block_size)
    dtype = columns.dtype

    def step(
        carry: _BlockedCarry,
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[_BlockedCarry, None]:
        block, block_live = block_and_live
        t = _candidate_terms(block=block, live=block_live, flat=flat)
        # Within-block winner: identical construction to the dense reduction,
        # through the very same certified-screen-then-exact resolution.
        tied = _exactly_maximal(terms=t, gather=lambda index: block[index], q=flat)
        block_ra, winner = _right_continuous_winner(
            tied=tied, terms=t, gather=lambda index: block[index]
        )
        column = winner[:, None]
        block_has = jnp.any(t.brackets, axis=1)
        block_carry = _BlockedCarry(
            any_bracket=block_has,
            poisoned=jnp.any(
                t.brackets & (jnp.isnan(t.value_hi) | jnp.isnan(t.value_lo)), axis=1
            ),
            has_winner=block_has,
            hi=jnp.take_along_axis(t.value_hi, column, axis=1)[:, 0],
            lo=jnp.take_along_axis(t.value_lo, column, axis=1)[:, 0],
            radius=jnp.take_along_axis(t.radius, column, axis=1)[:, 0],
            exact=jnp.take_along_axis(t.exact, column, axis=1)[:, 0],
            right_extending=block_ra,
            slope=jnp.take_along_axis(t.slope, column, axis=1)[:, 0],
            value=jnp.take_along_axis(t.value_hi, column, axis=1)[:, 0],
            policy=jnp.take_along_axis(t.policy, column, axis=1)[:, 0],
            marginal=jnp.take_along_axis(t.marginal, column, axis=1)[:, 0],
            cols=block[winner],
        )
        return _combine_blocked(carry=carry, block=block_carry, q=flat), None

    final, _ = jax.lax.scan(
        step,
        _BlockedCarry(
            any_bracket=jnp.zeros((n_query,), dtype=bool),
            poisoned=jnp.zeros((n_query,), dtype=bool),
            has_winner=jnp.zeros((n_query,), dtype=bool),
            hi=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            lo=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            radius=jnp.zeros((n_query,), dtype=dtype),
            # The seed carries no winner (`has_winner` is False), so it never
            # reaches a comparison; claiming exactness for it would be a lie the
            # fold could only ever act on by mistake.
            exact=jnp.zeros((n_query,), dtype=bool),
            right_extending=jnp.zeros((n_query,), dtype=bool),
            slope=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            value=jnp.full((n_query,), jnp.nan, dtype=dtype),
            policy=jnp.full((n_query,), jnp.nan, dtype=dtype),
            marginal=jnp.full((n_query,), jnp.nan, dtype=dtype),
            cols=jnp.zeros((n_query, columns.shape[1]), dtype=dtype),
        ),
        (blocks, live_blocks),
    )
    ok = final.any_bracket & ~final.poisoned
    env_value = jnp.where(ok, final.value, jnp.nan)
    env_policy = jnp.where(ok, final.policy, jnp.nan)
    env_marginal = jnp.where(ok, final.marginal, jnp.nan)
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )


def _envelope_at_query_407(
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
        winning segment's marginal at each query, each shaped like `x_query`
        and all gathered from the same winning segment. A query no live segment
        brackets yields NaN in all three.
    """
    # NO global rescaling happens here, deliberately. Round 9 normalized the
    # whole problem at this point to keep Dekker's splitting in range, and that
    # map is NOT INJECTIVE over a mixed-scale array: one exponent was chosen
    # from the largest candidate anywhere in the input, including candidates
    # that cannot bracket the query, so a remote segment at `2**126` collapsed
    # two adjacent local floats into the same subnormal BEFORE the certified
    # comparator ever saw them. Exact arithmetic downstream cannot rebuild bits
    # that were discarded upstream, and the published value, policy and marginal
    # all changed in response to a segment that is mathematically irrelevant
    # (round-9 audit F2).
    #
    # Exponent safety is therefore established PER CANDIDATE, inside
    # `_candidate_terms`, where the operands of each error-free transform are
    # known and nothing another candidate does can perturb them. Topology and
    # node events stay on the ORIGINAL represented coordinates.
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
    # exact-tie break, so multi-point chains and their interpolation are
    # unchanged; a self-bracket wins only where nothing brackets the query from
    # the right.
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
    columns = jnp.stack(
        [
            links.left_grid,
            links.right_grid,
            links.left_value,
            links.right_value,
            links.left_policy,
            links.right_policy,
            links.left_marginal,
            links.right_marginal,
        ],
        axis=1,
    )

    query = jnp.asarray(x_query)
    n_segment = links.live.shape[0]
    if 0 < segment_block_size < n_segment:
        return _envelope_at_query_blocked(
            columns=columns, live=links.live, query=query, block_size=segment_block_size
        )

    flat = query.reshape(-1)
    terms = _candidate_terms(block=columns, live=links.live, flat=flat)

    # Certified screen followed by exact resolution: the winner set is the
    # EXACTLY maximal candidates, so right-continuity is applied to genuine
    # ties and never bypassed by a low-word residue (requirement 3).
    exact_tie = _exactly_maximal(
        terms=terms, gather=lambda index: columns[index], q=flat
    )

    # Right-continuous break among the exactly-tied candidates only, then ONE
    # winner index per query; value, policy, and marginal are gathered from that
    # same index (requirements 4 and 5).
    _, winner = _right_continuous_winner(
        tied=exact_tie, terms=terms, gather=lambda index: columns[index]
    )
    best = winner[:, None]
    any_bracket = jnp.any(terms.brackets, axis=1)
    # A bracketed query whose candidates cannot be ranked — a non-finite
    # certified value pair (infinite endpoint arithmetic), which also empties
    # the exact-tie set — fails loud with NaN (requirement 3). The explicit
    # `poisoned` flag keeps the dense and blocked paths on the same rule.
    resolved = jnp.any(exact_tie, axis=1)
    poisoned = jnp.any(
        terms.brackets & (jnp.isnan(terms.value_hi) | jnp.isnan(terms.value_lo)),
        axis=1,
    )
    ok = any_bracket & resolved & ~poisoned
    env_value = jnp.where(
        ok, jnp.take_along_axis(terms.value_hi, best, axis=1)[:, 0], jnp.nan
    )
    env_policy = jnp.where(
        ok, jnp.take_along_axis(terms.policy, best, axis=1)[:, 0], jnp.nan
    )
    env_marginal = jnp.where(
        ok, jnp.take_along_axis(terms.marginal, best, axis=1)[:, 0], jnp.nan
    )
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )
