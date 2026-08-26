"""Exact query-side upper envelope of an EGM candidate correspondence.

The query backend evaluates every live consecutive branch link, plus every live
candidate as a zero-width self-bracket, directly at the requested abscissae.
The certified mode delegates the complete ownership reduction to one opaque
native operation. Stored IEEE operands are decoded as exact dyadics and ordered
lexicographically by exact affine value, right extension, exact slope, and the
stable stored-link index. Only after one winner and one status have been returned
does Python invoke the exact affine reader, exactly once for each of value,
policy, and marginal. Those channels publish together or all become NaN.

This boundary is deliberate. Same-format double-double words cannot encode every
nonzero residual of normal operands, so neither a zero product tail nor two equal
rounded slope words is an equality certificate. The native winner avoids that
representability limit and keeps the integer arithmetic outside the JAX trace.
An array of queries is handled by one custom call; an outer ``vmap`` is lowered
sequentially around that opaque call. ``segment_block_size`` therefore has no
numerical effect in certified mode, giving exact dense/blocked identity.

The ordinary mode remains a dense working-format interpolation and maximum. It
is cheaper, but it carries no exact ownership guarantee.
"""

import functools
from collections.abc import Callable
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.axis_boundaries import (
    ResolvedAxisPartition,
    axis_interval_indices,
    feasibility_region_indices,
)
from _lcm.egm.upper_envelope._exact_affine.ffi import (
    exact_affine_read,
    exact_query_winner,
)
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
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
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
        segment_block_size: How many candidates the ordinary read holds against
            every query at once. `0` lets the reduction pick a window; a positive
            value pins it. The certified path is one native exact reduction whose
            partition is internal, so the hint does not reach it and is
            bit-identical for every value either way.
        arithmetic: Which arithmetic decides ownership.
            - `"certified"` compares the stored affine candidates in native
              fixed-width integer arithmetic, using exact value, right extension,
              exact slope, and stable segment index. It performs three exact reads
              only after the winner is known and publishes all channels together.
            - `"ordinary"` reads each link in the working format and takes the
              largest. It decides every bracketed query — there is no certificate
              to abstain on — and costs roughly an order of magnitude less per
              read. Adequate where candidate values are separated by much more
              than the format's resolution at their own magnitude.
            The choice is made when the function is traced, so `"ordinary"`
            emits none of the error-free transforms rather than masking them.
            Implemented for the dense reduction only.
        feasibility_partition: Shared liquid-axis partition containing the
            feasibility boundaries that candidates and queries must respect.
            Must be supplied together with `feasible_interval_mask`.
        feasible_interval_mask: Whether each interval of
            `feasibility_partition` is feasible. Infeasible candidates are
            removed before links are built, and infeasible queries publish the
            carry contract: `-inf` value, NaN policy, and zero marginal.

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`. A
        query no live segment brackets yields NaN in all three.

    Note:
        Under `arithmetic="certified"` the read is differentiable in forward
        mode only: `jax.jvp`, `jax.jacfwd`, and forward-over-forward all carry
        the exact affine slope, while `jax.grad` and `jax.vjp` raise. The
        certified rule fails closed on a non-finite direction, which makes its
        published tangent a function of the tangent rather than a linear map of
        it, and only a tangent-linear rule can be transposed. Reverse mode over
        a scalar objective therefore needs `arithmetic="ordinary"`, or
        `jax.jacfwd` in place of `jax.grad`.
    """
    if (feasibility_partition is None) != (feasible_interval_mask is None):
        raise ValueError(
            "feasibility_partition and feasible_interval_mask must be supplied "
            "together."
        )

    if feasibility_partition is None:
        candidate_region = jnp.zeros_like(segment_id, dtype=jnp.int32)
    else:
        feasible_interval_mask = cast("BoolND", feasible_interval_mask)
        candidate_interval = axis_interval_indices(
            partition=feasibility_partition,
            values=endog_grid,
        )
        candidate_feasible = feasible_interval_mask[candidate_interval]
        candidate_region = feasibility_region_indices(
            partition=feasibility_partition,
            values=endog_grid,
        )
        endog_grid = jnp.where(candidate_feasible, endog_grid, jnp.nan)
        value = jnp.where(candidate_feasible, value, jnp.nan)
        policy = jnp.where(candidate_feasible, policy, jnp.nan)
        marginal = jnp.where(candidate_feasible, marginal, jnp.nan)

    dead = ~jnp.isfinite(endog_grid) | ~jnp.isfinite(value)
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
        live=(
            ~dead[:-1]
            & ~dead[1:]
            & (segment_id[:-1] == segment_id[1:])
            & (candidate_region[:-1] == candidate_region[1:])
        ),
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
    if arithmetic == "certified":
        # The exact kernel owns the whole segment reduction. A block size is now
        # a partition request with no numerical effect: the native operation
        # streams the same stored segments and returns one winner/status per
        # query, so dense and blocked calls are identical by construction.
        published = _envelope_exact(links=links, query=query).published
        return _mask_infeasible_queries(
            published=published,
            query=query,
            partition=feasibility_partition,
            feasible_interval_mask=feasible_interval_mask,
        )

    published = _envelope_blocked_ordinary(
        links=links,
        query=query,
        block_size=_ordinary_block_size(
            requested=segment_block_size, n_segment=n_segment
        ),
    ).published
    unreadable = _subnormal_operand_present(
        row=(endog_grid, value, policy, marginal), query=query
    ) | _derived_subnormal_possible(links=links, query=query)
    readable_value, readable_policy, readable_marginal = (
        jnp.where(unreadable, jnp.nan, channel) for channel in published
    )
    return _mask_infeasible_queries(
        published=(readable_value, readable_policy, readable_marginal),
        query=query,
        partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
    )


def _mask_infeasible_queries(
    *,
    published: tuple[FloatND, FloatND, FloatND],
    query: FloatND,
    partition: ResolvedAxisPartition | None,
    feasible_interval_mask: BoolND | None,
) -> tuple[FloatND, FloatND, FloatND]:
    """Apply the value, policy, and carry contracts to infeasible query rows."""
    if partition is None or feasible_interval_mask is None:
        return published

    interval = axis_interval_indices(partition=partition, values=query)
    feasible = feasible_interval_mask[interval]
    value, policy, marginal = published
    return (
        jnp.where(feasible, value, -jnp.inf),
        jnp.where(feasible, policy, jnp.nan),
        jnp.where(feasible, marginal, 0.0),
    )


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

    This predicate makes the unsupported read explicit. Supporting it requires
    arithmetic that reads operands through their significands and exponents
    rather than as working-dtype floats.
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


class _EnvelopeReduction(NamedTuple):
    """The envelope at every query."""

    published: tuple[FloatND, FloatND, FloatND]
    """Value, policy, and marginal of the winning candidate at each query."""


def _envelope_exact(*, links: _SegmentLinks, query: FloatND) -> _EnvelopeReduction:
    """Resolve one exact winner and read only its three published channels.

    The native winner operation compares exact stored affine values under the
    documented total order: value, right extension, exact slope, then stable
    segment index. Its status is coupled to the three exact affine reads below;
    a query either publishes all channels from one winner or NaN in all three.
    """
    if links.left_grid.shape[0] == 0:
        empty = jnp.full_like(query, jnp.nan)
        return _EnvelopeReduction(published=(empty, empty, empty))

    winner, winner_status = exact_query_winner(
        left_grid=links.left_grid,
        right_grid=links.right_grid,
        left_value=links.left_value,
        right_value=links.right_value,
        live=links.live,
        x_query=query,
    )
    flat_query = query.reshape(-1)
    flat_winner = winner.reshape(-1)

    def _take(column: FloatND) -> FloatND:
        return jnp.take(column, flat_winner, axis=0)

    stored_left_grid = _take(links.left_grid)
    stored_right_grid = _take(links.right_grid)
    descending = stored_left_grid > stored_right_grid
    zero_width = stored_left_grid == stored_right_grid
    left_grid = jnp.where(descending, stored_right_grid, stored_left_grid)
    right_grid = jnp.where(descending, stored_left_grid, stored_right_grid)
    # The exact reader intentionally retains its positive-width public contract.
    # A zero-width winner has no affine line; the query reduction defined it as
    # the stored left endpoint. Canonicalize that selected point to a flat unit
    # line and read it at zero, so all three channels still pass through exactly
    # one winner-only native reader without broadening the reader's API.
    read_x0 = jnp.where(zero_width, jnp.zeros_like(left_grid), left_grid)
    read_x1 = jnp.where(zero_width, jnp.ones_like(right_grid), right_grid)
    read_query = jnp.where(zero_width, jnp.zeros_like(flat_query), flat_query)

    def _read(*, left: FloatND, right: FloatND) -> tuple[FloatND, IntND]:
        stored_left = _take(left)
        stored_right = _take(right)
        lower_value = jnp.where(descending, stored_right, stored_left)
        upper_value = jnp.where(descending, stored_left, stored_right)
        upper_value = jnp.where(zero_width, lower_value, upper_value)
        return exact_affine_read(
            x0=read_x0,
            x1=read_x1,
            v0=lower_value,
            v1=upper_value,
            x_query=read_query,
        )

    value, value_status = _read(left=links.left_value, right=links.right_value)
    policy, policy_status = _read(left=links.left_policy, right=links.right_policy)
    marginal, marginal_status = _read(
        left=links.left_marginal, right=links.right_marginal
    )
    coupled_status = (
        winner_status.reshape(-1) | value_status | policy_status | marginal_status
    )
    decided = coupled_status == 0

    def _published(channel: FloatND) -> FloatND:
        return jnp.where(decided, channel, jnp.nan).reshape(query.shape)

    return _EnvelopeReduction(
        published=(_published(value), _published(policy), _published(marginal))
    )


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


def _block_query_terms(
    *,
    block: FloatND,
    live: BoolND,
    flat: Float1D,
    arithmetic: ComparisonArithmetic = "certified",
) -> _BlockTerms:
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
    read = functools.partial(
        _along_link,
        query=q,
        left_grid=spread_left_grid,
        right_grid=spread_right_grid,
        arithmetic=arithmetic,
    )
    value_interp = read(left=left_value[None, :], right=right_value[None, :])
    policy_interp = read(left=left_policy[None, :], right=right_policy[None, :])
    marginal_interp = read(left=left_marginal[None, :], right=right_marginal[None, :])
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


# Candidates the ordinary read holds against one query when nothing is asked for.
# A window in candidates, not a working-set budget in bytes: the reduction sees
# its own query count but not how many of these reductions the caller is running
# side by side, and under a vmapped solve that outer width multiplies every array
# here. A budget computed from what is visible would therefore be spent many
# times over, while a fixed window bounds the working set per query row whatever
# the caller wraps around it.
_ORDINARY_CANDIDATE_WINDOW = 1024


def _ordinary_block_size(*, requested: int, n_segment: int) -> int:
    """Resolve the candidate window the ordinary read streams in.

    Args:
        requested: The caller's `segment_block_size`; `0` asks for the default.
        n_segment: Candidate links available, the largest useful window.

    Returns:
        A positive window no larger than the candidate count.

    """
    window = requested if requested > 0 else _ORDINARY_CANDIDATE_WINDOW
    return max(1, min(window, n_segment))


def _link_blocks(*, links: _SegmentLinks, block_size: int) -> tuple[FloatND, BoolND]:
    """Pad the links to a whole number of blocks and stack them for a scan.

    The padding is dead segments, which never bracket a query, so a padded block
    contributes nothing a live one would not.

    Args:
        links: The candidate links.
        block_size: Candidates per block.

    Returns:
        Tuple of the `(n_block, block_size, 8)` endpoint columns and their
        `(n_block, block_size)` live flags.

    """
    n_segment = links.live.shape[0]
    pad = (-n_segment) % block_size

    def padded(column: FloatND, fill: float) -> FloatND:
        if pad == 0:
            return column
        return jnp.concatenate([column, jnp.full((pad,), fill, dtype=column.dtype)])

    columns = jnp.stack(
        [
            padded(links.left_grid, 0.0),
            padded(links.right_grid, 0.0),
            padded(links.left_value, 0.0),
            padded(links.right_value, 0.0),
            padded(links.left_policy, 0.0),
            padded(links.right_policy, 0.0),
            padded(links.left_marginal, 0.0),
            padded(links.right_marginal, 0.0),
        ],
        axis=1,
    )
    live = (
        links.live
        if pad == 0
        else jnp.concatenate([links.live, jnp.zeros((pad,), dtype=bool)])
    )
    return (
        columns.reshape(-1, block_size, columns.shape[1]),
        live.reshape(-1, block_size),
    )


class _OrdinaryRank(NamedTuple):
    """The fields the ordinary read compares, in the order it compares them.

    The largest read wins outright; among candidates reading exactly level the
    right-continuous tie-break decides, preferring one that extends strictly to
    the right of the query and then the larger value-slope. A candidate that does
    not bracket the query carries `-inf` in every field, so it loses on the first
    comparison and never reaches the later ones.
    """

    value: FloatND
    """Value read along the link at the query."""
    right_available: FloatND
    """Whether the link extends strictly right of the query, as a number."""
    slope_high: FloatND
    """Leading word of the link's value-slope."""
    slope_low: FloatND
    """Trailing word of the same slope, which orders what the leading word ties."""


def _envelope_blocked_ordinary(
    *, links: _SegmentLinks, query: FloatND, block_size: int
) -> _EnvelopeReduction:
    """Evaluate the ordinary envelope by streaming the candidates in blocks.

    Taking a maximum does not require the candidates to be present at once, so
    the scan holds one `(n_query, block_size)` window and a running winner. The
    winner is the largest read, ties broken right-continuously — one ordered
    comparison over the fields of `_OrdinaryRank`, applied first within a block
    and then between the block's best and the standing best. A challenger has to
    beat the standing winner outright to replace it, so an earlier candidate
    keeps a tie however the candidates are partitioned.

    Args:
        links: The candidate links.
        query: Abscissae to evaluate at.
        block_size: Candidates held against every query at once.

    Returns:
        The envelope's value, policy, and marginal at each query, NaN where no
        live link brackets it.

    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    dtype = links.left_grid.dtype
    blocks, live_blocks = _link_blocks(links=links, block_size=block_size)

    def block_best(
        block: FloatND, block_live: BoolND
    ) -> tuple[_OrdinaryRank, tuple[FloatND, ...]]:
        """The block's own winner per query, as its rank and its channels."""
        terms = _block_query_terms(
            block=block, live=block_live, flat=flat, arithmetic="ordinary"
        )
        excluded = jnp.full_like(terms.value, -jnp.inf)
        offered = _OrdinaryRank(
            value=jnp.where(terms.brackets, terms.value, excluded),
            right_available=jnp.where(
                terms.brackets,
                (flat[:, None] < terms.upper).astype(dtype),
                excluded,
            ),
            slope_high=jnp.where(terms.brackets, terms.slope_high, excluded),
            slope_low=jnp.where(terms.brackets, terms.slope_low, excluded),
        )
        still_tied = jnp.ones_like(offered.value, dtype=bool)
        for field in offered:
            best = jnp.max(
                jnp.where(still_tied, field, -jnp.inf), axis=1, keepdims=True
            )
            still_tied = still_tied & (field == best)
        index = jnp.argmax(still_tied, axis=1)[:, None]

        def pick(field: FloatND) -> FloatND:
            return jnp.take_along_axis(field, index, axis=1)[:, 0]

        return (
            _OrdinaryRank(*(pick(field) for field in offered)),
            (pick(terms.value), pick(terms.policy), pick(terms.marginal)),
        )

    def step(
        carry: tuple[FloatND, ...], block_and_live: tuple[FloatND, BoolND]
    ) -> tuple[tuple[FloatND, ...], None]:
        held = _OrdinaryRank(*carry[:4])
        standing = carry[4:]
        offered, channels = block_best(*block_and_live)
        decided = jnp.zeros((n_query,), dtype=bool)
        take = jnp.zeros((n_query,), dtype=bool)
        for challenging, holding in zip(offered, held, strict=True):
            take = take | (~decided & (challenging > holding))
            decided = decided | (challenging != holding)
        return (
            *(
                jnp.where(take, challenging, holding)
                for challenging, holding in zip(offered, held, strict=True)
            ),
            *(
                jnp.where(take, offered_channel, standing_channel)
                for offered_channel, standing_channel in zip(
                    channels, standing, strict=True
                )
            ),
        ), None

    empty_rank = jnp.full((n_query,), -jnp.inf, dtype=dtype)
    empty_channel = jnp.full((n_query,), jnp.nan, dtype=dtype)
    carry, _ = jax.lax.scan(
        step,
        (*(empty_rank,) * 4, *(empty_channel,) * 3),
        (blocks, live_blocks),
    )
    decided = jnp.isfinite(carry[0])
    value, policy, marginal = (
        jnp.where(decided, channel, jnp.nan).reshape(query.shape)
        for channel in carry[4:]
    )
    return _EnvelopeReduction(published=(value, policy, marginal))


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
    blocks, live_blocks = _link_blocks(links=links, block_size=block_size)
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
