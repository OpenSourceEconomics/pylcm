r"""Owner sequence of the affine envelope inside one node cell.

Node-cell boundaries are the sorted live candidate abscissae, so no run node
lies strictly inside a cell: every link that covers the cell covers *all* of it.
Inside a cell the candidates are therefore full lines rather than segments, and
the envelope is the pointwise maximum of at most `max_runs` lines. That maximum
is **convex**, its pieces appear in order of increasing slope, and each line owns
at most one interval.

Three consequences shape this module:

- the owner sequence can be walked left to right, taking at each step the
  earliest point some line overtakes the current owner;
- several lines can overtake at one abscissa. The steepest of them owns the
  ground beyond it; picking any other strands the true owner, whose own crossing
  sits at the same place that the walk has already passed;
- a line excluded from the walk can be cleared in **one** certified comparison.
  For an excluded line `k`, `k - envelope` is linear minus convex, hence concave,
  so its maximum sits at the single breakpoint where the envelope's slope
  brackets `k`'s. If `k` is not above the envelope there, it is above it nowhere.

The last point is what makes certifying the proposed owner against *every* live
line affordable: it costs one comparison per line, not one per pair.

The walk itself runs in ordinary arithmetic, so it is checked rather than
trusted. Which line leads at the cell's left edge is proposed from the readings
and then certified by `certified_sign.certified_margin_sign`; every line the walk
excludes is certified against the owners meeting at its bracketing breakpoint. A
line certified *above* them poisons the row rather than publishing a guess: the
walk gave away ground that line owns, so the row would carry the wrong branch's
policy across an interval.

A margin whose sign the certificate cannot settle is the opposite case and is not
refused. The two lines then differ by less than the arithmetic can represent, so
there is no interval on which either is demonstrably better and nothing is being
guessed — only selected, which the walk does deterministically by taking the
steepest. Refusing would discard correct rows over a distinction the model cannot
observe, and would do so routinely: cell edges *are* candidate abscissae, so a
line regularly ends exactly where the cell does and meets its neighbour there.

Readings taken at face value are not enough to *drive* the walk either. Where the
lines sit on a large common value level they collapse onto a single double, and
the walk then sees no crossing where a real one exists — so readings are formed
relative to a reference line, cancelling the level before anything rounds, and
are carried as a `(high, low)` pair until a decision consumes them. See
`_recentred_edge_readings`.

Which lines own the cell is settled by certified comparisons alone, but *where*
they hand over is not merely cosmetic. The refined row carries a switch as a
duplicated abscissa holding both records, and a right-continuous read there
returns the incoming line, so a breakpoint one float below the true crossing
publishes the incoming policy — and its marginal — on a state the outgoing line
still owns, and one float above withholds them after ownership has passed. The
handover belongs at the smallest float at or above the crossing.

The walk cannot supply that abscissa. Its readings travel at a rounded slope, so
the crossing it sees is right to within a reading and not to within a float, and
rounding a crossing that is itself uncertain up rather than to nearest does not
recover the missing digits. So the walk only *ranks* — which line leads, and in
what order they hand over — and the abscissae it settles on are re-derived
afterwards from the lines' stored endpoints, where the difference of two lines is
affine with exactly representable coefficients. See `_place_handovers`.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_add_float,
    dd_from_difference,
    dd_mul,
    dd_mul_float,
    dd_negate,
    dd_quotient,
    dd_quotient_bounded,
    normalizing_exponent,
    scale_by_power_of_two,
    two_prod,
    two_sum,
)
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarBool

# A per-link reading at one cell edge, as `(high, low)`. Collapsing the pair to
# its sum discards exactly the digits that decide a handover between links whose
# readings agree to the last bit, so the pair is carried until a decision is made.
type EdgeReading = tuple[Float1D, Float1D]


def hull_owners(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
    max_runs: int,
) -> tuple[Float1D, Int1D, ScalarBool]:
    """Return the envelope's breakpoints and owners across one node cell.

    Args:
        left: Abscissa of the cell's left edge.
        right: Abscissa of the cell's right edge.
        live: Boolean `(max_runs,)`; whether each run covers the cell.
        low: Candidate index of each link's lower endpoint, `(max_runs,)`.
        high: Candidate index of each link's upper endpoint, `(max_runs,)`.
        endog_grid: Candidate endogenous grid points in producer order.
        value: Candidate value-correspondence points at `endog_grid`.
        max_runs: Static capacity for the number of x-monotone runs.

    Returns:
        Tuple of the `max_runs + 1` breakpoints in ascending order, the
        `max_runs` owning link indices, and whether any ownership decision could
        not be certified. Once no link overtakes the owner the remaining
        breakpoints sit on the cell's right edge, so the pieces they would open
        have zero width.

    """
    at_left, at_right, slope = _recentred_edge_readings(
        left=left,
        right=right,
        live=live,
        low=low,
        high=high,
        endog_grid=endog_grid,
        value=value,
    )
    first = _leading_link_at(at_query=at_left[0] + at_left[1], live=live, slope=slope)
    bounds, owners = _walk_owners(
        left=left,
        right=right,
        live=live,
        at_left=at_left,
        at_right=at_right,
        slope=slope,
        first=first,
        max_runs=max_runs,
    )
    bounds, misplaced = _place_handovers(
        bounds=bounds,
        owners=owners,
        low=low,
        high=high,
        endog_grid=endog_grid,
        value=value,
        left=left,
        right=right,
    )
    uncertified = _fails_all_live_check(
        bounds=bounds,
        owners=owners,
        live=live,
        slope=slope,
        at_left=at_left,
        at_right=at_right,
        low=low,
        high=high,
        endog_grid=endog_grid,
        value=value,
    )
    return bounds, owners, uncertified | misplaced


def _recentred_edge_readings(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
) -> tuple[EdgeReading, EdgeReading, Float1D]:
    """Read every link at both cell edges, accurately enough to be compared.

    Evaluating each link and subtracting afterwards cannot drive the walk. Where
    the links sit on a large common value level, their readings at one edge
    collapse onto a single double while the differences that decide ownership
    live orders of magnitude below it — the walk then sees no crossing where a
    genuine one exists, and a branch that owns ground is silently dropped.

    The level is therefore cancelled *before* anything is rounded. Readings are
    published relative to one live reference link, and each is assembled as

    ```{math}
    (v_0 - v_0^{ref}) + \\big[(x - x_0)\\,s - (x - x_0^{ref})\\,s^{ref}\\big],
    ```

    where the leading difference is exact — two values on one level are within a
    factor of two of each other, so their difference is representable — and the
    bracket holds the link's *rise across one cell* relative to the reference's.
    The rise terms are formed with error-free transforms and their tails kept, so
    the bracket is exact too, and links that meet at one abscissa are seen to meet
    there rather than a rounding apart. That matters beyond accuracy: a
    simultaneous crossing read a rounding apart would open a sliver piece for a
    link that owns nothing, and publish its policy across it.

    Working in differences is free because differences are all the walk uses: a
    common reference cancels out of every margin and every crossing. The two
    edges take different references, so their difference carries a constant
    offset — the same one for every link, so it changes no comparison.

    Absolute readings are never published from here. Values in the refined row
    come from the owning link's own interpolant, and the decisions that must hold
    exactly go through `certified_sign`.

    Returns:
        Tuple of the readings at `left`, the readings at `right`, and each link's
        slope; all three are `-inf` where the link does not cover the cell.
    """
    x0 = endog_grid[low]
    x1 = endog_grid[high]
    v0 = value[low]
    v1 = value[high]

    # One exponent per group for the whole cell, never one per link: a reading is
    # homogeneous of degree one in the values, so a per-link value exponent would
    # rescale each link differently and leave nothing comparable. Normalizing the
    # abscissae keeps the slope, and hence Dekker's splitting of it, in range.
    abscissa_exponent = normalizing_exponent(
        jnp.maximum(
            _live_magnitude(x0, x1, live=live),
            jnp.maximum(jnp.abs(left), jnp.abs(right)),
        )
    )
    value_exponent = normalizing_exponent(_live_magnitude(v0, v1, live=live))
    x0, x1, left, right = (
        scale_by_power_of_two(term, -abscissa_exponent)
        for term in (x0, x1, left, right)
    )
    v0, v1 = (scale_by_power_of_two(term, -value_exponent) for term in (v0, v1))

    width = x1 - x0
    slope = jnp.where(live, (v1 - v0) / jnp.where(width == 0.0, 1.0, width), -jnp.inf)

    reference = jnp.argmax(live).astype(jnp.int32)

    def read(x_query: FloatND) -> EdgeReading:
        # Anchor each link at whichever of its own endpoints is nearer the query.
        # `slope` is a rounded quotient, so a reading carries the slope's error
        # multiplied by the distance travelled: over a link far longer than the
        # cell, that error alone can exceed the sub-ULP gap the reading exists to
        # resolve. Cell edges *are* candidate abscissae, so the query is often an
        # endpoint of the very link being read — anchored there the distance is
        # zero and the reading is the stored value, exactly.
        anchored_right = jnp.abs(x_query - x1) < jnp.abs(x_query - x0)
        anchor_x = jnp.where(anchored_right, x1, x0)
        anchor_v = jnp.where(anchored_right, v1, v0)
        level = anchor_v[reference]

        offset_high, offset_low = two_sum(x_query, -anchor_x)
        step_high, step_low = two_prod(offset_high, slope)
        step_low = step_low + offset_low * slope
        rise_high, rise_error = two_sum(step_high, -step_high[reference])
        rise_low = (step_low - step_low[reference]) + rise_error
        # Two anchors on one value level differ by less than a factor of two, so
        # their difference is exact; the only rounding left is the one this last
        # `two_sum` records rather than discards.
        high, low = two_sum(anchor_v - level, rise_high)
        high, low = two_sum(high, low + rise_low)
        dead_reading = jnp.full_like(high, -jnp.inf)
        return (
            jnp.where(live, high, dead_reading),
            jnp.where(live, low, jnp.zeros_like(low)),
        )

    return read(left), read(right), slope


def _reading_difference(
    minuend: tuple[FloatND, FloatND], subtrahend: tuple[FloatND, FloatND]
) -> tuple[FloatND, FloatND]:
    """Return `minuend - subtrahend`, keeping the low word.

    Rank-polymorphic on purpose: the owner's reading is a scalar and the links'
    are a vector, and the difference is taken between the two.

    A link that does not cover the cell reads `-inf`, and the error-free
    transforms are meaningless there — `two_sum` on two infinities yields NaN,
    which would spread through every comparison. Such a difference falls back to
    the ordinary one, which is the infinity the caller's predicates expect.
    """
    ordinary = minuend[0] - subtrahend[0]
    high, error = two_sum(minuend[0], -subtrahend[0])
    low = (minuend[1] - subtrahend[1]) + error
    refined_high, refined_low = two_sum(high, low)
    exact = jnp.isfinite(ordinary)
    return (
        jnp.where(exact, refined_high, ordinary),
        jnp.where(exact, refined_low, jnp.zeros_like(refined_low)),
    )


def _is_negative(reading: tuple[FloatND, FloatND]) -> BoolND:
    """Whether a reading is strictly below zero, low word included.

    The high word alone answers this only when it is nonzero. Where two links
    meet to within a rounding, it is exactly zero and the entire question lives
    in the low word — which is the case this module exists to get right.
    """
    high, low = reading
    return (high < 0.0) | ((high == 0.0) & (low < 0.0))


def _place_handovers(
    *,
    bounds: Float1D,
    owners: Int1D,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
    left: FloatND,
    right: FloatND,
) -> tuple[Float1D, ScalarBool]:
    """Re-place every breakpoint at the first state its incoming link owns.

    The refined row carries a switch as a duplicated abscissa holding both links'
    records, and a right-continuous read there returns the incoming link. The
    abscissa is therefore structural, not descriptive: one state below the true
    crossing the row publishes the incoming policy — and its marginal — over
    ground the outgoing link still owns; one state above it withholds them after
    ownership has passed.

    Neither placement can be read off the walk's edge readings, which travel at a
    rounded slope. Cross-multiplied, the difference of two links is

    ```{math}
    f(x) = N_a(x)\\,w_b - N_b(x)\\,w_a,
    ```

    affine in `x`, and both its value at the cell's left edge and its slope are
    exact double-double expressions in the two links' stored endpoints, each
    carrying its own error bound. Because the walk orders owners by increasing
    slope, `f` decreases across a handover, so the state to publish is the
    smallest representable one at or above the root — and *which* one that is
    the quotient's low word says: a positive low word puts the root above the
    high word, anything else at or below it.

    What the row is asked to certify is that the bound fits inside a state, so
    that only those two candidates are in play. It is deliberately not asked to
    beat the low word as well. Where the bound straddles zero the crossing is
    within the bound of the published state — a sub-state quantity, far finer
    than the states the row is written in — and refusing there would discard the
    row over a distinction it has no way to express. A bound wider than a state
    is the genuinely undecided case, and that one is refused.

    The margin predicate is not the instrument for this. Adjacent to a crossing
    the two links are within a rounding of each other, so
    `certified_sign.certified_margin_sign` reports a determinant under its own
    resolution there — by construction, not by accident — and a placement driven
    by it would hand over a state early whenever it did.

    This runs once per cell over all breakpoints at once, not inside the walk.
    The walk decides *who* owns what, which is certified separately; only the
    boundaries it settles on need locating, and there are at most `max_runs` of
    them however many steps the walk took.
    """
    outgoing, incoming = owners[:-1], owners[1:]
    a_x0, a_x1 = endog_grid[low[outgoing]], endog_grid[high[outgoing]]
    a_v0, a_v1 = value[low[outgoing]], value[high[outgoing]]
    b_x0, b_x1 = endog_grid[low[incoming]], endog_grid[high[incoming]]
    b_v0, b_v1 = value[low[incoming]], value[high[incoming]]

    width_a = dd_from_difference(a_x1, a_x0)
    width_b = dd_from_difference(b_x1, b_x0)

    # Subtracting one common constant from all four values leaves `f` unchanged
    # — each numerator loses `c` times its own width, and the cross-multiplication
    # cancels the two — so the level the lines sit on can be taken out before it
    # is ever multiplied. Leaving it in would set the coefficients' error bounds
    # by the *level* while the slope they are divided by is set by the value
    # *differences*, and where the two are orders of magnitude apart the located
    # root inherits an error far wider than the float it is meant to pick out.
    reference = a_v0
    numerator_a = _recentred_numerator(
        x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1, reference=reference, x_query=left
    )
    numerator_b = _recentred_numerator(
        x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1, reference=reference, x_query=left
    )
    at_edge = dd_add(
        dd_mul(numerator_a, width_b),
        dd_negate(dd_mul(numerator_b, width_a)),
    )
    rate = dd_add(
        dd_mul(dd_from_difference(a_v1, a_v0), width_b),
        dd_negate(dd_mul(dd_from_difference(b_v1, b_v0), width_a)),
    )
    degenerate = (rate[0] + rate[1]) == 0.0
    safe_rate = (
        jnp.where(degenerate, jnp.ones_like(rate[0]), rate[0]),
        jnp.where(degenerate, jnp.zeros_like(rate[1]), rate[1]),
        rate[2],
    )
    step_high, step_low, step_error = dd_quotient_bounded(dd_negate(at_edge), safe_rate)
    root_high, root_low, root_dropped = dd_add_float(
        (step_high, step_low, jnp.zeros_like(step_high)), left
    )

    # Everything the located root has to be read against: the two coefficients'
    # own bounds carried through the division, what the division could not clear,
    # and the tail the shift back to absolute coordinates discarded.
    magnitude = jnp.abs(rate[0] + rate[1])
    safe_magnitude = jnp.where(degenerate, jnp.ones_like(magnitude), magnitude)
    step = jnp.abs(step_high) + jnp.abs(step_low)
    root_error = (
        (at_edge[2] + step * rate[2]) / safe_magnitude + step_error + root_dropped
    )

    # What has to hold is that the bound fits inside a state: the crossing is
    # then known to lie in `root_high` or the state above it, and no third
    # candidate is in play. The low word is not part of that test — it is the
    # exact residual of the located pair, routinely half a state wide, and it is
    # what *chooses* between the two: a positive residual puts the crossing above
    # `root_high`, so ownership passes at the state above it, and anything else
    # puts it at or below, so `root_high` is already the incoming link's.
    resolution = jnp.nextafter(root_high, jnp.inf) - root_high
    resolved = 2.0 * root_error < resolution
    candidate = jnp.where(root_low > 0.0, jnp.nextafter(root_high, jnp.inf), root_high)

    # Breakpoints where ownership does not change are not handovers; the walk
    # parks them on the cell's right edge and they must stay there.
    interior = bounds[1:-1]
    hands_over = outgoing != incoming
    placed = jnp.where(hands_over, jnp.clip(candidate, left, right), interior)

    # Where several lines meet at one abscissa the walk gives the ones between
    # the outer two no width at all, and that is the topology, not an accident of
    # arithmetic: the middle lines own nothing and must emit nothing. Each pair's
    # root is derived on its own here, so two pairs meeting at one point can
    # round to states an ULP apart and hand a line an interval it does not own.
    # Re-placement therefore inherits the walk's coincidences: only the first
    # breakpoint of a coincident group is placed, and the rest follow it. The
    # running maximum does both jobs — it carries a group's placement across its
    # members and keeps the sequence ascending where a placement moved.
    opens_group = jnp.concatenate(
        [jnp.ones_like(interior[:1], dtype=bool), interior[1:] != interior[:-1]]
    )
    placed = jax.lax.cummax(jnp.where(opens_group, placed, -jnp.inf))
    finite = jnp.isfinite(at_edge[0] + at_edge[1]) & jnp.isfinite(rate[0] + rate[1])
    unresolved = jnp.any(hands_over & (~finite | ~resolved))
    return jnp.concatenate([bounds[:1], placed, bounds[-1:]]), unresolved


def _recentred_numerator(
    *,
    x0: FloatND,
    x1: FloatND,
    v0: FloatND,
    v1: FloatND,
    reference: FloatND,
    x_query: FloatND,
) -> DoubleDouble:
    """Return `(v0 - ref)*(x1 - x) + (v1 - ref)*(x - x0)` with nothing dropped.

    The width-scaled value at `x`, measured from `reference` rather than from
    zero. Each recentred value is formed as an exact pair, so the shift costs no
    accuracy even where the two values are too far apart to subtract exactly in
    one float.
    """
    return dd_add(
        dd_mul(dd_from_difference(x1, x_query), dd_from_difference(v0, reference)),
        dd_mul(dd_from_difference(x_query, x0), dd_from_difference(v1, reference)),
    )


def _approximate_crossing(
    *,
    margin_left: tuple[FloatND, FloatND],
    margin_right: tuple[FloatND, FloatND],
    left: FloatND,
    right: FloatND,
) -> Float1D:
    """Return roughly where each link catches the owner, for ranking only.

    The walk needs an order over the candidates before it can know which one it
    will take, and this supplies it from the edge readings. It is not where the
    handover is published: a reading travels from its link's anchor to the query
    at a *rounded* slope, so it carries that slope's error times the distance
    travelled, and the root of two such readings can sit on the wrong side of a
    representable state. Where the links share endpoints that distance is zero
    and the readings are the stored values exactly, which is why the error shows
    up only once the supports are shifted.

    Ranking on it is sound because the *choice* of owner is certified elsewhere:
    a link this order strands is caught by the all-live check, which poisons the
    row. `_place_handovers` then re-places the breakpoints the walk settled on,
    from the stored endpoints.
    """
    numerator = (margin_left[0], margin_left[1], jnp.zeros_like(margin_left[0]))
    span = dd_add(
        numerator,
        dd_negate((margin_right[0], margin_right[1], jnp.zeros_like(margin_right[0]))),
    )
    degenerate = (span[0] + span[1]) == 0.0
    safe_span = (
        jnp.where(degenerate, jnp.ones_like(span[0]), span[0]),
        jnp.where(degenerate, jnp.zeros_like(span[1]), span[1]),
        span[2],
    )
    weight_high, weight_low = dd_quotient(numerator, safe_span)
    width = right - left
    offset = dd_mul_float((weight_high, weight_low, jnp.zeros_like(weight_high)), width)
    state_high, state_low, _dropped = dd_add_float(offset, left)
    # Round the root up: a residual above the published float means the true
    # crossing lies beyond it, so the incoming link does not own it yet.
    return jnp.where(state_low > 0.0, jnp.nextafter(state_high, jnp.inf), state_high)


def _live_magnitude(*per_link: Float1D, live: BoolND) -> FloatND:
    """Return the largest magnitude any live link contributes, as a scalar."""
    magnitude = jnp.zeros((), dtype=per_link[0].dtype)
    for term in per_link:
        usable = live & jnp.isfinite(term)
        magnitude = jnp.maximum(
            magnitude, jnp.max(jnp.where(usable, jnp.abs(term), 0.0))
        )
    return magnitude


def _leading_link_at(*, at_query: Float1D, live: BoolND, slope: Float1D) -> IntND:
    """Return the link the walk starts from at a cell edge.

    This is a *proposal*, not a decision, and it carries no certificate of its
    own — it does not need one. A wrong proposal makes the walk publish an
    envelope the true leader rises above near the edge, and the all-live check
    then catches that link at its bracketing breakpoint and poisons the row. So
    the certificate is complete without a comparison here, and the readings only
    have to be good enough to keep a certifiable cell out of the poison path.

    A tie goes to the steeper link, which owns the ground to the right of the
    edge.
    """
    ranking = jnp.where(live, at_query, -jnp.inf)
    tied = live & (ranking == jnp.max(ranking))
    return jnp.argmax(jnp.where(tied, slope, -jnp.inf)).astype(jnp.int32)


def _walk_owners(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    at_left: EdgeReading,
    at_right: EdgeReading,
    slope: Float1D,
    first: IntND,
    max_runs: int,
) -> tuple[Float1D, Int1D]:
    """Walk the owner sequence from the cell's left edge to its right edge.

    Every link is read from the two edge readings the caller supplies rather than
    evaluated afresh: two evaluations of one affine link at one abscissa can land
    an ULP apart when the compiler vectorizes the call sites differently, and the
    walk would then be stepping on numbers no other stage ever saw.
    """

    def advance(
        carry: tuple[FloatND, IntND], _step: None
    ) -> tuple[tuple[FloatND, IntND], tuple[FloatND, IntND]]:
        x_owned_from, owner = carry
        owner_left = (at_left[0][owner], at_left[1][owner])
        owner_right = (at_right[0][owner], at_right[1][owner])
        margin_left = _reading_difference(owner_left, at_left)
        margin_right = _reading_difference(owner_right, at_right)

        # The crossing's *position* is an ordinary float — it becomes an abscissa
        # of the published row, which holds it to that precision anyway. Whether
        # there is a crossing at all is a different question, and it is settled
        # on the low word: two links whose readings agree to the last bit still
        # hand over, and rounding the margin first hides exactly that handover.
        # A link takes over when it is strictly above the owner at the cell's
        # right edge; the crossing only says *where*, and is clamped into the
        # ground still unowned rather than allowed to veto the handover. A cell
        # is bounded by consecutive live abscissae and can therefore be narrower
        # than one ULP of its own coordinates, leaving no representable interior
        # point at all — a position test would then reject every handover inside
        # it and poison the row over a rounding.
        #
        # The walk still terminates: a link is taken only while strictly above
        # the current owner there, which it can never be against itself.
        crossing = jnp.clip(
            _approximate_crossing(
                margin_left=margin_left,
                margin_right=margin_right,
                left=left,
                right=right,
            ),
            x_owned_from,
            right,
        )
        overtakes = live & _is_negative(margin_right)
        candidate = jnp.where(overtakes, crossing, jnp.inf)
        earliest = jnp.min(candidate)
        # Several links can meet the owner at one abscissa. The envelope is
        # convex, so the steepest of them owns the ground beyond that point; a
        # link between them owns nothing and taking it would leave the true
        # owner unreachable, since its crossing sits where the walk already is.
        simultaneous = candidate == earliest
        successor = jnp.argmax(jnp.where(simultaneous, slope, -jnp.inf)).astype(
            jnp.int32
        )
        found = jnp.isfinite(earliest)
        next_x = jnp.where(found, earliest, right)
        next_owner = jnp.where(found, successor, owner).astype(jnp.int32)
        return (next_x, next_owner), (next_x, next_owner)

    _carry, (later_bounds, later_owners) = jax.lax.scan(
        advance, (left, first), None, length=max_runs - 1
    )
    bounds = jnp.concatenate([left[None], later_bounds, right[None]])
    owners = jnp.concatenate([first[None], later_owners])
    return bounds, owners


def _fails_all_live_check(
    *,
    bounds: Float1D,
    owners: Int1D,
    live: BoolND,
    slope: Float1D,
    at_left: EdgeReading,
    at_right: EdgeReading,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
) -> ScalarBool:
    """Report whether any live link escapes certification against the owners.

    Owners appear in order of increasing slope, so the walk's breakpoints carry a
    non-decreasing slope sequence. For a link `k`, the difference `k - envelope`
    is concave, and it peaks at the breakpoint where that sequence crosses `k`'s
    own slope. One comparison there therefore settles `k` over the whole cell:
    certified below, and `k` is below everywhere; certified above, and the walk
    passed over a link that owns ground. This is the whole reason certifying
    against every live link is affordable — it is one comparison per link.

    Two owners meet at that breakpoint and the breakpoint is placed in ordinary
    arithmetic, so they agree there only to within its placement error. `k` is
    compared against whichever of them reads higher. The choice is made on
    readings and can in principle go the wrong way, but only in the safe
    direction: the envelope is at least as high as either owner, so `k` certified
    *below* the chosen one is below the envelope whatever the choice was. A wrong
    choice can only turn a publishable cell into a refused one.

    Locating the breakpoint presumes the owners' slopes ascend, which convexity
    guarantees and the walk should deliver. That is the premise the whole
    certificate rests on, so it is checked rather than assumed: a walk that
    handed back a link twice would search an unsorted sequence and could look
    past the very link that escaped it.
    """
    n_links = slope.shape[0]
    n_owners = owners.shape[0]
    index = jnp.arange(n_links, dtype=jnp.int32)

    owner_slope = slope[owners]
    not_convex = jnp.any(live) & jnp.any(jnp.diff(owner_slope) < 0.0)
    position = jnp.searchsorted(owner_slope, slope, side="left").astype(jnp.int32)
    x_query = bounds[position]
    before = owners[jnp.clip(position - 1, 0, n_owners - 1)]
    after = owners[jnp.clip(position, 0, n_owners - 1)]

    width = bounds[-1] - bounds[0]
    weight = (x_query - bounds[0]) / jnp.where(width == 0.0, 1.0, width)
    left_value = at_left[0] + at_left[1]
    right_value = at_right[0] + at_right[1]
    reads_before = left_value[before] + weight * (
        right_value[before] - left_value[before]
    )
    reads_after = left_value[after] + weight * (right_value[after] - left_value[after])
    rival = jnp.where(reads_before >= reads_after, before, after)

    order = certified_margin_sign(
        a_x0=endog_grid[low],
        a_x1=endog_grid[high],
        a_v0=value[low],
        a_v1=value[high],
        b_x0=endog_grid[low[rival]],
        b_x1=endog_grid[high[rival]],
        b_v0=value[low[rival]],
        b_v1=value[high[rival]],
        x_query=x_query,
    )
    # A link compared with itself is not a contest: the determinant is zero by
    # construction but carries the accumulated error bound of both products, so
    # it would be reported as undecidable.
    owns_the_breakpoint = (index == before) | (index == after)
    # Only a link certified strictly *above* the envelope is a defect: the walk
    # gave its ground to someone else, and the row would publish the wrong
    # branch's policy across an interval.
    #
    # A margin the certificate cannot *sign* is a different thing. The two links
    # then differ by less than the arithmetic can represent, so no interval
    # exists on which one is demonstrably better — the choice is a selection
    # from options that are indistinguishable at this precision, and the walk's
    # steepest-slope tie-break makes it deterministically. Refusing there would
    # discard correct rows over a coin toss the model cannot observe; and the
    # case is not exotic, since cell edges *are* candidate abscissae, so a link
    # routinely ends exactly where the cell does and meets its neighbour there.
    #
    # A margin the certificate could not *compute* is a defect again, and the
    # opposite one: no determinant was produced, so the links may be far apart
    # and the silence says nothing about which owns the ground.
    escapes = (order == 1) | (order == UNRESOLVED_SIGN)
    escaped = jnp.any(live & ~owns_the_breakpoint & escapes)
    return escaped | not_convex
