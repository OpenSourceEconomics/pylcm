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
line certified above them, or a comparison whose sign cannot be certified, poisons
the row rather than publishing a guess.

Readings taken at face value are not enough to *drive* the walk either. Where the
lines sit on a large common value level they collapse onto a single double, and
the walk then sees no crossing where a real one exists — so readings are formed
relative to a reference line, cancelling the level before anything rounds. See
`_recentred_edge_readings`.

The breakpoints themselves stay in ordinary arithmetic: they become abscissae of
the published row, which carries them to float precision anyway. A misplaced
breakpoint moves an owner boundary by an ULP; it cannot change which lines own
the cell, because that question is settled by certified comparisons alone.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from _lcm.egm.upper_envelope.double_double import (
    normalizing_exponent,
    two_prod,
    two_sum,
)
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarBool


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
    first = _leading_link_at(at_query=at_left, live=live, slope=slope)
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
    return bounds, owners, uncertified


def _recentred_edge_readings(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    low: IntND,
    high: IntND,
    endog_grid: Float1D,
    value: Float1D,
) -> tuple[Float1D, Float1D, Float1D]:
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
        jnp.ldexp(term, -abscissa_exponent) for term in (x0, x1, left, right)
    )
    v0, v1 = (jnp.ldexp(term, -value_exponent) for term in (v0, v1))

    width = x1 - x0
    slope = jnp.where(live, (v1 - v0) / jnp.where(width == 0.0, 1.0, width), -jnp.inf)

    reference = jnp.argmax(live).astype(jnp.int32)
    level = v0[reference]

    def read(x_query: FloatND) -> Float1D:
        offset_high, offset_low = two_sum(x_query, -x0)
        step_high, step_low = two_prod(offset_high, slope)
        step_low = step_low + offset_low * slope
        rise = (step_high - step_high[reference]) + (step_low - step_low[reference])
        return jnp.where(live, (v0 - level) + rise, -jnp.inf)

    return read(left), read(right), slope


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
    at_left: Float1D,
    at_right: Float1D,
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
        margin_left = at_left[owner] - at_left
        margin_right = at_right[owner] - at_right
        span = margin_left - margin_right
        safe_span = jnp.where(span == 0.0, 1.0, span)
        crossing = left + (margin_left / safe_span) * (right - left)
        # A link takes over where it crosses the owner and stays above it, which
        # inside this cell is exactly a crossing it enters from below.
        overtakes = (
            live & (margin_right < 0.0) & (crossing > x_owned_from) & (crossing < right)
        )
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
    at_left: Float1D,
    at_right: Float1D,
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
    reads_before = at_left[before] + weight * (at_right[before] - at_left[before])
    reads_after = at_left[after] + weight * (at_right[after] - at_left[after])
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
    escapes = (order == 1) | (order == UNRESOLVED_SIGN)
    escaped = jnp.any(live & ~owns_the_breakpoint & escapes)
    return escaped | not_convex
