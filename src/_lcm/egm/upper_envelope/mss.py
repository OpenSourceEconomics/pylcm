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
grid spacing. The refined arrays therefore track the FUES envelope tightly.
Unlike FUES it consumes no `jump_thresh` heuristic: the winner switch is read
directly off the evaluated values.

The rule is stated as a left-to-right sweep, but the executable path evaluates
it in parallel rather than sequentially: every query abscissa is scored against
every link at once, as a dense `(n_query, n_link)` bracket-and-compare block,
and the winner is a reduction along the link axis. Each query's answer is
independent of the others, so nothing about the rule needs the order — what the
parallel form costs instead is a working set proportional to the product, where
a true emitting scan would carry one query at a time.

Three decisions here are structural rather than numerical — which link owns a
query, whether the owner changed between two queries, and where the change
happened — and none of them is settled by a rounded comparison:

- **Ownership is certified before reading.** One exact reduction over the
  admitted links decides the winner from the stored operands. Admission depends
  only on each link's stored support, never on whether a floating read succeeded.
  Links certified level with the winner are separated by a
  right-continuous rule — the link that extends strictly right of the query,
  then the steeper one, then the earlier stored link — so the owner at a node
  where two branches meet is the one that owns the interval above it, and the
  switch is visible at that node rather than one interval later.
- **Only the owner's channels are read.** The native affine reader forms the
  exact rational from stored endpoints and rounds once to the working format,
  without floating weighted products. Endpoints and common levels are preserved;
  a failed read poisons the publication rather than removing its owner. Crossing
  ordinates are rounded upward only when an exact comparison requires it.
- **A crossing is located inside the interval it happened in.** The two winning
  chords' gap is certified at the two adjacent query abscissae. Its exact
  stored-operand root is rounded upward to the first representable state owned
  by the incoming branch. Record coalescence uses that emitted state, including
  when a nonrepresentable root hands over at an existing query node.

A crossing abscissa is inserted twice — same abscissa, left- and
right-extrapolated policy — so the refined arrays stay weakly ascending and the
policy discontinuity at a discrete-choice switch is preserved exactly, the same
convention FUES uses. Where a crossing lands *on* one of the two query nodes,
that node's own row is already one of the two records, and the emission
contributes only the other one:

- a crossing at the left node: that node published the outgoing owner, so only
  the incoming owner's record is inserted, after it;
- a crossing at the right node: that node publishes the incoming owner, so only
  the outgoing owner's record is inserted, before it.

Either way the kink abscissa carries exactly two rows, outgoing owner first, and
no row duplicates the node's own. Both orientations are stated so the ordering
is defined whichever way the geometry falls, but they are not equally reachable:
right-continuous ownership hands a node where two branches meet to the branch
that owns the interval above it, which puts the crossing on the *right* node of
the interval the switch is observed in. The left-node branch is the rule's other
half rather than a case the sweep is expected to take.

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope._exact_affine import (
    UNRESOLVED_STATUS,
    exact_affine_handover,
    exact_affine_read,
    exact_query_winner_batched,
)
from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarInt


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
    sorted ascending and swept left-to-right; at each abscissa the certified
    owner among the bracketing segments publishes both the value and the policy,
    and where the owning branch switches between two adjacent abscissae the
    crossing is inserted (twice — left and right policy — unless it lands on one
    of the two nodes, whose own row is then one of the two records). The refined
    arrays have static length `n_refined`, hold the envelope points in weakly
    ascending grid order, and are NaN-padded in the tail.

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
    grid_key = _stored_key(value=jnp.where(dead, jnp.inf, endog_grid))
    order = jnp.argsort(grid_key, stable=True)
    query_grid = jnp.where(dead, jnp.nan, endog_grid)[order]
    query_dead = dead[order]
    # Several candidate branches can supply the same query abscissa. They still
    # all participate as links, but the sweep publishes the owning node once.
    # Otherwise a node-aligned switch emits its outgoing record beside several
    # identical incoming-node records instead of one outgoing/incoming pair.
    repeated_query = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=bool),
            _stored_equal(left=query_grid[1:], right=query_grid[:-1]),
        )
    )
    query_dead = query_dead | repeated_query

    links, link_segment = _segment_chain(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        dead=dead,
        segment_id=segment_id,
    )

    envelope_value, envelope_policy, winner_link, winner_segment = _evaluate_envelope(
        query_grid=query_grid, links=links, segment_id=link_segment
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
    # crossing of the two winning links (as up to two records, left and right
    # policy). The first live query has no predecessor, so it emits only its node.
    crossing = _crossing_blocks(
        query_grid=query_grid,
        query_drop=query_drop,
        winner_link=winner_link,
        winner_segment=winner_segment,
        links=links,
    )

    # A crossing is a genuine envelope kink only if one of the two crossing
    # branches owns the envelope at the crossing abscissa. A switch across a
    # *gap* in the candidate cloud intersects two lines in an interval a third
    # branch dominates, or that no segment covers at all; evaluating the envelope
    # there names its owner, and a crossing whose owner is neither branch is not
    # on the envelope. The test is the identity of the owner, so nothing about it
    # depends on a tolerance.
    _, _, _, crossing_owner = _evaluate_envelope(
        query_grid=crossing.grid, links=links, segment_id=link_segment
    )
    # Owner identity, not read finiteness, determines event admission too. A
    # failed publication must remain an emitted NaN rather than erase the kink.
    on_envelope = (crossing_owner == crossing.segment_left) | (
        crossing_owner == crossing.segment_right
    )
    left_valid = crossing.left_valid & on_envelope
    right_valid = crossing.right_valid & on_envelope

    # A node-aligned crossing reuses one node record. Give that record the same
    # outward ordinate as the inserted one; its ordinary nearest node reading
    # can otherwise sit below both exact chords. This changes no record counts.
    node_index = jnp.arange(query_grid.shape[0], dtype=jnp.int32)
    live_index = jax.lax.associative_scan(
        jnp.maximum, jnp.where(~query_drop, node_index, -1)
    )
    previous_index = jnp.concatenate((jnp.full((1,), -1), live_index[:-1]))
    at_previous = (previous_index >= 0) & _stored_equal(
        left=crossing.grid, right=query_grid[jnp.maximum(previous_index, 0)]
    )
    crossing_node = jnp.where(
        _stored_equal(left=crossing.grid, right=query_grid),
        node_index,
        jnp.where(at_previous, previous_index, query_grid.shape[0]),
    )
    crossing_node = jnp.where(
        left_valid | right_valid, crossing_node, query_grid.shape[0]
    )
    node_value = node_value.at[crossing_node].set(crossing.value, mode="drop")
    # Refusing an event location must not silently publish a row without its
    # kink. Keep the observed query and poison its channels instead of making
    # up a finite crossing abscissa or treating a failed certificate as no event.
    node_value = jnp.where(crossing.unresolved, jnp.nan, node_value)
    node_policy = jnp.where(crossing.unresolved, jnp.nan, node_policy)

    # Per-query output block: up to three rows in ascending grid order — the two
    # crossing records (same abscissa, left then right policy) followed by the
    # query node. The crossing of step `i` lies in `[grid_{i-1}, grid_i]`, and a
    # crossing landing on a node contributes only the record that node's own row
    # is not, so the kink abscissa carries exactly two rows either way.
    nan_scalar = jnp.full((), jnp.nan, dtype=query_grid.dtype)
    node_valid = ~query_drop
    row_valid = jnp.stack([left_valid, right_valid, node_valid], axis=1).ravel()
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


class _Links(NamedTuple):
    """Every link as a comparable affine line plus the span it actually covers.

    `certified_sign` compares lines and says plainly what it will not invent: a
    link of zero width has no affine line and a non-finite operand is unresolved
    rather than false. The comparable fields give readings and crossing signs a
    line to work with. Ownership instead receives `lower`, `upper`, `v0` and
    `upper_value`: the original oriented endpoints, including a singleton's
    original upper endpoint/value. A readable line must never manufacture right
    extension, width or support for the owner decision.
    """

    lower: Float1D
    """Lower stored abscissa of the link's span."""
    upper: Float1D
    """Upper stored abscissa of the link's span."""
    upper_value: Float1D
    """Original value at `upper`, before making a singleton readable."""
    x0: Float1D
    """Lower abscissa of the comparable line."""
    x1: Float1D
    """Upper abscissa of the comparable line, strictly above `x0`."""
    v0: Float1D
    """Value at `x0`."""
    v1: Float1D
    """Value at `x1`."""
    p0: Float1D
    """Policy at `x0`."""
    p1: Float1D
    """Policy at `x1`."""
    live: BoolND
    """Whether the link is a real value segment at all."""


def _segment_chain(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    dead: BoolND,
    segment_id: Float1D | None,
) -> tuple[_Links, Int1D]:
    """Build the candidate links and their branch labels.

    Args:
        endog_grid: Candidate abscissae in input order; NaN marks a dead node.
        policy: Candidate policies in the same order.
        value: Candidate values in the same order.
        dead: Per-candidate dead mask.
        segment_id: Explicit per-candidate branch label, or `None` to derive the
            branches from the monotone split.

    Returns:
        Tuple of the comparable links and their per-link branch id.

    """
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
        decreases = _stored_less(left=right_grid, right=left_grid) | (
            _value_decrease_past_noise(left_value=left_value, right_value=right_value)
        )
        link_segment = jnp.cumsum(decreases.astype(jnp.int32))
        segment_live = ~dead[:-1] & ~dead[1:] & ~decreases
    else:
        same_segment = segment_id[:-1] == segment_id[1:]
        link_segment = segment_id[:-1].astype(jnp.int32)
        segment_live = ~dead[:-1] & ~dead[1:] & same_segment

    links = _comparable_links(
        left_grid=left_grid,
        right_grid=right_grid,
        left_policy=left_policy,
        right_policy=right_policy,
        left_value=left_value,
        right_value=right_value,
        segment_live=segment_live,
    )
    return links, link_segment


def _comparable_links(
    *,
    left_grid: Float1D,
    right_grid: Float1D,
    left_policy: Float1D,
    right_policy: Float1D,
    left_value: Float1D,
    right_value: Float1D,
    segment_live: BoolND,
) -> _Links:
    """Orient every link ascending and give a degenerate one a readable width.

    Stored-bit order, not floating arithmetic, orients descending endpoints and
    distinguishes a true singleton from a positive subnormal width. A singleton
    is read as a constant line on [0, 1], even at the largest finite coordinate;
    that surrogate is used only for readings/crossing signs. Its original span
    and endpoint values are retained separately for ownership and admission.
    """
    descending = _stored_less(left=right_grid, right=left_grid)
    x0 = jnp.where(descending, right_grid, left_grid)
    x1 = jnp.where(descending, left_grid, right_grid)
    v0 = jnp.where(descending, right_value, left_value)
    v1 = jnp.where(descending, left_value, right_value)
    p0 = jnp.where(descending, right_policy, left_policy)
    p1 = jnp.where(descending, left_policy, right_policy)

    degenerate = _stored_equal(left=x0, right=x1)
    finite_line = (
        jnp.isfinite(v0) & jnp.isfinite(v1) & jnp.isfinite(x0) & jnp.isfinite(x1)
    )
    return _Links(
        lower=x0,
        upper=x1,
        upper_value=v1,
        x0=jnp.where(degenerate, jnp.zeros_like(x0), x0),
        x1=jnp.where(degenerate, jnp.ones_like(x1), x1),
        v0=v0,
        v1=jnp.where(degenerate, v0, v1),
        p0=p0,
        p1=jnp.where(degenerate, p0, p1),
        live=segment_live & finite_line,
    )


def _chord_value(
    *, x: FloatND, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND
) -> FloatND:
    """Read a chord with one nearest rounding, or NaN on an unresolved read.

    The native reader forms the weighted numerator and width in fixed-width
    integers. Neither overflow nor underflow of an intermediate floating product
    can change a finite answer. Upward event publication is kept separate from
    this nearest reading, and localization uses the exact affine difference.
    """
    reading, _status = _chord_reading(x=x, x0=x0, x1=x1, v0=v0, v1=v1)
    return reading


def _stored_key(*, value: FloatND) -> jax.Array:
    """Materialize unsigned sort keys, identifying the two signed zeros.

    Positive IEEE encodings ascend with magnitude; negative ones descend.
    Normalizing only the zero *bits*, then complementing negative encodings and
    flipping the positive sign bit, gives an unsigned monotone key. No floating
    operation is needed to form the keys. They are used only as integer sort
    inputs, not compared next to their floating operands (see `_stored_equal`).
    Sorting replaces dead coordinates with +inf before building the key.
    """
    integer = jnp.uint64 if value.dtype == jnp.float64 else jnp.uint32
    sign = jnp.asarray(1 << (jnp.finfo(value.dtype).bits - 1), dtype=integer)
    bits = jax.lax.bitcast_convert_type(value, integer)
    bits = jnp.where((bits & ~sign) == 0, jnp.zeros_like(bits), bits)
    return jnp.where((bits & sign) != 0, ~bits, bits ^ sign)


def _stored_parts(*, value: FloatND) -> tuple[jax.Array, jax.Array, BoolND, BoolND]:
    """Decode bits, magnitude, sign and non-NaN status using integers only."""
    integer = jnp.uint64 if value.dtype == jnp.float64 else jnp.uint32
    sign = jnp.asarray(1 << (jnp.finfo(value.dtype).bits - 1), dtype=integer)
    infinity = jax.lax.bitcast_convert_type(jnp.full((), jnp.inf, value.dtype), integer)
    bits = jax.lax.bitcast_convert_type(value, integer)
    magnitude = bits & ~sign
    return bits, magnitude, (bits & sign) != 0, magnitude <= infinity


def _stored_equal(*, left: FloatND, right: FloatND) -> BoolND:
    """Same geometric location: signed zeros agree, distinct subnormals do not.

    Compare the raw encodings, not two normalized sortable keys. A compiler can
    recognize the latter as a floating equality and reintroduce flushing under
    JIT. The only additional equality here is the explicit two-zero bit case.
    """
    left_bits, left_magnitude, _, left_valid = _stored_parts(value=left)
    right_bits, right_magnitude, _, right_valid = _stored_parts(value=right)
    return (
        ((left_bits == right_bits) | ((left_magnitude | right_magnitude) == 0))
        & left_valid
        & right_valid
    )


def _stored_less(*, left: FloatND, right: FloatND) -> BoolND:
    """Strict sign/magnitude order, with signed zeros equal and NaNs unordered."""
    _, left_magnitude, left_negative, left_valid = _stored_parts(value=left)
    _, right_magnitude, right_negative, right_valid = _stored_parts(value=right)
    ordered = jnp.where(
        left_negative != right_negative,
        left_negative,
        jnp.where(
            left_negative,
            left_magnitude > right_magnitude,
            left_magnitude < right_magnitude,
        ),
    )
    return (
        ordered & ((left_magnitude | right_magnitude) != 0) & left_valid & right_valid
    )


def _stored_in_span(*, query: FloatND, lower: FloatND, upper: FloatND) -> BoolND:
    """Admit finite queries to live finite spans using the node-identity order."""
    return (
        jnp.isfinite(query)
        & ~_stored_less(left=query, right=lower)
        & ~_stored_less(left=upper, right=query)
    )


def _same_bits(*, left: FloatND, right: FloatND) -> BoolND:
    """Channel identity, unlike geometry, preserves the sign of a stored zero."""
    integer = jnp.uint64 if left.dtype == jnp.float64 else jnp.uint32
    return jax.lax.bitcast_convert_type(left, integer) == jax.lax.bitcast_convert_type(
        right, integer
    )


def _chord_reading(
    *, x: FloatND, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND
) -> tuple[FloatND, IntND]:
    """Read one selected channel with an explicit native publication status.

    No rounded slope or floating endpoint-weight product is formed. Status zero
    certifies a single nearest rounding; an invalid or overflowing read returns
    NaN and its nonzero status. Restoring stored endpoints/common levels after
    that check also preserves signed zeros, which rational arithmetic alone does
    not distinguish. These shortcuts must never override a failed status.
    """
    reading, status = exact_affine_read(x0=x0, x1=x1, v0=v0, v1=v1, x_query=x)
    reading = jnp.where(
        _stored_equal(left=x, right=x0),
        v0,
        jnp.where(
            _stored_equal(left=x, right=x1),
            v1,
            jnp.where(_same_bits(left=v0, right=v1), v0, reading),
        ),
    )
    return jnp.where(status == 0, reading, jnp.nan), status


def _chord_upper_value(
    *, x: FloatND, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND
) -> FloatND:
    """Publish the least working-format value at or above the exact chord.

    Compare the exact chord with the constant line at its nearest reading. Only
    a strict positive sign needs the next float toward +inf: exact endpoints,
    common levels, and readings already rounded upward remain untouched. The
    comparison is native/exact, including subnormals and negative values. A
    refused comparison or unrepresentable upper bound stays explicitly NaN.
    """
    reading = _chord_value(x=x, x0=x0, x1=x1, v0=v0, v1=v1)
    sign = certified_margin_sign(
        a_x0=x0,
        a_x1=x1,
        a_v0=v0,
        a_v1=v1,
        b_x0=jnp.zeros_like(reading),
        b_x1=jnp.ones_like(reading),
        b_v0=reading,
        b_v1=reading,
        x_query=x,
    )
    upper = jnp.where(
        sign == 1, jnp.nextafter(reading, jnp.full_like(reading, jnp.inf)), reading
    )
    return jnp.where((sign >= -1) & (sign <= 1) & jnp.isfinite(upper), upper, jnp.nan)


def _certified_owner(
    *,
    brackets: BoolND,
    links: _Links,
    query: FloatND,
    stable_index: IntND,
) -> tuple[Int1D, BoolND]:
    """Return the column of the link that owns each query, and whether it is exact.

    Ownership is one complete reduction over the stored operands: among the links
    a query admits, the owner is the greatest by exact affine value, then by
    reaching strictly right of the query, then by exact value slope, then by the
    earliest stable identity. No candidate value is rounded before the winner is
    chosen, so a query at which several links' values fall in one rounding bin is
    ordered by what the operands say rather than by what the reading shows.

    A node where two branches meet is therefore owned by the branch that owns the
    interval above it, so the switch is published at the node the geometry puts it
    at.

    Admission and comparison are kept apart. Which links a query admits is decided
    by the stored span. The selector also receives that ORIGINAL span, not the
    comparable line: otherwise a singleton would acquire artificial right
    extension in the tie order even with a correct admission mask.

    Returns:
        Tuple of the owning column per query and whether that query's order was
        resolved. A query whose comparison the kernel cannot decide is reported
        unresolved rather than settled by a rounded reading.
    """
    shape = brackets.shape
    winner, exact_status = exact_query_winner_batched(
        left_grid=jnp.broadcast_to(links.lower[None, :], shape),
        right_grid=jnp.broadcast_to(links.upper[None, :], shape),
        left_value=jnp.broadcast_to(links.v0[None, :], shape),
        right_value=jnp.broadcast_to(links.upper_value[None, :], shape),
        live=brackets,
        stable_index=stable_index,
        x_query=query,
    )
    owner = winner.reshape(-1).astype(jnp.int32)
    resolved = exact_status.reshape(-1) == 0
    return owner, resolved


def _evaluate_envelope(
    *, query_grid: Float1D, links: _Links, segment_id: Int1D
) -> tuple[Float1D, Float1D, Int1D, Int1D]:
    """Evaluate the upper envelope and its owning link/branch at every query.

    Builds the dense `(n_query, n_link)` block: each query is tested against
    every link, a link admits the query iff the query lies in its stored span,
    and the owner among the admitting links is certified rather than read off
    the values. The published value and policy are both the owner's own chord at
    the query, so they always name one branch. A query no link admits reports
    `-inf` value (the absent-envelope sentinel), owning link `0`, and branch
    `-1`.

    Args:
        query_grid: Abscissae at which to evaluate the envelope; NaN tail.
        links: The candidate links as comparable lines plus their stored spans.
        segment_id: Per-link monotone-branch id; equal across one branch.

    Returns:
        Tuple of the envelope value, the envelope policy, the owning link index
        (`0` where no link admits the query), and the owning branch id (`-1`
        where no link admits it) at each query.

    """
    query = query_grid[:, None]
    brackets = links.live[None, :] & _stored_in_span(
        query=query, lower=links.lower, upper=links.upper
    )
    stable_index = jnp.broadcast_to(
        jnp.arange(links.x0.shape[0], dtype=jnp.int32)[None, :], brackets.shape
    )
    owner, resolved = _certified_owner(
        brackets=brackets,
        links=links,
        query=query,
        stable_index=stable_index,
    )
    # Gather first: publication costs one read per selected channel/query, not
    # one exact read per candidate in the dense admission block.
    value, value_status = _chord_reading(
        x=query_grid,
        x0=links.x0[owner],
        x1=links.x1[owner],
        v0=links.v0[owner],
        v1=links.v1[owner],
    )
    policy, policy_status = _chord_reading(
        x=query_grid,
        x0=links.x0[owner],
        x1=links.x1[owner],
        v0=links.p0[owner],
        v1=links.p1[owner],
    )
    any_bracket = jnp.any(brackets, axis=1)
    published = resolved & (value_status == 0) & (policy_status == 0)
    # No support is -inf (drop the node); admitted but unreadable is NaN (keep
    # the owner and poison both channels). A failed read never selects a rival.
    envelope_value = jnp.where(
        any_bracket, jnp.where(published, value, jnp.nan), -jnp.inf
    )
    envelope_policy = jnp.where(any_bracket & published, policy, jnp.nan)
    winner_link = jnp.where(any_bracket, owner, 0).astype(jnp.int32)
    winner_segment = jnp.where(any_bracket, segment_id[owner], -1).astype(jnp.int32)
    return envelope_value, envelope_policy, winner_link, winner_segment


@dataclass(frozen=True, kw_only=True)
class _CrossingBlocks:
    """Per-query crossing candidate: abscissa, value, policies, per-record flags.

    The fields are aligned with the query sweep: entry `i` is the crossing
    inserted around query node `i`, present only when query `i`'s live owning
    branch differs from the previous live query's and the two chords' gap
    changes sign between the two abscissae.
    """

    grid: Float1D
    """Crossing abscissa per query."""
    value: Float1D
    """Envelope value at the crossing."""
    policy_left: Float1D
    """Policy of the outgoing owner at the crossing."""
    policy_right: Float1D
    """Policy of the incoming owner at the crossing."""
    left_valid: BoolND
    """Whether the outgoing owner's record is emitted."""
    right_valid: BoolND
    """Whether the incoming owner's record is emitted."""
    segment_left: Int1D
    """Branch id of the outgoing owner."""
    segment_right: Int1D
    """Branch id of the incoming owner."""
    unresolved: BoolND
    """Whether a switched pair's location could not be certified."""


def _crossing_blocks(
    *,
    query_grid: Float1D,
    query_drop: BoolND,
    winner_link: Int1D,
    winner_segment: Int1D,
    links: _Links,
) -> _CrossingBlocks:
    """Compute, per query, the crossing of its owner with the previous owner.

    Sweeps the live queries left-to-right (carrying the previous live query's
    owning link, branch id and abscissa) and, whenever the owning *branch*
    switches, locates the two chords' crossing inside the interval the switch
    happened in. A move from one link to the next within one monotone branch
    keeps the segment id, so it is not a switch and inserts nothing; only a
    genuine branch change is a kink.
    """
    n_query = query_grid.shape[0]
    step = functools.partial(
        _crossing_step,
        live=~query_drop,
        winner_link=winner_link,
        winner_segment=winner_segment,
        query_grid=query_grid,
        links=links,
    )
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
        left_valid=rows.left_valid,
        right_valid=rows.right_valid,
        segment_left=rows.segment_left,
        segment_right=rows.segment_right,
        unresolved=rows.unresolved,
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
    left_valid: BoolND
    """Whether this step emits the outgoing owner's record."""
    right_valid: BoolND
    """Whether this step emits the incoming owner's record."""
    segment_left: IntND
    """Branch id of the outgoing owner."""
    segment_right: IntND
    """Branch id of the incoming owner."""
    unresolved: BoolND
    """Whether this step must poison its query's publication."""


_CROSSING_ROW_FIELDS = (
    "grid",
    "value",
    "policy_left",
    "policy_right",
    "left_valid",
    "right_valid",
    "segment_left",
    "segment_right",
    "unresolved",
)


def _flatten_crossing_row(row: _CrossingRow) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(row, name) for name in _CROSSING_ROW_FIELDS), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
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


# keyword-only-exempt: library-callback=jax.lax.scan
def _crossing_step(
    carry: tuple[ScalarInt, ScalarInt, FloatND],
    idx: ScalarInt,
    *,
    live: BoolND,
    winner_link: Int1D,
    winner_segment: Int1D,
    query_grid: Float1D,
    links: _Links,
) -> tuple[tuple[ScalarInt, ScalarInt, FloatND], _CrossingRow]:
    """Emit the crossing query `idx` opens, and advance the previous-owner carry.

    The carry is the previous live query's owning link, its branch id, and its
    abscissa. A crossing is emitted where the owning branch switches and the two
    owners' chords cross inside the interval between the two abscissae. Where
    the crossing lands on one of the two nodes, that node's own row already
    carries one of the two owners and only the other record is emitted.
    """
    prev_link, prev_segment, prev_grid = carry
    is_live = live[idx]
    this_link = winner_link[idx]
    this_segment = winner_segment[idx]
    this_grid = query_grid[idx]

    switches = is_live & (prev_segment >= 0) & (this_segment != prev_segment)
    row = _crossing_in_interval(
        seg_a=prev_link,
        seg_b=this_link,
        prev_grid=prev_grid,
        this_grid=this_grid,
        links=links,
    )
    valid = switches & row.resolved
    emitted = _CrossingRow(
        grid=row.grid,
        value=row.value,
        policy_left=row.policy_a,
        policy_right=row.policy_b,
        left_valid=valid & ~row.at_left,
        right_valid=valid & ~row.at_right,
        segment_left=prev_segment,
        segment_right=this_segment,
        unresolved=switches & row.unresolved,
    )

    # Advance the previous-live-query carry only on a live query; a dropped
    # query leaves the comparison anchored at the last live owner/abscissa.
    new_link = jnp.where(is_live, this_link, prev_link).astype(jnp.int32)
    new_segment = jnp.where(is_live, this_segment, prev_segment).astype(jnp.int32)
    new_grid = jnp.where(is_live, this_grid, prev_grid)
    return (new_link, new_segment, new_grid), emitted


@dataclass(frozen=True, kw_only=True)
class _SegmentIntersection:
    """Where two owning chords cross inside one interval, with both policies."""

    grid: FloatND
    """Abscissa where the two chords meet."""
    value: FloatND
    """Envelope value of the two chords there."""
    policy_a: FloatND
    """Policy of the outgoing owner at the crossing."""
    policy_b: FloatND
    """Policy of the incoming owner at the crossing."""
    resolved: BoolND
    """Whether the two chords in fact cross inside the interval."""
    at_left: BoolND
    """Whether the emitted handover state coincides with the left query."""
    at_right: BoolND
    """Whether the emitted handover state coincides with the right query."""
    unresolved: BoolND
    """Whether the signs or a bracketed event's location were refused."""


def _crossing_in_interval(
    *,
    seg_a: ScalarInt,
    seg_b: ScalarInt,
    prev_grid: FloatND,
    this_grid: FloatND,
    links: _Links,
) -> _SegmentIntersection:
    """Locate where chords `seg_a` and `seg_b` cross between the two abscissae.

    *Whether* they cross is certified: the comparator settles the sign of their
    difference at each of the two abscissae from the stored operands, and a
    crossing exists exactly where the outgoing chord is at or above the incoming
    one at the left abscissa and at or below it at the right. Reading those two
    signs off a floating evaluation instead would put the existence of the
    crossing back at the mercy of the last bit, which is what leaves a switch
    unpublished; a gap that does not change sign is a switch that happened
    because a branch started or stopped covering the interval, not because the
    two met.

    Oriented endpoint equality is a node crossing; equality at both ends is
    collinearity, not an event. For an interior crossing, the native handover
    primitive forms the exact cross-multiplied affine difference from the same
    stored operands as the signs. It publishes the least representable state
    at or above the root, with no rounded chord subtraction or denominator
    fallback. Coalescence is determined from this emitted state, not from the
    endpoint signs: even a strictly interior root can hand over at the right
    node.

    The outgoing link's own stored span, rather than its widened comparable line,
    admits the event. The incoming link's span does not: a branch entered at the
    right node is represented there by the link that reaches beyond it, so a root
    inside the interval lies before that link's support by construction, and
    requiring it to be covered would drop every handover aligned with a node.
    """
    a_x0, a_x1 = links.x0[seg_a], links.x1[seg_a]
    a_v0, a_v1 = links.v0[seg_a], links.v1[seg_a]
    a_p0, a_p1 = links.p0[seg_a], links.p1[seg_a]
    b_x0, b_x1 = links.x0[seg_b], links.x1[seg_b]
    b_v0, b_v1 = links.v0[seg_b], links.v1[seg_b]
    b_p0, b_p1 = links.p0[seg_b], links.p1[seg_b]

    chords = {
        "a_x0": a_x0,
        "a_x1": a_x1,
        "a_v0": a_v0,
        "a_v1": a_v1,
        "b_x0": b_x0,
        "b_x1": b_x1,
        "b_v0": b_v0,
        "b_v1": b_v1,
    }
    sign_prev = certified_margin_sign(x_query=prev_grid, **chords)
    sign_this = certified_margin_sign(x_query=this_grid, **chords)
    at_left_root = (sign_prev == 0) & (sign_this == -1)
    at_right_root = (sign_prev == 1) & (sign_this == 0)
    crosses_inside = (sign_prev == 1) & (sign_this == -1)
    bracketed = at_left_root | at_right_root | crosses_inside
    handover, location_status = exact_affine_handover(
        left=prev_grid, right=this_grid, **chords
    )
    grid = jnp.where(bracketed & (location_status == 0), handover, jnp.nan)
    covered = (
        links.live[seg_a]
        & links.live[seg_b]
        & _stored_in_span(
            query=grid, lower=links.lower[seg_a], upper=links.upper[seg_a]
        )
    )
    resolved = bracketed & (location_status == 0) & covered
    at_left = resolved & _stored_equal(left=grid, right=prev_grid)
    at_right = resolved & _stored_equal(left=grid, right=this_grid)
    unresolved = (
        (sign_prev == UNRESOLVED_STATUS)
        | (sign_this == UNRESOLVED_STATUS)
        | (bracketed & (location_status != 0))
    )

    # Select the higher exact chord at the *emitted* abscissa before reading it.
    # The handover can lie above the root, so max(two nearest readings) is not an
    # upper certificate. One directed read of the higher chord bounds both.
    order = certified_margin_sign(x_query=grid, **chords)
    take_b = order == -1
    value = _chord_upper_value(
        x=grid,
        x0=jnp.where(take_b, b_x0, a_x0),
        x1=jnp.where(take_b, b_x1, a_x1),
        v0=jnp.where(take_b, b_v0, a_v0),
        v1=jnp.where(take_b, b_v1, a_v1),
    )
    policy_a, status_a = _chord_reading(x=grid, x0=a_x0, x1=a_x1, v0=a_p0, v1=a_p1)
    policy_b, status_b = _chord_reading(x=grid, x0=b_x0, x1=b_x1, v0=b_p0, v1=b_p1)
    published = (
        (order >= -1)
        & (order <= 1)
        & jnp.isfinite(value)
        & (status_a == 0)
        & (status_b == 0)
    )
    return _SegmentIntersection(
        grid=grid,
        value=jnp.where(published, value, jnp.nan),
        policy_a=jnp.where(published, policy_a, jnp.nan),
        policy_b=jnp.where(published, policy_b, jnp.nan),
        # Geometry is separate from reading: an unresolved ordinate/policy is a
        # NaN event, not a reason to silently omit a genuine branch switch.
        resolved=resolved,
        at_left=at_left,
        at_right=at_right,
        unresolved=unresolved,
    )


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
