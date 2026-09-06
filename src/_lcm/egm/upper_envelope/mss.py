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

- **Ownership is certified.** A provisional winner is read off the values and
  then challenged: `certified_sign` decides the sign of each bracketing link's
  value less the standing winner's from the stored operands, in exact integer
  arithmetic. Links certified level with the winner are separated by a
  right-continuous rule — the link that extends strictly right of the query,
  then the steeper one, then the earlier stored link — so the owner at a node
  where two branches meet is the one that owns the interval above it, and the
  switch is visible at that node rather than one interval later.
- **A value is read from its own chord's endpoints.** Each link is evaluated at
  a query by weighing its two stored endpoints against their distances to the
  query, with the products carried at twice the working precision. The reading
  is the stored value at either endpoint exactly, and elsewhere it is the
  chord's own value rather than a line extrapolated from one far anchor.
- **A crossing is located inside the interval it happened in.** The two winning
  chords' gap is evaluated at the two adjacent query abscissae; a crossing
  exists exactly where that gap changes sign across them, and its abscissa is
  the root of the gap's own secant between them.

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

from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from _lcm.egm.upper_envelope.double_double import (
    dd_add,
    dd_from_difference,
    dd_mul_float,
    dd_quotient,
)
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarInt

# How many times a certified challenger may replace the standing winner. The
# provisional winner is already the highest reading, so a challenger can only be
# a link the reading ordered wrongly; one promotion settles that, and the second
# is the margin for a challenger that is itself displaced.
_PROMOTION_ROUNDS = 2


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
    grid_key = jnp.where(dead, jnp.inf, endog_grid)
    order = jnp.argsort(grid_key)
    query_grid = jnp.where(dead, jnp.nan, endog_grid)[order]
    query_dead = dead[order]
    # Several candidate branches can supply the same query abscissa. They still
    # all participate as links, but the sweep publishes the owning node once.
    # Otherwise a node-aligned switch emits its outgoing record beside several
    # identical incoming-node records instead of one outgoing/incoming pair.
    repeated_query = jnp.concatenate(
        (jnp.zeros((1,), dtype=bool), query_grid[1:] == query_grid[:-1])
    )
    query_dead = query_dead | repeated_query

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

    links = _comparable_links(
        left_grid=left_grid,
        right_grid=right_grid,
        left_policy=left_policy,
        right_policy=right_policy,
        left_value=left_value,
        right_value=right_value,
        segment_live=segment_live,
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
    crossing_value, _, _, crossing_owner = _evaluate_envelope(
        query_grid=crossing.grid, links=links, segment_id=link_segment
    )
    on_envelope = jnp.isfinite(crossing_value) & (
        (crossing_owner == crossing.segment_left)
        | (crossing_owner == crossing.segment_right)
    )
    left_valid = crossing.left_valid & on_envelope
    right_valid = crossing.right_valid & on_envelope

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
    rather than false. The comparable fields are shaped once so every comparison
    downstream is handed operands it can decide on, while `lower`/`upper` keep
    the stored span, so widening a degenerate link's divisor never widens the
    set of queries it brackets.
    """

    lower: Float1D
    """Lower stored abscissa of the link's span."""
    upper: Float1D
    """Upper stored abscissa of the link's span."""
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

    A link stored right-to-left carries the same line as the same link stored
    left-to-right, so the endpoints are swapped rather than rejected. A link of
    zero width carries no line: its divisor is displaced by one representable
    step and both endpoint readings are set to the stored lower ones, which is
    the flat line it in fact is. One representable step is a readable width
    everywhere but at zero, where it is the smallest subnormal and a comparison
    would abstain, so a flat link at zero takes a width of one instead.
    """
    descending = right_grid < left_grid
    x0 = jnp.where(descending, right_grid, left_grid)
    x1 = jnp.where(descending, left_grid, right_grid)
    v0 = jnp.where(descending, right_value, left_value)
    v1 = jnp.where(descending, left_value, right_value)
    p0 = jnp.where(descending, right_policy, left_policy)
    p1 = jnp.where(descending, left_policy, right_policy)

    degenerate = x1 <= x0
    flat_at_zero = degenerate & (x0 == 0.0)
    step = jnp.nextafter(x0, jnp.full_like(x0, jnp.inf))
    finite_line = jnp.isfinite(v0) & jnp.isfinite(v1) & jnp.isfinite(x0)
    return _Links(
        lower=jnp.minimum(left_grid, right_grid),
        upper=jnp.maximum(left_grid, right_grid),
        x0=x0,
        x1=jnp.where(flat_at_zero, jnp.ones_like(x0), jnp.where(degenerate, step, x1)),
        v0=v0,
        v1=jnp.where(degenerate, v0, v1),
        p0=p0,
        p1=jnp.where(degenerate, p0, p1),
        live=segment_live & finite_line,
    )


def _chord_value(
    *, x: FloatND, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND
) -> FloatND:
    """Evaluate the chord through two stored endpoints at `x`.

    Each endpoint is weighed by its distance to the *other* one, so the reading
    has no anchor: a query far from `x0` is not reached by extrapolating a slope
    from it, and the endpoints' own magnitudes cancel in the numerator rather
    than in the answer. Both distances and the width are exact differences and
    the two products are carried at twice the working precision, so only the
    division rounds. At either stored endpoint the reading is that endpoint's
    stored value exactly.
    """
    left_weight = dd_from_difference(x1, x)
    right_weight = dd_from_difference(x, x0)
    numerator = dd_add(dd_mul_float(left_weight, v0), dd_mul_float(right_weight, v1))
    high, _low = dd_quotient(numerator, dd_from_difference(x1, x0))
    return jnp.where(x == x0, v0, jnp.where(x == x1, v1, high))


def _chord_reading(
    *, x: FloatND, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND
) -> FloatND:
    """Read a chord at `x` in the working precision, exactly at both endpoints.

    Carries a quantity no ordering is decided on — the policy — so it costs one
    rounded evaluation rather than the compensated one `_chord_value` pays for.
    Both endpoints are returned as stored, and a chord whose two endpoints carry
    the same value reads back that value at every abscissa, so a branch of
    constant policy publishes one policy rather than a spread of neighbours.
    """
    width = x1 - x0
    interior = v0 + (x - x0) / width * (v1 - v0)
    return jnp.where(x == x0, v0, jnp.where(x == x1, v1, interior))


def _certified_owner(
    *,
    brackets: BoolND,
    value: FloatND,
    links: _Links,
    query: FloatND,
    stable_index: IntND,
) -> Int1D:
    """Return the column of the link that owns each query.

    The highest reading is the provisional owner and is then challenged: any
    bracketing link certified strictly above it takes its place, twice over. The
    remaining question is which of the links certified *level* with the owner
    publishes the query, and that is settled right-continuously — the link
    reaching strictly right of the query, then the steeper one, then the earlier
    stored link. A node where two branches meet is therefore owned by the branch
    that owns the interval above it, so the switch is published at the node the
    geometry puts it at.

    Two things bound what that settles. The promotion budget is finite, so a
    query at which more links are mis-ordered by the reading than there are
    rounds can still publish a link another is certified above; a link can only
    be mis-ordered by the reading when the two sit within a few representable
    steps of each other, so the residual is of the same order as the published
    value's own. And the comparator refuses rather than guesses on operands it
    cannot decide, while this method has no channel through which to refuse in
    turn: a query whose comparisons all come back refused keeps the highest
    reading, which is the answer the method gave before any of them were
    certified.
    """
    provisional = _leader(brackets=brackets, value=value)
    reference = _take_link(links=links, index=provisional)
    for _ in range(_PROMOTION_ROUNDS):
        beats = brackets & (
            _sign_against(links=links, reference=reference, x=query) == 1
        )
        challenger = _take_link(links=links, index=_leader(brackets=beats, value=value))
        promoted = jnp.any(beats, axis=1, keepdims=True)
        reference = tuple(
            jnp.where(promoted, new, held)
            for new, held in zip(challenger, reference, strict=True)
        )

    level = brackets & (_sign_against(links=links, reference=reference, x=query) == 0)
    excluded = jnp.full_like(value, -jnp.inf)
    slope = _slope(x_a=links.x0, y_a=links.v0, x_b=links.x1, y_b=links.v1)
    reaches_right = (links.upper > query).astype(value.dtype)
    ordered = (
        jnp.where(level, reaches_right, excluded),
        jnp.where(level, jnp.broadcast_to(slope, value.shape), excluded),
    )

    still_tied = level
    for field in ordered:
        best = jnp.max(jnp.where(still_tied, field, -jnp.inf), axis=1, keepdims=True)
        still_tied = still_tied & (field == best)
    sentinel = jnp.iinfo(jnp.int32).max
    index_key = jnp.where(still_tied, stable_index, sentinel)
    earliest = jnp.min(index_key, axis=1, keepdims=True)
    settled = still_tied & (index_key == earliest)
    chosen = jnp.argmax(settled, axis=1).astype(jnp.int32)
    return jnp.where(jnp.any(settled, axis=1), chosen, provisional[:, 0]).astype(
        jnp.int32
    )


def _leader(*, brackets: BoolND, value: FloatND) -> IntND:
    """Column of the highest reading among the admitted links, per query."""
    index = jnp.argmax(jnp.where(brackets, value, -jnp.inf), axis=1)
    return index[:, None].astype(jnp.int32)


def _take_link(*, links: _Links, index: IntND) -> tuple[FloatND, ...]:
    """The comparable line of one column per query, as `(x0, x1, v0, v1)`."""
    return tuple(
        jnp.take(field, index, axis=0)
        for field in (links.x0, links.x1, links.v0, links.v1)
    )


def _sign_against(
    *, links: _Links, reference: tuple[FloatND, ...], x: FloatND
) -> IntND:
    """Certified sign of each link's value at `x` less the reference link's."""
    reference_x0, reference_x1, reference_v0, reference_v1 = reference
    return certified_margin_sign(
        a_x0=links.x0[None, :],
        a_x1=links.x1[None, :],
        a_v0=links.v0[None, :],
        a_v1=links.v1[None, :],
        b_x0=reference_x0,
        b_x1=reference_x1,
        b_v0=reference_v0,
        b_v1=reference_v1,
        x_query=x,
    )


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
    brackets = links.live[None, :] & (query >= links.lower) & (query <= links.upper)
    value = _chord_value(x=query, x0=links.x0, x1=links.x1, v0=links.v0, v1=links.v1)
    admits = brackets & jnp.isfinite(value)
    stable_index = jnp.broadcast_to(
        jnp.arange(links.x0.shape[0], dtype=jnp.int32)[None, :], value.shape
    )

    owner = _certified_owner(
        brackets=admits,
        value=value,
        links=links,
        query=query,
        stable_index=stable_index,
    )
    policy = _chord_reading(x=query, x0=links.x0, x1=links.x1, v0=links.p0, v1=links.p1)
    any_bracket = jnp.any(admits, axis=1)
    column = owner[:, None]
    envelope_value = jnp.where(
        any_bracket, jnp.take_along_axis(value, column, axis=1)[:, 0], -jnp.inf
    )
    envelope_policy = jnp.take_along_axis(policy, column, axis=1)[:, 0]
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


_CROSSING_ROW_FIELDS = (
    "grid",
    "value",
    "policy_left",
    "policy_right",
    "left_valid",
    "right_valid",
    "segment_left",
    "segment_right",
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
    """Whether the crossing sits exactly on the left query abscissa."""
    at_right: BoolND
    """Whether the crossing sits exactly on the right query abscissa."""


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

    A certified zero at one of the two abscissae is a crossing sitting exactly on
    that node, and the emitted abscissa is that node exactly. Only the interior
    case is located by arithmetic, and it is located inside the interval the
    switch was observed in — the root of the gap's own secant across it, clamped
    to the interval the certified signs bracket it in — rather than by
    extrapolating either chord from its stored endpoints.
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
    at_left = sign_prev == 0
    at_right = (sign_this == 0) & ~at_left
    crosses_inside = (sign_prev == 1) & (sign_this == -1)
    resolved = at_left | at_right | crosses_inside

    gap_prev = _chord_gap(x=prev_grid, **chords)
    gap_this = _chord_gap(x=this_grid, **chords)
    denominator = gap_prev - gap_this
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    interior = jnp.clip(
        (gap_prev * this_grid - gap_this * prev_grid) / safe_denominator,
        prev_grid,
        this_grid,
    )
    grid = jnp.where(at_left, prev_grid, jnp.where(at_right, this_grid, interior))

    value_a = _chord_value(x=grid, x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1)
    value_b = _chord_value(x=grid, x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1)
    return _SegmentIntersection(
        grid=grid,
        # The emitted abscissa is a rounding away from the exact root, so the
        # two chords no longer agree there to the last bit. Publishing the
        # higher of the two keeps the emitted row on the envelope: a value below
        # both branches would lose the node to any competitor between them.
        value=jnp.maximum(value_a, value_b),
        policy_a=_chord_reading(x=grid, x0=a_x0, x1=a_x1, v0=a_p0, v1=a_p1),
        policy_b=_chord_reading(x=grid, x0=b_x0, x1=b_x1, v0=b_p0, v1=b_p1),
        resolved=resolved,
        at_left=at_left,
        at_right=at_right,
    )


def _chord_gap(
    *,
    x: FloatND,
    a_x0: FloatND,
    a_x1: FloatND,
    a_v0: FloatND,
    a_v1: FloatND,
    b_x0: FloatND,
    b_x1: FloatND,
    b_v0: FloatND,
    b_v1: FloatND,
) -> FloatND:
    """How far the first chord sits above the second at `x`."""
    return _chord_value(x=x, x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1) - _chord_value(
        x=x, x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1
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
