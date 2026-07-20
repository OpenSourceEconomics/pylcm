"""The NEGM cash-on-hand outer envelope: stacked candidates, query-side max.

The NEGM kernel collapses the outer durable choice by `V = max_j W_j` on the
exogenous state grid, but the continuation it threads to the parent period is a
carry — value and marginal-utility rows on an endogenous (resources) grid. The
keeper and every adjuster outer-grid node produce one such candidate row per
durable state, each in its *own* resources space: the adjuster `j` pays a
credited durable cost, so its resources are the keeper's cash-on-hand shifted
down by `credited(z, z'_j)`.

The envelope is exact-to-grid only if the parent reads `max_j V_j(q)` at *every*
query `q` — interpolating a row that was already maximized on a node grid
overstates the envelope near a branch crossing (an aggregate-bridge error). So
the carry retains all candidates and the maximum is taken at the read:

- `build_stacked_outer_carry` lifts every candidate into a common cash-on-hand
  (coh) axis by adding back its credited cost (`∂coh/∂R = 1`, so the value and
  the resource-marginal carry over unchanged) and stacks the lifted candidates
  verbatim on a candidate axis just before the grid axis — no maximum is taken
  at build time.
- `outer_envelope_at_query` reads each candidate at the query through the
  parent's own interpolation convention and takes the pointwise maximum there,
  publishing the winning candidate's marginal (Danskin) — never an average
  across a crossing. Exact value ties resolve right-continuously by the value
  germ, publishing the winner's economic marginal.

The production read path (`continuation._collapse_stacked_candidates`) is the
authoritative implementation and follows the same right-continuous tie
convention; `outer_envelope_at_query` is a self-contained reference for the
query-side max, not a byte-for-byte oracle of the production aggregation.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import (
    interp_left_germ_on_padded_grid,
    interp_left_record_on_padded_grid,
    interp_on_padded_grid,
    interp_right_germ_on_padded_grid,
)
from lcm.typing import Bool1D, BoolND, Float1D, FloatND, IntND


def build_stacked_outer_carry(
    *,
    keeper_carry: EGMCarry,
    adjuster_carries: tuple[EGMCarry, ...],
    coh_shifts: FloatND,
) -> EGMCarry:
    """Stack the outer durable candidates into a common-coh candidate axis.

    Every candidate is lifted into a common cash-on-hand axis by adding back its
    credited cost (`∂coh/∂R = 1`, so value and resource-marginal transfer
    unchanged), then stacked on a new candidate axis inserted just before the
    grid axis. Unlike a node-sampled merge this takes no maximum at build time:
    the A+1 conditional carries are retained verbatim so the parent can take the
    exact `max_j V_j(q)` at its own query (`outer_envelope_at_query`). The keeper
    is candidate 0 (zero shift, `credited(z, z) = 0`); adjuster `j` is candidate
    `j + 1`, lifted by `coh_shifts[:, j]` per durable state. While the stack is
    assembled the unstacked candidate arrays and the stacked output coexist, a
    transient on top of the resident stacked carry itself.

    Args:
        keeper_carry: The keeper's continuation carry — one row per leading cell
            `(discrete..., durable)`, already in coh space.
        adjuster_carries: One carry per outer-grid node, each in its own resources
            space; aligned with the columns of `coh_shifts`.
        coh_shifts: Per durable state (rows) and adjuster node (columns), the
            credited cost added to that adjuster's endogenous grid to map it into
            coh space. Shape `(n_durable, n_adjusters)`.

    Returns:
        A carry whose leading shape is `(discrete..., durable, n_candidates)` and
        whose trailing axis is the grid: `carry[cell]` is the `(n_candidates,
        n_pad)` block `outer_envelope_at_query` consumes.

    """
    leading_shape = keeper_carry.endog_grid.shape[:-1]
    n_durable = leading_shape[-1]
    # The durable margin is the last leading axis; broadcast a per-durable shift
    # over the leading discrete axes and the grid axis.
    broadcast = (1,) * (len(leading_shape) - 1) + (n_durable, 1)

    lifted_endog = [keeper_carry.endog_grid]
    for adjuster_index, adjuster_carry in enumerate(adjuster_carries):
        shift = coh_shifts[:, adjuster_index].reshape(broadcast)
        lifted_endog.append(adjuster_carry.endog_grid + shift)

    candidates = (keeper_carry, *adjuster_carries)
    return EGMCarry(
        endog_grid=jnp.stack(lifted_endog, axis=-2),
        value=jnp.stack([c.value for c in candidates], axis=-2),
        marginal_utility=jnp.stack([c.marginal_utility for c in candidates], axis=-2),
        taste_shock_scale=keeper_carry.taste_shock_scale,
    )


def outer_envelope_at_query(
    *,
    candidate_endog: FloatND,
    candidate_value: FloatND,
    candidate_marginal: FloatND,
    x_query: Float1D,
) -> tuple[Float1D, Float1D]:
    """Pointwise upper envelope `max_j V_j(q)` of the lifted outer candidates.

    Each candidate row is read at every query through the parent's own
    interpolation convention — edge-clamped Fritsch-Carlson-limited cubic Hermite
    value with the marginal row as node slopes, a separate linear marginal read,
    and the value masked to `-inf` below the candidate's first finite coh node
    (its borrowing-constrained support has not started, and its marginal is
    zeroed alongside, so an all-infeasible query publishes the `(-inf, 0)`
    infeasible pair). Strictly above a candidate's own last finite node the
    value read clamps to a constant, so both marginal payloads are re-pinned to
    zero there per candidate — a winner queried past its own support publishes
    the locally constant envelope's zero slope, never its terminal record.
    The published value is the pointwise maximum over
    candidates; the published marginal is the *winning* candidate's resource
    slope (Danskin), so it is winner-consistent and never averaged across a
    branch crossing. At an exact value tie the winner is right-continuous in
    the value read itself: the tied candidates are compared by the complete
    right germ of their own value interpolants (`right_germ_winner`) — each
    local piece is a limited cubic Hermite or a constant clamp, so
    right-finiteness plus the first three one-sided derivatives determine the
    read on a right neighborhood exactly, and the branch whose read actually
    wins immediately to the right of the query owns the published (economic)
    marginal, matching the one-sided convention the parent's Euler inversion
    expects. Candidates whose right germs coincide — at a shared terminal
    abscissa every candidate clamps, so the right germ cannot discriminate —
    are compared by their *left* germs: the branch that carries the envelope
    on the left neighborhood wins ownership. Only candidates whose local pieces
    literally coincide on both sides fall back to the lowest index, a
    deterministic choice among identical branches. The germ decides ownership;
    the published payload is the winner's economic marginal — the two stay
    separate objects. On left-owned cells that payload is the winner's
    *left-record* marginal (`interp_left_record_on_padded_grid`), so a winner
    whose terminal abscissa is duplicated publishes the left duplicate's record
    — the one that justified ownership — not the right one.

    Taking the maximum at the query — rather than at a shared node grid and
    republishing a single interpolated row — is exact for the finite candidate set
    at every query: a candidate that wins only on an interval strictly between two
    nodes is read at its true value there instead of being bridged upward by a
    shared-node reinterpolation.

    Args:
        candidate_endog: Lifted common-coh grids, `(n_candidates, n_pad)`, each
            NaN-padded in the tail.
        candidate_value: Conditional value rows, `(n_candidates, n_pad)`.
        candidate_marginal: Conditional resource-marginal rows,
            `(n_candidates, n_pad)`.
        x_query: Query cash-on-hand points, `(n_query,)`.

    Returns:
        Tuple of the envelope value and the winner's marginal, each `(n_query,)`.

    """

    def read_one(
        endog: Float1D, value: Float1D, marginal: Float1D
    ) -> tuple[
        Float1D,
        Float1D,
        Float1D,
        tuple[Bool1D, Float1D, Float1D, Float1D],
        tuple[Bool1D, Float1D, Float1D, Float1D],
    ]:
        cand_lower = jnp.min(jnp.where(jnp.isfinite(endog), endog, jnp.inf))
        value_at_query = interp_on_padded_grid(
            x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
        )
        marginal_at_query = interp_on_padded_grid(
            x_query=x_query, xp=endog, fp=marginal
        )
        left_marginal_at_query = interp_left_record_on_padded_grid(
            x_query=x_query, xp=endog, fp=marginal
        )
        right_germ_at_query = interp_right_germ_on_padded_grid(
            x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
        )
        left_germ_at_query = interp_left_germ_on_padded_grid(
            x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
        )
        # The support mask applies only where a finite first node exists:
        # `cand_lower` is `+inf` on an all-NaN (poisoned) row, whose NaN read
        # must reach the maximum fail-loud instead of becoming an ordinary
        # infeasible `(-inf, 0)` pair.
        below_support = (x_query < cand_lower) & jnp.isfinite(cand_lower)
        value_at_query = jnp.where(below_support, -jnp.inf, value_at_query)
        marginal_at_query = jnp.where(below_support, 0.0, marginal_at_query)
        left_marginal_at_query = jnp.where(below_support, 0.0, left_marginal_at_query)
        # Strictly above a candidate's own last finite node its value read is
        # a constant clamp, so its marginal payload is exactly zero there —
        # the separate linear marginal read would republish the terminal
        # record of a node strictly below the query. Re-pinned per candidate,
        # BEFORE the collapse, so an earlier-ending clamp winner cannot
        # publish a stale record; at exact equality the node's own record
        # stands. `cand_upper` is `-inf` on an all-NaN row (mask off — the
        # NaN read stays poisonous).
        cand_upper = jnp.max(jnp.where(jnp.isfinite(endog), endog, -jnp.inf))
        above_support = (x_query > cand_upper) & jnp.isfinite(cand_upper)
        marginal_at_query = jnp.where(above_support, 0.0, marginal_at_query)
        left_marginal_at_query = jnp.where(above_support, 0.0, left_marginal_at_query)
        return (
            value_at_query,
            marginal_at_query,
            left_marginal_at_query,
            right_germ_at_query,
            left_germ_at_query,
        )

    values, marginals, left_marginals, right_germ, left_germ = jax.vmap(read_one)(
        candidate_endog, candidate_value, candidate_marginal
    )
    winner, left_owned = right_germ_winner(
        value=values.T,
        right_germ=tuple(component.T for component in right_germ),
        left_germ=tuple(component.T for component in left_germ),
    )
    # The payload follows the ownership side: the winner's ordinary
    # (right-continuous) marginal on right-decided cells, its left record on
    # left-owned ones — the two differ exactly at duplicated abscissae.
    envelope_marginal = jnp.where(
        left_owned,
        jnp.take_along_axis(left_marginals.T, winner, axis=-1),
        jnp.take_along_axis(marginals.T, winner, axis=-1),
    )[..., 0]
    # The published value is the maximum itself: identical to the winner's read
    # at any tie, and NaN-propagating when a poisoned candidate row (whose NaN
    # empties the tie set) must surface fail-loud.
    return jnp.max(values, axis=0), envelope_marginal


def right_germ_winner(
    *,
    value: FloatND,
    right_germ: tuple[BoolND, FloatND, FloatND, FloatND],
    left_germ: tuple[BoolND, FloatND, FloatND, FloatND],
) -> tuple[IntND, BoolND]:
    """Select the tie-owning candidate index along the trailing candidate axis.

    Staged lexicographic comparison, each stage exact (no packing, no
    tolerance — the claim is exact ordering of the reads' own local pieces):

    - only candidates attaining the maximum value compete,
    - a right-finite read beats one that dies to `-inf` immediately right,
    - then the first, second, and third right derivatives in turn (the local
      pieces are cubics or constant clamps, so agreement through the third
      derivative means the pieces coincide on a right neighborhood),
    - candidates still tied are right-identical — at a shared terminal
      abscissa every candidate clamps right — so the left germ decides:
      left-finite first, then the branch maximizing the read at `q - ε`
      (lexicographically the *smallest* first, *largest* second, *smallest*
      third left derivative), keeping the published marginal inside the
      envelope's generalized gradient at such a boundary,
    - `argmax` resolves what remains to the lowest index, a deterministic
      choice among branches identical on both sides.

    Ownership has a *side*, and the published payload must follow it: when
    the right stages fully separate the tie set, the winner owns a right
    neighborhood and the ordinary (right-continuous) marginal read is the
    consistent payload; when they cannot — the surviving candidates are
    right-identical and the left stages (or the index fallback among
    left-identical branches) decide — ownership is a statement about the left
    neighborhood, and the winner's *left-record* marginal is the payload that
    justified it (at a duplicated terminal abscissa the two differ).

    Args:
        value: Candidate value reads; the candidate axis is last.
        right_germ: Tuple of the right-finiteness flag and the first three
            right derivatives of the candidate value reads, same shape.
        left_germ: Tuple of the left-finiteness flag and the first three
            left derivatives of the candidate value reads, same shape.

    Returns:
        Tuple of the winning candidate index per query cell and the
        left-ownership flag (True where the right stages left the tie
        undecided, so the winner's left-record payload applies), each with
        the candidate axis kept as a trailing length-1 axis (for
        `take_along_axis`).

    """
    right_finite, first, second, third = right_germ
    left_finite, left_first, left_second, left_third = left_germ
    survivors = value >= jnp.max(value, axis=-1, keepdims=True)
    for stage_key in (right_finite.astype(value.dtype), first, second, third):
        stage = jnp.where(survivors, stage_key, -jnp.inf)
        survivors = survivors & (stage >= jnp.max(stage, axis=-1, keepdims=True))
    left_owned = jnp.sum(survivors, axis=-1, keepdims=True) > 1
    left_stage_keys = (
        left_finite.astype(value.dtype),
        -left_first,
        left_second,
        -left_third,
    )
    for stage_key in left_stage_keys:
        stage = jnp.where(survivors, stage_key, -jnp.inf)
        survivors = survivors & (stage >= jnp.max(stage, axis=-1, keepdims=True))
    # int32 winner indices: the candidate axis has at most a few hundred
    # entries, so the x64-default int64 only doubles the gather-index buffers.
    return jnp.argmax(survivors, axis=-1, keepdims=True).astype(jnp.int32), left_owned
