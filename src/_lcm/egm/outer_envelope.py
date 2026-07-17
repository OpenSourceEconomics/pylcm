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
  across a crossing. The production read path applies the same semantics inside
  the parent's child-carry aggregation.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import (
    interp_on_padded_grid,
    interp_right_derivative_on_padded_grid,
)
from lcm.typing import Float1D, FloatND


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
    infeasible pair). The published value is the pointwise maximum over
    candidates; the published marginal is the *winning* candidate's resource
    slope (Danskin), so it is winner-consistent and never averaged across a
    branch crossing. At an exact value tie the winner is right-continuous in
    the value read itself: the tied candidates are ranked by the right
    derivative of their own value interpolants — the Fritsch-Carlson-limited
    Hermite slope of the bracket right of the query, exactly zero on the clamp
    ray at or above a candidate's last node — so the branch whose read actually
    wins immediately to the right of the query owns the published (economic)
    marginal, matching the one-sided convention the parent's Euler inversion
    expects. Candidates whose reads are identical to first order on the right
    fall back to the lowest index, a deterministic choice among locally
    indistinguishable branches.

    Taking the maximum at the query — rather than at a shared node grid and
    republishing a single interpolated row — is exact for the finite candidate set
    at every query (`thm:nnbegm`): a candidate that wins only on an interval
    strictly between two nodes is read at its true value there instead of being
    bridged upward (`thm:aggregate-bridge`).

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
    ) -> tuple[Float1D, Float1D, Float1D]:
        cand_lower = jnp.min(jnp.where(jnp.isfinite(endog), endog, jnp.inf))
        value_at_query = interp_on_padded_grid(
            x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
        )
        marginal_at_query = interp_on_padded_grid(
            x_query=x_query, xp=endog, fp=marginal
        )
        right_slope_at_query = interp_right_derivative_on_padded_grid(
            x_query=x_query, xp=endog, fp=value, fp_slopes=marginal
        )
        # The support mask applies only where a finite first node exists:
        # `cand_lower` is `+inf` on an all-NaN (poisoned) row, whose NaN read
        # must reach the maximum fail-loud instead of becoming an ordinary
        # infeasible `(-inf, 0)` pair.
        below_support = (x_query < cand_lower) & jnp.isfinite(cand_lower)
        value_at_query = jnp.where(below_support, -jnp.inf, value_at_query)
        marginal_at_query = jnp.where(below_support, 0.0, marginal_at_query)
        right_slope_at_query = jnp.where(below_support, 0.0, right_slope_at_query)
        return value_at_query, marginal_at_query, right_slope_at_query

    values, marginals, right_slopes = jax.vmap(read_one)(
        candidate_endog, candidate_value, candidate_marginal
    )
    envelope_value = jnp.max(values, axis=0)
    # Staged lexicographic ownership: only candidates attaining the envelope
    # value compete (`-inf` rank otherwise); among them the largest right
    # derivative of the value read wins, and `argmax` resolves exact rank ties
    # to the lowest index. No packing of the two stages into one float — the
    # comparisons stay exact at any magnitude and any precision.
    rank = jnp.where(values >= envelope_value, right_slopes, -jnp.inf)
    winner = jnp.argmax(rank, axis=0)
    envelope_marginal = jnp.take_along_axis(marginals, winner[None, :], axis=0)[0]
    return envelope_value, envelope_marginal
