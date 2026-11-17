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
- The read path (`continuation._collapse_stacked_candidates`) reads each
  candidate at the query through the parent's own interpolation convention and
  takes the pointwise maximum there, publishing the winning candidate's marginal
  (Danskin) — never an average across a crossing. It takes the value `argmax`
  with the lowest candidate index on exact ties.
"""

from dataclasses import replace

import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry, stack_egm_carries
from lcm.typing import FloatND


def stack_candidate_carries(
    *, candidates: tuple[EGMCarry, ...], nan_is_infeasible: bool = False
) -> EGMCarry:
    """Stack finite outer candidates without taking a nodewise maximum."""
    if not candidates:
        raise ValueError("At least one outer candidate is required.")
    leaves = tuple(candidate.leaves() for candidate in candidates)
    if all(candidate.query_data_for_transport() is None for candidate in candidates):
        values = tuple(
            jnp.where(jnp.isnan(leaf[("value",)]), -jnp.inf, leaf[("value",)])
            if nan_is_infeasible
            else leaf[("value",)]
            for leaf in leaves
        )
        marginals = tuple(
            jnp.where(jnp.isnan(leaf[("value",)]), 0.0, leaf[("marginal_utility",)])
            if nan_is_infeasible
            else leaf[("marginal_utility",)]
            for leaf in leaves
        )
        return EGMCarry(
            endog_grid=jnp.stack([leaf[("endog_grid",)] for leaf in leaves], axis=-2),
            value=jnp.stack(values, axis=-2),
            marginal_utility=jnp.stack(marginals, axis=-2),
            taste_shock_scale=leaves[0][("taste_shock_scale",)],
        )
    normalized = tuple(
        replace(
            candidate,
            value=jnp.where(
                jnp.isnan(leaf[("value",)]),
                -jnp.inf,
                leaf[("value",)],
            ),
            marginal_utility=jnp.where(
                jnp.isnan(leaf[("value",)]),
                0.0,
                leaf[("marginal_utility",)],
            ),
        )
        if nan_is_infeasible and candidate.query_data_for_transport() is None
        else candidate
        for candidate, leaf in zip(candidates, leaves, strict=True)
    )
    return stack_egm_carries(candidates=normalized)


def build_stacked_outer_carry(
    *,
    keeper_carry: EGMCarry,
    adjuster_carries: tuple[EGMCarry, ...],
    coh_shifts: FloatND,
    durable_axis: int,
) -> EGMCarry:
    """Stack the outer durable candidates into a common-coh candidate axis.

    Every candidate is lifted into a common cash-on-hand axis by adding back its
    credited cost (`∂coh/∂R = 1`, so value and resource-marginal transfer
    unchanged), then stacked on a new candidate axis inserted just before the
    grid axis. Unlike a node-sampled merge this takes no maximum at build time:
    the A+1 conditional carries are retained verbatim so the parent can take the
    exact `max_j V_j(q)` at its own query. The keeper
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
        durable_axis: Position of the durable state among the carry's leading
            axes. The lift addresses this named axis and broadcasts over every
            other discrete or passive-state axis.

    Returns:
        A carry whose existing leading state axes are preserved, followed by
        `n_candidates`, with the row grid trailing. `carry[cell]` is the
        `(n_candidates, n_pad)` block the query-side maximum consumes.

    """
    leading_shape = keeper_carry.leaves()[("endog_grid",)].shape[:-1]
    normalized_axis = durable_axis % len(leading_shape)
    n_durable = leading_shape[normalized_axis]
    if coh_shifts.shape[0] != n_durable:
        msg = (
            "The credited-cost shift's durable dimension does not match the "
            f"carry axis: {coh_shifts.shape[0]} != {n_durable}."
        )
        raise ValueError(msg)
    # Address the durable margin by its named axis and broadcast over every
    # other leading state axis plus the trailing row grid.
    broadcast_list = [1] * (len(leading_shape) + 1)
    broadcast_list[normalized_axis] = n_durable
    broadcast = tuple(broadcast_list)

    lifted = [keeper_carry]
    for adjuster_index, adjuster_carry in enumerate(adjuster_carries):
        shift = coh_shifts[:, adjuster_index].reshape(broadcast)
        lifted.append(
            replace(adjuster_carry, endog_grid=adjuster_carry.endog_grid + shift)
        )
    return stack_candidate_carries(candidates=tuple(lifted))
