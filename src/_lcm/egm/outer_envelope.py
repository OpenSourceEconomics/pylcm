"""The NEGM cash-on-hand outer-max carry envelope.

The NEGM kernel collapses the outer durable choice by `V = max_j W_j` on the
exogenous state grid, but the continuation it threads to the parent period is a
carry — value and marginal-utility rows on an endogenous (resources) grid. The
keeper and every adjuster outer-grid node produce one such candidate row per
durable state, each in its *own* resources space: the adjuster `j` pays a
credited durable cost, so its resources are the keeper's cash-on-hand shifted
down by `credited(z, z'_j)`.

`build_outer_envelope_carry` lifts every candidate into a common cash-on-hand
(coh) axis by adding back its credited cost (`∂coh/∂R = 1`, so the value and the
resource-marginal carry over unchanged), interpolates each candidate's value and
marginal onto a shared coh grid per durable state, and takes the pointwise
maximum — the genuine upper envelope in coh space. An adjuster that wins strictly
between two keeper nodes therefore survives, because each candidate is read at
the shared coh nodes by interpolation, not only at its own abscissae. The
published marginal is the *winning* candidate's resource slope at each coh node,
never an average across a branch crossing. The parent's keeper-identity
continuation read then interpolates the published carry exactly as before; the
carry it reads is now the outer-max envelope rather than the keeper alone.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from lcm.typing import Float1D, FloatND


def build_outer_envelope_carry(
    *,
    keeper_carry: EGMCarry,
    adjuster_carries: tuple[EGMCarry, ...],
    coh_shifts: FloatND,
) -> EGMCarry:
    """Build the outer-max continuation carry as a coh-space upper envelope.

    Each candidate's endogenous grid is shifted into a common cash-on-hand axis
    (`+credited(z, z'_j)`, zero for the keeper), every candidate's value and
    marginal-utility rows are interpolated onto a shared coh grid per durable
    state, and the pointwise maximum over candidates is taken. The published
    value row is that maximum; the published marginal row is the winning
    candidate's marginal at each coh node, so the marginal stays
    winner-consistent rather than averaged across a crossing.

    The shared coh grid per durable state spans the candidates' joint finite coh
    range with the keeper's grid length, so a candidate that has no valid node
    below the shared grid's lower end (its borrowing-constrained support starts
    higher) is masked to `-inf` there and cannot win in that region.

    Args:
        keeper_carry: The keeper's continuation carry — one row per durable
            state, already in coh space (`credited(z, z) = 0`).
        adjuster_carries: One carry per outer-grid node, each in its own
            resources space; aligned with the columns of `coh_shifts`.
        coh_shifts: Per durable state (rows) and adjuster node (columns), the
            constant `credited(z, z'_j)` added to that adjuster's endogenous grid
            to map it into coh space. Shape `(n_durable, n_adjusters)`.

    Returns:
        The published continuation carry — the coh-space upper envelope of the
        keeper and all adjuster candidates, one row per durable state.

    """
    n_pad = keeper_carry.endog_grid.shape[-1]
    n_durable = keeper_carry.endog_grid.shape[0]

    # Stack the candidate rows on a leading axis: the keeper (shift 0) then every
    # adjuster shifted into coh space. Each block is (n_durable, n_pad); the
    # stack is (n_candidates, n_durable, n_pad).
    keeper_shift = jnp.zeros((n_durable, 1), dtype=coh_shifts.dtype)
    all_shifts = jnp.concatenate([keeper_shift, coh_shifts], axis=1)

    endog_stack = jnp.stack(
        [keeper_carry.endog_grid, *(c.endog_grid for c in adjuster_carries)], axis=0
    )
    value_stack = jnp.stack(
        [keeper_carry.value, *(c.value for c in adjuster_carries)], axis=0
    )
    marginal_stack = jnp.stack(
        [
            keeper_carry.marginal_utility,
            *(c.marginal_utility for c in adjuster_carries),
        ],
        axis=0,
    )

    # Shift each candidate's endogenous grid into coh space; the per-(durable,
    # candidate) shift broadcasts over the grid axis. NaN padding stays NaN, so
    # the per-row interpolation's padding handling is preserved.
    endog_coh = endog_stack + all_shifts.T[:, :, None]

    refined_grid, refined_value, refined_marginal = jax.vmap(
        _envelope_one_durable_state, in_axes=(1, 1, 1, None)
    )(endog_coh, value_stack, marginal_stack, n_pad)

    return EGMCarry(
        endog_grid=refined_grid,
        value=refined_value,
        marginal_utility=refined_marginal,
        taste_shock_scale=keeper_carry.taste_shock_scale,
    )


def _envelope_one_durable_state(
    endog_coh: FloatND,
    value: FloatND,
    marginal: FloatND,
    n_pad: int,
) -> tuple[Float1D, Float1D, Float1D]:
    """Upper-envelope the candidate rows of one durable state on a shared coh grid.

    `endog_coh`, `value`, `marginal` are `(n_candidates, n_pad)` — each row a
    candidate's coh-shifted carry. The shared coh grid spans the joint finite coh
    range; every candidate is interpolated onto it (queries below a candidate's
    own support clamp to that candidate's lowest value, so a candidate whose
    support starts higher is masked to `-inf` below its first finite node), and
    the pointwise maximum over candidates is the envelope. The winning
    candidate's marginal is carried at each node.
    """
    finite = jnp.isfinite(endog_coh) & jnp.isfinite(value)
    safe_coh = jnp.where(finite, endog_coh, jnp.inf)
    lower = jnp.min(safe_coh)
    upper = jnp.max(jnp.where(finite, endog_coh, -jnp.inf))
    shared_coh = jnp.linspace(lower, upper, n_pad)

    def _read_candidate(
        cand_endog: Float1D, cand_value: Float1D, cand_marginal: Float1D
    ) -> tuple[Float1D, Float1D]:
        """Interpolate one candidate's value and marginal onto the shared grid.

        Below the candidate's own first finite coh node the value is masked to
        `-inf` (its borrowing-constrained support has not started), so the
        candidate cannot win there.
        """
        cand_lower = jnp.min(jnp.where(jnp.isfinite(cand_endog), cand_endog, jnp.inf))
        value_on_shared = interp_on_padded_grid(
            x_query=shared_coh,
            xp=cand_endog,
            fp=cand_value,
            fp_slopes=cand_marginal,
        )
        marginal_on_shared = interp_on_padded_grid(
            x_query=shared_coh,
            xp=cand_endog,
            fp=cand_marginal,
        )
        below_support = shared_coh < cand_lower
        value_on_shared = jnp.where(below_support, -jnp.inf, value_on_shared)
        return value_on_shared, marginal_on_shared

    values_on_shared, marginals_on_shared = jax.vmap(_read_candidate)(
        endog_coh, value, marginal
    )
    winner = jnp.argmax(values_on_shared, axis=0)
    envelope_value = jnp.max(values_on_shared, axis=0)
    envelope_marginal = jnp.take_along_axis(
        marginals_on_shared, winner[None, :], axis=0
    )[0]
    return shared_coh, envelope_value, envelope_marginal
