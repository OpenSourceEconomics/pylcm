"""The NEGM cash-on-hand outer-max carry envelope.

The NEGM kernel collapses the outer durable choice by `V = max_j W_j` on the
exogenous state grid, but the continuation it threads to the parent period is a
carry — value and marginal-utility rows on an endogenous (resources) grid. The
keeper and every adjuster outer-grid node produce one such candidate row per
durable state, each in its *own* resources space: the adjuster `j` pays a
credited durable cost, so its resources are the keeper's cash-on-hand shifted
down by `credited(z, z'_j)`.

The envelope lifts every candidate into a common cash-on-hand (coh) axis by adding
back its credited cost (`∂coh/∂R = 1`, so the value and the resource-marginal
carry over unchanged), interpolates each candidate's value and marginal onto a
shared coh grid per durable state, and takes the pointwise maximum — the genuine
upper envelope in coh space. An adjuster that wins strictly between two keeper
nodes therefore survives, because each candidate is read at the shared coh nodes
by interpolation, not only at its own abscissae. The published marginal is the
*winning* candidate's resource slope at each coh node, never an average across a
branch crossing. The parent's keeper-identity continuation read then interpolates
the published carry exactly as before; the carry it reads is the outer-max
envelope rather than the keeper alone.

The envelope is built by *folding* candidates one at a time into a running
maximum (`init_outer_envelope` → `fold_outer_envelope` per candidate →
`finalize_outer_envelope`), so the caller never materialises all outer-grid
candidates at once. Folding is value-identical to a single stacked maximum:
`max` is associative, the shared coh grid is fixed at the keeper's grid for every
fold, and the strict `>` update keeps the earliest candidate on ties — matching a
stacked `argmax` that returns the first maximiser. `build_outer_envelope_carry`
wraps the fold over a candidate tuple for callers (and tests) that hold them all.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from lcm.typing import Float1D, FloatND, ScalarFloat


class OuterEnvelopeState(NamedTuple):
    """Running coh-space upper envelope over the outer durable candidates.

    The rows are flattened over the carry's leading cells `(discrete..., durable)`
    so a single candidate folds in with one batched interpolation; `finalize`
    restores the leading shape.
    """

    shared_coh: FloatND
    """The fixed shared coh grid (the keeper's), `(n_leading_cells, n_pad)`."""
    value: FloatND
    """Running envelope value at `shared_coh`, `(n_leading_cells, n_pad)`."""
    marginal: FloatND
    """Running winner's marginal at `shared_coh`, `(n_leading_cells, n_pad)`."""
    leading_shape: tuple[int, ...]
    """The carry's leading shape `(discrete..., durable)` for `finalize`."""
    n_durable: int
    """Number of durable-margin states — the last leading axis."""
    taste_shock_scale: ScalarFloat
    """The keeper carry's taste-shock scale, carried through to the result."""


def init_outer_envelope(keeper_carry: EGMCarry) -> OuterEnvelopeState:
    """Start the running envelope from the keeper's own coh grid.

    The keeper is already in coh space (`credited(z, z) = 0`), so its endogenous
    grid is the shared coh grid every candidate is read onto. The running value
    and marginal are the keeper read through the same identity interpolate-and-mask
    path every adjuster uses (its own grid, zero shift), so the keeper is the
    baseline an adjuster must strictly beat — matching a stacked `argmax` that
    returns the first maximiser (the keeper) on a tie.
    """
    n_pad = keeper_carry.endog_grid.shape[-1]
    leading_shape = keeper_carry.endog_grid.shape[:-1]
    n_durable = leading_shape[-1]
    shared_coh = keeper_carry.endog_grid.reshape(-1, n_pad)
    keeper_value = keeper_carry.value.reshape(-1, n_pad)
    keeper_marginal = keeper_carry.marginal_utility.reshape(-1, n_pad)
    value, marginal = jax.vmap(_read_candidate_row)(
        shared_coh, shared_coh, keeper_value, keeper_marginal
    )
    return OuterEnvelopeState(
        shared_coh=shared_coh,
        value=value,
        marginal=marginal,
        leading_shape=leading_shape,
        n_durable=n_durable,
        taste_shock_scale=keeper_carry.taste_shock_scale,
    )


def fold_outer_envelope(
    state: OuterEnvelopeState,
    candidate_carry: EGMCarry,
    coh_shift: Float1D,
) -> OuterEnvelopeState:
    """Fold one outer candidate into the running coh-space maximum.

    The candidate's endogenous grid is shifted into coh space by `coh_shift` (per
    durable state), its value and marginal are interpolated onto the shared coh
    grid, queries below its own first finite coh node are masked to `-inf`, and a
    strict `value > running` update keeps the candidate where it wins (so the
    earliest candidate survives a tie, matching a stacked `argmax`).

    Args:
        state: The running envelope.
        candidate_carry: The candidate's carry — one row per leading cell, in its
            own resources space.
        coh_shift: The credited cost `credited(z, z'_j)` added to the candidate's
            endogenous grid per durable state, shape `(n_durable,)`. Zero for the
            keeper.

    Returns:
        The updated running envelope.

    """
    n_pad = state.shared_coh.shape[-1]
    n_cells = state.shared_coh.shape[0]
    # The durable margin is the last leading axis, so a row-major flatten makes it
    # the fastest-varying index: leading cell `i` sits at durable `i % n_durable`.
    durable_of_cell = jnp.arange(n_cells) % state.n_durable
    shift_per_cell = coh_shift[durable_of_cell][:, None]

    cand_endog = candidate_carry.endog_grid.reshape(-1, n_pad) + shift_per_cell
    cand_value = candidate_carry.value.reshape(-1, n_pad)
    cand_marginal = candidate_carry.marginal_utility.reshape(-1, n_pad)

    value_on, marginal_on = jax.vmap(_read_candidate_row)(
        state.shared_coh, cand_endog, cand_value, cand_marginal
    )
    takes = value_on > state.value
    return state._replace(
        value=jnp.where(takes, value_on, state.value),
        marginal=jnp.where(takes, marginal_on, state.marginal),
    )


def finalize_outer_envelope(state: OuterEnvelopeState) -> EGMCarry:
    """Restore the leading shape and emit the published continuation carry."""
    n_pad = state.shared_coh.shape[-1]
    shape = (*state.leading_shape, n_pad)
    return EGMCarry(
        endog_grid=state.shared_coh.reshape(shape),
        value=state.value.reshape(shape),
        marginal_utility=state.marginal.reshape(shape),
        taste_shock_scale=state.taste_shock_scale,
    )


def build_outer_envelope_carry(
    *,
    keeper_carry: EGMCarry,
    adjuster_carries: tuple[EGMCarry, ...],
    coh_shifts: FloatND,
) -> EGMCarry:
    """Build the outer-max continuation carry as a coh-space upper envelope.

    Folds the keeper and every adjuster candidate into a running coh-space
    maximum (see the module docstring). The published value row is that maximum;
    the published marginal row is the winning candidate's marginal at each coh
    node, so the marginal stays winner-consistent rather than averaged across a
    crossing. A candidate whose borrowing-constrained support starts above the
    shared grid's lower end is masked to `-inf` there and cannot win in that
    region.

    The carry's leading axes are the inner DC-EGM's outer states — any
    discrete/process states first, then the single passive durable margin — so the
    durable is the *last* leading axis. The credited shift depends only on the
    durable margin and the adjuster node, so it is applied per durable state; the
    upper envelope is taken independently per leading cell `(discrete..., durable)`.

    Args:
        keeper_carry: The keeper's continuation carry — one row per leading cell
            `(discrete..., durable)`, already in coh space (`credited(z, z) = 0`).
        adjuster_carries: One carry per outer-grid node, each in its own resources
            space; aligned with the columns of `coh_shifts`.
        coh_shifts: Per durable state (rows) and adjuster node (columns), the
            constant `credited(z, z'_j)` added to that adjuster's endogenous grid
            to map it into coh space. Shape `(n_durable, n_adjusters)`.

    Returns:
        The published continuation carry — the coh-space upper envelope of the
        keeper and all adjuster candidates, one row per leading cell.

    """
    state = init_outer_envelope(keeper_carry)
    for adjuster_index, adjuster_carry in enumerate(adjuster_carries):
        state = fold_outer_envelope(
            state, adjuster_carry, coh_shifts[:, adjuster_index]
        )
    return finalize_outer_envelope(state)


def _read_candidate_row(
    shared_coh: Float1D,
    cand_endog: Float1D,
    cand_value: Float1D,
    cand_marginal: Float1D,
) -> tuple[Float1D, Float1D]:
    """Interpolate one candidate's value and marginal onto the shared coh grid.

    Below the candidate's own first finite coh node the value is masked to `-inf`
    (its borrowing-constrained support has not started), so the candidate cannot
    win there. The marginal is interpolated independently and carried only where
    the candidate wins.
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
