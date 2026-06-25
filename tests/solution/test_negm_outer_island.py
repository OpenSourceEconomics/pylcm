"""NEGM outer envelope keeps an adjuster that wins between keeper nodes.

The nested-EGM outer fold reads every adjuster candidate onto the keeper's shared
cash-on-hand grid and takes the running maximum at those nodes. A shared-node
maximum alone is a *sampled* maximum: an adjuster branch whose value exceeds the
keeper only on an interval strictly between two shared nodes — a thin (S, s)
adjustment island — is never sampled at its peak. To recover it the fold also
reads each adjuster at its own nodes and splices that adjuster's best win into
the published carry, so the envelope retains the island the sampled maximum
would miss.
"""

import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.outer_envelope import build_outer_envelope_carry


def test_outer_envelope_keeps_adjuster_island_between_shared_nodes():
    """An adjuster winning only between keeper nodes survives the outer envelope.

    The keeper value is flat zero on `linspace(0, 1, 11)`. The adjuster carries a
    positive island peaking at `x = 0.55` (value `0.1`) — a node on its own grid
    but between the keeper nodes `0.5` and `0.6` — and is negative elsewhere. The
    true outer maximum at `0.55` is `0.1`, so the published envelope must report
    at least that there.
    """
    keeper_coh = jnp.linspace(0.0, 1.0, 11)
    zeros11 = jnp.zeros(11)
    keeper = EGMCarry(
        endog_grid=keeper_coh[None, :],
        value=zeros11[None, :],
        marginal_utility=zeros11[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    adj_coh = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0])
    adj_val = jnp.array(
        [-0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.1, -0.1, -0.2, -0.2, -0.2]
    )
    adjuster = EGMCarry(
        endog_grid=adj_coh[None, :],
        value=adj_val[None, :],
        marginal_utility=zeros11[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_outer_envelope_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[0.0]]),
    )
    enveloped_at_peak = float(
        interp_on_padded_grid(
            x_query=jnp.array([0.55]),
            xp=carry.endog_grid[0],
            fp=carry.value[0],
            fp_slopes=carry.marginal_utility[0],
        )[0]
    )

    assert enveloped_at_peak >= 0.1 - 1e-6
