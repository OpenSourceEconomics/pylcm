"""Spec for the NEGM cash-on-hand outer-max carry envelope.

The NEGM kernel publishes one continuation carry per durable state. The keeper
and every adjuster outer-grid node each produce a candidate value row; the
published carry must be their genuine upper envelope in cash-on-hand (coh)
space, so a parent period that interpolates the carry sees the best durable
choice at every coh — not only the keeper's, and not only at the keeper's nodes.

`build_outer_envelope_carry` maps each candidate's endogenous (resources) grid
into coh space by a per-candidate constant shift (`+credited(z, z')`, zero for
the keeper), concatenates the shifted candidate rows per durable state, and runs
the regime's upper-envelope backend to produce the published carry. These tests
exercise that construction in isolation, with synthetic keeper and adjuster
rows, so the envelope logic is pinned independently of the toy model and oracle.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.outer_envelope import build_outer_envelope_carry
from lcm.typing import Float1D


def _crra_value_row(*, coh: Float1D, level: float) -> Float1D:
    """A concave, increasing synthetic value row plus a constant `level`."""
    return jnp.sqrt(coh) + level


def _identity_marginal(*, coh: Float1D) -> Float1D:
    """A positive, decreasing marginal row consistent with `sqrt` curvature."""
    return 0.5 / jnp.sqrt(coh)


def test_adjuster_winning_by_delta_lifts_the_envelope_by_delta() -> None:
    """An adjuster that beats the keeper by `delta` lifts the carry by `delta`.

    One durable state, one keeper row, one adjuster row. The adjuster's value is
    the keeper's plus a constant `delta` over a reachable coh interval, so the
    upper envelope must equal the adjuster there: reading the published carry at
    a coh in that interval returns the keeper value plus `delta`.
    """
    n_pad = 40
    coh = jnp.linspace(1.0, 9.0, n_pad)
    delta = 0.25

    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    # The adjuster's own resources grid is coh shifted down by its credited cost;
    # the shift maps it back into coh space. Its value beats the keeper by delta.
    shift = 2.0
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=(_crra_value_row(coh=coh, level=delta))[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_outer_envelope_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),  # one durable state, one adjuster
    )

    assert carry.value.shape == (1, n_pad)
    query = jnp.asarray([3.0, 5.0, 7.0])
    enveloped = interp_on_padded_grid(
        x_query=query,
        xp=carry.endog_grid[0],
        fp=carry.value[0],
        fp_slopes=carry.marginal_utility[0],
    )
    expected = jnp.sqrt(query) + delta
    np.testing.assert_allclose(np.asarray(enveloped), np.asarray(expected), atol=2e-2)


def test_adjuster_winning_only_on_a_subinterval_is_captured() -> None:
    """An adjuster winning only on a coh sub-interval lifts the envelope there.

    The adjuster beats the keeper by a hump centred in the coh range and is below
    it at both ends. The published envelope must equal the keeper outside the
    hump and the adjuster inside it, including at coh points strictly between the
    keeper's grid nodes — a nodewise maximum at the keeper's abscissae would miss
    the interior win.
    """
    n_pad = 60
    coh = jnp.linspace(1.0, 9.0, n_pad)
    # A hump peaking near coh = 5, negative at the ends: the adjuster wins only
    # in the middle of the coh range.
    hump = 0.5 - 0.08 * (coh - 5.0) ** 2
    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    shift = 2.0
    adjuster_value = jnp.sqrt(coh) + hump
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=adjuster_value[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_outer_envelope_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),
    )

    # An off-node query in the winning interior: the envelope must equal the
    # adjuster (keeper + hump), strictly above the keeper.
    interior = jnp.asarray([4.7, 5.3])
    enveloped_interior = interp_on_padded_grid(
        x_query=interior,
        xp=carry.endog_grid[0],
        fp=carry.value[0],
        fp_slopes=carry.marginal_utility[0],
    )
    expected_interior = jnp.sqrt(interior) + (0.5 - 0.08 * (interior - 5.0) ** 2)
    np.testing.assert_allclose(
        np.asarray(enveloped_interior), np.asarray(expected_interior), atol=3e-2
    )
    assert bool(jnp.all(enveloped_interior > jnp.sqrt(interior)))

    # At the coh ends the keeper wins: the envelope equals the keeper value.
    ends = jnp.asarray([1.5, 8.5])
    enveloped_ends = interp_on_padded_grid(
        x_query=ends,
        xp=carry.endog_grid[0],
        fp=carry.value[0],
        fp_slopes=carry.marginal_utility[0],
    )
    np.testing.assert_allclose(
        np.asarray(enveloped_ends), np.asarray(jnp.sqrt(ends)), atol=3e-2
    )


def test_keeper_wins_when_no_adjuster_improves() -> None:
    """With no adjuster beating the keeper, the carry reproduces the keeper.

    Every adjuster value lies strictly below the keeper, so the upper envelope is
    the keeper row itself: reading the published carry returns the keeper value.
    """
    n_pad = 40
    coh = jnp.linspace(1.0, 9.0, n_pad)

    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    shift = 2.0
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=_crra_value_row(coh=coh, level=-0.5)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_outer_envelope_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),
    )

    query = jnp.asarray([3.0, 5.0, 7.0])
    enveloped = interp_on_padded_grid(
        x_query=query,
        xp=carry.endog_grid[0],
        fp=carry.value[0],
        fp_slopes=carry.marginal_utility[0],
    )
    expected = jnp.sqrt(query)
    np.testing.assert_allclose(np.asarray(enveloped), np.asarray(expected), atol=2e-2)
