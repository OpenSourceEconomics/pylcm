"""The two-asset unconstrained inverse-Euler step satisfies the FOCs and budget.

The closed-form inverse is correct iff its outputs round-trip: the recovered
consumption and deposit reproduce the post-decision balances through the budget
identities, and they satisfy the consumption and deposit first-order conditions.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_inverse import invert_ucon_cloud

_DISCOUNT = 0.95
_CRRA = 2.0
_MATCH = 1.0


def _cloud():
    # A post-decision grid and gradients with w_a > w_b > 0, which is the
    # unconstrained-region regime (interior deposit, slack borrowing constraint).
    a = jnp.linspace(1.0, 20.0, 6)[:, None] * jnp.ones((6, 5))
    b = jnp.ones((6, 5)) * jnp.linspace(0.5, 10.0, 5)[None, :]
    w_a = jnp.linspace(0.20, 0.05, 6)[:, None] * jnp.ones((6, 5))
    w_b = 0.5 * w_a  # strictly between 0 and w_a -> interior deposit
    return (
        invert_ucon_cloud(
            a=a,
            b=b,
            w_a=w_a,
            w_b=w_b,
            post_decision_value=jnp.zeros((6, 5)),
            discount_factor=_DISCOUNT,
            crra=_CRRA,
            match_rate=_MATCH,
        ),
        a,
        b,
        w_a,
        w_b,
    )


def test_inverse_recovers_liquid_post_decision_balance():
    """`m - c - d` returns the liquid post-decision balance `a`."""
    cloud, a, _b, _wa, _wb = _cloud()
    recovered = cloud.m_endog - cloud.consumption - cloud.deposit
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(a), atol=1e-10)


def test_inverse_recovers_pension_post_decision_balance():
    """`n + d + chi*log(1 + d)` returns the pension post-decision balance `b`."""
    cloud, _a, b, _wa, _wb = _cloud()
    recovered = cloud.n_endog + cloud.deposit + _MATCH * jnp.log1p(cloud.deposit)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(b), atol=1e-10)


def test_inverse_satisfies_consumption_foc():
    """`u'(c) = c**(-rho)` equals `discount_factor * w_a`."""
    cloud, _a, _b, w_a, _wb = _cloud()
    marginal_utility = cloud.consumption ** (-_CRRA)
    np.testing.assert_allclose(
        np.asarray(marginal_utility), np.asarray(_DISCOUNT * w_a), rtol=1e-10
    )


def test_inverse_satisfies_deposit_foc():
    """`w_a = w_b * (1 + chi / (1 + d))` holds at the recovered deposit."""
    cloud, _a, _b, w_a, w_b = _cloud()
    rhs = w_b * (1.0 + _MATCH / (1.0 + cloud.deposit))
    np.testing.assert_allclose(np.asarray(w_a), np.asarray(rhs), rtol=1e-10)
