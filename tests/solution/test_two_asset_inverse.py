"""The two-asset unconstrained inverse-Euler step satisfies the FOCs and budget.

The closed-form inverse is correct iff its outputs round-trip: the recovered
consumption and deposit reproduce the post-decision balances through the budget
identities, and they satisfy the consumption and deposit first-order conditions.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_inverse import (
    invert_acon_cloud,
    invert_con_cloud,
    invert_dcon_cloud,
    invert_ucon_cloud,
)
from tests.conftest import X64_ENABLED

# Round-trip and FOC identities are float-eps-limited at the active precision.
_ATOL = 1e-10 if X64_ENABLED else 1e-5
_RTOL = 1e-10 if X64_ENABLED else 1e-5

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
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(a), atol=_ATOL)


def test_inverse_recovers_pension_post_decision_balance():
    """`n + d + chi*log(1 + d)` returns the pension post-decision balance `b`."""
    cloud, _a, b, _wa, _wb = _cloud()
    recovered = cloud.n_endog + cloud.deposit + _MATCH * jnp.log1p(cloud.deposit)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(b), atol=_ATOL)


def test_inverse_satisfies_consumption_foc():
    """`u'(c) = c**(-rho)` equals `discount_factor * w_a`."""
    cloud, _a, _b, w_a, _wb = _cloud()
    marginal_utility = cloud.consumption ** (-_CRRA)
    np.testing.assert_allclose(
        np.asarray(marginal_utility), np.asarray(_DISCOUNT * w_a), rtol=_RTOL
    )


def test_inverse_satisfies_deposit_foc():
    """`w_a = w_b * (1 + chi / (1 + d))` holds at the recovered deposit."""
    cloud, _a, _b, w_a, w_b = _cloud()
    rhs = w_b * (1.0 + _MATCH / (1.0 + cloud.deposit))
    np.testing.assert_allclose(np.asarray(w_a), np.asarray(rhs), rtol=_RTOL)


def _dcon_cloud():
    # Gradients with w_a >= w_b*(1+chi): the unconstrained deposit would be negative,
    # so the deposit is pinned to zero (the deposit-constrained region).
    a = jnp.linspace(1.0, 20.0, 6)[:, None] * jnp.ones((6, 5))
    b = jnp.ones((6, 5)) * jnp.linspace(0.5, 10.0, 5)[None, :]
    w_a = jnp.linspace(0.20, 0.05, 6)[:, None] * jnp.ones((6, 5))
    w_b = 0.2 * w_a  # w_a = 5 w_b > w_b*(1+chi) -> deposit pinned to 0
    return (
        invert_dcon_cloud(
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
    )


def test_dcon_pins_deposit_to_zero():
    """The deposit-constrained cloud has `d = 0` everywhere."""
    cloud, _a, _b, _wa = _dcon_cloud()
    np.testing.assert_array_equal(np.asarray(cloud.deposit), 0.0)


def test_dcon_recovers_liquid_budget_with_zero_deposit():
    """`m - c` returns the liquid post-decision balance `a` (since `d = 0`)."""
    cloud, a, _b, _wa = _dcon_cloud()
    np.testing.assert_allclose(
        np.asarray(cloud.m_endog - cloud.consumption), np.asarray(a), atol=_ATOL
    )


def test_dcon_leaves_pension_unchanged():
    """The endogenous pension equals the post-decision balance `b` (no deposit)."""
    cloud, _a, b, _wa = _dcon_cloud()
    np.testing.assert_array_equal(np.asarray(cloud.n_endog), np.asarray(b))


def test_dcon_satisfies_consumption_foc():
    """`u'(c) = c**(-rho)` equals `discount_factor * w_a`."""
    cloud, _a, _b, w_a = _dcon_cloud()
    np.testing.assert_allclose(
        np.asarray(cloud.consumption ** (-_CRRA)),
        np.asarray(_DISCOUNT * w_a),
        rtol=_RTOL,
    )


# The deposit-FOC ratio `1 + chi/(1 + d) = u'(c)/(beta*w_b)` is fixed to 1.6, which lies
# strictly in `(1, 1 + chi) = (1, 2)`, so the constructed deposit is interior (`d > 0`)
# while the borrowing constraint binds (`a = 0`).
_ACON_RATIO = 1.6


def _acon_cloud():
    consumption = jnp.linspace(0.5, 3.0, 6)[:, None] * jnp.ones((6, 5))
    b = jnp.ones((6, 5)) * jnp.linspace(0.5, 10.0, 5)[None, :]
    marginal_utility = consumption ** (-_CRRA)
    # Pin u'(c)/(beta*w_b) = _ACON_RATIO so the recovered deposit is interior.
    w_b = marginal_utility / (_DISCOUNT * _ACON_RATIO)
    return (
        invert_acon_cloud(
            consumption=consumption,
            b=b,
            post_decision_value_at_zero_a=jnp.zeros((6, 5)),
            w_b_at_zero_a=w_b,
            w_a_at_zero_a=marginal_utility / _DISCOUNT,
            discount_factor=_DISCOUNT,
            crra=_CRRA,
            match_rate=_MATCH,
        ),
        consumption,
        b,
        w_b,
    )


def test_acon_pins_liquid_post_decision_to_zero():
    """The liquid post-decision balance `m - c - d` is zero (borrowing binds)."""
    cloud, _c, _b, _wb = _acon_cloud()
    recovered = cloud.m_endog - cloud.consumption - cloud.deposit
    np.testing.assert_allclose(np.asarray(recovered), 0.0, atol=_ATOL)


def test_acon_recovers_pension_post_decision_balance():
    """`n + d + chi*log(1 + d)` returns the pension post-decision balance `b`."""
    cloud, _c, b, _wb = _acon_cloud()
    recovered = cloud.n_endog + cloud.deposit + _MATCH * jnp.log1p(cloud.deposit)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(b), atol=_ATOL)


def test_acon_has_interior_deposit():
    """The recovered deposit is strictly positive (the deposit margin is slack)."""
    cloud, _c, _b, _wb = _acon_cloud()
    assert np.all(np.asarray(cloud.deposit) > 0.0)


def test_acon_satisfies_deposit_foc():
    """`u'(c) = beta * w_b * (1 + chi / (1 + d))` holds at the recovered deposit."""
    cloud, _c, _b, w_b = _acon_cloud()
    rhs = _DISCOUNT * w_b * (1.0 + _MATCH / (1.0 + cloud.deposit))
    np.testing.assert_allclose(
        np.asarray(cloud.consumption ** (-_CRRA)), np.asarray(rhs), rtol=_RTOL
    )


def _con_cloud():
    consumption = jnp.linspace(0.5, 3.0, 6)[:, None] * jnp.ones((6, 5))
    b = jnp.ones((6, 5)) * jnp.linspace(0.5, 10.0, 5)[None, :]
    return (
        invert_con_cloud(
            consumption=consumption,
            b=b,
            post_decision_value_at_zero_a=jnp.zeros((6, 5)),
            w_b_at_zero_a=jnp.ones((6, 5)),
            w_a_at_zero_a=jnp.ones((6, 5)),
            discount_factor=_DISCOUNT,
            crra=_CRRA,
            match_rate=_MATCH,
        ),
        consumption,
        b,
    )


def test_con_pins_deposit_to_zero():
    """The fully-constrained corner has `d = 0` everywhere."""
    cloud, _c, _b = _con_cloud()
    np.testing.assert_array_equal(np.asarray(cloud.deposit), 0.0)


def test_con_consumes_entire_liquid_budget():
    """`m = c` at the corner (`a = 0` and `d = 0`)."""
    cloud, consumption, _b = _con_cloud()
    np.testing.assert_array_equal(np.asarray(cloud.m_endog), np.asarray(consumption))


def test_con_leaves_pension_unchanged():
    """The endogenous pension equals the post-decision balance `b` (no deposit)."""
    cloud, _c, b = _con_cloud()
    np.testing.assert_array_equal(np.asarray(cloud.n_endog), np.asarray(b))
