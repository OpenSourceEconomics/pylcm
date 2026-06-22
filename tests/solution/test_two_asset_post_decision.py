"""Post-decision value and gradients are exact for an affine next-period value.

When `V'` is affine on the regular `(m, n)` grid, bilinear interpolation reproduces
it exactly, so the post-decision value and its chain-rule gradients must equal their
closed forms: `w_a = beta_m*(1+r^a)` and `w_b = beta_n*(1+r^b)`.

At the working->retired boundary both post-decision balances feed a single retired
liquid state through the lump-sum payout, and the same affine-exactness pins the
boundary reader's value and chain-rule gradients.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_post_decision import (
    post_decision_value_and_grad,
    post_decision_value_and_grad_retiring,
)

_ALPHA, _BETA_M, _BETA_N = 1.0, 0.7, 0.3
_RETURN_LIQUID, _RETURN_PENSION, _WAGE = 0.03, 0.06, 5.0


def _setup():
    m_grid = jnp.linspace(0.0, 200.0, 41)
    n_grid = jnp.linspace(0.0, 120.0, 31)
    mesh_m, mesh_n = jnp.meshgrid(m_grid, n_grid, indexing="ij")
    next_value = _ALPHA + _BETA_M * mesh_m + _BETA_N * mesh_n
    a = jnp.linspace(5.0, 40.0, 6)[:, None] * jnp.ones((6, 4))
    b = jnp.ones((6, 4)) * jnp.linspace(2.0, 30.0, 4)[None, :]
    out = post_decision_value_and_grad(
        next_value=next_value,
        m_grid=m_grid,
        n_grid=n_grid,
        a=a,
        b=b,
        return_liquid=_RETURN_LIQUID,
        return_pension=_RETURN_PENSION,
        wage=_WAGE,
    )
    return out, a, b


def test_post_decision_value_matches_affine_closed_form():
    """`w(a,b)` equals `V'` evaluated at the carried-forward states."""
    out, a, b = _setup()
    m_next = (1.0 + _RETURN_LIQUID) * a + _WAGE
    n_next = (1.0 + _RETURN_PENSION) * b
    expected = _ALPHA + _BETA_M * m_next + _BETA_N * n_next
    np.testing.assert_allclose(np.asarray(out.value), np.asarray(expected), rtol=1e-5)


def test_post_decision_grad_a_is_return_scaled_marginal_value():
    """`w_a = beta_m * (1 + r^a)` everywhere (affine value, constant marginal)."""
    out, _a, _b = _setup()
    expected = _BETA_M * (1.0 + _RETURN_LIQUID)
    np.testing.assert_allclose(np.asarray(out.grad_a), expected, rtol=1e-5)


def test_post_decision_grad_b_is_return_scaled_marginal_value():
    """`w_b = beta_n * (1 + r^b)` everywhere (affine value, constant marginal)."""
    out, _a, _b = _setup()
    expected = _BETA_N * (1.0 + _RETURN_PENSION)
    np.testing.assert_allclose(np.asarray(out.grad_b), expected, rtol=1e-5)


_PAYOUT, _RETIREMENT_INCOME, _BETA_L = 1.05, 0.5, 0.4


def _setup_retiring():
    liquid_grid = jnp.linspace(0.0, 200.0, 41)
    next_value_retired = _ALPHA + _BETA_L * liquid_grid
    next_marginal_retired = _BETA_L * jnp.ones_like(liquid_grid)
    a = jnp.linspace(5.0, 40.0, 6)[:, None] * jnp.ones((6, 4))
    b = jnp.ones((6, 4)) * jnp.linspace(2.0, 30.0, 4)[None, :]
    out = post_decision_value_and_grad_retiring(
        next_value_retired=next_value_retired,
        next_marginal_retired=next_marginal_retired,
        liquid_grid=liquid_grid,
        a=a,
        b=b,
        return_liquid=_RETURN_LIQUID,
        pension_payout_return=_PAYOUT,
        retirement_income=_RETIREMENT_INCOME,
    )
    return out, a, b


def test_retiring_post_decision_value_reads_retired_value_through_the_payout():
    """`w(a,b)` equals the retired value at the lump-sum retired liquid state."""
    out, a, b = _setup_retiring()
    liquid_next = (1.0 + _RETURN_LIQUID) * a + _PAYOUT * b + _RETIREMENT_INCOME
    expected = _ALPHA + _BETA_L * liquid_next
    np.testing.assert_allclose(np.asarray(out.value), np.asarray(expected), rtol=1e-5)


def test_retiring_post_decision_grad_a_scales_retired_marginal_by_liquid_return():
    """`w_a = (1 + r^a) * V'_retired` (carried retired marginal, return-scaled)."""
    out, _a, _b = _setup_retiring()
    expected = (1.0 + _RETURN_LIQUID) * _BETA_L
    np.testing.assert_allclose(np.asarray(out.grad_a), expected, rtol=1e-5)


def test_retiring_post_decision_grad_b_scales_retired_marginal_by_payout():
    """`w_b = pension_payout_return * V'_retired` (the lump-sum chain rule)."""
    out, _a, _b = _setup_retiring()
    expected = _PAYOUT * _BETA_L
    np.testing.assert_allclose(np.asarray(out.grad_b), expected, rtol=1e-5)
