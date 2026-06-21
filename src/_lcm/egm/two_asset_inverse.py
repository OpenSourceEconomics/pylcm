"""Closed-form inverse-Euler step for the two-asset model (unconstrained interior).

The first kernel piece of the two-continuous-state EGM foundation, for the
unconstrained (`ucon`) region where both the liquid borrowing constraint and the
deposit lower bound are slack. Given a post-decision grid $(a, b)$ (liquid and
pension post-decision balances) and the post-decision value gradients
$w_a = \\partial_a w$, $w_b = \\partial_b w$, the two intratemporal first-order
conditions invert in closed form to the optimal consumption and deposit, and the
budget identities map each post-decision node back to the endogenous current state
$(m, n)$ (cash-on-hand and pension balance) it is optimal at. This is the
single-segment case — no upper envelope is involved.

The inverse is the verified core: $u'(c) = \\beta w_a$ gives $c$, and equating the
marginal value of a liquid dollar to that of a deposited dollar,
$w_a = w_b\\,(1 + \\chi/(1 + d))$, gives $d$.
"""

from typing import NamedTuple

import jax.numpy as jnp

from lcm.typing import FloatND


class UconCloud(NamedTuple):
    """Endogenous cloud produced by the unconstrained inverse-Euler step.

    Each field is shaped like the post-decision $(a, b)$ grid; entry $(i, j)$ is the
    current state and policy at which post-decision node $(a_i, b_j)$ is optimal.
    """

    m_endog: FloatND
    """Endogenous cash-on-hand `liquid` at which the node is optimal."""
    n_endog: FloatND
    """Endogenous pension balance at which the node is optimal."""
    consumption: FloatND
    """Optimal consumption `c`."""
    deposit: FloatND
    """Optimal pension deposit `d`."""
    value: FloatND
    """Value `u(c) + discount_factor * w` at the endogenous state."""
    value_grad_m: FloatND
    """Marginal value of liquid wealth, `discount_factor * w_a`."""
    value_grad_n: FloatND
    """Marginal value of pension wealth, `discount_factor * w_b`."""


def invert_ucon_cloud(
    *,
    a: FloatND,
    b: FloatND,
    w_a: FloatND,
    w_b: FloatND,
    post_decision_value: FloatND,
    discount_factor: float,
    crra: float,
    match_rate: float,
) -> UconCloud:
    """Invert the two intratemporal FOCs on the unconstrained region.

    Args:
        a: Liquid post-decision balance at each grid node.
        b: Pension post-decision balance at each grid node.
        w_a: Post-decision value gradient with respect to `a`.
        w_b: Post-decision value gradient with respect to `b`.
        post_decision_value: Post-decision value `w(a, b)` at each node.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        match_rate: Pension employer-match coefficient `chi` (match `chi*log(1+d)`).

    Returns:
        The endogenous cloud: current state, policy, value, and value gradient per node.

    """
    consumption = (discount_factor * w_a) ** (-1.0 / crra)
    deposit = match_rate * w_b / (w_a - w_b) - 1.0
    # Budget identities, inverted: a = m - c - d, b = n + d + chi*log(1 + d).
    m_endog = a + consumption + deposit
    n_endog = b - deposit - match_rate * jnp.log1p(deposit)
    value = _crra_utility(consumption, crra) + discount_factor * post_decision_value
    return UconCloud(
        m_endog=m_endog,
        n_endog=n_endog,
        consumption=consumption,
        deposit=deposit,
        value=value,
        value_grad_m=discount_factor * w_a,
        value_grad_n=discount_factor * w_b,
    )


def _crra_utility(consumption: FloatND, crra: float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
