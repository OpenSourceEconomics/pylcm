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

from lcm.typing import BoolND, FloatND, ScalarFloat


class RegionCloud(NamedTuple):
    """Endogenous cloud produced by a region's inverse-Euler step.

    Each field is shaped like the post-decision $(a, b)$ grid; entry $(i, j)$ is the
    current state and policy at which post-decision node $(a_i, b_j)$ is optimal,
    conditional on the constraint region (`ucon`, `dcon`, ...) the cloud was built for.
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
    """Marginal value of liquid wealth `dV/dm`, equal to `u'(c)` at the optimum.

    By the envelope theorem this is `discount_factor * w_a` where the liquid Euler
    equation holds (`ucon`/`dcon`), and `u'(c)` directly at the borrowing-constrained
    corner (`acon`/`con`), where all extra liquid is consumed — the two coincide at the
    optimal policy.
    """
    value_grad_n: FloatND
    """Marginal value of pension wealth `dV/dn`, equal to `discount_factor * w_b`."""
    valid_region: BoolND
    """Whether the node satisfies this region's complementary-slackness conditions.

    Each region inverts its first-order conditions *assuming* its own constraints
    bind; the solution is a genuine KKT candidate only where the region's
    inequalities also hold. A node failing them is a finite but spurious point
    that must be excluded before the upper-envelope selection, not merely a
    non-finite one.
    """


def invert_ucon_cloud(
    *,
    a: FloatND,
    b: FloatND,
    w_a: FloatND,
    w_b: FloatND,
    post_decision_value: FloatND,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
) -> RegionCloud:
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
    # KKT: liquid slack (a > 0, so the consumption Euler holds) and an interior
    # deposit (d > 0). `deposit > 0` already implies `w_a > w_b`, since a
    # non-positive denominator drives the deposit below zero.
    valid_region = (a > 0.0) & (deposit > 0.0) & (w_a > w_b)
    return RegionCloud(
        m_endog=m_endog,
        n_endog=n_endog,
        consumption=consumption,
        deposit=deposit,
        value=value,
        value_grad_m=discount_factor * w_a,
        value_grad_n=discount_factor * w_b,
        valid_region=valid_region,
    )


def invert_dcon_cloud(
    *,
    a: FloatND,
    b: FloatND,
    w_a: FloatND,
    w_b: FloatND,
    post_decision_value: FloatND,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
) -> RegionCloud:
    """Invert the consumption FOC on the deposit-constrained region (`dcon`, `d = 0`).

    Where the pension is unattractive enough that the optimal deposit hits its lower
    bound, the deposit is pinned to zero and only the consumption FOC `u'(c) = beta*w_a`
    is inverted. With `d = 0` the pension is unchanged (`n = b`) and the liquid budget
    identity is `a = m - c`.

    Args:
        a: Liquid post-decision balance at each node.
        b: Pension post-decision balance at each node.
        w_a: Post-decision value gradient with respect to `a`.
        w_b: Post-decision value gradient with respect to `b`.
        post_decision_value: Post-decision value `w(a, b)` at each node.
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        match_rate: Pension employer-match coefficient `chi`; enters only the
            complementary-slackness test for `d = 0` being optimal.

    Returns:
        The deposit-constrained endogenous cloud.

    """
    consumption = (discount_factor * w_a) ** (-1.0 / crra)
    value = _crra_utility(consumption, crra) + discount_factor * post_decision_value
    # KKT: liquid slack (a > 0) and `d = 0` optimal — the marginal gain from the
    # first deposited dollar, `beta*w_b*(1 + chi)`, must not exceed the marginal
    # value of the liquid dollar it costs, `u'(c) = beta*w_a`.
    valid_region = (a > 0.0) & (w_b * (1.0 + match_rate) <= w_a)
    return RegionCloud(
        m_endog=a + consumption,
        n_endog=b,
        consumption=consumption,
        deposit=jnp.zeros_like(consumption),
        value=value,
        value_grad_m=discount_factor * w_a,
        value_grad_n=discount_factor * w_b,
        valid_region=valid_region,
    )


def invert_acon_cloud(
    *,
    consumption: FloatND,
    b: FloatND,
    post_decision_value_at_zero_a: FloatND,
    w_b_at_zero_a: FloatND,
    w_a_at_zero_a: FloatND,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
) -> RegionCloud:
    """Invert the deposit FOC on the borrowing-constrained region (`acon`, `a = 0`).

    Where the liquid borrowing constraint binds the liquid post-decision balance is
    pinned at `a = 0`, so the liquid Euler holds only with a non-negative multiplier and
    cannot be inverted for consumption. Instead the region is parameterized by a
    consumption grid axis against the pension post-decision balance `b` at `a = 0`:
    consumption is exogenous, and the still-interior deposit is recovered from its FOC
    `u'(c) = beta * w_b * (1 + chi / (1 + d))`. The liquid budget at the corner is
    `m = c + d` (no liquid savings), and the pension budget inverts to
    `n = b - d - chi*log(1 + d)`.

    Args:
        consumption: Exogenous consumption sweep at each node (`a = 0` corner).
        b: Pension post-decision balance at each node (evaluated at `a = 0`).
        post_decision_value_at_zero_a: Post-decision value `w(0, b)` at each node.
        w_b_at_zero_a: Post-decision value gradient w.r.t. `b`, at `a = 0`.
        w_a_at_zero_a: Post-decision value gradient w.r.t. `a`, at `a = 0`; enters
            only the complementary-slackness test for the binding borrowing limit.
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        match_rate: Pension employer-match coefficient `chi`.

    Returns:
        The borrowing-constrained endogenous cloud.

    """
    marginal_utility = consumption ** (-crra)
    # Deposit FOC at the corner: u'(c) = beta*w_b*(1 + chi/(1 + d)); solve for d.
    deposit_ratio = marginal_utility / (discount_factor * w_b_at_zero_a)
    deposit = match_rate / (deposit_ratio - 1.0) - 1.0
    value = (
        _crra_utility(consumption, crra)
        + discount_factor * post_decision_value_at_zero_a
    )
    # KKT: the borrowing limit binds with a non-negative multiplier, i.e. the agent
    # would consume more liquid if it could (`u'(c) >= beta*w_a` at a = 0), and the
    # deposit stays interior (`d > 0`).
    valid_region = (marginal_utility >= discount_factor * w_a_at_zero_a) & (
        deposit > 0.0
    )
    return RegionCloud(
        # a = 0 -> m = c + d; pension budget inverts as in the unconstrained region.
        m_endog=consumption + deposit,
        n_endog=b - deposit - match_rate * jnp.log1p(deposit),
        consumption=consumption,
        deposit=deposit,
        value=value,
        # At the binding budget the marginal value of liquid wealth is the common
        # marginal value u'(c) (equal to the deposit FOC's right-hand side).
        value_grad_m=marginal_utility,
        value_grad_n=discount_factor * w_b_at_zero_a,
        valid_region=valid_region,
    )


def invert_con_cloud(
    *,
    consumption: FloatND,
    b: FloatND,
    post_decision_value_at_zero_a: FloatND,
    w_b_at_zero_a: FloatND,
    w_a_at_zero_a: FloatND,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
) -> RegionCloud:
    """Build the fully-constrained corner cloud (`con`, `a = 0` and `d = 0`).

    Where both the borrowing constraint and the deposit lower bound bind, neither FOC
    holds with equality and there is nothing to invert: the agent consumes its entire
    liquid budget (`m = c`, since `a = 0` and `d = 0`) and the pension is unchanged
    (`n = b`). The region is the deep-constrained corner of the state space,
    parameterized by a consumption grid axis against the pension balance at `a = 0`.

    Args:
        consumption: Exogenous consumption sweep at each node (`m = c` at the corner).
        b: Pension post-decision balance at each node (`n = b`, no deposit).
        post_decision_value_at_zero_a: Post-decision value `w(0, b)` at each node.
        w_b_at_zero_a: Post-decision value gradient w.r.t. `b`, at `a = 0`.
        w_a_at_zero_a: Post-decision value gradient w.r.t. `a`, at `a = 0`; enters
            only the complementary-slackness test for the binding borrowing limit.
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        match_rate: Pension employer-match coefficient `chi`; enters only the
            complementary-slackness test for `d = 0` being optimal.

    Returns:
        The fully-constrained endogenous cloud.

    """
    marginal_utility = consumption ** (-crra)
    value = (
        _crra_utility(consumption, crra)
        + discount_factor * post_decision_value_at_zero_a
    )
    # KKT: both corners bind with non-negative multipliers — the borrowing limit
    # (`u'(c) >= beta*w_a` at a = 0) and the deposit lower bound (`d = 0` optimal,
    # `u'(c) >= beta*w_b*(1 + chi)` at a = 0).
    valid_region = (marginal_utility >= discount_factor * w_a_at_zero_a) & (
        marginal_utility >= discount_factor * w_b_at_zero_a * (1.0 + match_rate)
    )
    return RegionCloud(
        m_endog=consumption,
        n_endog=b,
        consumption=consumption,
        deposit=jnp.zeros_like(consumption),
        value=value,
        # All extra liquid is consumed at the corner, so dV/dm = u'(c).
        value_grad_m=marginal_utility,
        value_grad_n=discount_factor * w_b_at_zero_a,
        valid_region=valid_region,
    )


def _crra_utility(consumption: FloatND, crra: ScalarFloat | float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
