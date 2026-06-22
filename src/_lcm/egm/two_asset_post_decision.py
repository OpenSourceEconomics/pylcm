"""Post-decision value and gradients for the deterministic two-asset model.

The inverse-Euler step consumes the post-decision value $w(a, b)$ and its gradients
$w_a, w_b$. For the deterministic two-asset model the post-decision value is next
period's value read at the transformed states $m' = (1 + r^a)\\,a + \\text{wage}$ and
$n' = (1 + r^b)\\,b$ (liquid and pension carry forward at their gross returns), and
the gradients follow by the chain rule: $w_a = \\partial_m V'\\,(1 + r^a)$,
$w_b = \\partial_n V'\\,(1 + r^b)$.

Value and gradient come from the same bilinear interpolant of $V'$ on the regular
$(m, n)$ state grid, so they are mutually consistent (the gradient is the derivative
of the value read). The interpolation is exact for an affine $V'$.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from lcm.typing import Float1D, FloatND


class PostDecision(NamedTuple):
    """Post-decision value and gradients on a post-decision $(a, b)$ grid."""

    value: FloatND
    """Post-decision value `w(a, b) = V'(m'(a), n'(b))`."""
    grad_a: FloatND
    """`w_a = d/da V'(m'(a), n'(b))`."""
    grad_b: FloatND
    """`w_b = d/db V'(m'(a), n'(b))`."""


def post_decision_value_and_grad(
    *,
    next_value: FloatND,
    m_grid: Float1D,
    n_grid: Float1D,
    a: FloatND,
    b: FloatND,
    return_liquid: float,
    return_pension: float,
    wage: float,
) -> PostDecision:
    """Evaluate the post-decision value and its gradients on the `(a, b)` grid.

    Args:
        next_value: Next period's value on the regular `(m, n)` grid, shape
            `(len(m_grid), len(n_grid))`.
        m_grid: Regular liquid-state grid (ascending, evenly spaced).
        n_grid: Regular pension-state grid (ascending, evenly spaced).
        a: Liquid post-decision balance at each node.
        b: Pension post-decision balance at each node.
        return_liquid: Liquid gross return minus one, `r^a`.
        return_pension: Pension gross return minus one, `r^b`.
        wage: Deterministic labor income added to next-period liquid wealth.

    Returns:
        Post-decision value and gradients, one entry per `(a, b)` node.

    """

    def value_at(a_node: FloatND, b_node: FloatND) -> FloatND:
        m_next = (1.0 + return_liquid) * a_node + wage
        n_next = (1.0 + return_pension) * b_node
        m_index = (m_next - m_grid[0]) / (m_grid[1] - m_grid[0])
        n_index = (n_next - n_grid[0]) / (n_grid[1] - n_grid[0])
        return map_coordinates(next_value, [m_index, n_index], order=1, mode="nearest")

    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    value = jax.vmap(value_at)(flat_a, flat_b)
    grad_a, grad_b = jax.vmap(jax.grad(value_at, argnums=(0, 1)))(flat_a, flat_b)
    return PostDecision(
        value=value.reshape(a.shape),
        grad_a=grad_a.reshape(a.shape),
        grad_b=grad_b.reshape(a.shape),
    )


def post_decision_value_and_grad_retiring(
    *,
    next_value_retired: Float1D,
    next_marginal_retired: Float1D,
    liquid_grid: Float1D,
    a: FloatND,
    b: FloatND,
    return_liquid: float,
    pension_payout_return: float,
    retirement_income: float,
) -> PostDecision:
    """Post-decision value and gradients at the working->retired boundary.

    On the working->retired transition the pension is paid out as a lump sum, so both
    post-decision balances feed a single retired liquid state
    $\\ell' = (1 + r^a)\\,a + \\pi\\,b + y_{\\text{ret}}$ ($\\pi$ the pension payout
    return, $y_{\\text{ret}}$ the first retirement income). The post-decision value is
    the 1-D retired value read there, $w(a, b) = V_{\\text{ret}}(\\ell')$, and the
    gradients follow by the chain rule with the **carried** retired marginal
    $V'_{\\text{ret}}$ (exact, not a finite difference of the value array):
    $w_a = (1 + r^a)\\,V'_{\\text{ret}}$ and $w_b = \\pi\\,V'_{\\text{ret}}$.

    Args:
        next_value_retired: Next period's retired value on `liquid_grid`, shape
            `(len(liquid_grid),)`.
        next_marginal_retired: Next period's retired marginal value of liquid on
            `liquid_grid`, shape `(len(liquid_grid),)`.
        liquid_grid: Regular retired liquid-state grid (ascending, evenly spaced).
        a: Liquid post-decision balance at each node.
        b: Pension post-decision balance at each node.
        return_liquid: Liquid gross return minus one, `r^a`.
        pension_payout_return: Factor the pension balance is paid out at on retirement.
        retirement_income: First retirement income added to the retired liquid state.

    Returns:
        Post-decision value and gradients, one entry per `(a, b)` node.

    """
    gross_return = 1.0 + return_liquid
    origin = liquid_grid[0]
    step = liquid_grid[1] - liquid_grid[0]

    def read_at(liquid_next: FloatND, values: Float1D) -> FloatND:
        index = (liquid_next - origin) / step
        return map_coordinates(values, [jnp.atleast_1d(index)], order=1, mode="nearest")

    liquid_next = gross_return * a + pension_payout_return * b + retirement_income
    flat_liquid = liquid_next.reshape(-1)
    value = read_at(flat_liquid, next_value_retired)
    marginal = read_at(flat_liquid, next_marginal_retired)
    return PostDecision(
        value=value.reshape(a.shape),
        grad_a=(gross_return * marginal).reshape(a.shape),
        grad_b=(pension_payout_return * marginal).reshape(a.shape),
    )
