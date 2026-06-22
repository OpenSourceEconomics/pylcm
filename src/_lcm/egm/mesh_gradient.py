"""Region-aware envelope gradient and the cross-segment switch mask.

The upper envelope publishes the marginal value of wealth at the selected policy. For
the two-asset model the envelope theorem gives, at the optimum,

$$V_m = u'(c), \\qquad V_n = \\beta\\, W_b,$$

with $u'(c) = c^{-\\rho}$. This holds in **every** KKT region: where the borrowing
constraint binds ($a = 0$) the consumption FOC is $u'(c) = \\beta w_a + \\mu_a$ with a
non-negative multiplier $\\mu_a$, so $V_m = u'(c)$ but $V_m \\ne \\beta w_a$. Publishing
$\\beta w_a$ there understates the marginal value. Using $u'(c)$ directly is correct in
all regions and matches the region inverses' `value_grad_m`.

At a **cross-segment switch** the pointwise maximum of the segment value functions is
generally not differentiable, so no unique gradient exists. The switch mask flags every
target whose winning segment differs from a grid neighbour's; the published gradient
there is the winning branch's one-sided derivative, explicitly marked.
"""

from typing import NamedTuple

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, IntND


class RegionGradient(NamedTuple):
    """The region-aware marginal value of liquid and pension wealth."""

    value_grad_m: FloatND
    """`V_m = u'(c) = c**(-crra)`, valid in every KKT region."""
    value_grad_n: FloatND
    """`V_n = discount_factor * W_b`."""


def region_aware_gradient(
    *,
    consumption: FloatND,
    post_decision_grad_b: FloatND,
    discount_factor: float,
    crra: float,
) -> RegionGradient:
    """Compute the envelope marginal values from the selected policy.

    Args:
        consumption: Selected consumption `c` per target.
        post_decision_grad_b: Post-decision value gradient `W_b` at the selected policy.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.

    Returns:
        The marginal value of liquid wealth `u'(c)` and of pension wealth
        `discount_factor * W_b`, region-independent.

    """
    return RegionGradient(
        value_grad_m=consumption ** (-crra),
        value_grad_n=discount_factor * post_decision_grad_b,
    )


def switch_mask(*, segment: IntND, grid_shape: tuple[int, int]) -> BoolND:
    """Flag targets at a cross-segment switch (the envelope is non-differentiable).

    A target is flagged when its winning segment differs from any of its four grid
    neighbours: the pointwise-max envelope has a kink across that boundary, so the
    published gradient is one-sided there.

    Args:
        segment: Winning segment index per target, shape `(n_m * n_n,)`, row-major over
            the `(m, n)` grid.
        grid_shape: The `(n_m, n_n)` shape the targets were flattened from.

    Returns:
        Boolean mask per target, shape `(n_m * n_n,)`, `True` at a switch.

    """
    seg = segment.reshape(grid_shape)
    mask = jnp.zeros(grid_shape, dtype=bool)
    row_diff = seg[1:, :] != seg[:-1, :]
    mask = mask.at[1:, :].set(mask[1:, :] | row_diff)
    mask = mask.at[:-1, :].set(mask[:-1, :] | row_diff)
    col_diff = seg[:, 1:] != seg[:, :-1]
    mask = mask.at[:, 1:].set(mask[:, 1:] | col_diff)
    mask = mask.at[:, :-1].set(mask[:, :-1] | col_diff)
    return mask.reshape(-1)
