"""One backward 2-D EGM step for the two-asset model (unconstrained interior).

Assembles the verified pieces into a single period of the multidimensional EGM
solve: from next period's value on the regular `(m, n)` grid, compute the
post-decision value and gradients, invert the Euler equations to the endogenous
cloud, and rasterize that cloud back onto the regular `(m, n)` grid with the
inverse-bilinear locator. This is the `ucon` (single-segment) case — no envelope.

Where the regular target is outside the endogenous cloud (the corner where the
borrowing or deposit constraint binds), the locator falls back to the nearest
boundary cell, so the published value there is an extrapolation rather than the
constrained solution; the value matches a dense grid-search solve only on the
unconstrained interior the cloud covers.
"""

import jax.numpy as jnp

from _lcm.egm.cell_locator import locate_in_quad_mesh, read_bilinear
from _lcm.egm.two_asset_inverse import invert_ucon_cloud
from _lcm.egm.two_asset_post_decision import post_decision_value_and_grad
from lcm.typing import Float1D, FloatND


def egm_step(
    *,
    next_value: FloatND,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    discount_factor: float,
    crra: float,
    match_rate: float,
    return_liquid: float,
    return_pension: float,
    wage: float,
) -> FloatND:
    """Solve one period of the two-asset model by 2-D EGM on the unconstrained region.

    Args:
        next_value: Next period's value on the regular `(m, n)` grid.
        m_grid: Regular liquid-state grid.
        n_grid: Regular pension-state grid.
        a_grid: Liquid post-decision grid.
        b_grid: Pension post-decision grid.
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        match_rate: Pension employer-match coefficient.
        return_liquid: Liquid net return `r^a`.
        return_pension: Pension net return `r^b`.
        wage: Deterministic labor income.

    Returns:
        This period's value on the regular `(m, n)` grid.

    """
    a_mesh, b_mesh = jnp.meshgrid(a_grid, b_grid, indexing="ij")
    post = post_decision_value_and_grad(
        next_value=next_value,
        m_grid=m_grid,
        n_grid=n_grid,
        a=a_mesh,
        b=b_mesh,
        return_liquid=return_liquid,
        return_pension=return_pension,
        wage=wage,
    )
    cloud = invert_ucon_cloud(
        a=a_mesh,
        b=b_mesh,
        w_a=post.grad_a,
        w_b=post.grad_b,
        post_decision_value=post.value,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    m_mesh, n_mesh = jnp.meshgrid(m_grid, n_grid, indexing="ij")
    queries = jnp.stack([m_mesh.reshape(-1), n_mesh.reshape(-1)], axis=1)
    located = locate_in_quad_mesh(
        m_image=cloud.m_endog, n_image=cloud.n_endog, queries=queries
    )
    read = read_bilinear(node_values=cloud.value, located=located)
    return read.value.reshape(m_mesh.shape)
