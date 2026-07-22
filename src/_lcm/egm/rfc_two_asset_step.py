"""One two-asset backward step that selects the upper envelope by the 2-D RFC.

The Roof-top Cut alternative to the G2EGM mesh envelope (`two_asset_g2egm_step`). It
builds the same four KKT candidate clouds (`ucon`, `dcon`, `acon`, `con`) by the
closed-form inverse Euler step, then — instead of triangulating each segment and taking
within- and across-segment maxima — merges all four clouds into one and applies the
global rooftop-cut delete (`rfc_delete_mask_2d`) followed by a single local-simplex
barycentric publish (`rfc_publish_2d`) at the regular `(m, n)` targets. This is the
Dobrescu-Shanker 2024 multidimensional method (their Box 1): a combined-cloud
delete-then-interpolate-once, with no per-segment republication.

Infeasible cloud points (NaN/inf value, state, or supgradient at a KKT corner that does
not bind) are marked invalid: their states are pushed to a far sentinel and value to
`-inf` so they neither dominate nor are published, and the validity mask is passed to
both the cut and the publisher.
"""

import jax.numpy as jnp

from _lcm.egm.two_asset_g2egm_step import G2EGMResult
from _lcm.egm.two_asset_inverse import (
    RegionCloud,
    invert_acon_cloud,
    invert_con_cloud,
    invert_dcon_cloud,
    invert_ucon_cloud,
)
from _lcm.egm.two_asset_post_decision import post_decision_value_and_grad
from _lcm.egm.upper_envelope.rfc_2d import (
    J_BAR_DEFAULT,
    K_PUBLISH_DEFAULT,
    RADIUS_DEFAULT,
    rfc_delete_mask_2d,
    rfc_publish_2d,
)
from lcm.typing import Float1D, FloatND, ScalarFloat

_FAR_SENTINEL = 1e9


def rfc_two_asset_step(
    *,
    next_value: FloatND,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    return_pension: ScalarFloat | float,
    wage: ScalarFloat | float,
    j_bar: float = J_BAR_DEFAULT,
    radius: float = RADIUS_DEFAULT,
    k_cut: int = 5,
    k_publish: int = K_PUBLISH_DEFAULT,
) -> G2EGMResult:
    """Solve one two-asset period by the combined-cloud 2-D RFC upper envelope.

    Args:
        next_value: Next period's value on the regular `(m, n)` grid.
        m_grid: Regular liquid-state grid (ascending, evenly spaced).
        n_grid: Regular pension-state grid (ascending, evenly spaced).
        a_grid: Liquid post-decision grid for `ucon`/`dcon` (should include 0).
        b_grid: Pension post-decision grid (shared by all segments).
        consumption_grid: Consumption sweep for `acon`/`con` at `a = 0`.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        match_rate: Pension employer-match coefficient `chi`.
        return_liquid: Liquid net return `r^a`.
        return_pension: Pension net return `r^b`.
        wage: Deterministic labor income.
        j_bar: RFC policy-jump threshold.
        radius: RFC neighbour distance threshold.
        k_cut: RFC neighbours (including self) the delete inspects.
        k_publish: Nearest survivors forming the publish local-simplex set.

    Returns:
        This period's upper-envelope value and policy on the regular `(m, n)` grid.
    """
    clouds = _build_region_clouds(
        next_value=next_value,
        m_grid=m_grid,
        n_grid=n_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        consumption_grid=consumption_grid,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
        return_liquid=return_liquid,
        return_pension=return_pension,
        wage=wage,
    )

    states = jnp.concatenate(
        [
            jnp.stack([c.m_endog.reshape(-1), c.n_endog.reshape(-1)], axis=1)
            for c in clouds
        ]
    )
    values = jnp.concatenate([c.value.reshape(-1) for c in clouds])
    supgradients = jnp.concatenate(
        [
            jnp.stack([c.value_grad_m.reshape(-1), c.value_grad_n.reshape(-1)], axis=1)
            for c in clouds
        ]
    )
    policies = jnp.concatenate(
        [
            jnp.stack([c.consumption.reshape(-1), c.deposit.reshape(-1)], axis=1)
            for c in clouds
        ]
    )

    # Each region cloud carries a `valid_region` KKT mask: the complementary-
    # slackness test its inverse assumes, false at finite-but-KKT-inconsistent
    # candidates (a region's FOCs solved where its inequalities do not hold). These
    # are dropped here so only genuine KKT candidates enter the cut and publish.
    # Dropping them thins the valid survivors at coarse grids, so the publisher
    # selects the *smallest* containing simplex (`rfc_publish_2d`) to keep the
    # affine fit local across the holes the mask opens.
    region_valid = jnp.concatenate([c.valid_region.reshape(-1) for c in clouds])
    valid = (
        jnp.isfinite(values)
        & jnp.all(jnp.isfinite(states), axis=1)
        & jnp.all(jnp.isfinite(supgradients), axis=1)
        & jnp.all(jnp.isfinite(policies), axis=1)
        & region_valid
    )
    # Sanitize invalid candidates so they neither dominate (value -inf, gradient 0) nor
    # sit near any target (state at a far sentinel); the validity mask keeps them out of
    # both the cut and the publish.
    safe_states = jnp.where(valid[:, None], states, _FAR_SENTINEL)
    safe_values = jnp.where(valid, values, -jnp.inf)
    safe_supgradients = jnp.where(valid[:, None], supgradients, 0.0)
    safe_policies = jnp.where(valid[:, None], policies, 0.0)

    # The reference cut constants (radius, J_bar) assume O(1) state and policy scales,
    # so normalize the cut to unit grid ranges: states by the state range, supgradients
    # by the state range (chain rule, which leaves the below-tangent test scale-
    # invariant), and the consumption selector by its range, so radius and the jump
    # ratio are meaningful on the pension state space. The publish stays in original
    # units (its barycentric local-simplex is affine-invariant).
    state_scale = jnp.array([m_grid[-1] - m_grid[0], n_grid[-1] - n_grid[0]])
    policy_scale = jnp.array([consumption_grid[-1] - consumption_grid[0], 1.0])
    keep = rfc_delete_mask_2d(
        states=safe_states / state_scale,
        supgradients=safe_supgradients * state_scale,
        values=safe_values,
        policies=safe_policies / policy_scale,
        j_bar=j_bar,
        radius=radius,
        k=k_cut,
    )
    survives = keep & valid

    m_mesh, n_mesh = jnp.meshgrid(m_grid, n_grid, indexing="ij")
    targets = jnp.stack([m_mesh.reshape(-1), n_mesh.reshape(-1)], axis=1)
    published_value, published_policy = rfc_publish_2d(
        survivor_states=safe_states,
        survivor_values=safe_values,
        survivor_policies=safe_policies,
        target_states=targets,
        valid=survives,
        k=k_publish,
    )

    return G2EGMResult(
        value=published_value.reshape(m_mesh.shape),
        consumption=published_policy[:, 0].reshape(m_mesh.shape),
        deposit=published_policy[:, 1].reshape(m_mesh.shape),
    )


def _build_region_clouds(
    *,
    next_value: FloatND,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    return_pension: ScalarFloat | float,
    wage: ScalarFloat | float,
) -> tuple[RegionCloud, RegionCloud, RegionCloud, RegionCloud]:
    """Build the four KKT candidate clouds, identical to the G2EGM step's clouds."""
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
    ucon = invert_ucon_cloud(
        a=a_mesh,
        b=b_mesh,
        w_a=post.grad_a,
        w_b=post.grad_b,
        post_decision_value=post.value,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    dcon = invert_dcon_cloud(
        a=a_mesh,
        b=b_mesh,
        w_a=post.grad_a,
        w_b=post.grad_b,
        post_decision_value=post.value,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    a_zero = jnp.zeros_like(b_grid)
    post_zero = post_decision_value_and_grad(
        next_value=next_value,
        m_grid=m_grid,
        n_grid=n_grid,
        a=a_zero,
        b=b_grid,
        return_liquid=return_liquid,
        return_pension=return_pension,
        wage=wage,
    )
    c_mesh, cb_mesh = jnp.meshgrid(consumption_grid, b_grid, indexing="ij")
    value_at_zero = jnp.broadcast_to(post_zero.value[None, :], c_mesh.shape)
    grad_b_at_zero = jnp.broadcast_to(post_zero.grad_b[None, :], c_mesh.shape)
    grad_a_at_zero = jnp.broadcast_to(post_zero.grad_a[None, :], c_mesh.shape)
    acon = invert_acon_cloud(
        consumption=c_mesh,
        b=cb_mesh,
        post_decision_value_at_zero_a=value_at_zero,
        w_b_at_zero_a=grad_b_at_zero,
        w_a_at_zero_a=grad_a_at_zero,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    con = invert_con_cloud(
        consumption=c_mesh,
        b=cb_mesh,
        post_decision_value_at_zero_a=value_at_zero,
        w_b_at_zero_a=grad_b_at_zero,
        w_a_at_zero_a=grad_a_at_zero,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    # A candidate built from a continuation that the reader fabricated by clamping an
    # out-of-domain transformed state (`post.valid` False) is dropped from its KKT
    # region, mirroring the G2EGM objective's `post_decision_valid` gate. `ucon`/`dcon`
    # read the full `(a, b)` mesh; `acon`/`con` read the `a = 0` slice over `b_grid`.
    ucon = ucon._replace(valid_region=ucon.valid_region & post.valid)
    dcon = dcon._replace(valid_region=dcon.valid_region & post.valid)
    acon = acon._replace(valid_region=acon.valid_region & post_zero.valid[None, :])
    con = con._replace(valid_region=con.valid_region & post_zero.valid[None, :])
    return ucon, dcon, acon, con
