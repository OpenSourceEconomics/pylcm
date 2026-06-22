"""One multi-segment 2-D G2EGM upper-envelope step for the two-asset model.

Assembles the four KKT constraint segments (`ucon`, `dcon`, `acon`, `con`) into a
single backward step. From next period's value on the regular $(m, n)$ grid it builds
each segment's candidate cloud by that segment's closed-form inverse Euler step,
triangulates the clouds into meshes in the current-state plane, and selects — at every
common $(m, n)$ target — the best feasible policy across all segments by recomputing
the Bellman objective: the first (within-segment) upper envelope over each segment's
admissible triangles, then the second (across-segment) maximum.

Unlike the `ucon`-only step (`two_asset_step.egm_step`), the constrained corners — low
liquid wealth (borrowing binds, `acon`/`con`) and low pension (deposit binds, `dcon`) —
are covered by their own segments rather than left to the unconstrained cloud's
poisoning extrapolation. `ucon`/`dcon` are built on the post-decision $(a, b)$ grid;
`acon`/`con` are built on the $(consumption, b)$ grid at $a = 0$.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.mesh_envelope import ObjectiveEvaluator, first_envelope, second_envelope
from _lcm.egm.two_asset_inverse import (
    invert_acon_cloud,
    invert_con_cloud,
    invert_dcon_cloud,
    invert_ucon_cloud,
)
from _lcm.egm.two_asset_objective import build_two_asset_objective
from _lcm.egm.two_asset_post_decision import post_decision_value_and_grad
from _lcm.egm.two_asset_segment_mesh import build_segment_mesh
from lcm.typing import Float1D, FloatND


def g2egm_step(
    *,
    next_value: FloatND,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    discount_factor: float,
    crra: float,
    match_rate: float,
    return_liquid: float,
    return_pension: float,
    wage: float,
    threshold: float = 0.25,
) -> FloatND:
    """Solve one period of the two-asset model by the four-segment G2EGM envelope.

    Args:
        next_value: Next period's value on the regular `(m, n)` grid.
        m_grid: Regular liquid-state grid (ascending, evenly spaced).
        n_grid: Regular pension-state grid (ascending, evenly spaced).
        a_grid: Liquid post-decision grid for `ucon`/`dcon` (should include 0 so the
            objective reads `W(0, b)` accurately at the borrowing-constrained corner).
        b_grid: Pension post-decision grid (shared by all segments).
        consumption_grid: Consumption sweep for `acon`/`con` at `a = 0`.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        match_rate: Pension employer-match coefficient `chi`.
        return_liquid: Liquid net return `r^a`.
        return_pension: Pension net return `r^b`.
        wage: Deterministic labor income.
        threshold: Barycentric extrapolation tolerance for triangle admissibility.

    Returns:
        This period's upper-envelope value on the regular `(m, n)` grid.

    """
    # ucon / dcon: candidate clouds on the (a, b) post-decision grid.
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
    )

    # acon / con: candidate clouds on the (consumption, b) grid at a = 0.
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
    acon = invert_acon_cloud(
        consumption=c_mesh,
        b=cb_mesh,
        post_decision_value_at_zero_a=value_at_zero,
        w_b_at_zero_a=grad_b_at_zero,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )
    con = invert_con_cloud(
        consumption=c_mesh,
        b=cb_mesh,
        post_decision_value_at_zero_a=value_at_zero,
        w_b_at_zero_a=grad_b_at_zero,
        discount_factor=discount_factor,
        crra=crra,
    )

    meshes = [
        build_segment_mesh(cloud=ucon, region_label=0),
        build_segment_mesh(cloud=dcon, region_label=1),
        build_segment_mesh(cloud=acon, region_label=2),
        build_segment_mesh(cloud=con, region_label=3),
    ]

    objective = build_two_asset_objective(
        post_decision_value=post.value,
        a_grid=a_grid,
        b_grid=b_grid,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
    )

    m_mesh, n_mesh = jnp.meshgrid(m_grid, n_grid, indexing="ij")
    targets = jnp.stack([m_mesh.reshape(-1), n_mesh.reshape(-1)], axis=1)

    segment_values = []
    segment_policies = []
    for mesh in meshes:
        values, policies = first_envelope(
            mesh=mesh, targets=targets, objective=objective, threshold=threshold
        )
        segment_values.append(values)
        segment_policies.append(policies)
    result = second_envelope(
        segment_values=jnp.stack(segment_values, axis=0),
        segment_policies=jnp.stack(segment_policies, axis=0),
    )

    # Targets no admissible segment candidate reaches (envelope value `-inf`) are filled
    # by a direct-Bellman search over a coarse policy grid — the v3 hole-fill. A target
    # whose optimal post-state leaves the post-decision grid stays a hole (no in-domain
    # candidate exists), which is a grid-coverage limit, not an algorithm gap.
    deposit_grid = jnp.linspace(0.0, b_grid[-1] - b_grid[0], consumption_grid.shape[0])
    hole_value = _direct_bellman_fill(
        targets=targets,
        objective=objective,
        consumption_grid=consumption_grid,
        deposit_grid=deposit_grid,
    )
    filled = jnp.where(jnp.isfinite(result.value), result.value, hole_value)
    return filled.reshape(m_mesh.shape)


def _direct_bellman_fill(
    *,
    targets: FloatND,
    objective: ObjectiveEvaluator,
    consumption_grid: Float1D,
    deposit_grid: Float1D,
) -> FloatND:
    """Maximize the recomputed objective over a coarse policy grid, per target.

    The fallback for common-grid targets no segment mesh covers: a direct-Bellman search
    over the `(consumption, deposit)` product grid, masking infeasible candidates. The
    result is `-inf` only where every coarse candidate is infeasible (the optimal
    post-state leaves the post-decision grid).
    """
    c_mesh, d_mesh = jnp.meshgrid(consumption_grid, deposit_grid, indexing="ij")
    candidates = jnp.stack([c_mesh.reshape(-1), d_mesh.reshape(-1)], axis=1)

    def at_target(target: FloatND) -> FloatND:
        def per_candidate(policy: FloatND) -> FloatND:
            value, feasible = objective(target, policy)
            return jnp.where(feasible & jnp.isfinite(value), value, -jnp.inf)

        return jnp.max(jax.vmap(per_candidate)(candidates))

    return jax.vmap(at_target)(targets)
