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

from collections.abc import Callable
from typing import NamedTuple

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
from _lcm.egm.two_asset_post_decision import (
    PostDecision,
    post_decision_value_and_grad,
    post_decision_value_and_grad_retiring,
)
from _lcm.egm.two_asset_segment_mesh import build_segment_mesh
from lcm.typing import Float1D, Float2D, FloatND, ScalarFloat

# Maps a post-decision `(a, b)` mesh to its value and gradients. The working step reads
# next period's working value on the `(m, n)` grid; the retirement-boundary step reads
# the 1-D retired value through the lump-sum payout. The envelope core is identical
# either way — only this reader differs.
PostDecisionReader = Callable[[FloatND, FloatND], PostDecision]


class G2EGMResult(NamedTuple):
    """One G2EGM step's value and policy on the regular working `(m, n)` grid."""

    value: FloatND
    """This period's upper-envelope value, shape `(len(m_grid), len(n_grid))`."""
    consumption: FloatND
    """Optimal consumption per `(m, n)` target (invalid at uncovered holes)."""
    deposit: FloatND
    """Optimal pension deposit per `(m, n)` target (invalid at uncovered holes)."""


def g2egm_step(
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
    threshold: float = 0.25,
) -> G2EGMResult:
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
        This period's upper-envelope value and policy on the regular `(m, n)` grid.

    """

    def post_reader(a: FloatND, b: FloatND) -> PostDecision:
        return post_decision_value_and_grad(
            next_value=next_value,
            m_grid=m_grid,
            n_grid=n_grid,
            a=a,
            b=b,
            return_liquid=return_liquid,
            return_pension=return_pension,
            wage=wage,
        )

    return _g2egm_envelope_step(
        post_reader=post_reader,
        m_grid=m_grid,
        n_grid=n_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        consumption_grid=consumption_grid,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
        threshold=threshold,
    )


def g2egm_retiring_step(
    *,
    next_value_retired: Float1D,
    next_marginal_retired: Float1D,
    liquid_grid: Float1D,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    pension_payout_return: ScalarFloat | float,
    retirement_income: ScalarFloat | float,
    threshold: float = 0.25,
) -> G2EGMResult:
    """Solve the working->retired boundary period by the four-segment G2EGM envelope.

    Identical to `g2egm_step` except the post-decision continuation is the 1-D retired
    value read through the lump-sum payout (`post_decision_value_and_grad_retiring`):
    on retirement the pension is paid out into liquid, so both post-decision balances
    feed a single retired liquid state. The endogenous-grid machinery, segments, and
    envelope are unchanged.

    Args:
        next_value_retired: Next period's retired value on `liquid_grid`.
        next_marginal_retired: Next period's retired marginal value of liquid on
            `liquid_grid`.
        liquid_grid: Regular retired liquid-state grid (ascending, evenly spaced).
        m_grid: Regular working liquid-state grid (ascending, evenly spaced).
        n_grid: Regular working pension-state grid (ascending, evenly spaced).
        a_grid: Liquid post-decision grid for `ucon`/`dcon`.
        b_grid: Pension post-decision grid (shared by all segments).
        consumption_grid: Consumption sweep for `acon`/`con` at `a = 0`.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        match_rate: Pension employer-match coefficient `chi`.
        return_liquid: Liquid net return `r^a`.
        pension_payout_return: Factor the pension balance is paid out at on retirement.
        retirement_income: First retirement income added to the retired liquid state.
        threshold: Barycentric extrapolation tolerance for triangle admissibility.

    Returns:
        This period's upper-envelope value and policy on the working `(m, n)` grid.

    """

    def post_reader(a: FloatND, b: FloatND) -> PostDecision:
        return post_decision_value_and_grad_retiring(
            next_value_retired=next_value_retired,
            next_marginal_retired=next_marginal_retired,
            liquid_grid=liquid_grid,
            a=a,
            b=b,
            return_liquid=return_liquid,
            pension_payout_return=pension_payout_return,
            retirement_income=retirement_income,
        )

    return _g2egm_envelope_step(
        post_reader=post_reader,
        m_grid=m_grid,
        n_grid=n_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        consumption_grid=consumption_grid,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
        threshold=threshold,
    )


def _g2egm_envelope_step(
    *,
    post_reader: PostDecisionReader,
    m_grid: Float1D,
    n_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
    threshold: float,
) -> G2EGMResult:
    """Run the four-segment G2EGM envelope given a post-decision reader.

    The reader supplies the post-decision value and gradients on the `(a, b)` mesh; the
    rest — the four KKT-segment inverses, triangulated meshes, within- and
    across-segment envelopes, and the direct-Bellman hole-fill — is independent of
    whether the continuation is the working or the retired value.
    """
    # ucon / dcon: candidate clouds on the (a, b) post-decision grid.
    a_mesh, b_mesh = jnp.meshgrid(a_grid, b_grid, indexing="ij")
    post = post_reader(a_mesh, b_mesh)
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

    # acon / con: candidate clouds on the (consumption, b) grid at a = 0.
    a_zero = jnp.zeros_like(b_grid)
    post_zero = post_reader(a_zero, b_grid)
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
        region_labels=jnp.array([m.region_label for m in meshes], dtype=jnp.int32),
    )

    # Targets no admissible segment candidate reaches (envelope value `-inf`) are filled
    # by a direct-Bellman search over a coarse policy grid — the v3 hole-fill. A target
    # whose optimal post-state leaves the post-decision grid stays a hole (no in-domain
    # candidate exists), which is a grid-coverage limit, not an algorithm gap.
    deposit_grid = jnp.linspace(0.0, b_grid[-1] - b_grid[0], consumption_grid.shape[0])
    hole_value, hole_policy = _direct_bellman_fill(
        targets=targets,
        objective=objective,
        consumption_grid=consumption_grid,
        deposit_grid=deposit_grid,
    )
    # A target the segment envelope misses takes the hole-fill's value *and* its policy,
    # so the published consumption and deposit stay consistent with the published value
    # rather than the stale failed-envelope policy.
    is_hole = ~jnp.isfinite(result.value)
    filled_value = jnp.where(is_hole, hole_value, result.value)
    filled_policy = jnp.where(is_hole[:, None], hole_policy, result.policy)
    return G2EGMResult(
        value=filled_value.reshape(m_mesh.shape),
        consumption=filled_policy[:, 0].reshape(m_mesh.shape),
        deposit=filled_policy[:, 1].reshape(m_mesh.shape),
    )


def _direct_bellman_fill(
    *,
    targets: FloatND,
    objective: ObjectiveEvaluator,
    consumption_grid: Float1D,
    deposit_grid: Float1D,
) -> tuple[FloatND, Float2D]:
    """Maximize the recomputed objective over a coarse policy grid, per target.

    The fallback for common-grid targets no segment mesh covers: a direct-Bellman search
    over the `(consumption, deposit)` product grid, masking infeasible candidates.

    Returns the per-target maximizing value and the `(consumption, deposit)` candidate
    that attains it, so a hole cell's published policy matches its published value. The
    value is `-inf` only where every coarse candidate is infeasible (the optimal
    post-state leaves the post-decision grid); the returned policy there is the first
    candidate and is meaningless, flagged by the `-inf` value.
    """
    c_mesh, d_mesh = jnp.meshgrid(consumption_grid, deposit_grid, indexing="ij")
    candidates = jnp.stack([c_mesh.reshape(-1), d_mesh.reshape(-1)], axis=1)

    def at_target(target: FloatND) -> tuple[FloatND, Float1D]:
        def per_candidate(policy: FloatND) -> FloatND:
            value, feasible = objective(target, policy)
            return jnp.where(feasible & jnp.isfinite(value), value, -jnp.inf)

        masked = jax.vmap(per_candidate)(candidates)
        winner = jnp.argmax(masked)
        return masked[winner], candidates[winner]

    return jax.vmap(at_target)(targets)
