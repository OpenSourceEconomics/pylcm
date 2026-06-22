"""The two-asset Bellman-objective evaluator recomputes value and feasibility.

The evaluator reconstructs the post-decision balances from the budget identities,
reads the post-decision value by bilinear interpolation, and adds CRRA utility. The
tests pin exact values against a known analytic post-decision value (affine, so
bilinear interpolation is exact off-grid; and the audit's `a/(1+b)` read exactly at a
grid node), and pin the feasibility flag that the envelope masks on.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_objective import build_two_asset_objective

_DISCOUNT = 0.95
_CRRA = 2.0
_MATCH = 1.0
_A_GRID = jnp.linspace(0.0, 10.0, 11)
_B_GRID = jnp.linspace(0.0, 10.0, 11)


def test_objective_is_exact_for_an_affine_post_decision_value():
    """For affine `W`, the recomputed objective equals `u(c) + beta*W(a, b)` exactly.

    With `W(a, b) = 2a + 3b + 1` (bilinear-exact), state `(5, 2)`, policy `(c, d) =
    (1, 0.5)`: `a = 3.5`, `b = 2.5 + log(1.5)`, so `W = 16.7163953` and the objective is
    `u(1) + 0.95*W = -1 + 15.8805756 = 14.8805756`.
    """
    a_mesh, b_mesh = jnp.meshgrid(_A_GRID, _B_GRID, indexing="ij")
    post_decision_value = 2.0 * a_mesh + 3.0 * b_mesh + 1.0
    objective = build_two_asset_objective(
        post_decision_value=post_decision_value,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
    )
    value, feasible = objective(jnp.array([5.0, 2.0]), jnp.array([1.0, 0.5]))
    np.testing.assert_allclose(float(value), 14.8805755576, rtol=1e-10)
    assert bool(feasible)


def test_objective_reads_the_audit_post_decision_value_at_a_grid_node():
    """At a grid node the bilinear read is exact, so `W(a,b)=a/(1+b)` is recovered.

    State `(4, 2)`, policy `(1, 0)` give `a = 3`, `b = 2` (both grid nodes), so
    `W = 3/3 = 1` and the objective is `u(1) + 0.95*1 = -0.05`.
    """
    a_mesh, b_mesh = jnp.meshgrid(_A_GRID, _B_GRID, indexing="ij")
    post_decision_value = a_mesh / (1.0 + b_mesh)
    objective = build_two_asset_objective(
        post_decision_value=post_decision_value,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
    )
    value, feasible = objective(jnp.array([4.0, 2.0]), jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(float(value), -0.05, rtol=1e-10)
    assert bool(feasible)


def test_objective_flags_negative_liquid_post_decision_as_infeasible():
    """A policy whose liquid post-decision balance `a = m - c - d` is negative fails."""
    a_mesh, b_mesh = jnp.meshgrid(_A_GRID, _B_GRID, indexing="ij")
    post_decision_value = 2.0 * a_mesh + 3.0 * b_mesh + 1.0
    objective = build_two_asset_objective(
        post_decision_value=post_decision_value,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
    )
    # m - c - d = 1 - 1 - 0.5 = -0.5 < 0.
    value, feasible = objective(jnp.array([1.0, 2.0]), jnp.array([1.0, 0.5]))
    assert not bool(feasible)
    assert np.isfinite(float(value))


def test_objective_flags_negative_consumption_as_infeasible():
    """A non-positive interpolated consumption is infeasible with a finite value."""
    a_mesh, b_mesh = jnp.meshgrid(_A_GRID, _B_GRID, indexing="ij")
    post_decision_value = 2.0 * a_mesh + 3.0 * b_mesh + 1.0
    objective = build_two_asset_objective(
        post_decision_value=post_decision_value,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
    )
    value, feasible = objective(jnp.array([5.0, 2.0]), jnp.array([-0.5, 0.0]))
    assert not bool(feasible)
    assert np.isfinite(float(value))
