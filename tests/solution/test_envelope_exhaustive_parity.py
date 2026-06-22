"""The JAX first envelope matches an independent exhaustive host-side reference.

The v3 oracle's mandatory gate for the upper-envelope kernel is an exhaustive
all-simplex reference: on a tiny mesh, enumerate every triangle for every target, apply
the exact admissibility and feasibility rules, recompute every objective, and take the
maximum on the host. This module implements that reference in NumPy — independent of the
production `first_envelope` — and asserts the two agree, including the audit's F3 case
where an admissible-but-extrapolated triangle holds the maximizer.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.mesh_envelope import SegmentMesh, first_envelope
from _lcm.egm.two_asset_inverse import RegionCloud
from _lcm.egm.two_asset_objective import build_two_asset_objective
from _lcm.egm.two_asset_segment_mesh import build_segment_mesh


def _barycentric(triangle, query):
    p0, p1, p2 = triangle
    e1, e2, rhs = p1 - p0, p2 - p0, query - p0
    det = e1[0] * e2[1] - e1[1] * e2[0]
    w1 = (rhs[0] * e2[1] - rhs[1] * e2[0]) / det
    w2 = (e1[0] * rhs[1] - e1[1] * rhs[0]) / det
    return np.array([1.0 - w1 - w2, w1, w2])


def _exhaustive_first_envelope(*, mesh, targets, objective, threshold):
    """Host reference: max recomputed objective over every admissible triangle.

    Mirrors `first_envelope`'s argmax semantics (the first maximizer wins) so the two
    are bit-comparable on the value and the winning policy.
    """
    node_state = np.asarray(mesh.node_state)
    node_policy = np.asarray(mesh.node_policy)
    simplices = np.asarray(mesh.simplices)
    valid = np.asarray(mesh.valid_node)
    values, policies = [], []
    for query in np.asarray(targets):
        candidate_values, candidate_policies = [], []
        for triple in simplices:
            weights = _barycentric(node_state[triple], query)
            policy = weights @ node_policy[triple]
            value, feasible = objective(jnp.asarray(query), jnp.asarray(policy))
            admissible = bool(np.all(weights > -threshold)) and bool(
                np.all(valid[triple])
            )
            keep = admissible and bool(feasible)
            candidate_values.append(float(value) if keep else -np.inf)
            candidate_policies.append(np.asarray(policy))
        best = int(np.argmax(candidate_values))
        values.append(candidate_values[best])
        policies.append(candidate_policies[best])
    return np.array(values), np.array(policies)


def _identity_objective(_state, policy):
    return policy[0], jnp.ones((), dtype=bool)


def test_first_envelope_matches_exhaustive_on_the_f3_extrapolation_case():
    """On the admissible-extrapolation mesh the JAX envelope equals the reference."""
    mesh = SegmentMesh(
        region_label=0,
        node_state=jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],  # triangle B: q extrapolated but admissible, value 1.0
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 3.0],  # triangle A: q covered, value 0.0
            ]
        ),
        node_policy=jnp.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]),
        simplices=jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32),
        valid_node=jnp.ones(6, dtype=bool),
    )
    targets = jnp.array([[1.2, 1.2]])
    jax_values, jax_policies = first_envelope(
        mesh=mesh, targets=targets, objective=_identity_objective, threshold=0.25
    )
    ref_values, ref_policies = _exhaustive_first_envelope(
        mesh=mesh, targets=targets, objective=_identity_objective, threshold=0.25
    )
    np.testing.assert_allclose(np.asarray(jax_values), ref_values, atol=1e-9)
    np.testing.assert_allclose(np.asarray(jax_policies), ref_policies, atol=1e-9)


def test_first_envelope_matches_exhaustive_with_the_two_asset_objective():
    """On a region-cloud mesh and the real objective the two implementations agree."""
    a_mesh, b_mesh = jnp.meshgrid(jnp.arange(3.0), jnp.arange(3.0), indexing="ij")
    cloud = RegionCloud(
        m_endog=2.0 + 1.5 * a_mesh,
        n_endog=2.0 + 1.5 * b_mesh,
        consumption=1.0 + 0.3 * a_mesh,
        deposit=0.2 * b_mesh,
        value=jnp.zeros((3, 3)),
        value_grad_m=jnp.zeros((3, 3)),
        value_grad_n=jnp.zeros((3, 3)),
    )
    mesh = build_segment_mesh(cloud=cloud, region_label=0)
    grid = jnp.linspace(0.0, 10.0, 11)
    value_mesh = jnp.meshgrid(grid, grid, indexing="ij")
    objective = build_two_asset_objective(
        post_decision_value=2.0 * value_mesh[0] + 3.0 * value_mesh[1] + 1.0,
        a_grid=grid,
        b_grid=grid,
        discount_factor=0.95,
        crra=2.0,
        match_rate=1.0,
    )
    targets = jnp.array([[3.0, 3.0], [4.0, 4.0], [3.5, 4.5], [2.5, 2.5]])
    jax_values, jax_policies = first_envelope(
        mesh=mesh, targets=targets, objective=objective, threshold=0.25
    )
    ref_values, ref_policies = _exhaustive_first_envelope(
        mesh=mesh, targets=targets, objective=objective, threshold=0.25
    )
    np.testing.assert_allclose(np.asarray(jax_values), ref_values, atol=1e-9)
    np.testing.assert_allclose(np.asarray(jax_policies), ref_policies, atol=1e-9)
