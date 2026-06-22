"""A region cloud triangulates into a `SegmentMesh` with the right topology.

The mesh built from an `n_rows` by `n_cols` cloud has one node per cloud entry, two
triangles per source cell, the endogenous states and policies of the cloud at its
nodes, and a validity mask that drops non-finite or non-positive-consumption nodes.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_inverse import RegionCloud
from _lcm.egm.two_asset_segment_mesh import build_segment_mesh


def _synthetic_cloud(*, consumption=None):
    """A 3x3 region cloud with simple, distinct node states and policies."""
    a_mesh, b_mesh = jnp.meshgrid(jnp.arange(3.0), jnp.arange(3.0), indexing="ij")
    if consumption is None:
        consumption = 1.0 + a_mesh
    return RegionCloud(
        m_endog=a_mesh + 10.0,
        n_endog=b_mesh + 20.0,
        consumption=consumption,
        deposit=0.5 * b_mesh,
        value=jnp.zeros((3, 3)),
        value_grad_m=jnp.zeros((3, 3)),
        value_grad_n=jnp.zeros((3, 3)),
    )


def test_mesh_has_one_node_per_cloud_entry_and_two_triangles_per_cell():
    """A 3x3 cloud yields 9 nodes and `2*2*2 = 8` triangles."""
    mesh = build_segment_mesh(cloud=_synthetic_cloud(), region_label=1)
    assert mesh.node_state.shape == (9, 2)
    assert mesh.node_policy.shape == (9, 2)
    assert mesh.simplices.shape == (8, 3)
    assert mesh.region_label == 1


def test_mesh_nodes_match_the_cloud_state_and_policy():
    """The mesh nodes are the cloud's endogenous states and policies, row-major."""
    cloud = _synthetic_cloud()
    mesh = build_segment_mesh(cloud=cloud, region_label=0)
    np.testing.assert_allclose(
        np.asarray(mesh.node_state[:, 0]), np.asarray(cloud.m_endog).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(mesh.node_state[:, 1]), np.asarray(cloud.n_endog).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(mesh.node_policy[:, 0]), np.asarray(cloud.consumption).reshape(-1)
    )


def test_mesh_marks_non_positive_consumption_and_non_finite_state_invalid():
    """A non-positive consumption or non-finite endogenous state masks the node."""
    consumption = 1.0 + jnp.meshgrid(jnp.arange(3.0), jnp.arange(3.0), indexing="ij")[0]
    consumption = consumption.at[0, 0].set(-1.0)  # invalid: non-positive consumption
    cloud = _synthetic_cloud(consumption=consumption)
    cloud = cloud._replace(m_endog=cloud.m_endog.at[2, 2].set(jnp.nan))  # invalid: NaN
    mesh = build_segment_mesh(cloud=cloud, region_label=0)
    valid = np.asarray(mesh.valid_node)
    assert not valid[0]  # node (0,0)
    assert not valid[8]  # node (2,2)
    assert valid[4]  # an interior node stays valid
