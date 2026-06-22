"""Build a triangulated `SegmentMesh` from a constraint segment's candidate cloud.

A region inverse (`two_asset_inverse`) returns a `RegionCloud` whose fields are shaped
like the segment's own regular source grid — the post-decision $(a, b)$ grid for
`ucon`/`dcon`, or the $(c, b)$ grid at $a = 0$ for `acon`/`con`. The upper envelope
needs that cloud as a triangulated mesh in the current-state $(m, n)$ plane: the nodes
are the cloud's endogenous states and policies, the triangles are the diagonal split of
the source cells, and a node is invalid when its endogenous state is non-finite or its
consumption is non-positive (an off-grid or NaN-inverse node).

The builder is identical for every segment — only the cloud's 2-D source shape is read,
not its coordinates — so the heterogeneous parameterization of the regions is preserved
while the geometric output is uniform.
"""

import jax.numpy as jnp

from _lcm.egm.mesh_envelope import SegmentMesh
from _lcm.egm.mesh_geometry import triangulate_regular_grid
from _lcm.egm.two_asset_inverse import RegionCloud


def build_segment_mesh(*, cloud: RegionCloud, region_label: int) -> SegmentMesh:
    """Triangulate a region's candidate cloud into a `SegmentMesh`.

    Args:
        cloud: The region inverse's endogenous cloud, fields shaped like the segment's
            2-D source grid `(n_rows, n_cols)`.
        region_label: Which KKT segment produced the cloud.

    Returns:
        The segment's triangulated mesh: endogenous `(m, n)` node states, `(c, d)` node
        policies, the source grid's triangle connectivity, and a per-node validity mask
        (finite endogenous state and positive consumption).

    """
    n_rows, n_cols = cloud.m_endog.shape
    m_endog = cloud.m_endog.reshape(-1)
    n_endog = cloud.n_endog.reshape(-1)
    consumption = cloud.consumption.reshape(-1)
    valid_node = jnp.isfinite(m_endog) & jnp.isfinite(n_endog) & (consumption > 0.0)
    return SegmentMesh(
        region_label=region_label,
        node_state=jnp.stack([m_endog, n_endog], axis=1),
        node_policy=jnp.stack([consumption, cloud.deposit.reshape(-1)], axis=1),
        simplices=triangulate_regular_grid(n_rows=n_rows, n_cols=n_cols),
        valid_node=valid_node,
    )
