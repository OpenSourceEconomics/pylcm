"""On-device multidimensional Roof-top Cut: the k-NN radius-masked delete kernel.

The multidimensional RFC of Dobrescu & Shanker (2024, their Box 1) selects the upper
envelope of a value *correspondence* over a 2-D (or higher) post-decision cloud by a
dense dominance test: at each candidate $i$ it forms the tangent plane from the value
supgradient $\\nabla v_i$ and deletes every $k$-nearest neighbour $j$ within a physical
radius that

- lies below $i$'s tangent plane — $v_j - v_i < \\nabla v_i \\cdot (x_j - x_i)$ — and
- sits across a policy jump (the jump selector is policy column 0) —
  $\\lVert\\sigma_j - \\sigma_i\\rVert / \\lVert x_j - x_i\\rVert > \\bar J$.

Unlike the 1-D backend's sorted-grid neighbourhood, the multidimensional neighbours are
the $k-1$ strict nearest by Euclidean distance. KD-trees are not `jit`-able, so the
kernel computes the full pairwise distance matrix and takes the $k$ smallest per row
with `jax.lax.top_k`; this yields the same neighbour set as the host KD-tree reference
(`tests/solution/_rfc_2d_host.py`), against which the kernel is validated. The per-pair
test has no sequential carry, so the whole cut is a static-shape, `jit`-able dense
computation. Delete-only (no crossing insertion) keeps the output a subset of the input,
faithful to Box 1.

The reference tuning constants are $\\bar J = 1 + 10^{-10}$, $\\rho = 0.5$, $k = 5$.
"""

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, Float2D

# Reference tuning constants, verbatim from RFCSimple.py / RFC.py.
J_BAR_DEFAULT = 1.0 + 1e-10
RADIUS_DEFAULT = 0.5
K_DEFAULT = 5


def rfc_delete_mask_2d(
    *,
    states: Float2D,
    supgradients: Float2D,
    values: Float1D,
    policies: Float2D,
    j_bar: float = J_BAR_DEFAULT,
    radius: float = RADIUS_DEFAULT,
    k: int = K_DEFAULT,
) -> BoolND:
    """Return a boolean keep-mask for the candidate cloud (True = survives the cut).

    The kernel reproduces the host RFC reference: for each candidate $i$ it deletes any
    of its $k-1$ strict nearest neighbours $j$ that lies below $i$'s tangent plane,
    across a policy jump, and within `radius`. A point is kept unless some anchor
    deletes it.

    Args:
        states: Candidate post-decision states, shape `(n, d)`.
        supgradients: Value supgradient at each candidate, shape `(n, d)`.
        values: Candidate value, shape `(n,)`.
        policies: Candidate policy vectors, shape `(n, dp)`; the jump selector uses
            column 0 only, matching the reference's `sigma[:, [0]]`.
        j_bar: Policy-jump threshold $\\bar J$.
        radius: Neighbour distance threshold $\\rho$.
        k: Number of KD-tree neighbours (including self) the reference queries; the cut
            inspects the `k - 1` strict nearest.

    Returns:
        Boolean array of shape `(n,)`; `True` for surviving (kept) candidates.
    """
    n = states.shape[0]
    n_neighbors = min(k - 1, n - 1)

    # diff[i, j] = x_j - x_i, so tangent[i, j] reads anchor i's gradient.
    diff = states[None, :, :] - states[:, None, :]
    distance = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    self_mask = jnp.eye(n, dtype=bool)
    distance_no_self = jnp.where(self_mask, jnp.inf, distance)

    # k-NN restriction: a neighbour is in-set iff its distance is within the
    # n_neighbors-th smallest of the row (self excluded by the +inf diagonal).
    nearest_neg, _ = jax.lax.top_k(-distance_no_self, n_neighbors)
    kth_distance = -nearest_neg[:, -1]
    in_knn = distance_no_self <= kth_distance[:, None]

    tangent = jnp.einsum("id,ijd->ij", supgradients, diff)
    value_gap = values[None, :] - values[:, None]
    below_tangent = value_gap < tangent

    selector = policies[:, 0]
    policy_gap = jnp.abs(selector[None, :] - selector[:, None])
    safe_distance = jnp.where(self_mask, 1.0, distance_no_self)
    policy_jump = (policy_gap / safe_distance) > j_bar

    within_radius = distance_no_self < radius

    deletes = in_knn & below_tangent & policy_jump & within_radius
    deleted = jnp.any(deletes, axis=0)
    return ~deleted
