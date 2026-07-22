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

import itertools

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, Float2D, ScalarFloat

# Reference tuning constants, verbatim from RFCSimple.py / RFC.py.
J_BAR_DEFAULT = 1.0 + 1e-10
RADIUS_DEFAULT = 0.5
K_DEFAULT = 5

# Publisher defaults: enough nearest survivors to bracket a target in a non-degenerate
# triangle, a small negative barycentric tolerance, and a squared-area floor that
# rejects near-collinear (degenerate) triangles.
K_PUBLISH_DEFAULT = 12
EXTRAPOLATION_THRESHOLD_DEFAULT = 1e-9
DEGENERATE_AREA_FLOOR = 1e-12
# Minimum normalized mean-ratio shape quality `Q in (0, 1]` (1 = equilateral) a
# publish simplex must clear to count as well-conditioned; below it a sliver's
# affine interpolant is unstable. Generous enough not to reject ordinary
# acute/obtuse triangles (only ~20:1-and-worse slivers fall below it).
SHAPE_QUALITY_FLOOR = 0.1


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
    # Delete a candidate only when it lies below the neighbour's tangent plane
    # past a scale-aware noise floor. `tangent` is an einsum reduction whose
    # rounding sign follows the backend's summation order, so a candidate exactly
    # on the plane would otherwise be dropped on one backend and kept on another.
    # Keeping on the tie is the conservative direction (an on-plane point is on
    # the envelope). Mirrors the savings-tie floor in `fues.py`.
    noise_floor = (
        16.0
        * jnp.finfo(values.dtype).eps
        * jnp.maximum(jnp.abs(value_gap), jnp.abs(tangent))
    )
    below_tangent = value_gap < tangent - noise_floor

    selector = policies[:, 0]
    policy_gap = jnp.abs(selector[None, :] - selector[:, None])
    safe_distance = jnp.where(self_mask, 1.0, distance_no_self)
    policy_jump = (policy_gap / safe_distance) > j_bar

    within_radius = distance_no_self < radius

    deletes = in_knn & below_tangent & policy_jump & within_radius
    deleted = jnp.any(deletes, axis=0)
    return ~deleted


def rfc_publish_2d(
    *,
    survivor_states: Float2D,
    survivor_values: Float1D,
    survivor_policies: Float2D,
    target_states: Float2D,
    valid: BoolND | None = None,
    k: int = K_PUBLISH_DEFAULT,
    extrapolation_threshold: float = EXTRAPOLATION_THRESHOLD_DEFAULT,
) -> tuple[Float1D, Float2D]:
    """Publish value and policy at target states by local-simplex barycentric weights.

    The on-device stand-in for the host Delaunay publisher (D1): for each target, take
    its `k` nearest survivors, enumerate every triangle among them, and select the
    *most-local well-conditioned* containing triangle — the smallest-area simplex
    (weights at or above `-extrapolation_threshold`) whose normalized shape quality
    clears `SHAPE_QUALITY_FLOOR`. Locality keeps the affine fit tight (it tracks the
    curved value surface instead of spanning a wide arc, the accuracy the KKT-masked
    clouds need); the shape gate rejects ill-conditioned slivers that pass the area
    floor. If every containing triangle is a sliver, the smallest containing one is
    used (coverage over conditioning); a target with no containing triangle (outside
    the survivor support) falls back to its nearest survivor. Value and policy are the
    barycentric combination of the chosen triangle's vertices, which reproduces
    survivor values exactly and is affine-exact in the hull.

    The `valid` mask lets a caller pass the *full* candidate cloud plus the cut's
    keep-mask rather than a pre-filtered survivor array — the jit-friendly form, since
    compacting survivors would change the array shape. Deleted candidates are pushed to
    infinite distance (so they never enter a target's neighbourhood) and any triangle
    that would use one as a vertex is rejected.

    Args:
        survivor_states: Candidate states, shape `(s, d)` with `d == 2`.
        survivor_values: Candidate values, shape `(s,)`.
        survivor_policies: Candidate policy vectors, shape `(s, dp)`.
        target_states: Query states, shape `(t, 2)`.
        valid: Optional keep-mask, shape `(s,)`; `True` for surviving candidates. When
            omitted every candidate is treated as a survivor.
        k: Number of nearest survivors that form the local simplex search set.
        extrapolation_threshold: Non-negative barycentric tolerance; a triangle counts
            as containing the target when every weight is at least its negation.

    Returns:
        Tuple of `(values, policies)`: published value `(t,)` and policy `(t, dp)`.
    """
    n_survivors = survivor_states.shape[0]
    k_eff = min(k, n_survivors)
    triangles = jnp.asarray(
        list(itertools.combinations(range(k_eff), 3)), dtype=jnp.int32
    )
    keep = jnp.ones(n_survivors, dtype=bool) if valid is None else valid

    def publish_one(query: Float1D) -> tuple[ScalarFloat, Float1D]:
        raw_distance = jnp.linalg.norm(survivor_states - query, axis=1)
        distance = jnp.where(keep, raw_distance, jnp.inf)
        _, nearest_idx = jax.lax.top_k(-distance, k_eff)
        verts = survivor_states[nearest_idx]
        local_values = survivor_values[nearest_idx]
        local_policies = survivor_policies[nearest_idx]
        local_valid = keep[nearest_idx]

        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = verts[a], verts[b], verts[c]
        edge1 = p1 - p0
        edge2 = p2 - p0
        to_query = query - p0
        d00 = jnp.sum(edge1 * edge1, axis=1)
        d01 = jnp.sum(edge1 * edge2, axis=1)
        d11 = jnp.sum(edge2 * edge2, axis=1)
        d20 = jnp.sum(to_query * edge1, axis=1)
        d21 = jnp.sum(to_query * edge2, axis=1)
        # `denom` is the squared area (times 4): zero for collinear vertices.
        denom = d00 * d11 - d01 * d01
        safe_denom = jnp.where(denom > 0.0, denom, 1.0)
        w1 = (d11 * d20 - d01 * d21) / safe_denom
        w2 = (d00 * d21 - d01 * d20) / safe_denom
        w0 = 1.0 - w1 - w2

        nondegenerate = denom > DEGENERATE_AREA_FLOOR
        triangle_valid = local_valid[a] & local_valid[b] & local_valid[c]
        contains = (
            (w0 >= -extrapolation_threshold)
            & (w1 >= -extrapolation_threshold)
            & (w2 >= -extrapolation_threshold)
            & nondegenerate
            & triangle_valid
        )
        # Shape quality `Q = 4 sqrt(3) A / (l0^2 + l1^2 + l2^2)` (the standard
        # normalized mean-ratio; `Q = 1` equilateral, `Q -> 0` sliver), from the
        # squared edge lengths `d00`, `d11`, and `|edge2 - edge1|^2 = d00 + d11 -
        # 2 d01`, with `4A = 2 sqrt(denom)`. The area floor alone passes
        # ill-conditioned slivers; gating on `Q` rejects them so the affine
        # interpolant stays stable.
        edge_sq_sum = 2.0 * (d00 + d11 - d01)
        safe_edge_sq_sum = jnp.where(edge_sq_sum > 0.0, edge_sq_sum, 1.0)
        quality = (
            2.0 * jnp.sqrt(3.0) * jnp.sqrt(jnp.maximum(denom, 0.0)) / safe_edge_sq_sum
        )
        well_conditioned = quality > SHAPE_QUALITY_FLOOR

        # Prefer the *most local* (smallest-area) *well-conditioned* containing
        # simplex: locality keeps the affine fit tight, the quality gate keeps it
        # stable. If every containing simplex is a sliver, fall back to the smallest
        # containing one (coverage over conditioning) before the nearest survivor.
        score_good = jnp.where(contains & well_conditioned, -denom, -jnp.inf)
        best_good = jnp.argmax(score_good)
        found_good = score_good[best_good] > -jnp.inf
        score_any = jnp.where(contains, -denom, -jnp.inf)
        best_any = jnp.argmax(score_any)
        found_any = score_any[best_any] > -jnp.inf
        best = jnp.where(found_good, best_good, best_any)
        found = found_good | found_any

        weights = jnp.array([w0[best], w1[best], w2[best]])
        vertex_values = jnp.array(
            [local_values[a[best]], local_values[b[best]], local_values[c[best]]]
        )
        vertex_policies = jnp.stack(
            [local_policies[a[best]], local_policies[b[best]], local_policies[c[best]]]
        )
        value_interp = weights @ vertex_values
        policy_interp = weights @ vertex_policies

        value = jnp.where(found, value_interp, local_values[0])
        policy = jnp.where(found, policy_interp, local_policies[0])
        return value, policy

    return jax.vmap(publish_one)(target_states)
