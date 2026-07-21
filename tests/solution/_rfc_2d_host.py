"""Host (NumPy/SciPy) reference for the multidimensional Roof-top Cut (RFC).

This is the off-device ground truth for the DS-2024 multidimensional RFC backend
(Dobrescu & Shanker 2024, their Box 1). It reproduces the reference implementation
`akshayshanker/InverseDCDP/RFC/RFCSimple.py` that produced the paper's RFC numbers:
a k-nearest-neighbour, radius-masked tangent-dominance delete, followed by a
Delaunay barycentric publish. The JAX kernel and on-device publisher are validated
against this module; it is never on a solve path.

The cut, per candidate $i$ over its $k-1$ strict KD-tree neighbours $j$, deletes $j$
iff all three hold:

- $j$ lies below $i$'s tangent plane: $v_j - v_i < \\nabla v_i \\cdot (x_j - x_i)$;
- a policy jump separates them, using policy column 0 as the selector:
  $|\\,\\lVert\\sigma_j - \\sigma_i\\rVert / \\lVert x_j - x_i\\rVert\\,| > \\bar J$;
- they are neighbours within the radius: $\\lVert x_j - x_i\\rVert < \\rho$.

The deleted set is the union of such $j$ over all $i$. The reference tuning constants
are $\\bar J = 1 + 10^{-10}$, $\\rho = 0.5$, $k = 5$.
"""

import numpy as np
from scipy.spatial import Delaunay, KDTree

# Reference tuning constants, verbatim from RFCSimple.py / RFC.py.
J_BAR_DEFAULT = 1.0 + 1e-10
RADIUS_DEFAULT = 0.5
K_DEFAULT = 5


def rfc_delete_mask(
    *,
    states: np.ndarray,
    supgradients: np.ndarray,
    values: np.ndarray,
    policies: np.ndarray,
    j_bar: float = J_BAR_DEFAULT,
    radius: float = RADIUS_DEFAULT,
    k: int = K_DEFAULT,
) -> np.ndarray:
    """Return a boolean keep-mask for the candidate cloud (True = survives the cut).

    Args:
        states: Candidate post-decision states, shape `(n, d)`.
        supgradients: Value supgradient at each candidate, shape `(n, d)`.
        values: Candidate value, shape `(n,)`.
        policies: Candidate policy vectors, shape `(n, dp)`; the jump selector uses
            column 0 only, matching the reference's `sigma[:, [0]]`.
        j_bar: Policy-jump threshold $\\bar J$.
        radius: Neighbour distance threshold $\\rho$.
        k: Number of KD-tree neighbours (including self) to query.

    Returns:
        Boolean array of shape `(n,)`; `True` for surviving (kept) candidates.
    """
    states = np.asarray(states, dtype=float)
    supgradients = np.asarray(supgradients, dtype=float)
    values = np.asarray(values, dtype=float)
    selector = np.asarray(policies, dtype=float)[:, 0]
    n = states.shape[0]

    keep = np.ones(n, dtype=bool)
    if n <= 1:
        return keep

    k_eff = min(k, n)
    tree = KDTree(states)
    dd, idx = tree.query(states, k=k_eff)
    dd = np.atleast_2d(dd)
    idx = np.atleast_2d(idx)
    # Drop self (column 0); clip any out-of-range fill index (KDTree returns n on
    # under-full queries) to the last valid point, where the inf distance disables it.
    neigh = idx[:, 1:].copy()
    neigh_dist = dd[:, 1:]
    neigh[neigh >= n] = n - 1

    for i in range(n):
        for col in range(neigh.shape[1]):
            j = neigh[i, col]
            dist = neigh_dist[i, col]
            if not np.isfinite(dist) or dist >= radius:
                continue
            delta_x = states[j] - states[i]
            tangent = float(delta_x @ supgradients[i])
            value_gap = values[j] - values[i]
            # Match the kernel's scale-aware noise floor so both keep a candidate
            # sitting on the tangent plane within rounding noise.
            noise_floor = (
                16.0 * np.finfo(values.dtype).eps * max(abs(value_gap), abs(tangent))
            )
            below_tangent = value_gap < tangent - noise_floor
            policy_jump = abs(abs(selector[j] - selector[i]) / dist) > j_bar
            if below_tangent and policy_jump:
                keep[j] = False
    return keep


def rfc_publish(
    *,
    survivor_states: np.ndarray,
    survivor_values: np.ndarray,
    survivor_policies: np.ndarray,
    target_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Publish value and policy at target states by Delaunay barycentric interpolation.

    Mirrors the reference's `interpolateEGM`: a `scipy.spatial.Delaunay` triangulation
    of the survivor cloud, linear (barycentric) interpolation inside the convex hull,
    and a nearest-survivor fallback for targets outside it. Barycentric weights
    reproduce survivor values exactly at survivor locations.

    Args:
        survivor_states: Surviving candidate states, shape `(s, d)`.
        survivor_values: Surviving candidate values, shape `(s,)`.
        survivor_policies: Surviving candidate policy vectors, shape `(s, dp)`.
        target_states: Query states, shape `(t, d)`.

    Returns:
        Tuple of `(values, policies)`: interpolated value `(t,)` and policy `(t, dp)`.
    """
    survivor_states = np.asarray(survivor_states, dtype=float)
    survivor_values = np.asarray(survivor_values, dtype=float)
    survivor_policies = np.asarray(survivor_policies, dtype=float)
    target_states = np.asarray(target_states, dtype=float)

    tri = Delaunay(survivor_states)
    simplex = tri.find_simplex(target_states)

    n_targets = target_states.shape[0]
    n_policies = survivor_policies.shape[1]
    values = np.empty(n_targets, dtype=float)
    policies = np.empty((n_targets, n_policies), dtype=float)

    tree = KDTree(survivor_states)

    for t in range(n_targets):
        s = simplex[t]
        if s < 0:
            # Outside the convex hull: nearest-survivor fallback.
            _, nearest = tree.query(target_states[t], k=1)
            values[t] = survivor_values[nearest]
            policies[t] = survivor_policies[nearest]
            continue
        vertices = tri.simplices[s]
        transform = tri.transform[s]
        d = target_states.shape[1]
        bary = transform[:d] @ (target_states[t] - transform[d])
        weights = np.append(bary, 1.0 - bary.sum())
        values[t] = float(weights @ survivor_values[vertices])
        policies[t] = weights @ survivor_policies[vertices]
    return values, policies


def rfc_envelope_host(
    *,
    states: np.ndarray,
    supgradients: np.ndarray,
    values: np.ndarray,
    policies: np.ndarray,
    target_states: np.ndarray,
    j_bar: float = J_BAR_DEFAULT,
    radius: float = RADIUS_DEFAULT,
    k: int = K_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full host RFC: cut the cloud, then publish at the target states.

    Returns:
        Tuple of `(keep_mask, published_values, published_policies)`.
    """
    keep = rfc_delete_mask(
        states=states,
        supgradients=supgradients,
        values=values,
        policies=policies,
        j_bar=j_bar,
        radius=radius,
        k=k,
    )
    published_values, published_policies = rfc_publish(
        survivor_states=np.asarray(states, dtype=float)[keep],
        survivor_values=np.asarray(values, dtype=float)[keep],
        survivor_policies=np.asarray(policies, dtype=float)[keep],
        target_states=target_states,
    )
    return keep, published_values, published_policies
