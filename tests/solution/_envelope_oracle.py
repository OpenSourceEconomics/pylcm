"""Branch-aware exact upper-envelope oracle for the EGM envelope backends.

An independent host-side reference for FUES/RFC/MSS/LTM. Given discrete-choice
candidate branches, it computes the *exact* pointwise upper envelope of their
piecewise-linear interpolants and the winning branch's policy. It makes no
concavity, monotonicity, or scan-window assumption.

## Topology contract

Each branch is an **ordered polyline**: the candidates carrying one `segment_id`
are taken **in input order** as the polyline's vertices, and consecutive
vertices are its edges. Input order — not a re-sort by `x` — is therefore the
branch's edge connectivity, so a non-monotone (folded) branch such as
`(0, 0) -> (2, 10) -> (1, 0)` is evaluated as the true polyline (value `7.5` at
`x = 1.5`), not the spurious `5` a sort-by-`x` would give. At a query the branch
value is the maximum over every edge whose `x`-range covers the query (a fold
contributes several), and the envelope is the maximum over the branches defined
there.

`NaN` in `endog_grid` marks a padding slot and is dropped. A live candidate with
a non-finite value, policy, or segment label is a malformed input and raises
`TopologyError`. A branch that revisits an abscissa with a *different* value or
policy is multivalued there and also raises `TopologyError`; an exact duplicate
vertex is collapsed.

This is a test tool only — host-side NumPy, never a registered JIT/GPU backend.
"""

from itertools import pairwise

import numpy as np

# Right-continuous tie probe: at an exact crossing the winning policy is the
# branch that also wins just to the right, matching the kernel's
# `searchsorted(side="right")` read. The probe offset is tiny relative to the
# data scale, so it stays inside the bracket between adjacent envelope events.
_RIGHT_EPS = 1e-7


class TopologyError(ValueError):
    """Raised when candidate topology is ambiguous or malformed."""


def _branches(
    *,
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    segment_id: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Group candidates into ordered-polyline branches, in input order.

    Return a list of `(label, x, value, policy)` per distinct `segment_id`, each
    in input (edge-connectivity) order with padding dropped. Fail loudly on a
    non-finite live entry or a branch that is multivalued at an abscissa.
    """
    x = np.asarray(endog_grid, dtype=float)
    v = np.asarray(value, dtype=float)
    p = np.asarray(policy, dtype=float)
    s = np.asarray(segment_id, dtype=float)

    live = ~np.isnan(x)
    if np.any(live & ~np.isfinite(v)):
        msg = "non-finite value at a live candidate (NaN endog_grid marks padding)"
        raise TopologyError(msg)
    if np.any(live & ~np.isfinite(p)):
        msg = "non-finite policy at a live candidate"
        raise TopologyError(msg)
    if np.any(live & ~np.isfinite(s)):
        msg = "non-finite segment label at a live candidate"
        raise TopologyError(msg)
    x, v, p, s = x[live], v[live], p[live], s[live]

    order: list[float] = []
    groups: dict[float, list[tuple[float, float, float]]] = {}
    for xi, vi, pi, si in zip(x, v, p, s, strict=True):
        if si not in groups:
            groups[si] = []
            order.append(si)
        groups[si].append((float(xi), float(vi), float(pi)))

    branches: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for label in order:
        pts = groups[label]
        seen: dict[float, tuple[float, float]] = {}
        keep: list[tuple[float, float, float]] = []
        for xi, vi, pi in pts:
            if xi in seen:
                if seen[xi] != (vi, pi):
                    msg = f"branch {label} is multivalued at x={xi}"
                    raise TopologyError(msg)
                continue
            seen[xi] = (vi, pi)
            keep.append((xi, vi, pi))
        branches.append(
            (
                float(label),
                np.array([q[0] for q in keep]),
                np.array([q[1] for q in keep]),
                np.array([q[2] for q in keep]),
            )
        )
    return branches


def _branch_at(
    branch: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    x_query: float,
    tol: float,
) -> tuple[float, float] | None:
    """Polyline value and policy of one branch at `x_query`, max over covering edges.

    Returns `None` if the query lies outside every edge's `x`-range. A folded
    branch contributes several covering edges; the branch value is their maximum.
    """
    _label, x, value, policy = branch
    if x.size == 1:
        return (
            (float(value[0]), float(policy[0])) if abs(x_query - x[0]) <= tol else None
        )

    best_value, best_policy = -np.inf, np.nan
    for i in range(x.size - 1):
        x0, x1 = x[i], x[i + 1]
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        if not (lo - tol <= x_query <= hi + tol):
            continue
        if x1 == x0:
            edge_value, edge_policy = (
                (value[i], policy[i])
                if value[i] >= value[i + 1]
                else (value[i + 1], policy[i + 1])
            )
        else:
            t = (x_query - x0) / (x1 - x0)
            edge_value = value[i] + t * (value[i + 1] - value[i])
            edge_policy = policy[i] + t * (policy[i + 1] - policy[i])
        if edge_value > best_value:
            best_value, best_policy = float(edge_value), float(edge_policy)
    return (best_value, best_policy) if best_value > -np.inf else None


def _resolve_tie(
    branches: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    tied: list[tuple[float, tuple[float, float]]],
    x_query: float,
    tol: float,
) -> tuple[float, float, float]:
    """Break an exact-value tie right-continuously: the branch winning just right.

    Probe each tied branch a hair to the right of the crossing; the one with the
    larger value there carries the published policy. If the crossing sits at the
    common right edge (no branch is defined further right), fall back to the
    lowest segment label for determinism.
    """
    label_to_branch = {b[0]: b for b in branches}
    eps = _RIGHT_EPS * max(1.0, abs(x_query))
    best_label, best_policy, best_right = None, np.nan, -np.inf
    for label, (_value, policy) in tied:
        probed = _branch_at(label_to_branch[label], x_query + eps, tol)
        right_value = probed[0] if probed is not None else -np.inf
        if right_value > best_right:
            best_right, best_label, best_policy = right_value, label, policy
    if best_label is None or best_right == -np.inf:
        # All tied branches end at this query: pick the lowest label deterministically.
        label, (_value, policy) = min(tied, key=lambda item: item[0])
        return label, policy, _value
    return best_label, best_policy, dict(tied)[best_label][0]


def exact_envelope(
    *,
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    segment_id: np.ndarray,
    x_query: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact branch-aware upper-envelope value and policy at the query points.

    Return `(env_value, env_policy, winner)` aligned with `x_query`. At each query
    the envelope value is the exact maximum, over the branches defined there, of
    the branch's piecewise-linear value; the policy is the winning branch's
    interpolated policy and `winner` its **segment label** (not a list index).
    `tol` only classifies ties — it never lowers the reported maximum. An exact
    value tie breaks right-continuously (the branch winning just to the right),
    matching the kernel's `side="right"` read. A query outside every branch's
    range yields `NaN` value/policy and winner `-1`.
    """
    branches = _branches(
        endog_grid=endog_grid, value=value, policy=policy, segment_id=segment_id
    )
    xq = np.asarray(x_query, dtype=float)
    env_value = np.full(xq.shape, np.nan)
    env_policy = np.full(xq.shape, np.nan)
    winner = np.full(xq.shape, -1.0)

    for k, x_point in enumerate(xq):
        evaluated = [
            (branch[0], result)
            for branch in branches
            if (result := _branch_at(branch, float(x_point), tol)) is not None
        ]
        if not evaluated:
            continue
        # Exact maximum first; tolerance only groups the tie set around it.
        max_value = max(result[0] for _label, result in evaluated)
        tied = [
            (label, result)
            for label, result in evaluated
            if result[0] >= max_value - tol
        ]
        if len(tied) == 1:
            win_label, (_win_value, win_policy) = tied[0]
        else:
            win_label, win_policy, _win_value = _resolve_tie(
                branches, tied, float(x_point), tol
            )
        env_value[k] = max_value
        env_policy[k] = win_policy
        winner[k] = win_label
    return env_value, env_policy, winner


def envelope_event_points(
    *,
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    segment_id: np.ndarray,
) -> np.ndarray:
    """Return an event-complete set of query abscissae for the envelope.

    Certifying a whole envelope by sampling can miss an arbitrarily narrow winning
    interval between samples. The events that can change the envelope winner are
    the branch vertices and the pairwise edge intersections; testing each event
    plus an interior point of every resulting interval is sufficient. Return the
    sorted unique union of all vertices, all pairwise edge-crossing abscissae, and
    the midpoints between consecutive events.
    """
    branches = _branches(
        endog_grid=endog_grid, value=value, policy=policy, segment_id=segment_id
    )
    events: set[float] = set()
    edges: list[tuple[float, float, float, float]] = []
    for _label, x, v, _p in branches:
        events.update(float(xi) for xi in x)
        for i in range(x.size - 1):
            edge = (float(x[i]), float(v[i]), float(x[i + 1]), float(v[i + 1]))
            edges.append(edge)

    for ax0, av0, ax1, av1 in edges:
        if ax1 == ax0:
            continue
        a_slope = (av1 - av0) / (ax1 - ax0)
        for bx0, bv0, bx1, bv1 in edges:
            if bx1 == bx0:
                continue
            b_slope = (bv1 - bv0) / (bx1 - bx0)
            if a_slope == b_slope:
                continue
            cross = (bv0 - av0 + a_slope * ax0 - b_slope * bx0) / (a_slope - b_slope)
            in_a = min(ax0, ax1) <= cross <= max(ax0, ax1)
            in_b = min(bx0, bx1) <= cross <= max(bx0, bx1)
            if in_a and in_b:
                events.add(float(cross))

    ordered = sorted(events)
    midpoints = [(a + b) / 2.0 for a, b in pairwise(ordered)]
    return np.array(sorted(set(ordered) | set(midpoints)))
