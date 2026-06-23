"""Branch-aware exact upper-envelope oracle for the EGM envelope backends.

An independent host-side reference for FUES/RFC/MSS/LTM. Given discrete-choice
candidate branches (grouped by `segment_id`), it computes the *exact* pointwise
upper envelope of their piecewise-linear interpolants and the winning branch's
policy. Unlike the production backends it makes no concavity, monotonicity, or
scan-window assumption: within each branch the candidates define an exact
piecewise-linear function, and the envelope is the pointwise maximum over the
branches defined at the query.

This is the *branch-aware exact oracle* — a simple per-branch evaluation rather
than Hershberger's O(n log n) segment-envelope algorithm, which is easier to
audit on the small candidate rows the EGM step produces and needs explicit
branch topology (the `segment_id` labels) to avoid spurious bridges between
unrelated branches. It is a test tool only — host-side NumPy, never a registered
JIT/GPU backend.
"""

import numpy as np


def branches_from_candidates(
    *,
    endog_grid: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray,
    segment_id: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group flat candidate arrays into per-segment sorted branches.

    Return a list of `(x, value, policy)` tuples, one per distinct segment id,
    each sorted ascending in `x` with NaN-padded entries dropped. A branch with a
    repeated abscissa keeps its maximal-value copy, since a dominated within-branch
    duplicate is not part of that branch's piecewise-linear function.
    """
    x = np.asarray(endog_grid, dtype=float)
    v = np.asarray(value, dtype=float)
    p = np.asarray(policy, dtype=float)
    s = np.asarray(segment_id, dtype=float)
    keep = ~np.isnan(x)
    x, v, p, s = x[keep], v[keep], p[keep], s[keep]

    branches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seg in np.unique(s):
        mask = s == seg
        order = np.argsort(x[mask], kind="stable")
        xs, vs, ps = x[mask][order], v[mask][order], p[mask][order]
        # Collapse exact-duplicate abscissae within the branch to the max value.
        keep_x: list[float] = []
        keep_v: list[float] = []
        keep_p: list[float] = []
        i = 0
        while i < len(xs):
            j = i
            best = i
            while j < len(xs) and xs[j] == xs[i]:
                if vs[j] > vs[best]:
                    best = j
                j += 1
            keep_x.append(float(xs[i]))
            keep_v.append(float(vs[best]))
            keep_p.append(float(ps[best]))
            i = j
        branches.append((np.array(keep_x), np.array(keep_v), np.array(keep_p)))
    return branches


def _branch_value_policy(
    branch: tuple[np.ndarray, np.ndarray, np.ndarray], x_query: float
) -> tuple[float, float] | None:
    """Linear value and policy of one branch at `x_query`, or `None` off-range."""
    x, value, policy = branch
    if x_query < x[0] or x_query > x[-1]:
        return None
    return float(np.interp(x_query, x, value)), float(np.interp(x_query, x, policy))


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

    Return `(env_value, env_policy, winner)` arrays aligned with `x_query`. At each
    query the envelope value is the maximum, over the branches defined there, of the
    branch's piecewise-linear value; the policy is the winning branch's interpolated
    policy and `winner` its branch index. Ties break toward the lower segment id for
    determinism. A query outside every branch's range yields NaN value/policy and
    winner `-1`.
    """
    branches = branches_from_candidates(
        endog_grid=endog_grid, value=value, policy=policy, segment_id=segment_id
    )
    xq = np.asarray(x_query, dtype=float)
    env_value = np.full(xq.shape, np.nan)
    env_policy = np.full(xq.shape, np.nan)
    winner = np.full(xq.shape, -1, dtype=int)
    for k, x_point in enumerate(xq):
        best_value, best_policy, best_branch = -np.inf, np.nan, -1
        for branch_index, branch in enumerate(branches):
            evaluated = _branch_value_policy(branch, float(x_point))
            if evaluated is None:
                continue
            branch_value, branch_policy = evaluated
            if branch_value > best_value + tol:
                best_value, best_policy, best_branch = (
                    branch_value,
                    branch_policy,
                    branch_index,
                )
        if best_branch >= 0:
            env_value[k] = best_value
            env_policy[k] = best_policy
            winner[k] = best_branch
    return env_value, env_policy, winner
