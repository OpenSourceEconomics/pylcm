"""The G2EGM direct-Bellman hole-fill returns the maximizing policy, not just its value.

Targets the regular `(m, n)` grid that no segment mesh covers are filled by a direct
search over a coarse `(consumption, deposit)` policy grid. The fill must publish the
argmax policy alongside its value, so a hole cell's consumption and pension deposit are
consistent with its filled value — not left as the failed envelope's stale policy.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_g2egm_step import _direct_bellman_fill


def test_direct_bellman_fill_returns_the_argmax_policy_and_value():
    """The fill returns the coarse-grid candidate that maximizes the objective.

    With a concave objective peaked at a target-specific `(c*, d*)`, the best feasible
    candidate is the grid point nearest that peak. The fill must return both that
    candidate's value and the candidate itself, per target.
    """
    consumption_grid = jnp.asarray([1.0, 2.0, 3.0])
    deposit_grid = jnp.asarray([0.0, 1.0, 2.0])

    # The target carries its own objective peak `(c*, d*)`; the objective is concave in
    # the policy around that peak and feasible everywhere.
    def objective(target, policy):
        peak_c, peak_d = target[0], target[1]
        value = -((policy[0] - peak_c) ** 2 + (policy[1] - peak_d) ** 2)
        return value, jnp.isfinite(policy[0])

    targets = jnp.asarray([[2.0, 1.0], [3.0, 0.0]])
    value, policy = _direct_bellman_fill(
        targets=targets,
        objective=objective,
        consumption_grid=consumption_grid,
        deposit_grid=deposit_grid,
    )

    # Peak (2, 1) sits exactly on a grid node → value 0 at policy (2, 1); peak (3, 0)
    # likewise at (3, 0).
    np.testing.assert_allclose(np.asarray(value), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(policy), [[2.0, 1.0], [3.0, 0.0]], atol=1e-12)


def test_direct_bellman_fill_skips_infeasible_candidates():
    """An infeasible best-by-value candidate is not selected.

    The objective peaks at a grid node that is masked infeasible, so the fill must fall
    back to the best *feasible* candidate, returning its value and policy.
    """
    consumption_grid = jnp.asarray([1.0, 2.0, 3.0])
    deposit_grid = jnp.asarray([0.0])

    def objective(target, policy):  # noqa: ARG001
        # The value rises toward the top consumption node, but candidates at or above
        # 2.5 are masked infeasible, so the best feasible node is the one just below.
        value = -((policy[0] - 3.0) ** 2)
        feasible = policy[0] < 2.5
        return value, feasible

    targets = jnp.asarray([[0.0, 0.0]])
    value, policy = _direct_bellman_fill(
        targets=targets,
        objective=objective,
        consumption_grid=consumption_grid,
        deposit_grid=deposit_grid,
    )

    np.testing.assert_allclose(np.asarray(policy), [[2.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(np.asarray(value), [-1.0], atol=1e-12)


def test_direct_bellman_fill_marks_all_infeasible_targets():
    """A target with no feasible candidate keeps a `-inf` filled value."""
    consumption_grid = jnp.asarray([1.0, 2.0])
    deposit_grid = jnp.asarray([0.0])

    def objective(target, policy):  # noqa: ARG001
        return jnp.asarray(0.0), policy[0] < 0.0

    targets = jnp.asarray([[0.0, 0.0]])
    value, _policy = _direct_bellman_fill(
        targets=targets,
        objective=objective,
        consumption_grid=consumption_grid,
        deposit_grid=deposit_grid,
    )

    assert not np.isfinite(np.asarray(value)).any()
