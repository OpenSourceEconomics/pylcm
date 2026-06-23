"""Tests for the branch-aware exact upper-envelope oracle.

The oracle is the independent reference the EGM envelope backends are checked
against. These tests pin the oracle's own correctness on hand-computed cases —
including the exact-node branch crossing the production FUES gets wrong (audit
F5) — and confirm it agrees with FUES on a crossing the FUES scan resolves.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.fues import refine_envelope
from tests.solution._envelope_oracle import exact_envelope


def test_oracle_recovers_v_shape_on_exact_node_crossing():
    """Two branches crossing on a shared node give the true V-shaped envelope.

    Branch 0 is `V(R) = R` (policy 0); branch 1 is `V(R) = 1 - R` (policy 10).
    They cross exactly at the node `R = 0.5`. The exact envelope is
    `max(R, 1 - R)` with its minimum `0.5` at the crossing, and the winning policy
    switches from branch 1 (left) to branch 0 (right). This is the case the
    production FUES collapses to a flat envelope (audit F5).
    """
    endog_grid = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    value = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    policy = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    segment_id = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    x_query = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    env_value, env_policy, _winner = exact_envelope(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        segment_id=segment_id,
        x_query=x_query,
    )

    np.testing.assert_allclose(env_value, [1.0, 0.75, 0.5, 0.75, 1.0], atol=1e-12)
    # Left of the crossing branch 1 (policy 10) wins; right of it branch 0
    # (policy 0). The exact-node tie resolves deterministically to the lower id.
    np.testing.assert_allclose(env_policy, [10.0, 10.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_oracle_is_the_interpolant_for_a_single_branch():
    """With one branch the envelope is exactly that branch's interpolant."""
    endog_grid = np.array([0.0, 1.0, 2.0, 3.0])
    value = np.array([0.0, 1.0, 1.5, 1.7])  # concave, increasing
    policy = np.array([0.0, 0.5, 0.9, 1.2])
    segment_id = np.zeros(4)
    x_query = np.array([0.5, 1.5, 2.5])

    env_value, env_policy, winner = exact_envelope(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        segment_id=segment_id,
        x_query=x_query,
    )

    np.testing.assert_allclose(env_value, np.interp(x_query, endog_grid, value))
    np.testing.assert_allclose(env_policy, np.interp(x_query, endog_grid, policy))
    np.testing.assert_array_equal(winner, [0, 0, 0])


def test_oracle_query_outside_all_branches_is_nan():
    """A query beyond every branch's support yields NaN and winner -1."""
    env_value, env_policy, winner = exact_envelope(
        endog_grid=np.array([1.0, 2.0]),
        value=np.array([1.0, 2.0]),
        policy=np.array([0.5, 1.0]),
        segment_id=np.zeros(2),
        x_query=np.array([0.0, 5.0]),
    )
    assert np.isnan(env_value).all()
    assert np.isnan(env_policy).all()
    np.testing.assert_array_equal(winner, [-1, -1])


def test_oracle_agrees_with_fues_on_a_between_node_crossing():
    """On a crossing the FUES scan resolves, oracle and FUES envelopes agree.

    Two labelled branches cross strictly between grid nodes (not on one), the case
    the FUES crossing-insertion handles. Over the region where both branches are
    defined, the FUES-refined value function and the exact oracle envelope match.
    """
    endog_grid = jnp.array([1.0, 1.5, 2.0, 2.5])
    policy = 0.5 * endog_grid
    value = jnp.array([2.0, 1.8, 2.2, 3.0])
    segment_id = jnp.array([0.0, 1.0, 0.0, 1.0])

    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=12,
        segment_id=segment_id,
    )
    # Branch 0 spans [1, 2], branch 1 spans [1.5, 2.5]; both are defined on
    # [1.5, 2.0], which brackets the crossing.
    x_query = jnp.linspace(1.5, 2.0, 11)
    fues_value = np.asarray(
        interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    )
    oracle_value, _policy, _winner = exact_envelope(
        endog_grid=np.asarray(endog_grid),
        value=np.asarray(value),
        policy=np.asarray(policy),
        segment_id=np.asarray(segment_id),
        x_query=np.asarray(x_query),
    )
    np.testing.assert_allclose(fues_value, oracle_value, atol=1e-6)
