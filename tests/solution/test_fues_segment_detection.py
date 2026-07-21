"""FUES segment identity is a heuristic: an undetected switch bridges branches.

FUES groups candidates into value-function segments by thresholding the
implied-savings slope (`fues_jump_thresh`); crossings are inserted only
*between* detected segments. When the cross-segment slope stays below the
threshold and no explicit `segment_id` labels are supplied — the DC-EGM kernel
supplies none — two distinct value branches merge into one row: no crossing is
inserted and the row linearly bridges the switch. This is why the off-grid
simulation policy read requires MSS and keeps every FUES regime on the
grid-argmax path.

The correspondence below is consistent with log utility: within each branch
`dV/dR = 1/c` (envelope theorem), policies are `0.3` and `0.1`, and the two
branch value lines cross at `R = 1.5`. The cross-branch implied-savings slope
is `(1.51 - 1.10) / (1.61 - 1.40) ≈ 1.95`, just below the default threshold
`2.0`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.fues import refine_envelope

_LEFT_GRID = jnp.array([1.0, 1.2, 1.4])
_RIGHT_GRID = jnp.array([1.61, 1.81, 2.01])
_LEFT_POLICY = 0.3
_RIGHT_POLICY = 0.1
_CROSSING = 1.5
_QUERY = 1.45


def _branch_value(grid: jnp.ndarray, policy: float) -> jnp.ndarray:
    return 20.0 + (1.0 / policy) * (grid - _CROSSING)


def _refine(segment_id: jnp.ndarray | None) -> tuple[jnp.ndarray, ...]:
    grid = jnp.concatenate([_LEFT_GRID, _RIGHT_GRID])
    policy = jnp.concatenate([jnp.full(3, _LEFT_POLICY), jnp.full(3, _RIGHT_POLICY)])
    value = jnp.concatenate(
        [
            _branch_value(_LEFT_GRID, _LEFT_POLICY),
            _branch_value(_RIGHT_GRID, _RIGHT_POLICY),
        ]
    )
    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=20,
        jump_thresh=2.0,
        n_points_to_scan=None,
        segment_id=segment_id,
    )
    return refined_grid, refined_policy, refined_value


def test_unlabeled_fues_bridges_a_sub_threshold_segment_switch():
    """Without labels, the exhaustive scan merges the branches and misses the crossing.

    The read at `R = 1.45` then over-states the value (the bridge between the
    two branch lines) and returns a consumption belonging to neither branch,
    instead of the true envelope value `20 + (1/0.3)(1.45 - 1.5)` on the left
    branch with policy `0.3`.
    """
    refined_grid, refined_policy, refined_value = _refine(segment_id=None)

    replay_value = interp_on_padded_grid(
        x_query=jnp.asarray(_QUERY), xp=refined_grid, fp=refined_value
    )
    replay_policy = interp_on_padded_grid(
        x_query=jnp.asarray(_QUERY), xp=refined_grid, fp=refined_policy
    )
    true_value = 20.0 + (1.0 / _LEFT_POLICY) * (_QUERY - _CROSSING)

    assert float(replay_value) == pytest.approx(20.0079, abs=1e-3)
    assert float(true_value) == pytest.approx(19.8333, abs=1e-3)
    assert float(replay_policy) == pytest.approx(0.2524, abs=1e-3)


def test_labeled_fues_inserts_the_crossing_the_heuristic_missed():
    """With true segment labels, the same routine recovers the exact envelope.

    The crossing at `R = 1.5` is inserted and the read at `R = 1.45` returns
    the left branch's value and policy — confirming the failure mode is
    segment identification, not the scan width.
    """
    labels = jnp.concatenate([jnp.zeros(3), jnp.ones(3)])
    refined_grid, refined_policy, refined_value = _refine(segment_id=labels)

    replay_value = interp_on_padded_grid(
        x_query=jnp.asarray(_QUERY), xp=refined_grid, fp=refined_value
    )
    replay_policy = interp_on_padded_grid(
        x_query=jnp.asarray(_QUERY), xp=refined_grid, fp=refined_policy
    )
    true_value = 20.0 + (1.0 / _LEFT_POLICY) * (_QUERY - _CROSSING)

    np.testing.assert_allclose(float(replay_value), float(true_value), atol=1e-4)
    np.testing.assert_allclose(float(replay_policy), _LEFT_POLICY, atol=1e-4)
