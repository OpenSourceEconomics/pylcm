"""The 2-D G2EGM upper envelope selects the correct feasible candidate.

The envelope must maximize the recomputed objective over every admissible triangle —
covering and mildly extrapolated alike — and drop infeasible interpolated policies
before the maximum. The tests pin the two selection behaviors an adversarial audit
flagged: an extrapolated-but-admissible triangle holding the true maximizer must beat a
covering triangle with a lower value (a cover-first rule would miss it), and an
infeasible interpolated policy must never win.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.mesh_envelope import (
    SegmentMesh,
    first_envelope,
    second_envelope,
)


def _value_is_policy(_state, policy):
    """Objective equal to the (scalar) interpolated policy; always feasible."""
    return policy[0], jnp.ones((), dtype=bool)


def _always_infeasible(_state, policy):
    return policy[0], jnp.zeros((), dtype=bool)


def test_first_envelope_prefers_admissible_extrapolated_over_covering():
    """An extrapolated-but-admissible triangle with the higher value wins.

    Triangle B has the target `(1.2, 1.2)` outside but admissible (weights
    `(-0.2, 0.6, 0.6)`) with interpolated value 1.0; triangle A covers the target with
    value 0.0. The within-segment envelope must return 1.0 — a cover-first rule that
    only consults A would return 0.0.
    """
    mesh = SegmentMesh(
        region_label=0,
        node_state=jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],  # triangle B: (1.2,1.2) extrapolated, admissible
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 3.0],  # triangle A: (1.2,1.2) covered
            ]
        ),
        node_policy=jnp.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]),
        simplices=jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32),
        valid_node=jnp.ones(6, dtype=bool),
    )
    values, policies = first_envelope(
        mesh=mesh,
        targets=jnp.array([[1.2, 1.2]]),
        objective=_value_is_policy,
        threshold=0.25,
    )
    np.testing.assert_allclose(np.asarray(values), [1.0], atol=1e-9)
    np.testing.assert_allclose(np.asarray(policies), [[1.0]], atol=1e-9)


def test_first_envelope_masks_infeasible_candidates():
    """An infeasible interpolated policy is masked to negative infinity."""
    mesh = SegmentMesh(
        region_label=0,
        node_state=jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]),
        node_policy=jnp.array([[1.0], [1.0], [1.0]]),
        simplices=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        valid_node=jnp.ones(3, dtype=bool),
    )
    values, _policies = first_envelope(
        mesh=mesh,
        targets=jnp.array([[1.0, 1.0]]),
        objective=_always_infeasible,
        threshold=0.25,
    )
    assert np.isneginf(np.asarray(values)[0])


def test_first_envelope_no_admissible_triangle_returns_neg_inf():
    """A target outside every admissible triangle gets no candidate."""
    mesh = SegmentMesh(
        region_label=0,
        node_state=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        node_policy=jnp.array([[1.0], [1.0], [1.0]]),
        simplices=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        valid_node=jnp.ones(3, dtype=bool),
    )
    values, _policies = first_envelope(
        mesh=mesh,
        targets=jnp.array([[5.0, 5.0]]),  # far outside the unit triangle
        objective=_value_is_policy,
        threshold=0.25,
    )
    assert np.isneginf(np.asarray(values)[0])


def test_first_envelope_drops_triangles_touching_invalid_nodes():
    """A triangle with an invalid node is not a candidate even when it covers."""
    mesh = SegmentMesh(
        region_label=0,
        node_state=jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]),
        node_policy=jnp.array([[1.0], [1.0], [1.0]]),
        simplices=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        valid_node=jnp.array([True, True, False]),
    )
    values, _policies = first_envelope(
        mesh=mesh,
        targets=jnp.array([[1.0, 1.0]]),
        objective=_value_is_policy,
        threshold=0.25,
    )
    assert np.isneginf(np.asarray(values)[0])


def test_second_envelope_takes_max_segment_and_gathers_policy():
    """The across-segment envelope picks the higher value and its segment's policy."""
    segment_values = jnp.array([[0.3, 0.9], [0.7, 0.2]])  # (n_segment=2, n_target=2)
    segment_policies = jnp.array(
        [
            [[10.0], [11.0]],  # segment 0 policies per target
            [[20.0], [21.0]],  # segment 1 policies per target
        ]
    )
    result = second_envelope(
        segment_values=segment_values, segment_policies=segment_policies
    )
    np.testing.assert_allclose(np.asarray(result.value), [0.7, 0.9], atol=1e-9)
    np.testing.assert_array_equal(np.asarray(result.segment), [1, 0])
    np.testing.assert_allclose(np.asarray(result.policy), [[20.0], [11.0]], atol=1e-9)
