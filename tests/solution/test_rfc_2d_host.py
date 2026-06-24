"""The host (NumPy/SciPy) 2-D RFC reference behaves as Dobrescu-Shanker Box 1.

These tests pin the off-device ground truth that the on-device multidimensional RFC
backend is validated against: the tangent-dominance cut spares concave segments,
cuts a dominated branch across a policy jump, and the Delaunay publisher reproduces
survivor values exactly inside the hull with a nearest fallback outside it.
"""

import numpy as np

from tests.solution._rfc_2d_host import (
    rfc_delete_mask,
    rfc_publish,
)


def test_affine_cloud_survives_the_cut_intact():
    """A cloud sampled from an affine value with constant policy loses no points.

    Every neighbour lies exactly on every other point's tangent plane (not strictly
    below it) and no policy jump exists, so the cut deletes nothing.
    """
    xs, ys = np.meshgrid(np.linspace(0.0, 0.4, 5), np.linspace(0.0, 0.4, 5))
    states = np.column_stack([xs.ravel(), ys.ravel()])
    slope = np.array([1.3, -0.7])
    values = states @ slope + 2.0
    supgradients = np.tile(slope, (states.shape[0], 1))
    policies = np.zeros((states.shape[0], 1))

    keep = rfc_delete_mask(
        states=states, supgradients=supgradients, values=values, policies=policies
    )

    assert keep.all()


def test_concave_segment_without_a_policy_jump_survives():
    """A single concave segment keeps every point — the jump gate spares concavity.

    On a concave surface every neighbour lies below its neighbours' tangents, but
    with a smoothly varying policy (no discontinuity) the jump gate is never tripped,
    so nothing is deleted.
    """
    xs, ys = np.meshgrid(np.linspace(0.0, 0.4, 5), np.linspace(0.0, 0.4, 5))
    states = np.column_stack([xs.ravel(), ys.ravel()])
    values = -(states[:, 0] ** 2 + states[:, 1] ** 2)
    supgradients = np.column_stack([-2.0 * states[:, 0], -2.0 * states[:, 1]])
    # Policy varies slowly: |dsigma| / dist stays well below the jump threshold.
    policies = (0.01 * states[:, 0]).reshape(-1, 1)

    keep = rfc_delete_mask(
        states=states, supgradients=supgradients, values=values, policies=policies
    )

    assert keep.all()


def test_dominated_branch_across_a_policy_jump_is_cut():
    """A lower branch below the upper's tangent across a policy jump is deleted.

    Two flat branches sit close in state: an upper branch (value 1, policy 0) and a
    lower branch (value 0.5, policy 10) a short distance away. From each upper point
    the nearby lower point lies below its (flat) tangent and across a large policy
    jump within the radius, so every lower point is cut; the upper points, which lie
    above the lower tangents, all survive.
    """
    upper = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
    lower = np.array([[0.0, 0.05], [0.2, 0.05], [0.4, 0.05]])
    states = np.vstack([upper, lower])
    values = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    supgradients = np.zeros((6, 2))
    policies = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]).reshape(-1, 1)

    keep = rfc_delete_mask(
        states=states, supgradients=supgradients, values=values, policies=policies
    )

    np.testing.assert_array_equal(keep, [True, True, True, False, False, False])


def test_publish_reproduces_survivor_values_and_interpolates_linearly():
    """The Delaunay publisher reproduces survivor values and is linear in the hull.

    With survivors sampled from `V = x + y` and policy `p = 2x`, a query at a survivor
    returns that survivor's value exactly, and a query at the cell centre returns the
    linear interpolant.
    """
    survivor_states = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    survivor_values = survivor_states[:, 0] + survivor_states[:, 1]
    survivor_policies = (2.0 * survivor_states[:, 0]).reshape(-1, 1)
    targets = np.array([[0.0, 0.0], [0.5, 0.5]])

    values, policies = rfc_publish(
        survivor_states=survivor_states,
        survivor_values=survivor_values,
        survivor_policies=survivor_policies,
        target_states=targets,
    )

    np.testing.assert_allclose(values, [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(policies[:, 0], [0.0, 1.0], atol=1e-9)


def test_publish_falls_back_to_nearest_survivor_outside_the_hull():
    """A target outside the survivor convex hull takes the nearest survivor's value."""
    survivor_states = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    survivor_values = np.array([0.0, 1.0, 1.0, 2.0])
    survivor_policies = np.array([0.0, 1.0, 1.0, 2.0]).reshape(-1, 1)
    targets = np.array([[5.0, 5.0]])

    values, policies = rfc_publish(
        survivor_states=survivor_states,
        survivor_values=survivor_values,
        survivor_policies=survivor_policies,
        target_states=targets,
    )

    np.testing.assert_allclose(values, [2.0], atol=1e-9)
    np.testing.assert_allclose(policies[:, 0], [2.0], atol=1e-9)
