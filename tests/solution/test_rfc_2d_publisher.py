"""The on-device 2-D RFC publisher is affine-exact and agrees with host Delaunay.

The local-simplex barycentric publisher is the on-device stand-in for the host Delaunay
reference (D1 of the P6 design). These tests pin that it reproduces survivor values
exactly, is affine-exact inside the survivor hull (matching both the analytic linear
ground truth and the host Delaunay publisher), and falls back to the nearest survivor
outside the hull.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.rfc_2d import rfc_publish_2d
from tests.solution._rfc_2d_host import rfc_publish as rfc_publish_host

SLOPE = np.array([1.3, -0.7])


def _linear_survivor_grid():
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    states = np.column_stack([xs.ravel(), ys.ravel()])
    values = states @ SLOPE + 2.0
    policies = (2.0 * states[:, 0] + states[:, 1]).reshape(-1, 1)
    return states, values, policies


def _publish_kernel(states, values, policies, targets):
    out_values, out_policies = rfc_publish_2d(
        survivor_states=jnp.asarray(states),
        survivor_values=jnp.asarray(values),
        survivor_policies=jnp.asarray(policies),
        target_states=jnp.asarray(targets),
    )
    return np.asarray(out_values), np.asarray(out_policies)


def test_publisher_reproduces_survivor_values_at_survivor_locations():
    """Querying at a survivor returns that survivor's value and policy exactly."""
    states, values, policies = _linear_survivor_grid()
    targets = states[[0, 7, 12, 24]]

    out_values, out_policies = _publish_kernel(states, values, policies, targets)

    np.testing.assert_allclose(out_values, values[[0, 7, 12, 24]], atol=1e-9)
    np.testing.assert_allclose(
        out_policies[:, 0], policies[[0, 7, 12, 24], 0], atol=1e-9
    )


def test_publisher_is_affine_exact_and_matches_host_delaunay():
    """Inside the hull the publisher reproduces the linear field and the host oracle."""
    states, values, policies = _linear_survivor_grid()
    targets = np.array([[0.25, 0.25], [0.4, 0.6], [0.7, 0.3], [0.55, 0.55]])
    analytic_values = targets @ SLOPE + 2.0
    analytic_policies = 2.0 * targets[:, 0] + targets[:, 1]

    out_values, out_policies = _publish_kernel(states, values, policies, targets)
    host_values, host_policies = rfc_publish_host(
        survivor_states=states,
        survivor_values=values,
        survivor_policies=policies,
        target_states=targets,
    )

    np.testing.assert_allclose(out_values, analytic_values, atol=1e-9)
    np.testing.assert_allclose(out_policies[:, 0], analytic_policies, atol=1e-9)
    np.testing.assert_allclose(out_values, host_values, atol=1e-9)
    np.testing.assert_allclose(out_policies[:, 0], host_policies[:, 0], atol=1e-9)


def test_publisher_valid_mask_excludes_deleted_candidates():
    """Publishing the full cloud with a keep-mask ignores the deleted candidates.

    The jit-friendly form passes every candidate plus the cut's keep-mask. Corrupt
    candidates marked invalid must never enter a target's neighbourhood or a triangle,
    so the published field still reproduces the linear ground truth carried by the
    valid candidates.
    """
    states, values, policies = _linear_survivor_grid()
    corrupt_states = np.array([[0.3, 0.3], [0.5, 0.4], [0.6, 0.6]])
    corrupt_values = np.array([99.0, -99.0, 99.0])
    corrupt_policies = np.array([99.0, 99.0, 99.0]).reshape(-1, 1)
    all_states = np.vstack([states, corrupt_states])
    all_values = np.concatenate([values, corrupt_values])
    all_policies = np.vstack([policies, corrupt_policies])
    valid = np.concatenate([np.ones(len(states), bool), np.zeros(3, bool)])
    targets = np.array([[0.25, 0.25], [0.4, 0.6], [0.55, 0.55]])
    analytic_values = targets @ SLOPE + 2.0

    out_values, out_policies = rfc_publish_2d(
        survivor_states=jnp.asarray(all_states),
        survivor_values=jnp.asarray(all_values),
        survivor_policies=jnp.asarray(all_policies),
        target_states=jnp.asarray(targets),
        valid=jnp.asarray(valid),
    )

    np.testing.assert_allclose(np.asarray(out_values), analytic_values, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(out_policies)[:, 0],
        2.0 * targets[:, 0] + targets[:, 1],
        atol=1e-9,
    )


def test_publisher_falls_back_to_nearest_survivor_outside_the_hull():
    """A target outside the survivor hull takes the nearest survivor's value/policy."""
    states, values, policies = _linear_survivor_grid()
    targets = np.array([[5.0, 5.0]])
    nearest = int(np.argmin(np.linalg.norm(states - targets[0], axis=1)))

    out_values, out_policies = _publish_kernel(states, values, policies, targets)

    np.testing.assert_allclose(out_values, [values[nearest]], atol=1e-9)
    np.testing.assert_allclose(out_policies[:, 0], [policies[nearest, 0]], atol=1e-9)
