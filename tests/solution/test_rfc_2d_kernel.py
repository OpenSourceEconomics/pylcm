"""The on-device 2-D RFC delete kernel matches the host reference mask exactly.

The JAX kernel computes k-NN neighbours by a dense pairwise `top_k` rather than a host
KD-tree, so these tests pin that it reproduces the host reference's keep-mask
(`tests/solution/_rfc_2d_host.py`) bit-for-bit on the canonical clouds and on a random
two-branch cloud — the gate that lets the kernel stand in for the reference on device.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.rfc_2d import rfc_delete_mask_2d
from tests.solution._rfc_2d_host import rfc_delete_mask


def _affine_cloud():
    xs, ys = np.meshgrid(np.linspace(0.0, 0.4, 5), np.linspace(0.0, 0.4, 5))
    states = np.column_stack([xs.ravel(), ys.ravel()])
    slope = np.array([1.3, -0.7])
    values = states @ slope + 2.0
    supgradients = np.tile(slope, (states.shape[0], 1))
    policies = np.zeros((states.shape[0], 1))
    return states, supgradients, values, policies


def _concave_cloud():
    xs, ys = np.meshgrid(np.linspace(0.0, 0.4, 5), np.linspace(0.0, 0.4, 5))
    states = np.column_stack([xs.ravel(), ys.ravel()])
    values = -(states[:, 0] ** 2 + states[:, 1] ** 2)
    supgradients = np.column_stack([-2.0 * states[:, 0], -2.0 * states[:, 1]])
    policies = (0.01 * states[:, 0]).reshape(-1, 1)
    return states, supgradients, values, policies


def _dominated_branch_cloud():
    upper = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
    lower = np.array([[0.0, 0.05], [0.2, 0.05], [0.4, 0.05]])
    states = np.vstack([upper, lower])
    values = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    supgradients = np.zeros((6, 2))
    policies = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]).reshape(-1, 1)
    return states, supgradients, values, policies


def _random_two_branch_cloud():
    rng = np.random.default_rng(20260624)
    base = rng.uniform(0.0, 1.0, size=(40, 2))
    # An upper branch (policy 0) and a lower, dominated branch (policy 5) sharing the
    # same state support so neighbours straddle the policy jump.
    states = np.vstack([base, base + np.array([0.0, 0.01])])
    values = np.concatenate([base.sum(axis=1), base.sum(axis=1) - 0.3])
    supgradients = np.tile(np.array([1.0, 1.0]), (states.shape[0], 1))
    policies = np.concatenate([np.zeros(40), np.full(40, 5.0)]).reshape(-1, 1)
    return states, supgradients, values, policies


@pytest.mark.parametrize(
    "cloud",
    [
        _affine_cloud,
        _concave_cloud,
        _dominated_branch_cloud,
        _random_two_branch_cloud,
    ],
)
def test_kernel_keep_mask_matches_host_reference(cloud):
    """The JAX kernel's keep-mask equals the host KD-tree reference's keep-mask."""
    states, supgradients, values, policies = cloud()

    host_keep = rfc_delete_mask(
        states=states, supgradients=supgradients, values=values, policies=policies
    )
    kernel_keep = np.asarray(
        rfc_delete_mask_2d(
            states=jnp.asarray(states),
            supgradients=jnp.asarray(supgradients),
            values=jnp.asarray(values),
            policies=jnp.asarray(policies),
        )
    )

    np.testing.assert_array_equal(kernel_keep, host_keep)
