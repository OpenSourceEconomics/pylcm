"""Upper-envelope selection is robust to sub-noise ties across backends.

Every keep/drop and winner decision in the envelope refinements compares
quantities that carry rounding noise whose sign follows the backend's reduction
order (a candidate exactly on a neighbour's tangent plane; two segments crossing
at a node). These pins assert that a candidate within the scale-aware noise floor
of such a boundary is resolved the same way regardless of a sub-floor
perturbation, so XLA:CPU and XLA:GPU cannot produce structurally different
envelopes from bit-identical inputs.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.rfc_2d import rfc_delete_mask_2d
from tests.conftest import X64_ENABLED


def test_candidate_just_below_a_tangent_plane_survives_the_2d_rfc_cut():
    """A candidate a sub-floor amount below a neighbour's tangent plane is kept.

    Anchor at the origin with unit gradient in the first coordinate defines the
    plane `value = x`. A second candidate at `x = 0.3` sits on that plane; its
    value is placed a few ulp below the plane — within the scale-aware noise
    floor — across a policy jump and inside the delete radius. The candidate is
    on the envelope, so the cut must keep it rather than delete it on rounding
    noise.
    """
    if not X64_ENABLED:
        pytest.skip("sub-ulp construction is float64-specific")

    on_plane = 0.3
    tiny = 5e-16  # ~9 ulp at 0.3; below the 16*eps*0.3 ~ 1.1e-15 floor
    states = jnp.array([[0.0, 0.0], [0.3, 0.0]])
    supgradients = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    values = jnp.array([0.0, on_plane - tiny])
    policies = jnp.array([[0.0], [1.0]])  # gap 1.0 / dist 0.3 = 3.3 > j_bar

    keep = rfc_delete_mask_2d(
        states=states, supgradients=supgradients, values=values, policies=policies
    )

    assert bool(keep[1])


def test_2d_rfc_keep_mask_is_invariant_to_a_one_ulp_value_perturbation():
    """A one-ulp nudge of the on-plane candidate's value never flips the mask."""
    if not X64_ENABLED:
        pytest.skip("sub-ulp construction is float64-specific")

    states = jnp.array([[0.0, 0.0], [0.3, 0.0]])
    supgradients = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    policies = jnp.array([[0.0], [1.0]])
    base = jnp.array([0.0, 0.3])
    nudged = base.at[1].set(jnp.nextafter(base[1], jnp.asarray(-jnp.inf)))

    keep_base = rfc_delete_mask_2d(
        states=states, supgradients=supgradients, values=base, policies=policies
    )
    keep_nudged = rfc_delete_mask_2d(
        states=states, supgradients=supgradients, values=nudged, policies=policies
    )

    np.testing.assert_array_equal(np.asarray(keep_base), np.asarray(keep_nudged))
