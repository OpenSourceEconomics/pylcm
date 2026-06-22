"""The region-aware envelope gradient and switch mask are correct.

The published marginal value of liquid wealth is `u'(c) = c**(-crra)` in every region —
matching the region inverses' `value_grad_m` — and the pension marginal is
`discount_factor * W_b`. The switch mask flags exactly the targets at a cross-segment
boundary, where the envelope is non-differentiable.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.mesh_gradient import region_aware_gradient, switch_mask
from _lcm.egm.two_asset_inverse import invert_acon_cloud

_DISCOUNT = 0.95
_CRRA = 2.0
_MATCH = 1.0


def test_value_grad_m_is_marginal_utility():
    """`V_m = u'(c) = c**(-crra)`, the region-independent envelope marginal."""
    consumption = jnp.array([0.5, 1.0, 2.0])
    gradient = region_aware_gradient(
        consumption=consumption,
        post_decision_grad_b=jnp.array([1.0, 1.0, 1.0]),
        discount_factor=_DISCOUNT,
        crra=_CRRA,
    )
    np.testing.assert_allclose(
        np.asarray(gradient.value_grad_m), np.asarray(consumption ** (-_CRRA))
    )


def test_value_grad_n_is_discounted_post_decision_gradient():
    """`V_n = discount_factor * W_b`."""
    grad_b = jnp.array([0.2, 0.4, 0.6])
    gradient = region_aware_gradient(
        consumption=jnp.array([1.0, 1.0, 1.0]),
        post_decision_grad_b=grad_b,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
    )
    np.testing.assert_allclose(
        np.asarray(gradient.value_grad_n), np.asarray(_DISCOUNT * grad_b)
    )


def test_value_grad_m_matches_the_constrained_region_inverse():
    """`V_m = u'(c)` reproduces `invert_acon_cloud.value_grad_m` (not beta*w_a).

    The borrowing-constrained inverse already publishes `consumption**(-crra)` for the
    liquid marginal; the envelope gradient uses the same rule, so they agree.
    """
    consumption = jnp.array([[0.5, 0.8], [1.2, 1.5]])
    b = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    w_b = consumption ** (-_CRRA) / (_DISCOUNT * 1.6)
    cloud = invert_acon_cloud(
        consumption=consumption,
        b=b,
        post_decision_value_at_zero_a=jnp.zeros((2, 2)),
        w_b_at_zero_a=w_b,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
    )
    gradient = region_aware_gradient(
        consumption=cloud.consumption,
        post_decision_grad_b=w_b,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
    )
    np.testing.assert_allclose(
        np.asarray(gradient.value_grad_m), np.asarray(cloud.value_grad_m), rtol=1e-12
    )


def test_switch_mask_flags_the_segment_boundary():
    """A vertical segment boundary flags the two columns straddling it, not the rest."""
    # Winning segment 0 in columns 0-1, segment 1 in column 2, on a 3x3 grid.
    segment = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=jnp.int32)
    mask = np.asarray(switch_mask(segment=segment, grid_shape=(3, 3))).reshape(3, 3)
    np.testing.assert_array_equal(mask[:, 0], [False, False, False])
    np.testing.assert_array_equal(mask[:, 1], [True, True, True])
    np.testing.assert_array_equal(mask[:, 2], [True, True, True])


def test_switch_mask_is_all_false_for_a_single_segment():
    """With one winning segment everywhere there is no switch."""
    segment = jnp.zeros(9, dtype=jnp.int32)
    mask = np.asarray(switch_mask(segment=segment, grid_shape=(3, 3)))
    assert not mask.any()
