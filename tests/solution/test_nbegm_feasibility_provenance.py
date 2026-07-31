"""Feasibility of an NBEGM action is decided pointwise, on that point's own budget.

Whether consuming the whole budget is an action depends on the sign of that
point's cash-on-hand and on nothing else. Two properties follow, and both are
contracts the solver owes its user:

- An exactly represented positive budget is an attainable action, however small.
  A finite grid can place a node anywhere, and a node whose budget is one ULP is
  as real an action as one whose budget is a thousand.
- The decision is local. Extending a grid with a far-away node changes no
  primitive at the nodes already there, so it cannot change whether any of them
  has a feasible action.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import _no_save_corner

CRRA = 2.0
DISCOUNT_FACTOR = 0.95


def _corner_value(coh: jnp.ndarray) -> np.ndarray:
    """The no-save corner's value channel on a budget grid."""
    _endog, value, _policy, _marginal = _no_save_corner(
        endog_grid=jnp.arange(coh.shape[0], dtype=jnp.result_type(1.0)),
        coh=coh,
        crra=CRRA,
        discount_factor=DISCOUNT_FACTOR,
        continuation=jnp.zeros_like(coh),
    )
    return np.asarray(value)


def _one_ulp_above(value: float) -> float:
    """The next representable float above `value` in the working precision."""
    dtype = np.dtype(jnp.result_type(1.0))
    return float(np.nextafter(dtype.type(value), dtype.type(np.inf)))


def test_no_save_corner_keeps_a_budget_positive_by_one_ulp():
    """A budget of one ULP is an attainable action and carries a finite value."""
    tiny = _one_ulp_above(0.0) * 2.0**60
    value = _corner_value(jnp.asarray([tiny, 10.0]))
    assert np.isfinite(value[0])


def test_no_save_corner_is_unchanged_by_a_far_away_grid_node():
    """Extending the budget grid upward leaves the other nodes' values untouched."""
    near = _corner_value(jnp.asarray([1.0, 2.0]))
    far = _corner_value(jnp.asarray([1.0, 1.0 / jnp.finfo(jnp.result_type(1.0)).eps]))
    np.testing.assert_array_equal(far[0], near[0])


def test_no_save_corner_kills_an_exactly_zero_budget():
    """A budget of exactly zero affords no action, at any grid scale."""
    value = _corner_value(jnp.asarray([0.0, 10.0]))
    assert np.isnan(value[0])


def test_no_save_corner_kills_a_negative_budget():
    """A negative budget affords no action, at any grid scale."""
    value = _corner_value(jnp.asarray([-1e-30, 10.0]))
    assert np.isnan(value[0])
