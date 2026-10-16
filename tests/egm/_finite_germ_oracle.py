"""Economic declarations and native geometry for finite query components."""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    refine_envelope_exact,
)
from lcm.typing import ContinuousAction, ContinuousState, FloatND


def _utility(*, consumption: ContinuousAction, new_stock: ContinuousState) -> FloatND:
    delta = consumption - 1.0
    return (
        jnp.log(consumption)
        + (delta**3 / 6.0 + delta**4 / 2.0) * jnp.exp(-100.0 * delta**2)
        - (new_stock - 1.0) ** 2
    )


def _gross(*, liquid: ContinuousState) -> FloatND:
    return liquid + 1.25


def _cost(*, stock: ContinuousState, new_stock: ContinuousState) -> FloatND:
    return (new_stock - stock) ** 2 / 4.0


def _new_stock(
    *, stock: ContinuousState, investment: ContinuousAction
) -> ContinuousState:
    return stock + investment


def _floor(*, stock: ContinuousState) -> FloatND:
    return stock / 4.0


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings


def _next_stock(*, new_stock: ContinuousState) -> ContinuousState:
    return new_stock


def _same_stock(*, stock: ContinuousState) -> ContinuousState:
    return stock


def _parent_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _terminal(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return jnp.zeros_like(liquid + stock)


def _emitted_row(*, capacity: int, empty: bool) -> OwnedEnvelope:
    """Use native provenance emission for a supplied original two-candidate link."""
    missing = jnp.asarray([jnp.nan, jnp.nan])
    result = refine_envelope_exact(
        endog_grid=missing if empty else jnp.asarray([1.0, 1.5]),
        policy=missing if empty else jnp.asarray([0.875, 1.25]),
        value=missing if empty else jnp.asarray([-37.0 / 128.0, 29.0 / 128.0]),
        savings=missing if empty else jnp.asarray([0.125, 0.25]),
        n_refined=capacity,
    )
    assert isinstance(result, OwnedEnvelope)
    assert int(result.n_kept) == (0 if empty else 2)
    if not empty:
        np.testing.assert_array_equal(result.source_left_index[:2], [0, 0])
        np.testing.assert_array_equal(result.source_right_index[:2], [1, 1])
        np.testing.assert_array_equal(result.savings[:2], [0.125, 0.25])
        assert bool(result.link_valid[0])
    return result
