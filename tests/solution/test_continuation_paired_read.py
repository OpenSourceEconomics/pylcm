"""Spec for the paired value/marginal carry read of the Epstein-Zin continuation.

The Epstein-Zin transform marginal `T = sum w V^(-gamma) dV/ds` is the
derivative of the same joint certainty equivalent the value channel carries, so
the child read must supply `dV/ds` as the derivative of the value interpolant it
actually evaluates. The value row is read with monotone cubic Hermite
interpolation (the marginal row as limited node slopes); a *separate* linear
interpolation of the marginal row is a different approximant — where the slope
limiter binds, the two disagree at leading order (endpoint slopes zero force the
linear marginal read to zero while the Hermite value still moves between the
nodes). The paired read differentiates the value interpolant itself, so the
marginal is exactly the value read's slope; the linear expected-utility read
keeps the separate marginal interpolation (a documented second-order
approximation there).
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _aggregate_child_choices
from _lcm.egm.interp import prepare_padded_grid


def _one_row_scene(query: float) -> dict:
    """One smooth carry row: values 1 -> 2 on [0, 1] with zero endpoint slopes."""
    endog = jnp.array([0.0, 1.0])
    value = jnp.array([1.0, 2.0])
    marginal = jnp.array([0.0, 0.0])
    carry = EGMCarry(
        endog_grid=endog,
        value=value,
        marginal_utility=marginal,
        taste_shock_scale=jnp.asarray(0.0),
    )
    search_grid, valid_length = prepare_padded_grid(endog)
    return {
        "carry": carry,
        "prepared_search_grid": search_grid,
        "prepared_valid_length": valid_length,
        "has_taste_shocks": False,
        "child_index": (),
        "child_passive_values": (),
        "child_passive_grids": (),
        "row_queries": jnp.asarray(query),
        "row_gradients": jnp.asarray(1.0),
    }


def test_paired_read_marginal_is_the_derivative_of_the_hermite_value_read() -> None:
    """The paired marginal equals the value interpolant's analytic slope.

    On values `(1, 2)` over `[0, 1]` with zero endpoint slopes, the monotone
    cubic Hermite value read has derivative `6 t (1 - t)` at relative position
    `t`; at `t = 0.25` that is `1.125`, while the marginal row itself reads
    zero everywhere.
    """
    _, marginal = _aggregate_child_choices(
        **_one_row_scene(0.25), paired_marginal_read=True
    )
    np.testing.assert_allclose(np.asarray(marginal), 1.125, rtol=1e-10)


def test_linear_read_keeps_the_separate_marginal_interpolation() -> None:
    """Without pairing, the marginal is the marginal row's own linear read."""
    _, marginal = _aggregate_child_choices(
        **_one_row_scene(0.25), paired_marginal_read=False
    )
    np.testing.assert_allclose(np.asarray(marginal), 0.0, atol=1e-12)


def test_paired_read_marginal_at_an_exact_node_uses_the_declared_side() -> None:
    """An on-node query reads the right piece's endpoint slope, not a subgradient.

    Transitions land on child grid nodes routinely (a zero-savings corner on a
    grid starting at zero), so the paired marginal at an exact node must be
    the declared right-side interpolation derivative. With zero endpoint
    slopes both adjacent Hermite pieces have derivative zero at the node —
    any other value would feed the Euler equation a marginal the value read
    does not have.
    """
    _, marginal = _aggregate_child_choices(
        **_one_row_scene(0.0), paired_marginal_read=True
    )
    np.testing.assert_allclose(np.asarray(marginal), 0.0, atol=1e-12)


def test_paired_read_marginal_above_support_is_zero() -> None:
    """Above the carry's support the value read clamps, so its slope is zero."""
    _, marginal = _aggregate_child_choices(
        **_one_row_scene(1.5), paired_marginal_read=True
    )
    np.testing.assert_allclose(np.asarray(marginal), 0.0, atol=1e-12)
