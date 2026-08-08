"""A lottery spanning more binades than the exponent field can hold.

A joint probability formed from several rare factors travels as a coefficient
beside a base-two scale. Where the spread between the likeliest and the rarest
node exceeds what one shared scale can carry, the rare node may be understated
or dropped — the declared approximation — but never made likelier than it is.
Under a power mean with a large exponent that ratio is what the continuation
value is made of, so enlarging it moves the value and reverses choices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import QuasiArithmeticMean
from _lcm.power_mean import weighted_power_mean
from _lcm.probability import (
    flattened_to_one_scale,
    is_live,
    reconcile_scales,
    scaled_exact_product,
)

# A competing continuation the exact lottery loses to and an enlarged one wins.
_ALTERNATIVE = 0.47


def _witness() -> tuple[int, int, int]:
    """Return the factor exponent, factor count and power exponent for the dtype."""
    if jnp.asarray(0.0).dtype == jnp.float32:
        return -100, 3, 252
    return -700, 4, 2044


def _decoded_ratio(coefficients: jnp.ndarray, shifts: jnp.ndarray) -> np.longdouble:
    """Return the rare node's probability relative to the common one, as stored."""
    per_entry = np.broadcast_to(np.asarray(shifts), np.asarray(coefficients).shape)
    decoded = [
        np.longdouble(np.asarray(coefficients)[i])
        * np.exp2(-np.longdouble(per_entry[i]))
        for i in (0, 1)
    ]
    return decoded[1] / decoded[0]


def _reconciled_witness() -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Return the reconciled two-node lottery and the true log2 of its ratio."""
    factor_exponent, n_factors, _ = _witness()
    dtype = jnp.asarray(0.0).dtype
    factors = jnp.full((n_factors, 1), jnp.asarray(2.0**factor_exponent, dtype=dtype))
    rare, rare_shift = scaled_exact_product(factors)
    coefficients, shifts = reconcile_scales(
        jnp.concatenate([jnp.ones((1,), dtype=dtype), rare]),
        jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.full((1,), rare_shift)]),
    )
    return coefficients, shifts, factor_exponent * n_factors


def test_a_capped_scale_never_makes_a_rare_node_likelier_than_it_is():
    """The stored probability of an unliftable node is at most its true one."""
    coefficients, shifts, true_log2_ratio = _reconciled_witness()

    assert _decoded_ratio(coefficients, shifts) <= np.exp2(
        np.longdouble(true_log2_ratio)
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_wide_spread_lottery_prices_at_its_exact_continuation(*, compiled: bool):
    """`(Σ w̃ v^p)^(1/p)` holds when the lottery spans more than the exponent range."""
    coefficients, shifts, true_log2_ratio = _reconciled_witness()
    _, _, power_exponent = _witness()
    dtype = jnp.asarray(0.0).dtype
    mean = jax.jit(weighted_power_mean) if compiled else weighted_power_mean

    got = mean(
        values=jnp.asarray([0.0, 1.0], dtype=dtype),
        weights=coefficients,
        exponent=jnp.asarray(power_exponent, dtype=dtype),
        shifts=shifts,
    )

    expected = float(np.exp2(np.longdouble(true_log2_ratio) / power_exponent))
    np.testing.assert_allclose(
        np.asarray(got), expected, rtol=5e-6 if dtype == jnp.float32 else 1e-12
    )


def test_a_wide_spread_lottery_loses_to_the_alternative_it_should_lose_to():
    """The exact continuation sits below the competing one, so the choice is it."""
    coefficients, shifts, true_log2_ratio = _reconciled_witness()
    _, _, power_exponent = _witness()
    dtype = jnp.asarray(0.0).dtype

    got = weighted_power_mean(
        values=jnp.asarray([0.0, 1.0], dtype=dtype),
        weights=coefficients,
        exponent=jnp.asarray(power_exponent, dtype=dtype),
        shifts=shifts,
    )

    assert (
        float(np.exp2(np.longdouble(true_log2_ratio) / power_exponent)) < _ALTERNATIVE
    )
    assert float(np.asarray(got)) < _ALTERNATIVE


def test_flattening_never_states_a_probability_larger_than_it_is():
    """Put on one scale as numbers, a node is understated rather than enlarged."""
    coefficients, shifts, true_log2_ratio = _reconciled_witness()

    flattened = flattened_to_one_scale(
        coefficients=coefficients,
        shifts=shifts,
        values=jnp.ones_like(coefficients),
    )

    ratio = np.longdouble(np.asarray(flattened)[1]) / np.longdouble(
        np.asarray(flattened)[0]
    )
    assert ratio <= np.exp2(np.longdouble(true_log2_ratio))


def test_flattening_keeps_an_event_that_can_occur_live():
    """A node too rare for the shared scale stays an event where `±inf` stands there.

    A represented zero is the null event, and it annihilates whatever value
    stands at its node. A state where no action is feasible carries `-inf`, and
    rounding its probability to zero would drop that infinity and report an
    ordinary number for a continuation that has none. Every strictly positive
    weight gives the same infinity, so flooring the weight loses nothing.
    """
    coefficients, shifts, _ = _reconciled_witness()

    flattened = flattened_to_one_scale(
        coefficients=coefficients,
        shifts=shifts,
        values=jnp.asarray([2.0, -jnp.inf], dtype=coefficients.dtype),
    )

    assert bool(np.asarray(is_live(flattened))[1])


def test_a_rare_node_keeps_its_infinity_through_a_general_transform():
    """`-inf` at a node that can occur survives a lottery reduced as numbers."""
    coefficients, shifts, _ = _reconciled_witness()
    dtype = jnp.asarray(0.0).dtype

    got = QuasiArithmeticMean(
        transform=lambda value: value, inverse=lambda value: value
    ).aggregate_scaled(
        values=jnp.asarray([2.0, -jnp.inf], dtype=dtype),
        coefficients=coefficients,
        shifts=shifts,
        params={},
    )

    assert float(np.asarray(got)) == float("-inf")
