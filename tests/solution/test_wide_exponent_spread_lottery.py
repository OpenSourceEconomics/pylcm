"""A lottery spanning more binades than the exponent field can hold.

A joint probability formed from several rare factors travels as a coefficient
beside a base-two scale, and every step that reads several such pairs together
keeps each one's own scale. The rarest node of such a lottery is therefore
neither dropped nor made likelier than it is, however far below the likeliest
one it sits. Under a power mean with a large exponent that ratio is what the
continuation value is made of, so losing it moves the value and reverses
choices.

A consumer that has to name the weights as plain numbers is the one place a
node may be understated — the declared approximation — and never enlarged.
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
    normalized_scaled_weights,
    restored_against_a_nonfinite_value,
    scaled_exact_product,
)
from _lcm.regime_building.Q_and_F import _expectation_over_stochastic_nodes
from _lcm.zero_safe import scaled_weighted_terms

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


def _wide_spread_witness() -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Return a two-node lottery as scaled pairs, and the true log2 of its ratio."""
    factor_exponent, n_factors, _ = _witness()
    dtype = jnp.asarray(0.0).dtype
    factors = jnp.full((n_factors, 1), jnp.asarray(2.0**factor_exponent, dtype=dtype))
    rare, rare_shift = scaled_exact_product(factors)
    coefficients = jnp.concatenate([jnp.ones((1,), dtype=dtype), rare])
    shifts = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.full((1,), rare_shift, dtype=jnp.int32)]
    )
    return coefficients, shifts, factor_exponent * n_factors


def _relative_tolerance() -> float:
    return 5e-6 if jnp.asarray(0.0).dtype == jnp.float32 else 1e-12


def test_normalizing_keeps_a_rare_node_at_its_own_probability():
    """Given unit mass, the rarest node's share is still the share it had."""
    coefficients, shifts, true_log2_ratio = _wide_spread_witness()

    normalized, normalized_shifts = normalized_scaled_weights(
        coefficients=coefficients, shifts=shifts
    )

    np.testing.assert_allclose(
        float(_decoded_ratio(normalized, normalized_shifts)),
        float(np.exp2(np.longdouble(true_log2_ratio))),
        rtol=_relative_tolerance(),
    )


def test_normalizing_leaves_no_node_at_a_represented_zero():
    """A node that can occur is still a coefficient the format holds."""
    coefficients, shifts, _ = _wide_spread_witness()

    normalized, _ = normalized_scaled_weights(coefficients=coefficients, shifts=shifts)

    assert bool(jnp.all(is_live(normalized)))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_wide_spread_lottery_prices_at_its_exact_continuation(*, compiled: bool):
    """`(Σ w̃ v^p)^(1/p)` holds when the lottery spans more than the exponent range."""
    coefficients, shifts, true_log2_ratio = _wide_spread_witness()
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
    np.testing.assert_allclose(np.asarray(got), expected, rtol=_relative_tolerance())


def test_a_wide_spread_lottery_loses_to_the_alternative_it_should_lose_to():
    """The exact continuation sits below the competing one, so the choice is it."""
    coefficients, shifts, true_log2_ratio = _wide_spread_witness()
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


def test_the_linear_mean_reads_a_node_at_the_scale_it_carries():
    """A node's scale is part of its probability, not a detail of its storage.

    Two nodes with the same coefficient and different scales are two different
    probabilities. Reading the coefficients alone would price a node `2**-300`
    likely as an even one, and a large value there would then carry the mean.
    """
    dtype = jnp.asarray(0.0).dtype
    large = jnp.asarray(2.0**100, dtype=dtype)

    got = _expectation_over_stochastic_nodes(
        values=jnp.asarray([1.0, large], dtype=dtype),
        weights=jnp.ones(2, dtype=dtype),
        shifts=jnp.asarray([0, 300], dtype=jnp.int32),
    )

    np.testing.assert_allclose(float(np.asarray(got)), 1.0, rtol=_relative_tolerance())


def test_flattening_never_states_a_probability_larger_than_it_is():
    """Put on one scale as numbers, a node is understated rather than enlarged."""
    coefficients, shifts, true_log2_ratio = _wide_spread_witness()

    flattened = flattened_to_one_scale(coefficients=coefficients, shifts=shifts)

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
    coefficients, shifts, _ = _wide_spread_witness()

    flattened = restored_against_a_nonfinite_value(
        coefficients=coefficients,
        lowered=flattened_to_one_scale(coefficients=coefficients, shifts=shifts),
        values=jnp.asarray([2.0, -jnp.inf], dtype=coefficients.dtype),
    )

    assert bool(np.asarray(is_live(flattened))[1])


def test_flattening_keeps_an_event_carrying_a_nan_live():
    """A node too rare for the shared scale stays an event where `NaN` stands there.

    A `NaN` at a node that can occur is a misspecified model, and the answer is
    that `NaN` whatever the node's probability is. Rounding the weight to a
    represented zero would let the zero-safe reduction read the node as the null
    event and report an ordinary number for a model that has none.
    """
    coefficients, shifts, _ = _wide_spread_witness()

    flattened = restored_against_a_nonfinite_value(
        coefficients=coefficients,
        lowered=flattened_to_one_scale(coefficients=coefficients, shifts=shifts),
        values=jnp.asarray([2.0, jnp.nan], dtype=coefficients.dtype),
    )

    assert bool(np.asarray(is_live(flattened))[1])


def test_a_rare_node_keeps_its_nan_through_a_general_transform():
    """`NaN` at a node that can occur survives a lottery reduced as numbers."""
    coefficients, shifts, _ = _wide_spread_witness()
    dtype = jnp.asarray(0.0).dtype

    got = QuasiArithmeticMean(
        transform=lambda value: value, inverse=lambda value: value
    ).aggregate_scaled(
        values=jnp.asarray([2.0, jnp.nan], dtype=dtype),
        coefficients=coefficients,
        shifts=shifts,
        params={},
    )

    assert bool(jnp.isnan(got))


def test_a_rare_node_keeps_its_infinity_through_a_general_transform():
    """`-inf` at a node that can occur survives a lottery reduced as numbers."""
    coefficients, shifts, _ = _wide_spread_witness()
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


def test_a_terms_scale_is_split_so_a_large_value_does_not_overflow_it():
    """A node's contribution stays in range when its value is near the top of it.

    An ordinary weight meeting a value near the format's largest, at a scale
    that brings the product back down, has an answer the format holds. Applying
    the whole scale to the product first would overflow on the way to it.
    """
    dtype = jnp.zeros(()).dtype
    largest = jnp.finfo(dtype).max

    terms = scaled_weighted_terms(
        coefficients=jnp.asarray([1.9, 1.0], dtype=dtype),
        shifts=jnp.asarray([10, 0], dtype=jnp.int32),
        values=jnp.asarray([largest, 1.0], dtype=dtype),
    )

    expected = np.ldexp(np.longdouble(1.9) * np.longdouble(largest), -10)
    np.testing.assert_allclose(
        np.asarray(terms)[0], np.float64(expected), rtol=_relative_tolerance()
    )


def test_a_node_that_cannot_occur_contributes_nothing_against_an_infinity():
    """A represented-zero weight annihilates its value however the scale falls.

    The scale is applied around the product rather than to the weight alone, so
    the null event has to survive that reordering: `0 * -inf` is the NaN the
    weighted term exists to prevent, and it would take the well-specified node
    beside it down too.
    """
    dtype = jnp.zeros(()).dtype

    terms = scaled_weighted_terms(
        coefficients=jnp.asarray([1.0, 0.0], dtype=dtype),
        shifts=jnp.zeros(2, dtype=jnp.int32),
        values=jnp.asarray([2.0, -jnp.inf], dtype=dtype),
    )

    np.testing.assert_allclose(np.asarray(terms), np.asarray([2.0, 0.0]))
