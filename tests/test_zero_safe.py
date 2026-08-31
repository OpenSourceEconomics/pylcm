"""A scaled weighted average prices nodes the format holds only once scaled.

`zero_safe_average` takes each node's weight as a significand with its own
base-two scale, so the node's probability is `coefficient * 2**-shift` and its
contribution is `coefficient * 2**-shift * value`. The three factors commute as
real numbers but not as floating-point intermediates: forming
`coefficient * value` first asks the format for the very product the downscale
exists to avoid, and a coefficient of order one meeting a value near the top of
the range overflows to infinity, which no later scaling recovers.

The downscale is therefore split — as much of it onto the coefficient as the
coefficient has room for while staying normal, the remainder onto the product —
which is what `scaled_weighted_terms` does and what these tests pin.
"""

import jax.numpy as jnp
from numpy.testing import assert_allclose

from _lcm.regime_building.zero_safe import zero_safe_average
from tests.conftest import DECIMAL_PRECISION


def test_zero_safe_average_prices_a_downscaled_node_against_a_near_max_value():
    """A node whose scale keeps it in range is priced, not sent to infinity.

    Both nodes here carry probability `2**-8`, so the mean is half the larger
    of the two values — an ordinary number, even though the larger value sits
    one binade below the top of the format.
    """
    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    values = jnp.asarray([largest, 0.0])
    coefficients = jnp.asarray([4.0, 1.0])
    shifts = jnp.asarray([10, 8], dtype=jnp.int32)

    # The specimen is in the overflow regime: the product of the two operands
    # as they arrive is not representable, so an answer that comes back finite
    # can only have formed the contribution some other way.
    assert not jnp.isfinite(coefficients[0] * values[0])

    result = zero_safe_average(a=values, weights=coefficients, shifts=shifts)

    assert_allclose(result, largest / 2, rtol=10.0**-DECIMAL_PRECISION)


def test_zero_safe_average_prices_a_downscaled_node_along_a_reduced_axis():
    """The same holds when the average reduces one axis of a larger array.

    Each row carries the two nodes at equal probability, so each row's mean is
    half its own larger value.
    """
    largest = jnp.finfo(jnp.asarray(1.0).dtype).max
    values = jnp.asarray([[largest, 0.0], [0.0, largest / 4]])
    coefficients = jnp.asarray([4.0, 1.0])
    shifts = jnp.asarray([10, 8], dtype=jnp.int32)

    assert not jnp.isfinite(coefficients[0] * values[0, 0])

    result = zero_safe_average(a=values, axis=1, weights=coefficients, shifts=shifts)

    assert_allclose(
        result,
        jnp.asarray([largest / 2, largest / 8]),
        rtol=10.0**-DECIMAL_PRECISION,
    )
