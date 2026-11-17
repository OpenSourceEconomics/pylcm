"""A scaled weighted average prices nodes the format holds only once scaled.

`zero_safe_average` takes each node's weight as a significand with its own
base-two scale, so the node's probability is `coefficient * 2**-shift` and its
contribution is `coefficient * 2**-shift * value`. The three factors commute as
real numbers but not as floating-point intermediates: forming
`coefficient * value` first asks the format for the very product the downscale
exists to avoid, and a coefficient of order one meeting a value near the top of
the range overflows to infinity, which no later scaling recovers.

Each node is therefore formed from normal significands and its exponent is
applied only relative to the largest term of its sum, so a node is lost only
when it is immaterial against that term. These tests pin that at both ends of
the range, against exact rational references.
"""

from fractions import Fraction
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from _lcm.probability import is_live
from _lcm.regime_building.zero_safe import zero_safe_average
from _lcm.zero_safe import _largest_selected
from lcm.certainty_equivalent import LinearExpectation
from lcm.typing import FloatND, IntND
from tests.conftest import DECIMAL_PRECISION


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("route", ["average", "linear"])
def test_scaled_mean_keeps_zero_on_its_mass_scale(
    *, mode: Literal["eager", "jit", "vmap"], route: Literal["average", "linear"]
) -> None:
    """Positive mass with zero values stays zero under equivalent weight encodings."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    tiny = float(np.finfo(dtype).tiny)
    smallest = float(np.nextafter(dtype.type(0.0), dtype.type(1.0)))
    host_weights = np.asarray(
        [
            [tiny / 4, tiny / 4],
            [smallest, smallest],
            [tiny / 4, tiny / 2],
            [tiny / 2, tiny / 4],
            [tiny / 4, 0.0],
            [tiny, tiny],
            [tiny / 4, tiny / 4],
            [tiny / 4, tiny / 4],
        ],
        dtype=dtype,
    )
    values = jnp.asarray(
        [[0.0, 0.0], [-0.0, -0.0], [0.0, -0.0], [-0.0, 0.0]] * 2,
        dtype=dtype,
    )
    weights = jnp.asarray(host_weights)
    shifts = jnp.asarray([[0, 0]] * 5 + [[2, 2], [-7, -7], [7, 7]], dtype=jnp.int32)
    np.testing.assert_array_equal(is_live(weights), host_weights > 0)

    def average(*, values: FloatND, weights: FloatND, shifts: IntND) -> FloatND:
        if route == "linear":
            return LinearExpectation().aggregate_scaled(
                values=values, coefficients=weights, shifts=shifts, params={}
            )
        return zero_safe_average(a=values, weights=weights, shifts=shifts, axis=-1)

    evaluate = average
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
    actual = evaluate(values=values, weights=weights, shifts=shifts)
    expected = [
        _scaled_mean_reference(
            values=tuple(float(v) for v in row),
            weights=tuple(float(w) for w in mass),
            shifts=tuple(int(s) for s in scale),
        )[0]
        for row, mass, scale in zip(
            np.asarray(values), host_weights, np.asarray(shifts), strict=True
        )
    ]
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))


@pytest.mark.parametrize("enable_jit", [False, True])
def test_selected_exponents_keep_the_full_integer_range(*, enable_jit: bool) -> None:
    """A real minimum exponent is distinct from a row with no selected entry."""
    exponents = jnp.asarray(
        [[-(2**30) - 1, 0], [-(2**30), 0], [np.iinfo(np.int32).min, 0], [1, 2]],
        dtype=jnp.int32,
    )
    selected = jnp.asarray([[True, False]] * 3 + [[False, False]])
    evaluate = jax.jit(_largest_selected) if enable_jit else _largest_selected
    actual = evaluate(selected=selected, exponents=exponents)
    expected = np.asarray(
        [[-(2**30) - 1], [-(2**30)], [np.iinfo(np.int32).min], [0]], dtype=np.int32
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_scaled_mean_preserves_a_constant_at_the_finite_limit(
    *, mode: str, sign: int
) -> None:
    """Unequal normal weights retain a constant at either finite endpoint."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    maximum = float(np.finfo(dtype).max)
    values = jnp.asarray([sign * maximum, sign * maximum], dtype=dtype)
    coefficients = jnp.asarray([1.0, 1.0 + np.finfo(dtype).eps], dtype=dtype)

    def average(*, values: FloatND, coefficients: FloatND) -> FloatND:
        return zero_safe_average(
            a=values, weights=coefficients, shifts=jnp.zeros(2, dtype=jnp.int32)
        )

    evaluate = average
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        values = jnp.stack([values] * 2)
        coefficients = jnp.stack([coefficients] * 2)
    actual = np.asarray(evaluate(values=values, coefficients=coefficients))
    expected, _, _ = _scaled_mean_reference(
        values=(sign * maximum, sign * maximum),
        weights=(1.0, float(coefficients.reshape(-1)[1])),
        shifts=(0, 0),
    )
    np.testing.assert_array_equal(actual, np.full(actual.shape, expected, dtype=dtype))


def _scaled_mean_reference(
    *, values: tuple[float, ...], weights: tuple[float, ...], shifts: tuple[int, ...]
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Evaluate the mean and its derivatives with exact represented-input rationals."""
    factors = tuple(Fraction(2) ** -shift for shift in shifts)
    probabilities = tuple(
        Fraction(weight) * factor
        for weight, factor in zip(weights, factors, strict=True)
    )
    mass = sum(probabilities, Fraction())
    mean = (
        sum(
            (p * Fraction(v) for p, v in zip(probabilities, values, strict=True)),
            Fraction(),
        )
        / mass
    )
    weight_slopes = tuple(
        float(factor * (Fraction(value) - mean) / mass)
        for factor, value in zip(factors, values, strict=True)
    )
    return float(mean), weight_slopes, tuple(float(p / mass) for p in probabilities)


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("multiple", [1.0, 1.25, 1.5, 1.75, 2.0])
def test_rare_large_value_keeps_a_normal_small_mean(
    *, mode: str, multiple: float
) -> None:
    """A negligible large tail does not downscale the dominant normal contribution."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    tiny = float(np.finfo(dtype).tiny)
    values = (multiple * tiny, float(np.finfo(dtype).max))
    weights = (1.0, tiny)
    shifts = (
        0,
        3 * (800 if dtype.itemsize == 8 else 100) + int(np.finfo(dtype).minexp),
    )
    expected, _, _ = _scaled_mean_reference(
        values=values, weights=weights, shifts=shifts
    )

    def average(*, values: FloatND, coefficients: FloatND) -> FloatND:
        return zero_safe_average(
            a=values, weights=coefficients, shifts=jnp.asarray(shifts, dtype=jnp.int32)
        )

    evaluate = average
    value_array, weight_array = jnp.asarray(values), jnp.asarray(weights)
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value_array = jnp.stack([value_array] * 2)
        weight_array = jnp.stack([weight_array] * 2)
    actual = evaluate(values=value_array, coefficients=weight_array)
    assert_allclose(actual, expected, rtol=10.0**-DECIMAL_PRECISION, atol=0)


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_scaled_average_keeps_a_finite_mean_when_its_numerator_overflows(
    *, axis: int | None, sign: float, enable_jit: bool
) -> None:
    """Large same-sign values have a finite mean, and dead infinities drop out."""
    peak = jnp.finfo(jnp.asarray(1.0).dtype).max
    values = jnp.asarray([sign * peak, sign * (peak / 2), -jnp.inf])
    expected = sign * (peak * 0.75)
    if axis is not None:
        values = jnp.stack([values, values / 2], axis=1 if axis == 0 else 0)
        expected = jnp.asarray([expected, expected / 2])
    average = (
        jax.jit(zero_safe_average, static_argnames="axis")
        if enable_jit
        else zero_safe_average
    )

    result = average(
        a=values,
        weights=jnp.asarray([1.0, 1.0, 0.0]),
        shifts=jnp.zeros(3, dtype=jnp.int32),
        axis=axis,
    )

    assert_allclose(result, expected, rtol=10.0**-DECIMAL_PRECISION)


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
