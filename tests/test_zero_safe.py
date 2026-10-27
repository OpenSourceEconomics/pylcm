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

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from _lcm.regime_building.zero_safe import zero_safe_average
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION


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
@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "positive",
        "negative",
        "opposite",
        "dominant",
        "permuted",
        "shifted",
        "midpoint",
        "unequal",
        "adjacent_negative",
        "adjacent_permuted",
        "half_scale",
        "sixteenth_scale",
        "equivalent_pairs",
        "common_shift",
        "three_nodes",
        "five_nodes",
        "tiny_adjacent",
        "tiny_adjacent_rare_large",
    ],
)
def test_scaled_mean_normalizes_finite_slopes_before_tangent_arithmetic(
    *, mode: str, case: str
) -> None:
    """Finite normalized derivatives survive extreme levels and opposite-sign gaps."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    peak = float(np.finfo(dtype).max)
    lower = float(np.nextafter(dtype.type(peak), dtype.type(0)))
    third = float(np.nextafter(dtype.type(lower), dtype.type(0)))
    tiny = float(np.finfo(dtype).tiny)
    tiny_level = tiny * 2.0 ** (np.finfo(dtype).nmant - 2)
    tiny_upper = float(np.nextafter(dtype.type(tiny_level), dtype.type(np.inf)))
    rare_shift = int(2 * np.finfo(dtype).maxexp + 2)
    specimens = {
        "zero": ((peak, peak / 2), (1.0, 0.0), (0, 0)),
        "positive": ((peak, peak / 2), (1.0, 0.5), (0, 0)),
        "negative": ((-peak, -peak / 2), (1.0, 0.0), (0, 0)),
        "opposite": ((peak, -peak), (1.0, 2.0), (0, 0)),
        "dominant": ((peak, -peak), (3.0, 1.0), (0, 0)),
        "permuted": ((peak / 2, peak), (0.0, 1.0), (0, 0)),
        "shifted": ((peak, peak / 2), (1.0, 0.0), (7, 7)),
        "midpoint": ((peak, lower), (1.0, 1.0), (1, 1)),
        "unequal": ((peak, lower), (1.0, 3.0), (2, 2)),
        "adjacent_negative": ((-peak, -lower), (1.0, 1.0), (1, 1)),
        "adjacent_permuted": ((lower, peak), (3.0, 1.0), (2, 2)),
        "half_scale": ((peak / 2, lower / 2), (1.0, 1.0), (1, 1)),
        "sixteenth_scale": ((peak / 16, lower / 16), (1.0, 3.0), (2, 2)),
        "equivalent_pairs": ((peak, lower), (2.0, 2.0), (2, 2)),
        "common_shift": ((peak, lower), (1.0, 3.0), (9, 9)),
        "three_nodes": ((peak, lower, third), (1.0, 3.0, 2.0), (1, 1, 1)),
        "tiny_adjacent": ((tiny_level, tiny_upper), (tiny, tiny), (0, 0)),
        "tiny_adjacent_rare_large": (
            (tiny_level, tiny_upper, peak),
            (tiny, tiny, tiny),
            (0, 0, rare_shift),
        ),
        "five_nodes": (
            (peak, lower, third, lower, peak),
            (1.0, 3.0, 2.0, 4.0, 2.0),
            (1, 1, 1, 1, 1),
        ),
    }
    values, weights, shifts = specimens[case]
    _, expected_weights, expected_values = _scaled_mean_reference(
        values=values, weights=weights, shifts=shifts
    )
    scales = jnp.asarray(shifts, dtype=jnp.int32)

    def derivatives(
        *, values: FloatND, coefficients: FloatND
    ) -> tuple[FloatND, FloatND, FloatND]:
        def weight_function(weight: FloatND) -> FloatND:
            return zero_safe_average(a=values, weights=weight, shifts=scales)

        def value_function(value: FloatND) -> FloatND:
            return zero_safe_average(a=value, weights=coefficients, shifts=scales)

        return (
            jax.grad(weight_function)(coefficients),
            jax.jacfwd(weight_function)(coefficients),
            jax.grad(value_function)(values),
        )

    evaluate = derivatives
    value_array, weight_array = jnp.asarray(values), jnp.asarray(weights)
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value_array = jnp.stack([value_array, value_array])
        weight_array = jnp.stack([weight_array, weight_array])
    actual_weights, actual_forward, actual_values = evaluate(
        values=value_array, coefficients=weight_array
    )
    expected_weight_array = np.asarray(expected_weights)
    expected_value_array = np.asarray(expected_values)
    if mode == "vmap":
        expected_weight_array = np.stack([expected_weight_array] * 2)
        expected_value_array = np.stack([expected_value_array] * 2)
    for actual in (actual_weights, actual_forward):
        assert_allclose(
            actual,
            expected_weight_array,
            rtol=10.0**-DECIMAL_PRECISION,
            atol=10.0**-DECIMAL_PRECISION,
        )
    assert_allclose(actual_values, expected_value_array, rtol=10.0**-DECIMAL_PRECISION)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_unperturbed_coefficients_do_not_poison_finite_value_derivatives(
    *,
    enable_jit: bool,
) -> None:
    """Value-only AD ignores an unperturbed, out-of-format coefficient partial."""
    peak = jnp.finfo(jnp.asarray(1.0).dtype).max
    values = jnp.asarray([peak, -peak])
    weights = jnp.asarray([1.0, 0.0])
    shifts = jnp.zeros(2, dtype=jnp.int32)

    def average(value: FloatND) -> FloatND:
        return zero_safe_average(a=value, weights=weights, shifts=shifts)

    def forward(value: FloatND) -> FloatND:
        return jax.jvp(average, (value,), (jnp.ones_like(value),))[1]

    gradient = jax.grad(average)
    evaluate_forward = forward
    if enable_jit:
        gradient, evaluate_forward = jax.jit(gradient), jax.jit(evaluate_forward)
    assert_allclose(gradient(values), np.asarray([1.0, 0.0]), rtol=0, atol=0)
    assert_allclose(evaluate_forward(values), 1.0, rtol=0, atol=0)


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


@pytest.mark.parametrize("weights", [(0.0, 1.0), (1.0, 2.0)])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_scaled_average_preserves_weight_and_value_derivatives(
    *, weights: tuple[float, float], enable_jit: bool
) -> None:
    """Finite nodes retain their weight derivative even at zero probability."""
    values = jnp.asarray([2.0, 4.0])
    coefficients = jnp.asarray(weights)
    shifts = jnp.zeros(2, dtype=jnp.int32)
    weight_gradient = jax.grad(
        lambda weight: zero_safe_average(a=values, weights=weight, shifts=shifts)
    )
    value_gradient = jax.grad(
        lambda value: zero_safe_average(a=value, weights=coefficients, shifts=shifts)
    )
    if enable_jit:
        weight_gradient = jax.jit(weight_gradient)
        value_gradient = jax.jit(value_gradient)
    total = jnp.sum(coefficients)
    expected = jnp.sum(coefficients * values) / total

    assert_allclose(
        weight_gradient(coefficients),
        (values - expected) / total,
        rtol=10.0**-DECIMAL_PRECISION,
        atol=10.0**-DECIMAL_PRECISION,
    )
    assert_allclose(
        value_gradient(values), coefficients / total, rtol=10.0**-DECIMAL_PRECISION
    )


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
