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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from _lcm.certainty_equivalent import _scaled_lottery_terms_with_scale
from _lcm.regime_building.zero_safe import zero_safe_average
from lcm.typing import FloatND, IntND
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


def _get_tiny_level_derivative_specimen(
    *, case: str
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Construct normal live values and exact normalized derivative references."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    tiny = float(np.finfo(dtype).tiny)
    level = float(np.ldexp(dtype.type(tiny), np.finfo(dtype).nmant - 2))
    upper = float(np.nextafter(dtype.type(level), dtype.type(np.inf)))
    values, coefficients, shifts = (level, upper), (tiny, tiny), (0, 0)
    if case in {"unequal", "permuted"}:
        coefficients = (tiny, 2 * tiny)
    if case == "negative":
        values = (-level, -upper)
    elif case == "permuted":
        values, coefficients = values[::-1], coefficients[::-1]
    elif case == "shifted":
        shifts = (7, 7)
    elif case == "dead_finite":
        values, coefficients, shifts = (
            (*values, 0.0),
            (*coefficients, 0.0),
            (*shifts, 0),
        )
    # Only finite values enter the independent rational oracle. The null-event
    # nonfinite coefficient partial follows the kernel's explicit zero convention.
    _, expected_coefficients, expected_values = _scaled_mean_reference(
        values=values, weights=coefficients, shifts=shifts
    )
    if case == "dead_nonfinite":
        values, coefficients, shifts = (
            (*values, -np.inf),
            (*coefficients, 0.0),
            (*shifts, 0),
        )
        expected_coefficients = (*expected_coefficients, 0.0)
        expected_values = (*expected_values, 0.0)
    return values, coefficients, shifts, expected_coefficients, expected_values


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
        "tiny_coefficient_adjacent",
        "tiny_coefficient_unequal",
        "tiny_coefficient_cancelled",
        "tiny_coefficient_multinode",
        "tiny_coefficient_rare_large",
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
    tiny_level = 1.0
    tiny_upper = float(np.nextafter(dtype.type(tiny_level), dtype.type(np.inf)))
    tiny_third = float(np.nextafter(dtype.type(tiny_upper), dtype.type(np.inf)))
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
        "tiny_coefficient_adjacent": ((tiny_level, tiny_upper), (tiny, tiny), (0, 0)),
        "tiny_coefficient_unequal": (
            (tiny_level, tiny_upper),
            (tiny, 2 * tiny),
            (0, 0),
        ),
        "tiny_coefficient_cancelled": (
            (tiny_level, tiny_upper, tiny_third),
            (tiny, 2 * tiny, tiny),
            (0, 0, 0),
        ),
        "tiny_coefficient_multinode": (
            (tiny_level, tiny_upper, tiny_third),
            (tiny, 2 * tiny, 3 * tiny),
            (0, 0, 0),
        ),
        "tiny_coefficient_rare_large": (
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
    expected_mean, expected_weights, expected_values = _scaled_mean_reference(
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
    assert np.isfinite(np.asarray(value_array)).all()
    assert not np.array_equal(value_array[0], value_array[1])
    assert_allclose(
        zero_safe_average(a=value_array, weights=weight_array, shifts=scales),
        expected_mean,
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0,
    )
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
    # The exact probability can lie below the active format's range.
    expected_value_array = np.asarray(expected_values, dtype=value_array.dtype)
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


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
@pytest.mark.parametrize(
    "case",
    [
        "equal",
        "unequal",
        "negative",
        "permuted",
        "shifted",
        "dead_finite",
        "dead_nonfinite",
    ],
)
def test_scaled_mean_retains_partials_below_its_numerator_range(
    *, mode: str, case: str
) -> None:
    """Local finite partials retain value spacing despite a tiny raw numerator."""
    values, coefficients, shifts, expected_coefficients, expected_values = (
        _get_tiny_level_derivative_specimen(case=case)
    )
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    scales = jnp.asarray(shifts, dtype=jnp.int32)

    def derivatives(
        *, values: FloatND, coefficients: FloatND
    ) -> tuple[FloatND, FloatND, FloatND]:
        def average(weight: FloatND) -> FloatND:
            return zero_safe_average(a=values, weights=weight, shifts=scales)

        def value_average(value: FloatND) -> FloatND:
            return zero_safe_average(a=value, weights=coefficients, shifts=scales)

        return (
            jax.grad(average)(coefficients),
            jax.jacfwd(average)(coefficients),
            jax.grad(value_average)(values),
        )

    value_array, coefficient_array = jnp.asarray(values), jnp.asarray(coefficients)
    assert np.isfinite(np.asarray(value_array[:2])).all()
    assert not np.array_equal(value_array[0], value_array[1])
    evaluate = derivatives
    expected_coefficient_array = np.asarray(expected_coefficients, dtype=dtype)
    expected_value_array = np.asarray(expected_values, dtype=dtype)
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value_array = jnp.stack([value_array, value_array])
        coefficient_array = jnp.stack([coefficient_array, coefficient_array])
        expected_coefficient_array = np.stack([expected_coefficient_array] * 2)
        expected_value_array = np.stack([expected_value_array] * 2)
    reverse, forward, value_partials = evaluate(
        values=value_array, coefficients=coefficient_array
    )
    for actual in (reverse, forward):
        assert_allclose(
            actual, expected_coefficient_array, rtol=10.0**-DECIMAL_PRECISION, atol=0
        )
    assert_allclose(
        value_partials, expected_value_array, rtol=10.0**-DECIMAL_PRECISION, atol=0
    )


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("coefficient_factor", [0.5, 1.0, 2.0, 1048576.0])
@pytest.mark.parametrize(
    "case",
    [
        "equal",
        "unequal",
        "negative",
        "permuted",
        "shifted",
        "dead_finite",
        "dead_nonfinite",
    ],
)
def test_scaled_mean_preserves_small_values_and_partials(
    *, mode: str, coefficient_factor: float, case: str
) -> None:
    """Coefficient magnitude cannot erase a normal normalized mean."""
    values, coefficients, shifts, _, _ = _get_tiny_level_derivative_specimen(case=case)
    coefficients = tuple(c * coefficient_factor for c in coefficients)
    finite_nodes = tuple(i for i, value in enumerate(values) if np.isfinite(value))
    expected_mean, expected_coefficients, expected_values = _scaled_mean_reference(
        values=tuple(values[i] for i in finite_nodes),
        weights=tuple(coefficients[i] for i in finite_nodes),
        shifts=tuple(shifts[i] for i in finite_nodes),
    )
    if case == "dead_nonfinite":
        expected_coefficients = (*expected_coefficients, 0.0)
        expected_values = (*expected_values, 0.0)
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    scales = jnp.asarray(shifts, dtype=jnp.int32)

    def mean_and_derivatives(
        *, values: FloatND, coefficients: FloatND
    ) -> tuple[FloatND, FloatND, FloatND, FloatND]:
        def average(weight: FloatND) -> FloatND:
            return zero_safe_average(a=values, weights=weight, shifts=scales)

        def value_average(value: FloatND) -> FloatND:
            return zero_safe_average(a=value, weights=coefficients, shifts=scales)

        return (
            average(coefficients),
            jax.grad(average)(coefficients),
            jax.jacfwd(average)(coefficients),
            jax.grad(value_average)(values),
        )

    operands = {
        "values": jnp.asarray(values),
        "coefficients": jnp.asarray(coefficients),
    }
    expected = (
        np.asarray(expected_mean, dtype=dtype),
        np.asarray(expected_coefficients, dtype=dtype),
        np.asarray(expected_coefficients, dtype=dtype),
        np.asarray(expected_values, dtype=dtype),
    )
    assert abs(expected_mean) >= np.finfo(dtype).tiny
    evaluate = mean_and_derivatives
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        operands = {name: jnp.stack([value, value]) for name, value in operands.items()}
        expected = tuple(np.stack([value, value]) for value in expected)
    for actual, reference in zip(evaluate(**operands), expected, strict=True):
        assert_allclose(actual, reference, rtol=10.0**-DECIMAL_PRECISION, atol=0)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("common_shift", [-(2**31), 2**31 - 1])
def test_small_scaled_mean_ignores_common_shift_gauge(
    *, enable_jit: bool, common_shift: int
) -> None:
    """Common endpoint offsets cancel before coefficient exponent arithmetic."""
    values, coefficients, _, _, _ = _get_tiny_level_derivative_specimen(case="unequal")
    expected, slopes, _ = _scaled_mean_reference(
        values=values, weights=coefficients, shifts=(0, 0)
    )
    shifts = jnp.full(2, common_shift, dtype=jnp.int32)
    value_array = jnp.asarray(values)

    def average(weight: FloatND) -> FloatND:
        return zero_safe_average(a=value_array, weights=weight, shifts=shifts)

    evaluate = jax.value_and_grad(average)
    if enable_jit:
        evaluate = jax.jit(evaluate)
    mean, actual_slopes = evaluate(jnp.asarray(coefficients))
    assert_allclose(mean, expected, rtol=10.0**-DECIMAL_PRECISION, atol=0)
    assert_allclose(actual_slopes, slopes, rtol=10.0**-DECIMAL_PRECISION, atol=0)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("dead_shift", [-(2**31), 2**31 - 1])
def test_zero_weight_extreme_shift_keeps_a_zero_finite_right_partial(
    *, enable_jit: bool, dead_shift: int
) -> None:
    """A finite null node equal to the mean has exactly zero weight slope."""
    dtype = jnp.asarray(1.0).dtype
    tiny = jnp.finfo(dtype).tiny
    level = jnp.asarray(tiny * 2.0 ** (jnp.finfo(dtype).nmant - 2), dtype=dtype)
    values = jnp.full(3, level)
    shifts = jnp.asarray([0, 0, dead_shift], dtype=jnp.int32)

    def average(weight: FloatND) -> FloatND:
        return zero_safe_average(a=values, weights=weight, shifts=shifts)

    evaluate = jax.value_and_grad(average)
    if enable_jit:
        evaluate = jax.jit(evaluate)
    mean, slopes = evaluate(jnp.asarray([tiny, tiny, 0.0], dtype=dtype))
    assert_allclose(mean, level, rtol=10.0**-DECIMAL_PRECISION, atol=0)
    assert_allclose(slopes, np.zeros(3), rtol=0, atol=0)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("coefficient_factor", [0.5, 1.0, 2.0, 1048576.0])
def test_normalized_lottery_terms_keep_the_physical_returned_scale(
    *, enable_jit: bool, coefficient_factor: float
) -> None:
    """The helper scale reconstructs the original represented-input measure."""

    values, coefficients, _, _, _ = _get_tiny_level_derivative_specimen(case="unequal")
    coefficients = tuple(c * coefficient_factor for c in coefficients)
    shifts = (0, 3)
    evaluate = _scaled_lottery_terms_with_scale
    if enable_jit:
        evaluate = jax.jit(evaluate)
    terms, masses, common_scale = evaluate(
        values=jnp.asarray(values),
        coefficients=jnp.asarray(coefficients),
        shifts=jnp.asarray(shifts, dtype=jnp.int32),
    )
    scale = int(np.asarray(common_scale).item())
    expected_masses = tuple(
        Fraction(c) * Fraction(2) ** (scale - shift)
        for c, shift in zip(coefficients, shifts, strict=True)
    )
    expected_terms = tuple(
        mass * Fraction(v) for mass, v in zip(expected_masses, values, strict=True)
    )
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    assert_allclose(
        masses,
        np.asarray(tuple(map(float, expected_masses)), dtype=dtype),
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0,
    )
    assert_allclose(
        terms,
        np.asarray(tuple(map(float, expected_terms)), dtype=dtype),
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0,
    )


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


def _get_floor_level_lotteries() -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[int, int]], ...
]:
    """Construct two-node lotteries whose values sit at the bottom of the normal range.

    Every value, every live coefficient and every mean is normal, and no lottery
    cancels. The coefficients are unequal ordinary weights, probability rows,
    equivalent scaled encodings and coefficients near the smallest normal, each
    with its sign, permutation and common-shift variants.
    """
    tiny = float(np.finfo(jnp.asarray(1.0).dtype).tiny)
    value_pairs = (
        (1.5 * tiny, 1.5 * tiny),
        (1.75 * tiny, 1.75 * tiny),
        (2 * tiny, 1.5 * tiny),
    )
    weightings = (
        ((1.0, 0.5), (0, 0)),
        ((1.0, 0.25), (0, 0)),
        ((0.75, 0.25), (0, 0)),
        ((0.75, 0.5), (0, 1)),
        ((1.0, 1.0), (0, 1)),
        ((2 * tiny, tiny), (0, 0)),
        ((4 * tiny, tiny), (0, 0)),
        ((4 * tiny, 4 * tiny), (1, 2)),
    )
    lotteries = []
    for values in value_pairs:
        for coefficients, shifts in weightings:
            lotteries.extend(
                [
                    (values, coefficients, shifts),
                    ((-values[0], -values[1]), coefficients, shifts),
                    (values[::-1], coefficients[::-1], shifts[::-1]),
                    (values, coefficients, (shifts[0] + 7, shifts[1] + 7)),
                    (values, coefficients, (shifts[0] - 7, shifts[1] - 7)),
                ]
            )
    return tuple(lotteries)


@pytest.mark.parametrize("quantity", ["mean", "value_partials", "weight_partials"])
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_scaled_mean_keeps_floor_level_nodes_against_unequal_weights(
    *, mode: str, quantity: str
) -> None:
    """A normal mean of floor-level values keeps every node's weighted contribution."""
    lotteries = _get_floor_level_lotteries()
    values = jnp.asarray([lottery[0] for lottery in lotteries])
    coefficients = jnp.asarray([lottery[1] for lottery in lotteries])
    shifts = jnp.asarray([lottery[2] for lottery in lotteries], dtype=jnp.int32)

    def mean_and_partials(
        *, values: FloatND, coefficients: FloatND, shifts: IntND
    ) -> tuple[FloatND, FloatND, FloatND]:
        def value_average(value: FloatND) -> FloatND:
            return zero_safe_average(
                a=value, weights=coefficients, shifts=shifts, axis=-1
            )

        def weight_average(weight: FloatND) -> FloatND:
            return zero_safe_average(a=values, weights=weight, shifts=shifts, axis=-1)

        if values.ndim == 1:
            return (
                value_average(values),
                jax.jacfwd(value_average)(values),
                jax.grad(weight_average)(coefficients),
            )
        # Rows are independent lotteries, so each row's partials are the
        # diagonal blocks of the batched Jacobian.
        return (
            value_average(values),
            jnp.einsum("iij->ij", jax.jacfwd(value_average)(values)),
            jnp.einsum("iij->ij", jax.jacfwd(weight_average)(coefficients)),
        )

    evaluate = mean_and_partials
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
    results = evaluate(values=values, coefficients=coefficients, shifts=shifts)
    index = ("mean", "value_partials", "weight_partials").index(quantity)
    actual = np.asarray(results[index])

    tolerance = Fraction(10) ** -DECIMAL_PRECISION
    # A partial below the normal range is not representable on a flushing
    # backend, so it may come back as zero; every other quantity is relative.
    tiny = Fraction(float(np.finfo(actual.dtype).tiny))
    worst = Fraction()
    for row, (v, c, s) in zip(actual, lotteries, strict=True):
        mean, weight_partials, value_partials = _scaled_mean_reference(
            values=tuple(float(x) for x in np.asarray(v, dtype=actual.dtype)),
            weights=tuple(float(x) for x in np.asarray(c, dtype=actual.dtype)),
            shifts=s,
        )
        expected = {
            "mean": (mean,),
            "value_partials": value_partials,
            "weight_partials": weight_partials,
        }[quantity]
        for observed, exact in zip(np.atleast_1d(row), expected, strict=True):
            error = abs(Fraction(float(observed)) - Fraction(exact))
            if abs(Fraction(exact)) < tiny:
                error = max(error - tiny, Fraction())
            worst = max(worst, error / abs(Fraction(exact)) if exact else error)
    assert worst <= tolerance
