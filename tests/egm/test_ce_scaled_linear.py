"""Independent finite-final IEEE transport and its explicit first-order transpose."""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.egm.ce_scaled_linear import (
    _coefficient_parts,
    _transpose,
    join_ce_direction,
    join_represented_ce_direction,
    scale_ce_direction,
)
from lcm.typing import Float1D, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("kind", ["normal", "tiny_final", "tiny_input", "large_scale"])
def test_ce_scaled_transport_has_independent_final_scale(*, kind: str) -> None:
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    if kind == "normal":
        coefficient_exponent, operand = -3, 24.0
    elif kind == "tiny_final":
        coefficient_exponent, operand = int(info.minexp) - 2, 1.0
    elif kind == "tiny_input":
        coefficient_exponent = 16
        operand = float(info.smallest_subnormal)
    else:
        coefficient_exponent, operand = int(info.maxexp) + 8, math.ldexp(1.0, -16)
    logarithm = jnp.asarray(coefficient_exponent * math.log(2.0), dtype=dtype)
    direction = jnp.asarray(operand, dtype=dtype)
    polarity = jnp.ones_like(logarithm)
    mantissa, exponent, _ = _coefficient_parts(logabs=logarithm)
    # The represented binary coefficient fixes the approximation being
    # transported, separately from CE/variance and exp/log approximation errors.
    with localcontext() as context:
        context.prec = 160
        expected_decimal = (
            Decimal(float(mantissa)) * Decimal(2) ** int(exponent) * Decimal(operand)
        )
        expected = np.asarray(float(expected_decimal), dtype=dtype)

    def apply(value: FloatND) -> FloatND:
        return scale_ce_direction(logabs=logarithm, sign=polarity, direction=value)

    for evaluate in (apply, jax.jit(apply)):
        actual = evaluate(direction)
        transposed = jax.linear_transpose(evaluate, direction)(direction)[0]
        # Forward/transpose compilation may round transport arithmetic
        # differently. Both must satisfy the same independent scale contract.
        for output in (actual, transposed):
            assert isinstance(output, jax.Array)
            assert output.shape == direction.shape
            assert output.dtype == direction.dtype
            actual_host = np.asarray(output)
            if 0.0 < abs(float(expected)) < float(info.tiny):
                # One subnormal ULP covers significand-product rounding followed
                # by the final RN-even scale, not exact single rounding.
                with localcontext() as context:
                    context.prec = 160
                    error = abs(Decimal(float(actual_host)) - Decimal(float(expected)))
                    assert error <= Decimal(float(info.smallest_subnormal))
            else:
                np.testing.assert_almost_equal(
                    float(actual_host) / float(expected), 1.0, decimal=DECIMAL_PRECISION
                )
            assert float(actual_host) != 0.0
            assert np.signbit(actual_host) == np.signbit(expected)
        opposite = evaluate(-direction)
        assert np.asarray(opposite).tobytes() == (-np.asarray(actual)).tobytes()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "kind", ["overflow", "underflow", "nonfinite", "invalid_scale"]
)
def test_ce_scaled_transport_refuses_invalid_or_unretained_final(*, kind: str) -> None:
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    exponent = (
        int(info.maxexp) + 8
        if kind == "overflow"
        else int(info.minexp) - info.nmant - 8
    )
    logarithm = jnp.asarray(exponent * math.log(2.0), dtype=dtype)
    direction = jnp.asarray(jnp.inf if kind == "nonfinite" else 1.0, dtype=dtype)
    sign = jnp.asarray(-1.0 if kind == "invalid_scale" else 1.0, dtype=dtype)
    for evaluate in (scale_ce_direction, jax.jit(scale_ce_direction)):
        actual = evaluate(logabs=logarithm, sign=sign, direction=direction)
        assert not bool(jnp.isfinite(actual))
        if kind != "invalid_scale":
            zero = evaluate(
                logabs=logarithm, sign=sign, direction=jnp.zeros_like(direction)
            )
            assert (
                np.asarray(zero).tobytes()
                == np.asarray(jnp.zeros_like(direction)).tobytes()
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ce_scaled_transport_batches_and_reduces_scalar_cotangent() -> None:
    logarithms = jnp.log(jnp.asarray([0.5, 2.0, 4.0]))
    signs = jnp.ones_like(logarithms)
    operands = jnp.asarray([2.0, -3.0, 0.25])

    # keyword-only-exempt: library-callback=jax.vmap
    def scalar(logarithm: ScalarFloat, operand: ScalarFloat) -> ScalarFloat:
        return scale_ce_direction(
            logabs=logarithm, sign=jnp.ones_like(logarithm), direction=operand
        )

    expected = jnp.asarray([1.0, -6.0, 1.0])
    for apply in (jax.vmap(scalar), jax.jit(jax.vmap(scalar))):
        np.testing.assert_array_almost_equal(
            apply(logarithms, operands), expected, decimal=DECIMAL_PRECISION
        )

    # The original scalar exponent is broadcast to rows; its reverse must sum
    # row cotangents, rather than return a vector with a plausible shape.
    def summed(parameter: ScalarFloat) -> ScalarFloat:
        return jnp.sum(
            scale_ce_direction(logabs=logarithms, sign=signs, direction=parameter)
        )

    np.testing.assert_almost_equal(
        jax.grad(summed)(jnp.asarray(2.0)), 6.5, decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ce_scaled_transport_differentiates_its_coefficient_to_second_order() -> None:
    one = jnp.asarray(1.0)
    direction = jnp.asarray(1.5)

    def logarithmic(logarithm: ScalarFloat) -> ScalarFloat:
        return scale_ce_direction(logabs=logarithm, sign=one, direction=direction)

    def represented(coefficient: ScalarFloat) -> ScalarFloat:
        return join_represented_ce_direction(
            coefficient=coefficient, sign=one, base=one, direction=direction
        )

    logarithm, coefficient = jnp.asarray(0.25), jnp.asarray(2.0)
    # d/dL (1.5 e^L) = 1.5 e^L at every order; d/dR (1 + 1.5 R) = 1.5, then 0.
    expected = 1.5 * math.exp(0.25)
    np.testing.assert_almost_equal(
        jax.grad(logarithmic)(logarithm), expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        jax.grad(jax.grad(logarithmic))(logarithm), expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        jax.grad(represented)(coefficient), 1.5, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        jax.grad(jax.grad(represented))(coefficient), 0.0, decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ce_scaled_join_preserves_tiny_base_and_true_linear_transpose() -> None:
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    tiny = jnp.asarray(float(info.tiny) / 4.0, dtype=dtype)
    zero = jnp.zeros_like(tiny)
    logarithm = jnp.log(jnp.asarray(2.0, dtype=dtype))
    one = jnp.ones_like(tiny)

    # keyword-only-exempt: library-callback=jax.linear_transpose
    def apply(base: ScalarFloat, direction: ScalarFloat) -> ScalarFloat:
        return join_ce_direction(
            logabs=logarithm, sign=one, base=base, direction=direction
        )

    for evaluate in (apply, jax.jit(apply)):
        assert np.asarray(evaluate(tiny, zero)).tobytes() == np.asarray(tiny).tobytes()
        expected = np.asarray(float(info.tiny) / 2.0, dtype=dtype)
        assert np.asarray(evaluate(zero, tiny)).tobytes() == expected.tobytes()
        actual = evaluate(jnp.asarray(3.0), jnp.asarray(4.0))
        np.testing.assert_almost_equal(actual, 11.0, decimal=DECIMAL_PRECISION)
        base_ct, direction_ct = jax.linear_transpose(evaluate, one, one)(one)
        np.testing.assert_array_almost_equal(
            jnp.stack((base_ct, direction_ct)), [1.0, 2.0], decimal=DECIMAL_PRECISION
        )
    refused = join_ce_direction(
        logabs=jnp.asarray(-80.0),
        sign=one,
        base=jnp.asarray(-1.0),
        direction=one,
    )
    assert not bool(jnp.isfinite(refused))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("invalid", ["coefficient", "cotangent"])
def test_ce_scaled_join_base_transpose_refuses_invalid_domain(*, invalid: str) -> None:
    one = jnp.asarray(1.0)
    logarithm = jnp.asarray(jnp.nan if invalid == "coefficient" else 0.0)
    cotangent = jnp.asarray(jnp.inf if invalid == "cotangent" else 1.0)

    def apply(base: ScalarFloat) -> ScalarFloat:
        return join_ce_direction(
            logabs=logarithm, sign=one, base=base, direction=jnp.zeros_like(base)
        )

    for evaluate in (apply, jax.jit(apply)):
        actual = jax.linear_transpose(evaluate, one)(cotangent)[0]
        assert not bool(jnp.isfinite(actual))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ce_scaled_join_zero_coefficient_base_transpose_preserves_tiny_bits() -> None:
    dtype = jnp.asarray(0.0).dtype
    tiny = jnp.asarray(float(np.finfo(dtype).tiny) / 4.0, dtype=dtype)
    zero = jnp.zeros_like(tiny)
    logarithm = jnp.asarray(-jnp.inf, dtype=dtype)

    def apply(base: ScalarFloat) -> ScalarFloat:
        return join_ce_direction(logabs=logarithm, sign=zero, base=base, direction=zero)

    for evaluate in (apply, jax.jit(apply)):
        actual = jax.linear_transpose(evaluate, jnp.ones_like(tiny))(tiny)[0]
        assert np.asarray(actual).tobytes() == np.asarray(tiny).tobytes()
        np.testing.assert_almost_equal(
            jax.grad(evaluate)(jnp.ones_like(tiny)), 1.0, decimal=DECIMAL_PRECISION
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "kind",
    [
        "ordinary",
        "tiny_base",
        "tiny_input",
        "tiny_coefficient",
        "underflow",
        "overflow",
        "invalid",
    ],
)
def test_represented_ce_coefficient_keeps_ordinary_and_exceptional_directions(
    *, kind: str
) -> None:
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    cases = {
        "ordinary": (1.0, 2.0**-70, 0.5),
        "tiny_base": (2.0, float(info.tiny) / 4.0, 0.0),
        "tiny_input": (8.0, 0.0, float(info.smallest_subnormal)),
        "tiny_coefficient": (float(info.tiny) / 4.0, 0.0, 4.0),
        "underflow": (0.125, 0.0, float(info.smallest_subnormal)),
        "overflow": (8.0, 0.0, float(info.max)),
        "invalid": (-1.0, 0.0, 1.0),
    }
    coefficient, original, operand = cases[kind]
    factor = jnp.asarray(coefficient, dtype=dtype)
    base = jnp.asarray(original, dtype=dtype)
    direction = jnp.asarray(operand, dtype=dtype)

    # keyword-only-exempt: library-callback=jax.linear_transpose
    def apply(original: ScalarFloat, operand: ScalarFloat) -> ScalarFloat:
        return join_represented_ce_direction(
            coefficient=factor,
            sign=jnp.ones_like(factor),
            base=original,
            direction=operand,
        )

    with localcontext() as context:
        context.prec = 160
        expected_decimal = Decimal(original) + Decimal(coefficient) * Decimal(operand)
        expected = float(expected_decimal)
    for evaluate in (apply, jax.jit(apply)):
        actual = evaluate(base, direction)
        if kind in ("underflow", "overflow", "invalid"):
            assert not bool(jnp.isfinite(actual))
            continue
        assert bool(jnp.isfinite(actual))
        if kind == "tiny_base":
            assert np.asarray(actual).tobytes() == np.asarray(base).tobytes()
        elif kind == "tiny_input":
            with localcontext() as context:
                context.prec = 160
                assert abs(Decimal(float(actual)) - expected_decimal) <= Decimal(
                    float(info.smallest_subnormal)
                )
        else:
            np.testing.assert_allclose(
                float(actual) / expected,
                1.0,
                rtol=0,
                atol=max(1e-4, 8 * math.sqrt(info.eps)),
            )
        base_ct, direction_ct = jax.linear_transpose(evaluate, base, direction)(
            jnp.ones_like(base)
        )
        assert float(base_ct) == 1.0
        if kind == "tiny_coefficient":
            np.testing.assert_allclose(
                float(direction_ct) / coefficient,
                1.0,
                rtol=0,
                atol=max(1e-4, 8 * math.sqrt(info.eps)),
            )
        else:
            np.testing.assert_allclose(
                float(direction_ct), coefficient, rtol=64 * info.eps, atol=0
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("kind", ["log", "represented"])
@pytest.mark.parametrize("compiled", [False, True])
def test_ce_transport_mixed_second_derivatives_and_differentiated_transpose(
    *, kind: str, compiled: bool
) -> None:
    """Coefficient, base and direction perturbations retain both mixed terms."""
    point = jnp.asarray([0.25 if kind == "log" else 2.0, 1.0, 1.5])
    tangent = jnp.asarray([0.2, 0.3, 0.4])
    other = jnp.asarray([0.5, 0.6, 0.7])
    coefficient = jnp.exp(point[0]) if kind == "log" else point[0]

    def evaluate(point: Float1D) -> ScalarFloat:
        factor, base, direction = point
        if kind == "log":
            return join_ce_direction(
                logabs=factor, sign=jnp.asarray(1.0), base=base, direction=direction
            )
        return join_represented_ce_direction(
            coefficient=factor,
            sign=jnp.asarray(1.0),
            base=base,
            direction=direction,
        )

    def first(point: Float1D) -> ScalarFloat:
        return jax.jvp(evaluate, (point,), (tangent,))[1]

    def second(point: Float1D) -> ScalarFloat:
        return jax.jvp(first, (point,), (other,))[1]

    expected_first = tangent[1] + coefficient * tangent[2]
    if kind == "log":
        expected_first += coefficient * point[2] * tangent[0]
        expected_second = coefficient * (
            point[2] * tangent[0] * other[0]
            + tangent[2] * other[0]
            + other[2] * tangent[0]
        )
    else:
        expected_first += point[2] * tangent[0]
        expected_second = tangent[2] * other[0] + other[2] * tangent[0]
    run_first, run_second = (
        (jax.jit(first), jax.jit(second)) if compiled else (first, second)
    )
    np.testing.assert_almost_equal(
        run_first(point), expected_first, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        run_second(point), expected_second, decimal=DECIMAL_PRECISION
    )

    def coefficient_value(factor: ScalarFloat) -> ScalarFloat:
        return evaluate(point.at[0].set(factor))

    curvature = jax.grad(jax.grad(coefficient_value))
    expected_curvature = coefficient * point[2] if kind == "log" else 0.0
    np.testing.assert_almost_equal(
        (jax.jit(curvature) if compiled else curvature)(point[0]),
        expected_curvature,
        decimal=DECIMAL_PRECISION,
    )

    def transpose(parameters: Float1D) -> Float1D:
        factor, cotangent = parameters

        def linear(payload: Float1D) -> ScalarFloat:
            if kind == "log":
                return join_ce_direction(
                    logabs=factor,
                    sign=jnp.asarray(1.0),
                    base=payload[0],
                    direction=payload[1],
                )
            return join_represented_ce_direction(
                coefficient=factor,
                sign=jnp.asarray(1.0),
                base=payload[0],
                direction=payload[1],
            )

        return jax.linear_transpose(linear, jnp.zeros(2))(cotangent)[0]

    derivative = jax.jacfwd(transpose)
    expected = jnp.asarray(
        [
            [0.0, 1.0],
            [coefficient * 0.8 if kind == "log" else 0.8, coefficient],
        ]
    )
    np.testing.assert_array_almost_equal(
        (jax.jit(derivative) if compiled else derivative)(jnp.asarray([point[0], 0.8])),
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("forged", [False, True])
def test_ce_transpose_refuses_unauthenticated_scalar_metadata(*, forged: bool) -> None:
    """Only the actual JAX typed scalar may substitute for a callback array."""
    scalar_class = (
        type("TypedFloat", (float,), {"__module__": "jax._src.literals"})
        if forged
        else float
    )
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter sign"):
        _transpose(
            cotangent=jnp.asarray(1.0),
            logabs=jnp.asarray(0.0),
            sign=scalar_class(1.0),  # ty: ignore[invalid-argument-type]
            base=jnp.asarray(0.0),
            direction=jnp.asarray(1.0),
        )
