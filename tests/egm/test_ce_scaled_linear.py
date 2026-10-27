"""Independent finite-final IEEE transport and its explicit first-order transpose."""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ce_scaled_linear import (
    _coefficient_parts,
    join_ce_direction,
    scale_ce_direction,
)
from lcm.typing import FloatND, ScalarFloat
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
def test_ce_scaled_transport_does_not_silently_differentiate_residuals() -> None:
    one = jnp.asarray(1.0)

    def changed_residual(logarithm: ScalarFloat) -> ScalarFloat:
        return scale_ce_direction(logabs=logarithm, sign=one, direction=one)

    with pytest.raises(NotImplementedError, match="first-order residuals only"):
        jax.jvp(changed_residual, (one,), (one,))


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
