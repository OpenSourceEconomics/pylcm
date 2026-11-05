"""Independent IEEE scaling oracle for actual signed-log CE coefficients."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_power_math import _materialize, _SignedLog
from _lcm.probability import scaled_by_power_of_two
from lcm.typing import FloatND, IntND


def _read_and_parts(value: _SignedLog) -> tuple[FloatND, FloatND, IntND]:
    """Expose the unchanged floating decomposition, not its binary encoding.

    math.ldexp below tests encoding independently. This deliberately separates
    represented exponential coefficients from any backend exp/log ULP claim.
    """
    info = jnp.finfo(value.logabs.dtype)
    ln_two = jnp.log(jnp.asarray(2.0, dtype=value.logabs.dtype))
    bounded = jnp.clip(
        value.logabs,
        (info.minexp - info.nmant - 2) * ln_two,
        (info.maxexp + 1) * ln_two,
    )
    bounded = jnp.where(jnp.isfinite(bounded), bounded, 0.0)
    exponent = jnp.floor(bounded / ln_two).astype(jnp.int32)
    coefficient = value.sign * jnp.exp(bounded - exponent * ln_two)
    return _materialize(value=value), coefficient, exponent


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
def test_signed_log_materializer_matches_independent_ldexp_near_ieee_boundaries(
    *, compiled: bool
) -> None:
    """Normal/subnormal/zero/overflow values follow the actual dtype's IEEE map."""
    dtype = jnp.asarray(1.0).dtype
    info = jnp.finfo(dtype)
    shifts = (
        0,
        -1,
        info.minexp,
        info.minexp - 1,
        info.minexp - info.nmant,
        info.minexp - info.nmant - 1,
        info.minexp - info.nmant - 2,
        info.maxexp - 1,
        info.maxexp,
        info.maxexp + 1,
    )
    coefficients = jnp.asarray([1.0, 1.5, 2.0 - info.eps], dtype=dtype)
    exponents = jnp.repeat(jnp.asarray(shifts, dtype=jnp.int32), coefficients.size)
    logabs = exponents.astype(dtype) * jnp.log(jnp.asarray(2.0, dtype=dtype))
    logabs += jnp.tile(jnp.log(coefficients), len(shifts))
    record = _SignedLog(
        jnp.tile(logabs, 2),
        jnp.repeat(jnp.asarray([1.0, -1.0], dtype=dtype), logabs.size),
    )
    read = jax.jit(_read_and_parts) if compiled else _read_and_parts
    actual, coefficient, exponent = read(record)
    host_coefficient = np.asarray(coefficient)
    host_exponent = np.asarray(exponent)
    reference = []
    for mantissa, power in zip(host_coefficient, host_exponent, strict=True):
        try:
            reference.append(math.ldexp(float(mantissa), int(power)))
        except OverflowError:
            reference.append(math.copysign(math.inf, float(mantissa)))
    with np.errstate(over="ignore", under="ignore"):
        expected = np.asarray(reference, dtype=dtype)
    assert actual.dtype == coefficient.dtype == dtype
    assert actual.shape == expected.shape
    assert np.any((np.abs(expected) > 0.0) & (np.abs(expected) < info.tiny))
    assert np.any(np.isinf(expected))
    np.testing.assert_array_equal(actual, expected)
    normal = np.isfinite(expected) & (np.abs(expected) >= info.tiny)
    old_normal = np.asarray(
        scaled_by_power_of_two(
            values=jnp.asarray(host_coefficient[normal]),
            shift=jnp.asarray(host_exponent[normal]),
        )
    )
    assert np.asarray(actual)[normal].tobytes() == old_normal.tobytes()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_signed_log_materializer_retains_zero_and_invalid_semantics() -> None:
    logabs = jnp.asarray([0.0, jnp.inf, -jnp.inf, jnp.nan, jnp.nan])
    sign = jnp.asarray([0.0, -1.0, 1.0, 1.0, 0.0])
    actual = _materialize(value=_SignedLog(logabs, sign))
    np.testing.assert_array_equal(actual[:3], [0.0, -jnp.inf, 0.0])
    assert bool(jnp.all(jnp.isnan(actual[3:])))
