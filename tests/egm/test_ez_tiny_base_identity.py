"""Actual continuation CE must preserve its original tiny value direction."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ez_power_mean import _unit_exponent_data, continuation_power_mean
from _lcm.power_mean import weighted_power_mean
from lcm.typing import FloatND


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_ez_zero_exponent_direction_preserves_original_tiny_value_jvp() -> (
    None
):
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    rare = float(info.tiny)
    incoming = rare / 4.0
    values = jnp.asarray([2.0, 1.0], dtype=dtype)
    weights = jnp.asarray([1.0, rare], dtype=dtype)
    shifts = jnp.zeros_like(values, dtype=jnp.int32)
    exponent = jnp.asarray(0.0, dtype=dtype)
    directions = jnp.asarray([incoming, 0.0], dtype=dtype)

    def reference(lottery: FloatND) -> FloatND:
        return weighted_power_mean(
            values=lottery, weights=weights, exponent=exponent, shifts=shifts
        )

    def continuation(lottery: FloatND) -> FloatND:
        return continuation_power_mean(
            values=lottery, weights=weights, exponent=exponent, shifts=shifts
        )

    # Independent geometric-mean value partial and exponent coefficient.
    # Extended Decimal precision retains the normal but extremely rare mass.
    with localcontext() as context:
        context.prec = 400
        p = Decimal(1) / (Decimal(1) + Decimal(rare))
        log_two = Decimal(2).ln()
        certainty = (p * log_two).exp()
        value_direction = certainty * p / 2 * Decimal(incoming)
        coefficient = certainty * p * (1 - p) * log_two**2 / 2
        assert 0 < coefficient < Decimal(rare)
        assert 0 < value_direction < Decimal(rare)

    for compiled in (False, True):
        original = jax.jit(reference) if compiled else reference
        wrapped = jax.jit(continuation) if compiled else continuation
        # Establish the actual original JVP before checking the continuation.
        # No manufactured base or substituted power-mean rule is admitted.
        primal, original_direction = jax.jvp(original, (values,), (directions,))
        assert isinstance(primal, jax.Array)
        assert isinstance(original_direction, jax.Array)
        original_host = np.asarray(original_direction)
        assert float(original_host) != 0.0
        with localcontext() as context:
            context.prec = 400
            assert abs(Decimal(float(original_host)) - value_direction) <= Decimal(
                float(info.smallest_subnormal)
            )

        represented_coefficient, logarithm, sign = _unit_exponent_data(
            values=values, weights=weights, exponent=exponent, shifts=shifts
        )
        assert not bool(jnp.isfinite(represented_coefficient))
        assert bool(jnp.isfinite(logarithm))
        assert float(sign) == 1.0

        actual, actual_direction = jax.jvp(wrapped, (values,), (directions,))
        assert isinstance(actual, jax.Array)
        assert isinstance(actual_direction, jax.Array)
        assert (actual.shape, actual.dtype, actual.weak_type) == (
            primal.shape,
            primal.dtype,
            primal.weak_type,
        )
        assert actual_direction.shape == original_direction.shape
        assert actual_direction.dtype == original_direction.dtype
        assert np.asarray(actual).tobytes() == np.asarray(primal).tobytes()
        assert np.asarray(actual_direction).tobytes() == original_host.tobytes()
