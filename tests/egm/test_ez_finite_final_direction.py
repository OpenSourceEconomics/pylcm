"""Analytic CE directions whose final scale fits but coefficient scale does not."""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ez_power_mean import continuation_power_mean
from _lcm.power_mean import weighted_power_mean
from lcm.typing import ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _literal_direction(
    *,
    certainty: float,
    payoffs: list[float],
    probability: float,
    tiny: float,
    largest: float,
    direction: float,
    coefficient_range: str,
) -> tuple[Decimal, float]:
    with localcontext() as context:
        context.prec = 160
        p = Decimal(probability)
        delta = Decimal(payoffs[1]).ln() - Decimal(payoffs[0]).ln()
        coefficient = Decimal(certainty) * p * (1 - p) * delta**2 / 2
        if coefficient_range == "subnormal":
            assert 0 < coefficient < Decimal(tiny)
        else:
            assert coefficient > Decimal(largest)
        final = coefficient * Decimal(direction)
        assert Decimal(tiny) <= final <= Decimal(largest)
        return coefficient, float(final)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("coefficient_range", ["subnormal", "overflow"])
def test_original_ez_normal_final_direction_keeps_unmaterialized_coefficient(
    *, coefficient_range: str
) -> None:
    """The original CE's variance coefficient is scaled only with its direction."""
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    if coefficient_range == "subnormal":
        payoffs = [float(info.tiny), 2.0 * float(info.tiny)]
        probabilities = [0.5, 0.5]
        direction = math.ldexp(1.0, 16)
    else:
        # Exact binary probabilities keep a rare low payoff beside a huge one.
        # The CE itself is normal; CE*Var(log V)/2 exceeds the dtype maximum.
        high_exponent = int(info.maxexp) - 1
        denominator = 1 << (high_exponent.bit_length())
        low_probability = 1.0 / denominator
        payoffs = [1.0, math.ldexp(1.0, high_exponent)]
        probabilities = [low_probability, 1.0 - low_probability]
        direction = math.ldexp(1.0, -16)
    values = jnp.asarray(payoffs, dtype=dtype)
    weights = jnp.asarray(probabilities, dtype=dtype)
    shifts = jnp.zeros_like(values, dtype=jnp.int32)
    exponent = jnp.asarray(0.0, dtype=dtype)
    tangent = jnp.asarray(direction, dtype=dtype)
    original = weighted_power_mean(
        values=values, weights=weights, exponent=exponent, shifts=shifts
    )
    assert isinstance(original, jax.Array)
    assert float(info.tiny) <= float(original) <= float(info.max)
    assert float(info.tiny) <= direction <= float(info.max)

    # The unchanged represented primal fixes the CE scale. Decimal evaluates
    # the independent binary-lottery variance and final product without ever
    # rounding the out-of-range coefficient to the tested floating dtype.
    coefficient, expected = _literal_direction(
        certainty=float(original),
        payoffs=payoffs,
        probability=probabilities[0],
        tiny=float(info.tiny),
        largest=float(info.max),
        direction=direction,
        coefficient_range=coefficient_range,
    )

    def evaluate(parameter: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=values, weights=weights, exponent=parameter, shifts=shifts
        )

    for apply in (evaluate, jax.jit(evaluate)):
        actual, actual_direction = jax.jvp(apply, (exponent,), (tangent,))
        assert isinstance(actual, jax.Array)
        assert isinstance(actual_direction, jax.Array)
        assert (actual.shape, actual.dtype, actual.weak_type) == (
            original.shape,
            original.dtype,
            original.weak_type,
        )
        assert np.asarray(actual).tobytes() == np.asarray(original).tobytes()
        assert (actual_direction.shape, actual_direction.dtype) == ((), original.dtype)
        assert bool(jnp.isfinite(actual_direction))
        assert float(actual_direction) >= float(info.tiny)
        # Normalize before the suite precision comparison so a tiny expected
        # direction cannot pass through an absolute-error tolerance at zero.
        np.testing.assert_almost_equal(
            float(actual_direction) / expected, 1.0, decimal=DECIMAL_PRECISION
        )
        if coefficient_range == "subnormal":
            # Same primal with unit direction must preserve the final subnormal,
            # rather than turn a legitimate derivative into a flushed zero.
            _, retained = jax.jvp(apply, (exponent,), (jnp.ones_like(exponent),))
            with localcontext() as context:
                context.prec = 160
                ratio = float(Decimal(float(retained)) / coefficient)
            np.testing.assert_almost_equal(ratio, 1.0, decimal=DECIMAL_PRECISION)
            assert float(retained) != 0.0
