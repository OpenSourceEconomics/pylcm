"""Original EZ primal and independently derived first-order unit-risk chain."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ez_kernel import (
    ez_blend_partials,
    ez_continuation,
    ez_invert_partials,
    ez_transform_partials,
)
from _lcm.egm.ez_power_mean import continuation_power_mean
from _lcm.power_mean import weighted_power_mean
from lcm.typing import FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("exponent", [0.0, 0.25, -0.5])
def test_original_ez_power_mean_primal_and_nonunit_tangent(*, exponent: float) -> None:
    """The original represented primal stays exact, including batched lotteries."""
    values = jnp.asarray([[3.0, 5.0], [7.0, 11.0]])
    weights = jnp.asarray([0.25, 0.75])
    shifts = jnp.asarray([0, 1], dtype=jnp.int32)
    p = jnp.asarray(exponent)
    old = weighted_power_mean(values=values, weights=weights, exponent=p, shifts=shifts)
    new = continuation_power_mean(
        values=values, weights=weights, exponent=p, shifts=shifts
    )
    assert (new.shape, new.dtype, new.weak_type) == (
        old.shape,
        old.dtype,
        old.weak_type,
    )
    assert np.asarray(new).tobytes() == np.asarray(old).tobytes()
    if exponent != 0.0:

        def original(parameter: ScalarFloat) -> FloatND:
            return weighted_power_mean(
                values=values, weights=weights, exponent=parameter, shifts=shifts
            )

        def corrected(parameter: ScalarFloat) -> FloatND:
            return continuation_power_mean(
                values=values, weights=weights, exponent=parameter, shifts=shifts
            )

        tangent = jnp.ones_like(p)
        _, before = jax.jvp(original, (p,), (tangent,))
        _, after = jax.jvp(corrected, (p,), (tangent,))
        np.testing.assert_array_equal(after, before)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_ez_unit_risk_has_analytic_forward_and_reverse_direction() -> None:
    values = jnp.asarray([3.0, 5.0])
    weights = jnp.asarray([0.25, 0.75])

    def evaluate(risk: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=values,
            weights=weights,
            exponent=1.0 - risk,
            shifts=jnp.zeros_like(values, dtype=jnp.int32),
        )

    risk = jnp.asarray(1.0)
    expected = 3.0**0.25 * 5.0**0.75
    variance = 0.25 * 0.75 * math.log(5.0 / 3.0) ** 2
    expected_direction = -expected * variance / 2.0
    for apply in (evaluate, jax.jit(evaluate)):
        value, direction = jax.jvp(apply, (risk,), (jnp.ones_like(risk),))
        np.testing.assert_almost_equal(value, expected, decimal=DECIMAL_PRECISION)
        np.testing.assert_almost_equal(
            direction, expected_direction, decimal=DECIMAL_PRECISION
        )
        np.testing.assert_almost_equal(
            jax.grad(apply)(risk), expected_direction, decimal=DECIMAL_PRECISION
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_ez_split_and_joint_lotteries_keep_total_unit_risk_variance() -> None:
    """Within-block plus between-block terms recover the original whole lottery."""
    payoffs = np.asarray([2.0, 3.0, 5.0, 8.0])
    probabilities = np.asarray([0.125, 0.125, 0.25, 0.5])
    values = jnp.asarray(payoffs)
    weights = jnp.asarray(probabilities)
    marginals = jnp.ones_like(values)
    log_mean = float(probabilities @ np.log(payoffs))
    nu = math.exp(log_mean)
    variance = float(probabilities @ (np.log(payoffs) - log_mean) ** 2)
    nu_gamma = -nu * variance / 2.0
    nu_s = nu * float(probabilities @ (1.0 / payoffs))
    nu_s_gamma = nu_s * (log_mean + nu_gamma / nu) - nu * float(
        probabilities @ (np.log(payoffs) / payoffs)
    )

    def joint(risk: ScalarFloat) -> FloatND:
        return jnp.stack(
            ez_continuation(
                child_values=values,
                child_marginals=marginals,
                weights=weights,
                risk_aversion=risk,
            )
        )

    def split(risk: ScalarFloat) -> FloatND:
        parts = tuple(
            ez_transform_partials(
                child_values=values[start : start + 2],
                child_marginals=marginals[start : start + 2],
                weights=weights[start : start + 2],
                risk_aversion=risk,
            )
            for start in (0, 2)
        )
        blended = ez_blend_partials(
            certainty_equivalents=jnp.stack([part[0] for part in parts]),
            weight_sums=jnp.stack([part[1] for part in parts]),
            marginal_log_scales=jnp.stack([part[2] for part in parts]),
            marginal_mantissas=jnp.stack([part[3] for part in parts]),
            probs=jnp.ones((2,), dtype=values.dtype),
            risk_aversion=risk,
        )
        return jnp.stack(
            ez_invert_partials(
                certainty_equivalent=blended[0],
                weight_sum=blended[1],
                marginal_log_scale=blended[2],
                marginal_mantissa=blended[3],
                risk_aversion=risk,
            )
        )

    risk = jnp.asarray(1.0)
    for apply in (joint, split):
        value, direction = jax.jvp(apply, (risk,), (jnp.ones_like(risk),))
        np.testing.assert_array_almost_equal(
            value, [nu, nu_s], decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_almost_equal(
            direction, [nu_gamma, nu_s_gamma], decimal=DECIMAL_PRECISION
        )
        np.testing.assert_almost_equal(
            jax.grad(lambda parameter, evaluate=apply: evaluate(parameter)[0])(risk),
            nu_gamma,
            decimal=DECIMAL_PRECISION,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "kind", ["zero_mass", "negative_weight", "nonpositive_payoff", "tiny_coefficient"]
)
def test_original_ez_unit_risk_invalid_or_unretained_direction_refuses(
    *, kind: str
) -> None:
    values = jnp.asarray([3.0, 5.0])
    weights = jnp.asarray([0.5, 0.5])
    if kind == "zero_mass":
        weights = jnp.zeros_like(weights)
    elif kind == "negative_weight":
        weights = jnp.asarray([-0.5, 1.5])
    elif kind == "nonpositive_payoff":
        values = jnp.asarray([0.0, 5.0])
    else:
        values = jnp.asarray([2.0, 4.0]) * jnp.finfo(values.dtype).tiny

    def evaluate(exponent: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=values,
            weights=weights,
            exponent=exponent,
            shifts=jnp.zeros_like(values, dtype=jnp.int32),
        )

    exponent = jnp.asarray(0.0)
    _, direction = jax.jvp(evaluate, (exponent,), (jnp.ones_like(exponent),))
    assert not bool(jnp.isfinite(direction))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_ez_unit_risk_broadcast_weights_batched_forward_and_reverse() -> None:
    """Original broadcast-weight rows keep their own variance and CE scale."""
    payoffs = np.asarray([[3.0, 5.0], [7.0, 11.0]])
    probabilities = np.asarray([0.25, 0.75])
    values = jnp.asarray(payoffs)
    weights = jnp.asarray(probabilities)
    expected = np.exp(np.log(payoffs) @ probabilities)
    variance = (
        probabilities[0] * probabilities[1] * np.log(payoffs[:, 1] / payoffs[:, 0]) ** 2
    )
    expected_direction = -expected * variance / 2.0

    def evaluate(risk: ScalarFloat) -> FloatND:
        return continuation_power_mean(
            values=values,
            weights=weights,
            exponent=1.0 - risk,
            shifts=jnp.zeros((2,), dtype=jnp.int32),
        )

    def objective(risk: ScalarFloat) -> ScalarFloat:
        return jnp.sum(evaluate(risk))

    risk = jnp.asarray(1.0)
    for apply in (evaluate, jax.jit(evaluate)):
        value, direction = jax.jvp(apply, (risk,), (jnp.ones_like(risk),))
        np.testing.assert_array_almost_equal(value, expected, decimal=DECIMAL_PRECISION)
        np.testing.assert_array_almost_equal(
            direction, expected_direction, decimal=DECIMAL_PRECISION
        )
    np.testing.assert_almost_equal(
        jax.grad(objective)(risk), expected_direction.sum(), decimal=DECIMAL_PRECISION
    )
