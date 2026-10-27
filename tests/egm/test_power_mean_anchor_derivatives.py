"""Differentiate smooth power means independently of numerical anchor ties."""

import math
from collections.abc import Sequence
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ez_power_mean import continuation_power_mean
from _lcm.power_mean import weighted_power_mean, weighted_power_mean_of_pair
from lcm.koopmans_aggregation import CESAggregator
from lcm.typing import FloatND, ScalarFloat


def equal_payoff_direction(
    *, weights: Sequence[float], directions: Sequence[float]
) -> float:
    """Return the defining economic derivative on the positive equal-payoff ray."""
    if len(weights) != len(directions) or not weights:
        raise ValueError("The oracle requires one direction per lottery node.")
    with localcontext() as context:
        context.prec = 80
        w = tuple(Decimal(float(value)) for value in weights)
        d = tuple(Decimal(float(value)) for value in directions)
        if any(value < 0 for value in w) or sum(w, start=Decimal(0)) <= 0:
            raise ValueError("The oracle requires nonnegative, positive-total mass.")
        return float(
            sum(
                (weight * direction for weight, direction in zip(w, d, strict=True)),
                start=Decimal(0),
            )
            / sum(w, start=Decimal(0))
        )


def power_mean_value_direction(
    *,
    values: Sequence[float],
    weights: Sequence[float],
    exponent: float,
    value_directions: Sequence[float],
    weight_directions: Sequence[float] | None = None,
    exponent_direction: float = 0.0,
) -> tuple[float, float]:
    """Differentiate the defining normalized moment with 100-digit arithmetic.

    Inputs are the exact real values of supplied binary floats. Values must be
    positive; weights must be nonnegative with positive total mass. A direction
    at a zero weight must be interpreted on an admissible one-sided path.
    This scalar oracle uses no production anchoring, signed logs or AD rules.
    """
    if not len(values) == len(weights) == len(value_directions) or not values:
        raise ValueError("Expected a nonempty common event roster.")
    if weight_directions is None:
        weight_directions = [0.0] * len(weights)
    if len(weight_directions) != len(weights):
        raise ValueError("Expected one weight direction per event.")

    def D(number: float) -> Decimal:
        return Decimal(float(number))

    with localcontext() as context:
        context.prec = 100
        v, w = (tuple(map(D, values)), tuple(map(D, weights)))
        dv, dw = (tuple(map(D, value_directions)), tuple(map(D, weight_directions)))
        p, dp = (D(exponent), D(exponent_direction))
        z, dz = (sum(w, start=Decimal(0)), sum(dw, start=Decimal(0)))
        if any(x <= 0 for x in v) or any(x < 0 for x in w) or z <= 0:
            raise ValueError(
                "Positive payoffs and a valid positive-total lottery required."
            )
        logs = tuple(x.ln() for x in v)
        if p == 0:
            mu = (
                sum((a * ell for a, ell in zip(w, logs, strict=True)), start=Decimal(0))
                / z
            )
            value = mu.exp()
            variance = (
                sum(
                    (a * (ell - mu) ** 2 for a, ell in zip(w, logs, strict=True)),
                    start=Decimal(0),
                )
                / z
            )
            child = (
                sum(
                    (
                        a * direction / payoff
                        for a, direction, payoff in zip(w, dv, v, strict=True)
                    ),
                    start=Decimal(0),
                )
                / z
            )
            law = (
                sum(
                    (
                        direction * (ell - mu)
                        for direction, ell in zip(dw, logs, strict=True)
                    ),
                    start=Decimal(0),
                )
                / z
            )
            direction = value * (child + law + dp * variance / 2)
        else:
            powers = tuple((p * ell).exp() for ell in logs)
            n = sum(
                (a * power for a, power in zip(w, powers, strict=True)),
                start=Decimal(0),
            )
            log_average = (n / z).ln()
            value = (log_average / p).exp()
            child_and_law = sum(
                (
                    power * (weight_dot + weight * p * value_dot / payoff)
                    for power, weight_dot, weight, value_dot, payoff in zip(
                        powers, dw, w, dv, v, strict=True
                    )
                ),
                start=Decimal(0),
            )
            dp_channel = sum(
                (
                    weight * power * ell
                    for weight, power, ell in zip(w, powers, logs, strict=True)
                ),
                start=Decimal(0),
            )
            direction = value * (
                (child_and_law / n - dz / z) / p
                + dp * (dp_channel / (p * n) - log_average / p**2)
            )
        return (float(value), float(direction))


pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _budget(dtype: np.dtype[np.float32 | np.float64]) -> float:
    return max(1e-4, 8 * math.sqrt(np.finfo(dtype).eps))


def test_minimal_equal_payoff_continuation_witness() -> None:
    """sqrt((1+theta)*(1-theta)) has derivative zero at theta=0."""
    weights = jnp.asarray([0.5, 0.5])
    slopes = jnp.asarray([1.0, -1.0])
    zero = jnp.asarray(0.0)

    def evaluate(theta: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=1.0 + theta * slopes,
            weights=weights,
            exponent=zero,
            shifts=jnp.zeros((2,), dtype=jnp.int32),
        )

    expected = equal_payoff_direction(weights=[0.5, 0.5], directions=[1.0, -1.0])
    for apply in (evaluate, jax.jit(evaluate)):
        value, direction = jax.jvp(apply, (zero,), (jnp.ones_like(zero),))
        assert float(value) == 1.0
        np.testing.assert_allclose(
            float(direction), expected, rtol=0, atol=_budget(value.dtype)
        )
        np.testing.assert_allclose(
            float(jax.grad(apply)(zero)), expected, rtol=0, atol=_budget(value.dtype)
        )


@pytest.mark.parametrize("psi", [1.0, 2.0])
def test_public_ces_equal_payoff_stationary_point(*, psi: float) -> None:
    """A smooth current/future tradeoff has zero derivative at its optimum."""
    aggregator = CESAggregator()
    zero = jnp.asarray(0.0)

    def evaluate(theta: ScalarFloat) -> ScalarFloat:
        return aggregator(
            utility=1.0 + theta,
            CE=1.0 - theta,
            discount_factor=jnp.asarray(0.5),
            intertemporal_elasticity_of_substitution=jnp.asarray(psi),
        )

    # psi=1: sqrt(1-theta**2); psi=2: (1+sqrt(1-theta**2))/2.
    # Both are smooth, symmetric and strictly maximized at theta=0 locally.
    for apply in (evaluate, jax.jit(evaluate)):
        value, derivative = jax.jvp(apply, (zero,), (jnp.ones_like(zero),))
        assert float(value) == 1.0
        np.testing.assert_allclose(
            float(derivative), 0.0, rtol=0, atol=_budget(value.dtype)
        )
        np.testing.assert_allclose(
            float(jax.grad(apply)(zero)), 0.0, rtol=0, atol=_budget(value.dtype)
        )


@pytest.mark.parametrize("exponent", [0.0, 0.25, -0.5, 1e-8, -1e-8])
@pytest.mark.parametrize(
    ("weights", "directions", "payoff"),
    [
        ([0.25, 0.75], [1.0, -1.0], 1.0),
        ([0.75, 0.25], [-1.0, 1.0], 1.0),  # identical lottery, permuted
        ([0.125, 0.125, 0.75], [1.0, 1.0, -1.0], 1.0),  # duplicated first node
        ([0.25, 0.75], [1.0, 1.0], 8.0),  # positive common-slope control
    ],
)
def test_equal_payoff_direction_class(
    *,
    exponent: float,
    weights: Sequence[float],
    directions: Sequence[float],
    payoff: float,
) -> None:
    """Near/unit/nonunit risk, permutation, duplication, scale and control."""
    w = jnp.asarray(weights)
    slopes = jnp.asarray(directions)
    zero = jnp.asarray(0.0)
    power = jnp.asarray(exponent)
    expected = equal_payoff_direction(weights=weights, directions=directions)

    def evaluate(theta: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=payoff + theta * slopes,
            weights=w,
            exponent=power,
            shifts=jnp.zeros_like(w, dtype=jnp.int32),
        )

    value, direction = jax.jvp(evaluate, (zero,), (jnp.ones_like(zero),))
    assert float(value) == payoff
    np.testing.assert_allclose(
        float(direction), expected, rtol=0, atol=_budget(value.dtype)
    )


@pytest.mark.parametrize("kernel", ["general", "pair", "continuation"])
def test_equal_payoff_value_gradient(*, kernel: str) -> None:
    """At equality the value gradient is the probability vector, not an owner."""
    weights = jnp.asarray([0.25, 0.75])
    values = jnp.asarray([2.0, 2.0])
    exponent = jnp.asarray(-0.5)

    def evaluate(lottery: FloatND) -> ScalarFloat:
        if kernel == "pair":
            return weighted_power_mean_of_pair(
                first=lottery[0],
                second=lottery[1],
                first_weight=weights[0],
                second_weight=weights[1],
                exponent=exponent,
            )
        apply = (
            continuation_power_mean if kernel == "continuation" else weighted_power_mean
        )
        return apply(
            values=lottery,
            weights=weights,
            exponent=exponent,
            shifts=jnp.zeros((2,), dtype=jnp.int32),
        )

    for apply in (evaluate, jax.jit(evaluate)):
        actual = jax.grad(apply)(values)
        assert actual.shape == values.shape
        assert actual.dtype == values.dtype
        np.testing.assert_allclose(
            np.asarray(actual), [0.25, 0.75], rtol=0, atol=_budget(values.dtype)
        )


def test_equal_payoff_batched_rows_and_vmap() -> None:
    """The same smooth identity holds independently for each broadcast row."""
    weights = jnp.asarray([0.25, 0.75])
    values = jnp.asarray([[1.0, 1.0], [8.0, 8.0]])
    directions = jnp.asarray([[1.0, -1.0], [0.0, 2.0]])
    shifts = jnp.zeros((2,), dtype=jnp.int32)

    def row(lottery: FloatND) -> FloatND:
        return continuation_power_mean(
            values=lottery,
            weights=weights,
            exponent=jnp.asarray(0.0),
            shifts=shifts,
        )

    expected = np.asarray(
        [
            equal_payoff_direction(weights=[0.25, 0.75], directions=[1.0, -1.0]),
            equal_payoff_direction(weights=[0.25, 0.75], directions=[0.0, 2.0]),
        ]
    )
    for apply in (row, jax.vmap(row), jax.jit(jax.vmap(row))):
        primal, actual = jax.jvp(apply, (values,), (directions,))
        assert actual.shape == primal.shape == (2,)
        assert actual.dtype == values.dtype
        np.testing.assert_allclose(
            np.asarray(actual), expected, rtol=0, atol=_budget(values.dtype)
        )


@pytest.mark.parametrize("kernel", ["general", "continuation"])
@pytest.mark.parametrize(
    ("exponent", "values", "expected_value"),
    [(1.0, [1.0, 1.0, 0.5], 5.0 / 6.0), (-1.0, [0.5, 0.5, 1.0], 3.0 / 5.0)],
)
def test_tied_extremum_in_nonconstant_lottery(
    *, kernel: str, exponent: float, values: Sequence[float], expected_value: float
) -> None:
    """Tied anchor AD must agree even when the final deviation is nonzero."""
    lottery = jnp.asarray(values)
    weights = jnp.ones_like(lottery)  # normalized by the mathematical mean
    slopes = jnp.asarray([1.0, -1.0, 0.0])
    zero = jnp.asarray(0.0)

    def evaluate(theta: ScalarFloat) -> ScalarFloat:
        apply = (
            continuation_power_mean if kernel == "continuation" else weighted_power_mean
        )
        return apply(
            values=lottery + theta * slopes,
            weights=weights,
            exponent=jnp.asarray(exponent),
            shifts=jnp.zeros_like(weights, dtype=jnp.int32),
        )

    # The first two terms exchange under theta -> -theta. The normalized
    # defining moment is smooth and even here, hence its derivative is zero.
    # p=1 has constant arithmetic sum 5/2, hence mean5/6. p=-1 at theta0
    # has inverse-value sum2+2+1=5, hence harmonic mean3/5.
    for apply in (evaluate, jax.jit(evaluate)):
        value, derivative = jax.jvp(apply, (zero,), (jnp.ones_like(zero),))
        np.testing.assert_allclose(
            float(value),
            expected_value,
            rtol=0,
            atol=max(1e-6, 64 * np.finfo(value.dtype).eps),
        )
        np.testing.assert_allclose(
            float(derivative), 0.0, rtol=0, atol=_budget(value.dtype)
        )
        np.testing.assert_allclose(
            float(jax.grad(apply)(zero)), 0.0, rtol=0, atol=_budget(value.dtype)
        )


def test_unit_exponent_correction_composes_once_with_payoff_and_weight_directions() -> (
    None
):
    """The defining geometric-limit derivative includes each economic channel once."""
    values = jnp.asarray([3.0, 5.0])
    weights = jnp.asarray([0.25, 0.75])
    value_directions = jnp.asarray([1.0, -0.5])
    weight_directions = jnp.asarray([0.125, -0.125])
    zero = jnp.asarray(0.0)
    expected_value, expected_direction = power_mean_value_direction(
        values=[3.0, 5.0],
        weights=[0.25, 0.75],
        exponent=0.0,
        value_directions=[1.0, -0.5],
        weight_directions=[0.125, -0.125],
        exponent_direction=1.0,
    )

    def evaluate(parameter: ScalarFloat) -> ScalarFloat:
        return continuation_power_mean(
            values=values + parameter * value_directions,
            weights=weights + parameter * weight_directions,
            exponent=parameter,
            shifts=jnp.zeros((2,), dtype=jnp.int32),
        )

    for apply in (evaluate, jax.jit(evaluate)):
        value, direction = jax.jvp(apply, (zero,), (jnp.ones_like(zero),))
        np.testing.assert_allclose(
            float(value),
            expected_value,
            rtol=0,
            atol=max(1e-6, 64 * np.finfo(value.dtype).eps),
        )
        np.testing.assert_allclose(
            float(direction), expected_direction, rtol=0, atol=_budget(value.dtype)
        )
        np.testing.assert_allclose(
            float(jax.grad(apply)(zero)),
            expected_direction,
            rtol=0,
            atol=_budget(value.dtype),
        )
