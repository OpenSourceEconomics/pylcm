"""Economic derivative cancellation is checked before publishing a valid CE read."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import PowerMean
from _lcm.egm.ce_scaled_linear import join_ce_direction
from _lcm.egm.current_iid_power_mean import (
    PowerMeanNodeBank,
    reduce_power_mean_economic,
)
from lcm.typing import FloatND, ScalarBool, ScalarFloat

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _evaluate(
    economic: FloatND,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    values = jnp.ones_like(economic)
    bank = PowerMeanNodeBank(
        values=values,
        economic=economic,
        savings_coordinates=values,
        source_coordinates=jnp.asarray([1.0, -1.0], dtype=economic.dtype),
        coefficients=jnp.full_like(values, 0.5),
        shifts=jnp.zeros(2, dtype=jnp.int32),
        differentiated_coefficients=jnp.zeros((1, 2), dtype=economic.dtype),
        differentiated_shifts=jnp.zeros((1, 2), dtype=jnp.int32),
        needed=jnp.ones(2, dtype=jnp.bool_),
        valid=jnp.ones((), dtype=jnp.bool_),
    )
    return reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(1.0, dtype=economic.dtype)},
        parameter_direction={"risk_aversion": jnp.asarray(0.0, dtype=economic.dtype)},
        bank=bank,
    )


def _check_economic_result(
    *, operands: np.ndarray, compiled: bool, ordinary: bool
) -> None:
    pair = jnp.asarray(operands)
    assert np.asarray(pair).tobytes() == operands.tobytes()
    reference = float((Fraction(float(operands[0])) - Fraction(float(operands[1]))) / 2)
    evaluate = jax.jit(_evaluate) if compiled else _evaluate
    value, _, source, valid = jax.device_get(evaluate(pair))
    if ordinary:
        assert bool(valid), "Ordinary finite economic cancellation must stay supported."
    if bool(valid):
        assert abs(float(value) - 1.0) <= 1e-4
        assert abs(float(source) - reference) <= 1e-4 * max(1.0, abs(reference))


@pytest.mark.parametrize("compiled", [False, True])
def test_material_economic_log_collision_is_accurate_or_refused(
    *, compiled: bool
) -> None:
    """Distinct finite economic inputs cannot collapse to a valid inaccurate zero."""
    dtype = jnp.asarray(0.0).dtype
    power = 40 if dtype == jnp.float64 else 20
    decrement = 2.0**-10 if dtype == jnp.float64 else 2.0**-3
    operands = np.asarray([2.0**power, 2.0**power - decrement], dtype=dtype)
    assert operands[0] > operands[1] > 0.0
    _check_economic_result(operands=operands, compiled=compiled, ordinary=False)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("mutation", ["original", "reverse", "equality"])
@pytest.mark.parametrize("ordinary", [False, True])
def test_economic_cancellation_preserves_scale_order_and_ordinary_support(
    *, compiled: bool, scale: float, mutation: str, ordinary: bool
) -> None:
    """Original finite input arithmetic defines the economic direction."""
    dtype = jnp.asarray(0.0).dtype
    power = 53 if dtype == jnp.float64 else 24
    operands = np.asarray(
        [4.0, 3.0] if ordinary else [2.0**power, 2.0**power - 1.0], dtype=dtype
    )
    operands *= scale
    if mutation == "reverse":
        operands = operands[::-1].copy()
    elif mutation == "equality":
        operands[1] = operands[0]
    _check_economic_result(operands=operands, compiled=compiled, ordinary=ordinary)


@pytest.mark.parametrize("compiled", [False, True])
def test_finite_ce_join_retains_the_original_opposing_operands(
    *, compiled: bool
) -> None:
    """A unit coefficient joins actual finite directions before any log compression."""
    dtype = jnp.asarray(0.0).dtype
    power = 53 if dtype == jnp.float64 else 24
    base = jnp.asarray(2.0**power, dtype=dtype)
    direction = jnp.asarray(-(2.0**power - 1.0), dtype=dtype)
    assert float(base) + float(direction) == 1.0
    evaluate = jax.jit(join_ce_direction) if compiled else join_ce_direction
    result = evaluate(
        logabs=jnp.zeros_like(base),
        sign=jnp.ones_like(base),
        base=base,
        direction=direction,
    )
    assert not bool(jnp.isfinite(result)) or abs(float(result) - 1.0) <= 1e-4


def _evaluate_channels(
    *, values: FloatND, economic: FloatND, weight_direction: FloatND
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    """Use the original economic and probability operands in a linear mean."""
    ones = jnp.ones_like(values)
    bank = PowerMeanNodeBank(
        values=values,
        economic=economic,
        savings_coordinates=ones,
        source_coordinates=ones,
        coefficients=jnp.full_like(values, 0.5),
        shifts=jnp.zeros(2, dtype=jnp.int32),
        differentiated_coefficients=weight_direction[jnp.newaxis, :],
        differentiated_shifts=jnp.zeros((1, 2), dtype=jnp.int32),
        needed=jnp.ones(2, dtype=jnp.bool_),
        valid=jnp.ones((), dtype=jnp.bool_),
    )
    return reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(0.0, dtype=values.dtype)},
        parameter_direction={"risk_aversion": jnp.asarray(0.0, dtype=values.dtype)},
        bank=bank,
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_same_sign_large_directions_remain_supported(
    *, compiled: bool, empty: bool
) -> None:
    """Magnitude alone is not a reason to refuse a well-conditioned direction."""
    dtype = jnp.asarray(0.0).dtype
    magnitude = 0.0 if empty else 2.0**40
    evaluate = jax.jit(_evaluate_channels) if compiled else _evaluate_channels
    value, saving, source, valid = evaluate(
        values=jnp.ones(2, dtype=dtype),
        economic=jnp.full(2, magnitude, dtype=dtype),
        weight_direction=jnp.zeros(2, dtype=dtype),
    )
    assert bool(valid)
    assert float(value) == 1.0
    np.testing.assert_allclose((saving, source), magnitude, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("ordinary", [False, True])
def test_child_and_weight_opposition_cannot_hide_inner_uncertainty(
    *, compiled: bool, ordinary: bool
) -> None:
    """For the arithmetic mean, DC=E + A*(1-3) from actual finite inputs."""
    dtype = jnp.asarray(0.0).dtype
    power = 39 if dtype == jnp.float64 else 19
    delta = 2.0**-10 if dtype == jnp.float64 else 2.0**-3
    amplitude = 1.0 if ordinary else 2.0**power
    economic = np.asarray([2.0 * amplitude + delta] * 2, dtype=dtype)
    weights = np.asarray([amplitude, -amplitude], dtype=dtype)
    reference = float(
        Fraction(float(economic[0]))
        + Fraction(float(weights[0]))
        + 3 * Fraction(float(weights[1]))
    )
    evaluate = jax.jit(_evaluate_channels) if compiled else _evaluate_channels
    value, _, source, valid = evaluate(
        values=jnp.asarray([1.0, 3.0], dtype=dtype),
        economic=jnp.asarray(economic),
        weight_direction=jnp.asarray(weights),
    )
    if ordinary:
        assert bool(valid)
    if bool(valid):
        assert abs(float(value) - 2.0) <= 1e-4
        assert abs(float(source) - reference) <= 1e-4 * max(1.0, abs(reference))
