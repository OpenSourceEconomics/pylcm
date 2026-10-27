"""Economic derivative cancellation is checked before publishing a valid CE read."""

from decimal import Decimal, localcontext
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
from _lcm.probability import log_of_nonnegative
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


def _evaluate_law_direction(
    values: FloatND,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    """Differentiate only the original lottery weights at arithmetic exponent one."""
    zeros = jnp.zeros_like(values)
    bank = PowerMeanNodeBank(
        values=values,
        economic=zeros,
        savings_coordinates=zeros,
        source_coordinates=zeros,
        coefficients=jnp.full_like(values, 0.5),
        shifts=jnp.zeros(2, dtype=jnp.int32),
        differentiated_coefficients=jnp.asarray([[1.0, -1.0]], dtype=values.dtype),
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
@pytest.mark.parametrize("lottery", ["close_large", "ordinary", "equal"])
def test_original_payoff_law_contrast_is_accurate_or_refused(
    *, compiled: bool, lottery: str
) -> None:
    """Accepted law derivatives retain the contrast in the original payoffs."""
    dtype = jnp.asarray(0.0).dtype
    power = 53 if dtype == jnp.float64 else 24
    if lottery == "close_large":
        pair = [2.0**power, 2.0**power - 1.0]
    elif lottery == "ordinary":
        pair = [4.0, 3.0]
    else:
        pair = [4.0, 4.0]
    operands = np.asarray(pair, dtype=dtype)
    reference = Fraction(float(operands[0])) - Fraction(float(operands[1]))
    assert reference == (0 if lottery == "equal" else 1)
    values = jnp.asarray(operands)
    assert np.asarray(values).tobytes() == operands.tobytes()
    evaluate = jax.jit(_evaluate_law_direction) if compiled else _evaluate_law_direction
    value, saving, source, valid = jax.device_get(evaluate(values))
    if lottery != "close_large":
        assert bool(valid), "Ordinary and equal lotteries must remain supported."
    if bool(valid):
        budget = max(1e-4, 8.0 * np.sqrt(np.finfo(operands.dtype).eps))
        expected_value = float(
            (Fraction(float(operands[0])) + Fraction(float(operands[1]))) / 2
        )
        assert abs(float(value) - expected_value) <= budget * max(1.0, expected_value)
        assert abs(float(saving)) <= budget
        assert abs(float(source) - float(reference)) <= budget * max(
            1.0, abs(float(reference))
        )


def _evaluate_law_power_direction(
    *, values: FloatND, exponent: ScalarFloat
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    """Read the actual bank with only its represented probability-law direction."""
    zeros = jnp.zeros_like(values)
    bank = PowerMeanNodeBank(
        values=values,
        economic=zeros,
        savings_coordinates=zeros,
        source_coordinates=zeros,
        coefficients=jnp.full_like(values, 0.5),
        shifts=jnp.zeros(2, dtype=jnp.int32),
        differentiated_coefficients=jnp.asarray([[1.0, -1.0]], dtype=values.dtype),
        differentiated_shifts=jnp.zeros((1, 2), dtype=jnp.int32),
        needed=jnp.ones(2, dtype=jnp.bool_),
        valid=jnp.ones((), dtype=jnp.bool_),
    )
    return reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": 1.0 - exponent},
        parameter_direction={"risk_aversion": jnp.asarray(0.0, dtype=values.dtype)},
        bank=bank,
    )


def _original_payoff_law_reference(
    *, first: float, second: float, exponent: float
) -> float:
    """Return the defining normalized-law direction from original binary payoffs."""
    if exponent == 1.0:
        return float(Fraction(first) - Fraction(second))
    if exponent != 0.0:
        raise ValueError("The reference requires the geometric or arithmetic mean.")
    with localcontext() as context:
        context.prec = 100
        logs = (Decimal(first).ln(), Decimal(second).ln())
        value = ((logs[0] + logs[1]) / 2).exp()
        return float(value * (logs[0] - logs[1]))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("mutation", ["original", "reverse", "equality"])
@pytest.mark.parametrize("power", [None, 40, 45, 53])
@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_law_contrast_mutations_use_original_payoff_economics(
    *,
    compiled: bool,
    scale: float,
    mutation: str,
    power: int | None,
    exponent: float,
) -> None:
    """Ordinary/equal laws stay usable; material unresolved contrasts refuse."""
    dtype = jnp.asarray(0.0).dtype
    ordinary = power is None
    if ordinary:
        operands = np.asarray([4.0, 3.0], dtype=dtype)
    else:
        actual_power = (
            power if dtype == jnp.float64 else {40: 12, 45: 18, 53: 24}[power]
        )
        anchor = np.asarray(2.0**actual_power, dtype=dtype)
        previous = np.nextafter(anchor, np.asarray(-np.inf, dtype=dtype))
        operands = np.asarray([anchor, previous], dtype=dtype)
        assert operands[0] > operands[1] > 0.0
    operands *= scale
    if mutation == "reverse":
        operands = operands[::-1].copy()
    elif mutation == "equality":
        operands[1] = operands[0]
    assert np.isfinite(operands).all()
    assert np.all(operands > 0.0)
    values = jnp.asarray(operands)
    assert np.asarray(values).tobytes() == operands.tobytes()
    reference = _original_payoff_law_reference(
        first=float(operands[0]), second=float(operands[1]), exponent=exponent
    )
    evaluate = (
        jax.jit(_evaluate_law_power_direction)
        if compiled
        else _evaluate_law_power_direction
    )
    _, saving, source, valid = jax.device_get(
        evaluate(values=values, exponent=jnp.asarray(exponent, dtype=dtype))
    )
    if ordinary or mutation == "equality":
        assert bool(valid), "Ordinary and exact-equality laws must remain supported."
    if bool(valid):
        budget = max(1e-4, 8.0 * np.sqrt(np.finfo(dtype).eps))
        assert np.isfinite(float(saving))
        assert np.isfinite(float(source))
        assert abs(float(saving)) <= budget
        assert abs(float(source) - reference) <= budget * max(1.0, abs(reference))


@pytest.mark.coverage(backends=("cpu",), precisions="64")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mutation", ["original", "reverse", "equality"])
def test_tiny_law_uncertainty_preserves_ordinary_economic_support(
    *, compiled: bool, mutation: str
) -> None:
    """A tiny law contribution does not invalidate an ordinary unit direction."""
    dtype = jnp.asarray(0.0).dtype
    anchor = np.asarray(8.0, dtype=dtype)
    previous = np.nextafter(anchor, np.asarray(-np.inf, dtype=dtype))
    operands = np.asarray([anchor, previous], dtype=dtype)
    if mutation == "reverse":
        operands = operands[::-1].copy()
    elif mutation == "equality":
        operands[1] = operands[0]
    economic = np.ones(2, dtype=dtype)
    weights = np.asarray([2.0**-70, -(2.0**-70)], dtype=dtype)
    reference_value = (Fraction(float(operands[0])) + Fraction(float(operands[1]))) / 2
    reference_source = (
        Fraction(float(economic[0])) + Fraction(float(economic[1]))
    ) / 2 + sum(
        (
            Fraction(float(weight)) * Fraction(float(payoff))
            for weight, payoff in zip(weights, operands, strict=True)
        ),
        Fraction(0),
    )
    values = jnp.asarray(operands)
    assert np.asarray(values).tobytes() == operands.tobytes()
    represented_logs = jax.device_get(log_of_nonnegative(values))
    print(  # noqa: T201 - Record actual production-log collision operands for attribution.
        "ordinary-law-payoffs",
        operands,
        "production-logs",
        represented_logs,
        flush=True,
    )
    assert represented_logs[0].tobytes() == represented_logs[1].tobytes(), (
        "The bounded witness requires a production-log collision.",
        represented_logs,
    )
    evaluate = jax.jit(_evaluate_channels) if compiled else _evaluate_channels
    value, saving, source, valid = jax.device_get(
        evaluate(
            values=values,
            economic=jnp.asarray(economic),
            weight_direction=jnp.asarray(weights),
        )
    )
    assert bool(valid), "Ordinary finite directions must remain supported."
    budget = max(1e-4, 8.0 * np.sqrt(np.finfo(dtype).eps))
    assert abs(float(value) - float(reference_value)) <= budget * max(
        1.0, abs(float(reference_value))
    )
    assert abs(float(saving) - 1.0) <= budget
    assert abs(float(source) - float(reference_source)) <= budget * max(
        1.0, abs(float(reference_source))
    )
