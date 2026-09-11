"""Operand-scaled comparison for the additive multi-regime simulation witness.

For ``Q = U + CE``, ``abs(delta_Q) / (abs(U) + abs(CE))`` is the minimum
componentwise relative backward perturbation explaining a width difference.
The eight-ULP budget therefore uses this input norm, evaluated independently
from the retained reference solution. It is neither a larger empirical Q-ULP
threshold nor a claim that every internal CE lies within eight of its own ULP.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.regime_building.Q_and_F import _QAndF
from _lcm.simulation.programs import _StreamedArgmaxQOverA, _SubjectTiled
from _lcm.typing import FlatParams
from benchmarks.asv._simulation_witnesses import multi_regime
from lcm import CESAggregator, LinearAggregator, Model
from lcm.result import SimulationResult
from tests.test_models.processes import MultiRegimeId


def additive_operand_norm(
    *, utility: np.ndarray, continuation: np.ndarray
) -> np.ndarray:
    """Return the finite additive input norm in the operands' working format."""
    assert utility.dtype == continuation.dtype
    assert utility.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    assert utility.shape == continuation.shape
    assert np.isfinite(utility).all()
    assert np.isfinite(continuation).all()
    with np.errstate(over="ignore"):
        norm = np.abs(utility) + np.abs(continuation)
    assert np.isfinite(norm).all(), "Additive operand norm must remain finite."
    return np.asarray(norm)


def assert_additive_value_parity(
    *, got: np.ndarray, expected: np.ndarray, reference_operand_norm: np.ndarray
) -> None:
    """Require exact nonfinite patterns and eight ULP at each reference input norm.

    A zero input norm permits no departure. Neither compared values nor their
    difference can increase the bound; malformed/nonfinite norms are refused.
    """
    assert got.dtype == expected.dtype == reference_operand_norm.dtype
    assert got.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    assert got.shape == expected.shape == reference_operand_norm.shape
    assert np.isfinite(reference_operand_norm).all()
    assert (reference_operand_norm >= 0).all()
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(got), finite)
    np.testing.assert_array_equal(got[~finite], expected[~finite])
    norm = reference_operand_norm[finite]
    with np.errstate(over="ignore"):
        spacing = np.spacing(norm).astype(np.float64)
    assert np.isfinite(spacing).all(), "Additive operand spacing must remain finite."
    bound = np.where(norm > 0, 8 * spacing, 0)
    with np.errstate(over="ignore"):
        gap = np.abs(
            got[finite].astype(np.float64) - expected[finite].astype(np.float64)
        )
    assert (gap <= bound).all(), (
        f"Value width difference exceeds eight-ULP additive operand bound: "
        f"gaps={gap[gap > bound]!r}, bounds={bound[gap > bound]!r}."
    )


def _unwrap(function: Callable[..., Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Identify the canonical functor and preserve partialled fixed parameters."""
    bound = {}
    while True:
        if isinstance(function, partial):
            assert not function.args
            bound = dict(function.keywords) | bound
            function = function.func
        elif inspect.ismethod(function):
            owner = function.__self__
            assert callable(owner)
            function = owner
        else:
            unwrapped = inspect.unwrap(function)
            if unwrapped is function:
                return function, bound
            function = unwrapped


def _decision_q(function: Callable[..., Any]) -> tuple[_QAndF, dict[str, Any]]:
    """Reach the declared streamed decision's original canonical Q functor."""
    subject, bound = _unwrap(function)
    assert isinstance(subject, _SubjectTiled)
    reducer, reducer_bound = _unwrap(subject.func)
    assert isinstance(reducer, _StreamedArgmaxQOverA)
    q, q_bound = _unwrap(reducer.Q_and_F)
    assert isinstance(q, _QAndF)
    aggregator, _ = _unwrap(q.koopmans_aggregator)
    assert type(aggregator) is LinearAggregator
    return q, q_bound | reducer_bound | bound


def _evaluate_operands(q: _QAndF) -> Callable[..., Any]:
    """Evaluate actual canonical U and CE; never infer CE from a reported value."""

    @jax.jit
    def evaluate(
        *, cell: dict[str, Any], values: MappingProxyType
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        utility, feasible = q.U_and_F(**cell)
        continuation, _ = q.compute_CE(
            next_regime_to_V_arr=values,
            zero=jnp.zeros_like(utility),
            states_actions_params=cell,
        )
        aggregation = q.build_W_kwargs(cell)
        assert len(aggregation) == 1
        name, discount = next(iter(aggregation.items()))
        assert name.split("__")[-1] == "discount_factor"
        return utility, continuation, discount, feasible

    return evaluate


def assert_additive_witness(*, model: Model, flat_params: FlatParams) -> None:
    """Refuse to apply the additive scale without its exact declared assumptions."""
    for name, regime in model.user_regimes.items():
        if not regime.terminal:
            assert type(regime.koopmans_aggregator) is LinearAggregator
            np.testing.assert_array_equal(
                flat_params[name]["koopmans_aggregator__discount_factor"], 1
            )


def reference_additive_norms(
    *,
    model: Model,
    result: SimulationResult,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Compute per-row norms for this zero-terminal, beta-one additive witness.

    This pointwise canonical reevaluation provides an independent operand scale;
    it does not expose the vectorized kernel's live continuation operand. The
    returned host norms retain no model, solution array or JIT callable.
    """
    flat_params = result.flat_params
    assert_additive_witness(model=model, flat_params=flat_params)
    values = result.period_to_regime_to_V_arr
    norms = np.zeros((len(frame), 1), dtype=canonical_float_dtype())
    evaluators = {}
    for index, row in enumerate(frame.to_dict("records")):
        if not np.isfinite(row["value"]):
            continue
        name = row["regime_name"]
        regime = model._regimes[name]
        if regime.terminal:
            np.testing.assert_array_equal(row["value"], 0)
            continue
        period, subject = int(row["period"]), int(row["subject_id"])
        q, bound = _decision_q(regime.simulation.programs.decision[period].function)
        raw = result.raw_results[name][period]
        cell = dict(flat_params[name]) | {
            key: value[subject]
            for key, value in (dict(raw.states) | dict(raw.actions)).items()
        }
        cell |= {"period": jnp.int32(period), "age": model.ages.values[period]}
        cell |= bound
        key = (name, period)
        if key not in evaluators:
            evaluators[key] = _evaluate_operands(q)
        utility, continuation, discount, feasible = evaluators[key](
            cell=cell, values=values[period + 1]
        )
        assert bool(feasible)
        np.testing.assert_array_equal(discount, 1)
        norms[index, 0] = additive_operand_norm(
            utility=np.asarray(utility), continuation=np.asarray(continuation)
        )
    norms.setflags(write=False)
    return norms


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("exponent", [-30, 0, 30])
def test_adjacent_operand_error_survives_cancellation_and_rescaling(
    *, dtype: type[np.floating], exponent: int
) -> None:
    """A known one-input-ULP error can span many result ULP without growing."""
    utility = dtype(2.0**exponent)
    remainder = dtype(utility * 2.0 ** (-10 if dtype is np.float32 else -30))
    continuation = dtype(-utility + remainder)
    adjacent = np.nextafter(continuation, dtype(0))
    expected, got = dtype(utility + continuation), dtype(utility + adjacent)
    assert adjacent > continuation
    assert np.isfinite(got)
    assert abs(float(got) - float(expected)) > 8 * float(np.spacing(expected))
    norm = additive_operand_norm(
        utility=np.asarray(utility), continuation=np.asarray(continuation)
    )
    assert_additive_value_parity(
        got=np.asarray(got), expected=np.asarray(expected), reference_operand_norm=norm
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eight_unit_boundary_and_seeded_value_error(
    *, dtype: type[np.floating]
) -> None:
    """The independently chosen budget boundary and a substantive mutant are live."""
    norm = np.asarray(1.9375, dtype=dtype)
    expected = np.asarray(0.0625, dtype=dtype)
    spacing = float(np.spacing(norm))
    assert_additive_value_parity(
        got=np.asarray(float(expected) + 8 * spacing, dtype=dtype),
        expected=expected,
        reference_operand_norm=norm,
    )
    for perturbation in (9 * spacing, 0.001, 1e10):
        with pytest.raises(AssertionError, match="additive operand bound"):
            assert_additive_value_parity(
                got=np.asarray(float(expected) + perturbation, dtype=dtype),
                expected=expected,
                reference_operand_norm=norm,
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_norm_requires_exact_zero(*, dtype: type[np.floating]) -> None:
    """No underflow allowance is invented for an exactly zero additive input."""
    zero = np.zeros(1, dtype=dtype)
    norm = additive_operand_norm(utility=zero, continuation=zero)
    np.testing.assert_array_equal(norm, zero)
    assert_additive_value_parity(got=zero, expected=zero, reference_operand_norm=norm)
    adjacent = np.nextafter(zero, np.ones(1, dtype=dtype))
    assert (adjacent > zero).all()
    with pytest.raises(AssertionError, match="additive operand bound"):
        assert_additive_value_parity(
            got=adjacent, expected=zero, reference_operand_norm=norm
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_publications_keep_exact_patterns(
    *, dtype: type[np.floating]
) -> None:
    """NaN, either infinity and finite values cannot mask one another."""
    expected = np.asarray([np.nan, np.inf, -np.inf, 0], dtype=dtype)
    norm = np.zeros(4, dtype=dtype)
    assert_additive_value_parity(
        got=expected.copy(), expected=expected, reference_operand_norm=norm
    )
    for index, replacement in enumerate([0, -np.inf, np.inf, np.nan]):
        got = expected.copy()
        got[index] = replacement
        with pytest.raises(AssertionError):
            assert_additive_value_parity(
                got=got, expected=expected, reference_operand_norm=norm
            )


@pytest.mark.parametrize("norm_value", [np.nan, np.inf, -np.inf, -1])
def test_invalid_reference_norm_is_refused(*, norm_value: float) -> None:
    """Malformed scales cannot turn the comparison into a silent pass."""
    value = np.asarray([1.0])
    with pytest.raises(AssertionError):
        assert_additive_value_parity(
            got=value,
            expected=value,
            reference_operand_norm=np.asarray([norm_value], dtype=value.dtype),
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_or_overflowing_operand_norm_is_refused(
    *, dtype: type[np.floating]
) -> None:
    """An invalid norm has no infinite-error-budget fallback."""
    for operand in (np.nan, np.inf, np.finfo(dtype).max):
        value = np.asarray([operand], dtype=dtype)
        with pytest.raises(AssertionError):
            additive_operand_norm(utility=value, continuation=value)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_finite_max_norm_cannot_grant_infinite_budget(
    *, dtype: type[np.floating]
) -> None:
    """A finite norm whose next representable neighbor overflows is refused."""
    norm = np.asarray([np.finfo(dtype).max], dtype=dtype)
    assert np.isfinite(norm).all()
    with pytest.raises(AssertionError, match="spacing must remain finite"):
        assert_additive_value_parity(
            got=np.asarray([2], dtype=dtype),
            expected=np.asarray([1], dtype=dtype),
            reference_operand_norm=norm,
        )


def test_nonunit_discount_refuses_additive_witness_contract() -> None:
    """The witness must still declare and supply beta exactly one."""
    model, params, _ = multi_regime()
    assert_additive_witness(model=model, flat_params=model._process_params(params))
    work_params = params["work"]
    assert isinstance(work_params, Mapping)
    params = dict(params) | {"work": dict(work_params) | {"discount_factor": 0.9}}
    with pytest.raises(AssertionError):
        assert_additive_witness(model=model, flat_params=model._process_params(params))


def test_other_aggregator_refuses_additive_witness_contract() -> None:
    """A nonlinear user declaration cannot acquire the additive error scale."""
    original, params, _ = multi_regime()
    model = Model(
        regimes={
            name: regime
            if regime.terminal
            else replace(regime, koopmans_aggregator=CESAggregator())
            for name, regime in original.user_regimes.items()
        },
        ages=original.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=original.fixed_params,
    )
    with pytest.raises(AssertionError):
        assert_additive_witness(
            model=model, flat_params=original._process_params(params)
        )
