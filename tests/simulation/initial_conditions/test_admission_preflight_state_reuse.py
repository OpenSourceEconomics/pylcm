"""Identical state-law inputs may share reduced flags within one preflight."""

import dataclasses
import gc
import logging
import weakref
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.transition_checks as checks
from _lcm.engine import StateActionSpace, _StochasticStateTransition
from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.typing import FlatRegimeParams
from _lcm.utils.logging import LogLevel, get_logger
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.typing import DiscreteState, FloatND, IntND, ScalarInt, UserParams
from tests.simulation.test_compile_requests import _lcm_log_output_held_fixed
from tests.test_models.processes import next_health


class _Arguments(TypedDict):
    transition: _StochasticStateTransition
    regime_params: FlatRegimeParams
    state_action_space: StateActionSpace
    regime_name: str
    age: ScalarInt
    period: int
    logger: logging.Logger


def _law(*, health: DiscreteState, probability: FloatND) -> FloatND:
    return probability[health]


def _other_law(*, health: DiscreteState, probability: FloatND) -> FloatND:
    return probability[health]


def _age_law(*, health: DiscreteState, probability: FloatND, age: ScalarInt) -> FloatND:
    return probability[health] + jnp.where(age > 0, 0.1, 0.0)


def _period_law(
    *, health: DiscreteState, probability: FloatND, period: ScalarInt
) -> FloatND:
    return probability[health] + jnp.where(period > 0, 0.1, 0.0)


def _mapping_law(*, health: DiscreteState, probability: MappingLeaf) -> FloatND:
    array = probability.data["values"]
    assert isinstance(array, jax.Array)
    return array[health]


def _arguments(*, function: Callable[..., FloatND] = _law) -> _Arguments:
    return {
        "transition": _StochasticStateTransition(
            func=function,
            state_name="health",
            target_regime_name="target",
            n_outcomes=2,
            indexing_params=(),
            phase="solve",
        ),
        "regime_params": MappingProxyType(
            {"probability": jnp.array([[0.4, 0.6], [0.5, 0.5]])}
        ),
        "state_action_space": StateActionSpace(
            states=MappingProxyType({"health": jnp.array([0, 1], dtype=jnp.int32)}),
            discrete_actions=MappingProxyType({}),
            continuous_actions=MappingProxyType({}),
            state_and_discrete_action_names=("health",),
        ),
        "regime_name": "source",
        "age": jnp.int32(0),
        "period": 0,
        "logger": get_logger(log_level="debug"),
    }


@pytest.fixture
def evaluations(*, monkeypatch: pytest.MonkeyPatch) -> list[Callable[..., Any]]:
    """Count actual user-law evaluation at the existing vmap callable seam."""
    original = checks._GridPointCall.__call__
    observed: list[Callable[..., Any]] = []

    def record(self: checks._GridPointCall, *arguments: FloatND | IntND) -> Any:
        observed.append(self.func)
        return original(self, *arguments)

    monkeypatch.setattr(checks._GridPointCall, "__call__", record)
    return observed


def test_public_preflight_evaluates_each_identical_state_law_once_per_call(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real five-period health sweep has two distinct bound law inputs."""
    model, params, initial = WITNESSES["multi_regime"]()
    solution = model.solve(params=params, log_level="off")
    original: Callable[..., Any] = checks._GridPointCall.__call__
    evaluations: list[object] = []

    def observed(self: checks._GridPointCall, *arguments: FloatND | IntND) -> Any:
        if self.func is next_health:
            evaluations.append(self.func)
        return original(self, *arguments)

    monkeypatch.setattr(checks._GridPointCall, "__call__", observed)
    with _lcm_log_output_held_fixed():
        baseline = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
            seed=0,
        )
        for _ in range(2):
            evaluations.clear()
            result = model.simulate(
                params=params,
                initial_conditions=initial,
                solution=solution,
                log_level="progress",
                seed=0,
            )
            jax.block_until_ready(result.raw_results)
            assert len(evaluations) == 2
            expected, expected_tree = jax.tree.flatten(baseline.raw_results)
            actual, actual_tree = jax.tree.flatten(result.raw_results)
            assert actual_tree == expected_tree
            for left, right in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(right, left)


@pytest.mark.parametrize(
    "changed",
    ["parameter", "grid", "target", "phase", "source", "callable", "age", "period"],
)
def test_every_consumed_binding_and_context_is_rechecked(
    *,
    evaluations: list[Callable[..., Any]],
    changed: str,
) -> None:
    """Repeated inputs reuse flags; each distinct real binding is evaluated again."""
    function = {"age": _age_law, "period": _period_law}.get(changed, _law)
    arguments = _arguments(function=function)
    summary = checks._ValidationSummary()
    try:
        checks._validate_state_transition_single(**arguments, summary=summary)
        checks._validate_state_transition_single(**arguments, summary=summary)
        assert len(evaluations) == 1
        replacement: _Arguments = {**arguments}
        if changed == "parameter":
            replacement["regime_params"] = MappingProxyType(
                {"probability": jnp.array([[0.2, 0.2], [0.5, 0.5]])}
            )
        elif changed == "grid":
            replacement["state_action_space"] = arguments["state_action_space"].replace(
                states=MappingProxyType({"health": jnp.array([1, 0], dtype=jnp.int32)})
            )
        elif changed == "target":
            replacement["transition"] = dataclasses.replace(
                arguments["transition"], target_regime_name="other"
            )
        elif changed == "phase":
            replacement["transition"] = dataclasses.replace(
                arguments["transition"], phase="simulate"
            )
        elif changed == "source":
            replacement["regime_name"] = "other"
        elif changed == "callable":
            replacement["transition"] = dataclasses.replace(
                arguments["transition"], func=_other_law
            )
        elif changed == "age":
            replacement["age"] = jnp.int32(1)
        else:
            replacement["period"] = 1
        checks._validate_state_transition_single(**replacement, summary=summary)
        assert len(evaluations) == 2
        assert len(summary.flags) == 3
        assert summary.valid() is (changed not in ("parameter", "age", "period"))
    finally:
        summary.close()
    assert not summary.state_probabilities
    again = checks._ValidationSummary()
    try:
        checks._validate_state_transition_single(**arguments, summary=again)
        assert len(evaluations) == 3
    finally:
        again.close()


def test_changed_outcome_count_is_checked_even_for_reused_numerical_inputs(
    *,
    evaluations: list[Callable[..., Any]],
) -> None:
    """The same probability grid cannot satisfy a different declared outcome size."""
    arguments = _arguments()
    summary = checks._ValidationSummary()
    try:
        checks._validate_state_transition_single(**arguments, summary=summary)
        arguments["transition"] = dataclasses.replace(
            arguments["transition"], n_outcomes=3
        )
        with pytest.raises(checks._SerialValidationRequired):
            checks._validate_state_transition_single(**arguments, summary=summary)
        assert len(evaluations) == 1
    finally:
        summary.close()


def test_reuse_keeps_reduced_flags_without_retaining_probability_grids(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-open summary does not root the potentially large law output."""
    original = checks._check_state_probs
    probabilities: list[weakref.ReferenceType[jax.Array]] = []

    def observe(*, probs: FloatND, **kwargs: Any) -> None:
        probabilities.append(weakref.ref(probs))
        original(probs=probs, **kwargs)

    monkeypatch.setattr(checks, "_check_state_probs", observe)
    arguments = _arguments()
    summary = checks._ValidationSummary()
    try:
        checks._validate_state_transition_single(**arguments, summary=summary)
        assert summary.valid()
        assert summary.state_probabilities
        gc.collect()
        assert probabilities
        assert all(reference() is None for reference in probabilities)
    finally:
        summary.close()


def test_mutable_canonical_wrapper_is_evaluated_again(
    *,
    evaluations: list[Callable[..., Any]],
) -> None:
    """A wrapper can replace its data even though each contained array is immutable."""
    arguments = _arguments(function=_mapping_law)
    wrapper = MappingLeaf({"values": jnp.array([[0.4, 0.6], [0.5, 0.5]])})
    arguments["regime_params"] = MappingProxyType({"probability": wrapper})
    summary = checks._ValidationSummary()
    try:
        checks._validate_state_transition_single(**arguments, summary=summary)
        wrapper.data = MappingProxyType({"values": jnp.array([[0.2, 0.2], [0.5, 0.5]])})
        checks._validate_state_transition_single(**arguments, summary=summary)
        assert len(evaluations) == 2
        assert not summary.state_probabilities
        assert not summary.valid()
    finally:
        summary.close()


def test_budgeted_preflight_keeps_profiled_checks_without_reuse(
    *,
    evaluations: list[Callable[..., Any]],
) -> None:
    """The new reuse route cannot bypass profiled budgeted validation operations."""
    arguments = _arguments()
    devices = (jax.devices()[0],)
    memory = SimulationMemory(
        budget_bytes=2**20,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(
            tree=(
                arguments["regime_params"],
                arguments["state_action_space"].states,
            )
        ),
    )
    summary = checks._ValidationSummary(memory=memory)
    try:
        checks._validate_state_transition_single(**arguments, summary=summary)
        checks._validate_state_transition_single(**arguments, summary=summary)
        assert len(evaluations) == 2
        assert not summary.state_probabilities
        assert summary.valid()
    finally:
        summary.close()


def test_close_releases_every_identity_keyed_input_owner() -> None:
    """Strong keys prevent identity reuse during the call and vanish at close."""
    references: list[weakref.ReferenceType[jax.Array]] = []

    def build_summary() -> checks._ValidationSummary:
        arguments = _arguments()
        for value in (
            arguments["regime_params"]["probability"],
            arguments["state_action_space"].states["health"],
        ):
            assert isinstance(value, jax.Array)
            references.append(weakref.ref(value))
        summary = checks._ValidationSummary()
        checks._validate_state_transition_single(**arguments, summary=summary)
        return summary

    summary = build_summary()
    assert summary.valid()
    gc.collect()
    assert all(reference() is not None for reference in references)
    summary.close()
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("log_level", ["warning", "progress", "debug"])
def test_cached_invalid_flags_preserve_complete_ordered_serial_diagnostics(
    *,
    log_level: LogLevel,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    evaluations: list[Callable[..., Any]],
) -> None:
    """An invalid reused flag still runs the original period-by-period retry."""
    model, _, _ = WITNESSES["multi_regime"]()
    invalid = jnp.array([[1.2, -0.2], [0.5, 0.5]])
    params: UserParams = {
        "work": {
            "discount_factor": 1.0,
            "next_health": {"probs_array": invalid},
            "income": {"mu": 0.0, "sigma": 1.0},
        },
        "retire": {
            "discount_factor": 1.0,
            "next_health": {"probs_array": invalid},
            "income": {"mu": 0.0, "sigma": 1.0},
        },
        "dead": {},
    }
    flat_params = model._process_params(params=params)
    logger = get_logger(log_level=log_level)
    monkeypatch.setattr(logger, "propagate", True)

    def diagnostics(*, serial: bool) -> tuple[str | None, list[str], int]:
        caplog.clear()
        evaluations.clear()
        error = None
        try:
            if serial:
                checks._validate_transition_sequence(
                    regimes=model._regimes,
                    flat_params=flat_params,
                    ages=model.ages,
                    logger=logger,
                    summary=None,
                )
            else:
                checks.validate_transitions(
                    regimes=model._regimes,
                    flat_params=flat_params,
                    ages=model.ages,
                    logger=logger,
                )
        except InvalidStateTransitionProbabilitiesError as caught:
            error = str(caught)
        messages = [
            record.getMessage() for record in caplog.records if record.name == "lcm"
        ]
        count = sum(function is next_health for function in evaluations)
        return error, messages, count

    expected_error, expected_messages, serial_count = diagnostics(serial=True)
    actual_error, actual_messages, actual_count = diagnostics(serial=False)
    assert actual_error == expected_error
    assert actual_messages == expected_messages
    assert actual_count == serial_count + 2
    if log_level == "debug":
        assert actual_error is not None
        assert "in regime 'work' at age 0" in actual_error
        assert not actual_messages
    else:
        assert actual_error is None
        assert len(actual_messages) == 5
        assert all(
            "returned values outside [0, 1]" in message for message in actual_messages
        )
