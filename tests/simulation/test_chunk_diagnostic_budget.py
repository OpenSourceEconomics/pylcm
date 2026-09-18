"""Forward diagnostic operations must admit allocation with current carry live."""

import dataclasses
import inspect
import logging
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulation
import _lcm.solution.validate_V as validation
from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation import diagnostic_operations as diagnostics
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.solution.validate_V import validate_V, value_function_nan_error
from _lcm.utils.logging import LogLevel, get_logger, log_regime_transitions
from lcm.exceptions import ExecutionPlanningError, InvalidValueFunctionError
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)
from tests.simulation.test_population_allocation_budget import (
    _forbid_concrete,
    _memory,
    _UnadmittedAllocationError,
)


class _AdmissionObservedError(Exception):
    """Stop the public run exactly where the intended budget refusal was checked."""


@pytest.fixture(scope="module")
def public_simulation() -> Callable[..., object]:
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")

    def run(*, log_level: LogLevel = "debug") -> object:
        return model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level=log_level,
        )

    return run


def _capture_memory(monkeypatch: pytest.MonkeyPatch) -> list[SimulationMemory]:
    memories: list[SimulationMemory] = []

    def create(**arguments: Any) -> SimulationMemory:
        memory = SimulationMemory(**arguments)
        memories.append(memory)
        return memory

    monkeypatch.setattr(simulation, "SimulationMemory", create)
    return memories


@pytest.mark.parametrize(
    "operation",
    ["_validate_period_values", "_validate_simulated_value", "log_regime_transitions"],
)
def test_public_diagnostic_refuses_before_concrete_allocation(
    *,
    monkeypatch: pytest.MonkeyPatch,
    public_simulation: Callable[..., object],
    operation: str,
) -> None:
    """A real completed period reaches admission before any diagnostic dispatch."""
    memories = _capture_memory(monkeypatch)
    original = getattr(simulation, operation)
    reached: list[str] = []
    if operation == "_validate_simulated_value":
        period_validation = simulation._validate_period_values

        def inject_nan(**arguments: Any) -> None:
            rows = arguments["period_results"]
            arguments["period_results"] = tuple(
                (
                    name,
                    dataclasses.replace(data, V_arr=jnp.full_like(data.V_arr, jnp.nan)),
                )
                for name, data in rows
            )
            period_validation(**arguments)

        monkeypatch.setattr(simulation, "_validate_period_values", inject_nan)

    def inspect_admission(**arguments: Any) -> None:
        reached.append(operation)
        memories[-1].budget_bytes = 1
        with monkeypatch.context() as guard:
            flags = simulation.non_finite_by_regime
            where = jnp.where

            def guard_flags(**operands: Any) -> object:
                if not any(
                    isinstance(leaf, jax.core.Tracer)
                    for leaf in jax.tree.leaves(operands)
                ):
                    raise _UnadmittedAllocationError("Unprofiled period flag dispatch")
                return flags(**operands)

            def guard_where(condition: object, *args: Any, **kwargs: Any) -> object:
                if not isinstance(condition, jax.core.Tracer):
                    raise _UnadmittedAllocationError("Unprofiled owned-value mask")
                return where(condition, *args, **kwargs)

            guard.setattr(simulation, "non_finite_by_regime", guard_flags)
            guard.setattr(jnp, "where", guard_where)
            guard.setattr(jax, "device_put", _forbid_concrete)
            guard.setattr(
                jax._src.core.EvalTrace, "process_primitive", _forbid_concrete
            )
            with pytest.raises(_UnadmittedAllocationError):
                jnp.zeros(3)
            with pytest.raises(ExecutionPlanningError):
                original(**arguments)
        raise _AdmissionObservedError

    monkeypatch.setattr(simulation, operation, inspect_admission)
    with pytest.raises(_AdmissionObservedError):
        public_simulation()
    assert reached == [operation]


def test_period_diagnostics_charge_new_carry_after_last_unit_closes(
    *, monkeypatch: pytest.MonkeyPatch, public_simulation: Callable[..., object]
) -> None:
    """Current state, membership, role and RNG arrays survive the unit handoff."""
    memories = _capture_memory(monkeypatch)
    original = simulation._validate_period_values
    inspected: list[bool] = []

    def observe(**arguments: Any) -> None:
        frame = inspect.currentframe()
        assert frame is not None
        assert frame.f_back is not None
        caller = frame.f_back.f_locals
        roots = tuple(
            caller[name]
            for name in (
                "states",
                "prev_regime_ids",
                "subject_regime_ids",
                "new_subject_regime_ids",
                "own_stakeholder",
                "new_own_stakeholder",
                "key",
                "taste_key",
                "age",
            )
        )
        expected = measure_buffer_footprint(tree=roots)
        observed = memories[-1].snapshot()
        missing = resident_bytes_by_device(
            live=expected, arguments=observed, devices=memories[-1].devices
        )
        assert not any(missing.values()), missing
        inspected.append(True)
        original(**arguments)

    monkeypatch.setattr(simulation, "_validate_period_values", observe)
    public_simulation()
    assert inspected


@pytest.mark.parametrize("collective", [False, True])
def test_profiled_value_flags_and_count_preserve_owned_rows(
    *, collective: bool
) -> None:
    value = jnp.asarray([[1.0, jnp.nan], [jnp.nan, jnp.inf], [jnp.inf, -0.0]])
    if not collective:
        value = value[:, 0]
    mask = jnp.asarray([True, False, True])
    memory = _memory(inputs=(value, mask), budget=1_000_000)
    flags = memory.run(
        function=diagnostics.period_value_flags,
        arguments={"values": (value,), "in_regime": (mask,)},
        subject_arg_names=("values", "in_regime"),
    )
    count = memory.run(
        function=diagnostics.owned_value_nan_count,
        arguments={"value": value, "in_regime": mask},
        subject_arg_names=("value", "in_regime"),
    )
    np.testing.assert_array_equal(flags, [[collective], [True]])
    assert int(count) == int(collective)
    assert count.dtype == jnp.sum(jnp.asarray([True])).dtype
    if collective:
        with pytest.raises(InvalidValueFunctionError) as original:
            simulation._validate_simulated_value(
                value=value,
                subject_ids_in_regime=mask,
                age=jnp.asarray(0, dtype=jnp.int32),
                regime_name="alive",
                logger=get_logger(log_level="debug"),
            )
        with pytest.raises(InvalidValueFunctionError) as profiled:
            simulation._validate_simulated_value(
                value=value,
                subject_ids_in_regime=mask,
                age=jnp.asarray(0, dtype=jnp.int32),
                regime_name="alive",
                logger=get_logger(log_level="debug"),
                memory=memory,
            )
        assert str(profiled.value) == str(original.value)
        assert "1 of 6 values are NaN" in str(profiled.value)


def test_profiled_transition_counts_keep_exact_sorted_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    previous = jnp.asarray([5, -1, 2, 5, 2], dtype=jnp.int32)
    current = jnp.asarray([2, 2, 5, 5, 2], dtype=jnp.int32)
    names = MappingProxyType({5: "five", 2: "two"})
    memory = _memory(inputs=(previous, current), budget=1_000_000)
    counts = diagnostics.profiled_transition_counts(
        memory=memory,
        prev_regime_ids=previous,
        new_regime_ids=current,
        sorted_ids=(2, 5),
    )
    assert counts == [[1, 1], [1, 1]]
    logger = get_logger(log_level="debug")
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        log_regime_transitions(
            logger=logger,
            prev_regime_ids=previous,
            new_regime_ids=current,
            regime_ids_to_names=names,
        )
        log_regime_transitions(
            logger=logger,
            prev_regime_ids=previous,
            new_regime_ids=current,
            regime_ids_to_names=names,
            counts_factory=lambda: counts,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert (
        messages[-2]
        == messages[-1]
        == (
            "  transitions:\n  - two → two = 1\n  - two → five = 1\n"
            "  - five → two = 1\n  - five → five = 1"
        )
    )


@pytest.mark.parametrize("level", ["off", "warning", "debug"])
def test_period_diagnostic_levels_and_error_order_are_preserved(
    *, caplog: pytest.LogCaptureFixture, level: LogLevel
) -> None:
    mask = jnp.asarray([True, True, False])
    first = jnp.asarray([jnp.nan, 1.0, jnp.nan])
    second = jnp.asarray([jnp.inf, 2.0, jnp.nan])
    records = tuple(
        (
            name,
            PeriodRegimeSimulationData(
                V_arr=value,
                actions=MappingProxyType({}),
                states=MappingProxyType({}),
                in_regime=mask,
                own_stakeholder=jnp.full(3, -1, dtype=jnp.int32),
                nested_policy_fallback=jnp.zeros(3, dtype=bool),
            ),
        )
        for name, value in (("first", first), ("second", second))
    )
    memory = _memory(
        inputs=(first, second, mask), budget=1 if level == "off" else 1_000_000
    )
    logger = get_logger(log_level=level)
    arguments = {
        "logger": logger,
        "age": jnp.asarray(0, dtype=jnp.int32),
        "period_results": records,
        "memory": memory,
    }
    if level == "debug":
        with pytest.raises(InvalidValueFunctionError, match="1 of 3 values are NaN"):
            simulation._validate_period_values(**arguments)
    else:
        simulation._validate_period_values(**arguments)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "NaN/Inf in V_arr" in record.getMessage()
    ]
    assert warnings == (
        []
        if level == "off"
        else [
            "NaN/Inf in V_arr for regime 'first' at age 0",
            "NaN/Inf in V_arr for regime 'second' at age 0",
        ]
    )
    assert len(memory.operations.cache) == (0 if level == "off" else 2)


def test_host_nan_report_keeps_exception_payload_and_enrichment_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = object()
    seen: list[InvalidValueFunctionError] = []

    def enrich(**arguments: Any) -> None:
        error = arguments["exc"]
        assert error.partial_solution is payload
        seen.append(error)
        error.add_note("enrichment ran after exception construction")

    monkeypatch.setattr(validation, "_enrich_with_diagnostics", enrich)
    value = jnp.asarray([jnp.nan, 1.0])
    with pytest.raises(InvalidValueFunctionError) as caught:
        validate_V(
            V_arr=value,
            age=20.0,
            regime_name="alive",
            partial_solution=payload,
            entered_process_names=("z", "a"),
            compute_intermediates=lambda: None,
            state_action_space=simulation.StateActionSpace(
                states=MappingProxyType({}),
                discrete_actions=MappingProxyType({}),
                continuous_actions=MappingProxyType({}),
                state_and_discrete_action_names=(),
            ),
        )
    expected = value_function_nan_error(
        n_nan=1,
        total=2,
        age=20.0,
        regime_name="alive",
        partial_solution=payload,
        entered_process_names=("z", "a"),
    )
    assert type(caught.value) is type(expected) is InvalidValueFunctionError
    assert str(caught.value) == str(expected)
    assert "'a', 'z'" in str(expected)
    assert caught.value.partial_solution is expected.partial_solution is payload
    assert seen == [caught.value]
    assert caught.value.__notes__ == ["enrichment ran after exception construction"]


@pytest.mark.parametrize("level", ["off", "warning", "debug"])
def test_diagnostic_bindings_lower_actual_operations_without_allocating(
    *, monkeypatch: pytest.MonkeyPatch, level: LogLevel
) -> None:
    value = jnp.asarray([1.0, jnp.nan, 3.0])
    mask = jnp.asarray([True, False, True])
    ids = jnp.asarray([2, 5, 2], dtype=jnp.int32)
    memory = _memory(inputs=(value, mask, ids), budget=1_000_000)
    abstract = jax.tree.map(
        lambda array: jax.ShapeDtypeStruct(
            array.shape, array.dtype, sharding=array.sharding
        ),
        (value, mask, ids),
    )
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        bindings = diagnostics.diagnostic_bindings(
            values=(abstract[0],),
            in_regime=(abstract[1],),
            prev_regime_ids=abstract[2],
            new_regime_ids=abstract[2],
            sorted_ids=(2, 5),
            log_level=level,
        )
        profiles = tuple(
            memory.operations.prepare_abstract(
                function=binding.function,
                arguments=binding.arguments,
                subject_arg_names=binding.subject_arg_names,
                static_arguments=binding.static_arguments,
                subject_outputs=binding.subject_outputs,
                devices=memory.subject_devices,
            )
            for binding in bindings
        )
    assert tuple(binding.function for binding in bindings) == (
        ()
        if level == "off"
        else (
            diagnostics.period_value_flags,
            diagnostics.owned_value_nan_count,
            *((diagnostics.transition_counts,) if level == "debug" else ()),
        )
    )
    assert tuple(profile.executable.out_info.shape for profile in profiles) == (
        () if level == "off" else ((2, 1), (), *(((2, 2),) if level == "debug" else ()))
    )
    assert all(profile.peak_bytes > 0 for profile in profiles)
    with pytest.raises(ExecutionPlanningError, match="abstract"):
        diagnostics.DiagnosticBinding(
            function=diagnostics.owned_value_nan_count,
            arguments={"value": value, "in_regime": mask},
            subject_arg_names=("value", "in_regime"),
        )
