"""Known owners must remain charged across host adapters and public preflight."""

import dataclasses
import weakref
from collections.abc import Callable, Mapping
from functools import partialmethod
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.transitions as transitions_module
import lcm._solver_api.entries as entries_module
from _lcm.execution.core_program import CoreProgram
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.program_types import DECISION_PROGRAM, ROUTE_PROGRAM
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.simulation.runtime import SimulationDispatchContext, SimulationRuntime
from _lcm.solution.retained_buffers import retained_solution_buffers
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.solution.test_solution_result import _small_grid_search_inputs


@categorical(ordered=False)
class _LifecycleRegimeId:
    alive: ScalarInt
    done: ScalarInt


def _lifecycle_utility(*, wealth: ContinuousState, saving: ContinuousAction) -> FloatND:
    return wealth + saving


def _lifecycle_terminal_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _lifecycle_next_wealth(
    *, wealth: ContinuousState, saving: ContinuousAction
) -> FloatND:
    return wealth + saving


def _lifecycle_next_regime() -> ScalarInt:
    return _LifecycleRegimeId.done


def _only_initial_age(age: float) -> bool:
    return age == 0


def _stateful_target_model() -> Model:
    return Model(
        regimes={
            "alive": Regime(
                transition=_lifecycle_next_regime,
                active=_only_initial_age,
                functions={"utility": _lifecycle_utility},
                actions={"saving": LinSpacedGrid(start=1, stop=2, n_points=2)},
            ),
            "done": Regime(
                transition=None, functions={"utility": _lifecycle_terminal_utility}
            ),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=5)},
        state_transitions={"wealth": _lifecycle_next_wealth},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ObservedStateMerge:
    """Weak observations do not extend either raw or merged array ownership."""

    original: Callable[..., object]
    raw_outputs: list[weakref.ReferenceType[jax.Array]]
    merged_carrier: list[weakref.ReferenceType[jax.Array]]
    reached: list[str]


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_profiled_merge(
    self: ProfiledSimulationOperations,
    *,
    original: Callable[..., object],
    observed: _ObservedStateMerge,
    **kwargs: Any,
) -> object:
    is_merge = kwargs["function"] is observed.original
    if is_merge:
        observed.raw_outputs[:] = [
            weakref.ref(leaf)
            for leaf in jax.tree.leaves(kwargs["arguments"]["next_states_per_regime"])
            if isinstance(leaf, jax.Array)
        ]
    result = original(self, **kwargs)
    if is_merge:
        observed.merged_carrier[:] = [
            weakref.ref(leaf)
            for leaf in jax.tree.leaves(result)
            if isinstance(leaf, jax.Array)
        ]
    return result


# keyword-only-exempt: library-callback=functools.partialmethod
def _inspect_route_inventory(
    self: SimulationRuntime,
    *,
    original: Callable[..., object],
    observed: _ObservedStateMerge,
    channel: str,
    program: CoreProgram,
    arguments: Mapping[str, object],
    period: int,
    n_subjects: int,
    residency: SimulationDispatchContext | None = None,
) -> object:
    if program.name == ROUTE_PROGRAM and observed.raw_outputs:
        assert residency is not None
        observed.reached.append(channel)
        if channel == "raw_outputs":
            released = sum(reference() is None for reference in observed.raw_outputs)
            assert released == 0, (
                f"{released} raw output owners died before unit commit"
            )
        else:
            merged = tuple(reference() for reference in observed.merged_carrier)
            assert all(array is not None for array in merged)
            missing = resident_bytes_by_device(
                live=measure_buffer_footprint(tree=merged),
                arguments=residency.live_footprint(),
                devices=residency.budget_devices,
            )
            assert all(size == 0 for size in missing.values()), (
                f"Live merged carrier is absent from route inventory: {dict(missing)}"
            )
    return original(
        self,
        program=program,
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
        residency=residency,
    )


def _assert_transition_inventory(
    *, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    """Run the public stateful-target model with exact route-time owner observations."""
    observed = _ObservedStateMerge(
        original=transitions_module._advance_states_for_subjects,
        raw_outputs=[],
        merged_carrier=[],
        reached=[],
    )
    monkeypatch.setattr(
        ProfiledSimulationOperations,
        "dispatch",
        partialmethod(
            _observe_profiled_merge,
            original=ProfiledSimulationOperations.dispatch,
            observed=observed,
        ),
    )
    monkeypatch.setattr(
        SimulationRuntime,
        "dispatch",
        partialmethod(
            _inspect_route_inventory,
            original=SimulationRuntime.dispatch,
            observed=observed,
            channel=channel,
        ),
    )
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="off"
    )
    assert observed.reached


@pytest.mark.parametrize("channel", ["raw_outputs", "merged_carrier"])
def test_transition_owners_survive_and_are_counted_before_route(
    *, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    """An actual public GridSearch transition keeps both sides of a host merge live."""
    _assert_transition_inventory(monkeypatch=monkeypatch, channel=channel)


# keyword-only-exempt: library-callback=SimulationMemory.hold
def _drop_host_roots(self: SimulationMemory, tree: object) -> None:
    del self, tree


def test_missing_host_root_handoff_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the explicit host merge handoff makes the independent inventory fail."""
    monkeypatch.setattr(SimulationMemory, "hold", _drop_host_roots)
    with pytest.raises(AssertionError, match="Live merged carrier"):
        _assert_transition_inventory(monkeypatch=monkeypatch, channel="merged_carrier")


# keyword-only-exempt: library-callback=functools.partialmethod
def _remember_decision_values(
    self: SimulationRuntime,
    *,
    original: Callable[..., object],
    values: list[weakref.ReferenceType[jax.Array]],
    **kwargs: Any,
) -> object:
    result = original(self, **kwargs)
    if kwargs["program"].name == DECISION_PROGRAM:
        assert isinstance(result, tuple)
        values[:] = [weakref.ref(leaf) for leaf in jax.tree.leaves(result[1])]
    return result


# keyword-only-exempt: library-callback=functools.partialmethod
def _inspect_lookup_inventory(
    self: ProfiledSimulationOperations,
    *,
    original: Callable[..., object],
    values: list[weakref.ReferenceType[jax.Array]],
    reached: list[bool],
    **kwargs: Any,
) -> object:
    if kwargs["function"].__name__ == "_lookup_values_from_indices" and values:
        arrays = tuple(reference() for reference in values)
        assert all(array is not None for array in arrays)
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=arrays),
            arguments=kwargs["live_footprint"](),
            devices=kwargs["budget_devices"],
        )
        assert all(size == 0 for size in missing.values()), (
            f"Decision values absent from action lookup inventory: {dict(missing)}"
        )
        reached.append(True)
    return original(self, **kwargs)


def test_raw_decision_values_are_counted_during_action_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The action gather's live inventory includes the distinct raw value output."""
    values: list[weakref.ReferenceType[jax.Array]] = []
    reached: list[bool] = []
    monkeypatch.setattr(
        SimulationRuntime,
        "dispatch",
        partialmethod(
            _remember_decision_values,
            original=SimulationRuntime.dispatch,
            values=values,
        ),
    )
    monkeypatch.setattr(
        ProfiledSimulationOperations,
        "dispatch",
        partialmethod(
            _inspect_lookup_inventory,
            original=ProfiledSimulationOperations.dispatch,
            values=values,
            reached=reached,
        ),
    )
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="off"
    )
    assert reached


def _reject_private_copy(*, value: object, label: str) -> object:
    del value, label
    raise AssertionError("Foreign solution payload copied before the one-byte refusal")


def test_insufficient_budget_refuses_before_foreign_payload_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse existing retained arrays before a private copy can start."""
    source_model, params, initial = _small_grid_search_inputs()
    solution = source_model.solve(params=params, log_level="off")
    assert any(
        array.nbytes > 1 for array in retained_solution_buffers(solution=solution)
    )
    budgeted_model, _, _ = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=1)
    )
    monkeypatch.setattr(entries_module, "_copy_solution_value", _reject_private_copy)
    with pytest.raises(ExecutionPlanningError, match="budget"):
        budgeted_model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
        )
