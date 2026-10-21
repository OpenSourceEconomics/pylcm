"""A budgeted simulation counts every buffer a dissolution route holds.

Under a finite device-memory budget the simulation measures what each held
owner keeps alive. The carriers a dissolution route produces are held for the
rest of the unit, so the footprint measured through the held owner must be
exactly the footprint of its array fields, named one by one.

The open-branch entry carrier is built and updated inside admitted operations,
and stays a live root from the period that writes it through the diagnostics
of that period and the setup of the next one, where it is read.
"""

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation import chunk_operations, simulate
from _lcm.simulation.dissolution_route import (
    RerouteOutcome,
    empty_gated_entries,
    record_open_entries,
)
from _lcm.simulation.host_operations import StaticArgument
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.typing import FootprintTree, PytreeByPeriod, PytreeValue, RegimeName
from lcm import ExecutionConfig, Model
from lcm.typing import ReferenceName, ScalarFloat, ScalarInt
from tests.simulation.test_realised_dissolution_route import _PARAMS, _build_model


def _outcome_arrays(*, outcome: RerouteOutcome) -> FootprintTree:
    return (
        outcome.states,
        outcome.subject_regime_ids,
        outcome.own_stakeholder,
        outcome.new_subject_regime_ids,
        outcome.new_own_stakeholder,
        outcome.entered_through,
        outcome.rerouted,
        outcome.fallback_ids,
    )


def test_a_held_reroute_outcome_measures_every_array_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    original_hold = SimulationMemory.hold

    # keyword-only-exempt: library-callback=SimulationUnitExecutor.on_output
    def observe_hold(self: SimulationMemory, tree: PytreeByPeriod) -> None:
        original_hold(self, tree=tree)
        if isinstance(tree, RerouteOutcome):
            explicit = _outcome_arrays(outcome=tree)
            jax.block_until_ready(explicit)
            observed.append(
                (
                    measure_buffer_footprint(tree=tree).spans,
                    measure_buffer_footprint(tree=explicit).spans,
                )
            )

    monkeypatch.setattr(SimulationMemory, "hold", observe_hold)
    model = _build_model(
        payoff_name="bowl",
        execution_config=ExecutionConfig(
            devices=(0,), axis_widths={"subject": 2}, device_memory_bytes=2**30
        ),
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.array([3.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["source"], jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    assert observed
    assert all(
        explicit != MappingProxyType({}) and held == explicit
        for held, explicit in observed
    )


def _budgeted_model() -> Model:
    return _build_model(
        payoff_name="bowl",
        execution_config=ExecutionConfig(
            devices=(0,), axis_widths={"subject": 2}, device_memory_bytes=2**30
        ),
    )


def _simulate_budgeted(*, model: Model) -> None:
    solution = model.solve(params=_PARAMS, log_level="off")
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.array([3.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["source"], jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )


def _record_entry_operations(
    *, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, bool, PytreeValue]]:
    """Record each entry operation the simulation runs, budgeted or not."""
    recorded: list[tuple[str, bool, PytreeValue]] = []
    original_run = simulate.run_simulation_operation

    def run_and_record[T: PytreeValue](
        *,
        memory: SimulationMemory | None,
        function: Callable[..., T],
        arguments: Mapping[ReferenceName, PytreeValue],
        subject_arg_names: tuple[ReferenceName, ...] = (),
        static_arguments: Mapping[str, StaticArgument] = MappingProxyType({}),
        subject_outputs: bool = False,
    ) -> T:
        result = original_run(
            memory=memory,
            function=function,
            arguments=arguments,
            subject_arg_names=subject_arg_names,
            static_arguments=static_arguments,
            subject_outputs=subject_outputs,
        )
        if function in (empty_gated_entries, record_open_entries):
            name = "empty" if function is empty_gated_entries else "record"
            recorded.append((name, memory is not None, result))
        return result

    monkeypatch.setattr(simulate, "run_simulation_operation", run_and_record)
    return recorded


def test_open_entries_are_built_and_recorded_inside_admitted_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a budget both entry operations run through the budget's admission."""
    recorded = _record_entry_operations(monkeypatch=monkeypatch)
    _simulate_budgeted(model=_budgeted_model())
    assert {(name, budgeted) for name, budgeted, _ in recorded} == {
        ("empty", True),
        ("record", True),
    }


def test_entry_carriers_stay_live_roots_through_diagnostics_and_next_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry carrier a period writes is a root while it can still be read."""
    recorded = _record_entry_operations(monkeypatch=monkeypatch)
    observed: list[tuple[str, bool]] = []
    original_validate = simulate._validate_period_values
    original_period_age = chunk_operations.period_age

    def is_root(*, memory: SimulationMemory | None) -> bool:
        assert memory is not None
        leaves = jax.tree.leaves(memory.unit_inputs)
        return any(leaf is recorded[-1][2] for leaf in leaves)

    def validate(
        *,
        logger: logging.Logger,
        age: ScalarInt | ScalarFloat,
        period_results: tuple[tuple[RegimeName, PeriodRegimeSimulationData], ...],
        memory: SimulationMemory | None = None,
        time_kind: Literal["age", "period"] = "age",
    ) -> None:
        observed.append(("diagnostics", is_root(memory=memory)))
        original_validate(
            logger=logger,
            age=age,
            period_results=period_results,
            memory=memory,
            time_kind=time_kind,
        )

    def period_age(
        *, values: jax.Array, period: int, memory: SimulationMemory | None
    ) -> jax.Array:
        if recorded:
            observed.append(("next_setup", is_root(memory=memory)))
        return original_period_age(values=values, period=period, memory=memory)

    monkeypatch.setattr(simulate, "_validate_period_values", validate)
    monkeypatch.setattr(chunk_operations, "period_age", period_age)
    _simulate_budgeted(model=_budgeted_model())
    assert {phase for phase, _ in observed} == {"diagnostics", "next_setup"}
    assert all(is_live for _, is_live in observed)
