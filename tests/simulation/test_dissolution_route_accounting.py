"""A budgeted simulation counts every buffer a dissolution route holds.

Under a finite device-memory budget the simulation measures what each held
owner keeps alive. The carriers a dissolution route produces are held for the
rest of the unit, so the footprint measured through the held owner must be
exactly the footprint of its array fields, named one by one.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest

from _lcm.simulation.dissolution_route import RerouteOutcome
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.typing import FootprintTree, PytreeByPeriod
from lcm import ExecutionConfig
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
