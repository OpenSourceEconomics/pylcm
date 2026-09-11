"""A complete outer profile covers the actual allocation stages and publications."""

import dataclasses
import importlib
from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.numpy as jnp
import pytest

import _lcm.simulation.simulate as simulation
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.runtime import SimulationRuntime
from tests.simulation.test_abstract_simulation_profiles import (
    _ConcreteAllocationError,
    _forbid_allocation,
)
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


@pytest.mark.parametrize("width", [3, 2])
def test_complete_profile_covers_actual_stages_and_retained_records_without_allocating(
    *,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
) -> None:
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    stages: set[str] = set()
    runtimes: list[SimulationRuntime] = []
    host_dispatch = ProfiledSimulationOperations.dispatch
    core_dispatch = SimulationRuntime.dispatch
    chunk_dispatch = simulation._simulate_subject_chunk
    inside_chunk = [False]

    def observe_chunk(**call: Any) -> object:
        inside_chunk[0] = True
        try:
            return chunk_dispatch(**call)
        finally:
            inside_chunk[0] = False

    def observe_host(self: ProfiledSimulationOperations, **call: Any) -> object:
        result = host_dispatch(self, **call)
        if inside_chunk[0]:
            stages.add(call["function"].__name__)
        return result

    def observe_core(self: SimulationRuntime, **call: Any) -> object:
        if not runtimes:
            runtimes.append(self)
        return core_dispatch(self, **call)

    with monkeypatch.context() as capture:
        capture.setattr(ProfiledSimulationOperations, "dispatch", observe_host)
        capture.setattr(SimulationRuntime, "dispatch", observe_core)
        capture.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
        )
    assert "_generate_windowed_simulation_keys" in stages
    assert "_advance_states_for_subjects" in stages
    assert "_lookup_values_from_indices" in stages
    record_bytes = sum(
        leaf.nbytes
        for period_data in result.raw_results.values()
        for record in period_data.values()
        for leaf in jax.tree.leaves(
            tuple(getattr(record, field.name) for field in dataclasses.fields(record))
        )
    )
    spaces = MappingProxyType(
        {
            name: regime.solution.state_action_space(
                regime_params=result.flat_params[name]
            )
            for name, regime in model._regimes.items()
        }
    )
    module = importlib.import_module("_lcm.simulation.chunk_profiles")
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(_ConcreteAllocationError):
            jnp.zeros(3)
        profile = module.profile_simulation_chunk(
            runtime=runtimes[0],
            regimes=model._regimes,
            flat_params=result.flat_params,
            base_spaces=spaces,
            values=result.period_to_regime_to_V_arr,
            ages=model.ages,
            initial_conditions=initial,
            regime_names_to_ids=model.regime_names_to_ids,
            n_subjects=width,
            population=3,
            original_population=3,
            widths={"subject": width},
            independent_taste=False,
            log_level="off",
        )
    assert stages <= {stage.name for stage in profile.stages}
    assert sum(profile.output_reservation.values()) >= record_bytes
    assert profile.n_subjects == width
    assert profile.padded_population == -(-3 // width) * width
    if width == 2:
        assert {
            "_pad_initial_leaf",
            "_slice_population",
            "_concatenate_arrays",
            "_slice_array",
        } <= {stage.name for stage in profile.stages}
    assert all(stage.peak_bytes > 0 for stage in profile.stages)
