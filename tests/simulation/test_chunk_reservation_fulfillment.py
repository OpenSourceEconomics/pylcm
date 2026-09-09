"""Only actual completed reservation owners can discharge future setup storage."""

from types import MappingProxyType

import jax
import numpy as np
import pytest

from _lcm.simulation.chunk_admission import PreparedSimulationChunks
from _lcm.simulation.chunk_inputs import SimulationCallInputs
from _lcm.simulation.chunk_planning import (
    SimulationChunkPlan,
    SimulationChunkProfile,
    SimulationStageProfile,
)
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint, union_buffer_footprints
from lcm.exceptions import ExecutionPlanningError


def _copy_output(*, source: jax.Array) -> jax.Array:
    return source + 1


def test_unrelated_retained_input_does_not_fulfill_future_setup() -> None:
    device = jax.devices()[0]
    source = jax.device_put(np.arange(32, dtype=np.int32), device)
    compiled = jax.jit(_copy_output).lower(source=source).compile()
    stage = SimulationStageProfile(name="copy", executable=compiled, devices=(device,))
    profile = SimulationChunkProfile(
        n_subjects=32,
        padded_population=32,
        stages=(stage,),
        fixed_reservation={device: source.nbytes},
        setup_reservation={device: source.nbytes},
        output_reservation={},
    )
    original = measure_buffer_footprint(tree=source)
    budget = source.nbytes + source.nbytes + stage.peak_bytes
    prepared = PreparedSimulationChunks(
        plan=SimulationChunkPlan(profile=profile, required_bytes={device: budget}),
        call_inputs=SimulationCallInputs(
            devices=(device,),
            flat_params=MappingProxyType({}),
            base_state_action_spaces=MappingProxyType({}),
        ),
        admitted_inputs=original,
    )
    memory = SimulationMemory(
        budget_bytes=budget,
        devices=(device,),
        subject_devices=(device,),
        operations=ProfiledSimulationOperations(),
        inputs=original,
    )
    prepared.require_chunk(memory=memory)
    unrelated = jax.device_put(np.full(32, -7, dtype=np.int32), device)
    memory.inputs = union_buffer_footprints(
        footprints=(original, measure_buffer_footprint(tree=unrelated))
    )
    with pytest.raises(ExecutionPlanningError, match="no longer fits"):
        prepared.require_chunk(memory=memory)
    np.testing.assert_array_equal(unrelated, np.full(32, -7, dtype=np.int32))
