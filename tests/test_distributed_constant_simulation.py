"""Constant simulation programs retain the selected execution devices after DCE."""

from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.chunk_planning import SimulationStageProfile
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationDispatchContext, SimulationRuntime
from lcm.typing import IntND

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


def _constant_outputs(
    *, dead: IntND, __lcm_subject_width__: int
) -> tuple[IntND, IntND]:
    """Return scalar and subject outputs without reading the input array."""
    del dead, __lcm_subject_width__
    return jnp.asarray(7, dtype=jnp.int32), jnp.full((4,), 5, dtype=jnp.int32)


@pytest.mark.parametrize("order", [(1,), (2, 1), (3, 1, 2)])
@pytest.mark.parametrize("budget", [None, 2**28])
def test_constant_program_uses_selected_devices_after_all_inputs_are_dropped(
    *, order: tuple[int, ...], budget: int | None
) -> None:
    """Abstract profiles and warm live dispatch agree on the real execution set."""
    devices = tuple(jax.devices()[index] for index in order)
    runtime = SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=order,
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=budget,
        ),
        enable_jit=True,
        subject_devices=devices,
    )
    program = CoreProgram(
        name="constant_outputs",
        function=_constant_outputs,
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="constant_outputs", subject_arg_names=("dead",)
        ),
        requirements=CoreExecutionRequirements(),
        output_roles=("scalar", "subjects"),
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )
    # The dead operand is divisible by every selected mesh. Output rank and
    # length deliberately differ; no subject-axis assumption may shard a scalar.
    layout = subject_operand_sharding(devices=devices)
    compiled = runtime.prepare_abstract(
        program=program,
        arguments={"dead": jax.ShapeDtypeStruct((12,), np.int32, sharding=layout)},
        period=0,
        n_subjects=12,
        widths=MappingProxyType({}),
    )
    assert isinstance(compiled.executable, jax.stages.Compiled)
    SimulationStageProfile(
        name="constant_outputs", executable=compiled.executable, devices=devices
    )
    assert compiled.executable.input_shardings[1]["dead"] is None
    source = jax.device_put(np.arange(12, dtype=np.int32), layout)
    for increment in (0, 10):
        arguments = {"dead": source + increment}
        residency = SimulationDispatchContext(
            live_footprint=partial(measure_buffer_footprint, tree=arguments),
            budget_devices=tuple(jax.devices()),
        )
        outputs = runtime.dispatch(
            program=program,
            arguments=arguments,
            period=0,
            n_subjects=12,
            residency=residency if budget is not None else None,
        )
        assert isinstance(outputs, tuple)
        scalar, vector = outputs
        assert scalar.devices() == vector.devices() == set(devices)
        np.testing.assert_array_equal(scalar, np.int32(7))
        np.testing.assert_array_equal(vector, np.full(4, 5, dtype=np.int32))
    assert len(runtime.cache) == 1
