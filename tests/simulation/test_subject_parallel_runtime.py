"""Use the real runtime, including abstract compilation and live admission."""

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
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import (
    CompiledSimulationProgram,
    SimulationDispatchContext,
    SimulationRuntime,
)
from lcm.exceptions import ExecutionPlanningError


def _devices() -> tuple[jax.Device, ...]:
    devices = tuple(jax.devices())
    if len(devices) < 8:
        pytest.skip("Select this test with eight visible devices.")
    return devices[:8]


def _increment(*, state: jax.Array) -> jax.Array:
    return state + 1


def _program() -> CoreProgram:
    return CoreProgram(
        name="subject_partition_transition",
        function=_SubjectTiled(func=_increment, subject_arg_names=("state",)),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="subject_partition_transition", subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),)
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _runtime(
    *, devices: tuple[jax.Device, ...], budget: int | None
) -> SimulationRuntime:
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=tuple(device.id for device in devices),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({"subject": 3}),
            device_memory_bytes=budget,
            simulation_sharding="subjects",
        ),
        enable_jit=True,
        subject_devices=devices,
    )


def test_real_runtime_abstract_profile_and_repeated_dispatch() -> None:
    devices = _devices()
    mesh = jax.make_mesh((8,), ("X",), (jax.sharding.AxisType.Auto,), devices=devices)
    layout = jax.NamedSharding(mesh, jax.P("X"))
    runtime = _runtime(devices=devices, budget=None)
    program = _program()
    runtime.prepare_abstract(
        program=program,
        arguments={"state": jax.ShapeDtypeStruct((16,), jnp.float32, sharding=layout)},
        period=0,
        n_subjects=16,
        widths={"subject": 3},
    )
    for offset in (0, 10):
        source = jax.device_put(np.arange(16, dtype=np.float32) + offset, layout)
        result = runtime.dispatch(
            program=program, arguments={"state": source}, period=0, n_subjects=16
        )
        assert isinstance(result, jax.Array)
        np.testing.assert_array_equal(result, np.arange(16) + offset + 1)
        assert [shard.data.shape for shard in result.addressable_shards] == [(2,)] * 8
        assert not source.is_deleted()
    assert len(runtime.cache) == 1


def test_last_device_retained_owner_refuses_before_numerical_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = _devices()
    runtime = _runtime(devices=devices, budget=1024)
    retained = jax.device_put(np.ones(1024, dtype=np.float32), devices[-1])
    source = jax.device_put(np.arange(16, dtype=np.float32), devices[0])
    calls = []
    original = CompiledSimulationProgram.__call__

    def observe(self: CompiledSimulationProgram, **arguments: object) -> object:
        calls.append(None)
        return original(self, **arguments)

    monkeypatch.setattr(CompiledSimulationProgram, "__call__", observe)
    with pytest.raises(ExecutionPlanningError):
        runtime.dispatch(
            program=_program(),
            arguments={"state": source},
            period=0,
            n_subjects=16,
            residency=SimulationDispatchContext(
                live_footprint=lambda: measure_buffer_footprint(
                    tree=(retained, source)
                ),
                budget_devices=devices,
                axis_widths={"subject": 3},
            ),
        )
    assert calls == []
    assert not source.is_deleted()
    assert not retained.is_deleted()
    np.testing.assert_array_equal(retained, np.ones(1024))
