"""Retained shape-only inputs remain charged when compiled arithmetic drops them."""

import logging
from collections.abc import Callable, Hashable, Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.footprint import (
    ScheduledUnit,
    per_device_footprint,
    plan_resident_inventory,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import VALUE, resolve_output_layout
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import (
    CompiledSimulationProgram,
    SimulationDispatchContext,
    SimulationRuntime,
)
from _lcm.solution import backward_induction
from lcm import AgeGrid
from lcm.exceptions import ExecutionPlanningError

_PAYLOAD_BYTES = 1024 * 1024
_INSUFFICIENT_BYTES = 3 * _PAYLOAD_BYTES // 2
_GENEROUS_BYTES = 4 * _PAYLOAD_BYTES


def _shape_only_simulation(*, state: jax.Array, **_static: int) -> jax.Array:
    """The declared argument's shape matters; its contents do not."""
    return jnp.arange(state.size, dtype=state.dtype).reshape(state.shape)


def _shape_only_solve(*, next_regime_to_V_arr: Mapping[str, jax.Array]) -> jax.Array:
    """A real declared local value read supplies only shape and dtype."""
    value = next_regime_to_V_arr["done"]
    return jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)


def _retained_source() -> jax.Array:
    """Commit one MiB on the requested device before the allocation is admitted."""
    dtype = jnp.zeros(()).dtype
    source = jax.device_put(
        np.full(_PAYLOAD_BYTES // dtype.itemsize, -3, dtype=dtype),
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    source.block_until_ready()
    assert source.nbytes == _PAYLOAD_BYTES
    return source


def _assert_exact_distinct_output(*, source: jax.Array, output: jax.Array) -> None:
    """A generous admission preserves both allocations, values and placement."""
    output.block_until_ready()
    assert source.unsafe_buffer_pointer() != output.unsafe_buffer_pointer()
    assert output.sharding == source.sharding
    assert not source.is_deleted()
    np.testing.assert_array_equal(np.asarray(source), np.full(source.shape, -3))
    np.testing.assert_array_equal(np.asarray(output), np.arange(source.size))
    spans = measure_buffer_footprint(tree=(source, output)).spans[jax.devices()[0]]
    assert sum(stop - start for start, stop in spans) == 2 * _PAYLOAD_BYTES


def _simulation_runtime(*, budget: int) -> SimulationRuntime:
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(jax.devices()[0].id,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=budget,
        ),
        enable_jit=True,
        subject_devices=(jax.devices()[0],),
    )


def test_simulation_refuses_a_shape_only_live_input_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live argument subtraction must agree with the actual compiler inputs."""
    source = _retained_source()
    program = CoreProgram(
        name="shape_only_transition",
        function=_shape_only_simulation,
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="shape_only_transition", subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )
    context = SimulationDispatchContext(
        live_footprint=lambda: measure_buffer_footprint(tree=source),
        budget_devices=(jax.devices()[0],),
    )
    generous = _simulation_runtime(budget=_GENEROUS_BYTES)
    output = generous.dispatch(
        program=program,
        arguments={"state": source},
        period=0,
        n_subjects=source.size,
        residency=context,
    )
    assert isinstance(output, jax.Array)
    _assert_exact_distinct_output(source=source, output=output)
    del output

    # Planning may compile, but refusal must precede every executable dispatch.
    executions: list[None] = []
    original = CompiledSimulationProgram.__call__

    def observe(self: CompiledSimulationProgram, **arguments: object) -> object:
        executions.append(None)
        return original(self, **arguments)

    monkeypatch.setattr(CompiledSimulationProgram, "__call__", observe)
    insufficient = _simulation_runtime(budget=_INSUFFICIENT_BYTES)
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        insufficient.dispatch(
            program=program,
            arguments={"state": source},
            period=0,
            n_subjects=source.size,
            residency=context,
        )
    assert executions == []
    np.testing.assert_array_equal(np.asarray(source), np.full(source.shape, -3))


def _compile_solve_read(
    source: jax.Array,
) -> tuple[jax.stages.Compiled, Callable[[jax.stages.Compiled], int]]:
    """Use the solve owner's actual local-read exclusion, schedule and lowerer."""
    address = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="done"
    )
    read = ValueRead(
        target=address,
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="acting",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("done",),
        ),
    )
    resolved = ResolvedCoreProgram(
        name="main",
        function=_shape_only_solve,
        arguments={"next_regime_to_V_arr": {"done": source}},
        static_kwargs={},
        requirements=CoreExecutionRequirements(value_reads=(read,)),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths={},
        specialization_key=(),
        input_transfer_plan=(),
    )
    metadata = backward_induction._ProgramExecutionMetadata(
        requirements=resolved.requirements,
        disposition=resolved.disposition,
        scope=resolved.scope,
        input_transfer_plan=(),
    )
    consumed = backward_induction._aligned_input_artifacts(metadata=metadata)
    assert consumed == (address,)
    device_ids = (jax.devices()[0].id,)
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "done"): (), (0, "acting"): consumed},
        retained_artifacts=(address,),
    )
    producer = ScheduledUnit(
        period=1,
        regime="done",
        device_ids=device_ids,
        produces=(address,),
        consumes=(),
        output_bytes_per_device=_PAYLOAD_BYTES,
    )
    consumer = ScheduledUnit(
        period=0,
        regime="acting",
        device_ids=device_ids,
        produces=(),
        consumes=consumed,
        output_bytes_per_device=_PAYLOAD_BYTES,
    )
    inventory = plan_resident_inventory(
        waves_by_period={1: ((producer,),), 0: ((consumer,),)},
        fold_dispatches={},
        ledger=ledger,
        footprints={address: per_device_footprint(array=source)},
    )[(0, "acting")]
    assert inventory.resident_bytes() == 0
    triple = ("acting", 0, "main")
    candidate = (triple, ())
    compiled: dict[Hashable, jax.stages.Compiled] = {}
    backward_induction._lower_and_compile_wave(
        new_lowerings={"shape-only": candidate},
        resolved_programs={candidate: resolved},
        all_layouts={
            triple: resolve_output_layout(
                core_key="main",
                value_template=source,
                state_order=("wealth",),
                output_roles=VALUE,
            )
        },
        internal_templates={candidate: {}},
        donations={candidate: ()},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        n_triples_per_lowering={"shape-only": 1},
        log_kernel_memory=False,
        n_workers=1,
        logger=logging.getLogger("unused-input-test"),
        compiled=compiled,
        labels={},
    )

    def resident(executable: jax.stages.Compiled) -> int:
        return backward_induction._candidate_resident_bytes(
            compiled=executable,
            program=resolved,
            internal_arguments={},
            inventory=inventory,
        )

    return compiled["shape-only"], resident


def test_solve_refuses_a_declared_shape_only_read_over_budget() -> None:
    """An aligned retained read cannot disappear from peak plus resident bytes."""
    source = _retained_source()
    executable, resident = _compile_solve_read(source)

    def compiler(_widths: Mapping[str, int]) -> jax.stages.Compiled:
        return executable

    generous = plan_workspace(
        axes=(),
        compile_candidate=compiler,
        budget_bytes=_GENEROUS_BYTES,
        resident_bytes_for=resident,
    )
    output = generous.compiled(next_regime_to_V_arr={"done": source})
    _assert_exact_distinct_output(source=source, output=output)
    del output

    executions: list[None] = []

    def dispatch_if_admitted() -> None:
        insufficient = plan_workspace(
            axes=(),
            compile_candidate=compiler,
            budget_bytes=_INSUFFICIENT_BYTES,
            resident_bytes_for=resident,
        )
        executions.append(None)
        insufficient.compiled(next_regime_to_V_arr={"done": source})

    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        dispatch_if_admitted()
    assert executions == []
    np.testing.assert_array_equal(np.asarray(source), np.full(source.shape, -3))
