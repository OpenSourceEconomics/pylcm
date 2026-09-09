"""Each forward dispatch fits actual live buffers, even after compilation is cached."""

import dataclasses
import gc
import logging
import weakref
from collections.abc import Callable, Mapping
from functools import partial, partialmethod
from types import MappingProxyType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.compile as compile_module
import _lcm.simulation.runtime as runtime_module
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    MaterializedCoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.residency import DeviceBufferFootprint, measure_buffer_footprint
from _lcm.simulation.runtime import CompiledSimulationProgram, SimulationRuntime
from _lcm.simulation.unit_executor import SimulationUnitExecutor
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from tests.simulation.test_program_runtime import _program


@dataclasses.dataclass(frozen=True, kw_only=True)
class _LiveArrays:
    """The outer owner retains these arrays; snapshots contain no array references."""

    arrays: list[jax.Array]

    def __call__(self) -> DeviceBufferFootprint:
        jax.block_until_ready(self.arrays)
        return measure_buffer_footprint(tree=self.arrays)


def _runtime(*, budget: int, enable_jit: bool = True) -> SimulationRuntime:
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=budget,
        ),
        enable_jit=enable_jit,
        subject_devices=(jax.devices()[0],),
    )


def _controlled_width_output(*, state: jax.Array, width: int) -> jax.Array:
    """Make a selected test width visible through a genuine compiled output."""
    return jnp.full_like(state, width)


# keyword-only-exempt: library-callback=functools.partialmethod
def _controlled_memory_analysis(
    self: jax.stages.Compiled,
    *,
    original: Callable[..., object],
    executable_widths: dict[int, int],
) -> object:
    width = executable_widths.get(id(self))
    if width is None:
        return original(self)
    return SimpleNamespace(peak_memory_in_bytes={4: 80, 2: 67, 1: 40}[width])


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_controlled_execution(
    self: jax.stages.Compiled,
    *args: Any,
    original: Callable[..., object],
    executable_widths: dict[int, int],
    executions: list[int],
    **kwargs: Any,
) -> object:
    width = executable_widths.get(id(self))
    if width is not None:
        executions.append(width)
    return original(self, *args, **kwargs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ControlledCompiler:
    """Compile one controlled report per actual width requested by the planner."""

    program: MaterializedCoreProgram
    enable_jit: bool
    subject_width: int
    abstract_inputs: bool
    compilations: list[int]
    executable_widths: dict[int, int]

    def __call__(self, widths: Mapping[str, int]) -> CompiledSimulationProgram:
        width = widths["subject"]
        self.compilations.append(width)
        # The controlled peak includes this still-owned shape-only state operand.
        # A real Compiled object publishes its actual kept input-sharding tree.
        compiled = (
            jax.jit(partial(_controlled_width_output, width=width), keep_unused=True)
            .lower(state=self.program.arguments["state"])
            .compile()
        )
        self.executable_widths[id(compiled)] = width
        return CompiledSimulationProgram(
            executable=compiled,
            static_kwargs=MappingProxyType({}),
        )


def test_cached_candidates_are_rechecked_after_retained_outputs_grow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wider cached code cannot reuse an admission made before outputs accumulated."""
    compilations: list[int] = []
    executions: list[int] = []
    executable_widths: dict[int, int] = {}
    monkeypatch.setattr(
        runtime_module,
        "_SimulationCandidateCompiler",
        partial(
            _ControlledCompiler,
            compilations=compilations,
            executable_widths=executable_widths,
        ),
    )
    monkeypatch.setattr(
        jax.stages.Compiled,
        "memory_analysis",
        partialmethod(
            _controlled_memory_analysis,
            original=jax.stages.Compiled.memory_analysis,
            executable_widths=executable_widths,
        ),
    )
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _observe_controlled_execution,
            original=jax.stages.Compiled.__call__,
            executable_widths=executable_widths,
            executions=executions,
        ),
    )
    runtime = _runtime(budget=100)
    program = _program()
    state = jnp.arange(4, dtype=jnp.uint8)
    provider = _LiveArrays(arrays=[state, jnp.ones(10, dtype=jnp.uint8)])
    executor = SimulationUnitExecutor(
        runtime=runtime,
        live_footprint=provider,
        budget_devices=(jax.devices()[0],),
    )
    outputs = [
        executor.dispatch(
            program=program, arguments={"state": state}, period=0, n_subjects=4
        )
    ]
    provider.arrays.append(jnp.ones(20, dtype=jnp.uint8))
    outputs.append(
        executor.dispatch(
            program=program, arguments={"state": state}, period=0, n_subjects=4
        )
    )
    provider.arrays.append(jnp.ones(40, dtype=jnp.uint8))
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        executor.dispatch(
            program=program, arguments={"state": state}, period=0, n_subjects=4
        )
    # The first four-byte output is decisive: without the unit's output metadata,
    # external=30 would admit width 2 (peak 67), whereas actual external=34 does not.
    assert executions == [4, 1]
    assert compilations == [4, 2, 1]
    np.testing.assert_array_equal(outputs[0], np.full(4, 4))
    np.testing.assert_array_equal(outputs[1], np.full(4, 1))
    assert len(runtime.cache) == 3
    provider_ref = weakref.ref(provider)
    output_refs = [weakref.ref(output) for output in outputs]
    del outputs, executor, provider
    gc.collect()
    assert provider_ref() is None
    assert all(reference() is None for reference in output_refs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ObservedPlanner:
    original: Callable[..., Any]
    observations: list[tuple[int, int | None]]

    def __call__(self, **kwargs: Any) -> Any:
        plan = self.original(**kwargs)
        self.observations.append((kwargs["resident_bytes"], plan.peak_bytes))
        return plan


def test_real_compiler_peak_excludes_only_actual_argument_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real XLA report and a retained unrelated array both reach budget planning."""
    observations: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        runtime_module,
        "plan_workspace",
        _ObservedPlanner(
            original=runtime_module.plan_workspace, observations=observations
        ),
    )
    runtime = _runtime(budget=1_000_000)
    state = jnp.arange(4.0)
    retained = jnp.ones(32)
    executor = SimulationUnitExecutor(
        runtime=runtime,
        live_footprint=_LiveArrays(arrays=[state, retained]),
        budget_devices=(jax.devices()[0],),
    )
    result = executor.dispatch(
        program=_program(), arguments={"state": state}, period=0, n_subjects=4
    )
    compiled = next(iter(runtime.cache.values()))
    actual_peak = compiler_peak_bytes(
        compiled=compiled.executable, widths={"subject": 4}
    )
    assert actual_peak > 0
    assert observations == [(retained.nbytes, actual_peak)]
    np.testing.assert_array_equal(result, np.arange(4.0) + 1)


def test_unit_owns_raw_outputs_until_explicit_close() -> None:
    """A host replacement cannot invalidate the unit's intermediate buffer metadata."""
    state = jnp.arange(4.0)
    executor = SimulationUnitExecutor(
        runtime=_runtime(budget=1_000_000),
        live_footprint=_LiveArrays(arrays=[state]),
        budget_devices=(jax.devices()[0],),
    )
    result = executor.dispatch(
        program=_program(), arguments={"state": state}, period=0, n_subjects=4
    )
    reference = weakref.ref(result)
    del result
    gc.collect()
    assert reference() is not None, "Raw output lost its owner before unit close"
    executor.close()
    gc.collect()
    assert reference() is None
    with pytest.raises(ExecutionPlanningError, match="closed"):
        executor.dispatch(
            program=_program(), arguments={"state": state}, period=0, n_subjects=4
        )


def test_budgeted_dispatch_requires_a_call_scoped_live_inventory() -> None:
    """A persistent runtime cannot authorize execution from argument-only residency."""
    with pytest.raises(ExecutionPlanningError, match=r"live.*context"):
        _runtime(budget=1_000_000).dispatch(
            program=_program(),
            arguments={"state": jnp.arange(4.0)},
            period=0,
            n_subjects=4,
        )


def test_ready_unit_outputs_reach_the_host_inventory() -> None:
    """Host adapters see every raw core output through the same explicit handoff."""
    state = jnp.arange(4.0)
    outputs: list[object] = []
    runtime = _runtime(budget=1_000_000)
    executor = SimulationUnitExecutor(
        runtime=runtime,
        live_footprint=_LiveArrays(arrays=[state]),
        budget_devices=(jax.devices()[0],),
        on_output=outputs.append,
    )
    result = executor.dispatch(
        program=_program(), arguments={"state": state}, period=0, n_subjects=4
    )
    assert len(outputs) == 1, "The host owner did not receive the raw core output"
    assert outputs[0] is result
    assert all(leaf.is_ready() for leaf in jax.tree.leaves(outputs))
    reference = weakref.ref(result)
    executor.close()
    del executor, result
    outputs.clear()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("adapter", ["eager", "host"])
def test_budgeted_unprofiled_adapters_refuse_before_dispatch(*, adapter: str) -> None:
    """An explicit inventory does not supply an unmeasured adapter workspace bound."""
    state = jnp.arange(4.0)
    program = _program()
    if adapter == "host":
        program = dataclasses.replace(
            program,
            disposition=CoreExecutionDisposition.HOST_DRIVEN,
            requirements=dataclasses.replace(program.requirements, tiled_axes=()),
        )
    runtime = _runtime(budget=1_000_000, enable_jit=adapter != "eager")
    executor = SimulationUnitExecutor(
        runtime=runtime,
        live_footprint=_LiveArrays(arrays=[state]),
        budget_devices=(jax.devices()[0],),
    )
    with pytest.raises(ExecutionPlanningError, match="no profiled workspace bound"):
        executor.dispatch(
            program=program, arguments={"state": state}, period=0, n_subjects=4
        )
    assert runtime.cache == {}


def _refuse_template_construction(**kwargs: object) -> object:
    del kwargs
    raise AssertionError("Budgeted prewarming allocated concrete templates")


def test_budgeted_prewarming_defers_before_building_templates(
    *, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An AOT hint cannot certify future output residency from zero templates."""
    model, params, _ = WITNESSES["multi_regime"](
        execution_config=ExecutionConfig(device_memory_bytes=1_000_000), n_subjects=7
    )
    monkeypatch.setattr(
        compile_module,
        "_get_regime_V_shapes_and_shardings",
        _refuse_template_construction,
    )
    with caplog.at_level(logging.INFO):
        model._ensure_simulate_compiled(
            compile_batch_size=7,
            flat_params=model._process_params(params),
            max_compilation_workers=2,
            log=logging.getLogger("budgeted-prewarm"),
        )
    assert "live residency" in caplog.text
