"""Cached code rechecks physical ownership of compiler-eliminated arguments."""

from collections.abc import Callable, Mapping
from functools import partialmethod
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationDispatchContext, SimulationRuntime
from lcm.exceptions import ExecutionPlanningError


def _increment_used(*, used: jax.Array, dead: jax.Array, **_static: int) -> jax.Array:
    del dead
    return used + 1


def _program() -> CoreProgram:
    return CoreProgram(
        name="alias_sensitive_transition",
        function=_increment_used,
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="alias_sensitive_transition",
            subject_arg_names=("used", "dead"),
        ),
        requirements=CoreExecutionRequirements(),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _runtime(*, budget: int, device: jax.Device) -> SimulationRuntime:
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(device.id,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=budget,
        ),
        enable_jit=True,
        subject_devices=(device,),
    )


def _dispatch(
    *,
    runtime: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[str, jax.Array],
) -> jax.Array:
    result = runtime.dispatch(
        program=program,
        arguments=arguments,
        period=0,
        n_subjects=arguments["used"].size,
        residency=SimulationDispatchContext(
            live_footprint=lambda: measure_buffer_footprint(tree=arguments),
            budget_devices=runtime.subject_devices,
        ),
    )
    assert isinstance(result, jax.Array)
    return result.block_until_ready()


def _payload_bytes(*, tree: object, device: jax.Device) -> int:
    return sum(
        stop - start
        for start, stop in measure_buffer_footprint(tree=tree).spans.get(device, ())
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _refuse_cached_execution(
    self: jax.stages.Compiled,
    *args: Any,
    original: Callable[..., object],
    forbidden: jax.stages.Compiled,
    attempted: list[bool],
    **kwargs: Any,
) -> object:
    if self is forbidden:
        attempted.append(True)
        raise AssertionError("Cached executable ran without room for its dead input")
    return original(self, *args, **kwargs)


def test_cached_runtime_counts_a_new_dead_owner_but_not_an_input_alias(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same executable admits aliases and refuses a new same-shape live owner."""
    device = jax.devices()[0]
    expected_source = np.arange(262_144, dtype=np.int32)
    source = jax.device_put(expected_source, device).block_until_ready()
    aliased = {"used": source, "dead": source}
    program = _program()
    generous = _runtime(budget=8 * source.nbytes, device=device)
    output = _dispatch(runtime=generous, program=program, arguments=aliased)
    assert output.sharding == source.sharding
    assert _payload_bytes(tree=(source, output), device=device) == 2 * source.nbytes
    np.testing.assert_array_equal(output, expected_source + 1)
    (profile,) = generous.cache.values()
    assert isinstance(profile.executable, jax.stages.Compiled)
    analysis = profile.executable.memory_analysis()
    assert analysis is not None
    assert analysis.argument_size_in_bytes == source.nbytes
    budget = compiler_peak_bytes(compiled=profile.executable, widths={})
    del output

    runtime = _runtime(budget=budget, device=device)
    output = _dispatch(runtime=runtime, program=program, arguments=aliased)
    np.testing.assert_array_equal(output, expected_source + 1)
    (selected,) = runtime.cache.values()
    assert isinstance(selected.executable, jax.stages.Compiled)
    del output

    # Allocate the distinct owner only after the first tight admission succeeds.
    dead = jax.device_put(np.full(source.shape, -17, dtype=np.int32), device)
    dead.block_until_ready()
    distinct = {"used": source, "dead": dead}
    assert _payload_bytes(tree=distinct, device=device) == 2 * source.nbytes
    assert _payload_bytes(tree=distinct, device=device) <= budget
    assert 3 * source.nbytes > budget
    output = _dispatch(runtime=generous, program=program, arguments=distinct)
    assert next(iter(generous.cache.values())) is profile
    assert output.sharding == source.sharding
    assert _payload_bytes(tree=(distinct, output), device=device) == 3 * source.nbytes
    np.testing.assert_array_equal(output, expected_source + 1)
    del output

    attempted: list[bool] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _refuse_cached_execution,
            original=jax.stages.Compiled.__call__,
            forbidden=selected.executable,
            attempted=attempted,
        ),
    )
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        _dispatch(runtime=runtime, program=program, arguments=distinct)
    assert attempted == []
    assert len(runtime.cache) == 1
    assert next(iter(runtime.cache.values())) is selected
    np.testing.assert_array_equal(source, expected_source)
    np.testing.assert_array_equal(dead, np.full(source.shape, -17, dtype=np.int32))
