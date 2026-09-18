"""An outer reservation fixes code even when instantaneous residency is smaller."""

from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    TiledOutputAxis,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationDispatchContext, SimulationRuntime
from lcm.exceptions import ExecutionPlanningError


def _increment(*, x: jax.Array) -> jax.Array:
    return x + 1


def _fixture(
    *, fixed: int | None = None
) -> tuple[SimulationRuntime, CoreProgram, jax.Array]:
    device = jax.devices()[0]
    source = jax.device_put(np.arange(8, dtype=np.int32), device)
    program = CoreProgram(
        name="reserved_subjects",
        function=_SubjectTiled(func=_increment, subject_arg_names=("x",)),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="reserved_subjects", subject_arg_names=("x",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(
                TiledOutputAxis(
                    name="subject",
                    state_names=("x",),
                    extent=8,
                    width_keyword=SUBJECT_WIDTH_KEYWORD,
                ),
            )
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )
    runtime = SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(device.id,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({} if fixed is None else {"subject": fixed}),
            device_memory_bytes=2**20,
        ),
        enable_jit=True,
        subject_devices=(device,),
    )
    return runtime, program, source


def test_actual_dispatch_preserves_the_outer_reserved_width(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, program, source = _fixture()
    selected: list[int] = []
    prepare = SimulationRuntime._prepare_materialized

    def observe(self: SimulationRuntime, **call: Any) -> object:
        compiled = prepare(self, **call)
        selected.append(compiled.widths["subject"])
        return compiled

    monkeypatch.setattr(SimulationRuntime, "_prepare_materialized", observe)
    context = SimulationDispatchContext(
        live_footprint=lambda: measure_buffer_footprint(tree=source),
        budget_devices=runtime.subject_devices,
    )
    expected = runtime.dispatch(
        program=program,
        arguments={"x": source},
        period=0,
        n_subjects=8,
        residency=context,
    )
    assert selected == [8], (
        "The current-residency counterfactual must choose wider code"
    )
    caller_widths = {"subject": 2}
    reserved = SimulationDispatchContext(
        live_footprint=context.live_footprint,
        budget_devices=context.budget_devices,
        axis_widths=caller_widths,
    )
    caller_widths["subject"] = 8
    actual = runtime.dispatch(
        program=program,
        arguments={"x": source},
        period=0,
        n_subjects=8,
        residency=reserved,
    )
    assert selected == [8, 2]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, np.arange(8, dtype=np.int32) + 1)
    assert isinstance(actual, jax.Array)
    assert actual.sharding == source.sharding
    np.testing.assert_array_equal(source, np.arange(8, dtype=np.int32))


def test_reserved_width_conflicting_with_explicit_configuration_is_refused() -> None:
    runtime, program, source = _fixture(fixed=4)
    with pytest.raises(ExecutionPlanningError, match="conflict"):
        runtime.dispatch(
            program=program,
            arguments={"x": source},
            period=0,
            n_subjects=8,
            residency=SimulationDispatchContext(
                live_footprint=lambda: measure_buffer_footprint(tree=source),
                budget_devices=runtime.subject_devices,
                axis_widths={"subject": 2},
            ),
        )
    assert not runtime.cache
