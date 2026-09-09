"""Prepare and re-admit a complete call's chunks without retaining admission in code.

The selector deliberately chooses one common width per axis across the forward
programs. A smaller program clamps that choice to its declared extent. Every
candidate uses the same actual completed grids and retained solution owners.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import jax

from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.simulation.chunk_inputs import (
    SimulationCallInputs,
    prepare_simulation_call_inputs,
)
from _lcm.simulation.chunk_planning import (
    SimulationChunkPlan,
    SimulationChunkProfile,
    _required_bytes,
    plan_simulation_chunks,
)
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
    resolve_budget_devices,
    union_buffer_footprints,
)
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds
from _lcm.utils.logging import LogLevel
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError


@dataclass(frozen=True, kw_only=True)
class PreparedSimulationChunks:
    """Call-local code selection and actual shared owners, never a Model cache value."""

    plan: SimulationChunkPlan
    call_inputs: SimulationCallInputs
    admitted_inputs: DeviceBufferFootprint

    def require_chunk(
        self,
        *,
        memory: SimulationMemory,
        completed_setup: DeviceBufferFootprint | None = None,
    ) -> None:
        """Recheck before slices, retaining actual fulfilled outputs in live storage.

        Only actual newly owned setup spans and published result spans fulfill
        reservations. Shape equality and current aliases never erase future slots.
        Logical duplicate slots can remain conservatively reserved after publication.
        """
        profile = self.plan.profile
        setup = resident_bytes_by_device(
            live=DeviceBufferFootprint(spans={})
            if completed_setup is None
            else completed_setup,
            arguments=self.admitted_inputs,
            devices=memory.devices,
        )
        outputs = resident_bytes_by_device(
            live=memory.outputs,
            arguments=DeviceBufferFootprint(spans={}),
            devices=memory.devices,
        )
        remaining = replace(
            profile,
            fixed_reservation={
                device: value
                - min(
                    value,
                    profile.setup_reservation.get(device, 0),
                    setup.get(device, 0),
                )
                for device, value in profile.fixed_reservation.items()
            },
            output_reservation={
                device: value - min(value, outputs.get(device, 0))
                for device, value in profile.output_reservation.items()
            },
            setup_reservation={},
        )
        resident = resident_bytes_by_device(
            live=memory.snapshot(),
            arguments=DeviceBufferFootprint(spans={}),
            devices=memory.devices,
        )
        required = _required_bytes(
            profile=remaining, resident=resident, devices=memory.devices
        )
        if any(value > memory.budget_bytes for value in required.values()):
            raise ExecutionPlanningError(
                "The reserved simulation chunk no longer fits current retained storage."
            )


def prepare_simulation_chunks(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    values: Mapping[int, Mapping[str, jax.Array]],
    ages: AgeGrid,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    original_population: int,
    retained_footprint: DeviceBufferFootprint,
    independent_taste: bool,
    log_level: LogLevel,
    policies: Mapping[int, Mapping[str, object]] | None = None,
) -> PreparedSimulationChunks:
    """Exhaust inner choices before shrinking an outer chunk's admitted extent."""
    runtime = next(iter(regimes.values())).simulation.programs.executor
    if (
        not isinstance(runtime, SimulationRuntime)
        or runtime.execution.device_memory_bytes is None
    ):
        raise ExecutionPlanningError(
            "Chunk admission requires a budgeted simulation runtime."
        )
    if not runtime.enable_jit or any(
        regime.gated_edges
        or (
            regime.simulation.replay_route.policy_applicable
            and regime.simulation.replay_route.consumer_route != "nnbegm_finite"
        )
        or regime.simulation.external_replay_route is not None
        for regime in regimes.values()
    ):
        raise ExecutionPlanningError(
            "Budgeted chunk admission requires compiled decision programs; "
            "host replay routes need complete stage profiles."
        )
    inputs = union_buffer_footprints(
        footprints=(
            retained_footprint,
            measure_buffer_footprint(
                tree=(flat_params, initial_conditions, values, ages.values)
            ),
        )
    )
    devices = resolve_budget_devices(
        execution_devices=placed_devices_for_ids(
            submesh_device_ids=(), visible_device_ids=runtime.execution.device_ids
        ),
        live=inputs,
    )
    memory = SimulationMemory(
        budget_bytes=runtime.execution.device_memory_bytes,
        devices=devices,
        subject_devices=runtime.subject_devices,
        operations=runtime.operations,
        inputs=inputs,
    )
    memory.check_resident()
    call_inputs = prepare_simulation_call_inputs(
        flat_params=flat_params,
        regimes=regimes,
        device_ids=runtime.execution.device_ids,
        memory=memory,
    )
    memory.inputs = union_buffer_footprints(
        footprints=(
            memory.inputs,
            measure_buffer_footprint(tree=call_inputs.array_roots),
        )
    )
    memory.close_unit()
    population = initial_conditions["regime_id"].shape[0]
    alignment = len(runtime.subject_devices)
    outer = TiledOutputAxis(
        name="subject",
        state_names=("subject",),
        extent=population,
        width_keyword="__lcm_subject_width__",
        alignment=alignment,
    )
    configured = runtime.execution.axis_widths
    if "subject" in configured:
        extent = min(configured["subject"], population)
        candidates = (-(-extent // alignment) * alignment,)
    elif population == 1:
        candidates = (1,)
    else:
        candidates = tuple(
            choice["subject"]
            for choice in workspace_width_candidates(
                axes=(outer,), budget_bytes=memory.budget_bytes
            )
        )
    resident = resident_bytes_by_device(
        live=memory.inputs, arguments=DeviceBufferFootprint(spans={}), devices=devices
    )
    profiler = _ChunkProfiler(
        runtime=runtime,
        regimes=regimes,
        call_inputs=call_inputs,
        values=values,
        policies=policies,
        ages=ages,
        initial_conditions=initial_conditions,
        regime_names_to_ids=regime_names_to_ids,
        population=population,
        original_population=original_population,
        independent_taste=independent_taste,
        log_level=log_level,
        resident=resident,
        devices=devices,
    )
    plan = plan_simulation_chunks(
        candidates=candidates,
        profile_candidate=profiler,
        live=memory.inputs,
        budget_bytes=memory.budget_bytes,
        devices=devices,
    )
    return PreparedSimulationChunks(
        plan=plan, call_inputs=call_inputs, admitted_inputs=memory.inputs
    )


@dataclass(frozen=True, kw_only=True)
class _ChunkProfiler:
    """Transient search inputs, released after selecting one immutable profile."""

    runtime: SimulationRuntime
    regimes: Mapping[str, Regime]
    call_inputs: SimulationCallInputs
    values: Mapping[int, Mapping[str, jax.Array]]
    ages: AgeGrid
    initial_conditions: Mapping[str, jax.Array]
    regime_names_to_ids: RegimeNamesToIds
    population: int
    original_population: int
    independent_taste: bool
    log_level: LogLevel
    resident: Mapping[jax.Device, int]
    devices: tuple[jax.Device, ...]
    policies: Mapping[int, Mapping[str, object]] | None = None

    def __call__(self, *, n_subjects: int) -> SimulationChunkProfile:
        """Return a fitting common inner choice, or the smallest required bound."""
        budget = self.runtime.execution.device_memory_bytes
        if budget is None:
            raise ExecutionPlanningError("Chunk profiling requires a device budget.")
        axes = _common_axes(regimes=self.regimes, n_subjects=n_subjects)
        choices = workspace_width_candidates(
            axes=axes,
            fixed_widths=self.runtime.execution.axis_widths,
            budget_bytes=self.runtime.execution.device_memory_bytes,
        )
        best = None
        best_peak = math.inf
        for widths in choices:
            profile = profile_simulation_chunk(
                runtime=self.runtime,
                regimes=self.regimes,
                flat_params=self.call_inputs.flat_params,
                base_spaces=self.call_inputs.base_state_action_spaces,
                values=self.values,
                policies=self.policies,
                ages=self.ages,
                initial_conditions=self.initial_conditions,
                regime_names_to_ids=self.regime_names_to_ids,
                n_subjects=n_subjects,
                population=self.population,
                original_population=self.original_population,
                widths=widths,
                independent_taste=self.independent_taste,
                log_level=self.log_level,
            )
            peak = max(
                _required_bytes(
                    profile=profile, resident=self.resident, devices=self.devices
                ).values()
            )
            if peak <= budget:
                return profile
            if peak < best_peak:
                best, best_peak = profile, peak
        if best is None:
            raise ExecutionPlanningError(
                "No compiled simulation chunk candidate was declared."
            )
        return best


def _common_axes(
    *, regimes: Mapping[str, Regime], n_subjects: int
) -> tuple[ReducedAxis | TiledOutputAxis, ...]:
    """Build a deliberately common axis policy from actual program declarations."""
    axes: dict[str, ReducedAxis | TiledOutputAxis] = {}
    for regime in regimes.values():
        programs = regime.simulation.programs
        for family in (
            programs.policy_prepare,
            programs.decision,
            programs.transition,
            programs.route,
        ):
            for program in family.values():
                for declared in program.requirements.axes:
                    axis = (
                        replace(declared, extent=n_subjects)
                        if declared.name == "subject"
                        else declared
                    )
                    if axis.extent == 1:
                        continue
                    previous = axes.get(axis.name)
                    if previous is not None:
                        representative = (
                            previous if previous.extent >= axis.extent else axis
                        )
                        axis = replace(
                            representative,
                            minimum_width=max(
                                previous.minimum_width, axis.minimum_width
                            ),
                            alignment=math.lcm(previous.alignment, axis.alignment),
                        )
                    axes[axis.name] = axis
    return tuple(axes[name] for name in sorted(axes))
