"""Prepare and re-admit a complete call's chunks without retaining admission in code.

The selector deliberately chooses one common width per axis across the forward
programs. A smaller program clamps that choice to its declared extent. Every
candidate uses the same actual completed grids and retained solution owners.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from types import MappingProxyType

import jax

from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.simulation.chunk_inputs import (
    SimulationCallInputs,
    prepare_simulation_call_inputs,
)
from _lcm.simulation.chunk_planning import (
    ChunkCandidateReceipt,
    ChunkDeviceReceipt,
    IndependentChunkReceipt,
    SimulationChunkPlan,
    SimulationChunkProfile,
    _required_bytes,
)
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.programs import gated_simulation_programs_ready
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
    flags: Mapping[int, Mapping[str, jax.Array]],
    ages: AgeGrid,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    original_population: int,
    retained_footprint: DeviceBufferFootprint,
    independent_taste: bool,
    log_level: LogLevel,
    policies: Mapping[int, Mapping[str, object]] | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> PreparedSimulationChunks:
    """Select a complete admitted outer cohort with the top-first planner."""
    runtime = next(iter(regimes.values())).simulation.programs.executor
    if (
        not isinstance(runtime, SimulationRuntime)
        or runtime.execution.device_memory_bytes is None
    ):
        raise ExecutionPlanningError(
            "Chunk admission requires a budgeted simulation runtime."
        )
    if not runtime.enable_jit or any(
        (
            (regime.gated_edges and not gated_simulation_programs_ready(regime=regime))
            or (
                regime.simulation.replay_route.policy_applicable
                and regime.simulation.replay_route.consumer_route != "nnbegm_finite"
            )
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
                tree=(flat_params, initial_conditions, values, flags, ages.values)
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
        process_grid_resolver=process_grid_resolver,
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
    resident = resident_bytes_by_device(
        live=memory.inputs, arguments=DeviceBufferFootprint(spans={}), devices=devices
    )
    profiler = _ChunkProfiler(
        runtime=runtime,
        regimes=regimes,
        call_inputs=call_inputs,
        values=values,
        flags=flags,
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
    plan = _plan_independent_chunks(profiler=profiler, alignment=alignment)
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
    flags: Mapping[int, Mapping[str, jax.Array]]
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
            profile = self.profile_widths(n_subjects=n_subjects, widths=widths)
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

    def profile_widths(
        self, *, n_subjects: int, widths: Mapping[str, int]
    ) -> SimulationChunkProfile:
        """Profile exactly one whole inner map without reopening its search."""
        return profile_simulation_chunk(
            runtime=self.runtime,
            regimes=self.regimes,
            flat_params=self.call_inputs.flat_params,
            base_spaces=self.call_inputs.base_state_action_spaces,
            values=self.values,
            flags=self.flags,
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


def _resolve_subject_anchor_width(
    *,
    configured: Mapping[str, int],
    population: int,
    alignment: int,
    budget: int,
) -> int:
    """Return the pinned subject width, or auto-resolve one (AUTO_ANCHOR).

    A pinned width is clamped to the population and aligned to the device
    count. Without a pin, the existing single-axis workspace search proposes
    the widest representable subject width the budget admits as the frontier's
    starting anchor, exactly as it always has for an unpinned budgeted call.
    A unit population needs no search. This anchor only seeds the frontier and
    the fallback width map; the outer-cohort planner still profiles the actual
    admitted extent before accepting it.
    """
    if "subject" in configured:
        extent = min(configured["subject"], population)
        return -(-extent // alignment) * alignment
    if population == 1:
        return 1
    outer = TiledOutputAxis(
        name="subject",
        state_names=("subject",),
        extent=population,
        width_keyword="__lcm_subject_width__",
        alignment=alignment,
    )
    return workspace_width_candidates(axes=(outer,), budget_bytes=budget)[0]["subject"]


def _independent_outer_candidates(
    *, population: int, alignment: int, subject_width: int
) -> tuple[int, ...]:
    """Return the doubling integer frontier, deduplicated in anchor-first order.

    The anchor is the inner subject width; each further candidate doubles the
    previous one until a candidate covers the whole population, and every
    candidate is rounded up to the device alignment. The frontier is finite and
    ordered geometrically; admission chooses the order in which to profile it.
    """
    candidates: dict[int, None] = {}
    scale = 1
    while True:
        extent = min(scale * subject_width, population)
        candidates[-(-extent // alignment) * alignment] = None
        if extent >= population:
            return tuple(candidates)
        scale *= 2


def _independent_anchor_widths(
    *, axes: tuple[ReducedAxis | TiledOutputAxis, ...], configured: Mapping[str, int]
) -> tuple[Mapping[str, int], ...]:
    """Generate two maps and retain pins absent from a scalar anchor's axes.

    A subject pin remains relevant when the larger candidate introduces that axis.
    Existing axes keep the enumerator's admissible clamp, without inventing axes
    for scalar programs or reopening the map at the larger extent.
    """
    full = {axis.name: axis.extent for axis in axes} | dict(configured)
    choices = tuple(
        MappingProxyType(
            dict(configured)
            | dict(
                workspace_width_candidates(
                    axes=axes, fixed_widths=pins, budget_bytes=None
                )[0]
            )
        )
        for pins in (full, configured)
    )
    return choices[:1] if choices[0] == choices[1] else choices


def _plan_independent_chunks(
    *, profiler: _ChunkProfiler, alignment: int
) -> SimulationChunkPlan:
    """Try the largest extent first, with exact profiles and anchor-derived widths.

    A successful first profile avoids compiling the smaller shapes. After refusal,
    select the existing full/bootstrap map at the anchor, freeze it, and descend
    through the remaining frontier. No fit or refusal is extrapolated between shapes.
    """
    started = perf_counter()
    budget = profiler.runtime.execution.device_memory_bytes
    if budget is None:
        raise ExecutionPlanningError("Independent chunk admission needs a budget.")
    configured = profiler.runtime.execution.axis_widths
    subject_width = _resolve_subject_anchor_width(
        configured=configured,
        population=profiler.population,
        alignment=alignment,
        budget=budget,
    )
    candidates = _independent_outer_candidates(
        population=profiler.population, alignment=alignment, subject_width=subject_width
    )
    anchor = candidates[0]
    largest = candidates[-1]
    axes = _common_axes(regimes=profiler.regimes, n_subjects=anchor)
    choices = _independent_anchor_widths(axes=axes, configured=configured)
    attempts: list[ChunkCandidateReceipt] = []
    selected = _profile_independent_candidate(
        profiler=profiler, n_subjects=largest, widths=choices[0], attempts=attempts
    )
    reason = "largest candidate admitted; smaller profiles skipped"
    map_reason = "preferred anchor-derived map admitted at largest candidate"
    if selected is None:
        selected, reason, map_reason = _admit_anchor_then_descend(
            profiler=profiler,
            candidates=candidates,
            choices=choices,
            attempts=attempts,
            budget=budget,
        )
    return replace(
        selected,
        receipt=IndependentChunkReceipt(
            original_population=profiler.original_population,
            entry_population=profiler.population,
            alignment=alignment,
            subject_width=subject_width,
            candidates=candidates,
            attempts=tuple(attempts),
            selected_subjects=selected.profile.n_subjects,
            axis_widths=tuple(sorted(selected.profile.axis_widths.items())),
            stopping_reason=reason,
            anchor_map_reason=map_reason,
            planning_seconds=perf_counter() - started,
            frontier_version=3,
        ),
    )


def _admit_anchor_then_descend(
    *,
    profiler: _ChunkProfiler,
    candidates: tuple[int, ...],
    choices: tuple[Mapping[str, int], ...],
    attempts: list[ChunkCandidateReceipt],
    budget: int,
) -> tuple[SimulationChunkPlan, str, str]:
    """Admit an anchor map after a largest-candidate refusal, then descend.

    The full map is tried at the anchor first, then a distinct bootstrap map. The
    admitted map is frozen while the remaining frontier is profiled largest first;
    the pair already refused at the top is not repeated. Returns the selected plan
    with its stopping and anchor-map reasons.
    """
    anchor = candidates[0]
    largest = candidates[-1]
    map_reason = "full/bootstrap maps identical; duplicate suppressed"
    if len(choices) > 1:
        map_reason = "bootstrap skipped after full anchor admitted"
    selected = None
    for index, widths in enumerate(choices):
        # When alignment collapses the frontier, this exact pair just refused.
        if largest == anchor and index == 0:
            continue
        if index:
            map_reason = "full anchor rejected; bootstrap profiled"
        selected = _profile_independent_candidate(
            profiler=profiler, n_subjects=anchor, widths=widths, attempts=attempts
        )
        if selected is not None:
            break
    if selected is None:
        raise ExecutionPlanningError(
            "no anchor map fits after top-first admission in the bounded "
            f"independent frontier with the {budget}-byte device budget "
            f"({len(attempts)} complete profiles; {map_reason}); "
            "rejected attempts: " + repr(tuple(attempts))
        )
    reason = "descending candidates rejected; admitted anchor retained"
    frozen_widths = selected.profile.axis_widths
    for extent in reversed(candidates[1:]):
        # Revisit the top only if anchor fallback changed the entire width map.
        if extent == largest and dict(frozen_widths) == dict(choices[0]):
            continue
        larger = _profile_independent_candidate(
            profiler=profiler,
            n_subjects=extent,
            widths=frozen_widths,
            attempts=attempts,
        )
        if larger is not None:
            selected = larger
            reason = "first fitting descending candidate admitted"
            break
    return selected, reason, map_reason


def _profile_independent_candidate(
    *,
    profiler: _ChunkProfiler,
    n_subjects: int,
    widths: Mapping[str, int],
    attempts: list[ChunkCandidateReceipt],
) -> SimulationChunkPlan | None:
    """Profile once, keeping scalar evidence and no rejected executable owners."""
    started = perf_counter()
    profile = profiler.profile_widths(n_subjects=n_subjects, widths=widths)
    elapsed = perf_counter() - started
    if profile.n_subjects != n_subjects or dict(profile.axis_widths) != dict(widths):
        raise ExecutionPlanningError(
            "Independent chunk profile changed its extent or widths."
        )
    required = _required_bytes(
        profile=profile, resident=profiler.resident, devices=profiler.devices
    )
    budget = profiler.runtime.execution.device_memory_bytes
    if budget is None:
        raise ExecutionPlanningError("Independent chunk admission needs a budget.")
    admitted = all(value <= budget for value in required.values())
    devices = []
    for device in profiler.devices:
        limiting = max(
            (stage for stage in profile.stages if device in stage.devices),
            key=lambda stage: stage.reservation_bytes,
            default=None,
        )
        devices.append(
            ChunkDeviceReceipt(
                platform=device.platform,
                device_id=device.id,
                resident_bytes=profiler.resident[device],
                fixed_bytes=profile.fixed_reservation.get(device, 0),
                output_bytes=profile.output_reservation.get(device, 0),
                max_stage_bytes=0 if limiting is None else limiting.reservation_bytes,
                limiting_stage=None if limiting is None else limiting.name,
                required_bytes=required[device],
            )
        )
    attempts.append(
        ChunkCandidateReceipt(
            n_subjects=n_subjects,
            padded_population=profile.padded_population,
            chunk_count=profile.padded_population // n_subjects,
            axis_widths=tuple(sorted(profile.axis_widths.items())),
            admitted=admitted,
            devices=tuple(devices),
            profile_seconds=elapsed,
            stage_entries=len(profile.stages) + len(profile.host_stages),
        )
    )
    return (
        SimulationChunkPlan(profile=profile, required_bytes=required)
        if admitted
        else None
    )


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
            programs.gate_fold,
            programs.gate_route,
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
