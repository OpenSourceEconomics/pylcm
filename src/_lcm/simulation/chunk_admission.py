"""Prepare and re-admit a complete call's chunks without retaining admission in code.

The selector deliberately chooses one common width per axis across the forward
programs. A smaller program clamps that choice to its declared extent. Every
candidate uses the same actual completed grids and retained solution owners.
"""

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax

from _lcm.egm.current_iid_replay_inputs import CurrentIIDReplayInputs
from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.footprint import per_device_footprint
from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.regime_building.projected_forward import ProjectedForwardInputs
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
from _lcm.simulation.chunk_profile_cache import profile_cache_registry
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.simulation.current_iid_overrides import CurrentIIDOverrideSchema
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.period_inputs import period_read_targets
from _lcm.simulation.programs import (
    budgeted_simulation_programs_ready,
    forward_regimes_by_period,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
    resolve_budget_devices,
    union_buffer_footprints,
)
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.subject_groups import grouped_extent
from _lcm.solution.backward_induction import _abstract_value_key, _hashable_metadata
from _lcm.solution.host_values import is_device_read_view, planning_stand_in
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds, SimulationPolicy
from _lcm.utils.logging import LogLevel
from lcm.exceptions import ExecutionPlanningError

if TYPE_CHECKING:
    type _StoredValues = Mapping[int, Mapping[str, jax.Array]]
else:
    # A runtime check samples one entry, and sampling a read-through view of
    # values kept off the device copies that value onto the device.
    type _StoredValues = object


@dataclass(frozen=True, kw_only=True)
class PreparedSimulationChunks:
    """Call-local code selection and actual shared owners, never a Model cache value."""

    plan: SimulationChunkPlan
    call_inputs: SimulationCallInputs
    admitted_inputs: DeviceBufferFootprint
    budget_note: str = ""
    """Clause naming the budget and the remedies, appended to a chunk refusal."""
    reserves_population_carry: bool = False
    """Whether the width was admitted beside every subject's carry, as simulating
    values kept off the device period by period across several chunks needs."""

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
            live=DeviceBufferFootprint(spans=MappingProxyType({}))
            if completed_setup is None
            else completed_setup,
            arguments=self.admitted_inputs,
            devices=memory.devices,
        )
        outputs = resident_bytes_by_device(
            live=memory.outputs,
            arguments=DeviceBufferFootprint(spans=MappingProxyType({})),
            devices=memory.devices,
        )
        remaining = replace(
            profile,
            fixed_reservation=MappingProxyType(
                {
                    device: value
                    - min(
                        value,
                        profile.setup_reservation.get(device, 0),
                        setup.get(device, 0),
                    )
                    for device, value in profile.fixed_reservation.items()
                }
            ),
            output_reservation=MappingProxyType(
                {
                    device: value - min(value, outputs.get(device, 0))
                    for device, value in profile.output_reservation.items()
                }
            ),
            setup_reservation=MappingProxyType({}),
        )
        resident = resident_bytes_by_device(
            live=memory.snapshot(),
            arguments=DeviceBufferFootprint(spans=MappingProxyType({})),
            devices=memory.devices,
        )
        required = _required_bytes(
            profile=remaining, resident=resident, devices=memory.devices
        )
        if any(value > memory.budget_bytes for value in required.values()):
            raise ExecutionPlanningError(
                "The reserved simulation chunk no longer fits current retained storage."
                f"{self.budget_note}"
            )


def prepare_simulation_chunks(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    values: _StoredValues,
    flags: Mapping[int, Mapping[str, jax.Array]],
    ages: TimeAxis,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    original_population: int,
    retained_footprint: DeviceBufferFootprint,
    independent_taste: bool,
    prng_impl: str | None = None,
    taste_prng_impl: str = "threefry2x32",
    log_level: LogLevel,
    runtime_checks: bool = True,
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None = None,
    current_iid_replay_inputs: CurrentIIDReplayInputs = MappingProxyType({}),
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
    process_grid_resolver: ProcessGridResolver | None = None,
    max_compilation_workers: int | None = None,
    group_sizes: tuple[int, ...] | None = None,
    current_iid_override_schema: CurrentIIDOverrideSchema = (),
) -> PreparedSimulationChunks:
    """Select a complete admitted outer cohort with the top-first planner.

    `max_compilation_workers` bounds the threads compiling each profiled
    candidate's forward programs, as it does for the solve's compilation waves.
    `group_sizes` holds the number of real subjects of each code when the
    regimes group subjects by an invariant state: no candidate then exceeds the
    largest group, and each candidate is profiled for every group's chunks.
    """
    runtime = next(iter(regimes.values())).simulation.programs.executor
    if (
        not isinstance(runtime, SimulationRuntime)
        or runtime.execution.device_memory_bytes is None
    ):
        raise ExecutionPlanningError(
            "Chunk admission requires a budgeted simulation runtime."
        )
    if not budgeted_simulation_programs_ready(
        regimes=regimes, n_periods=ages.n_periods, enable_jit=runtime.enable_jit
    ):
        raise ExecutionPlanningError(
            "Budgeted chunk admission requires compiled decision programs; "
            "host replay routes need complete stage profiles."
            f"{runtime.execution.device_memory_cap_note()}"
        )
    inputs = union_buffer_footprints(
        footprints=(
            retained_footprint,
            measure_buffer_footprint(
                tree=(
                    flat_params,
                    initial_conditions,
                    values,
                    flags,
                    ages.values,
                    projected_forward_inputs,
                )
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
        budget_note=runtime.execution.device_memory_cap_note(),
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
        live=memory.inputs,
        arguments=DeviceBufferFootprint(spans=MappingProxyType({})),
        devices=devices,
    )
    profiler = _ChunkProfiler(
        runtime=runtime,
        regimes=regimes,
        call_inputs=call_inputs,
        # Profiles and their cache keys need only the values' shapes and
        # placement; reading values kept off the device would copy them.
        values=planning_stand_in(values),
        flags=flags,
        policies=planning_stand_in(policies),
        current_iid_replay_inputs=current_iid_replay_inputs,
        projected_forward_inputs=projected_forward_inputs,
        ages=ages,
        initial_conditions=initial_conditions,
        regime_names_to_ids=regime_names_to_ids,
        population=population,
        original_population=original_population,
        independent_taste=independent_taste,
        prng_impl=prng_impl,
        taste_prng_impl=taste_prng_impl,
        log_level=log_level,
        runtime_checks=runtime_checks,
        resident=resident,
        devices=devices,
        max_compilation_workers=max_compilation_workers,
        group_sizes=group_sizes,
        current_iid_override_schema=current_iid_override_schema,
    )
    plan = _plan_independent_chunks(profiler=profiler, alignment=alignment)
    # Values kept off the device are simulated period by period when there are
    # several chunks, so every chunk's carry is live while one chunk runs. That
    # carry grows with the population, not the width; admit beside it.
    reserves_population_carry = (
        group_sizes is None
        and is_device_read_view(values)
        and plan.profile.n_subjects < plan.profile.padded_population
    )
    if reserves_population_carry:
        carry = _population_carry_bytes(
            initial_conditions=initial_conditions,
            subject_devices=runtime.subject_devices,
        )
        prefetch = _prefetch_bytes(
            regimes=regimes,
            values=planning_stand_in(values),
            flags=planning_stand_in(flags),
            policies=planning_stand_in(policies),
            n_periods=ages.n_periods,
            devices=devices,
        )
        try:
            plan = _plan_independent_chunks(
                profiler=replace(
                    profiler,
                    resident=MappingProxyType(
                        {
                            device: bytes_
                            + carry.get(device, 0)
                            + prefetch.get(device, 0)
                            for device, bytes_ in resident.items()
                        }
                    ),
                ),
                alignment=alignment,
            )
        except ExecutionPlanningError as error:
            msg = (
                "Simulating values kept off the device in several chunks runs every "
                "chunk through a period before the next, so every subject's state "
                f"stays on the device: about {dict(sorted(carry.items(), key=str))!r} "
                "bytes, and the next period's reads are copied early: about "
                f"{dict(sorted(prefetch.items(), key=str))!r} bytes, beside a chunk's "
                "workspace. Simulate fewer subjects per call or raise "
                "ExecutionConfig.device_memory_bytes."
                f"{runtime.execution.device_memory_cap_note()}"
            )
            raise ExecutionPlanningError(msg) from error
    return PreparedSimulationChunks(
        plan=plan,
        call_inputs=call_inputs,
        admitted_inputs=memory.inputs,
        budget_note=runtime.execution.device_memory_cap_note(),
        reserves_population_carry=reserves_population_carry,
    )


def _population_carry_bytes(
    *,
    initial_conditions: Mapping[str, jax.Array],
    subject_devices: tuple[jax.Device, ...],
) -> dict[jax.Device, int]:
    """Estimate each subject device's share of every subject's carry.

    A carry holds a subject's states, regime and role, all shaped like its
    initial conditions, and two route trackers of one 32-bit integer each; its
    keys are per chunk. The estimate charges the whole population, the running
    chunk included, whose carry its own profile also charges.
    """
    population = int(initial_conditions["regime_id"].shape[0])
    total = sum(int(column.nbytes) for column in initial_conditions.values())
    # Role (when the initial conditions name none) and the two route trackers.
    total += 3 * population * 4
    share = math.ceil(total / len(subject_devices))
    return dict.fromkeys(subject_devices, share)


def _prefetch_bytes(
    *,
    regimes: Mapping[RegimeName, Regime],
    values: Mapping[int, Mapping[str, jax.Array | jax.ShapeDtypeStruct]],
    flags: Mapping[int, Mapping[str, jax.Array | jax.ShapeDtypeStruct]],
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None,
    n_periods: int,
    devices: tuple[jax.Device, ...],
) -> dict[jax.Device, int]:
    """Estimate each device's bytes of the largest read set one period copies early.

    The period-by-period loop starts copying the next period's declared values,
    flags and policies while a period runs, so they sit beside that period's own
    reads. Every argument is a planning stand-in: nothing is read.
    """
    by_id = {device.id: device for device in devices}
    active_by_period = forward_regimes_by_period(regimes=regimes, n_periods=n_periods)
    largest: dict[jax.Device, int] = {}
    for period in range(1, n_periods):
        active = active_by_period[period]
        value_targets, flag_targets = period_read_targets(
            active=active, period=period, values=values, flags=flags
        )
        arrays = [
            stored[target_period][regime]
            for stored, targets in ((values, value_targets), (flags, flag_targets))
            for target_period, regime in targets
            if regime in stored.get(target_period, {})
        ]
        arrays.extend(
            leaf
            for name in active
            for leaf in jax.tree.leaves((policies or {}).get(period, {}).get(name))
            if isinstance(leaf, jax.Array | jax.ShapeDtypeStruct)
        )
        need: dict[jax.Device, int] = {}
        for array in arrays:
            footprint = per_device_footprint(array=array)
            for device_id in footprint.device_ids:
                if device_id in by_id:
                    device = by_id[device_id]
                    need[device] = need.get(device, 0) + footprint.bytes_per_device
        for device, bytes_ in need.items():
            largest[device] = max(largest.get(device, 0), bytes_)
    return largest


def _simulation_chunk_profile_key(
    *,
    runtime: SimulationRuntime,
    regimes: Mapping[str, Regime],
    call_inputs: SimulationCallInputs,
    values: Mapping[int, Mapping[str, jax.Array | jax.ShapeDtypeStruct]],
    flags: Mapping[int, Mapping[str, jax.Array]],
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None,
    current_iid_replay_inputs: CurrentIIDReplayInputs = MappingProxyType({}),
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
    ages: TimeAxis,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    n_subjects: int,
    population: int,
    original_population: int,
    widths: Mapping[str, int],
    independent_taste: bool,
    prng_impl: str | None = None,
    taste_prng_impl: str = "threefry2x32",
    log_level: LogLevel,
    runtime_checks: bool = True,
    group_sizes: tuple[int, ...] | None = None,
    current_iid_override_schema: CurrentIIDOverrideSchema = (),
) -> tuple[Hashable, ...]:
    """Return the canonical, versioned cache key for `profile_simulation_chunk`.

    Every component is either immutable static identity or an abstract
    (shape/dtype/weak-type/sharding) description of a dynamic leaf — never a
    concrete value. Two calls whose dynamic parameters, initial states,
    solution/replay arrays or process probabilities differ only in value (not
    shape/dtype/sharding) hash to the same key and share the cached profile;
    the caller's admission formula still re-runs against current owners.

    Conservative, documented limitations:
    - `policies` is keyed by `_abstract_value_key`, exactly like every other
      dynamic argument: its period/regime/leaf-name structure plus each
      leaf's shape, dtype, weak type and sharding. Two policy mappings that
      agree on all of that share a cached profile even when their numerical
      contents differ, which is the intended behaviour (the profile depends
      only on abstract metadata). What this does *not* distinguish is a
      policy's internal wiring: if two policy payloads present identical
      abstract leaves but route them through different callables, the key
      cannot tell them apart.
    - The regime read/transfer graph is not walked
      independently; it is approximated by the regime/base-space identity and
      shape metadata already covered below. A change to the transfer graph
      that leaves every covered shape untouched (extremely unlikely given how
      regimes are constructed) would not be distinguished by this key.
    """
    version = 4  # bump when any component below changes meaning
    key_version = ("chunk-profile-key", version)
    program_identity = (
        tuple(sorted(regimes)),
        tuple(
            (name, tuple(sorted(regime.active_periods)))
            for name, regime in regimes.items()
        ),
        _hashable_metadata(regime_names_to_ids),
        ages.n_periods,
    )
    axes_and_widths = tuple(sorted(widths.items()))
    argument_metadata = (
        current_iid_override_schema,
        _abstract_value_key(value=call_inputs.flat_params),
        _abstract_value_key(
            value=tuple(
                (space.states, space.actions)
                for space in call_inputs.base_state_action_spaces.values()
            )
        ),
        _abstract_value_key(value=current_iid_replay_inputs),
        _abstract_value_key(value=projected_forward_inputs),
        _abstract_value_key(value=values),
        _abstract_value_key(value=flags),
        _abstract_value_key(value=initial_conditions),
        _abstract_value_key(value=ages.values),
    )
    population_metadata = (n_subjects, population, original_population, group_sizes)
    devices_and_backend = (
        tuple((device.platform, device.id) for device in runtime.subject_devices),
        runtime.enable_jit,
        runtime.execution.simulation_sharding,
        runtime.execution.device_ids,
        runtime.execution.device_memory_bytes,
    )
    diagnostics_and_precision = (
        log_level,
        runtime_checks,
        independent_taste,
        prng_impl or jax.config.jax_default_prng_impl,
        taste_prng_impl,
        jax.config.jax_enable_x64,
    )
    # Policies are rebuilt per solve, so their Python identity is never stable
    # across calls even for numerically identical content; key their abstract
    # (type/shape-only, never value) structure instead, exactly like every
    # other dynamic argument above.
    policy_identity = None if policies is None else _abstract_value_key(value=policies)
    return (
        key_version,
        program_identity,
        axes_and_widths,
        argument_metadata,
        population_metadata,
        devices_and_backend,
        diagnostics_and_precision,
        policy_identity,
    )


@dataclass(frozen=True, kw_only=True)
class _ChunkProfiler:
    """Transient search inputs, released after selecting one immutable profile."""

    runtime: SimulationRuntime
    regimes: Mapping[str, Regime]
    call_inputs: SimulationCallInputs
    values: Mapping[int, Mapping[str, jax.Array | jax.ShapeDtypeStruct]]
    flags: Mapping[int, Mapping[str, jax.Array]]
    ages: TimeAxis
    initial_conditions: Mapping[str, jax.Array]
    regime_names_to_ids: RegimeNamesToIds
    population: int
    original_population: int
    independent_taste: bool
    prng_impl: str | None = None
    taste_prng_impl: str = "threefry2x32"
    log_level: LogLevel
    runtime_checks: bool = True
    resident: Mapping[jax.Device, int]
    devices: tuple[jax.Device, ...]
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None = None
    current_iid_replay_inputs: CurrentIIDReplayInputs = MappingProxyType({})
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({})
    max_compilation_workers: int | None = None
    group_sizes: tuple[int, ...] | None = None
    current_iid_override_schema: CurrentIIDOverrideSchema = ()

    def planning_population(self, *, alignment: int) -> int:
        """Return the largest extent one chunk can need: a group, or everyone."""
        if self.group_sizes is None:
            return self.population
        return grouped_extent(sizes=self.group_sizes, alignment=alignment)

    def __call__(self, *, n_subjects: int) -> SimulationChunkProfile:
        """Return a fitting common inner choice, or the smallest required bound."""
        budget = self.runtime.execution.device_memory_bytes
        if budget is None:
            raise ExecutionPlanningError("Chunk profiling requires a device budget.")
        axes = _common_axes(regimes=self.regimes, n_subjects=n_subjects)
        choices = workspace_width_candidates(
            axes=axes,
            fixed_widths=self.runtime.execution.axis_widths,
            width_ceilings=self.runtime.execution.axis_width_ceilings,
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
        """Profile exactly one whole inner map without reopening its search.

        A hit reuses the immutable multi-stage profile (compiled executables
        and byte counts) built for an identical abstract key; it never skips
        the caller's admission formula, which always re-runs against the
        returned profile and this call's own resident-bytes ledger.

        The builder is a `functools.partial` over the module-level
        `profile_simulation_chunk`, never a nested `def`: a nested `def` here
        would be re-wrapped by the beartype claw on every call and would close
        over `self`, keeping this call's arrays and compiled executables
        reachable from the retained wrapper (see `_lcm/utils/functools.py` and
        the module-level callbacks in `chunk_profile_inventory.py`). The
        partial itself is dropped as soon as `get_or_build` returns.
        """
        arguments = {
            "runtime": self.runtime,
            "regimes": self.regimes,
            "values": self.values,
            "flags": self.flags,
            "policies": self.policies,
            "current_iid_replay_inputs": self.current_iid_replay_inputs,
            "projected_forward_inputs": self.projected_forward_inputs,
            "ages": self.ages,
            "initial_conditions": self.initial_conditions,
            "regime_names_to_ids": self.regime_names_to_ids,
            "n_subjects": n_subjects,
            "population": self.population,
            "original_population": self.original_population,
            "widths": widths,
            "independent_taste": self.independent_taste,
            "prng_impl": self.prng_impl,
            "taste_prng_impl": self.taste_prng_impl,
            "log_level": self.log_level,
            "runtime_checks": self.runtime_checks,
            "group_sizes": self.group_sizes,
            "current_iid_override_schema": self.current_iid_override_schema,
        }
        return profile_cache_registry().get_or_build(
            runtime_token=self.runtime.profile_cache_token,
            key=_simulation_chunk_profile_key(
                call_inputs=self.call_inputs, **arguments
            ),
            build=partial(
                profile_simulation_chunk,
                flat_params=self.call_inputs.flat_params,
                base_spaces=self.call_inputs.base_state_action_spaces,
                max_compilation_workers=self.max_compilation_workers,
                **arguments,
            ),
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
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    configured: Mapping[str, int],
    ceilings: Mapping[str, int],
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
                    axes=axes,
                    fixed_widths=pins,
                    width_ceilings=ceilings,
                    budget_bytes=None,
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
    population = profiler.planning_population(alignment=alignment)
    subject_width = _resolve_subject_anchor_width(
        configured=configured,
        population=population,
        alignment=alignment,
        budget=budget,
    )
    candidates = _independent_outer_candidates(
        population=population, alignment=alignment, subject_width=subject_width
    )
    anchor = candidates[0]
    largest = candidates[-1]
    axes = _common_axes(regimes=profiler.regimes, n_subjects=anchor)
    choices = _independent_anchor_widths(
        axes=axes,
        configured=configured,
        ceilings=profiler.runtime.execution.axis_width_ceilings,
    )
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
            f"({len(attempts)} complete profiles; {map_reason});"
            f"{profiler.runtime.execution.device_memory_cap_note()} "
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
