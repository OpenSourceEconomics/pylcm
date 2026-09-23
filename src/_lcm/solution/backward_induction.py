"""The backward-induction loop that drives whichever solver each regime declares.

`solve` walks periods from the last to the first and, for every active regime,
dispatches the core programs that regime's solver declares, rolls their
published values and continuation artifacts into the next period up, and
releases each buffer whose last declared consumer has returned. Under a
device-memory budget it also selects the period's workspace widths and admits
the resulting residency before anything is compiled.
"""

import dataclasses
import functools
import gc
import inspect
import logging
import os
import time
from collections.abc import (
    Callable,
    Container,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType
from typing import cast

import jax
from jax._src import config as jax_config
from jax._src import core as jax_core

from _lcm.engine import (
    Regime,
    StateActionSpace,
    _build_regime_sharding,
    placed_devices_for_ids,
)
from _lcm.execution.abstract_program_inputs import abstract_program_inputs
from _lcm.execution.compiler_inputs import compiler_input_paths
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    ProgramScope,
    ReducedAxis,
    ResolvedCoreProgram,
    TiledOutputAxis,
    ValueRead,
    _value_read_argument_leaf,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
    resolve_core_program_candidates,
    select_programs,
)
from _lcm.execution.donation import (
    ResolvedDonation,
    resolve_donations,
    unit_input_readers,
    withhold_shared_donations,
)
from _lcm.execution.eager_core import make_eager_core
from _lcm.execution.execution_plan import (
    ResolvedExecution,
    execution_over_visible_devices,
)
from _lcm.execution.footprint import (
    ArtifactFootprint,
    ResidentInventory,
    ScheduledUnit,
    concrete_device_bytes,
    layout_footprint,
    per_device_footprint,
    plan_resident_inventory,
)
from _lcm.execution.hlo_fusions import ReduceFusionVerdict, classify_reduce_fusions
from _lcm.execution.internal_outputs import (
    ResolvedProducer,
    assert_width_invariant_internal_outputs,
    consumed_producer_names,
    internal_input_templates,
    resolve_producer,
    topological_program_order,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import (
    ExpectedOutputLeaf,
    PlannedCore,
    ResolvedOutputLayout,
    assert_value_leaf_layout,
    resolve_output_layout,
)
from _lcm.execution.pending_work import BeforeArrayDelete, PendingSolveWork
from _lcm.execution.scheduler import (
    BufferIdentity,
    BufferRegistry,
    DispatchUnit,
    PeriodTransferCache,
    ScheduledNode,
    buffer_identity,
    plan_period_waves,
    release_closed_artifacts,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    classify_value_transfer,
    resolve_value_transfer,
)
from _lcm.execution.workspace_planning import (
    BoundedWidthSelector,
    CompilerMemoryReservation,
    WorkspacePlan,
    _admissible_width,
    compiler_memory_reservation,
    plan_workspace,
    workspace_width_candidates,
)
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.regime_building.gated_edges import (
    EDGE_PERIOD_CONTEXT_ARGS,
    CompiledEdgeFold,
    bind_edge_period_context,
    build_reference_params_mapping_for_fold,
    build_same_period_mapping_for_fold,
    edge_may_fold_at_period,
    gate_reads_dissolution_flag,
    source_reads_folded_wbar,
)
from _lcm.regime_building.Q_and_F import (
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
)
from _lcm.solution.continuation_reads import published_continuation_template
from _lcm.solution.contract import (
    BackwardInductionResult,
    ContinuationPayload,
    GeneratedReplayAuthority,
)
from _lcm.solution.diagnostics import (
    _emit_post_loop_diagnostics,
    _fold_period_diagnostics,
    _init_diagnostic_accumulators,
    _states_for_period,
)
from _lcm.solution.kernel_attribution import (
    log_executed_kernel,
    log_module_fanout,
)
from _lcm.solution.kernel_output import (
    ConsumedKernelOutput,
    consume_kernel_output,
)
from _lcm.solution.period_capture import (
    PeriodCaptureTarget,
    capture_kernel_inputs,
    resolve_capture_target,
)
from _lcm.solution.solve_inputs import (
    SolveInputMappings,
    locate_artifact,
    register_rolled_inputs,
    substitute_artifact,
)
from _lcm.solution.solve_phase_records import CallId, solve_phase
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.solution.undeclared_reads import undeclared_read_pins
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
    _RegimeVTopology,
    placed_V_sharding,
)
from _lcm.typing import FlatParams, RegimeName, SimulationPolicy
from _lcm.utils.logging import (
    format_duration,
    log_period_header,
    log_period_timing,
    raise_or_warn,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidValueFunctionError,
    ModelInitializationError,
)
from lcm.execution import WidthSearch
from lcm.solver_api import (
    SIMULATION_POLICY,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    KernelOutput,
)
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarFloat,
    ScalarInt,
)

# Stands in for a period's flag mapping when the model retains no dissolution
# flags, so every period key is present with nothing behind it. One shared
# instance: it is immutable and carries no arrays.
_NO_DISSOLUTION_FLAGS: MappingProxyType[RegimeName, BoolND] = MappingProxyType({})


def solve(  # noqa: C901, PLR0912, PLR0915
    *,
    flat_params: FlatParams,
    ages: AgeGrid,
    regimes: MappingProxyType[RegimeName, Regime],
    program_fingerprint: str,
    logger: logging.Logger,
    enable_jit: bool,
    execution: ResolvedExecution | None = None,
    collect_solver_diagnostics: bool = False,
    max_compilation_workers: int | None = None,
    retain_dissolution_flags: bool = True,
    retain_replay: bool = True,
    retain_all_artifacts: bool = False,
    persistable_artifact_refs: frozenset[ArtifactRef] = frozenset(),
    retained_input_arrays: object = (),
    process_grid_resolver: ProcessGridResolver | None = None,
    call_id: CallId | None = None,
) -> BackwardInductionResult:
    """Solve a model by backward induction, whatever solver each regime declares.

    Args:
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        program_fingerprint: Digest of the model facts every lowered program
            depends on — structure, fixed parameters, grid support, no
            solve-time parameter values; enters every executable's compilation
            key, so equivalent programs of equivalent models share one
            executable and two models never do.
        logger: Logger that logs to stdout, and carries the runtime-validation
            policy. `log_level="debug"` stops backward induction at the first
            NaN period and raises; `"warning"` / `"progress"` let induction run
            to completion and log a warning, so `solve` returns a complete
            (NaN-bearing) solution; `"off"` skips the NaN check.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        execution: The hardware-local facts the model resolved — its devices,
            the optional per-device workspace budget, and fixed planner axis
            widths. `None` resolves the inert configuration against every
            visible device.
        collect_solver_diagnostics: Whether to retain a kernel's numerical
            self-report. Public ``Model.solve()`` and automatic simulation request
            it; ``log_level`` still decides whether diagnostics are calculated and
            retained. Internal callers may disable collection.
        max_compilation_workers: Maximum number of threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.
        retain_dissolution_flags: Whether a caller wants the per-period
            dissolution flags on the result for their own sake. A model whose
            gates read `D_target` retains them regardless — the flags are a
            simulate-side input there, not an inspection artifact.
        retain_replay: Whether replay artifacts are retained. Selects, per
            period kernel, the scoped programs that are dispatched (a kernel may
            publish a values-only and a replay variant of one body) and whether
            a published simulation policy is copied to the host and kept. A
            policy is kept only for a regime whose declared simulation route
            reads it; a values-only solve drops every policy at the period
            boundary instead of retaining one device-sized artifact per period.
        retain_all_artifacts: Whether independently persistable continuation and
            auxiliary outputs may be retained.
        persistable_artifact_refs: Exact model-authoritative artifact addresses
            selected by ``ALL_PERSISTABLE_ARTIFACTS``. Only these addresses are
            dispatched and copied for that mode; ordinary replay retention keeps
            its separate in-memory behavior, including ``NOT_PERSISTED`` routes.
        retained_input_arrays: Caller's already materialized inputs kept alive
            beside this solve, including automatic simulation's original and
            normalized initial conditions. These enter fixed residency by actual
            physical storage; they are not solve operands or cache entries.
        call_id: Identifier of the public call this solve serves, stamped on
            the host-phase records. `None` emits no phase records.

    Returns:
        The named backward-induction outputs: the immutable mapping of periods
        to regime value-function arrays, the immutable mapping of periods to each
        regime's published simulation policy (the off-grid policy artifact
        simulation can interpolate; regimes whose kernels publish none, or whose
        simulation route does not read one, have no entry, and the whole
        mapping is empty when `retain_replay` is false), and the immutable mapping of
        periods to each COLLECTIVE regime's dissolution flag `D` — `True` on the
        state cells whose action mask is empty, distinct from a numeric `-inf`
        value; empty inner mappings for models without collective regimes, so
        the default path only gains an empty dissolution mapping.

    """
    resolved_execution = (
        execution_over_visible_devices() if execution is None else execution
    )
    if resolved_execution.device_memory_bytes is not None and not enable_jit:
        msg = (
            "ExecutionConfig.device_memory_bytes requires JIT compilation so the "
            "compiler can report peak workspace."
        )
        raise ExecutionPlanningError(msg)

    capture_target = resolve_capture_target()

    # The state-action spaces and the fence that reads them depend only on
    # `regimes` and `flat_params`, so a colliding model is rejected before a
    # single kernel is compiled rather than after every regime-period has been
    # AOT-compiled.
    with solve_phase(name="state_action_spaces", logger=logger, call_id=call_id):
        base_state_action_spaces = _build_base_state_action_spaces(
            regimes=regimes,
            flat_params=flat_params,
            process_grid_resolver=process_grid_resolver,
        )
        _reject_edge_fold_state_param_collisions(
            regimes=regimes,
            base_state_action_spaces=base_state_action_spaces,
            flat_params=flat_params,
        )

    with solve_phase(name="continuation_templates", logger=logger, call_id=call_id):
        next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr = (
            _build_continuation_templates(
                regimes=regimes,
                flat_params=flat_params,
                device_ids=resolved_execution.device_ids,
                process_grid_resolver=process_grid_resolver,
            )
        )

    # Resolve every solve program, then compile unique lowerings when enabled.
    compiled_programs = _compile_all_functions(
        regimes=regimes,
        program_fingerprint=program_fingerprint,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        next_edge_to_V_arr=next_edge_to_V_arr,
        enable_jit=enable_jit,
        execution=resolved_execution,
        retain_replay=retain_replay,
        retain_all_artifacts=retain_all_artifacts,
        persistable_artifact_refs=persistable_artifact_refs,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
        fixed_input_arrays=(
            retained_input_arrays,
            tuple(
                (space.states, space.discrete_actions, space.continuous_actions)
                for space in base_state_action_spaces.values()
            ),
        ),
        process_grid_resolver=process_grid_resolver,
        call_id=call_id,
    )
    compiled_functions = compiled_programs.executables
    replay_dispatches = {
        (regime_name, period)
        for (regime_name, period, _core_name), metadata in (
            compiled_programs.metadata.items()
        )
        if metadata.scope is ProgramScope.REPLAY
    }
    # The executables were lowered against this ledger — a donation set is part
    # of a compilation key — so the loop commits to the one they were keyed by
    # rather than building a second.
    input_liveness = compiled_programs.input_liveness
    input_templates = SolveInputMappings(
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        next_edge_to_V_arr=next_edge_to_V_arr,
    )
    buffer_registry = BufferRegistry()
    # The templates live for the whole solve and stand in the mappings wherever
    # a key's buffer has gone — a period that rolls one forward unchanged, a
    # released key, a donated one. Declaring them here makes that a property of
    # the buffer rather than of each release decision.
    buffer_registry.declare_not_produced(
        tree=(
            flat_params,
            tuple(
                (space.states, space.discrete_actions, space.continuous_actions)
                for space in base_state_action_spaces.values()
            ),
            tuple(
                regime.solution.period_state_axes
                for regime in regimes.values()
                if regime.solution.period_state_axes is not None
            ),
            input_templates.next_regime_to_V_arr,
            input_templates.next_regime_to_continuation,
            input_templates.next_edge_to_V_arr,
        )
    )
    # An eager solve's dispatches are ordinary Python calls, so any object an
    # input contained can come back out as an output; nothing is released.
    if not enable_jit:
        logger.debug(
            "release skipped: eager dispatch",
            extra={"release_skipped": "eager dispatch"},
        )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    simulation_policies: dict[int, MappingProxyType[RegimeName, SimulationPolicy]] = {}
    generated_replay_authorities: dict[
        int, MappingProxyType[RegimeName, GeneratedReplayAuthority]
    ] = {}
    dissolution_flags: dict[int, MappingProxyType[RegimeName, BoolND]] = {}
    solver_diagnostics: dict[int, MappingProxyType[RegimeName, SolverDiagnostics]] = {}
    retained_continuations: dict[ArtifactRef, object] = {}
    replay_artifacts: dict[ArtifactRef, object] = {}
    auxiliary_artifacts: dict[ArtifactRef, object] = {}

    # Every collective kernel publishes `D`, but only two things read the
    # ACCUMULATED per-period mapping: forward simulation, for a gate that
    # declares the `D_target` operand, and a caller that asked for the flags.
    # A gate's own signature settles the first, so the answer is known before
    # the first kernel runs; where it is `False` and nobody asked, each period's
    # flags go out of scope with the period that produced them instead of
    # staying live for the whole induction. The per-period flags themselves are
    # built either way — the edge fold below reads them while they are current.
    publish_dissolution_flags = retain_dissolution_flags or any(
        gate_reads_dissolution_flag(edge=edge)
        for regime in regimes.values()
        for edge in regime.gated_edges.values()
    )

    # Async diagnostics accumulators: per-period NaN/Inf flags (and the
    # debug min/max/mean trio) live here as device-side scalars during
    # the hot loop. The two NaN/Inf flags are folded into single running
    # scalars via `v_array_has_nan` / `v_array_has_inf` — both jit-wrapped,
    # so XLA partitions each reduction across the V-array's devices instead
    # of gathering V onto the default device. The per-period min/max/mean
    # trio is appended to a list (only emitted at debug, where we genuinely
    # want every number on host).
    #
    # Per-period `block_until_ready()` after the running update forces
    # the device kernel to finish before the next period dispatches.
    # This frees the per-period `isnan(V_arr)` / `isinf(V_arr)`
    # intermediate buffers (V_arr-shaped, so model-dependent) so they
    # don't stack up across the loop. `block_until_ready` is a
    # *device-only* sync — no host transfer, no PCIe round-trip — so
    # it doesn't introduce a host stall: if `max_Q_over_a` (the
    # dominant per-period kernel) is in flight, the call returns
    # immediately when the small reduction is done.
    #
    # One host transfer per stat at end of solve (`.item()` on the
    # running scalars) decides whether to enter the failure-path
    # localisation. On a healthy solve no per-row materialisation
    # happens.
    #
    # Two gates, both falling out of the public log level:
    # - NaN/Inf tracking feeds runtime validation, so it runs whenever
    #   validation is not `"off"` (log levels `"warning"`/`"progress"`/
    #   `"debug"`). It skips even the NaN fail-fast when validation is off.
    # - The min/max/mean trio is a pure logging extra, gated on the
    #   logger's debug level.
    diagnostics_enabled = validation_enabled(logger)
    stats_enabled = logger.isEnabledFor(logging.DEBUG)
    (
        diagnostic_rows,
        diagnostic_min,
        diagnostic_max,
        diagnostic_mean,
        running_any_nan,
        running_any_inf,
    ) = _init_diagnostic_accumulators()

    logger.info("Starting solution")
    total_start = time.monotonic()

    # A published simulation policy is a solve output; no backward step reads
    # it. Its buffers can alias the period's continuation buffer, so retaining
    # one per period pins a continuation-sized device buffer per period for the
    # whole induction. Value-only solves therefore discard it at the period
    # boundary; a requesting consumer receives host copies, which simulation
    # re-materializes on device.
    host_device = (
        jax.devices("cpu")[0]
        if retain_replay
        or retain_all_artifacts
        or persistable_artifact_refs
        or (collect_solver_diagnostics and diagnostics_enabled)
        else None
    )

    # Every regime's node set runs on the devices the planner assigned it,
    # which is a fact of the model rather than of a period, so the device sets
    # the wave plan reads are resolved once for the whole solve.
    device_ids_by_regime = MappingProxyType(
        {
            regime_name: _regime_device_ids(
                regime=regime, visible_device_ids=resolved_execution.device_ids
            )
            for regime_name, regime in regimes.items()
        }
    )

    pending_work = (
        PendingSolveWork()
        if resolved_execution.device_memory_bytes is not None
        else None
    )
    with solve_phase(name="backward_induction", logger=logger, call_id=call_id):
        try:
            for period in reversed(range(ages.n_periods)):
                period_start = time.monotonic()
                period_solution: dict[RegimeName, FloatND] = {}
                period_continuations: dict[RegimeName, ContinuationPayload] = {}
                period_simulation_policies: dict[RegimeName, SimulationPolicy] = {}
                period_generated_replay_authorities: dict[
                    RegimeName, GeneratedReplayAuthority
                ] = {}
                period_dissolution_flags: dict[RegimeName, BoolND] = {}
                period_solver_diagnostics: dict[RegimeName, SolverDiagnostics] = {}
                period_retained_continuations: dict[
                    tuple[RegimeName, ArtifactKey], object
                ] = {}
                period_replay_artifacts: dict[
                    tuple[RegimeName, ArtifactKey], object
                ] = {}
                period_auxiliary_artifacts: dict[
                    tuple[RegimeName, ArtifactKey], object
                ] = {}

                period_inputs = SolveInputMappings(
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    next_edge_to_V_arr=next_edge_to_V_arr,
                )
                register_rolled_inputs(
                    inputs=period_inputs,
                    next_period=period + 1,
                    ledger=input_liveness,
                    registry=buffer_registry,
                )
                period_pending_outputs: list[FloatND] = []
                period_release_candidates: dict[
                    ValueArtifactAddress, _InputDispatch
                ] = {}

                active_regimes = {
                    regime_name: regime
                    for regime_name, regime in regimes.items()
                    if period in regime.active_periods
                }

                log_period_header(
                    logger=logger,
                    age=ages.values[period],
                    n_active_regimes=len(active_regimes),
                )

                shared_transfer_counts, regime_shared_transfer_keys = (
                    _period_shared_transfer_plan(
                        compiled_cores_by_regime=MappingProxyType(
                            {
                                regime_name: compiled_functions[(regime_name, period)]
                                for regime_name in active_regimes
                            }
                        )
                    )
                )
                period_transfer_cache = PeriodTransferCache(
                    registry=buffer_registry,
                    consumer_counts=shared_transfer_counts,
                    pending_outputs=period_pending_outputs,
                    release_enabled=enable_jit,
                    logger=logger,
                    before_delete=None
                    if pending_work is None
                    else pending_work.before_delete,
                )

                # Regimes declaring `same_period_refs` read other regimes' V of
                # THIS period, so a reference regime is planned into an earlier wave
                # than its reader. Independent regimes whose device sets are disjoint
                # share a wave and dispatch back to back, which is what a submesh
                # placement buys; regimes sharing a device keep one unit per wave, in
                # declaration order among regimes without a reference.
                waves = plan_period_waves(
                    nodes=tuple(
                        ScheduledNode(
                            period=period, regime=regime_name, program=core_key
                        )
                        for regime_name in active_regimes
                        for core_key in compiled_functions[(regime_name, period)]
                    ),
                    same_period_dependencies=MappingProxyType(
                        {
                            regime_name: regime.same_period_ref_regimes
                            for regime_name, regime in active_regimes.items()
                        }
                    ),
                    device_sets=MappingProxyType(
                        {
                            regime_name: device_ids_by_regime[regime_name]
                            for regime_name in active_regimes
                        }
                    ),
                )
                for wave in waves:
                    for unit in wave:
                        regime_name = unit.regime
                        regime = active_regimes[regime_name]
                        regime_retains_replay = (
                            regime_name,
                            period,
                        ) in replay_dispatches
                        selected_artifact_keys = _selected_artifact_keys_for_cell(
                            persistable_artifact_refs=persistable_artifact_refs,
                            regime_name=regime_name,
                            period=period,
                        )
                        selected_cores, selected_donations = (
                            _select_runtime_donation_cores(
                                compiled_programs=compiled_programs,
                                unit=unit,
                                inputs=SolveInputMappings(
                                    next_regime_to_V_arr=next_regime_to_V_arr,
                                    next_regime_to_continuation=next_regime_to_continuation,
                                    next_edge_to_V_arr=next_edge_to_V_arr,
                                ),
                                templates=input_templates,
                                registry=buffer_registry,
                                logger=logger,
                            )
                        )
                        donated_inputs = _donated_input_arrays(
                            donations=selected_donations,
                            unit=unit,
                            inputs=SolveInputMappings(
                                next_regime_to_V_arr=next_regime_to_V_arr,
                                next_regime_to_continuation=next_regime_to_continuation,
                                next_edge_to_V_arr=next_edge_to_V_arr,
                            ),
                            templates=input_templates,
                            registry=buffer_registry,
                        )
                        output = _run_period_kernel(
                            regime=regime,
                            regime_name=regime_name,
                            period=period,
                            compiled_cores=_cores_with_transfer_cache(
                                cores=selected_cores,
                                cache=period_transfer_cache,
                                pending_work=pending_work,
                            ),
                            capture_target=capture_target,
                            state_action_space=base_state_action_spaces[regime_name],
                            flat_params=flat_params,
                            ages=ages,
                            next_regime_to_V_arr=next_regime_to_V_arr,
                            next_regime_to_continuation=next_regime_to_continuation,
                            logger=logger,
                            next_edge_to_V_arr=next_edge_to_V_arr,
                            period_solution=period_solution,
                            retain_replay=_regime_retains_replay(
                                regime=regime,
                                retain_replay=retain_replay,
                            ),
                            selected_artifact_keys=selected_artifact_keys,
                        )
                        continuation_spec = regime.solution.continuation_spec
                        result = consume_kernel_output(
                            output=output,
                            continuation_key=(
                                None
                                if continuation_spec is None
                                else continuation_spec.artifact_key
                            ),
                            regime_name=regime_name,
                            period=period,
                            artifact_authorities=regime.solution.artifact_authorities,
                        )
                        V_arr = result.value
                        # The published V mapping is the calling convention for every
                        # downstream consumer — the parents' cores and the AOT-lowered
                        # simulate programs are both compiled against the per-regime V
                        # topology — so a kernel value must leave its compiled
                        # program on the template's placement; it is asserted
                        # here, never re-placed.
                        V_arr = _publish_kernel_value(
                            value=V_arr,
                            compiled_cores=compiled_functions[(regime_name, period)],
                        )
                        _fail_if_continuation_publisher_returned_none(
                            result=result,
                            regime_name=regime_name,
                            period=period,
                            continuation_publishers=next_regime_to_continuation,
                        )
                        if result.continuation is not None:
                            period_continuations[regime_name] = result.continuation
                        if retain_all_artifacts:
                            period_retained_continuations.update(
                                {
                                    (regime_name, key): payload
                                    for key, payload in (
                                        result.continuation_artifacts.items()
                                    )
                                    if ArtifactRef(
                                        period=period,
                                        regime=regime_name,
                                        key=key,
                                    )
                                    in persistable_artifact_refs
                                }
                            )
                        # A policy is kept only where the regime's declared
                        # simulation route reads it; replay authority travels
                        # with its policy.
                        if (
                            result.simulation_policy is not None
                            and regime_retains_replay
                        ):
                            period_simulation_policies[regime_name] = (
                                result.simulation_policy
                            )
                            if result.generated_replay_authority is not None:
                                period_generated_replay_authorities[regime_name] = (
                                    result.generated_replay_authority
                                )
                        period_replay_artifacts.update(
                            {
                                (regime_name, key): payload
                                for key, payload in result.replay_artifacts.items()
                                if key != SIMULATION_POLICY
                                and (
                                    retain_replay
                                    or ArtifactRef(
                                        period=period,
                                        regime=regime_name,
                                        key=key,
                                    )
                                    in persistable_artifact_refs
                                )
                            }
                        )
                        if retain_all_artifacts:
                            period_auxiliary_artifacts.update(
                                {
                                    (regime_name, key): payload
                                    for key, payload in (
                                        result.auxiliary_artifacts.items()
                                    )
                                    if ArtifactRef(
                                        period=period,
                                        regime=regime_name,
                                        key=key,
                                    )
                                    in persistable_artifact_refs
                                }
                            )
                        # A collective regime publishes its
                        # empty-mask dissolution flag D alongside V; singleton regimes
                        # leave it None and never touch this mapping.
                        if result.dissolution is not None:
                            period_dissolution_flags[regime_name] = result.dissolution
                        if (
                            collect_solver_diagnostics
                            and diagnostics_enabled
                            and result.diagnostics is not None
                        ):
                            period_solver_diagnostics[regime_name] = result.diagnostics
                        running_any_nan, running_any_inf = _fold_period_diagnostics(
                            V_arr=V_arr,
                            regime_name=regime_name,
                            period=period,
                            ages=ages,
                            diagnostics_enabled=diagnostics_enabled,
                            stats_enabled=stats_enabled,
                            diagnostic_rows=diagnostic_rows,
                            diagnostic_min=diagnostic_min,
                            diagnostic_max=diagnostic_max,
                            diagnostic_mean=diagnostic_mean,
                            running_any_nan=running_any_nan,
                            running_any_inf=running_any_inf,
                        )

                        period_solution[regime_name] = V_arr
                        period_pending_outputs.append(V_arr)
                        if result.continuation is not None:
                            period_pending_outputs.extend(
                                jax.tree.leaves(result.continuation)
                            )
                        dispatch_outputs = (
                            V_arr,
                            result.continuation,
                            result.continuation_artifacts,
                            result.replay_artifacts,
                            result.auxiliary_artifacts,
                            result.simulation_policy,
                            result.dissolution,
                            _diagnostic_arrays(
                                diagnostics=()
                                if result.diagnostics is None
                                else (result.diagnostics,)
                            ),
                        )
                        # Whatever this dispatch handed straight back out, it did not
                        # produce. The inputs are read the way the dispatch reads them —
                        # through the same period-axis overlay — so an age-specialized
                        # axis is compared as the dispatch actually saw it, and
                        # including this period's own values, which a
                        # same-period-ref regime reads.
                        buffer_registry.declare_passed_through(
                            inputs=(
                                _states_for_period(
                                    regime=regime,
                                    state_action_space=base_state_action_spaces[
                                        regime_name
                                    ],
                                    period=period,
                                ),
                                next_regime_to_V_arr,
                                next_regime_to_continuation,
                                next_edge_to_V_arr,
                                period_solution,
                            ),
                            outputs=dispatch_outputs,
                        )
                        # The result keeps these payloads, and what would make them
                        # independent runs only once the period is finished — after the
                        # releases below, and for the dissolution flags not at all. Two
                        # channels carrying one array is enough for a release
                        # addressed at one of them to reach the other, so every
                        # retained channel is declared before this period's first
                        # release.
                        buffer_registry.declare_not_produced(
                            tree=(
                                period_retained_continuations,
                                period_replay_artifacts,
                                period_auxiliary_artifacts,
                                period_simulation_policies,
                                period_dissolution_flags,
                                _diagnostic_arrays(
                                    diagnostics=tuple(
                                        period_solver_diagnostics.values()
                                    )
                                ),
                            )
                        )
                        (
                            next_regime_to_V_arr,
                            next_regime_to_continuation,
                            next_edge_to_V_arr,
                        ) = _retire_donated_inputs(
                            donated_inputs=donated_inputs,
                            dispatch=(period, regime_name),
                            inputs=SolveInputMappings(
                                next_regime_to_V_arr=next_regime_to_V_arr,
                                next_regime_to_continuation=next_regime_to_continuation,
                                next_edge_to_V_arr=next_edge_to_V_arr,
                            ),
                            templates=input_templates,
                            pending_outputs=(
                                tuple(period_pending_outputs),
                                dispatch_outputs,
                            ),
                            registry=buffer_registry,
                            logger=logger,
                            before_delete=None
                            if pending_work is None
                            else pending_work.before_delete,
                        )
                    for unit in wave:
                        for key in regime_shared_transfer_keys.get(
                            unit.regime, frozenset()
                        ):
                            period_transfer_cache.commit_consumer(key=key)
                        for closed in input_liveness.commit_successful_dispatch(
                            dispatch=(period, unit.regime)
                        ):
                            period_release_candidates.setdefault(
                                closed, (period, unit.regime)
                            )
                    (
                        next_regime_to_V_arr,
                        next_regime_to_continuation,
                        next_edge_to_V_arr,
                    ) = _release_closed_period_inputs(
                        ledger=input_liveness,
                        registry=buffer_registry,
                        candidates=period_release_candidates,
                        inputs=SolveInputMappings(
                            next_regime_to_V_arr=next_regime_to_V_arr,
                            next_regime_to_continuation=next_regime_to_continuation,
                            next_edge_to_V_arr=next_edge_to_V_arr,
                        ),
                        templates=input_templates,
                        pending_outputs=period_pending_outputs,
                        logger=logger,
                        release_enabled=enable_jit,
                        before_delete=None
                        if pending_work is None
                        else pending_work.before_delete,
                    )

                # Force the device-side reduction kernels to finish before the
                # next period dispatches, so each period's `isnan` / `isinf`
                # (and min/max/mean) intermediate buffers can be freed instead
                # of stacking up. `block_until_ready` does NOT transfer to host
                # — it is a device-side wait, cheap when the dominant
                # per-period kernel (`max_Q_over_a`) is the actual bottleneck.
                if diagnostics_enabled:
                    running_any_nan.block_until_ready()
                    running_any_inf.block_until_ready()
                    if stats_enabled and diagnostic_mean:
                        # Blocking on the last-appended stat suffices: XLA
                        # serialises dispatch order, so a finished `mean`
                        # implies a finished `min`/`max` too.
                        diagnostic_mean[-1].block_until_ready()

                # Fold each declared gated edge whose target
                # was solved this period onto the target grid, and roll the resulting
                # Wbar into the edge continuation the source reads next period. Reads
                # only the still-live period-t arrays (`period_solution`,
                # `period_dissolution_flags`). The node fold is streamed to cap peak
                # memory; parents then read Wbar in place of the raw target V via the
                # existing next_regime_to_V_arr threading.
                folded_edge_to_V_arr = _roll_gated_edges(
                    regimes=regimes,
                    ages=ages,
                    period=period,
                    period_solution=period_solution,
                    period_dissolution_flags=period_dissolution_flags,
                    base_state_action_spaces=base_state_action_spaces,
                    flat_params=flat_params,
                    next_edge_to_V_arr=next_edge_to_V_arr,
                )
                # The fold is a dispatch unit in its own right, and it passes a great
                # deal through: an edge it does not fold keeps its previous Wbar, and
                # the sharding match a folded one ends in returns its argument whenever
                # the shardings already agree. Its grids and params need no mention —
                # those are declared for the whole solve before the first dispatch.
                buffer_registry.declare_passed_through(
                    inputs=(
                        next_edge_to_V_arr,
                        period_solution,
                        period_dissolution_flags,
                    ),
                    outputs=folded_edge_to_V_arr,
                )
                next_edge_to_V_arr = folded_edge_to_V_arr
                # A fold is a consumer of the period's raw values in its own right, so
                # each edge folded above commits the dispatch declared for it. An edge
                # the same enumeration left unfolded declared none and commits nothing.
                for folded_edge in _folded_edge_keys_at_period(
                    regimes=regimes,
                    period=period,
                    solved_regimes=period_solution,
                ):
                    for closed in input_liveness.commit_successful_dispatch(
                        dispatch=(period, *folded_edge)
                    ):
                        period_release_candidates.setdefault(
                            closed, (period, *folded_edge)
                        )
                (
                    next_regime_to_V_arr,
                    next_regime_to_continuation,
                    next_edge_to_V_arr,
                ) = _release_closed_period_inputs(
                    ledger=input_liveness,
                    registry=buffer_registry,
                    candidates=period_release_candidates,
                    inputs=SolveInputMappings(
                        next_regime_to_V_arr=next_regime_to_V_arr,
                        next_regime_to_continuation=next_regime_to_continuation,
                        next_edge_to_V_arr=next_edge_to_V_arr,
                    ),
                    templates=input_templates,
                    pending_outputs=period_pending_outputs,
                    logger=logger,
                    release_enabled=enable_jit,
                    before_delete=None
                    if pending_work is None
                    else pending_work.before_delete,
                )
                next_regime_to_V_arr, next_regime_to_continuation = (
                    _roll_continuation_inputs(
                        regimes=regimes,
                        period_solution=period_solution,
                        period_continuations=period_continuations,
                        next_regime_to_V_arr=next_regime_to_V_arr,
                        next_regime_to_continuation=next_regime_to_continuation,
                    )
                )
                solution[period] = MappingProxyType(period_solution)
                # Publish each collective regime's dissolution
                # flag D alongside V, where a reader exists. Kept as a plain per-period
                # mapping (not rolled like `next_regime_to_V_arr`): nothing consumes a
                # NEXT-period D — a gated edge's gate reads the still-live per-period
                # flags at each period's end, before the roll (above). The period keys
                # match `solution`'s either way; only the arrays behind them differ.
                dissolution_flags[period] = (
                    MappingProxyType(period_dissolution_flags)
                    if publish_dissolution_flags
                    else _NO_DISSOLUTION_FLAGS
                )
                if retain_replay or period_simulation_policies:
                    assert host_device is not None  # noqa: S101
                    simulation_policies[period] = MappingProxyType(
                        {
                            regime_name: jax.block_until_ready(
                                jax.device_put(simulation_policy, host_device)
                            )
                            for regime_name, simulation_policy in (
                                period_simulation_policies.items()
                            )
                        }
                    )
                if period_generated_replay_authorities:
                    generated_replay_authorities[period] = MappingProxyType(
                        period_generated_replay_authorities
                    )
                if period_solver_diagnostics:
                    assert host_device is not None  # noqa: S101
                    solver_diagnostics[period] = MappingProxyType(
                        {
                            regime_name: _copy_solver_diagnostics_to_host(
                                diagnostics=diagnostics,
                                host_device=host_device,
                            )
                            for regime_name, diagnostics in (
                                period_solver_diagnostics.items()
                            )
                        }
                    )
                if retain_all_artifacts:
                    assert host_device is not None  # noqa: S101
                    for (
                        regime_name,
                        key,
                    ), payload in period_retained_continuations.items():
                        retained_continuations[
                            ArtifactRef(period=period, regime=regime_name, key=key)
                        ] = jax.block_until_ready(jax.device_put(payload, host_device))
                    for (
                        regime_name,
                        key,
                    ), payload in period_auxiliary_artifacts.items():
                        auxiliary_artifacts[
                            ArtifactRef(period=period, regime=regime_name, key=key)
                        ] = jax.block_until_ready(jax.device_put(payload, host_device))
                if retain_replay or period_replay_artifacts:
                    assert host_device is not None  # noqa: S101
                    for (regime_name, key), payload in period_replay_artifacts.items():
                        replay_artifacts[
                            ArtifactRef(period=period, regime=regime_name, key=key)
                        ] = jax.block_until_ready(jax.device_put(payload, host_device))

                elapsed = time.monotonic() - period_start
                log_period_timing(logger=logger, elapsed=elapsed)

                # Fail-fast on NaN: surface the offending period immediately
                # instead of finishing the whole backward induction. Costs one
                # host transfer of a scalar bool per period — negligible next
                # to the per-period `max_Q_over_a` kernel. Inf is non-fatal so
                # we don't break on it; the post-loop emitter still raises a
                # warning if any period flagged Inf.
                #
                # Only raise mode fails fast. Raise mode is the loudest level, so
                # diagnostics are on and `running_any_nan` has been tracked. In warn
                # mode induction runs to completion so `solve` returns a complete
                # (NaN-bearing) solution rather than a truncated one.
                if validation_raises(logger) and running_any_nan.item():
                    break

                _release_rolled_continuations(period_continuations=period_continuations)

            if diagnostics_enabled:
                try:
                    _emit_post_loop_diagnostics(
                        logger=logger,
                        diagnostic_rows=diagnostic_rows,
                        solution=MappingProxyType(solution),
                        regimes=regimes,
                        flat_params=flat_params,
                        running_any_nan=running_any_nan,
                        running_any_inf=running_any_inf,
                        diagnostic_min=diagnostic_min if stats_enabled else None,
                        diagnostic_max=diagnostic_max if stats_enabled else None,
                        diagnostic_mean=diagnostic_mean if stats_enabled else None,
                        process_grid_resolver=process_grid_resolver,
                    )
                except InvalidValueFunctionError as error:
                    raise_or_warn(logger=logger, error=error)

            _drain_V_arr_shards(solution=solution, dissolution_flags=dissolution_flags)
            input_liveness.assert_solve_complete()

        finally:
            if pending_work is not None:
                pending_work.close()

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return BackwardInductionResult(
        value_functions=MappingProxyType(solution),
        simulation_policies=MappingProxyType(simulation_policies),
        generated_replay_authorities=MappingProxyType(generated_replay_authorities),
        dissolution_flags=MappingProxyType(dissolution_flags),
        diagnostics=MappingProxyType(solver_diagnostics),
        retained_continuations=ArtifactStore(retained_continuations),
        replay_artifacts=ArtifactStore(replay_artifacts),
        auxiliary_artifacts=ArtifactStore(auxiliary_artifacts),
    )


def _diagnostic_arrays(
    *, diagnostics: Sequence[SolverDiagnostics]
) -> tuple[object, ...]:
    """Return the field values of each diagnostic payload, flattened.

    `SolverDiagnostics` is a registered pytree, so walking one reaches exactly
    the arrays this returns. The enumeration is the solve loop's route because
    it makes the release invariant it feeds — every retained diagnostics buffer
    declared before the period's first release — a property of the payload's
    own fields rather than of a registration made for persistence.
    """
    return tuple(
        getattr(payload, field.name)
        for payload in diagnostics
        for field in dataclasses.fields(payload)
    )


def _copy_solver_diagnostics_to_host(
    *, diagnostics: SolverDiagnostics, host_device: jax.Device
) -> SolverDiagnostics:
    """Copy one retained diagnostic payload off the accelerator."""
    return dataclasses.replace(
        diagnostics,
        **{
            field.name: (
                None
                if (value := getattr(diagnostics, field.name)) is None
                else jax.block_until_ready(jax.device_put(value, host_device))
            )
            for field in dataclasses.fields(diagnostics)
        },
    )


def _release_rolled_continuations(
    *, period_continuations: dict[RegimeName, ContinuationPayload]
) -> None:
    """Free the device buffers rolled off the period just solved that no key
    still names.

    The superseded continuation inputs and the period's transient working set
    are unreferenced once the period rolls, but a rolled continuation payload
    sits in a registered pytree that CPython's cyclic collector frees only when
    it next runs — forcing a collection here frees the device pool promptly,
    capping peak resident across the loop (mirrors the forward-sim memory
    rework in `result.py`).

    Gated on whether this period actually produced a continuation (the generic
    per-period kernel output the loop already tracks), not on the solver type:
    a period whose kernels publish none rolls no such buffer, so the collection
    — which otherwise dominates small warm solves with no memory gain — is
    skipped for it.
    """
    if period_continuations:
        gc.collect()


def _run_period_kernel(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    compiled_cores: MappingProxyType[str, PlannedCore],
    capture_target: PeriodCaptureTarget | None,
    state_action_space: StateActionSpace,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    logger: logging.Logger,
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    period_solution: Mapping[RegimeName, FloatND],
    retain_replay: bool,
    selected_artifact_keys: frozenset[ArtifactKey],
) -> KernelOutput:
    """Invoke one regime's period adapter for one period.

    Every regime exposes the same kind of adapter; the loop never branches on
    solver type. The adapter wraps the regime's compiled programs (passed in
    AOT-compiled as `compiled_cores`), calls them with the solver's own argument
    layout, and returns the public `KernelOutput`: the value-function array plus
    the artifacts it publishes on its keyed channels, which the loop reads through
    `consume_kernel_output` and accumulates where something consumes them.

    A regime declaring `same_period_refs` additionally
    receives the referenced regimes' V arrays of THIS period, read off
    `period_solution` — the within-period topological order guarantees they were
    solved earlier in this period's loop. a source
    declaring `gated_edges` receives its own rolled Wbar arrays, keyed by target
    regime name, which the grid-search kernel substitutes for the raw target V in
    `next_regime_to_V_arr`. Every other regime's adapter is called with the
    unchanged uniform signature.

    `period`/`age` are passed as JAX arrays (not Python scalars) so a shared
    `jax.jit` function is traced once with abstract shapes, not recompiled
    for every distinct (period, age) pair.

    The adapter is handed its full per-key compiled-core map (`compiled_cores`):
    a single-core kernel reads `["main"]`, a multi-core kernel reads each of its
    own core keys.

    Returns:
        The kernel's output for this regime-period, exactly as returned.

    """
    period_kernel = regime.solution.period_kernels[period]

    # Captured before the period-specific state axes are substituted below. Replay
    # re-enters this funnel with capture explicitly disabled.
    capture_kernel_inputs(
        capture_target=capture_target,
        regime=regime,
        regime_name=regime_name,
        period=period,
        compiled_cores=compiled_cores,
        kernel_kwargs={
            "regime_name": regime_name,
            "period": period,
            "state_action_space": state_action_space,
            "flat_params": flat_params,
            "ages": ages,
            "next_regime_to_V_arr": next_regime_to_V_arr,
            "next_regime_to_continuation": next_regime_to_continuation,
            "logger": logger,
            "next_edge_to_V_arr": next_edge_to_V_arr,
            "period_solution": period_solution,
            "retain_replay": retain_replay,
            "selected_artifact_keys": selected_artifact_keys,
        },
    )

    # AGE-SPECIALIZED STATES: tabulate period-t's value function on period-t's grid
    # nodes, not on the representative base axis. This is what keeps the tabulation
    # on the same grid as the continuation, which reads V_{t+1} on period-(t+1)'s
    # grid; the two halves disagreeing makes the solved value function wrong at
    # every node, not merely imprecise. Same shape as the base, so the shared
    # compiled core is not retraced.
    #
    # This consumer was DROPPED by cascade merge 80f5e79 ("Cascade
    # feat/age-specialized into feat/dcegm"). The age-specialized side called
    # `_states_for_period` in exactly two places -- the solve hot loop and the
    # failure-path reconstruction -- and the merge kept only the second, which moved
    # into `diagnostics.py`. `_build_period_state_axes` kept computing the axes and
    # `SolutionPhase.period_state_axes` kept carrying them, so nothing looked broken:
    # the data was still built and stored, just never read by the solver. Every
    # period then solved on the base axis, which is wrong exactly where the
    # age-specific grid diverges from it -- the last pre-retirement ages -- and
    # showed up as `-inf` in the worker value function at ages 57-59 in
    # blundellFemaleLaborSupply2016.
    state_action_space = dataclasses.replace(
        state_action_space,
        states=MappingProxyType(
            dict(
                _states_for_period(
                    regime=regime,
                    state_action_space=state_action_space,
                    period=period,
                )
            )
        ),
    )

    log_executed_kernel(
        regime_name=regime_name,
        period=period,
        ages=ages,
        state_action_space=state_action_space,
        core_keys=tuple(compiled_cores),
        logger=logger,
    )

    same_period_kwargs: dict[str, object] = {}
    if regime.same_period_ref_regimes:
        same_period_kwargs["same_period_regime_to_V_arr"] = MappingProxyType(
            {
                ref_regime_name: period_solution[ref_regime_name]
                for ref_regime_name in regime.same_period_ref_regimes
            }
        )
    same_period_kwargs.update(
        _edge_kwargs(
            regime=regime,
            regime_name=regime_name,
            next_edge_to_V_arr=next_edge_to_V_arr,
        )
    )
    return period_kernel(
        compiled_cores=compiled_cores,
        state_action_space=state_action_space,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        flat_params=flat_params,
        period=period,
        ages=ages,
        logger=logger,
        **same_period_kwargs,
    )


def _cores_with_transfer_cache(
    *,
    cores: MappingProxyType[str, PlannedCore],
    cache: PeriodTransferCache,
    pending_work: PendingSolveWork | None = None,
) -> MappingProxyType[str, PlannedCore]:
    """Hand one period's transfer cache to every core a kernel dispatches."""
    return MappingProxyType(
        {
            core_key: dataclasses.replace(
                core, transfer_cache=cache, pending_work=pending_work
            )
            for core_key, core in cores.items()
        }
    )


def _regime_device_ids(
    *, regime: Regime, visible_device_ids: tuple[int, ...]
) -> frozenset[int]:
    """Return the ids of the devices one regime's nodes are dispatched on.

    An empty placement names no submesh, so the regime runs on every device the
    model uses.

    Args:
        regime: The canonical regime whose nodes are being placed.
        visible_device_ids: The model's device ids, ascending.

    Returns:
        Frozenset of the device ids the regime's dispatches occupy.

    """
    return frozenset(regime.solution.submesh_device_ids) or frozenset(
        visible_device_ids
    )


def _period_shared_transfer_plan(
    *, compiled_cores_by_regime: Mapping[RegimeName, MappingProxyType[str, PlannedCore]]
) -> tuple[
    MappingProxyType[tuple[Hashable, Hashable], int],
    MappingProxyType[RegimeName, frozenset[tuple[Hashable, Hashable]]],
]:
    """Count, per shared transfer, how many of this period's regimes read it.

    A regime dispatch commits once for every core it runs, so the count that
    matters for release is per regime, not per core: two cores of one regime
    reading a shared transfer still leave it needing only that regime's own
    commit. An `ALIGNED_LOCAL` transfer names no copy for the cache to hold and
    is excluded, whatever its `reused_by_several_consumers` mark.

    Returns:
        Tuple of the declared consumer count per shared-transfer key, and the
        set of shared-transfer keys each regime's dispatch commits.

    """
    keys_by_regime: dict[RegimeName, set[tuple[Hashable, Hashable]]] = {}
    for regime_name, cores in compiled_cores_by_regime.items():
        regime_keys: set[tuple[Hashable, Hashable]] = set()
        for core in cores.values():
            for transfer in core.input_transfer_plan:
                if (
                    transfer.reused_by_several_consumers
                    and transfer.kind is not ValueTransferKind.ALIGNED_LOCAL
                ):
                    regime_keys.add((transfer.target, transfer.source_sharding))
        if regime_keys:
            keys_by_regime[regime_name] = regime_keys
    consumer_counts: dict[tuple[Hashable, Hashable], int] = {}
    for regime_keys in keys_by_regime.values():
        for key in regime_keys:
            consumer_counts[key] = consumer_counts.get(key, 0) + 1
    return (
        MappingProxyType(consumer_counts),
        MappingProxyType(
            {
                regime_name: frozenset(keys)
                for regime_name, keys in keys_by_regime.items()
            }
        ),
    )


def _roll_continuation_inputs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    period_solution: dict[RegimeName, FloatND],
    period_continuations: dict[RegimeName, ContinuationPayload],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
]:
    """Roll the per-period continuation mappings forward by one period.

    Both mappings keep their full template key sets — V for every regime,
    carries for every carry-producing regime — and update only the entries
    solved this period, so the pytree structure stays JIT-stable.

    The `.get(..., prior)` fallback is for regimes *inactive* this period: they
    keep the prior period's entry. It relies on the invariant that every
    continuation-publishing regime publishes on each of its active periods — the
    solve loop enforces this before rolling, so an active publisher can never
    fall through to the stale prior carry here.

    Returns:
        Tuple of the rolled V mapping and the rolled carry mapping.

    """
    rolled_V_arr = MappingProxyType(
        {
            regime_name: _match_leaf_template_sharding(
                leaf=period_solution[regime_name],
                template_leaf=next_regime_to_V_arr[regime_name],
            )
            if regime_name in period_solution
            else next_regime_to_V_arr[regime_name]
            for regime_name in regimes
        }
    )
    rolled_continuation = MappingProxyType(
        {
            regime_name: _match_continuation_template_sharding(
                continuation=period_continuations[regime_name],
                template=next_regime_to_continuation[regime_name],
            )
            if regime_name in period_continuations
            else next_regime_to_continuation[regime_name]
            for regime_name in next_regime_to_continuation
        }
    )
    return rolled_V_arr, rolled_continuation


# A gated edge's continuation slot is keyed by the
# (source regime, target regime) pair — a source has at most one edge per target,
# and the same target is read raw by other regimes, so the edge cannot share the
# plain regime-keyed V slot.
type _EdgeKey = tuple[RegimeName, RegimeName]


def _roll_gated_edges(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    ages: AgeGrid,
    period: int,
    period_solution: dict[RegimeName, FloatND],
    period_dissolution_flags: dict[RegimeName, BoolND],
    base_state_action_spaces: dict[RegimeName, StateActionSpace],
    flat_params: FlatParams,
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
) -> MappingProxyType[_EdgeKey, FloatND]:
    """Fold every gated edge whose target was solved this period; roll the rest.

    For each declared edge whose target regime (and
    every reference regime it reads) was solved in the period just completed,
    evaluate its `Wbar` producer on the still-live period-`t` arrays and
    store it; edges whose target is inactive this period keep their previous
    `Wbar` (the roll semantics of `next_regime_to_V_arr`). Keeps the full key
    set so the pytree structure stays JIT-stable.

    Which of the two an edge gets is `edge_may_fold_at_period`'s answer, the
    same one forward simulation consults. The fold at period `t` is read by
    the source at `t - 1`, so whether a source exists there decides what an
    unsolved reference regime means: at an unread period it is the legitimate
    boundary no-op of a self-loop edge at its target's earliest active period,
    and the previous `Wbar` stands; at a read period it is a misconfigured
    edge, and the fold refuses rather than feed the source a stale value.

    The gate and the projections are evaluated on the target's grid nodes at
    `period` — the same nodes the target's value function being folded was
    tabulated on. An `AgeSpecializedGrid` keeps `n_points` fixed while its
    bounds move with age, so reading the representative axis instead passes
    every shape check and folds the value at the wrong coordinates.
    """
    if not next_edge_to_V_arr:
        return next_edge_to_V_arr
    rolled: dict[_EdgeKey, FloatND] = dict(next_edge_to_V_arr)
    for source_name, target_name in _folded_edge_keys_at_period(
        regimes=regimes,
        period=period,
        solved_regimes=period_solution,
    ):
        edge = regimes[source_name].gated_edges[target_name]
        # The fold compiled for THIS period: the gate references and leg
        # fallbacks are interpolated on their own regimes' grids as of the
        # period being folded, which an `AgeSpecializedGrid` moves without
        # changing their shape.
        fold = edge.fold_at(period=period)
        same_period_mapping = build_same_period_mapping_for_fold(
            edge=edge,
            period_solution=period_solution,
            period_dissolution_flags=period_dissolution_flags,
        )
        wbar = _evaluate_edge_fold(
            fold=fold,
            fold_period=period,
            fold_age=ages.period_to_age(period),
            target_states=cast(
                "Mapping[str, ContinuousState | DiscreteState]",
                _states_for_period(
                    regime=regimes[target_name],
                    state_action_space=base_state_action_spaces[target_name],
                    period=period,
                ),
            ),
            same_period_mapping=same_period_mapping,
            source_flat_params=flat_params[source_name],
            reference_flat_params=build_reference_params_mapping_for_fold(
                edge=edge, flat_params=flat_params
            ),
        )
        rolled[(source_name, target_name)] = _match_leaf_template_sharding(
            leaf=wbar,
            template_leaf=next_edge_to_V_arr[(source_name, target_name)],
        )
    return MappingProxyType(rolled)


def _reject_edge_fold_state_param_collisions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace],
    flat_params: FlatParams,
) -> None:
    """Reject a gated edge whose fold binds one leaf as BOTH a target state and a
    source param.

    A gate / gate-ref projection / fallback projection declares its arguments by
    bare name. `get_edge_fold` exposes the target's state grids and the source's
    gate/projection params in ONE flat signature, so a name that is simultaneously
    a TARGET STATE of the target regime and a key of `flat_params[source]` occupies
    a single leaf that two binders both claim: `_evaluate_edge_fold` (below)
    overwrites the state grid with the source param, so the SOLVE-side `Wbar`
    reads the param, while the simulate evaluator's `_expose`
    (`get_edge_simulate_gate_evaluator`) classifies the same name as a state
    BEFORE it would record a source param, so the SIMULATE-side gate reads the
    realized target state. Solve and simulate then evaluate DIFFERENT predicates
    for the same edge -- the gate flips, `Wbar` changes, or a fallback
    coordinate is written from the wrong value, all silently.

    Why this is a solve-time (not construction-time) fence: a gate/projection
    param is bound from a BARE key the user adds to `flat_params[source]`, never
    from the function-qualified regime params template, so it is absent from
    `regime_to_flat_param_names[source]` and the collision is only visible once
    `flat_params` is in hand. A LEGITIMATE direct target-state read (a gate that
    reads a target state the source never supplies as a param -- e.g. a reused
    state NAME across two regimes) is untouched, because that name is not a key of
    `flat_params[source]`.
    """
    for source_name, source in regimes.items():
        if not source.gated_edges:
            continue
        source_param_names = set(flat_params[source_name])
        for target_name, edge in source.gated_edges.items():
            compiled_folds = tuple(edge.folds_by_period.values())
            if not compiled_folds:
                # The target regime is active in no period, so it holds no
                # value to fold and no fold was compiled — there is no
                # signature to check, and no `Wbar` this edge could ever feed.
                continue
            # Any compiled period answers: a fold's signature is built from
            # names — the target's states, the gate's and the projections'
            # parameters — and an `AgeSpecializedGrid` may vary only its nodes,
            # never a grid's class, shape, or points mode. So every period's
            # fold exposes the same leaves, and the collisions this rejects are
            # a property of the edge rather than of one period.
            # Every name this edge binds, on BOTH sides of the seam: the fold's
            # operand surfaces, the combiner that gates them (which carries the
            # projected readers), and the simulate gate evaluator. The check
            # below is about a name meaning one thing in solve and another in
            # simulate, so reading one side's signature alone would miss exactly
            # the names only the other side declares.
            evaluators = tuple(edge.simulate_gate_evaluators_by_period.values())
            sig_params = set().union(
                *(
                    set(inspect.signature(func).parameters)
                    for func in (
                        compiled_folds[0].surfaces,
                        compiled_folds[0].combine.combine,
                        *evaluators[:1],
                    )
                )
            )
            target_state_names = set(base_state_action_spaces[target_name].states)
            collisions = sorted(sig_params & target_state_names & source_param_names)
            if collisions:
                msg = (
                    f"The gated edge '{source_name}' -> '{target_name}' has a gate "
                    f"or projection argument {collisions} that is simultaneously a "
                    f"TARGET state of '{target_name}' and a source parameter in "
                    f"`flat_params['{source_name}']`. The fold's single leaf for "
                    "each such name is bound as the source param on the solve side "
                    "(`_evaluate_edge_fold`) but as the realized target state on the "
                    "simulate side (`get_edge_simulate_gate_evaluator`), so the "
                    "solved `Wbar` and the simulate router would evaluate different "
                    "gates. Rename the source parameter (or the target state) so the "
                    "two namespaces are disjoint."
                )
                raise ModelInitializationError(msg)
            # A source flat-param key (or target state) that shadows one of the
            # internal ENGINE argument names is a second solve/simulate divergence
            # of the same class: on the solve side
            # `_evaluate_edge_fold` binds `SAME_PERIOD_V_ARG` to the value mapping
            # and `SAME_PERIOD_PARAMS_ARG` to the reference params, then overwrites
            # those slots with any same-named source flat-param; on the simulate
            # side `_expose` classifies the identical spelling as the engine arg
            # BEFORE it can be recorded as a source param. So a source scalar named
            # `same_period_regime_to_params` opens the gate on solve (scalar) but
            # closes it on simulate (mapping), and a source
            # `same_period_regime_to_V_arr` overwrites the value mapping outright.
            # Reserve the engine names against both source params and target
            # states, whether or not THIS edge binds them. Which of the two
            # params mappings an edge names depends on its topology — a gate
            # reference reads one, a leg fallback the other — so keying the
            # reservation on the signature would let the same spelling be a
            # source param under one topology and engine vocabulary under a
            # neighbouring one. The absence of the name from a fold is what
            # makes it dangerous, not its presence: it is then qualified into
            # the target namespace and fails much later, inside solve.
            engine_args = {
                SAME_PERIOD_V_ARG,
                SAME_PERIOD_PARAMS_ARG,
                EDGE_REF_V_ARG,
                EDGE_REF_PARAMS_ARG,
                *EDGE_PERIOD_CONTEXT_ARGS,
            }
            engine_collisions = sorted(
                engine_args & (source_param_names | target_state_names)
            )
            if engine_collisions:
                msg = (
                    f"The gated edge '{source_name}' -> '{target_name}' has a gate "
                    f"or projection argument {engine_collisions} that shadows a "
                    "reserved internal engine argument name "
                    f"({sorted(engine_args)}). Such a name is bound as the source "
                    "parameter / target state on one side of the solve/simulate seam "
                    "but as the engine's value/params mapping on the other, so the "
                    "solved `Wbar` and the simulate router would evaluate different "
                    "gates (or crash when a source value overwrites the value "
                    "mapping). Rename the source parameter (or the target state) so "
                    "it does not collide with a reserved engine argument."
                )
                raise ModelInitializationError(msg)


def _evaluate_edge_fold(
    *,
    fold: CompiledEdgeFold,
    fold_period: int,
    fold_age: object,
    target_states: Mapping[str, ContinuousState | DiscreteState],
    same_period_mapping: Mapping[RegimeName, FloatND],
    source_flat_params: Mapping[str, object],
    reference_flat_params: Mapping[RegimeName, Mapping[str, object]],
    shared_sharding: jax.sharding.Sharding | None = None,
) -> FloatND:
    """Call one edge's fold with exactly the arguments its signature declares.

    Every parameter the fold needs is bound from the SOURCE regime — the fold is
    the source's own continuation object, and its gate / projections are declared
    on the source, so this is the namespace they are written against. (It is also
    the contract the simulate-side gate evaluator and leg projectors must match
    argument for argument; see `_lcm.regime_building.gated_edges
    .EdgeArgProvenance`.) The one exception is a REFERENCE regime's own
    interpolation grid, which belongs to neither the source nor the target:
    those params ride in `reference_flat_params` under
    `Q_and_F.SAME_PERIOD_PARAMS_ARG`, keyed by regime, and the reference readers
    resolve them internally.

    The target regime's grid may carry DISCRETE state axes (an encoded
    categorical, or any other `DiscreteGrid` state) alongside continuous ones,
    so `target_states` is typed as `base_state_action_spaces[target_name].
    states` is at the source — `ContinuousState | DiscreteState`
    (`_lcm.engine.StateActionSpace.states`), not float-only. Narrowing it to
    `FloatND` makes a discrete state raise `BeartypeCallHintParamViolation` at
    the `int32`-vs-float check inside `fold`, even though `get_edge_fold`'s
    `jnp.meshgrid` state broadcast tolerates either dtype.
    """
    surfaces = fold.surfaces
    sig_params = set(inspect.signature(surfaces).parameters)
    kwargs: dict[str, object] = {
        name: arr for name, arr in target_states.items() if name in sig_params
    }
    kwargs.update(
        {
            name: value
            for name, value in source_flat_params.items()
            if name in sig_params
        }
    )
    kwargs.update(
        bind_edge_period_context(
            func=surfaces,
            fold_period=fold_period,
            fold_age=cast("float | ScalarFloat | ScalarInt | None", fold_age),
        )
    )
    kwargs[SAME_PERIOD_V_ARG] = same_period_mapping
    if SAME_PERIOD_PARAMS_ARG in sig_params:
        kwargs[SAME_PERIOD_PARAMS_ARG] = reference_flat_params
    if shared_sharding is not None:
        # The caller owns value-copy lifetimes. Only ordinary fold operands
        # (including the period and age just bound above) are placed here.
        kwargs = {
            name: value
            if name == SAME_PERIOD_V_ARG
            else jax.device_put(value, shared_sharding)
            for name, value in kwargs.items()
        }
    return surfaces(**kwargs)


def _match_continuation_template_sharding(
    *, continuation: ContinuationPayload, template: ContinuationPayload
) -> ContinuationPayload:
    """Place a solved period's continuation on its template's device sharding.

    The parent's cores are AOT-compiled against the continuation template, so
    the template's per-leaf sharding is the calling convention. A producer can
    emit mixed-sharding leaves (value rows derived from the sharded value
    array, endogenous-grid rows broadcast replicated from the asset grid);
    every leaf is placed onto its template counterpart's sharding, a no-op
    where they already agree. Assumes the template of a distributed regime is
    itself sharded — an unsharded template under a distributed state would
    pull the continuation onto one device.
    """
    return jax.tree.map(_match_leaf_pair_sharding, continuation, template)


# keyword-only-exempt: library-callback=jax.tree.map
def _match_leaf_pair_sharding(leaf: FloatND, template_leaf: FloatND) -> FloatND:
    """Place one continuation leaf on its template leaf's sharding."""
    return _match_leaf_template_sharding(leaf=leaf, template_leaf=template_leaf)


def _publish_kernel_value(
    *,
    value: FloatND,
    compiled_cores: Mapping[str, Callable],
) -> FloatND:
    """Publish a period value after asserting its planned placement.

    Every compiled core has already asserted its complete runtime output tree
    against the layout used to lower it at the compiled-core seam; here only the
    value leaf the loop publishes is checked again, since the kernel may have
    unpacked the rest of its tree into channels. The check reads the first
    compiled core the period dispatched, whatever its name, since a
    retention-scoped graph compiles one program under one name and another under
    another. A core without a resolved layout cannot publish a value: nothing is
    re-placed here. Continuation rolling keeps its independent placement repair,
    which is a different producer/consumer boundary.
    """
    planned = tuple(
        core for core in compiled_cores.values() if isinstance(core, PlannedCore)
    )
    if len(planned) != len(compiled_cores):
        unplanned = tuple(
            name
            for name, core in compiled_cores.items()
            if not isinstance(core, PlannedCore)
        )
        msg = (
            "A period value can be published only through a PlannedCore; "
            f"{unplanned!r} carry no resolved layout."
        )
        raise TypeError(msg)
    if not planned:
        msg = "A period dispatched no compiled core, so it publishes no value."
        raise ValueError(msg)
    assert_value_leaf_layout(value=value, layout=planned[0].layout)
    return value


def _match_leaf_template_sharding(*, leaf: FloatND, template_leaf: FloatND) -> FloatND:
    """Place one solved array on its template's device sharding (no-op on match).

    Applied where a solved value array is published and where the continuation
    mappings roll forward, for the same reason as the continuations: a compiled
    kernel's output sharding is the backend's choice, so a value array can
    arrive replicated while the templates every consumer (parent cores and the
    AOT-lowered simulate programs) was lowered against are sharded.
    """
    if leaf.sharding == template_leaf.sharding:
        return leaf
    return jax.device_put(leaf, template_leaf.sharding)


def _fail_if_continuation_publisher_returned_none(
    *,
    result: ConsumedKernelOutput,
    regime_name: RegimeName,
    period: int,
    continuation_publishers: Mapping[RegimeName, ContinuationPayload],
) -> None:
    """Fail loud if a continuation-publishing regime published nothing.

    A regime with a continuation template MUST publish a continuation on every
    active period. If its kernel returns None, `_roll_continuation_inputs` would
    silently roll the stale prior period's carry forward — wrong numbers, not a
    crash — so surface the offending (regime, period) instead.
    """
    if result.continuation is None and regime_name in continuation_publishers:
        msg = (
            f"Regime '{regime_name}' declares a continuation template but its "
            f"kernel returned no continuation in active period {period}. A "
            f"continuation-based solver must publish a continuation on every "
            f"active period."
        )
        raise RuntimeError(msg)


def _build_continuation_templates(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    device_ids: tuple[int, ...] = (),
    process_grid_resolver: ProcessGridResolver | None = None,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
    MappingProxyType[_EdgeKey, FloatND],
]:
    """Build the period-invariant continuation-input templates.

    All mappings keep the same pytree structure (keys and shapes) across all
    periods, avoiding JIT re-compilation from pytree mismatches:

    - the V template holds a zero array per regime, shaped (and sharded) like
      the regime's V array;
    - the continuation template holds entries only for continuation-publishing
      regimes, in the key order reused every period;
    - the gated-edge template holds a zero `Wbar` per declared edge,
      shaped like the target regime's V state grid plus the source regime's
      stakeholder axis (a singleton source: the target grid alone). Empty for
      models without gated edges, so the default path only gains an empty third
      mapping.
    """
    regime_V_topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
        device_ids=device_ids,
        process_grid_resolver=process_grid_resolver,
    )
    next_regime_to_V_arr = MappingProxyType(
        {
            regime_name: _build_zero_V_arr(topology=topology)
            for regime_name, topology in regime_V_topology.items()
        }
    )
    next_regime_to_continuation = MappingProxyType(
        {
            regime_name: regime.solution.continuation_template
            for regime_name, regime in regimes.items()
            if regime.solution.continuation_template is not None
        }
    )
    next_edge_to_V_arr = MappingProxyType(
        {
            (source_name, target_name): _build_zero_V_arr(topology=topology)
            for source_name, target_name, topology in _iter_edge_topologies(
                regimes=regimes,
                flat_params=flat_params,
                device_ids=device_ids,
                process_grid_resolver=process_grid_resolver,
            )
        }
    )
    return next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr


def _edge_kwargs(
    *,
    regime: Regime,
    regime_name: RegimeName,
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
) -> dict[str, object]:
    """Build a source kernel's gated-edge `Wbar` argument, keyed by target.

    The kernel substitutes each entry for the raw target V in
    `next_regime_to_V_arr`. Lowering and execution both go through this one
    function, so the pytree the kernel is compiled against is the pytree it is
    called with. Empty for a regime declaring no gated edge.
    """
    if not regime.gated_edges:
        return {}
    return {
        "edge_regime_to_V_arr": MappingProxyType(
            {
                target_name: next_edge_to_V_arr[(regime_name, target_name)]
                for target_name in regime.gated_edges
            }
        )
    }


def _iter_edge_topologies(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    device_ids: tuple[int, ...] = (),
    process_grid_resolver: ProcessGridResolver | None = None,
) -> Iterator[tuple[RegimeName, RegimeName, _RegimeVTopology]]:
    """Yield `(source, target, Wbar topology)` for every declared gated edge.

    An edge's continuation lands on the target regime's state grid, so its axes
    — and the device sharding a `distributed=True` target state asks for — are
    the target's, built by the same sharding plan the target's own V template
    goes through. On top of them sits one replicated channel axis carrying the
    operands the gate and the branches are built from: the channels differ in
    which surface they hold, not in which slice of the target grid they read.

    Both the state-action space and the sharding plan are the target's alone,
    so they are built once per target however many sources reach it. The space
    completes runtime grids from params, which is the expensive half.

    `device_ids` names the model's devices, ascending; empty names every device
    JAX reports.
    """
    target_shapes: dict[RegimeName, tuple[int, ...]] = {}
    target_shardings: dict[RegimeName, jax.sharding.Sharding] = {}
    for source_name, source in regimes.items():
        if not source.gated_edges:
            continue
        for target_name in source.gated_edges:
            if target_name not in target_shapes:
                target = regimes[target_name]
                target_states = target.solution.state_action_space(
                    regime_params=flat_params[target_name],
                    process_grid_resolver=process_grid_resolver,
                ).states
                target_shapes[target_name] = tuple(
                    len(v) for v in target_states.values()
                )
                devices = placed_devices_for_ids(
                    submesh_device_ids=target.solution.submesh_device_ids,
                    visible_device_ids=device_ids,
                )
                target_shardings[target_name] = placed_V_sharding(
                    sharding_plan=_build_regime_sharding(
                        grids=target.solution.grids,
                        sharded_state_names=target.solution.sharded_state_names,
                        devices=devices,
                    ),
                    state_order=tuple(target_states),
                    devices=devices,
                )
            shape = target_shapes[target_name]
            sharding = target_shardings[target_name]
            n_channels = source.gated_edges[target_name].channels.count
            if n_channels:
                shape = (*shape, n_channels)
                if isinstance(sharding, jax.NamedSharding):
                    sharding = jax.NamedSharding(
                        mesh=sharding.mesh, spec=jax.P(*sharding.spec, None)
                    )
            yield (
                source_name,
                target_name,
                _RegimeVTopology(shape=shape, sharding=sharding),
            )


def _build_base_state_action_spaces(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> dict[RegimeName, StateActionSpace]:
    """Build each regime's params-completed state-action space once.

    The space is period-invariant within one solve (params are fixed), so
    runtime-grid completion (e.g. process gridpoint computation) runs once
    per regime instead of once per period-regime iteration.
    """
    return {
        regime_name: regime.solution.state_action_space(
            regime_params=flat_params[regime_name],
            process_grid_resolver=process_grid_resolver,
        )
        for regime_name, regime in regimes.items()
    }


def _drain_V_arr_shards(
    *,
    solution: dict[int, MappingProxyType[RegimeName, FloatND]],
    dissolution_flags: dict[int, MappingProxyType[RegimeName, BoolND]] | None = None,
) -> None:
    """Block until every V_arr (and dissolution-flag) shard is materialised.

    Solve → simulate barrier: backward induction returns sharded V_arrs,
    but the simulate phase must consume materialised arrays rather than
    in-flight kernels. Explicitly traverse the period → regime return schema
    before handing its array leaves to JAX: the immutable inner mappings are a
    public return boundary, not a synchronization mechanism whose correctness
    should depend on global pytree registration. The batched barrier blocks
    per-shard (no host transfer, no cross-device collective); free when kernels
    are already done, the minimum necessary sync when they are not. V stays
    sharded across devices. The collective dissolution flags ride along in the
    same barrier.
    """
    array_leaves = tuple(
        array
        for period_mapping in (solution, dissolution_flags)
        if period_mapping is not None
        for regime_mapping in period_mapping.values()
        for array in regime_mapping.values()
    )
    jax.block_until_ready(array_leaves)


type _InputDispatch = tuple[int, RegimeName] | tuple[int, RegimeName, RegimeName]
type _CoreTriple = tuple[RegimeName, int, str]
type _WidthKey = tuple[tuple[str, int], ...]
type _CoreCandidate = tuple[_CoreTriple, _WidthKey]
type _ConsumerKey = tuple[int, ValueArtifactAddress, Hashable]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ProgramExecutionMetadata:
    """Resolved declaration facts retained without pinning argument templates."""

    requirements: CoreExecutionRequirements
    disposition: CoreExecutionDisposition
    scope: ProgramScope
    input_transfer_plan: tuple[ResolvedValueTransfer, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CompiledPrograms:
    """Executable graph plus the metadata liveness reads through the same seam."""

    executables: dict[tuple[RegimeName, int], MappingProxyType[str, PlannedCore]]
    metadata: MappingProxyType[_CoreTriple, _ProgramExecutionMetadata]

    input_liveness: PlannedInputLiveness[_InputDispatch, ValueArtifactAddress]
    """The ledger the executables were lowered against; the loop commits to it."""

    donations: MappingProxyType[_CoreTriple, tuple[ResolvedDonation, ...]]
    """Per selected program, the donation decisions its executable carries."""

    donation_fallbacks: MappingProxyType[_CoreTriple, PlannedCore] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    """Admitted ordinary alternatives at exactly the selected donating widths."""


def _select_runtime_donation_cores(
    *,
    compiled_programs: _CompiledPrograms,
    unit: DispatchUnit,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    registry: BufferRegistry,
    logger: logging.Logger,
) -> tuple[
    MappingProxyType[str, PlannedCore],
    MappingProxyType[_CoreTriple, tuple[ResolvedDonation, ...]],
]:
    """Use an already-admitted ordinary core when actual ownership bars donation."""
    cores = dict(compiled_programs.executables[(unit.regime, unit.period)])
    decisions: dict[_CoreTriple, tuple[ResolvedDonation, ...]] = {}
    for name in unit.programs:
        triple = (unit.regime, unit.period, name)
        donations = compiled_programs.donations.get(triple, ())
        reasons = tuple(
            reason
            for donation in donations
            if donation.donated
            for artifact in donation.artifacts
            if (
                reason := _donation_ownership_refusal(
                    artifact=artifact,
                    inputs=inputs,
                    templates=templates,
                    registry=registry,
                )
            )
            is not None
        )
        if reasons:
            cores[name] = compiled_programs.donation_fallbacks[triple]
            decisions[triple] = ()
            logger.debug("donation withheld at %r: %s", triple, "; ".join(reasons))
        else:
            decisions[triple] = donations
    return MappingProxyType(cores), MappingProxyType(decisions)


def _donation_ownership_refusal(
    *,
    artifact: ValueArtifactAddress,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    registry: BufferRegistry,
) -> str | None:
    """Explain an actual input's ownership conflict without consuming the input."""
    array = locate_artifact(inputs=inputs, artifact=artifact)
    if array is None or array is locate_artifact(inputs=templates, artifact=artifact):
        return f"{artifact!r} is not a solve input of its own"
    if registry.is_not_produced(array=array):
        return f"{artifact!r} has a buffer no compiled executable produced"
    partners = registry.artifacts_sharing(array=array) - {artifact}
    if partners:
        return f"{artifact!r} has a buffer {sorted(partners, key=repr)!r} still name"
    return None


def _build_planned_input_liveness(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    retain_all_artifacts: bool,
    persistable_artifact_refs: frozenset[ArtifactRef],
) -> PlannedInputLiveness[_InputDispatch, ValueArtifactAddress]:
    """Build one exact-dispatch ledger without authorizing physical release.

    Every declared read of every program is a counted consumer, whatever the
    program's disposition. What the ledger retains is what the selected
    retention keeps in the solve result: every regime value, and every leaf of
    every continuation payload the persistence-oriented retention holds. What
    stays pinned is what no declaration covers: the host-read continuation
    leaves of the EGM family and every reachable input of a program that
    declares no reads. A rolled entry whose producer does not run at its period
    aliases the same address one period later.
    """
    dispatch_accesses: dict[_InputDispatch, tuple[ValueArtifactAddress, ...]] = dict(
        _gated_edge_fold_dispatches(regimes=regimes)
    )
    pinned_artifacts: set[ValueArtifactAddress] = set()

    metadata_by_dispatch: dict[
        tuple[RegimeName, int], dict[str, _ProgramExecutionMetadata]
    ] = {}
    for (regime_name, period, core_name), metadata in program_metadata.items():
        metadata_by_dispatch.setdefault((regime_name, period), {})[core_name] = metadata

    for (regime_name, period), programs in metadata_by_dispatch.items():
        declared, declares_no_reads = _classify_dispatch_value_artifacts(
            programs=programs,
        )
        dispatch_accesses[(period, regime_name)] = declared
        pinned_artifacts.update(
            undeclared_read_pins(
                regimes=regimes,
                regime_name=regime_name,
                period=period,
                declares_no_reads=declares_no_reads,
            )
        )

    retained = _retained_solution_artifacts(
        regimes=regimes,
        retain_all_artifacts=retain_all_artifacts,
        persistable_artifact_refs=persistable_artifact_refs,
    )
    known = {
        *retained,
        *pinned_artifacts,
        *(artifact for reads in dispatch_accesses.values() for artifact in reads),
    }
    return PlannedInputLiveness(
        dispatch_accesses=MappingProxyType(dispatch_accesses),
        pinned_artifacts=pinned_artifacts,
        retained_artifacts=retained,
        aliases=_rolled_aliases(regimes=regimes, artifacts=known),
    )


def _release_closed_period_inputs(
    *,
    ledger: PlannedInputLiveness[_InputDispatch, ValueArtifactAddress],
    registry: BufferRegistry,
    candidates: dict[ValueArtifactAddress, _InputDispatch],
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    pending_outputs: Sequence[FloatND],
    logger: logging.Logger,
    release_enabled: bool,
    before_delete: BeforeArrayDelete | None = None,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
    MappingProxyType[_EdgeKey, FloatND],
]:
    """Release every candidate the period's inputs still hold and substitute it.

    Candidates are the keys the period's commits closed, each with the dispatch
    that closed it; they are consumed here, so a key is released once. A key the
    mappings do not address (its buffer left them at an earlier roll), and one
    whose array is the solve-lifetime template leaf, need no physical action and
    are dropped.

    With `release_enabled` false nothing is freed and the mappings come back
    unchanged: an eager dispatch is an ordinary Python call whose outputs may be
    any object its inputs contained, so no buffer it touched is known to be one
    the engine produced. The ledger is unaffected either way — it counts
    consumers, and releasing is a separate decision — but its eligibility is
    never consulted on that path, so an `ExecutionPlanningError` from a release
    of a still-read key cannot fire under `enable_jit=False`.
    """
    if not release_enabled:
        candidates.clear()
        return (
            inputs.next_regime_to_V_arr,
            inputs.next_regime_to_continuation,
            inputs.next_edge_to_V_arr,
        )
    located: dict[Hashable, jax.Array] = {}
    for artifact in tuple(candidates):
        array = locate_artifact(inputs=inputs, artifact=artifact)
        if array is None:
            del candidates[artifact]
            continue
        if array is locate_artifact(inputs=templates, artifact=artifact):
            # The mappings hold the solve-lifetime template here: either the
            # last period read it, or a donated key closed at a later commit
            # and its leaf has already been substituted. Neither is a buffer
            # this solve may free.
            del candidates[artifact]
            continue
        located[artifact] = array
    by_dispatch: dict[_InputDispatch, list[ValueArtifactAddress]] = {}
    for artifact, dispatch in candidates.items():
        by_dispatch.setdefault(dispatch, []).append(artifact)
    for dispatch, artifacts in by_dispatch.items():
        for record in release_closed_artifacts(
            ledger=ledger,
            registry=registry,
            artifacts=artifacts,
            arrays_by_artifact=MappingProxyType(located),
            pending_outputs=pending_outputs,
            closing_dispatch=dispatch,
            logger=logger,
            before_delete=before_delete,
        ):
            released_artifact = cast("ValueArtifactAddress", record.artifact)
            if locate_artifact(inputs=inputs, artifact=released_artifact) is not None:
                inputs = substitute_artifact(
                    inputs=inputs, templates=templates, artifact=released_artifact
                )
    candidates.clear()
    return (
        inputs.next_regime_to_V_arr,
        inputs.next_regime_to_continuation,
        inputs.next_edge_to_V_arr,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DonatedInput:
    """One array a dispatch donates, with the buffer identity it had beforehand.

    The identity is taken before the dispatch because a donated array cannot
    report it afterwards; the registry is told to forget the buffer by it.
    """

    artifact: ValueArtifactAddress
    """The artifact key the donated array carries."""

    array: jax.Array
    """The array handed to the executable."""

    identity: BufferIdentity
    """The buffer identity of `array` before the dispatch."""


def _donated_input_arrays(
    *,
    donations: Mapping[_CoreTriple, tuple[ResolvedDonation, ...]],
    unit: DispatchUnit,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    registry: BufferRegistry,
) -> tuple[_DonatedInput, ...]:
    """Locate every array this unit's executables donate and check its aliases.

    The executable was lowered to donate the argument, so a buffer that another
    key still names, or one no dispatch produced, would be freed under its
    owner's feet; the plan is refused here rather than executed. The template
    leaf is never a donated input, and no two programs of one unit may donate
    one buffer: the second would receive what the first handed away.
    """
    located: list[_DonatedInput] = []
    seen: set[ValueArtifactAddress] = set()
    for program in unit.programs:
        for donation in donations.get((unit.regime, unit.period, program), ()):
            if not donation.donated:
                continue
            for artifact in donation.artifacts:
                if artifact in seen:
                    msg = (
                        f"Dispatch {(unit.period, unit.regime)!r} lowered two "
                        f"programs to donate {artifact!r}; one buffer is handed "
                        "over once."
                    )
                    raise ExecutionPlanningError(msg)
                seen.add(artifact)
                located.append(
                    _locate_donated_input(
                        artifact=artifact,
                        argument=donation.argument,
                        unit=unit,
                        inputs=inputs,
                        templates=templates,
                        registry=registry,
                    )
                )
    return tuple(located)


def _locate_donated_input(
    *,
    artifact: ValueArtifactAddress,
    argument: str,
    unit: DispatchUnit,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    registry: BufferRegistry,
) -> _DonatedInput:
    """Find one donated array and refuse a buffer the solve does not own."""
    array = locate_artifact(inputs=inputs, artifact=artifact)
    if array is None or array is locate_artifact(inputs=templates, artifact=artifact):
        msg = (
            f"Dispatch {(unit.period, unit.regime)!r} was lowered to donate "
            f"{argument!r}, but {artifact!r} is not a solve input of its own."
        )
        raise ExecutionPlanningError(msg)
    if registry.is_not_produced(array=array):
        msg = (
            f"Dispatch {(unit.period, unit.regime)!r} would donate {artifact!r}, "
            "whose buffer no compiled executable produced."
        )
        raise ExecutionPlanningError(msg)
    partners = registry.artifacts_sharing(array=array) - {artifact}
    if partners:
        msg = (
            f"Dispatch {(unit.period, unit.regime)!r} would donate {artifact!r}, "
            f"whose buffer {sorted(partners, key=repr)!r} still name."
        )
        raise ExecutionPlanningError(msg)
    return _DonatedInput(
        artifact=artifact, array=array, identity=buffer_identity(array=array)
    )


def _retire_donated_inputs(
    *,
    donated_inputs: tuple[_DonatedInput, ...],
    dispatch: _InputDispatch,
    inputs: SolveInputMappings,
    templates: SolveInputMappings,
    pending_outputs: Sequence[object],
    registry: BufferRegistry,
    logger: logging.Logger,
    before_delete: BeforeArrayDelete | None = None,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
    MappingProxyType[_EdgeKey, FloatND],
]:
    """Forget every donated buffer and put its template leaf in the mappings.

    Nothing is decided before one `block_until_ready` over the outputs of every
    dispatch of the period so far, this one included: a donated buffer may
    reappear as an output shard, and a backend that accepted the annotation
    without reusing the buffer leaves an array a computation still in flight
    reads. Such an array is deleted here, since the ledger has no consumer left
    for it, and the fallback is logged. The registry forgets the buffer by the
    identity recorded before the dispatch, which is the only form a deleted
    array still admits.
    """
    if not donated_inputs:
        return (
            inputs.next_regime_to_V_arr,
            inputs.next_regime_to_continuation,
            inputs.next_edge_to_V_arr,
        )
    jax.block_until_ready(tuple(pending_outputs))
    for donated in donated_inputs:
        registry.forget_identity(identity=donated.identity)
        if not donated.array.is_deleted():
            if before_delete is not None:
                before_delete(arrays=(donated.array,))
            donated.array.delete()
            logger.debug(
                "donation of %r by dispatch %r fell back to release",
                donated.artifact,
                dispatch,
                extra={
                    "artifact_key": donated.artifact,
                    "donating_dispatch": dispatch,
                },
            )
        else:
            logger.debug(
                "donated %r of dispatch %r",
                donated.artifact,
                dispatch,
                extra={
                    "artifact_key": donated.artifact,
                    "donating_dispatch": dispatch,
                },
            )
        inputs = substitute_artifact(
            inputs=inputs, templates=templates, artifact=donated.artifact
        )
    return (
        inputs.next_regime_to_V_arr,
        inputs.next_regime_to_continuation,
        inputs.next_edge_to_V_arr,
    )


def _classify_dispatch_value_artifacts(
    *,
    programs: Mapping[str, _ProgramExecutionMetadata],
) -> tuple[tuple[ValueArtifactAddress, ...], bool]:
    """Collect one dispatch's declared reads and whether any program declares none.

    A planned program's resolved transfer plan must name exactly its declared
    targets; a dense or host-driven program has no plan and contributes its
    declarations directly. A program that declares nothing may still read any
    reachable value through its builder, which the caller pins conservatively.
    """
    declared: list[ValueArtifactAddress] = []
    declares_no_reads = False

    for core_name, metadata in programs.items():
        declared_targets = tuple(
            read.target for read in metadata.requirements.value_reads
        )
        if not declared_targets:
            declares_no_reads = True
            continue
        if metadata.disposition is CoreExecutionDisposition.PLANNED:
            planned_targets = tuple(
                transfer.target for transfer in metadata.input_transfer_plan
            )
            if planned_targets != declared_targets:
                msg = (
                    "A resolved input plan disagrees with its CoreProgram "
                    f"declaration for core {core_name!r}: planned="
                    f"{planned_targets!r}, declared={declared_targets!r}."
                )
                raise RuntimeError(msg)
        declared.extend(declared_targets)

    return _unique_value_artifacts(declared), declares_no_reads


def _rolled_aliases(
    *,
    regimes: Mapping[RegimeName, Regime],
    artifacts: Iterable[ValueArtifactAddress],
) -> MappingProxyType[ValueArtifactAddress, ValueArtifactAddress]:
    """Map each rolled key to the key of the same buffer one period later.

    A regime value or continuation leaf of a regime inactive at its period, and
    a gated continuation with no fold dispatch at its period, are entries the
    period roll carries forward unchanged; each is the buffer the same address
    names one period later.
    """
    n_periods = _model_n_periods(regimes=regimes)
    fold_dispatches = frozenset(_gated_edge_fold_dispatches(regimes=regimes))
    aliases: dict[ValueArtifactAddress, ValueArtifactAddress] = {}
    pending = list(artifacts)
    while pending:
        artifact = pending.pop()
        if artifact in aliases or artifact.period >= n_periods - 1:
            continue
        if artifact.kind is ValueArtifactKind.GATED_CONTINUATION:
            produced = (
                artifact.period,
                artifact.regime,
                artifact.target_regime,
            ) in fold_dispatches
        else:
            produced = artifact.period in regimes[artifact.regime].active_periods
        if produced:
            continue
        later = dataclasses.replace(artifact, period=artifact.period + 1)
        aliases[artifact] = later
        pending.append(later)
    return MappingProxyType(aliases)


def _unique_value_artifacts(
    artifacts: Iterable[ValueArtifactAddress],
) -> tuple[ValueArtifactAddress, ...]:
    """Deduplicate one dispatch in declaration order."""
    return tuple(dict.fromkeys(artifacts))


def _retained_solution_artifacts(
    *,
    regimes: Mapping[RegimeName, Regime],
    retain_all_artifacts: bool,
    persistable_artifact_refs: frozenset[ArtifactRef],
) -> tuple[ValueArtifactAddress, ...]:
    """Pin every artifact the selected retention keeps in the public result.

    Every regime value is kept under every retention. A continuation payload is
    kept only where persistence-oriented retention selected its exact address,
    and then every leaf of it is retained: the result hands the whole payload
    back, so freeing one leaf leaves an unreadable artifact behind.

    The ledger speaks only in releasable input addresses, and the continuation
    payload is the one retained channel that has them — a replay or auxiliary
    key names a payload that appears in no rolling input mapping, so there is no
    address for the ledger to retain on its behalf. That those payloads can
    still *share a buffer* with a releasable leaf is a physical question, and
    `BufferRegistry.declare_passed_through` answers it at each dispatch.
    """
    values = tuple(
        ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE,
            period=period,
            regime=regime_name,
        )
        for regime_name, regime in regimes.items()
        for period in regime.active_periods
    )
    if not retain_all_artifacts:
        return values
    return values + _retained_continuation_leaves(
        regimes=regimes, persistable_artifact_refs=persistable_artifact_refs
    )


def _retained_continuation_leaves(
    *,
    regimes: Mapping[RegimeName, Regime],
    persistable_artifact_refs: frozenset[ArtifactRef],
) -> tuple[ValueArtifactAddress, ...]:
    """Address every leaf of every continuation payload persistence keeps."""
    continuation_specs = MappingProxyType(
        {
            regime_name: regime.solution.continuation_spec
            for regime_name, regime in regimes.items()
            if regime.solution.continuation_spec is not None
        }
    )
    leaves: list[ValueArtifactAddress] = []
    for ref in sorted(persistable_artifact_refs):
        spec = continuation_specs.get(ref.regime)
        template = published_continuation_template(
            continuation_specs=continuation_specs, target=ref.regime
        )
        if spec is None or template is None or spec.artifact_key != ref.key:
            continue
        leaves.extend(
            ValueArtifactAddress(
                kind=ValueArtifactKind.CONTINUATION_LEAF,
                period=ref.period,
                regime=ref.regime,
                artifact_key=ref.key,
                leaf_path=leaf_path,
            )
            for leaf_path in template.leaves()
        )
    return tuple(dict.fromkeys(leaves))


def gated_edge_fold_value_reads(
    *, regimes: Mapping[RegimeName, Regime], period: int
) -> tuple[ValueRead, ...]:
    """Declare every same-period value the gated-edge folds of one period read.

    The fold of an edge evaluates the target's value and each reference regime's
    value on the target's own grid nodes at the folded period, so each is one
    counted consumer of that period's value.
    """
    reads: list[ValueRead] = []
    for source_name, source in regimes.items():
        for target_name, edge in source.gated_edges.items():
            readers = (edge.target, *edge.reference_regimes)
            if not all(period in regimes[name].active_periods for name in readers):
                continue
            reads.extend(
                ValueRead(
                    target=ValueArtifactAddress(
                        kind=ValueArtifactKind.REGIME_VALUE,
                        period=period,
                        regime=name,
                    ),
                    source=ValueConsumerAddress(
                        source_period=period,
                        source_regime=source_name,
                        core_key=_fold_core_key(target_name=target_name),
                        channel=ValueInputChannel.SAME_PERIOD_VALUE,
                        path=(name,),
                    ),
                )
                for name in readers
            )
    return tuple(reads)


def _fold_core_key(*, target_name: RegimeName) -> str:
    """Name the engine-owned fold of the edge into one target regime."""
    return f"gated_edge_fold:{target_name}"


def _folded_edge_keys_at_period(
    *,
    regimes: Mapping[RegimeName, Regime],
    period: int,
    solved_regimes: Container[RegimeName],
) -> tuple[_EdgeKey, ...]:
    """Name every gated edge whose `Wbar` is folded on one period's values.

    The single enumeration the liveness declaration, the fold itself, and the
    fold's commit all read, so a declared fold dispatch and a committed one name
    the same edges. `solved_regimes` is the set of regimes holding a value at
    `period`: the regimes active there while the ledger is declared, the
    period's published values while the solve loop runs.
    """
    return tuple(
        (source_name, target_name)
        for source_name, source in regimes.items()
        for target_name, edge in source.gated_edges.items()
        if edge_may_fold_at_period(
            edge=edge,
            source_name=source_name,
            fold_period=period,
            solved_regimes=solved_regimes,
            source_reads_wbar=source_reads_folded_wbar(
                source_active_periods=source.active_periods,
                fold_period=period,
            ),
        )
    )


def _gated_edge_fold_dispatches(
    *,
    regimes: Mapping[RegimeName, Regime],
) -> dict[_InputDispatch, tuple[ValueArtifactAddress, ...]]:
    """Plan one dispatch per gated-edge fold the induction will evaluate.

    A fold's dispatch id is `(period, source, target)`, so it is identified
    apart from the `(period, regime)` core dispatches of the same period and
    from every other edge folded there.
    """
    dispatches: dict[_InputDispatch, tuple[ValueArtifactAddress, ...]] = {}
    for period in range(_model_n_periods(regimes=regimes)):
        reads_by_consumer: dict[tuple[RegimeName, str], list[ValueArtifactAddress]] = {}
        for read in gated_edge_fold_value_reads(regimes=regimes, period=period):
            reads_by_consumer.setdefault(
                (read.source.source_regime, read.source.core_key), []
            ).append(read.target)
        for source_name, target_name in _folded_edge_keys_at_period(
            regimes=regimes,
            period=period,
            solved_regimes=_regimes_active_at_period(regimes=regimes, period=period),
        ):
            reads = reads_by_consumer.get(
                (source_name, _fold_core_key(target_name=target_name)), []
            )
            if reads:
                dispatches[(period, source_name, target_name)] = (
                    _unique_value_artifacts(reads)
                )
    return dispatches


def _regimes_active_at_period(
    *,
    regimes: Mapping[RegimeName, Regime],
    period: int,
) -> frozenset[RegimeName]:
    """Name the regimes holding a solved value at one period."""
    return frozenset(
        regime_name
        for regime_name, regime in regimes.items()
        if period in regime.active_periods
    )


def _model_n_periods(*, regimes: Mapping[RegimeName, Regime]) -> int:
    """Return the number of periods the model spans, as its regimes record it."""
    return max(
        (regime.solution.reachability.n_periods for regime in regimes.values()),
        default=0,
    )


def _regime_retains_replay(*, regime: Regime, retain_replay: bool) -> bool:
    """Whether one regime's normal retention dispatches every replay program.

    A simulation policy is consumed only through the regime's declared replay
    route. A regime without one (a standalone case-piece NB-EGM regime, whose
    simulation reads the grid argmax) dispatches its values-only programs under
    every retention, so a replay output is never assembled only to be discarded.
    Persistence-oriented retention is resolved separately through exact artifact
    identities rather than widening this boolean.
    """
    return retain_replay and (
        regime.simulation.egm_policy_read is not None
        or regime.simulation.external_replay_route is not None
    )


def _resident_bytes_by_triple(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    ledger: PlannedInputLiveness,
    templates: SolveInputMappings,
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    device_ids: tuple[int, ...],
) -> MappingProxyType[_CoreTriple, int]:
    """Return the declared-read lower bound before candidates are compiled."""
    inventory = _resident_inventory_by_triple(
        regimes=regimes,
        ledger=ledger,
        templates=templates,
        program_metadata=program_metadata,
        device_ids=device_ids,
    )
    return MappingProxyType(
        {triple: position.resident_bytes() for triple, position in inventory.items()}
    )


def _resident_inventory_by_triple(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    ledger: PlannedInputLiveness,
    templates: SolveInputMappings,
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    device_ids: tuple[int, ...],
    fixed_bytes: Mapping[int, int] = MappingProxyType({}),
) -> MappingProxyType[_CoreTriple, ResidentInventory]:
    """Predict, per core triple, the live inventory at its scheduled position.

    The schedule walked is the one the loop will run: `plan_period_waves` over
    each period's active regimes at their placement's device sets, periods
    descending, a period's gated-edge folds after its last wave. Every core of
    a regime-period cell shares the cell's number, because a kernel dispatches
    its cores together as one unit.

    The snapshot retains every live alias group. Only a candidate's own aligned
    reads surviving compiler pruning may later be excluded from its resident
    bytes. Stored sources of copies remain charged. Concrete fixed owners,
    whole-period shared-copy reservations and the declared transfer-operator
    scratch add conservative per-device burdens; they may overlap compiler
    storage and do not predict exact allocator peaks.

    Every route charges that scratch, on every endpoint device of every planned
    transfer operator, so a cell whose reads cross a mesh boundary is admitted on
    the complete footprint of the copy and not only on what it delivers.

    Sizes come from the solve-lifetime templates, which are period-invariant,
    so an artifact of any period finds the template of what it names.
    """
    program_keys_by_cell = _program_keys_by_cell(triples=program_metadata)
    footprints = _artifact_footprints(ledger=ledger, templates=templates)
    device_ids_by_regime = MappingProxyType(
        {
            regime_name: _regime_device_ids(
                regime=regime, visible_device_ids=device_ids
            )
            for regime_name, regime in regimes.items()
        }
    )
    resident = plan_resident_inventory(
        waves_by_period=MappingProxyType(
            {
                period: _scheduled_waves(
                    regimes=regimes,
                    period=period,
                    program_keys_by_cell=program_keys_by_cell,
                    program_metadata=program_metadata,
                    footprints=footprints,
                    device_ids_by_regime=device_ids_by_regime,
                )
                for period in range(_model_n_periods(regimes=regimes))
            }
        ),
        fold_dispatches=_fold_output_artifacts(regimes=regimes),
        ledger=ledger,
        footprints=MappingProxyType(
            cast("dict[Hashable, ArtifactFootprint]", footprints)
        ),
    )
    copies_by_period = {
        period: _period_copy_reservations(period=period, metadata=program_metadata)
        for period in range(_model_n_periods(regimes=regimes))
    }
    scratch_by_period = {
        period: _period_transfer_scratch_reservations(
            period=period, metadata=program_metadata, device_ids=device_ids
        )
        for period in range(_model_n_periods(regimes=regimes))
    }
    return MappingProxyType(
        {
            (regime_name, period, core_key): dataclasses.replace(
                resident[(period, regime_name)],
                fixed_bytes=fixed_bytes,
                shared_copies=copies_by_period[period],
                transfer_scratch_bytes=scratch_by_period[period],
            )
            for (regime_name, period), core_keys in program_keys_by_cell.items()
            for core_key in core_keys
        }
    )


def _period_copy_reservations(
    *,
    period: int,
    metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
) -> Mapping[Hashable, ArtifactFootprint]:
    """Reserve each shared destination throughout its period, including aliases.

    Runtime caches a destination by artifact and required layout. Its release can
    be delayed when a copy shares a source shard, so whole-period retention is a
    conservative bound instead of a prediction of the allocator's release instant.
    """
    return MappingProxyType(
        {
            (transfer.target, transfer.source_sharding): ArtifactFootprint(
                bytes_per_device=transfer.cost.per_device_bytes,
                device_ids=tuple(
                    sorted(device.id for device in transfer.source_sharding.device_set)
                ),
            )
            for triple, program in metadata.items()
            if triple[1] == period
            for transfer in program.input_transfer_plan
            if transfer.reused_by_several_consumers
            and transfer.kind is not ValueTransferKind.ALIGNED_LOCAL
        }
    )


def _period_transfer_scratch_reservations(
    *,
    period: int,
    metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    device_ids: tuple[int, ...],
) -> Mapping[int, int]:
    """Bound all pending copy scratch on every endpoint device of one period.

    An operator's declared temporary bytes are what it holds beyond its result,
    on each device it touches — for a copy onto another mesh, a second whole
    value, charged on the stored value's devices as well as the reader's. The
    charge lands on every endpoint, so a device that only sources a copy is
    admitted on the stored shards plus that scratch rather than escaping the
    comparison because no kernel of the cell runs there.

    This deliberately overlaps every copy miss with every core's compiler peak;
    it is a declared conservative envelope, not measured allocator scratch.
    Shared artifact/layout copies count once, while unshared occurrences can all
    be pending simultaneously. An endpoint outside the admission devices has no
    planned ceiling to be compared against, so it is refused rather than charged.
    """
    scratch: dict[int, int] = {}
    shared: set[Hashable] = set()
    for triple, program in metadata.items():
        if triple[1] != period:
            continue
        for transfer in program.input_transfer_plan:
            if transfer.kind is ValueTransferKind.ALIGNED_LOCAL:
                continue
            cost = transfer.cost
            unplanned = sorted(frozenset(cost.devices) - frozenset(device_ids))
            if unplanned:
                msg = (
                    f"A planned transfer of {transfer.target} has endpoints "
                    f"{unplanned} outside the admission devices."
                )
                raise ExecutionPlanningError(msg)
            if transfer.reused_by_several_consumers:
                key = (transfer.target, transfer.source_sharding)
                if key in shared:
                    continue
                shared.add(key)
            for device in cost.devices:
                scratch[device] = scratch.get(device, 0) + cost.temporary_bytes
    return MappingProxyType(scratch)


def _internal_reservations_by_cell(
    *,
    programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    templates: Mapping[_CoreCandidate, Mapping[str, object]],
) -> Mapping[tuple[RegimeName, int], int]:
    """Reserve future producer subtrees across all cores of their runtime cell.

    These are not allocations owned by abstract templates. The existing producer
    validator establishes width-invariant subtrees; taking the maximum across
    candidate templates also keeps this reservation conservative if that contract
    is extended. A known abstract output sharding supplies its per-device payload;
    otherwise the full logical payload is reserved on every cell device. Neither
    fallback placement nor alias overlap is claimed to be exact allocator storage.
    """
    by_cell: dict[tuple[RegimeName, int], dict[tuple[str, str], int]] = {}
    for candidate, program in programs.items():
        cell = candidate[0][:2]
        sizes = by_cell.setdefault(cell, {})
        for name, reference in program.requirements.internal_inputs.items():
            size = sum(
                _internal_leaf_bytes(leaf=leaf)
                for leaf in jax.tree.leaves(templates[candidate][name])
                if isinstance(leaf, jax.ShapeDtypeStruct)
            )
            key = (reference.producer, reference.label)
            sizes[key] = max(sizes.get(key, 0), size)
    return MappingProxyType(
        {cell: sum(sizes.values()) for cell, sizes in by_cell.items()}
    )


def _internal_leaf_bytes(*, leaf: jax.ShapeDtypeStruct) -> int:
    """Use declared abstract output placement, or a full-device upper bound."""
    if isinstance(leaf.sharding, jax.sharding.Sharding):
        return layout_footprint(
            sharding=leaf.sharding,
            shape=leaf.shape,
            item_bytes=leaf.dtype.itemsize,
        ).bytes_per_device
    return leaf.size * leaf.dtype.itemsize


def _program_keys_by_cell(
    *, triples: Iterable[_CoreTriple]
) -> MappingProxyType[tuple[RegimeName, int], tuple[str, ...]]:
    """Group core keys by regime-period cell, in producer order."""
    cells: dict[tuple[RegimeName, int], list[str]] = {}
    for regime_name, period, core_key in triples:
        cells.setdefault((regime_name, period), []).append(core_key)
    return MappingProxyType({cell: tuple(keys) for cell, keys in cells.items()})


def _fold_output_artifacts(
    *, regimes: MappingProxyType[RegimeName, Regime]
) -> MappingProxyType[tuple[int, RegimeName, RegimeName], ValueArtifactAddress]:
    """Address the gated continuation every edge fold of the solve produces."""
    folds: dict[tuple[int, RegimeName, RegimeName], ValueArtifactAddress] = {}
    for dispatch in _gated_edge_fold_dispatches(regimes=regimes):
        period, source_name, target_name = cast(
            "tuple[int, RegimeName, RegimeName]", dispatch
        )
        folds[(period, source_name, target_name)] = ValueArtifactAddress(
            kind=ValueArtifactKind.GATED_CONTINUATION,
            period=period,
            regime=source_name,
            target_regime=target_name,
        )
    return MappingProxyType(folds)


def _artifact_footprints(
    *, ledger: PlannedInputLiveness, templates: SolveInputMappings
) -> dict[ValueArtifactAddress, ArtifactFootprint]:
    """Size every planned artifact the solve-lifetime templates address."""
    footprints: dict[ValueArtifactAddress, ArtifactFootprint] = {}
    for artifact in ledger.remaining_counts:
        template = locate_artifact(inputs=templates, artifact=artifact)
        if template is not None:
            footprints[artifact] = per_device_footprint(array=template)
    return footprints


def _scheduled_waves(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    period: int,
    program_keys_by_cell: Mapping[tuple[RegimeName, int], tuple[str, ...]],
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    footprints: Mapping[ValueArtifactAddress, ArtifactFootprint],
    device_ids_by_regime: Mapping[RegimeName, frozenset[int]],
) -> tuple[tuple[ScheduledUnit, ...], ...]:
    """Describe one period's dispatch waves the way the loop plans them."""
    active = {
        regime_name: regime
        for regime_name, regime in regimes.items()
        if period in regime.active_periods
        and program_keys_by_cell.get((regime_name, period))
    }
    waves = plan_period_waves(
        nodes=tuple(
            ScheduledNode(period=period, regime=regime_name, program=core_key)
            for regime_name in active
            for core_key in program_keys_by_cell[(regime_name, period)]
        ),
        same_period_dependencies=MappingProxyType(
            {
                regime_name: regime.same_period_ref_regimes
                for regime_name, regime in active.items()
            }
        ),
        device_sets=MappingProxyType(
            {regime_name: device_ids_by_regime[regime_name] for regime_name in active}
        ),
    )
    return tuple(
        tuple(
            _scheduled_unit(
                unit=unit,
                program_metadata=program_metadata,
                footprints=footprints,
                device_ids=tuple(sorted(device_ids_by_regime[unit.regime])),
            )
            for unit in wave
        )
        for wave in waves
    )


def _scheduled_unit(
    *,
    unit: DispatchUnit,
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
    footprints: Mapping[ValueArtifactAddress, ArtifactFootprint],
    device_ids: tuple[int, ...],
) -> ScheduledUnit:
    """Describe one dispatch unit's outputs and its pass-through inputs.

    The unit produces the sized artifacts addressed at its own cell; a gated
    continuation is produced by an edge fold instead, which the walk registers
    at the end of the period.
    """
    produced = tuple(
        artifact
        for artifact in footprints
        if artifact.kind is not ValueArtifactKind.GATED_CONTINUATION
        and artifact.period == unit.period
        and artifact.regime == unit.regime
    )
    return ScheduledUnit(
        period=unit.period,
        regime=unit.regime,
        device_ids=device_ids,
        produces=produced,
        consumes=_unique_value_artifacts(
            artifact
            for core_key in unit.programs
            for artifact in _aligned_input_artifacts(
                metadata=program_metadata[(unit.regime, unit.period, core_key)]
            )
        ),
        output_bytes_per_device=sum(
            footprints[artifact].bytes_per_device for artifact in produced
        ),
    )


def _aligned_input_artifacts(
    *, metadata: _ProgramExecutionMetadata
) -> tuple[ValueArtifactAddress, ...]:
    """Name the values one program's executable is handed without a copy.

    A planned program's resolved transfer plan says which reads reach the
    executable in their stored layout; every other transfer kind allocates a
    copy that the stored buffer outlives, so both are live. A program with no
    plan reads its declared values directly.
    """
    if metadata.input_transfer_plan:
        return tuple(
            transfer.target
            for transfer in metadata.input_transfer_plan
            if transfer.kind is ValueTransferKind.ALIGNED_LOCAL
        )
    return tuple(read.target for read in metadata.requirements.value_reads)


def _candidate_resident_bytes(
    *,
    compiled: jax.stages.Compiled,
    program: ResolvedCoreProgram,
    internal_arguments: Mapping[str, object],
    inventory: ResidentInventory,
) -> int:
    """Exclude only this specialization's aligned, compiler-live read buffers.

    Logical read occurrences identify scheduled storage. Template pointer aliases
    are deliberately irrelevant: representatives need not share the identities of
    the future runtime artifacts they size. Pruning changes neither declared reads
    nor their release times; it changes which inputs compiler accounting covers.
    """
    arguments = {**program.arguments, **internal_arguments}
    compiler_input_paths(compiled=compiled, arguments=arguments)
    _, input_shardings = compiled.input_shardings
    aligned_sources = (
        frozenset(
            transfer.source
            for transfer in program.input_transfer_plan
            if transfer.kind is ValueTransferKind.ALIGNED_LOCAL
        )
        if program.input_transfer_plan
        else frozenset(read.source for read in program.requirements.value_reads)
    )
    consumes = _unique_value_artifacts(
        read.target
        for read in program.requirements.value_reads
        if read.source in aligned_sources
        and _compiler_reads_source(shardings=input_shardings, source=read.source)
    )
    copied_inputs: set[Hashable] = set()
    temporary_bytes: dict[int, int] = {}
    for transfer in program.input_transfer_plan:
        if transfer.kind is ValueTransferKind.ALIGNED_LOCAL:
            continue
        kept = _compiler_reads_source(shardings=input_shardings, source=transfer.source)
        if transfer.reused_by_several_consumers:
            if kept:
                copied_inputs.add((transfer.target, transfer.source_sharding))
        elif not kept:
            # Unshared occurrences are allocated separately, even if they name
            # the same artifact. A pruned occurrence still reaches device_put.
            size = transfer.cost.per_device_bytes
            for device in transfer.source_sharding.device_set:
                temporary_bytes[device.id] = temporary_bytes.get(device.id, 0) + size
    return inventory.resident_bytes(
        consumes=consumes,
        consumed_copies=frozenset(copied_inputs),
        temporary_bytes=temporary_bytes,
    )


def _compiler_reads_source(
    *,
    shardings: Mapping[str, object],
    source: ValueConsumerAddress,
) -> bool:
    """Read one exact declared locator from the validated public input tree.

    JAX reconstructs original mapping keys and container structure even when a
    custom registration's flattened keys are positional. No pointer matching or
    assumptions about registration order enter the logical read occurrence.
    """
    node = shardings[source.argument or source.channel.value]
    for segment in source.path:
        if isinstance(node, Mapping):
            node = node[segment]
        elif isinstance(node, tuple):
            node = node[cast("int", segment)]
        else:
            node = getattr(node, str(segment))
    return isinstance(node, jax.sharding.Sharding)


def _triples_within_budget(
    *,
    candidates_by_triple: Mapping[_CoreTriple, Sequence[_CoreCandidate]],
    resident_bytes_by_triple: Mapping[_CoreTriple, int],
    budget_bytes: int | None,
) -> tuple[_CoreTriple, ...]:
    """Name the cores whose position still leaves room for a workspace.

    A core whose resident bytes already reach the budget is served by no width,
    so lowering its frontier would compile, on the very device that cannot host
    them, candidates that are refused either way. It is left out of the
    compilation waves entirely and refused by name when selection reaches it,
    with no compiled candidate of its own.
    """
    if budget_bytes is None:
        return tuple(candidates_by_triple)
    return tuple(
        triple
        for triple in candidates_by_triple
        if resident_bytes_by_triple[triple] < budget_bytes
    )


def _width_selection_failure(
    *,
    triple: _CoreTriple,
    resident_bytes: int,
    transfer_scratch_bytes: Mapping[int, int],
    budget_bytes: int,
    execution: ResolvedExecution,
    error: ExecutionPlanningError,
    skipped_fallback_evaluations: int = 0,
) -> ExecutionPlanningError:
    """Name the cell a workspace refusal belongs to and what it competed with.

    The budget named here is the effective one, so the message also states the
    request it came from whenever the devices capped it. The resident bytes
    already contain the transfer charge; it is named separately because a cell
    refused over a copy it only reads is otherwise indistinguishable from one
    refused over its own stored values.
    """
    regime_name, period, core_key = triple
    scratch = max(transfer_scratch_bytes.values(), default=0)
    msg = (
        f"Regime {regime_name!r} at period {period} cannot select a workspace "
        f"width for core {core_key!r}: the plan keeps {resident_bytes} bytes "
        f"resident on its busiest device against a {budget_bytes}-byte budget, "
        f"of which the period's transfer operators reserve up to {scratch} bytes "
        f"on a single device."
        f"{execution.device_memory_cap_note()} {error}"
    )
    if skipped_fallback_evaluations:
        msg += (
            f" Fallback evaluation skipped for {skipped_fallback_evaluations} "
            "rejected candidate(s) whose donating variant already exceeded the "
            "budget."
        )
    return ExecutionPlanningError(msg)


def _select_period_programs(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    retain_replay: bool,
    persistable_artifact_refs: frozenset[ArtifactRef],
) -> MappingProxyType[str, CoreProgram]:
    """Select one cell's programs from exact model-authoritative artifact keys."""
    selected_artifact_keys = _selected_artifact_keys_for_cell(
        persistable_artifact_refs=persistable_artifact_refs,
        regime_name=regime_name,
        period=period,
    )
    native_graph = core_program_graph(kernel=regime.solution.period_kernels[period])
    return select_programs(
        graph=native_graph,
        retain_replay=_regime_retains_replay(
            regime=regime, retain_replay=retain_replay
        ),
        selected_artifact_keys=selected_artifact_keys,
    )


def _selected_artifact_keys_for_cell(
    *,
    persistable_artifact_refs: frozenset[ArtifactRef],
    regime_name: RegimeName,
    period: int,
) -> frozenset[ArtifactKey]:
    """Project exact persistence retention onto one period/regime cell."""
    return frozenset(
        ref.key
        for ref in persistable_artifact_refs
        if ref.period == period and ref.regime == regime_name
    )


def _retained_base_space_arrays(*, regime: Regime) -> object:
    """Read owned base arrays without constructing another completed state space."""
    # The canonical phase retains this original even when runtime params replace
    # its placeholders. Accounting intentionally observes that owning field.
    space = regime.solution._base_state_action_space  # noqa: SLF001
    return space.states, space.discrete_actions, space.continuous_actions


def _compile_all_functions(  # noqa: C901, PLR0912, PLR0915
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    program_fingerprint: str,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    enable_jit: bool,
    execution: ResolvedExecution,
    retain_replay: bool,
    retain_all_artifacts: bool,
    persistable_artifact_refs: frozenset[ArtifactRef],
    max_compilation_workers: int | None,
    logger: logging.Logger,
    call_id: CallId | None = None,
    fixed_input_arrays: object = (),
    process_grid_resolver: ProcessGridResolver | None = None,
) -> _CompiledPrograms:
    """Resolve every solve program and optionally compile unique lowerings.

    Each regime exposes named cores through its period adapter. For every core, the
    engine first materializes the adapter's exact `CoreProgram`. The program supplies
    the planner-resolved callable, static choices, and output roles. The program's
    durable identity, abstract arguments, specialization, and
    output layout form the lowering key. Each unique program is lowered once
    (sequentially, because tracing is single-threaded), then the XLA programs compile
    in parallel via a thread pool. The loop stays free of solver-type forks.

    When JIT is disabled (`enable_jit=False`), executes the same resolved programs
    without the lowering and compilation steps.

    Args:
        regimes: The internal regimes containing the period adapters.
        program_fingerprint: Digest of the model facts every lowered program
            depends on; opens every program's identity, so two models never
            share an executable.
        flat_params: Regime parameters for constructing lowering args.
        ages: Age grid for the model.
        next_regime_to_V_arr: Template with consistent keys and V array shapes
            for constructing lowering arguments.
        next_regime_to_continuation: Template with consistent keys and carry
            shapes for constructing lowering arguments.
        next_edge_to_V_arr: Template with consistent keys and `Wbar` shapes
            for constructing a source kernel's gated-edge lowering arguments;
            empty for models without gated edges.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        execution: The hardware-local facts the model resolved — its devices,
            the optional per-device workspace budget, and fixed planner axis
            widths.
        retain_replay: Whether the solve retains replay artifacts; with the
            regime's declared replay route it selects which scoped programs of
            each kernel's graph are dispatched.
        retain_all_artifacts: Whether the result keeps every persistable
            continuation payload, which the ledger retains and never donates.
        persistable_artifact_refs: Exact model-authoritative addresses whose
            replay programs are selected for persistence-oriented retention.
        max_compilation_workers: Maximum threads for parallel compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.
        fixed_input_arrays: Already-built runtime space arrays retained by solve.

    Returns:
        Executable mappings by regime-period, the resolved metadata used by
        input liveness, the ledger the executables were lowered against, and the
        donation decisions each selected executable carries. Eager entries call
        the resolved functions directly; AOT entries call compiled executables
        carrying the same plans.

    """
    # Collect every kernel's native graph, narrowed to the retention's scope.
    with solve_phase(name="program_graphs", logger=logger, call_id=call_id):
        all_programs: dict[_CoreTriple, CoreProgram] = {}
        for regime_name, regime in regimes.items():
            for period in regime.active_periods:
                graph = _select_period_programs(
                    regime=regime,
                    regime_name=regime_name,
                    period=period,
                    retain_replay=retain_replay,
                    persistable_artifact_refs=persistable_artifact_refs,
                )
                for core_name, program in graph.items():
                    all_programs[(regime_name, period, core_name)] = program

    # Materialize each named core's exact program before representative selection.
    # The resulting function, arguments, roles, specialization, and layout form
    # one lowering source of truth.
    with solve_phase(name="structural_resolution", logger=logger, call_id=call_id):
        (
            all_layouts,
            lowering_keys,
            resolved_programs,
            internal_templates,
            input_liveness,
            donations,
            representative_metadata,
            frontier,
        ) = _resolve_output_layouts_and_lowering_keys(
            all_programs=all_programs,
            regimes=regimes,
            program_fingerprint=program_fingerprint,
            flat_params=flat_params,
            ages=ages,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            next_edge_to_V_arr=next_edge_to_V_arr,
            budget_bytes=execution.device_memory_bytes,
            execution_widths=execution,
            enable_jit=enable_jit,
            continuous_sharded_state=execution.continuous_sharded_state,
            donate_buffers=execution.donate_buffers,
            retain_all_artifacts=retain_all_artifacts,
            persistable_artifact_refs=persistable_artifact_refs,
            process_grid_resolver=process_grid_resolver,
        )

        _fail_if_one_key_covers_two_callables(
            lowering_keys=lowering_keys, resolved_programs=resolved_programs
        )
        fallback_programs = {
            candidate: resolved_programs[candidate]
            for candidate, decisions in donations.items()
            if _donated_arguments(donations=decisions)
        }
        fallback_donations: dict[_CoreCandidate, tuple[ResolvedDonation, ...]] = (
            dict.fromkeys(fallback_programs, ())
        )
        fallback_argument_keys: dict[_CoreTriple, Hashable] = {}
        fallback_keys = _lowering_keys(
            resolved_programs=fallback_programs,
            internal_templates=internal_templates,
            layouts=all_layouts,
            donations=fallback_donations,
            regimes=regimes,
            program_fingerprint=program_fingerprint,
            argument_keys=fallback_argument_keys,
        )
        frontier.bind_fallbacks(
            fallback_keys=fallback_keys,
            fallback_donations=fallback_donations,
            argument_keys=fallback_argument_keys,
        )

    # Bound candidates, in rank order, of each core. The frontier appends to these
    # lists as refusals ask for narrower candidates; `frontier_lengths` is the
    # full ranked frontier a core still has left to offer.
    candidates_by_triple = frontier.candidates_by_triple

    # Eager execution uses the same resolved function, arguments, requirements,
    # roles, static widths, and transfers as AOT. Only the final JAX compilation
    # step is omitted.
    if not enable_jit:
        if execution.device_memory_bytes is not None:
            msg = (
                "ExecutionConfig.device_memory_bytes requires JIT compilation so the "
                "compiler can report peak workspace."
            )
            raise ExecutionPlanningError(msg)
        # The eager route resolves no residency, compiles no wave and plans no
        # workspace; the empty brackets keep one record sequence for every solve.
        for skipped in (
            "residency_inventory",
            "compilation_waves",
            "workspace_selection",
        ):
            with solve_phase(name=skipped, logger=logger, call_id=call_id):
                pass
        selected_candidates = {
            triple: candidates[0] for triple, candidates in candidates_by_triple.items()
        }
        selected_programs = {
            triple: resolved_programs[candidate]
            for triple, candidate in selected_candidates.items()
        }
        eager = {
            triple: _attach_resolved_output_layout(
                compiled=make_eager_core(
                    program=program,
                    execution_sharding=all_layouts[triple].expected_leaves[0].sharding,
                    internal_input_templates=internal_templates[
                        selected_candidates[triple]
                    ],
                ),
                layout=all_layouts[triple],
                tile_widths=program.tile_widths,
                input_transfer_plan=program.input_transfer_plan,
                internal_input_templates=internal_templates[
                    selected_candidates[triple]
                ],
                name=triple[2],
            )
            for triple, program in selected_programs.items()
        }
        return _CompiledPrograms(
            executables=_group_cores_by_regime_period(eager),
            metadata=_execution_metadata(programs=selected_programs),
            input_liveness=input_liveness,
            donations=MappingProxyType(dict.fromkeys(selected_programs, ())),
        )

    # Candidates are ranked widest-first within each triple. Lowering proceeds in
    # waves: wave 0 holds every triple's top-ranked candidate — the bootstrap width
    # without a budget, the full extent under one, or the requested width — so an
    # unbudgeted solve lowers exactly one candidate per core. Under a budget, only
    # the triples whose current candidate exceeds it
    # advance to their next candidate in the following wave, and only then is that
    # candidate bound at all: a core whose widest candidate is admitted resolves
    # one program, not its whole frontier. Each wave deduplicates
    # by lowering key across triples, lowers sequentially (tracing is
    # single-threaded), and compiles in parallel.
    with solve_phase(name="residency_inventory", logger=logger, call_id=call_id):
        budget_bytes = execution.device_memory_bytes
        fixed_bytes = (
            MappingProxyType({})
            if budget_bytes is None
            else concrete_device_bytes(
                tree=(
                    fixed_input_arrays,
                    flat_params,
                    ages.values,
                    next_regime_to_V_arr,
                    next_regime_to_continuation,
                    next_edge_to_V_arr,
                    tuple(
                        (
                            regime.solution.period_state_axes,
                            regime.solution.resolved_fixed_params,
                            _retained_base_space_arrays(regime=regime),
                        )
                        for regime in regimes.values()
                    ),
                ),
            )
        )
        # A candidate competes with what the plan already keeps on its device at the
        # node's scheduled position. Without a budget no peak is consulted, so the
        # position is not walked either.
        resident_inventory = (
            MappingProxyType({})
            if budget_bytes is None
            else _resident_inventory_by_triple(
                regimes=regimes,
                ledger=input_liveness,
                templates=SolveInputMappings(
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    next_edge_to_V_arr=next_edge_to_V_arr,
                ),
                program_metadata=representative_metadata,
                device_ids=execution.device_ids,
                fixed_bytes=fixed_bytes,
            )
        )
        if budget_bytes is not None:
            internal_bytes = _internal_reservations_by_cell(
                programs=resolved_programs,
                templates=internal_templates,
            )
            resident_inventory = MappingProxyType(
                {
                    triple: dataclasses.replace(
                        inventory,
                        internal_bytes=internal_bytes[triple[:2]],
                    )
                    for triple, inventory in resident_inventory.items()
                }
            )
        resident_bytes_by_triple = MappingProxyType(
            {
                triple: 0
                if budget_bytes is None
                else resident_inventory[triple].resident_bytes()
                for triple in candidates_by_triple
            }
        )
    n_workers = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    compiled: dict[Hashable, jax.stages.Compiled] = {}
    labels: dict[Hashable, str] = {}
    memory_by_lowering_key: dict[Hashable, CompilerMemoryReservation] = {}
    resident_bytes_by_candidate: dict[_CoreCandidate, int] = {}
    admission_keys: dict[_CoreCandidate, Hashable] = {}
    eligible = _triples_within_budget(
        candidates_by_triple=candidates_by_triple,
        resident_bytes_by_triple=resident_bytes_by_triple,
        budget_bytes=budget_bytes,
    )
    # The bounded search only has a question to answer under a budget: without
    # one no candidate is ever refused, so the ranked walk lowers the one
    # bootstrap candidate per core either way.
    bounded = (
        execution.width_search.kind is WidthSearch.BOUNDED and budget_bytes is not None
    )
    source: _RankedCandidateSource | _BoundedCandidateSource
    if bounded:
        source = _bounded_candidate_source(
            frontier=frontier,
            triples=eligible,
            candidates_by_triple=candidates_by_triple,
            resolved_programs=resolved_programs,
            execution=execution,
        )
    else:
        source = _RankedCandidateSource(
            frontier=frontier, positions=dict.fromkeys(eligible, 0)
        )
    with solve_phase(name="compilation_waves", logger=logger, call_id=call_id):
        wave = 0
        skipped_fallback_evaluations: dict[_CoreTriple, int] = {}
        pending = source.pending()
        while pending:
            wave_candidates = {
                triple: source.candidate(triple=triple) for triple in pending
            }
            for triple, candidate in wave_candidates.items():
                logger.debug(
                    "candidate evaluation %r %r",
                    triple,
                    dict(resolved_programs[candidate].tile_widths),
                )
            wave_lowering_keys = {
                candidate: lowering_keys[candidate]
                for candidate in wave_candidates.values()
            }
            compiled_before_lowerings = frozenset(compiled)
            new_lowerings = _uncompiled(keys=wave_lowering_keys, compiled=compiled)
            logger.info(
                "AOT compilation wave %d: %d unique lowerings for %d "
                "regime-period-core triples (%d workers)",
                wave,
                len(new_lowerings),
                len(pending),
                n_workers,
            )
            _lower_and_compile_wave(
                new_lowerings=new_lowerings,
                resolved_programs=resolved_programs,
                all_layouts=all_layouts,
                internal_templates=internal_templates,
                donations=donations,
                ages=ages,
                n_triples_per_lowering=_count_triples_per_lowering_key(
                    lowering_keys=wave_lowering_keys
                ),
                log_kernel_memory=budget_bytes is None,
                n_workers=n_workers,
                logger=logger,
                compiled=compiled,
                labels=labels,
            )
            admission_keys.update(wave_lowering_keys)
            for triple, candidate in wave_candidates.items():
                source.note_variants(
                    triple=triple,
                    keys=(lowering_keys[candidate],),
                    compiled_before=compiled_before_lowerings,
                )
            unique_variants_compiled = len(new_lowerings)
            wave_skipped_fallback_evaluations = 0
            # A candidate whose donating variant alone exceeds the budget is
            # refused at every variant, because the paired admission below takes
            # the larger of the two. Its fallback is therefore never asked for.
            if budget_bytes is None:
                survivors = dict(wave_candidates)
            else:
                survivors = {}
                primary_residency: dict[_CoreCandidate, int] = {}
                for triple in pending:
                    candidate = wave_candidates[triple]
                    primary_key = lowering_keys[candidate]
                    resident = _measure_variant(
                        variant_key=primary_key,
                        candidate=candidate,
                        triple=triple,
                        compiled=compiled,
                        labels=labels,
                        memory_by_lowering_key=memory_by_lowering_key,
                        resolved_programs=resolved_programs,
                        internal_templates=internal_templates,
                        resident_inventory=resident_inventory,
                        logger=logger,
                    )
                    resident_bytes_by_candidate[candidate] = resident
                    primary_residency[candidate] = resident
                    logger.debug(
                        "  resident at %r period %d core %r: %d bytes",
                        triple[0],
                        triple[1],
                        triple[2],
                        resident,
                    )
                    logger.debug(
                        "  conservative fixed owners: %r; "
                        "cell internal output reservation: %d bytes/device",
                        dict(resident_inventory[triple].fixed_bytes),
                        resident_inventory[triple].internal_bytes,
                    )
                    if (
                        memory_by_lowering_key[primary_key].reservation_bytes + resident
                        <= budget_bytes
                    ):
                        survivors[triple] = candidate
                        continue
                    if candidate in fallback_keys:
                        skipped_fallback_evaluations[triple] = (
                            skipped_fallback_evaluations.get(triple, 0) + 1
                        )
                        wave_skipped_fallback_evaluations += 1
                    source.record(
                        triple=triple,
                        reservation_bytes=memory_by_lowering_key[
                            primary_key
                        ].reservation_bytes,
                        resident_bytes=resident,
                        peak_bytes=memory_by_lowering_key[primary_key].peak_bytes,
                        admitted=False,
                    )
            wave_fallback_keys = {
                candidate: fallback_keys[candidate]
                for candidate in survivors.values()
                if candidate in fallback_keys
            }
            avoided_fallback_requests = len(
                {
                    fallback_keys[wave_candidates[triple]]
                    for triple in pending
                    if wave_candidates[triple] in fallback_keys
                    and triple not in survivors
                }
                - set(wave_fallback_keys.values())
                - set(compiled)
            )
            compiled_before_fallbacks = frozenset(compiled)
            new_fallbacks = _uncompiled(keys=wave_fallback_keys, compiled=compiled)
            _lower_and_compile_wave(
                new_lowerings=new_fallbacks,
                resolved_programs=resolved_programs,
                all_layouts=all_layouts,
                internal_templates=internal_templates,
                donations=fallback_donations,
                ages=ages,
                n_triples_per_lowering=_count_triples_per_lowering_key(
                    lowering_keys=wave_fallback_keys
                ),
                log_kernel_memory=budget_bytes is None,
                n_workers=n_workers,
                logger=logger,
                compiled=compiled,
                labels=labels,
            )
            for triple, candidate in survivors.items():
                if candidate in fallback_keys:
                    source.note_variants(
                        triple=triple,
                        keys=(fallback_keys[candidate],),
                        compiled_before=compiled_before_fallbacks,
                    )
            unique_variants_compiled += len(new_fallbacks)
            logger.info(
                "AOT compilation wave %d: %d unique variants compiled, %d fallback "
                "evaluations skipped, %d fallback requests avoided",
                wave,
                unique_variants_compiled,
                wave_skipped_fallback_evaluations,
                avoided_fallback_requests,
            )
            if budget_bytes is None:
                break
            for triple, candidate in survivors.items():
                variant_keys = tuple(
                    dict.fromkeys(
                        (
                            lowering_keys[candidate],
                            fallback_keys.get(candidate, lowering_keys[candidate]),
                        )
                    )
                )
                # The donating variant was measured when the candidate was
                # admitted on its own reservation; only the fallback is new.
                variant_residency = {
                    variant_key: primary_residency[candidate]
                    if variant_key == lowering_keys[candidate]
                    else _measure_variant(
                        variant_key=variant_key,
                        candidate=candidate,
                        triple=triple,
                        compiled=compiled,
                        labels=labels,
                        memory_by_lowering_key=memory_by_lowering_key,
                        resolved_programs=resolved_programs,
                        internal_templates=internal_templates,
                        resident_inventory=resident_inventory,
                        logger=logger,
                    )
                    for variant_key in variant_keys
                }
                # Keep each variant's compiler reservation paired with its own
                # kept-input subtraction and retained owners.
                lowering_key = max(
                    variant_keys,
                    key=lambda key: (
                        memory_by_lowering_key[key].reservation_bytes
                        + variant_residency[key]
                    ),
                )
                admission_keys[candidate] = lowering_key
                resident = variant_residency[lowering_key]
                resident_bytes_by_candidate[candidate] = resident
                source.record(
                    triple=triple,
                    reservation_bytes=memory_by_lowering_key[
                        lowering_key
                    ].reservation_bytes,
                    resident_bytes=resident,
                    peak_bytes=memory_by_lowering_key[lowering_key].peak_bytes,
                    admitted=(
                        memory_by_lowering_key[lowering_key].reservation_bytes
                        + resident
                        <= budget_bytes
                    ),
                )
            # A triple that fits at no width has its whole frontier compiled; the
            # planner below reports it with every candidate's peak in hand.
            pending = source.advance()
            wave += 1
        if isinstance(source, _BoundedCandidateSource):
            for triple in candidates_by_triple:
                source.report(triple=triple, logger=logger)

    with solve_phase(name="workspace_selection", logger=logger, call_id=call_id):
        memory_by_compiled_id = {
            id(compiled[lowering_key]): memory
            for lowering_key, memory in memory_by_lowering_key.items()
        }

        # Select within each triple through the planner, which walks the same ranked
        # frontier and stops at the same first feasible candidate; the waves above
        # compiled exactly the candidates it asks for, so the winner reaches dispatch
        # without a second lowering or compilation.
        selected_programs: dict[_CoreTriple, ResolvedCoreProgram] = {}
        selected_cores: dict[_CoreTriple, PlannedCore] = {}
        selected_fallbacks: dict[_CoreTriple, PlannedCore] = {}
        for triple, candidates in candidates_by_triple.items():
            programs_by_width = {
                candidate[1]: resolved_programs[candidate] for candidate in candidates
            }
            compiled_by_width = {
                candidate[1]: compiled[admission_keys[candidate]]
                for candidate in candidates
                if candidate in admission_keys
            }
            representative = resolved_programs[candidates[0]]
            try:
                plan = (
                    _bounded_plan(
                        triple=triple,
                        source=source,
                        compiled_by_width=compiled_by_width,
                        memory_by_compiled_id=memory_by_compiled_id,
                        budget_bytes=cast("int", budget_bytes),
                    )
                    if isinstance(source, _BoundedCandidateSource)
                    else plan_workspace(
                        axes=representative.requirements.axes,
                        fixed_widths=execution.widths_for(regime_name=triple[0]),
                        width_ceilings=execution.axis_width_ceilings,
                        compile_candidate=_CompiledCandidateLookup(
                            compiled_by_width=compiled_by_width
                        ),
                        budget_bytes=budget_bytes,
                        memory_for=(
                            None
                            if budget_bytes is None
                            else _CompilerMemoryLookup(
                                memory_by_compiled_id=memory_by_compiled_id
                            )
                        ),
                        resident_bytes=resident_bytes_by_triple[triple],
                        resident_bytes_for=(
                            None
                            if budget_bytes is None
                            else _CandidateResidencyLookup(
                                compiled_by_width=compiled_by_width,
                                resident_bytes_by_width={
                                    candidate[1]: resident_bytes_by_candidate[candidate]
                                    for candidate in candidates
                                    if candidate in resident_bytes_by_candidate
                                },
                            )
                        ),
                    )
                )
            except ExecutionPlanningError as error:
                if budget_bytes is None:
                    raise
                raise _width_selection_failure(
                    triple=triple,
                    resident_bytes=resident_bytes_by_triple[triple],
                    transfer_scratch_bytes=resident_inventory[
                        triple
                    ].transfer_scratch_bytes,
                    budget_bytes=budget_bytes,
                    execution=execution,
                    error=error,
                    skipped_fallback_evaluations=skipped_fallback_evaluations.get(
                        triple, 0
                    ),
                ) from error
            if execution.halve_on_materialised_gather:
                plan = _halve_while_gather_materialises(
                    triple=triple,
                    plan=plan,
                    axes=representative.requirements.axes,
                    fixed_widths=execution.widths_for(regime_name=triple[0]),
                    width_ceilings=execution.axis_width_ceilings,
                    frontier=frontier,
                    compile_candidate=functools.partial(
                        _lower_and_compile_candidate,
                        fallback_keys=fallback_keys,
                        fallback_donations=fallback_donations,
                        lowering_keys=lowering_keys,
                        resolved_programs=resolved_programs,
                        all_layouts=all_layouts,
                        internal_templates=internal_templates,
                        donations=donations,
                        ages=ages,
                        budget_bytes=budget_bytes,
                        n_workers=n_workers,
                        logger=logger,
                        compiled=compiled,
                        labels=labels,
                        memory_by_lowering_key=memory_by_lowering_key,
                        resident_inventory=resident_inventory,
                    ),
                    logger=logger,
                )
                programs_by_width[_width_key(widths=plan.widths)] = resolved_programs[
                    (triple, _width_key(widths=plan.widths))
                ]
            selected_candidate = (triple, _width_key(widths=plan.widths))
            selected = programs_by_width[selected_candidate[1]]
            selected_programs[triple] = selected
            selected_cores[triple] = _attach_resolved_output_layout(
                compiled=compiled[lowering_keys[selected_candidate]],
                layout=all_layouts[triple],
                tile_widths=plan.widths,
                input_transfer_plan=selected.input_transfer_plan,
                internal_input_templates=internal_templates[
                    (triple, _width_key(widths=plan.widths))
                ],
                donated_arguments=_donated_arguments(
                    donations=donations[(triple, _width_key(widths=plan.widths))]
                ),
                name=triple[2],
            )
            if selected_candidate in fallback_keys:
                selected_fallbacks[triple] = dataclasses.replace(
                    selected_cores[triple],
                    compiled=compiled[fallback_keys[selected_candidate]],
                    donated_arguments=(),
                )

    return _CompiledPrograms(
        executables=_group_cores_by_regime_period(selected_cores),
        metadata=_execution_metadata(programs=selected_programs),
        input_liveness=input_liveness,
        donations=MappingProxyType(
            {
                triple: donations[
                    (triple, _width_key(widths=selected_cores[triple].tile_widths))
                ]
                for triple in selected_cores
            }
        ),
        donation_fallbacks=MappingProxyType(selected_fallbacks),
    )


def _halve_while_gather_materialises(
    *,
    triple: _CoreTriple,
    plan: WorkspacePlan[jax.stages.Compiled],
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int],
    width_ceilings: Mapping[str, int],
    frontier: _LazyCandidateFrontier,
    compile_candidate: Callable[..., jax.stages.Compiled],
    logger: logging.Logger,
) -> WorkspacePlan[jax.stages.Compiled]:
    """Halve the cell width while the compiled program materialises a gather table.

    Whether XLA recomputes a gathered table inside the reduction or writes it to
    device memory and reads it back depends on the device, the whole program and
    the cell count, so it is read off the compiled program rather than predicted.
    Inside the fused region the width does not move the kernel's throughput, so
    halving is enough; no finer search follows. Each halving is rounded onto the
    widths the axis admits and never exceeds its ceiling.

    Raises:
        ExecutionPlanningError: The program still materialises a gather table at
            the narrowest width its axis admits.

    """
    axis = _halvable_axis(axes=axes, fixed_widths=fixed_widths)
    if axis is None:
        return plan
    widths = plan.widths
    executable = plan.compiled
    while (
        fusion := _materialised_gather_fusion(compiled=executable, widths=widths)
    ) is not None:
        width = widths[axis.name]
        narrower = _admissible_width(
            axis=axis, width=width // 2, ceiling=width_ceilings.get(axis.name)
        )
        if narrower >= width:
            msg = (
                f"The compiled program {_describe_candidate(candidate=(triple, ()))} "
                f"materialises a gather table in reduce fusion {fusion!r} at "
                f"{axis.name!r} width {width}, the narrowest width the axis admits. "
                "Set `ExecutionConfig(halve_on_materialised_gather=False)` to keep "
                "the materialised program."
            )
            raise ExecutionPlanningError(msg)
        logger.info(
            "  %r at %r period %d: reduce fusion %r materialises a gather table at "
            "%r width %d; recompiling at %d",
            triple[2],
            triple[0],
            triple[1],
            fusion,
            axis.name,
            width,
            narrower,
        )
        widths = MappingProxyType({**widths, axis.name: narrower})
        executable = compile_candidate(
            candidate=frontier.bind_widths(triple=triple, widths=widths)
        )
    return WorkspacePlan(widths=widths, peak_bytes=None, compiled=executable)


def _halvable_axis(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int],
) -> TiledOutputAxis | None:
    """Return the axis halved while a gather materialises, unless its width is fixed."""
    for axis in axes:
        if (
            isinstance(axis, TiledOutputAxis)
            and axis.halve_on_materialised_gather
            and axis.name not in fixed_widths
        ):
            return axis
    return None


def _materialised_gather_fusion(
    *, compiled: jax.stages.Compiled, widths: Mapping[str, int]
) -> str | None:
    """Name one reduce fusion reading a gather table another fusion wrote, if any."""
    del widths
    text = compiled.as_text()
    if text is None:
        msg = "A compiled solve program has no HLO text to classify."
        raise RuntimeError(msg)
    return next(
        (
            name
            for name, verdict in classify_reduce_fusions(text).items()
            if verdict is ReduceFusionVerdict.MATERIALISED_GATHER
        ),
        None,
    )


def _lower_and_compile_candidate(
    *,
    candidate: _CoreCandidate,
    fallback_keys: Mapping[_CoreCandidate, Hashable],
    fallback_donations: Mapping[_CoreCandidate, tuple[ResolvedDonation, ...]],
    lowering_keys: Mapping[_CoreCandidate, Hashable],
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    all_layouts: Mapping[_CoreTriple, ResolvedOutputLayout],
    internal_templates: Mapping[_CoreCandidate, Mapping[str, object]],
    donations: Mapping[_CoreCandidate, tuple[ResolvedDonation, ...]],
    ages: AgeGrid,
    budget_bytes: int | None,
    n_workers: int,
    logger: logging.Logger,
    compiled: dict[Hashable, jax.stages.Compiled],
    labels: dict[Hashable, str],
    memory_by_lowering_key: dict[Hashable, CompilerMemoryReservation],
    resident_inventory: Mapping[_CoreTriple, ResidentInventory],
) -> jax.stages.Compiled:
    """Compile one bound candidate and its donation-free variant outside the waves.

    Under a budget every variant is admitted as a wave admits it, on its compiler
    reservation plus the bytes it leaves resident at the core's position.

    Raises:
        ExecutionPlanningError: A variant of the candidate exceeds the budget.

    """
    variants = (
        (lowering_keys, donations),
        *(((fallback_keys, fallback_donations),) if candidate in fallback_keys else ()),
    )
    for keys, variant_donations in variants:
        new = _uncompiled(keys={candidate: keys[candidate]}, compiled=compiled)
        _lower_and_compile_wave(
            new_lowerings=new,
            resolved_programs=resolved_programs,
            all_layouts=all_layouts,
            internal_templates=internal_templates,
            donations=variant_donations,
            ages=ages,
            n_triples_per_lowering=dict.fromkeys(new, 1),
            log_kernel_memory=budget_bytes is None,
            n_workers=n_workers,
            logger=logger,
            compiled=compiled,
            labels=labels,
        )
        if budget_bytes is None:
            continue
        triple = candidate[0]
        resident = _measure_variant(
            variant_key=keys[candidate],
            candidate=candidate,
            triple=triple,
            compiled=compiled,
            labels=labels,
            memory_by_lowering_key=memory_by_lowering_key,
            resolved_programs=resolved_programs,
            internal_templates=internal_templates,
            resident_inventory=resident_inventory,
            logger=logger,
        )
        reservation = memory_by_lowering_key[keys[candidate]].reservation_bytes
        if reservation + resident > budget_bytes:
            msg = (
                f"The narrower width {dict(resolved_programs[candidate].tile_widths)!r}"
                f" of {_describe_candidate(candidate=candidate)} needs {reservation} "
                f"reservation bytes beside {resident} resident bytes, exceeding the "
                f"{budget_bytes}-byte budget."
            )
            raise ExecutionPlanningError(msg)
    return compiled[lowering_keys[candidate]]


# The planner's lookups are instances of module-level classes rather than functions
# defined per solve. The package's beartype claw decorates every `def` it imports,
# including one executed inside a call, and keeps each decorated function in a
# process-wide registry; a nested function would therefore pin its closure — here
# every compiled executable of a core — for the life of the process.
@dataclasses.dataclass(frozen=True, kw_only=True)
class _CompiledCandidateLookup:
    """Serve the planner's width requests from one triple's compiled candidates."""

    compiled_by_width: Mapping[_WidthKey, jax.stages.Compiled]

    def __call__(self, widths: Mapping[str, int]) -> jax.stages.Compiled:
        try:
            return self.compiled_by_width[_width_key(widths=widths)]
        except KeyError:
            msg = (
                "The workspace planner asked for a width candidate that no "
                f"compilation wave lowered: {dict(widths)!r}."
            )
            raise RuntimeError(msg) from None


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CompilerMemoryLookup:
    """Serve complete memory accounting from the compilation waves' reports."""

    memory_by_compiled_id: Mapping[int, CompilerMemoryReservation]

    def __call__(self, executable: jax.stages.Compiled) -> CompilerMemoryReservation:
        return self.memory_by_compiled_id[id(executable)]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CandidateResidencyLookup:
    """Use this triple's width-specific residency, never a shared executable cache."""

    compiled_by_width: Mapping[_WidthKey, jax.stages.Compiled]
    resident_bytes_by_width: Mapping[_WidthKey, int]

    def __call__(self, executable: jax.stages.Compiled) -> int:
        return max(
            self.resident_bytes_by_width[width]
            for width, candidate in self.compiled_by_width.items()
            if candidate is executable and width in self.resident_bytes_by_width
        )


def _bounded_candidate_source(
    *,
    frontier: _LazyCandidateFrontier,
    triples: tuple[_CoreTriple, ...],
    candidates_by_triple: Mapping[_CoreTriple, Sequence[_CoreCandidate]],
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    execution: ResolvedExecution,
) -> _BoundedCandidateSource:
    """Open one bounded width search per core, seeded and ready to be compiled.

    The axes come from the core's top-ranked candidate, whose declarations no
    width changes, and the hint from the policy's entry for the core's regime; a
    hint the declarations refuse is logged by the search and skipped. A core
    whose declarations admit a single width runs no search and is offered that
    width.
    """
    policy = execution.width_search
    selectors: dict[_CoreTriple, BoundedWidthSelector | None] = {}
    proposals: dict[_CoreTriple, Mapping[str, int]] = {}
    for triple in triples:
        representative = resolved_programs[candidates_by_triple[triple][0]]
        if triple not in frontier.frontiers:
            selectors[triple] = None
            proposals[triple] = representative.tile_widths
            continue
        selector = BoundedWidthSelector(
            axes=representative.requirements.axes,
            fixed_widths=execution.widths_for(regime_name=triple[0]),
            width_ceilings=execution.axis_width_ceilings,
            policy=policy,
            hint=policy.hints.get(triple[0]),
            label=_describe_candidate(candidate=(triple, ())),
        )
        selectors[triple] = selector
        proposals[triple] = cast("Mapping[str, int]", selector.propose())
    return _BoundedCandidateSource(
        frontier=frontier, selectors=selectors, proposals=proposals
    )


def _bounded_plan(
    *,
    triple: _CoreTriple,
    source: _BoundedCandidateSource,
    compiled_by_width: Mapping[_WidthKey, jax.stages.Compiled],
    memory_by_compiled_id: Mapping[int, CompilerMemoryReservation],
    budget_bytes: int,
) -> WorkspacePlan[jax.stages.Compiled]:
    """Return the plan one core's bounded search kept, or say what it spent.

    Raises:
        ExecutionPlanningError: The search spent its evaluations without an
            admission, or the core's single admissible width was refused.

    """
    if triple not in source.selectors:
        msg = (
            "No width was searched for "
            f"{_describe_candidate(candidate=(triple, ()))}: the bytes the plan "
            "already keeps resident at its position reach the budget, so no "
            "workspace fits beside them."
        )
        raise ExecutionPlanningError(msg)
    widths = source.selected(triple=triple)
    if widths is None:
        selector = source.selectors[triple]
        if selector is None:
            msg = (
                "The only width "
                f"{_describe_candidate(candidate=(triple, ()))} admits was refused, "
                "so its bounded width search had nothing narrower to propose."
            )
            raise ExecutionPlanningError(msg)
        raise ExecutionPlanningError(selector.exhaustion_message(budget=budget_bytes))
    executable = compiled_by_width[_width_key(widths=widths)]
    memory = memory_by_compiled_id[id(executable)]
    return WorkspacePlan(
        widths=widths,
        peak_bytes=memory.peak_bytes,
        reservation_bytes=memory.reservation_bytes,
        compiled=executable,
    )


@dataclasses.dataclass(kw_only=True)
class _RankedCandidateSource:
    """Offer every core its next ranked width candidate, widest first.

    One position per core walks the ranked frontier. A refusal advances the
    position by one, so the next wave asks the frontier for the next candidate
    and binds it on the way; a core whose frontier is spent is simply left out
    of the next wave and reported by name when selection reaches it.
    """

    frontier: _LazyCandidateFrontier
    positions: dict[_CoreTriple, int]
    """The rank each still-pending core is being offered."""

    def __post_init__(self) -> None:
        """Start with no core queued for the wave after this one."""
        self._queued: dict[_CoreTriple, int] = {}

    def pending(self) -> tuple[_CoreTriple, ...]:
        """Name the cores this wave evaluates, in their established order."""
        return tuple(self.positions)

    def candidate(self, *, triple: _CoreTriple) -> _CoreCandidate:
        """Bind and return the candidate this core is being offered."""
        return self.frontier.candidate(triple=triple, position=self.positions[triple])

    def record(
        self,
        *,
        triple: _CoreTriple,
        reservation_bytes: int,
        resident_bytes: int,
        peak_bytes: int,
        admitted: bool,
    ) -> None:
        """Advance a refused core to its next rank and leave an admitted one."""
        del reservation_bytes, resident_bytes, peak_bytes
        if not admitted:
            _queue_next_candidate(
                triple=triple,
                position=self.positions[triple],
                frontier_lengths=self.frontier.frontier_lengths,
                next_pending=self._queued,
            )

    def note_variants(
        self,
        *,
        triple: _CoreTriple,
        keys: Iterable[Hashable],
        compiled_before: Container[Hashable],
    ) -> None:
        """Ignore the variants a wave lowered; the ranked walk reports none."""
        del triple, keys, compiled_before

    def advance(self) -> tuple[_CoreTriple, ...]:
        """Open the next wave over the cores a refusal queued for it."""
        self.positions = self._queued
        self._queued = {}
        return tuple(self.positions)


@dataclasses.dataclass(kw_only=True)
class _BoundedCandidateSource:
    """Offer every core the width its bounded search proposes next.

    Each core carries its own `BoundedWidthSelector`, which owns the seed,
    shrink and refine walk and nothing else: the waves compile what it proposes,
    admission decides, and the verdict goes back to the selector. A core whose
    declarations admit a single width has no search to run — it is offered that
    width once, and a refusal exhausts it.
    """

    frontier: _LazyCandidateFrontier
    selectors: Mapping[_CoreTriple, BoundedWidthSelector | None]
    """The search of each core, or `None` for a core with one admissible width."""
    proposals: dict[_CoreTriple, Mapping[str, int]]
    """The width mapping each still-pending core is being offered."""

    def __post_init__(self) -> None:
        """Start with no core queued and no single-width core decided."""
        self._queued: dict[_CoreTriple, Mapping[str, int]] = {}
        self._single_admitted: dict[_CoreTriple, Mapping[str, int]] = {}
        self._variants: dict[_CoreTriple, dict[Hashable, bool]] = {}

    def pending(self) -> tuple[_CoreTriple, ...]:
        """Name the cores this wave evaluates, in their established order."""
        return tuple(self.proposals)

    def candidate(self, *, triple: _CoreTriple) -> _CoreCandidate:
        """Bind and return the candidate for the width this core was offered."""
        return self.frontier.bind_widths(triple=triple, widths=self.proposals[triple])

    def record(
        self,
        *,
        triple: _CoreTriple,
        reservation_bytes: int,
        resident_bytes: int,
        peak_bytes: int,
        admitted: bool,
    ) -> None:
        """Hand one verdict to the core's search and queue what it proposes next."""
        selector = self.selectors[triple]
        if selector is None:
            if admitted:
                self._single_admitted[triple] = self.proposals[triple]
            return
        selector.record(
            widths=self.proposals[triple],
            reservation_bytes=reservation_bytes,
            resident_bytes=resident_bytes,
            peak_bytes=peak_bytes,
            admitted=admitted,
        )
        proposal = selector.propose()
        if proposal is not None:
            self._queued[triple] = proposal

    def note_variants(
        self,
        *,
        triple: _CoreTriple,
        keys: Iterable[Hashable],
        compiled_before: Container[Hashable],
    ) -> None:
        """Attribute the lowering keys one wave's evaluation of a core asked for.

        A key is a compiler request when no wave had compiled it by the time
        this wave opened, and a cache hit otherwise; `compiled_before` is the
        set of compiled keys as the wave began. Two cores evaluated in the same
        wave that share a key therefore each count it as a request, because
        neither waited on the other. A key reached again by a later evaluation
        of the same core keeps the verdict of its first appearance.
        """
        seen = self._variants.setdefault(triple, {})
        for key in keys:
            if key not in seen:
                seen[key] = key not in compiled_before

    def advance(self) -> tuple[_CoreTriple, ...]:
        """Open the next wave over the cores whose search proposed another width."""
        self.proposals = self._queued
        self._queued = {}
        return tuple(self.proposals)

    def selected(self, *, triple: _CoreTriple) -> Mapping[str, int] | None:
        """Return the widths this core's search kept, or `None` when it found none."""
        selector = self.selectors[triple]
        if selector is None:
            return self._single_admitted.get(triple)
        decision = selector.selected
        return None if decision is None else decision.widths

    def report(self, *, triple: _CoreTriple, logger: logging.Logger) -> None:
        """Log what one core's search spent and what it kept.

        The counts are the ones the waves measured: the evaluations the search
        spent, the distinct lowering keys those evaluations asked for — the
        donating variant and, where one exists, its fallback — and the split of
        those keys into the requests that reached the compiler and the hits an
        earlier wave had already served.

        A core the waves never opened a search for — one with a single
        admissible width, or one whose residency already reaches the budget —
        has nothing to report.
        """
        selector = self.selectors.get(triple)
        if selector is None:
            return
        variants = self._variants.get(triple, {})
        requests = sum(variants.values())
        logger.info(
            "bounded width search %s: %d evaluations, %d unique variants compiled, "
            "%d compiler requests, %d cache hits; selected %r",
            _describe_candidate(candidate=(triple, ())),
            len(selector.decisions),
            len(variants),
            requests,
            len(variants) - requests,
            None if selector.selected is None else dict(selector.selected.widths),
        )


def _uncompiled(
    *,
    keys: Mapping[_CoreCandidate, Hashable],
    compiled: Mapping[Hashable, jax.stages.Compiled],
) -> dict[Hashable, _CoreCandidate]:
    """Name one representative candidate per key no wave has compiled yet."""
    new: dict[Hashable, _CoreCandidate] = {}
    for candidate, key in keys.items():
        if key not in compiled:
            new.setdefault(key, candidate)
    return new


def _measure_variant(
    *,
    variant_key: Hashable,
    candidate: _CoreCandidate,
    triple: _CoreTriple,
    compiled: Mapping[Hashable, jax.stages.Compiled],
    labels: Mapping[Hashable, str],
    memory_by_lowering_key: dict[Hashable, CompilerMemoryReservation],
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    internal_templates: Mapping[_CoreCandidate, Mapping[str, object]],
    resident_inventory: Mapping[_CoreTriple, ResidentInventory],
    logger: logging.Logger,
) -> int:
    """Report one variant's residency, reserving its compiler memory once.

    The reservation is cached under the variant's own key, so two triples
    sharing a key read one report; the residency is taken per candidate,
    because the bytes a variant leaves resident depend on the triple's
    position in the plan.
    """
    if variant_key not in memory_by_lowering_key:
        memory = compiler_memory_reservation(
            compiled=compiled[variant_key],
            widths=resolved_programs[candidate].tile_widths,
        )
        memory_by_lowering_key[variant_key] = memory
        _log_kernel_memory(
            compiled=compiled[variant_key],
            label=labels[variant_key],
            logger=logger,
            precomputed_peak_bytes=memory.peak_bytes,
        )
    return _candidate_resident_bytes(
        compiled=compiled[variant_key],
        program=resolved_programs[candidate],
        internal_arguments=internal_templates[candidate],
        inventory=resident_inventory[triple],
    )


def _queue_next_candidate(
    *,
    triple: _CoreTriple,
    position: int,
    frontier_lengths: Mapping[_CoreTriple, int],
    next_pending: dict[_CoreTriple, int],
) -> None:
    """Queue the triple's next ranked candidate when its frontier offers one.

    A triple whose frontier is spent is simply left out of the next wave; the
    planner reports it by name when selection reaches it.
    """
    if position + 1 < frontier_lengths[triple]:
        next_pending[triple] = position + 1


def _lower_and_compile_wave(
    *,
    new_lowerings: Mapping[Hashable, _CoreCandidate],
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    all_layouts: Mapping[_CoreTriple, ResolvedOutputLayout],
    internal_templates: Mapping[_CoreCandidate, Mapping[str, object]],
    donations: Mapping[_CoreCandidate, tuple[ResolvedDonation, ...]],
    ages: AgeGrid,
    n_triples_per_lowering: Mapping[Hashable, int],
    log_kernel_memory: bool,
    n_workers: int,
    logger: logging.Logger,
    compiled: dict[Hashable, jax.stages.Compiled],
    labels: dict[Hashable, str],
) -> None:
    """Lower one wave's new candidates sequentially and compile them in parallel.

    Tracing is single-threaded, so lowering runs on the calling thread; XLA releases
    the GIL, so compilation fans out over a thread pool. Executables and their log
    labels land in `compiled` and `labels`, keyed by lowering key.
    """
    lowered: dict[Hashable, jax.stages.Lowered] = {}
    n_unique = len(new_lowerings)
    for i, (lowering_key, candidate) in enumerate(new_lowerings.items(), 1):
        triple, _ = candidate
        regime_name, period, core_key = triple
        resolved = resolved_programs[candidate]
        static_kwargs = resolved.static_kwargs
        label = (
            f"{regime_name} {core_key} (age {ages.values[period].item()}, "
            f"widths={dict(resolved.tile_widths)!r})"
        )
        labels[lowering_key] = label
        log_module_fanout(
            label=label,
            n_triples=n_triples_per_lowering[lowering_key],
            logger=logger,
        )
        logger.info("%d/%d  %s", i, n_unique, label)
        logger.info("  lowering ...")
        start = time.monotonic()
        layout = all_layouts[triple]
        donated = _donated_arguments(donations=donations[candidate])
        jitted = jax.jit(
            resolved.function,
            static_argnames=tuple(static_kwargs),
            out_shardings=layout.out_shardings,
            donate_argnames=donated or None,
        )
        low = jitted.lower(
            **resolved.arguments, **internal_templates[candidate], **static_kwargs
        )
        _assert_lowered_output_roles(
            lowered=low,
            output_roles=resolved.output_roles,
            layout=layout,
            label=label,
        )
        lowered[lowering_key] = low
        elapsed = time.monotonic() - start
        logger.info("  lowered in %s", format_duration(seconds=elapsed))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _compile_and_log,
                lowering_key=lowering_key,
                low=low,
                label=labels[lowering_key],
                log_kernel_memory=log_kernel_memory,
                logger=logger,
            )
            for lowering_key, low in lowered.items()
        ]
        for future in as_completed(futures):
            lowering_key, comp = future.result()
            compiled[lowering_key] = comp


def _compile_and_log(
    *,
    lowering_key: Hashable,
    low: jax.stages.Lowered,
    label: str,
    log_kernel_memory: bool,
    logger: logging.Logger,
) -> tuple[Hashable, jax.stages.Compiled]:
    """Compile one lowered program on a pool thread and log its timing."""
    logger.info("  compiling %s ...", label)
    start = time.monotonic()
    result = low.compile()
    elapsed = time.monotonic() - start
    logger.info("  compiled  %s  %s", label, format_duration(seconds=elapsed))
    if log_kernel_memory:
        _log_kernel_memory(compiled=result, label=label, logger=logger)
    return lowering_key, result


def _execution_metadata(
    *, programs: Mapping[_CoreTriple, ResolvedCoreProgram]
) -> MappingProxyType[_CoreTriple, _ProgramExecutionMetadata]:
    """Retain selected declaration facts without pinning argument templates."""
    return MappingProxyType(
        {
            triple: _ProgramExecutionMetadata(
                requirements=program.requirements,
                disposition=program.disposition,
                scope=program.scope,
                input_transfer_plan=program.input_transfer_plan,
            )
            for triple, program in programs.items()
        }
    )


def _count_triples_per_lowering_key(
    *,
    lowering_keys: Mapping[_CoreCandidate, Hashable],
) -> dict[Hashable, int]:
    """Count the candidate addresses each compiled module will serve.

    A shared callable with distinct output layouts is deliberately counted as
    distinct lowered modules.
    """
    counts: dict[Hashable, int] = {}
    for key in lowering_keys.values():
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fail_if_one_key_covers_two_callables(
    *,
    lowering_keys: Mapping[_CoreCandidate, Hashable],
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
) -> None:
    """Reject a compilation key that two different callables would answer to.

    A key is an assertion about what a program computes, and the callables are
    the oracle for it: two candidates that agree on the key while carrying
    different callables mean the key promises a sharing the callables do not
    support, and compiling one of them once would run one period's closure in
    another period's place. Two declarations reach that state:

    - the published identity is coarser than what the solver specialized;
    - the solver builds an equivalent but distinct callable per period it groups
      under one key, where the key promises one callable object.

    Both colliding addresses are named, so a reader can see which two programs
    the identity failed to tell apart.

    Args:
        lowering_keys: The compilation key of every resolved candidate.
        resolved_programs: The candidates' resolved programs.

    Raises:
        ExecutionPlanningError: If two candidates share a key over different
            callables.

    """
    seen_by_key: dict[Hashable, tuple[_CoreCandidate, Hashable]] = {}
    for candidate, lowering_key in lowering_keys.items():
        callable_key = _func_dedup_key(func=resolved_programs[candidate].function)
        known_candidate, known_callable = seen_by_key.setdefault(
            lowering_key, (candidate, callable_key)
        )
        if known_callable != callable_key:
            msg = (
                "Two core programs share one compilation key but are different "
                f"callables: {_describe_candidate(candidate=known_candidate)} and "
                f"{_describe_candidate(candidate=candidate)}. Either the program "
                "identity is too coarse for this solver's specialization, or the "
                "solver builds an equivalent but distinct callable for each period "
                "it groups under one key; periods grouped under one key must reach "
                "one callable object."
            )
            raise ExecutionPlanningError(msg)


def _describe_candidate(*, candidate: _CoreCandidate) -> str:
    """Name one candidate's address in the words a model author uses."""
    regime_name, period, core_name = candidate[0]
    return f"regime {regime_name!r}, core {core_name!r}, period {period}"


def _resolve_candidate_donations(
    *,
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    representatives: Mapping[_CoreTriple, ResolvedCoreProgram],
    input_liveness: PlannedInputLiveness[_InputDispatch, ValueArtifactAddress],
    n_periods: int,
    enable_jit: bool,
    donate_buffers: bool,
) -> tuple[
    dict[_CoreCandidate, tuple[ResolvedDonation, ...]],
    dict[
        tuple[int, RegimeName],
        MappingProxyType[ValueArtifactAddress, frozenset[ValueConsumerAddress]],
    ],
    dict[_CoreCandidate, tuple[ResolvedDonation, ...]],
]:
    """Decide what each bound candidate donates, and against which read census.

    Returns:
        The ledger's nominations, the unit read census every dispatch is judged
        against, and the nominations that survived that census.

    """
    nominations = {
        candidate: (
            resolve_donations(
                program=resolved,
                dispatch=(candidate[0][1], candidate[0][0]),
                ledger=input_liveness,
                n_periods=n_periods,
            )
            if enable_jit and donate_buffers
            else ()
        )
        for candidate, resolved in resolved_programs.items()
    }
    # Two cores nominating one artifact is a declaration defect, so it is
    # refused here, over the nominations: withholding one of them afterwards
    # would resolve the competition instead of reporting it.
    _fail_if_a_unit_donates_one_artifact_twice(donations=nominations)
    # Donation consumes one executable input, so the surviving nominations need
    # the unit's read census next to the ledger's per-dispatch count. The
    # representatives carry it: every core of the unit appears once, and the
    # declared reads a census counts do not vary with the width.
    unit_programs: dict[tuple[int, RegimeName], list[ResolvedCoreProgram]] = {}
    for (regime_name, period, _core_key), resolved in representatives.items():
        unit_programs.setdefault((period, regime_name), []).append(resolved)
    readers_by_dispatch = {
        dispatch: unit_input_readers(programs=programs)
        for dispatch, programs in unit_programs.items()
    }
    donations = {
        candidate: withhold_shared_donations(
            program=resolved_programs[candidate],
            donations=decisions,
            unit_readers=readers_by_dispatch[(candidate[0][1], candidate[0][0])],
        )
        for candidate, decisions in nominations.items()
    }
    return nominations, readers_by_dispatch, donations


def _checked_producer_records(
    *,
    candidates: Sequence[ResolvedCoreProgram],
    templates: Mapping[str, object],
) -> MappingProxyType[Hashable, ResolvedProducer]:
    """Publish one producer record per width candidate, invariance established.

    A consumer is lowered against the record of the producer's top-ranked
    candidate, before the budget selects which width runs, so every candidate of
    the frontier has to publish the same subtree per label — the same shape, the
    same dtype and the same weak typing. That is a property of the program, not
    of the plan, so it is established over the whole ranked frontier rather than
    over the candidates admission happens to reach.

    Args:
        candidates: The core's resolved width candidates, ranked widest first.
        templates: The internal-input templates the core was resolved with.

    Returns:
        The records, keyed by width and ordered as the frontier ranks them, so
        that the first is the top-ranked candidate the consumers are lowered
        against.

    Raises:
        ExecutionPlanningError: Two candidates publish different subtrees.

    """
    records: dict[Hashable, ResolvedProducer] = {
        _width_key(widths=candidate.tile_widths): resolve_producer(
            program=candidate, templates=templates
        )
        for candidate in candidates
    }
    assert_width_invariant_internal_outputs(candidates=records)
    return MappingProxyType(records)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CoreFrontier:
    """What one core's not-yet-bound width candidates are bound from.

    Everything here is decided before any width is: the abstract program, the
    resolved transfer plan it shares with every width, the internal-input
    templates its cell published, and the ranked width frontier. Binding a later
    candidate therefore asks nothing of the cell it was materialized in, and the
    trees retained are abstract.
    """

    program: MaterializedCoreProgram
    transfer_plan: tuple[ResolvedValueTransfer, ...]
    templates: Mapping[str, object]
    widths: tuple[Mapping[str, object] | None, ...]
    consumed: bool
    prebound: tuple[ResolvedCoreProgram, ...] | None
    """Every candidate already resolved abstractly, for a consumed producer.

    A consumed producer's whole frontier is resolved before any width is
    selected, so that its published internal outputs can be held against one
    another; binding a later candidate then reads the resolution off this tuple
    instead of repeating it. `None` for a core no consumer reads, whose
    candidates are resolved one refusal at a time.
    """


@dataclasses.dataclass(kw_only=True)
class _LazyCandidateFrontier:
    """Bind a core's ranked width candidates one refusal at a time.

    `plan_workspace` walks the ranked frontier widest first and keeps the first
    candidate admission admits, so a core whose widest candidate fits never needs
    a narrower one resolved, lowered or compiled. This holds the state the
    whole-frontier binding built once — the marks census, the liveness ledger, the
    unit read census and the per-core argument descriptions — and extends it by
    exactly one candidate whenever a compilation wave reports a refusal.

    The candidate order is the width frontier's, unchanged, and every derived
    fact a later candidate needs is width-invariant and already complete, so the
    plan admission sees is the plan it saw when the whole frontier was bound
    ahead of it.
    """

    frontiers: Mapping[_CoreTriple, _CoreFrontier]
    candidates_by_triple: dict[_CoreTriple, list[_CoreCandidate]]
    frontier_lengths: Mapping[_CoreTriple, int]
    layouts: Mapping[_CoreTriple, ResolvedOutputLayout]
    resolved_programs: dict[_CoreCandidate, ResolvedCoreProgram]
    internal_templates: dict[_CoreCandidate, Mapping[str, object]]
    nominations: dict[_CoreCandidate, tuple[ResolvedDonation, ...]]
    donations: dict[_CoreCandidate, tuple[ResolvedDonation, ...]]
    lowering_keys: dict[_CoreCandidate, Hashable]
    argument_keys: dict[_CoreTriple, Hashable]
    transfer_consumers: Mapping[_ConsumerKey, set[_CoreTriple]]
    readers_by_dispatch: Mapping[
        tuple[int, RegimeName],
        Mapping[ValueArtifactAddress, frozenset[ValueConsumerAddress]],
    ]
    input_liveness: PlannedInputLiveness[_InputDispatch, ValueArtifactAddress]
    regimes: MappingProxyType[RegimeName, Regime]
    program_fingerprint: str
    n_periods: int
    enable_jit: bool
    donate_buffers: bool
    fallback_keys: dict[_CoreCandidate, Hashable] = dataclasses.field(
        default_factory=dict
    )
    fallback_donations: dict[_CoreCandidate, tuple[ResolvedDonation, ...]] = (
        dataclasses.field(default_factory=dict)
    )
    fallback_argument_keys: dict[_CoreTriple, Hashable] = dataclasses.field(
        default_factory=dict
    )

    def bind_fallbacks(
        self,
        *,
        fallback_keys: dict[_CoreCandidate, Hashable],
        fallback_donations: dict[_CoreCandidate, tuple[ResolvedDonation, ...]],
        argument_keys: dict[_CoreTriple, Hashable],
    ) -> None:
        """Extend the caller's donation-free variant maps alongside the frontier."""
        self.fallback_keys = fallback_keys
        self.fallback_donations = fallback_donations
        self.fallback_argument_keys = argument_keys

    def candidate(self, *, triple: _CoreTriple, position: int) -> _CoreCandidate:
        """Return the candidate at one rank, binding the ranks up to it on the way."""
        bound = self.candidates_by_triple[triple]
        while len(bound) <= position:
            self._bind_next(triple=triple)
        return bound[position]

    def _bind_next(self, *, triple: _CoreTriple) -> None:
        """Bind one core's next-ranked candidate and everything derived from it."""
        frontier = self.frontiers[triple]
        position = len(self.candidates_by_triple[triple])
        # A consumed producer was resolved at every width before any consumer
        # of it was lowered, so that the subtrees it publishes could be held
        # against one another over the whole frontier; this reads that
        # resolution rather than repeating it.
        resolved = (
            frontier.prebound[position]
            if frontier.prebound is not None
            else resolve_core_program_candidates(
                program=frontier.program,
                tile_widths=frontier.widths[position : position + 1],
                input_transfer_plan=frontier.transfer_plan,
                abstract_inputs=True,
            )[0]
        )
        self._bind_resolved(triple=triple, resolved=resolved)

    def bind_widths(
        self, *, triple: _CoreTriple, widths: Mapping[str, int]
    ) -> _CoreCandidate:
        """Bind one core at a width mapping the ranked frontier never offered.

        A bounded width search proposes widths between the ranked candidates, so
        the mapping handed here is resolved through the same
        `resolve_core_program_candidates` call a ranked candidate is resolved
        through, with the same abstract inputs and the same transfer plan. A
        consumed producer's new candidate is then held against every candidate
        already resolved for it, so the width-invariance of the subtrees it
        publishes is established for the proposed width too.

        Args:
            triple: The regime, period and core the width belongs to.
            widths: One width per declared execution axis.

        Returns:
            The bound candidate, which is the one already bound when this width
            was evaluated before.

        Raises:
            ValueError: An axis declaration refuses the width.
            ExecutionPlanningError: The core retains no frontier to bind from,
                or a candidate publishes a width-dependent internal output.

        """
        width_key = _width_key(widths=widths)
        for bound in self.candidates_by_triple[triple]:
            if bound[1] == width_key:
                return bound
        frontier = self.frontiers.get(triple)
        if frontier is None:
            msg = (
                "A width search asked for widths "
                f"{dict(widths)!r} at {_describe_candidate(candidate=(triple, ()))}, "
                "whose single admissible width was bound before the search began."
            )
            raise ExecutionPlanningError(msg)
        resolved = resolve_core_program_candidates(
            program=frontier.program,
            tile_widths=(MappingProxyType(dict(widths)),),
            input_transfer_plan=frontier.transfer_plan,
            abstract_inputs=True,
        )[0]
        if frontier.prebound is not None:
            _checked_producer_records(
                candidates=(*frontier.prebound, resolved),
                templates=frontier.templates,
            )
        return self._bind_resolved(triple=triple, resolved=resolved)

    def _bind_resolved(
        self, *, triple: _CoreTriple, resolved: ResolvedCoreProgram
    ) -> _CoreCandidate:
        """Derive everything one resolved candidate needs and record it."""
        frontier = self.frontiers[triple]
        bound = self.candidates_by_triple[triple]
        templates = frontier.templates
        candidate = (triple, _width_key(widths=resolved.tile_widths))
        resolved = _apply_transfer_marks(
            programs={candidate: resolved}, consumers=self.transfer_consumers
        )[candidate]
        self.resolved_programs[candidate] = resolved
        self.internal_templates[candidate] = templates
        regime_name, period, _core_key = triple
        self.nominations[candidate] = (
            resolve_donations(
                program=resolved,
                dispatch=(period, regime_name),
                ledger=self.input_liveness,
                n_periods=self.n_periods,
            )
            if self.enable_jit and self.donate_buffers
            else ()
        )
        _fail_if_a_unit_donates_one_artifact_twice(donations=self.nominations)
        self.donations[candidate] = withhold_shared_donations(
            program=resolved,
            donations=self.nominations[candidate],
            unit_readers=self.readers_by_dispatch[(period, regime_name)],
        )
        self.lowering_keys.update(
            _lowering_keys(
                resolved_programs={candidate: resolved},
                internal_templates={candidate: templates},
                layouts=self.layouts,
                donations=self.donations,
                regimes=self.regimes,
                program_fingerprint=self.program_fingerprint,
                argument_keys=self.argument_keys,
            )
        )
        _fail_if_one_key_covers_two_callables(
            lowering_keys=self.lowering_keys,
            resolved_programs=self.resolved_programs,
        )
        if _donated_arguments(donations=self.donations[candidate]):
            self.fallback_donations[candidate] = ()
            self.fallback_keys.update(
                _lowering_keys(
                    resolved_programs={candidate: resolved},
                    internal_templates={candidate: templates},
                    layouts=self.layouts,
                    donations=self.fallback_donations,
                    regimes=self.regimes,
                    program_fingerprint=self.program_fingerprint,
                    argument_keys=self.fallback_argument_keys,
                )
            )
        bound.append(candidate)
        return candidate


def _resolve_output_layouts_and_lowering_keys(
    *,
    all_programs: Mapping[_CoreTriple, CoreProgram],
    regimes: MappingProxyType[RegimeName, Regime],
    program_fingerprint: str,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    budget_bytes: int | None,
    execution_widths: ResolvedExecution,
    enable_jit: bool,
    continuous_sharded_state: str | None = None,
    donate_buffers: bool = True,
    retain_all_artifacts: bool,
    persistable_artifact_refs: frozenset[ArtifactRef],
    process_grid_resolver: ProcessGridResolver | None = None,
) -> tuple[
    dict[_CoreTriple, ResolvedOutputLayout],
    dict[_CoreCandidate, Hashable],
    dict[_CoreCandidate, ResolvedCoreProgram],
    dict[_CoreCandidate, Mapping[str, object]],
    PlannedInputLiveness[_InputDispatch, ValueArtifactAddress],
    dict[_CoreCandidate, tuple[ResolvedDonation, ...]],
    MappingProxyType[_CoreTriple, _ProgramExecutionMetadata],
    _LazyCandidateFrontier,
]:
    """Materialize once, then bind the top-ranked width candidate of every core.

    Programs are visited so every producer of an internal output is materialized
    before the consumers that read it, and each consumer is lowered against the
    producer's abstract output rather than a stand-in.

    Only the top-ranked candidate of each core is *bound* here. The planner
    accepts the first candidate of the ranked frontier that admission lets it
    keep, so every narrower candidate is waste on a core whose widest one fits.
    The returned `_LazyCandidateFrontier` binds candidate `k + 1` of a core once
    the compilation waves have seen candidate `k` refused, from the same
    materialized program, the same declaration order and the same width
    frontier, so both the candidates offered to admission and the order they are
    offered in are the ones the whole-frontier binding produced.

    A core some consumer reads is the exception: every candidate of its frontier
    is resolved abstractly here, before any consumer of it is traced. Each
    producer is traced with everything it is lowered with — its dynamic
    arguments, the templates of the internal inputs it reads itself, and its
    planner-owned static widths — and its candidates must publish one subtree per
    label, since a consumer is lowered against the top-ranked subtree before the
    producer's width is selected. That is a property of the program rather than
    of the plan, so it is established over the whole ranked frontier and not only
    over the candidates admission happens to reach. Only what a candidate costs
    to *lower* — its compilation key and its donation decision — waits for the
    refusal that binds it, and the resolution it waits with is this one.

    Each candidate's lowering key opens with the program's durable identity —
    the model, the regime, the core, and both groupings of its period — so a
    key says what the program computes rather than which object computes it.

    Reuse marks are applied once every core has published its plan, because a
    transfer shared by two source cores is only visible across the whole set. The
    census ranges over cores, not over widths, so it is complete here and every
    later candidate is marked against it. The keys are computed over the marked
    programs, and the mark is outside every specialization key, so it moves none
    of them.

    The ledger is built from one representative per triple — the top-ranked
    candidate, whose declared reads and transfer plan no width changes — the
    donation set of every candidate is decided against it, and only then are the
    keys computed, so a key names everything the executable is lowered with.
    """
    layouts: dict[_CoreTriple, ResolvedOutputLayout] = {}
    resolved_programs: dict[_CoreCandidate, ResolvedCoreProgram] = {}
    internal_templates: dict[_CoreCandidate, Mapping[str, object]] = {}
    # The frontier of a core with one candidate is exhausted by the binding
    # below, so nothing is kept for it; only a budgeted frontier retains what a
    # later candidate is bound from.
    frontiers: dict[_CoreTriple, _CoreFrontier] = {}
    candidates_by_triple: dict[_CoreTriple, list[_CoreCandidate]] = {}
    frontier_lengths: dict[_CoreTriple, int] = {}
    # A producer's records are kept only while its own graph is being resolved,
    # and only when some consumer of that graph names it, so no argument tree is
    # held across the whole solve. Each record adds one abstract-shape tree.
    producers: dict[str, MappingProxyType[Hashable, ResolvedProducer]] = {}
    consumed_names: frozenset[str] = frozenset()
    current_cell: tuple[RegimeName, int] | None = None
    ordered_programs = _programs_in_producer_order(all_programs=all_programs)
    for (regime_name, period, core_key), declaration in ordered_programs.items():
        triple = (regime_name, period, core_key)
        if (regime_name, period) != current_cell:
            current_cell = (regime_name, period)
            producers = {}
            consumed_names = _consumed_producer_names(
                all_programs=all_programs, regime_name=regime_name, period=period
            )
        regime = regimes[regime_name]
        state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name],
            process_grid_resolver=process_grid_resolver,
        )
        edge_kwargs = _edge_kwargs(
            regime=regime,
            regime_name=regime_name,
            next_edge_to_V_arr=next_edge_to_V_arr,
        )
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            edge_regime_to_V_arr=cast(
                "Mapping[str, object] | None",
                edge_kwargs.get("edge_regime_to_V_arr"),
            ),
        )
        materialized = materialize_core_program(program=declaration, context=context)
        templates = internal_input_templates(program=materialized, producers=producers)
        materialized, transfer_plan = _prepare_abstract_program(
            program=materialized,
            source_value_template=next_regime_to_V_arr[regime_name],
            source=triple,
            require_full_next_value=_continuous_value_replica_required(
                regime=regime, state_name=continuous_sharded_state
            ),
        )
        width_candidates = workspace_width_candidates(
            axes=materialized.requirements.axes,
            fixed_widths=execution_widths.widths_for(regime_name=regime_name),
            width_ceilings=execution_widths.axis_width_ceilings,
            budget_bytes=budget_bytes,
        )
        state_order = tuple(
            name
            for name in state_action_space.states
            if name not in regime.fold_state_names
        )
        consumed = core_key in consumed_names
        # Width invariance of a published internal output is a property of the
        # program, not of the plan: a consumer is lowered against the producer's
        # top-ranked subtree, so a candidate whose published shape, dtype or weak
        # typing follows the width is a defect however the budget later selects.
        # Every candidate of a consumed producer is therefore resolved
        # abstractly here, before any consumer of it is lowered, while the parts
        # that cost a lowering — the compilation key and the donation decision —
        # stay with the lazy frontier and bind one refusal at a time.
        eager = resolve_core_program_candidates(
            program=materialized,
            tile_widths=width_candidates if consumed else width_candidates[:1],
            input_transfer_plan=transfer_plan,
            abstract_inputs=True,
        )
        # A core whose cell axis may be halved after compilation keeps its
        # frontier even at one ranked width, so the narrower width can be bound.
        if len(width_candidates) > 1 or (
            execution_widths.halve_on_materialised_gather
            and _halvable_axis(
                axes=materialized.requirements.axes,
                fixed_widths=execution_widths.widths_for(regime_name=regime_name),
            )
            is not None
        ):
            frontiers[triple] = _CoreFrontier(
                program=materialized,
                transfer_plan=transfer_plan,
                templates=templates,
                widths=width_candidates,
                consumed=consumed,
                prebound=tuple(eager) if consumed else None,
            )
        resolved = eager[0]
        layouts[triple] = resolve_output_layout(
            core_key=core_key,
            value_template=next_regime_to_V_arr[regime_name],
            state_order=state_order,
            output_roles=resolved.output_roles,
        )
        candidate = (triple, _width_key(widths=resolved.tile_widths))
        resolved_programs[candidate] = resolved
        internal_templates[candidate] = templates
        candidates_by_triple[triple] = [candidate]
        frontier_lengths[triple] = len(width_candidates)
        if consumed:
            producers[core_key] = _checked_producer_records(
                candidates=eager, templates=templates
            )
    transfer_consumers = _transfer_consumer_counts(resolved_programs=resolved_programs)
    resolved_programs.update(
        _apply_transfer_marks(programs=resolved_programs, consumers=transfer_consumers)
    )
    # The first candidate of a triple is its representative: the width frontier
    # is listed widest first, and neither the declared reads nor the transfer
    # plan the ledger consults depends on the width.
    representatives: dict[_CoreTriple, ResolvedCoreProgram] = {}
    for candidate, resolved in resolved_programs.items():
        representatives.setdefault(candidate[0], resolved)
    # One derivation per program group, handed to both the liveness ledger and
    # the resident-bytes walk, so the two read the same declaration facts.
    representative_metadata = _execution_metadata(programs=representatives)
    input_liveness = _build_planned_input_liveness(
        regimes=regimes,
        program_metadata=representative_metadata,
        retain_all_artifacts=retain_all_artifacts,
        persistable_artifact_refs=persistable_artifact_refs,
    )
    n_periods = _model_n_periods(regimes=regimes)
    nominations, readers_by_dispatch, donations = _resolve_candidate_donations(
        resolved_programs=resolved_programs,
        representatives=representatives,
        input_liveness=input_liveness,
        n_periods=n_periods,
        enable_jit=enable_jit,
        donate_buffers=donate_buffers,
    )
    argument_keys: dict[_CoreTriple, Hashable] = {}
    lowering_keys = _lowering_keys(
        resolved_programs=resolved_programs,
        internal_templates=internal_templates,
        layouts=layouts,
        donations=donations,
        regimes=regimes,
        program_fingerprint=program_fingerprint,
        argument_keys=argument_keys,
    )
    frontier = _LazyCandidateFrontier(
        frontiers=frontiers,
        candidates_by_triple=candidates_by_triple,
        frontier_lengths=MappingProxyType(frontier_lengths),
        layouts=layouts,
        resolved_programs=resolved_programs,
        internal_templates=internal_templates,
        nominations=nominations,
        donations=donations,
        lowering_keys=lowering_keys,
        argument_keys=argument_keys,
        transfer_consumers=transfer_consumers,
        readers_by_dispatch=readers_by_dispatch,
        input_liveness=input_liveness,
        regimes=regimes,
        program_fingerprint=program_fingerprint,
        n_periods=n_periods,
        enable_jit=enable_jit,
        donate_buffers=donate_buffers,
    )
    return (
        layouts,
        lowering_keys,
        resolved_programs,
        internal_templates,
        input_liveness,
        donations,
        representative_metadata,
        frontier,
    )


def _lowering_keys(
    *,
    resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    internal_templates: Mapping[_CoreCandidate, Mapping[str, object]],
    layouts: Mapping[_CoreTriple, ResolvedOutputLayout],
    donations: Mapping[_CoreCandidate, tuple[ResolvedDonation, ...]],
    regimes: MappingProxyType[RegimeName, Regime],
    program_fingerprint: str,
    argument_keys: dict[_CoreTriple, Hashable] | None = None,
) -> dict[_CoreCandidate, Hashable]:
    """Compute every candidate's lowering key, donation set included.

    Width candidates of one core share the exact dynamic arguments assembled before
    their static widths are resolved. Describe that argument tree once per frontier;
    specialization, donation, placement, layout, and compiler choices remain in each
    candidate's key.

    The frontier is bound one candidate at a time, so the caller may hand over the
    cache that description lives in. A width candidate bound after a refusal then
    reuses the description its core already published instead of assembling a
    second one.
    """
    keys: dict[_CoreCandidate, Hashable] = {}
    if argument_keys is None:
        argument_keys = {}
    for candidate, resolved in resolved_programs.items():
        triple = candidate[0]
        regime_name, period, core_key = triple
        regime = regimes[regime_name]
        if triple not in argument_keys:
            argument_keys[triple] = _abstract_arguments_key(
                arguments={**resolved.arguments, **internal_templates[candidate]}
            )
        keys[candidate] = (
            _program_identity(
                program_fingerprint=program_fingerprint,
                regime_name=regime_name,
                core_name=core_key,
                period_signature=regime.solution.period_signatures[period],
                solver_group_key=regime.solution.solver_period_group_keys.get(period),
            ),
            argument_keys[triple],
            resolved.specialization_key,
            _output_roles_key(output_roles=resolved.output_roles),
            layouts[triple].compilation_key,
            _donated_arguments(donations=donations[candidate]),
            regime.solution.submesh_device_ids,
            resolved.compiler_options,
            _trace_settings_key(),
        )
    return keys


def _fail_if_a_unit_donates_one_artifact_twice(
    *, donations: Mapping[_CoreCandidate, tuple[ResolvedDonation, ...]]
) -> None:
    """Refuse a plan in which two programs of one unit donate one artifact.

    A donated buffer is handed to its executable once, so the second program
    naming it would receive what the first gave away. The donation set is
    plan-time data, so the refusal names the artifact before any candidate is
    lowered; the dispatch loop asks the same question again over the programs
    it is about to call.

    Width candidates of one core are alternatives, not two donors, so the
    programs of a unit are counted by core name.

    Args:
        donations: Mapping of width candidate to that candidate's resolved
            donations.

    Raises:
        ExecutionPlanningError: Two cores of one regime and period donate one
            artifact.

    """
    donors: dict[tuple[RegimeName, int, ValueArtifactAddress], set[str]] = {}
    for (triple, _widths), resolved in donations.items():
        regime_name, period, core_key = triple
        for donation in resolved:
            if not donation.donated:
                continue
            for artifact in donation.artifacts:
                donors.setdefault((regime_name, period, artifact), set()).add(core_key)
    for (regime_name, period, artifact), cores in donors.items():
        if len(cores) > 1:
            msg = (
                f"Dispatch {(period, regime_name)!r} lowers programs "
                f"{sorted(cores)!r} to donate {artifact!r}; one buffer is "
                "handed over once."
            )
            raise ExecutionPlanningError(msg)


def _donated_arguments(*, donations: tuple[ResolvedDonation, ...]) -> tuple[str, ...]:
    """Return the argument names one candidate's executable donates."""
    return tuple(donation.argument for donation in donations if donation.donated)


def _mark_reused_transfers(
    *, resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram]
) -> dict[_CoreCandidate, ResolvedCoreProgram]:
    """Mark every transfer whose result more than one source core of a period reads.

    Two source cores of one period that read one stored artifact into one required
    layout need one copy between them, so the plan says so and the scheduler makes
    it once. Consumers are counted by core triple, not by width candidate: two
    widths of one core are alternatives, not two readers. The mark is a scheduling
    fact and stays out of the specialization key, so every lowering key is the key
    the unmarked program had. A program whose transfers already carry the marks the
    count implies is returned unchanged, so the single-source case rebuilds nothing.
    """
    return _apply_transfer_marks(
        programs=resolved_programs,
        consumers=_transfer_consumer_counts(resolved_programs=resolved_programs),
    )


def _transfer_consumer_counts(
    *, resolved_programs: Mapping[_CoreCandidate, ResolvedCoreProgram]
) -> dict[_ConsumerKey, set[_CoreTriple]]:
    """Count, per shareable transfer result, the source cores of a period that read it.

    A core's resolved input transfer plan is the same at every width — the plan is
    resolved once for the materialized program, before any width is bound — and the
    count ranges over triples, so one candidate per triple names the whole census.
    """
    consumers: dict[_ConsumerKey, set[_CoreTriple]] = {}
    for (triple, _widths), resolved in resolved_programs.items():
        for transfer in resolved.input_transfer_plan:
            key = _consumer_key(triple=triple, transfer=transfer)
            consumers.setdefault(key, set()).add(triple)
    return consumers


def _apply_transfer_marks(
    *,
    programs: Mapping[_CoreCandidate, ResolvedCoreProgram],
    consumers: Mapping[_ConsumerKey, set[_CoreTriple]],
) -> dict[_CoreCandidate, ResolvedCoreProgram]:
    """Write a census of shared transfer results onto each candidate's plan."""
    marked: dict[_CoreCandidate, ResolvedCoreProgram] = {}
    for candidate, resolved in programs.items():
        triple = candidate[0]
        plan: list[ResolvedValueTransfer] = []
        rebuilt = False
        for transfer in resolved.input_transfer_plan:
            key = _consumer_key(triple=triple, transfer=transfer)
            reused = len(consumers[key]) > 1
            if reused == transfer.reused_by_several_consumers:
                plan.append(transfer)
                continue
            plan.append(
                dataclasses.replace(transfer, reused_by_several_consumers=reused)
            )
            rebuilt = True
        marked[candidate] = (
            dataclasses.replace(resolved, input_transfer_plan=tuple(plan))
            if rebuilt
            else resolved
        )
    return marked


def _consumer_key(
    *, triple: _CoreTriple, transfer: ResolvedValueTransfer
) -> _ConsumerKey:
    """Name the one transfer result a period's source cores can share.

    A stored artifact read into one required layout is one copy, whichever core
    of the period asks for it. The source regime and core are deliberately absent:
    they are what the count ranges over.
    """
    return (triple[1], transfer.target, transfer.source_sharding)


def _consumed_producer_names(
    *,
    all_programs: Mapping[_CoreTriple, CoreProgram],
    regime_name: RegimeName,
    period: int,
) -> frozenset[str]:
    """Return the programs of one regime-period graph whose outputs are consumed."""
    return consumed_producer_names(
        graph={
            core_key: program
            for (other_regime, other_period, core_key), program in all_programs.items()
            if (other_regime, other_period) == (regime_name, period)
        }
    )


def _programs_in_producer_order(
    *, all_programs: Mapping[_CoreTriple, CoreProgram]
) -> dict[_CoreTriple, CoreProgram]:
    """Reorder the flat program map so producers precede their consumers.

    Ordering is per regime-period graph, since internal outputs never cross one.
    Declaration order breaks ties, so a graph without internal edges keeps the
    order its kernel published.
    """
    cells: dict[tuple[RegimeName, int], dict[str, CoreProgram]] = {}
    for (regime_name, period, core_key), program in all_programs.items():
        cells.setdefault((regime_name, period), {})[core_key] = program
    ordered: dict[_CoreTriple, CoreProgram] = {}
    for (regime_name, period), graph in cells.items():
        for core_key in topological_program_order(graph=graph):
            ordered[(regime_name, period, core_key)] = graph[core_key]
    return ordered


def _width_key(*, widths: Mapping[str, int]) -> _WidthKey:
    """Freeze a width map while preserving axis declaration order."""
    return tuple(widths.items())


def _continuous_value_replica_required(
    *, regime: Regime, state_name: str | None
) -> bool:
    """Read the construction-validated capability, never infer it from a grid."""
    if state_name is None:
        return False
    if state_name not in regime.solution.sharded_state_names:
        raise ExecutionPlanningError(
            "The validated continuous sharded state is missing from a source regime."
        )
    return True


def _resolve_program_for_execution(
    *,
    program: MaterializedCoreProgram,
    tile_widths: Mapping[str, int],
    source_value_template: object,
    source: _CoreTriple,
    require_full_next_value: bool = False,
    input_transfer_plan: tuple[ResolvedValueTransfer, ...] | None = None,
    abstract_inputs: bool = False,
) -> ResolvedCoreProgram:
    """Resolve the one program contract shared by eager, AOT, and replay."""
    if input_transfer_plan is None:
        input_transfer_plan = (
            _resolve_value_input_transfer_plan(
                program=program,
                source_value_template=source_value_template,
                source=source,
                require_full_next_value=require_full_next_value,
            )
            if program.disposition is CoreExecutionDisposition.PLANNED
            else ()
        )
    return resolve_core_program(
        program=program,
        tile_widths=tile_widths,
        input_transfer_plan=input_transfer_plan,
        abstract_inputs=abstract_inputs,
    )


def _prepare_abstract_program(
    *,
    program: MaterializedCoreProgram,
    source_value_template: FloatND,
    source: _CoreTriple,
    require_full_next_value: bool = False,
) -> tuple[MaterializedCoreProgram, tuple[ResolvedValueTransfer, ...]]:
    """Resolve one core's exact read destinations before enumerating widths."""
    transfers = (
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=source_value_template,
            source=source,
            require_full_next_value=require_full_next_value,
        )
        if program.disposition is CoreExecutionDisposition.PLANNED
        else ()
    )
    return (
        abstract_program_inputs(
            program=program,
            transfers=transfers,
            execution_sharding=source_value_template.sharding,
        ),
        transfers,
    )


def _resolve_value_input_transfer_plan(
    *,
    program: MaterializedCoreProgram,
    source_value_template: object,
    source: _CoreTriple,
    require_full_next_value: bool = False,
) -> tuple[ResolvedValueTransfer, ...]:
    """Resolve every declared value read against its source core's placement.

    Absolute artifact and consumer addresses remain on each transfer for dispatch and
    liveness. The specialization key omits absolute periods and source-node
    coordinates, while retaining the argument-tree path, so equivalent period nodes
    can still share a compiled executable without conflating different tree roles.
    """
    source_execution_sharding = getattr(source_value_template, "sharding", None)
    if not isinstance(source_execution_sharding, jax.sharding.Sharding):
        msg = "A source core's value template must expose a concrete JAX sharding."
        raise TypeError(msg)

    result: list[ResolvedValueTransfer] = []
    for read in program.requirements.value_reads:
        declared_source = (
            read.source.source_regime,
            read.source.source_period,
            read.source.core_key,
        )
        if declared_source != source:
            msg = (
                "A value read's source must match the actual compiled core: "
                f"declared={declared_source!r}, actual={source!r}."
            )
            raise ValueError(msg)
        stored_template = _value_read_argument_leaf(program=program, read=read)
        stored_sharding = getattr(stored_template, "sharding", None)
        kind, source_sharding = _resolve_value_transfer_layout(
            stored_sharding=stored_sharding,
            source_execution_sharding=source_execution_sharding,
            require_full_replica=(
                require_full_next_value
                and read.source.channel is ValueInputChannel.NEXT_REGIME_VALUE
            ),
            target_regime=read.target.regime,
            source_regime=read.source.source_regime,
        )
        result.append(
            resolve_value_transfer(
                target=read.target,
                source=read.source,
                kind=kind,
                stored_template=stored_template,
                source_sharding=source_sharding,
            )
        )
    return tuple(result)


def _resolve_value_transfer_layout(
    *,
    stored_sharding: object,
    source_execution_sharding: jax.sharding.Sharding,
    require_full_replica: bool = False,
    target_regime: RegimeName | None = None,
    source_regime: RegimeName | None = None,
) -> tuple[ValueTransferKind, jax.sharding.Sharding]:
    """Choose the required value layout and name the operator that reaches it.

    Args:
        stored_sharding: Concrete layout the target regime's value is stored on.
        source_execution_sharding: Concrete layout the reading core runs on.
        require_full_replica: Whether the read consumes the complete target line
            and so needs a replica of it on the source mesh.
        target_regime: Regime owning the stored value, named in a refusal.
        source_regime: Regime whose core reads it, named in a refusal.

    Returns:
        Tuple of the operator and the layout the read is delivered on.

    Raises:
        ExecutionPlanningError: No single operator serves the two layouts.

    """
    if not isinstance(stored_sharding, jax.sharding.Sharding):
        msg = "A stored target value must expose a concrete JAX sharding."
        raise TypeError(msg)

    if require_full_replica:
        # Continuous interpolation consumes the complete target line, including
        # remote shard crossings. Declare the replica before lowering so transfer
        # classification, liveness and budget admission own it outside the core.
        source_sharding = (
            jax.NamedSharding(
                mesh=source_execution_sharding.mesh,
                spec=jax.P(),
                memory_kind=source_execution_sharding.memory_kind,
            )
            if isinstance(source_execution_sharding, jax.NamedSharding)
            else source_execution_sharding
        )
    elif stored_sharding == source_execution_sharding or (
        isinstance(stored_sharding, jax.NamedSharding)
        and isinstance(source_execution_sharding, jax.NamedSharding)
        and stored_sharding.mesh == source_execution_sharding.mesh
    ):
        # A value already resident on the source mesh remains in its stored
        # representation. Its rank-specific partition spec need not equal the source
        # core's own output spec.
        source_sharding = stored_sharding
    elif isinstance(source_execution_sharding, jax.NamedSharding):
        # A value stored anywhere but the source core's own mesh — on one device,
        # or on the mesh of a regime that retains a different sharded state — is
        # moved onto that mesh as a replicated input. The source core's output
        # spec names that core's own axes and rank, neither of which a value read
        # from elsewhere shares, so reusing it would give the value the wrong axis
        # interpretation or no representable one at all.
        #
        # A partitioned delivery would be cheaper, and is refused rather than
        # chosen, even where the reading mesh happens to carry an axis of the
        # stored value's own name and extent. Splitting the value over that axis
        # is valid only for a read each device can answer from its own slice, and
        # a value read is not such a read: a state transition puts probability on
        # every category of the axis and interpolation spans the whole continuous
        # line, so each device consumes the complete value. Partitioning it would
        # move the missing pieces inside compiled work, as a collective the plan
        # does not name and admission never charged.
        source_sharding = jax.NamedSharding(
            mesh=source_execution_sharding.mesh,
            spec=jax.P(),
            memory_kind=source_execution_sharding.memory_kind,
        )
    else:
        # A source core running on one device collects the whole value there.
        source_sharding = source_execution_sharding

    try:
        kind = classify_value_transfer(
            stored_sharding=stored_sharding,
            required_sharding=source_sharding,
        )
    except ExecutionPlanningError as refusal:
        raise _unsupported_value_route(
            target_regime=target_regime,
            source_regime=source_regime,
            stored_sharding=stored_sharding,
            source_sharding=source_sharding,
            reason=str(refusal),
        ) from refusal
    return kind, source_sharding


def _unsupported_value_route(
    *,
    target_regime: RegimeName | None,
    source_regime: RegimeName | None,
    stored_sharding: jax.sharding.Sharding,
    source_sharding: jax.sharding.Sharding,
    reason: str,
) -> ExecutionPlanningError:
    """Name both regimes and their device axes on a route no operator serves."""
    target = (
        "the stored value" if target_regime is None else f"regime {target_regime!r}"
    )
    source = (
        "the reading core" if source_regime is None else f"regime {source_regime!r}"
    )
    return ExecutionPlanningError(
        f"{source} cannot read the value of {target}: "
        f"{target} is {_device_axis_description(sharding=stored_sharding)}, "
        f"{source} is {_device_axis_description(sharding=source_sharding)}, and "
        f"no planned transfer serves that pair. {reason}"
    )


def _device_axis_description(*, sharding: jax.sharding.Sharding) -> str:
    """Describe one endpoint by the device axes it holds and the devices it spans."""
    device_ids = sorted(device.id for device in sharding.device_set)
    if isinstance(sharding, jax.NamedSharding):
        axes = ", ".join(
            f"{name}={size}"
            for name, size in zip(
                sharding.mesh.axis_names, sharding.mesh.devices.shape, strict=True
            )
        )
        return f"sharded over ({axes}) on devices {device_ids}"
    return f"held on devices {device_ids} with no named axis"


def _program_identity(
    *,
    program_fingerprint: str,
    regime_name: RegimeName,
    core_name: str,
    period_signature: Hashable,
    solver_group_key: Hashable,
) -> Hashable:
    """Return what a program computes, independent of which object computes it.

    Five components, each durable across model constructions:

    - `program_fingerprint` — the model the program belongs to, digested
      without solve-time parameter values, since those are traced inputs;
    - `regime_name` and `core_name` — which named core of which regime it is;
    - `period_signature` — the engine's groupings of the period it serves
      (`SolutionPhase.period_signatures`);
    - `solver_group_key` — the solver's own grouping of that period
      (`SolutionPhase.solver_period_group_keys`), `None` where the solver does
      not group.

    The engine's groupings and the solver's are independent: neither implies
    the other, so both belong here.

    Args:
        program_fingerprint: Parameter-free digest of the model being solved.
        regime_name: Name of the regime whose core this is.
        core_name: The core's name within that regime's period graph.
        period_signature: The engine's signature for the core's period.
        solver_group_key: The solver's group key for the core's period.

    Returns:
        The hashable identity.

    """
    return (
        "program",
        program_fingerprint,
        regime_name,
        core_name,
        period_signature,
        solver_group_key,
    )


def _lowering_key(
    *,
    program_identity: Hashable,
    layout_key: Hashable,
    arguments: Mapping[str, object] | None = None,
    specialization_key: Hashable | None = None,
    output_roles: object | None = None,
    donated_arguments: tuple[str, ...] = (),
    placement_key: Hashable | None = None,
    compiler_options: tuple[tuple[str, int], ...] = (),
) -> Hashable:
    """Identify a program's tree, specialization, layout, donations and devices.

    The trace settings that change what a traced program computes are part of the
    identity, so a program traced under different settings is a different
    executable.
    """
    return (
        program_identity,
        (None if arguments is None else _abstract_arguments_key(arguments=arguments)),
        specialization_key,
        _output_roles_key(output_roles=output_roles),
        layout_key,
        donated_arguments,
        placement_key,
        compiler_options,
        _trace_settings_key(),
    )


def _trace_settings_key() -> Hashable:
    """Return JAX's effective trace context, which a traced program depends on.

    It is the context JAX keys its own trace caches on. It changes with, among
    others:
    - `jax_enable_x64`, the default integer and float widths;
    - `jax_numpy_dtype_promotion`, which decides whether a mixed-type operation
      traces at all;
    - `jax_default_matmul_precision`, the contraction precision;
    - the ambient mesh set by `jax.set_mesh`.

    The context holds JAX's interned axis environment as an object whose repr
    is its address, so it is spelled by its contents: a key then reads the same
    in every process that traces under the same context.
    """
    return (
        "trace_context",
        tuple(_spelled_trace_value(value) for value in jax_config.trace_context()),
    )


def _spelled_trace_value(value: object) -> object:
    """Replace an axis environment by the axis names and sizes it binds."""
    if not isinstance(value, jax_core.AxisEnv):
        return value
    return (
        "axis_env",
        tuple(sorted(value.axis_sizes.items(), key=repr)),
        tuple(sorted(value.spmd_axis_names, key=repr)),
        tuple(sorted(value.explicit_mesh_axis_names, key=repr)),
    )


def _abstract_arguments_key(
    *,
    arguments: Mapping[str, object],
) -> Hashable:
    """Describe dynamic kwargs by pytree and abstract leaf metadata."""
    return tuple(
        (name, _abstract_value_key(value=value)) for name, value in arguments.items()
    )


def _abstract_value_key(*, value: object) -> Hashable:
    """Describe one dynamic argument without retaining its concrete value."""
    tree = jax.tree.structure(value)
    leaves = jax.tree.leaves(value)
    return (
        _hashable_metadata(tree),
        tuple(_abstract_leaf_key(leaf=leaf) for leaf in leaves),
    )


def _abstract_leaf_key(*, leaf: object) -> Hashable:
    """Return the tracing-relevant metadata for one dynamic leaf."""
    raw_shape = getattr(leaf, "shape", None)
    shape = (
        None if raw_shape is None else tuple(int(dimension) for dimension in raw_shape)
    )
    return (
        jax.Array
        if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct))
        else type(leaf),
        shape,
        _hashable_metadata(getattr(leaf, "dtype", None)),
        getattr(leaf, "weak_type", None),
        _hashable_metadata(getattr(leaf, "sharding", None)),
    )


def _hashable_metadata(value: object) -> Hashable:
    """Return metadata directly when hashable and a stable spelling otherwise."""
    try:
        hash(value)
    except TypeError:
        return (type(value), repr(value))
    return cast("Hashable", value)


def _output_roles_key(*, output_roles: object | None) -> Hashable:
    """Encode a declared logical output tree in the lowering identity."""
    if output_roles is None:
        return None
    return (
        _hashable_metadata(jax.tree.structure(output_roles)),
        tuple(_hashable_metadata(leaf) for leaf in jax.tree.leaves(output_roles)),
    )


def _assert_lowered_output_roles(
    *,
    lowered: jax.stages.Lowered,
    output_roles: object,
    layout: ResolvedOutputLayout,
    label: str,
) -> None:
    """Reject lowered output that violates the declared role contract."""
    _assert_lowered_output_tree(
        output_roles=output_roles,
        output_info=lowered.out_info,
        label=label,
    )
    output_leaves = jax.tree.leaves(lowered.out_info)
    for output_info, expected in zip(
        output_leaves, layout.expected_leaves, strict=True
    ):
        _assert_lowered_output_leaf(
            output_info=output_info,
            label=label,
            expected=expected,
        )


def _assert_lowered_output_tree(
    *, output_roles: object, output_info: object, label: str
) -> None:
    """Require the lowered pytree to match the solver's declared role tree."""
    expected = jax.tree.structure(output_roles)
    actual = jax.tree.structure(output_info)
    if actual != expected:
        msg = (
            f"{label} lowered output tree {actual} does not match declared "
            f"output roles {expected}."
        )
        raise TypeError(msg)


def _assert_lowered_output_leaf(
    *,
    output_info: object,
    label: str,
    expected: ExpectedOutputLeaf,
) -> None:
    """Check one lowered leaf's declared shape and dtype, and its placement."""
    if expected.shape is not None:
        actual_shape = getattr(output_info, "shape", None)
        if actual_shape != expected.shape:
            msg = (
                f"{label} {expected.label} output shape mismatch: "
                f"expected {expected.shape}, got {actual_shape}."
            )
            raise TypeError(msg)
    if expected.dtype is not None:
        actual_dtype = getattr(output_info, "dtype", None)
        if actual_dtype != expected.dtype:
            msg = (
                f"{label} {expected.label} output dtype mismatch: "
                f"expected {expected.dtype}, got {actual_dtype}."
            )
            raise TypeError(msg)
    actual_sharding = getattr(output_info, "sharding", None)
    if actual_sharding != expected.sharding:
        msg = (
            f"{label} {expected.label} output sharding mismatch: "
            f"expected {expected.sharding}, got {actual_sharding}."
        )
        raise TypeError(msg)


def _attach_resolved_output_layout(
    *,
    compiled: Callable[..., object],
    layout: ResolvedOutputLayout,
    tile_widths: Mapping[str, int],
    input_transfer_plan: tuple[ResolvedValueTransfer, ...] = (),
    internal_input_templates: Mapping[str, object] = MappingProxyType({}),
    donated_arguments: tuple[str, ...] = (),
    name: str,
) -> PlannedCore:
    """Carry one node's resolved output and input plans to runtime dispatch."""
    return PlannedCore(
        compiled=compiled,
        layout=layout,
        tile_widths=tile_widths,
        input_transfer_plan=input_transfer_plan,
        internal_input_templates=internal_input_templates,
        donated_arguments=donated_arguments,
        name=name,
    )


def _group_cores_by_regime_period(
    cores_by_triple: Mapping[_CoreTriple, PlannedCore],
) -> dict[tuple[RegimeName, int], MappingProxyType[str, PlannedCore]]:
    """Group (regime, period, core_key) -> core into (regime, period) -> {key: core}.

    The solve loop dispatches each period adapter with its full per-key core map,
    so a multi-core kernel receives all its compiled cores while a single-core
    kernel receives `{"main": ...}`.
    """
    grouped: dict[tuple[RegimeName, int], dict[str, PlannedCore]] = {}
    for (regime_name, period, core_key), core in cores_by_triple.items():
        grouped.setdefault((regime_name, period), {})[core_key] = core
    return {key: MappingProxyType(cores) for key, cores in grouped.items()}


def _log_kernel_memory(
    *,
    compiled: jax.stages.Compiled,
    label: str,
    logger: logging.Logger,
    precomputed_peak_bytes: int | None = None,
) -> None:
    """Log XLA's compile-time memory analysis for one compiled kernel.

    Gated on the `LCM_LOG_KERNEL_MEMORY` env var (off by default, zero cost),
    independently of the solve `log_level`: the env var is the opt-in, so the
    `[mem]` lines are emitted at a level that always clears the logger's
    threshold — even at `log_level="off"`, where the debug NaN/Inf diagnostic
    (its own per-period full-V transient) would otherwise have to be enabled to
    see them, masking the real kernel peak.

    The temporary, argument, output, and raw peak counters describe compiler
    buffer accounting and are available without executing the kernel. Admission
    separately reserves the larger of the raw peak and argument + output - alias
    + temporary bytes. Neither diagnostic establishes a complete runtime memory
    bound. Pair with `XLA_FLAGS=--xla_dump_to=DIR` to inspect the compiler buffers.
    """
    if os.environ.get("LCM_LOG_KERNEL_MEMORY", "0") == "0":
        return
    level = max(logger.getEffectiveLevel(), logging.INFO)
    if precomputed_peak_bytes is not None:
        logger.log(
            level,
            "  [mem] %s: compiler peak=%.3f GiB (planner cache)",
            label,
            precomputed_peak_bytes / 1024**3,
        )
        return
    try:
        stats = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 - backend may not support analysis
        logger.log(level, "  [mem] %s: memory_analysis unavailable (%s)", label, exc)
        return
    if stats is None:
        logger.log(level, "  [mem] %s: memory_analysis returned None", label)
        return
    gib = 1024**3
    logger.log(
        level,
        "  [mem] %s: temp=%.3f GiB  args=%.3f GiB  output=%.3f GiB  peak=%.3f GiB",
        label,
        stats.temp_size_in_bytes / gib,
        stats.argument_size_in_bytes / gib,
        stats.output_size_in_bytes / gib,
        stats.peak_memory_in_bytes / gib,
    )


def _resolve_compilation_workers(*, max_compilation_workers: int | None) -> int:
    """Return the number of threads to use for parallel XLA compilation."""
    if max_compilation_workers is None:
        return os.cpu_count() or 1
    if max_compilation_workers < 1:
        msg = f"max_compilation_workers must be >= 1, got {max_compilation_workers}."
        raise ValueError(msg)
    return max_compilation_workers


def _func_dedup_key(*, func: Callable) -> Hashable:
    """Return a hashable deduplication key for a callable.

    For `functools.partial` objects wrapping shared JIT functions, deduplicate
    by the underlying function's identity together with the `id()` of every
    positional- and keyword-argument value. This is correct even when different
    partials bind different value objects — two partials share a compiled
    program only when every bound value is the same object.

    For plain callables, use object identity.

    """
    if isinstance(func, functools.partial):
        return (
            id(func.func),
            tuple((k, id(v)) for k, v in sorted(func.keywords.items())),
            tuple(id(value) for value in func.args),
        )
    return id(func)
