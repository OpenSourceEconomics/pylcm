import dataclasses
import functools
import gc
import inspect
import logging
import os
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType
from typing import cast

import jax

from _lcm.engine import Regime, StateActionSpace, _build_regime_sharding
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    ResolvedCoreProgram,
    _target_value_argument_leaf,
    core_program_graph,
    initial_core_tile_widths,
    materialize_core_program,
    resolve_core_program,
    select_programs,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import (
    UNPLANNED,
    ExpectedOutputLeaf,
    PlannedCore,
    ResolvedOutputLayout,
    assert_value_leaf_layout,
    planned_output_layout,
    resolve_output_layout,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueTransferKind,
    resolve_value_transfer,
)
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
from _lcm.solution.contract import (
    BackwardInductionResult,
    ContinuationPayload,
    GeneratedReplayAuthority,
    KernelResult,
    SimulationPolicy,
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
from _lcm.solution.kernel_output import normalize_kernel_output
from _lcm.solution.period_capture import (
    PeriodCaptureTarget,
    capture_kernel_inputs,
    resolve_capture_target,
)
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
    _RegimeVTopology,
)
from _lcm.typing import FlatParams, RegimeName
from _lcm.utils.logging import (
    format_duration,
    log_period_header,
    log_period_timing,
    raise_or_warn,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND

# Stands in for a period's flag mapping when the model retains no dissolution
# flags, so every period key is present with nothing behind it. One shared
# instance: it is immutable and carries no arrays.
_NO_DISSOLUTION_FLAGS: MappingProxyType[RegimeName, BoolND] = MappingProxyType({})


def solve(  # noqa: C901, PLR0912, PLR0915
    *,
    flat_params: FlatParams,
    ages: AgeGrid,
    regimes: MappingProxyType[RegimeName, Regime],
    logger: logging.Logger,
    enable_jit: bool,
    collect_simulation_policies: bool = False,
    simulation_policy_regimes: frozenset[RegimeName] | None = None,
    collect_solver_diagnostics: bool = False,
    track_artifact_publication: bool = False,
    max_compilation_workers: int | None = None,
    retain_dissolution_flags: bool = True,
    retain_replay: bool = True,
) -> BackwardInductionResult:
    """Solve a model by backward induction, whatever solver each regime declares.

    Args:
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout, and carries the runtime-validation
            policy. `log_level="debug"` stops backward induction at the first
            NaN period and raises; `"warning"` / `"progress"` let induction run
            to completion and log a warning, so `solve` returns a complete
            (NaN-bearing) solution; `"off"` skips the NaN check.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        collect_simulation_policies: Whether to retain and copy published off-grid
            policies to the host. The solve kernels may publish an internal policy
            alongside their other outputs, but a value-only solve drops it at the
            period boundary instead of retaining one device-sized artifact per period.
        simulation_policy_regimes: Optional canonical-regime allowlist for policy
            collection. ``None`` permits every publisher.
        collect_solver_diagnostics: Whether to retain a kernel's numerical
            self-report. Public ``Model.solve()`` and automatic simulation request
            it; ``log_level`` still decides whether diagnostics are calculated and
            retained. Internal callers may disable collection.
        track_artifact_publication: Whether to retain the tiny set of cells whose
            kernels produced a simulation policy. Used to write truthful omission
            records without retaining values-only replay arrays.
        max_compilation_workers: Maximum number of threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.
        retain_dissolution_flags: Whether a caller wants the per-period
            dissolution flags on the result for their own sake. A model whose
            gates read `D_target` retains them regardless — the flags are a
            simulate-side input there, not an inspection artifact.
        retain_replay: Whether replay artifacts are retained. Selects, per
            period kernel, the scoped programs that are dispatched: a kernel may
            publish a values-only and a replay variant of one body.

    Returns:
        The named backward-induction outputs: the immutable mapping of periods
        to regime value-function arrays, the immutable mapping of periods to each
        regime's published simulation policy (the off-grid policy artifact
        simulation can interpolate; regimes whose kernels publish none have no
        entry, and the whole mapping is empty when
        `collect_simulation_policies` is false), and the immutable mapping of
        periods to each COLLECTIVE regime's dissolution flag `D` — `True` on the
        state cells whose action mask is empty, distinct from a numeric `-inf`
        value; empty inner mappings for models without collective regimes, so
        the default path only gains an empty dissolution mapping.

    """
    capture_target = resolve_capture_target()

    # The state-action spaces and the fence that reads them depend only on
    # `regimes` and `flat_params`, so a colliding model is rejected before a
    # single kernel is compiled rather than after every regime-period has been
    # AOT-compiled.
    base_state_action_spaces = _build_base_state_action_spaces(
        regimes=regimes, flat_params=flat_params
    )
    _reject_edge_fold_state_param_collisions(
        regimes=regimes,
        base_state_action_spaces=base_state_action_spaces,
        flat_params=flat_params,
    )

    next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr = (
        _build_continuation_templates(regimes=regimes, flat_params=flat_params)
    )

    # Resolve every solve program, then compile unique lowerings when enabled.
    compiled_programs = _compile_all_functions(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        next_edge_to_V_arr=next_edge_to_V_arr,
        enable_jit=enable_jit,
        retain_replay=retain_replay,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
    )
    compiled_functions = compiled_programs.executables
    input_liveness = _build_planned_input_liveness(
        regimes=regimes, program_metadata=compiled_programs.metadata
    )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    simulation_policies: dict[int, MappingProxyType[RegimeName, SimulationPolicy]] = {}
    generated_replay_authorities: dict[
        int, MappingProxyType[RegimeName, GeneratedReplayAuthority]
    ] = {}
    dissolution_flags: dict[int, MappingProxyType[RegimeName, BoolND]] = {}
    solver_diagnostics: dict[int, MappingProxyType[RegimeName, SolverDiagnostics]] = {}
    published_simulation_policy_cells: set[tuple[int, RegimeName]] = set()

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
        if collect_simulation_policies
        or (collect_solver_diagnostics and diagnostics_enabled)
        else None
    )

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

        # Regimes declaring `same_period_refs` read
        # other regimes' V of THIS period, so those references must be solved
        # first — order the period's active regimes topologically by the
        # reference edges (stable: dict order among independent regimes).
        # Models without references keep the plain dict order.
        for regime_name in _order_regime_names_by_same_period_refs(
            active_regimes=active_regimes
        ):
            regime = active_regimes[regime_name]
            result = _run_period_kernel(
                regime=regime,
                regime_name=regime_name,
                period=period,
                compiled_cores=compiled_functions[(regime_name, period)],
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
                    regime=regime, retain_replay=retain_replay
                ),
            )
            input_liveness.commit_successful_dispatch(dispatch=(period, regime_name))
            V_arr = result.V_arr
            # The published V mapping is the calling convention for every
            # downstream consumer — the parents' cores and the AOT-lowered
            # simulate programs are both compiled against the per-regime V
            # topology — so a kernel output arriving with a different
            # sharding (the compiled program's output sharding is the
            # backend's choice) is placed back on the template's mesh here.
            V_arr = _publish_kernel_value(
                value=V_arr,
                template=next_regime_to_V_arr[regime_name],
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
            if result.simulation_policy is not None:
                if collect_simulation_policies and (
                    simulation_policy_regimes is None
                    or regime_name in simulation_policy_regimes
                ):
                    period_simulation_policies[regime_name] = result.simulation_policy
                if track_artifact_publication:
                    published_simulation_policy_cells.add((period, regime_name))
            if result.generated_replay_authority is not None:
                if result.simulation_policy is None:
                    msg = (
                        "A generated replay authority has no matching simulation "
                        f"policy at ({period}, {regime_name!r})."
                    )
                    raise TypeError(msg)
                if collect_simulation_policies and (
                    simulation_policy_regimes is None
                    or regime_name in simulation_policy_regimes
                ):
                    period_generated_replay_authorities[regime_name] = (
                        result.generated_replay_authority
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
        next_edge_to_V_arr = _roll_gated_edges(
            regimes=regimes,
            ages=ages,
            period=period,
            period_solution=period_solution,
            period_dissolution_flags=period_dissolution_flags,
            base_state_action_spaces=base_state_action_spaces,
            flat_params=flat_params,
            next_edge_to_V_arr=next_edge_to_V_arr,
        )
        next_regime_to_V_arr, next_regime_to_continuation = _roll_continuation_inputs(
            regimes=regimes,
            period_solution=period_solution,
            period_continuations=period_continuations,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
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
        if collect_simulation_policies:
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
                    for regime_name, diagnostics in period_solver_diagnostics.items()
                }
            )

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
            )
        except InvalidValueFunctionError as error:
            raise_or_warn(logger=logger, error=error)

    _drain_V_arr_shards(solution=solution, dissolution_flags=dissolution_flags)
    input_liveness.assert_solve_complete()

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return BackwardInductionResult(
        value_functions=MappingProxyType(solution),
        simulation_policies=MappingProxyType(simulation_policies),
        generated_replay_authorities=MappingProxyType(generated_replay_authorities),
        dissolution_flags=MappingProxyType(dissolution_flags),
        diagnostics=MappingProxyType(solver_diagnostics),
        published_simulation_policy_cells=frozenset(published_simulation_policy_cells),
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
    """Free the device buffers rolled off the period just solved.

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
    compiled_cores: MappingProxyType[str, Callable],
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
) -> KernelResult:
    """Invoke one regime's period adapter for one period.

    Every regime exposes the same kind of adapter; the loop never branches on
    solver type. The adapter wraps the regime's shared jitted core(s) (passed in
    AOT-compiled as `compiled_cores`), calls them with the solver's own argument
    layout, and returns either the public `KernelOutput` envelope or a legacy
    `KernelResult`. The fail-closed bridge below normalizes both to the latter —
    the value-function array plus the optional generic outputs (`continuation`,
    `simulation_policy`, and the collective `dissolution` flag D), which the
    backward-induction loop accumulates.

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
        The kernel's result for this regime-period.

    """
    period_kernel = regime.solution.period_kernels[period]

    # Captured before the period-specific state axes are substituted below. Replay
    # re-enters this funnel with capture explicitly disabled.
    capture_kernel_inputs(
        capture_target=capture_target,
        regime=regime,
        regime_name=regime_name,
        period=period,
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
    output = period_kernel(
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
    continuation_spec = regime.solution.continuation_spec
    return normalize_kernel_output(
        output=output,
        continuation_key=(
            None if continuation_spec is None else continuation_spec.artifact_key
        ),
        regime_name=regime_name,
        period=period,
    )


def _order_regime_names_by_same_period_refs(
    *,
    active_regimes: dict[RegimeName, Regime],
) -> tuple[RegimeName, ...]:
    """Topologically order one period's active regimes by `same_period_refs`.

    A regime reading another regime's same-period V
    must be solved after it. Stable Kahn ordering: at each step the first (in
    dict order) not-yet-placed regime whose active references are all placed is
    emitted, so models without references keep the plain dict order exactly. A
    cycle is rejected at model build (`_fail_if_same_period_ref_cycle`); the
    raise here is a defensive backstop for direct engine callers.
    """
    if not any(regime.same_period_ref_regimes for regime in active_regimes.values()):
        return tuple(active_regimes)
    placed: dict[RegimeName, None] = {}
    remaining = dict(active_regimes)
    while remaining:
        ready = next(
            (
                regime_name
                for regime_name, regime in remaining.items()
                if all(ref not in remaining for ref in regime.same_period_ref_regimes)
            ),
            None,
        )
        if ready is None:
            msg = (
                "same_period_refs form a cycle among the period's active "
                f"regimes: {sorted(remaining)}. This should have been "
                "rejected at model build."
            )
            raise RuntimeError(msg)
        placed[ready] = None
        del remaining[ready]
    return tuple(placed)


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
    for source_name, source in regimes.items():
        for target_name, edge in source.gated_edges.items():
            if not edge_may_fold_at_period(
                edge=edge,
                source_name=source_name,
                fold_period=period,
                solved_regimes=period_solution,
                source_reads_wbar=source_reads_folded_wbar(
                    source_active_periods=source.active_periods,
                    fold_period=period,
                ),
            ):
                continue
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
    fold_age: float | None,
    target_states: Mapping[str, ContinuousState | DiscreteState],
    same_period_mapping: Mapping[RegimeName, FloatND],
    source_flat_params: Mapping[str, object],
    reference_flat_params: Mapping[RegimeName, Mapping[str, object]],
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
            fold_age=fold_age,
        )
    )
    kwargs[SAME_PERIOD_V_ARG] = same_period_mapping
    if SAME_PERIOD_PARAMS_ARG in sig_params:
        kwargs[SAME_PERIOD_PARAMS_ARG] = reference_flat_params
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
    return jax.tree.map(
        lambda leaf, template_leaf: _match_leaf_template_sharding(
            leaf=leaf, template_leaf=template_leaf
        ),
        continuation,
        template,
    )


def _publish_kernel_value(
    *,
    value: FloatND,
    template: FloatND,
    compiled_cores: Mapping[str, Callable],
) -> FloatND:
    """Publish a period value in the engine-owned layout.

    An output-layout-aware core has already asserted its complete runtime
    output tree against the layout used to lower it at the compiled-core seam;
    here only the value leaf the loop publishes is checked again, since the
    kernel may have unpacked the rest of its tree into channels. The check reads
    the first planned core the period dispatched, whatever its name, since a
    retention-scoped graph compiles one program under one name and another under
    another. Legacy kernels retain the existing repair at this boundary.
    Continuation rolling deliberately keeps its independent repair: it is a
    different producer/consumer boundary.
    """
    for core in compiled_cores.values():
        layout = planned_output_layout(core)
        if layout is not UNPLANNED:
            assert_value_leaf_layout(value=value, layout=layout)
            return value
    return _repair_unplanned_kernel_value(value=value, template=template)


def _repair_unplanned_kernel_value(*, value: FloatND, template: FloatND) -> FloatND:
    """Place a legacy kernel output onto the published value template."""
    return _match_leaf_template_sharding(leaf=value, template_leaf=template)


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
    result: KernelResult,
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
                regimes=regimes, flat_params=flat_params
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
    """
    n_devices = len(jax.devices())
    target_shapes: dict[RegimeName, tuple[int, ...]] = {}
    target_shardings: dict[RegimeName, jax.NamedSharding | None] = {}
    for source_name, source in regimes.items():
        if not source.gated_edges:
            continue
        for target_name in source.gated_edges:
            if target_name not in target_shapes:
                target = regimes[target_name]
                target_states = target.solution.state_action_space(
                    regime_params=flat_params[target_name]
                ).states
                target_shapes[target_name] = tuple(
                    len(v) for v in target_states.values()
                )
                sharding_plan = _build_regime_sharding(
                    grids=target.solution.grids, n_devices=n_devices
                )
                target_shardings[target_name] = (
                    sharding_plan.V_arr_sharding(tuple(target_states))
                    if sharding_plan is not None
                    else None
                )
            shape = target_shapes[target_name]
            sharding = target_shardings[target_name]
            n_channels = source.gated_edges[target_name].channels.count
            if n_channels:
                shape = (*shape, n_channels)
                if sharding is not None:
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
) -> dict[RegimeName, StateActionSpace]:
    """Build each regime's params-completed state-action space once.

    The space is period-invariant within one solve (params are fixed), so
    runtime-grid completion (e.g. process gridpoint computation) runs once
    per regime instead of once per period-regime iteration.
    """
    return {
        regime_name: regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
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


type _InputDispatch = tuple[int, RegimeName]
type _CoreTriple = tuple[RegimeName, int, str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ProgramExecutionMetadata:
    """Resolved declaration facts retained without pinning argument templates."""

    requirements: CoreExecutionRequirements
    disposition: CoreExecutionDisposition
    input_transfer_plan: tuple[ResolvedValueTransfer, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CompiledPrograms:
    """Executable graph plus the metadata liveness reads through the same seam."""

    executables: dict[
        tuple[RegimeName, int], MappingProxyType[str, Callable[..., object]]
    ]
    metadata: MappingProxyType[_CoreTriple, _ProgramExecutionMetadata]


def _build_planned_input_liveness(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    program_metadata: Mapping[_CoreTriple, _ProgramExecutionMetadata],
) -> PlannedInputLiveness[_InputDispatch, ValueArtifactAddress]:
    """Build one exact-dispatch ledger without authorizing physical release."""
    dispatch_accesses: dict[_InputDispatch, tuple[ValueArtifactAddress, ...]] = {}
    pinned_artifacts = set(_retained_solution_value_artifacts(regimes=regimes))
    pinned_artifacts.update(_gated_fold_raw_value_artifacts(regimes=regimes))

    metadata_by_dispatch: dict[
        tuple[RegimeName, int], dict[str, _ProgramExecutionMetadata]
    ] = {}
    for (regime_name, period, core_name), metadata in program_metadata.items():
        metadata_by_dispatch.setdefault((regime_name, period), {})[core_name] = metadata

    for (regime_name, period), programs in metadata_by_dispatch.items():
        regime = regimes[regime_name]
        planned, unplanned_exact, has_unknown = _classify_dispatch_value_artifacts(
            programs=programs,
        )
        dispatch_accesses[(period, regime_name)] = planned
        pinned_artifacts.update(unplanned_exact)
        if has_unknown:
            pinned_artifacts.update(
                _conservative_legacy_value_artifacts(
                    regime=regime,
                    regime_name=regime_name,
                    period=period,
                )
            )

    return PlannedInputLiveness(
        dispatch_accesses=MappingProxyType(dispatch_accesses),
        pinned_artifacts=pinned_artifacts,
    )


def _classify_dispatch_value_artifacts(
    *,
    programs: Mapping[str, _ProgramExecutionMetadata],
) -> tuple[
    tuple[ValueArtifactAddress, ...],
    tuple[ValueArtifactAddress, ...],
    bool,
]:
    """Separate finite planned reads from pinned dense or unknown reads."""
    planned: list[ValueArtifactAddress] = []
    unplanned_exact: list[ValueArtifactAddress] = []
    has_unknown = False

    for core_name, metadata in programs.items():
        declared_targets = tuple(
            access.target for access in metadata.requirements.target_value_accesses
        )
        if metadata.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED:
            if not declared_targets:
                has_unknown = True
            else:
                unplanned_exact.extend(declared_targets)
            continue
        if metadata.disposition is CoreExecutionDisposition.DENSE:
            unplanned_exact.extend(declared_targets)
            continue

        plan = metadata.input_transfer_plan
        planned_targets = tuple(transfer.target for transfer in plan)
        if planned_targets != declared_targets:
            msg = (
                "A resolved input plan disagrees with its CoreProgram declaration for "
                f"core {core_name!r}: planned={planned_targets!r}, "
                f"declared={declared_targets!r}."
            )
            raise RuntimeError(msg)
        planned.extend(planned_targets)

    return (
        _unique_value_artifacts(planned),
        _unique_value_artifacts(unplanned_exact),
        has_unknown,
    )


def _unique_value_artifacts(
    artifacts: Iterable[ValueArtifactAddress],
) -> tuple[ValueArtifactAddress, ...]:
    """Deduplicate one dispatch in declaration order."""
    return tuple(dict.fromkeys(artifacts))


def _retained_solution_value_artifacts(
    *,
    regimes: Mapping[RegimeName, Regime],
) -> tuple[ValueArtifactAddress, ...]:
    """Pin every value retained in the public backward-induction result."""
    return tuple(
        ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE,
            period=period,
            regime=regime_name,
        )
        for regime_name, regime in regimes.items()
        for period in regime.active_periods
    )


def _gated_fold_raw_value_artifacts(
    *,
    regimes: Mapping[RegimeName, Regime],
) -> tuple[ValueArtifactAddress, ...]:
    """Pin raw same-period values read by the engine-owned gated-edge fold."""
    artifacts: list[ValueArtifactAddress] = []
    for source in regimes.values():
        for edge in source.gated_edges.values():
            readers = (edge.target, *edge.reference_regimes)
            for period in range(source.solution.reachability.n_periods):
                if not all(period in regimes[name].active_periods for name in readers):
                    continue
                artifacts.extend(
                    ValueArtifactAddress(
                        kind=ValueArtifactKind.REGIME_VALUE,
                        period=period,
                        regime=name,
                    )
                    for name in readers
                )
    return _unique_value_artifacts(artifacts)


def _conservative_legacy_value_artifacts(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
) -> tuple[ValueArtifactAddress, ...]:
    """Pin graph-declared values when a core has no complete input plan."""
    artifacts: list[ValueArtifactAddress] = [
        ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE,
            period=period,
            regime=reference,
        )
        for reference in regime.same_period_ref_regimes
    ]
    reachability = regime.solution.reachability
    if period == reachability.n_periods - 1:
        return _unique_value_artifacts(artifacts)

    for target in reachability.targets(period=period, source=regime_name):
        edge = regime.gated_edges.get(target)
        if edge is None:
            artifacts.append(
                ValueArtifactAddress(
                    kind=ValueArtifactKind.REGIME_VALUE,
                    period=period + 1,
                    regime=target,
                )
            )
            continue
        artifacts.append(
            ValueArtifactAddress(
                kind=ValueArtifactKind.GATED_CONTINUATION,
                period=period + 1,
                regime=regime_name,
                target_regime=target,
            )
        )
        artifacts.extend(
            ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=period + 1,
                regime=reference,
            )
            for reference in edge.reference_regimes
        )
    return _unique_value_artifacts(artifacts)


def _regime_retains_replay(*, regime: Regime, retain_replay: bool) -> bool:
    """Whether one regime's solve dispatches its replay-scoped programs.

    A simulation policy is consumed only through the regime's declared replay
    route. A regime without one (a standalone case-piece NB-EGM regime, whose
    simulation reads the grid argmax) dispatches its values-only programs under
    every retention, so a replay output is never assembled only to be discarded.
    Programs scoped `ANY` are unaffected.
    """
    return retain_replay and regime.simulation.egm_policy_read is not None


def _compile_all_functions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    enable_jit: bool,
    retain_replay: bool,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> _CompiledPrograms:
    """Resolve every solve program and optionally compile unique lowerings.

    Each regime exposes named cores through its period adapter. For every core, the
    engine first materializes the adapter's exact `CoreProgram`. The program supplies
    the planner-resolved callable, static choices, and output roles. The complete
    callable, abstract arguments, specialization, and
    output layout form the lowering key. Each unique program is lowered once
    (sequentially, because tracing is single-threaded), then the XLA programs compile
    in parallel via a thread pool. The loop stays free of solver-type forks.

    When JIT is disabled (`enable_jit=False`), executes the same resolved programs
    without the lowering and compilation steps.

    Args:
        regimes: The internal regimes containing the period adapters.
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
        retain_replay: Whether the solve retains replay artifacts; with the
            regime's declared replay route it selects which scoped programs of
            each kernel's graph are dispatched.
        max_compilation_workers: Maximum threads for parallel compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.

    Returns:
        Executable mappings by regime-period plus the resolved metadata used by
        input liveness. Eager entries call the resolved functions directly; AOT
        entries call compiled executables carrying the same plans.

    """
    # Collect the authoritative native graphs or their centralized legacy adapters.
    all_programs: dict[_CoreTriple, CoreProgram] = {}
    for regime_name, regime in regimes.items():
        regime_retains_replay = _regime_retains_replay(
            regime=regime, retain_replay=retain_replay
        )
        for period in regime.active_periods:
            graph = select_programs(
                graph=core_program_graph(kernel=regime.solution.period_kernels[period]),
                retain_replay=regime_retains_replay,
            )
            for core_name, program in graph.items():
                all_programs[(regime_name, period, core_name)] = program

    # Materialize each named core's exact program before representative selection.
    # The resulting function, arguments, roles, specialization, and layout form
    # one lowering source of truth.
    (
        all_layouts,
        lowering_keys,
        resolved_programs,
    ) = _resolve_output_layouts_and_lowering_keys(
        all_programs=all_programs,
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        next_edge_to_V_arr=next_edge_to_V_arr,
    )

    metadata = MappingProxyType(
        {
            triple: _ProgramExecutionMetadata(
                requirements=program.requirements,
                disposition=program.disposition,
                input_transfer_plan=program.input_transfer_plan,
            )
            for triple, program in resolved_programs.items()
        }
    )

    # Eager execution uses the same resolved function, arguments, requirements,
    # roles, static widths, and transfers as AOT. Only the final JAX compilation
    # step is omitted.
    if not enable_jit:
        eager = {
            triple: _attach_resolved_output_layout(
                compiled=(
                    functools.partial(program.function, **program.static_kwargs)
                    if program.static_kwargs
                    else program.function
                ),
                layout=all_layouts[triple],
                input_transfer_plan=program.input_transfer_plan,
            )
            for triple, program in resolved_programs.items()
        }
        return _CompiledPrograms(
            executables=_group_cores_by_regime_period(eager), metadata=metadata
        )

    # Keep one representative per lowering key so its adapter can build the
    # matching arguments.  Selection happens only after layout resolution.
    unique: dict[Hashable, tuple[Callable, RegimeName, int, str]] = {}
    for triple, program in resolved_programs.items():
        lowering_key = lowering_keys[triple]
        if lowering_key not in unique:
            regime_name, period, core_key = triple
            unique[lowering_key] = (
                program.function,
                regime_name,
                period,
                core_key,
            )

    n_triples_per_lowering = _count_triples_per_lowering_key(
        lowering_keys=lowering_keys
    )

    n_workers = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    n_unique = len(unique)

    logger.info(
        "AOT compilation: %d unique functions (%d regime-period-core triples, "
        "%d workers)",
        n_unique,
        len(all_programs),
        n_workers,
    )

    # Phase 1: Lower all unique cores (sequential — tracing is not thread-safe
    # and must happen on the main thread). Arguments were materialized before
    # representative selection and are reused verbatim here.
    lowered: dict[Hashable, jax.stages.Lowered] = {}
    labels: dict[Hashable, str] = {}
    for i, (lowering_key, (func, regime_name, period, core_key)) in enumerate(
        unique.items(), 1
    ):
        triple = (regime_name, period, core_key)
        resolved = resolved_programs[triple]
        lower_args = resolved.arguments
        static_kwargs = resolved.static_kwargs
        label = f"{regime_name} {core_key} (age {ages.values[period].item()})"
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
        jitted = (
            jax.jit(func, static_argnames=tuple(static_kwargs))
            if layout is UNPLANNED
            else jax.jit(
                func,
                static_argnames=tuple(static_kwargs),
                out_shardings=cast("ResolvedOutputLayout", layout).out_shardings,
            )
        )
        low = jitted.lower(**lower_args, **static_kwargs)
        _assert_lowered_output_roles(
            lowered=low,
            output_roles=resolved.output_roles,
            layout=layout,
            label=label,
        )
        lowered[lowering_key] = low
        elapsed = time.monotonic() - start
        logger.info("  lowered in %s", format_duration(seconds=elapsed))

    # Phase 2: Compile all lowered programs in parallel (XLA releases the GIL).
    compiled: dict[Hashable, jax.stages.Compiled] = {}

    def _compile_and_log(
        *,
        lowering_key: Hashable,
        low: jax.stages.Lowered,
        label: str,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        logger.info("  compiling %s ...", label)
        start = time.monotonic()
        result = low.compile()
        elapsed = time.monotonic() - start
        logger.info("  compiled  %s  %s", label, format_duration(seconds=elapsed))
        _log_kernel_memory(compiled=result, label=label, logger=logger)
        return lowering_key, result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _compile_and_log,
                lowering_key=lowering_key,
                low=low,
                label=labels[lowering_key],
            )
            for lowering_key, low in lowered.items()
        ]
        for future in as_completed(futures):
            lowering_key, comp = future.result()
            compiled[lowering_key] = comp

    # Map back to (regime, period) keys, grouping the compiled cores by core key.
    executables = _group_cores_by_regime_period(
        {
            triple: _attach_resolved_output_layout(
                compiled=compiled[lowering_keys[triple]],
                layout=all_layouts[triple],
                input_transfer_plan=resolved_programs[triple].input_transfer_plan,
            )
            for triple in all_programs
        }
    )
    return _CompiledPrograms(executables=executables, metadata=metadata)


def _count_triples_per_lowering_key(
    *,
    lowering_keys: Mapping[tuple[RegimeName, int, str], Hashable],
) -> dict[Hashable, int]:
    """Count the triples each callable-and-output-layout module will serve.

    A shared callable with distinct output layouts is deliberately counted as
    distinct lowered modules.
    """
    counts: dict[Hashable, int] = {}
    for key in lowering_keys.values():
        counts[key] = counts.get(key, 0) + 1
    return counts


def _resolve_output_layouts_and_lowering_keys(
    *,
    all_programs: Mapping[_CoreTriple, CoreProgram],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
) -> tuple[
    dict[_CoreTriple, ResolvedOutputLayout | object],
    dict[_CoreTriple, Hashable],
    dict[_CoreTriple, ResolvedCoreProgram],
]:
    """Materialize each core's complete, immutable lowering description."""
    layouts: dict[_CoreTriple, ResolvedOutputLayout | object] = {}
    lowering_keys: dict[_CoreTriple, Hashable] = {}
    resolved_programs: dict[_CoreTriple, ResolvedCoreProgram] = {}
    for (regime_name, period, core_key), declaration in all_programs.items():
        regime = regimes[regime_name]
        state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
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
        resolved = _resolve_program_for_execution(
            program=materialized,
            source_value_template=next_regime_to_V_arr[regime_name],
            source=(regime_name, period, core_key),
        )

        state_order = tuple(
            name
            for name in state_action_space.states
            if name not in regime.fold_state_names
        )
        layout = (
            UNPLANNED
            if resolved.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED
            else resolve_output_layout(
                core_key=core_key,
                value_template=next_regime_to_V_arr[regime_name],
                state_order=state_order,
                output_roles=resolved.output_roles,
            )
        )
        triple = (regime_name, period, core_key)
        layouts[triple] = layout
        resolved_programs[triple] = resolved
        layout_key = UNPLANNED if layout is UNPLANNED else layout.compilation_key
        lowering_keys[triple] = _lowering_key(
            func=resolved.function,
            layout_key=layout_key,
            arguments=resolved.arguments,
            specialization_key=resolved.specialization_key,
            output_roles=resolved.output_roles,
        )
    return layouts, lowering_keys, resolved_programs


def _resolve_program_for_execution(
    *,
    program: MaterializedCoreProgram,
    source_value_template: object,
    source: _CoreTriple,
) -> ResolvedCoreProgram:
    """Resolve the one program contract shared by eager, AOT, and replay."""
    input_transfer_plan = (
        _resolve_value_input_transfer_plan(
            program=program,
            source_value_template=source_value_template,
            source=source,
        )
        if program.disposition is CoreExecutionDisposition.PLANNED
        else ()
    )
    return resolve_core_program(
        program=program,
        tile_widths=initial_core_tile_widths(program=program),
        input_transfer_plan=input_transfer_plan,
    )


def _resolve_value_input_transfer_plan(
    *,
    program: MaterializedCoreProgram,
    source_value_template: object,
    source: _CoreTriple,
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
    for access in program.requirements.target_value_accesses:
        declared_source = (
            access.source.source_regime,
            access.source.source_period,
            access.source.core_key,
        )
        if declared_source != source:
            msg = (
                "A target-value access source must match the actual compiled core: "
                f"declared={declared_source!r}, actual={source!r}."
            )
            raise ValueError(msg)
        stored_template = _target_value_argument_leaf(program=program, access=access)
        stored_sharding = getattr(stored_template, "sharding", None)
        kind, source_sharding = _resolve_value_transfer_layout(
            stored_sharding=stored_sharding,
            source_execution_sharding=source_execution_sharding,
        )
        result.append(
            resolve_value_transfer(
                target=access.target,
                source=access.source,
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
) -> tuple[ValueTransferKind, jax.sharding.Sharding]:
    """Choose one of the two supported value-input representation adapters."""
    if not isinstance(stored_sharding, jax.sharding.Sharding):
        msg = "A stored target value must expose a concrete JAX sharding."
        raise TypeError(msg)

    if stored_sharding == source_execution_sharding or (
        isinstance(stored_sharding, jax.NamedSharding)
        and isinstance(source_execution_sharding, jax.NamedSharding)
        and stored_sharding.mesh == source_execution_sharding.mesh
    ):
        # A value already resident on the source mesh remains in its stored
        # representation. Its rank-specific partition spec need not equal the source
        # core's own output spec.
        return ValueTransferKind.ALIGNED_LOCAL, stored_sharding

    if isinstance(stored_sharding, jax.sharding.SingleDeviceSharding) and isinstance(
        source_execution_sharding, jax.NamedSharding
    ):
        # A partially distributed model moves the unsharded target onto the source
        # mesh as a replicated input. Reusing the source output's rank-specific spec
        # would give an unrelated target value the wrong axis interpretation.
        source_sharding = jax.NamedSharding(
            mesh=source_execution_sharding.mesh,
            spec=jax.P(),
            memory_kind=source_execution_sharding.memory_kind,
        )
        return ValueTransferKind.COPY_TO_SOURCE_LAYOUT, source_sharding

    # The reverse NamedSharding -> SingleDeviceSharding route is unreachable for a
    # valid value-consuming source: distributed states are model-level, and model
    # construction refuses to prune one from a nonterminal regime. Keep it
    # unsupported here so a broken construction invariant fails closed.
    msg = (
        "Unsupported target-value layout conversion: "
        f"{type(stored_sharding).__name__} -> "
        f"{type(source_execution_sharding).__name__}. "
        "Only values already aligned with the source execution placement and "
        "single-device values copied as replicated inputs onto a named source mesh "
        "are supported."
    )
    raise ValueError(msg)


def _lowering_key(
    *,
    func: Callable,
    layout_key: Hashable,
    arguments: Mapping[str, object] | None = None,
    specialization_key: Hashable | None = None,
    output_roles: object | None = None,
) -> Hashable:
    """Identify one callable, abstract input tree, specialization, and layout."""
    return (
        _func_dedup_key(func=func),
        (None if arguments is None else _abstract_arguments_key(arguments=arguments)),
        specialization_key,
        _output_roles_key(output_roles=output_roles),
        layout_key,
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
        type(leaf),
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
    output_roles: object | None,
    layout: ResolvedOutputLayout | object,
    label: str,
) -> None:
    """Reject lowered output that violates the declared role contract."""
    if output_roles is None:
        return
    if not isinstance(layout, ResolvedOutputLayout):
        msg = f"{label} declares output roles but has no resolved output layout."
        raise TypeError(msg)
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
    layout: ResolvedOutputLayout | object,
    input_transfer_plan: tuple[ResolvedValueTransfer, ...] = (),
) -> Callable:
    """Carry one node's resolved output and input plans to runtime dispatch."""
    if layout is UNPLANNED:
        if input_transfer_plan:
            msg = "An unplanned core cannot carry a resolved input transfer plan."
            raise ValueError(msg)
        return compiled
    return PlannedCore(
        compiled=compiled,
        layout=cast("ResolvedOutputLayout", layout),
        input_transfer_plan=input_transfer_plan,
    )


def _group_cores_by_regime_period(
    cores_by_triple: dict[tuple[RegimeName, int, str], Callable],
) -> dict[tuple[RegimeName, int], MappingProxyType[str, Callable]]:
    """Group (regime, period, core_key) -> core into (regime, period) -> {key: core}.

    The solve loop dispatches each period adapter with its full per-key core map,
    so a multi-core kernel receives all its compiled cores while a single-core
    kernel receives `{"main": ...}`.
    """
    grouped: dict[tuple[RegimeName, int], dict[str, Callable]] = {}
    for (regime_name, period, core_key), core in cores_by_triple.items():
        grouped.setdefault((regime_name, period), {})[core_key] = core
    return {key: MappingProxyType(cores) for key, cores in grouped.items()}


def _log_kernel_memory(
    *,
    compiled: jax.stages.Compiled,
    label: str,
    logger: logging.Logger,
) -> None:
    """Log XLA's compile-time memory analysis for one compiled kernel.

    Gated on the `LCM_LOG_KERNEL_MEMORY` env var (off by default, zero cost),
    independently of the solve `log_level`: the env var is the opt-in, so the
    `[mem]` lines are emitted at a level that always clears the logger's
    threshold — even at `log_level="off"`, where the debug NaN/Inf diagnostic
    (its own per-period full-V transient) would otherwise have to be enabled to
    see them, masking the real kernel peak.

    `temp_size_in_bytes` is the peak scratch buffer XLA plans for the kernel —
    the transient that binds the device at run time. Because it is computed at
    compile, it is available even for configs whose *execution* would OOM, so
    the egm_step working set can be sized (and swept against grid knobs) without
    running or exhausting the device. `argument`/`output` sizes bound the
    per-call resident inputs/outputs (the carry and V). Pair with
    `XLA_FLAGS=--xla_dump_to=DIR` to name the HLO op behind the peak buffer.
    """
    if os.environ.get("LCM_LOG_KERNEL_MEMORY", "0") == "0":
        return
    level = max(logger.getEffectiveLevel(), logging.INFO)
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
