import functools
import gc
import logging
import os
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import Regime, StateActionSpace, _build_regime_sharding
from _lcm.solution.validate_V import validate_V
from _lcm.typing import FlatParams, RegimeName, StateName
from _lcm.utils.logging import (
    format_duration,
    log_period_header,
    log_period_timing,
    raise_or_warn,
    v_array_has_inf,
    v_array_has_nan,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import BoolND, FloatND


def solve(
    *,
    flat_params: FlatParams,
    ages: AgeGrid,
    regimes: MappingProxyType[RegimeName, Regime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
) -> tuple[
    MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    MappingProxyType[int, MappingProxyType[RegimeName, EGMSimPolicy]],
]:
    """Solve a model using grid search.

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
        max_compilation_workers: Maximum number of threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.

    Returns:
        Tuple of (the immutable mapping of periods to regime value-function
        arrays, the immutable mapping of periods to each DC-EGM regime's
        published `EGMSimPolicy` — the off-grid consumption function simulation
        interpolates; empty for periods/regimes with no DC-EGM kernel).

    """
    next_regime_to_V_arr, next_regime_to_egm_carry = _build_continuation_templates(
        regimes=regimes, flat_params=flat_params
    )

    # AOT-compile all unique solve kernels in parallel.
    compiled_functions = _compile_all_functions(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_egm_carry=next_regime_to_egm_carry,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
    )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    sim_policies: dict[int, MappingProxyType[RegimeName, EGMSimPolicy]] = {}

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

    # backwards induction loop
    base_state_action_spaces = _build_base_state_action_spaces(
        regimes=regimes, flat_params=flat_params
    )

    # Simulation policies are a DC-EGM solve *output*, accumulated for every
    # period; no backward step reads them. Their `endog_grid` aliases the
    # period's carry buffer, so leaving them on device pins one carry-sized
    # buffer per period for the whole induction. Evict each period's policies
    # to host as they are produced; simulation re-materializes them on device.
    host_device = jax.devices("cpu")[0]

    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}
        period_egm_carries: dict[RegimeName, EGMCarry] = {}
        period_sim_policies: dict[RegimeName, EGMSimPolicy] = {}

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

        for regime_name, regime in active_regimes.items():
            V_arr = _solve_regime_period(
                regime=regime,
                regime_name=regime_name,
                period=period,
                solve_kernel=compiled_functions[(regime_name, period)],
                state_action_space=base_state_action_spaces[regime_name],
                flat_params=flat_params,
                ages=ages,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                period_egm_carries=period_egm_carries,
                period_sim_policies=period_sim_policies,
            )
            # Async reductions: gated on log level. `"off"` skips
            # everything — no kernel launches, no host syncs, no
            # NaN fail-fast. `"warning"` / `"progress"` folds two
            # cheap isnan/isinf reductions into the running scalars;
            # `"debug"` adds the min/max/mean trio. Each extra full-V
            # read is a memory-bandwidth tax on the larger models, so
            # the default keeps it to two reductions per (regime,
            # period).
            if diagnostics_enabled:
                if stats_enabled:
                    diagnostic_min.append(jnp.min(V_arr))
                    diagnostic_max.append(jnp.max(V_arr))
                    diagnostic_mean.append(jnp.mean(V_arr))
                running_any_nan = running_any_nan | v_array_has_nan(V_arr)
                running_any_inf = running_any_inf | v_array_has_inf(V_arr)
                diagnostic_rows.append(
                    _DiagnosticRow(
                        regime_name=regime_name,
                        period=period,
                        age=float(ages.values[period]),
                    )
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

        next_regime_to_V_arr, next_regime_to_egm_carry = _roll_continuation_inputs(
            regimes=regimes,
            period_solution=period_solution,
            period_egm_carries=period_egm_carries,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
        )
        solution[period] = MappingProxyType(period_solution)
        sim_policies[period] = MappingProxyType(
            {
                regime_name: jax.block_until_ready(
                    jax.device_put(sim_policy, host_device)
                )
                for regime_name, sim_policy in period_sim_policies.items()
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

        # Release the device buffers rolled off this period (the superseded
        # continuation V/carry and the period's transient working set) before
        # the next period's kernel allocates. They are unreferenced after the
        # roll, but their JAX arrays sit in registered pytrees that CPython's
        # cyclic collector frees only when it next runs — forcing a collection
        # here returns the device pool promptly, capping peak resident across
        # the loop (mirrors the forward-sim memory rework in `result.py`).
        gc.collect()

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

    _drain_V_arr_shards(solution=solution)

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return MappingProxyType(solution), MappingProxyType(sim_policies)


def _reachable_carry_subset(
    *,
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    reachable_targets: frozenset[RegimeName],
) -> MappingProxyType[RegimeName, EGMCarry]:
    """The carries a regime's EGM kernel actually reads.

    Each kernel only ever indexes `next_regime_to_egm_carry[target]` for its
    reachable targets, so the full all-regimes mapping is needlessly large.
    Filtering to the reachable subset keeps the kernel's carry pytree input
    minimal — only this subset is passed per call rather than every regime's
    carry at once.

    Iterates the source mapping's key order (stable across rolls) so the
    filtered pytree structure matches between lowering and call. Membership is
    tested defensively; reachable targets are always carry-producing.
    """
    return MappingProxyType(
        {
            name: next_regime_to_egm_carry[name]
            for name in next_regime_to_egm_carry
            if name in reachable_targets
        }
    )


def _init_diagnostic_accumulators() -> tuple[
    list[_DiagnosticRow],
    list[FloatND],
    list[FloatND],
    list[FloatND],
    BoolND,
    BoolND,
]:
    """Initialize the per-period async diagnostics accumulators.

    Returns the empty diagnostic-row, min, max, and mean lists, and the two
    running NaN/Inf flag scalars (folded into across the backward-induction
    loop). The two flags share the same immutable zero scalar initially; each
    is reassigned independently inside the loop.
    """
    zero: BoolND = jnp.zeros((), dtype=bool)
    rows: list[_DiagnosticRow] = []
    mins: list[FloatND] = []
    maxs: list[FloatND] = []
    means: list[FloatND] = []
    return rows, mins, maxs, means, zero, zero


def _solve_regime_period(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    solve_kernel: Callable,
    state_action_space: StateActionSpace,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    period_egm_carries: dict[RegimeName, EGMCarry],
    period_sim_policies: dict[RegimeName, EGMSimPolicy],
) -> FloatND:
    """Invoke one regime's solve kernel for one period.

    Dispatch on the regime's kernel kind:

    - DC-EGM kernel ⇒ Euler inversion on the savings grid (no action grids
      enter); returns the V array on the exogenous state grid plus the carry
      the regime's parents interpolate.
    - max-Q-over-a kernel ⇒ evaluate Q on states and actions and maximize
      over actions; a terminal regime with a carry producer additionally
      yields its closed-form carry.

    Produced carries are stored in `period_egm_carries` in place.

    `period`/`age` are passed as JAX arrays (not Python scalars) so a shared
    `jax.jit` function is traced once with abstract shapes, not recompiled
    for every distinct (period, age) pair.

    Returns:
        The regime's value-function array.

    """
    if regime.solution.egm_step is not None:
        V_arr, egm_carry, sim_policy = solve_kernel(
            **state_action_space.states,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=_reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=regime.solution.egm_reachable_targets,
            ),
            **_egm_kernel_params(
                regime=regime, regime_name=regime_name, flat_params=flat_params
            ),
            period=jnp.int32(period),
            age=ages.values[period],
        )
        period_egm_carries[regime_name] = egm_carry
        period_sim_policies[regime_name] = sim_policy
        return V_arr

    V_arr = solve_kernel(
        **state_action_space.states,
        **state_action_space.actions,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **flat_params[regime_name],
        period=jnp.int32(period),
        age=ages.values[period],
    )
    egm_carry_producer = regime.solution.egm_carry_producer
    if egm_carry_producer is not None:
        period_egm_carries[regime_name] = egm_carry_producer(
            V_arr=V_arr,
            **state_action_space.states,
            **flat_params[regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
        )
    return V_arr


def _roll_continuation_inputs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    period_solution: dict[RegimeName, FloatND],
    period_egm_carries: dict[RegimeName, EGMCarry],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
) -> tuple[
    MappingProxyType[RegimeName, FloatND], MappingProxyType[RegimeName, EGMCarry]
]:
    """Roll the per-period continuation mappings forward by one period.

    Both mappings keep their full template key sets — V for every regime,
    carries for every carry-producing regime — and update only the entries
    solved this period, so the pytree structure stays JIT-stable.

    Returns:
        Tuple of the rolled V mapping and the rolled carry mapping.

    """
    rolled_V_arr = MappingProxyType(
        {
            regime_name: period_solution.get(
                regime_name, next_regime_to_V_arr[regime_name]
            )
            for regime_name in regimes
        }
    )
    rolled_egm_carry = MappingProxyType(
        {
            regime_name: period_egm_carries.get(
                regime_name, next_regime_to_egm_carry[regime_name]
            )
            for regime_name in next_regime_to_egm_carry
        }
    )
    return rolled_V_arr, rolled_egm_carry


def _build_continuation_templates(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> tuple[
    MappingProxyType[RegimeName, FloatND], MappingProxyType[RegimeName, EGMCarry]
]:
    """Build the period-invariant continuation-input templates.

    Both mappings keep the same pytree structure (keys and shapes) across all
    periods, avoiding JIT re-compilation from pytree mismatches:

    - the V template holds a zero array per regime, shaped (and sharded) like
      the regime's V array;
    - the EGM-carry template holds entries only for carry-producing regimes
      (DC-EGM regimes and, in models with one, terminal regimes), in the key
      order reused every period.
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
    next_regime_to_egm_carry = MappingProxyType(
        {
            regime_name: regime.solution.egm_carry_template
            for regime_name, regime in regimes.items()
            if regime.solution.egm_carry_template is not None
        }
    )
    return next_regime_to_V_arr, next_regime_to_egm_carry


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
) -> None:
    """Block until every V_arr shard is materialised on its device.

    Solve → simulate barrier: backward induction returns sharded V_arrs,
    but the simulate phase must consume materialised arrays rather than
    in-flight kernels. `jax.block_until_ready` walks the pytree of V_arrs
    and blocks per-shard (no host transfer, no cross-device collective);
    free when kernels are already done, the minimum necessary sync when
    they are not. V stays sharded across devices.
    """
    jax.block_until_ready(solution)


def _compile_all_functions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    enable_jit: bool,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> dict[tuple[RegimeName, int], Callable]:
    """AOT-compile all unique solve kernels in parallel.

    With shared-JIT, many periods share the same `jax.jit`-wrapped function
    object. This function deduplicates by object identity, traces each unique
    function once (sequential), then compiles the XLA programs in parallel
    via a thread pool (XLA releases the GIL during compilation).

    A regime's kernel is its DC-EGM step when one is configured and its
    max-Q-over-a grid search otherwise; the two take different lowering
    arguments (the EGM step takes no action grids but the rolling carry
    template).

    When JIT is disabled (`enable_jit=False`), returns the raw functions
    without compilation.

    Args:
        regimes: The internal regimes containing solve functions.
        flat_params: Regime parameters for constructing lowering args.
        ages: Age grid for the model.
        next_regime_to_V_arr: Template with consistent keys and V array shapes
            for constructing lowering arguments.
        next_regime_to_egm_carry: Template with consistent keys and carry
            shapes for constructing EGM lowering arguments.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        max_compilation_workers: Maximum threads for parallel compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.

    Returns:
        Dict of (regime_name, period) to callable (compiled or raw) functions.

    """
    # Collect all (regime, period) -> function mappings.
    all_functions: dict[tuple[RegimeName, int], Callable] = {}
    egm_keys: set[tuple[RegimeName, int]] = set()
    for regime_name, regime in regimes.items():
        egm_step = regime.solution.egm_step
        for period in regime.active_periods:
            if egm_step is not None:
                all_functions[(regime_name, period)] = egm_step[period]
                egm_keys.add((regime_name, period))
            else:
                all_functions[(regime_name, period)] = regime.solution.max_Q_over_a[
                    period
                ]

    # If JIT is disabled, return raw functions directly.
    if not enable_jit:
        return all_functions

    # Deduplicate by identity (or by underlying function for partials).
    unique: dict[Hashable, tuple[Callable, RegimeName, int]] = {}
    for (regime_name, period), func in all_functions.items():
        func_id = _func_dedup_key(func=func)
        if func_id not in unique:
            unique[func_id] = (func, regime_name, period)

    n_workers = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    n_unique = len(unique)

    logger.info(
        "AOT compilation: %d unique functions (%d regime-period pairs, %d workers)",
        n_unique,
        len(all_functions),
        n_workers,
    )

    # Phase 1: Lower all unique functions (sequential — tracing is not
    # thread-safe and must happen on the main thread).
    lowered: dict[Hashable, jax.stages.Lowered] = {}
    labels: dict[Hashable, str] = {}
    for i, (func_id, (func, regime_name, period)) in enumerate(unique.items(), 1):
        lower_args = _build_lower_args(
            regime=regimes[regime_name],
            regime_name=regime_name,
            period=period,
            is_egm_kernel=(regime_name, period) in egm_keys,
            flat_params=flat_params,
            ages=ages,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
        )
        label = f"{regime_name} (age {ages.values[period].item()})"
        labels[func_id] = label
        logger.info("%d/%d  %s", i, n_unique, label)
        logger.info("  lowering ...")
        start = time.monotonic()
        lowered[func_id] = jax.jit(func).lower(**lower_args)
        elapsed = time.monotonic() - start
        logger.info("  lowered in %s", format_duration(seconds=elapsed))

    # Phase 2: Compile all lowered programs in parallel (XLA releases the GIL).
    compiled: dict[Hashable, jax.stages.Compiled] = {}

    def _compile_and_log(
        *,
        func_id: Hashable,
        low: jax.stages.Lowered,
        label: str,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        logger.info("  compiling %s ...", label)
        start = time.monotonic()
        result = low.compile()
        elapsed = time.monotonic() - start
        logger.info("  compiled  %s  %s", label, format_duration(seconds=elapsed))
        _log_kernel_memory(compiled=result, label=label, logger=logger)
        return func_id, result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _compile_and_log, func_id=func_id, low=low, label=labels[func_id]
            )
            for func_id, low in lowered.items()
        ]
        for future in as_completed(futures):
            func_id, comp = future.result()
            compiled[func_id] = comp

    # Map back to (regime, period) keys.
    return {
        key: compiled[_func_dedup_key(func=func)] for key, func in all_functions.items()
    }


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


def _build_lower_args(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    is_egm_kernel: bool,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
) -> dict[str, object]:
    """Build the lowering arguments for one solve kernel.

    EGM kernels take no action grids but the rolling carry template;
    max-Q-over-a kernels take the full state-action product.
    """
    state_action_space = regime.solution.state_action_space(
        regime_params=flat_params[regime_name],
    )
    if is_egm_kernel:
        return {
            **dict(state_action_space.states),
            "next_regime_to_egm_carry": _reachable_carry_subset(
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                reachable_targets=regime.solution.egm_reachable_targets,
            ),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **_egm_kernel_params(
                regime=regime, regime_name=regime_name, flat_params=flat_params
            ),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
    common = {
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **dict(flat_params[regime_name]),
        "period": jnp.int32(period),
        "age": ages.values[period],
    }
    return {
        **dict(state_action_space.states),
        **dict(state_action_space.actions),
        **common,
    }


def _egm_kernel_params(
    *,
    regime: Regime,
    regime_name: RegimeName,
    flat_params: FlatParams,
) -> dict[str, object]:
    """Flat params fed into a DC-EGM kernel: the source's plus its targets'.

    A DC-EGM source carrying into a *different* target regime evaluates that
    target's resources / transition functions in its per-asset-node solve,
    reading the target's params (e.g. a pension factor the source never
    reads). These are model-level shared values, so the target's
    `flat_params` entry carries the right value; union them in. The kernel
    threads its `**kwargs` into the per-combo pool, and its captured functions
    read only the keys they need, so a target's extra params are harmless to
    the source functions that do not. Mirrors the fixed-param binding done at
    model build (`_partial_fixed_params_into_regimes`) for the free-param path.
    """
    params: dict[str, object] = dict(flat_params[regime_name])
    for target_name in regime.solution.transitions:
        for key, value in flat_params.get(target_name, MappingProxyType({})).items():
            params.setdefault(key, value)
    return params


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
    keyword-argument value. This is correct even when different partials
    bind different value objects — two partials share a compiled program
    only when every keyword value is the same object.

    For plain callables, use object identity.

    """
    if isinstance(func, functools.partial):
        return (
            id(func.func),
            tuple((k, id(v)) for k, v in sorted(func.keywords.items())),
        )
    return id(func)


@dataclass(frozen=True)
class _RegimeVTopology:
    """Shape and (optional) sharding of a single regime's V-array."""

    shape: tuple[int, ...]
    """V-array shape, with one entry per state."""

    sharding: jax.NamedSharding | None
    """Device sharding for the V-array, or `None` when no state is distributed."""


def _get_regime_V_shapes_and_shardings(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> dict[RegimeName, _RegimeVTopology]:
    """Compute V-array shapes and shardings for every regime.

    The V-array has one dimension per state variable, sized by that state's
    grid. When at least one state grid in a regime is distributed, the
    V-array is sharded across devices along those axes; otherwise the
    sharding is `None`.

    Args:
        regimes: Immutable mapping of regime names to internal regimes.
        flat_params: Regime parameters (needed for runtime grid shapes).

    Returns:
        Dict of regime names to `_RegimeVTopology` (shape and sharding).

    """
    n_devices = len(jax.devices())
    topology: dict[RegimeName, _RegimeVTopology] = {}
    for regime_name, regime in regimes.items():
        state_action_space = regime.solution.state_action_space(
            regime_params=flat_params[regime_name],
        )
        state_order: tuple[StateName, ...] = tuple(state_action_space.states.keys())
        shape = tuple(len(v) for v in state_action_space.states.values())
        sharding_plan = _build_regime_sharding(
            grids=regime.solution.grids, n_devices=n_devices
        )
        sharding = (
            sharding_plan.V_arr_sharding(state_order)
            if sharding_plan is not None
            else None
        )
        topology[regime_name] = _RegimeVTopology(shape=shape, sharding=sharding)
    return topology


def _build_zero_V_arr(*, topology: _RegimeVTopology) -> FloatND:
    """Build the zero V-array template for a regime, sharded where requested."""
    zeros = jnp.zeros(topology.shape)
    if topology.sharding is None:
        return zeros
    return jax.device_put(zeros, topology.sharding)


@dataclass(frozen=True)
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata — no device-array references — so
    every (regime, period) row stays at a few bytes regardless of grid
    size. State-action space, next-period V mapping, regime params, and
    the `compute_intermediates` closure are reconstructed lazily on the
    failure path from `regimes`, `flat_params`, and the
    partial `solution` built up to that point.
    """

    regime_name: RegimeName
    """Name of the regime whose V-array this row summarises."""
    period: int
    """Period index in the backward-induction loop."""
    age: float
    """Age corresponding to `period` (pulled off `AgeGrid.values`)."""


def _emit_post_loop_diagnostics(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    running_any_nan: BoolND,
    running_any_inf: BoolND,
    diagnostic_min: list[FloatND] | None,
    diagnostic_max: list[FloatND] | None,
    diagnostic_mean: list[FloatND] | None,
) -> None:
    """Flush async diagnostics: raise on NaN, warn on Inf, log debug stats.

    Only enters the per-row failure path when the running NaN or Inf
    accumulators are set, so a healthy solve incurs no host-side scalar
    materialisation here.
    """
    if running_any_nan.item():
        _raise_first_nan_row(
            diagnostic_rows=diagnostic_rows,
            solution=solution,
            regimes=regimes,
            flat_params=flat_params,
        )
    if running_any_inf.item():
        _warn_inf_rows(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            solution=solution,
        )
    if diagnostic_min is not None and diagnostic_max is not None and diagnostic_mean:
        _log_per_period_stats(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            mins=jnp.stack(diagnostic_min),
            maxs=jnp.stack(diagnostic_max),
            means=jnp.stack(diagnostic_mean),
        )


def _raise_first_nan_row(
    *,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Find the first NaN-bearing (regime, period) and raise.

    Failure-path only — walks rows until the first NaN hit.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isnan(V_arr)).item():
            _raise_at(
                row=row,
                solution=solution,
                regimes=regimes,
                flat_params=flat_params,
            )


def _raise_at(
    *,
    row: _DiagnosticRow,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Run the enriched NaN diagnostic on a single offending row and raise."""
    regime = regimes[row.regime_name]
    regime_params = flat_params[row.regime_name]
    # `compute_intermediates` was built from the regime's full `flat_param_names`
    # (per-iteration params + fixed params); the live solve loop merges
    # `resolved_fixed_params` into `regime_params` implicitly via the partialled
    # closures, but we have to do it by hand here to call the diagnostic
    # directly. Same merge order as `engine.state_action_space` and
    # `simulation.result`.
    effective_regime_params = MappingProxyType(
        {**regime.resolved_fixed_params, **regime_params}
    )
    state_action_space = regime.solution.state_action_space(regime_params=regime_params)
    next_regime_to_V_arr = _reconstruct_next_regime_to_V_arr(
        period=row.period,
        regimes=regimes,
        flat_params=flat_params,
        solution=solution,
    )
    # The intermediates closure mirrors the brute-force Q evaluation; for a
    # row solved by the EGM kernel it cannot reproduce the failing
    # computation, so the error is raised without the U/F/E/Q breakdown.
    compute_intermediates = (
        None
        if regime.solution.egm_step is not None
        else regime.solution.compute_intermediates.get(row.period)
    )
    V_arr = solution[row.period][row.regime_name]
    validate_V(
        V_arr=V_arr,
        age=row.age,
        regime_name=row.regime_name,
        partial_solution=solution,
        compute_intermediates=compute_intermediates,
        state_action_space=state_action_space,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=effective_regime_params,
        period=row.period,
    )


def _reconstruct_next_regime_to_V_arr(
    *,
    period: int,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> MappingProxyType[RegimeName, FloatND]:
    """Recreate the rolling `next_regime_to_V_arr` that was used at `period`.

    The hot loop rolls the per-regime V forward via `period_solution.get(name,
    next_regime_to_V_arr[name])`, so at iteration `period` each regime's slot
    holds its V from the smallest later period where it was active, falling
    back to a zeros template otherwise.

    Rebuild the same mapping post-hoc from `solution`. Shape and device
    sharding both come from `_get_regime_V_shapes_and_shardings` so the
    reconstructed templates have the same pytree structure and placement as
    the live ones in `solve()`.
    """
    regime_V_topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
    )
    later_periods = sorted(p for p in solution if p > period)
    result: dict[RegimeName, FloatND] = {}
    for regime_name, topology in regime_V_topology.items():
        rolled: FloatND | None = None
        for q in later_periods:
            if regime_name in solution[q]:
                rolled = solution[q][regime_name]
                break
        result[regime_name] = (
            rolled if rolled is not None else _build_zero_V_arr(topology=topology)
        )
    return MappingProxyType(result)


def _warn_inf_rows(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> None:
    """Emit a warning per (regime, period) with Inf values.

    Only invoked on the failure path (`running_any_inf` was True).
    Materialises one host-side bool per row.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isinf(V_arr)).item():
            logger.warning(
                "Inf in V_arr for regime '%s' at age %s",
                row.regime_name,
                row.age,
            )


def _log_per_period_stats(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    mins: FloatND,
    maxs: FloatND,
    means: FloatND,
) -> None:
    """Emit one debug log line per (regime, period) with V min/max/mean."""
    for row, V_min, V_max, V_mean in zip(
        diagnostic_rows, mins.tolist(), maxs.tolist(), means.tolist(), strict=True
    ):
        logger.debug(
            "  %s  age %s   V min=%.3g  max=%.3g  mean=%.3g",
            row.regime_name,
            row.age,
            V_min,
            V_max,
            V_mean,
        )
