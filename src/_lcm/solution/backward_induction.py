import functools
import logging
import os
import time
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

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


def _states_for_period(
    regime: Regime, state_action_space: StateActionSpace, period: int
) -> Mapping[str, object]:
    """Current-period state axes, overriding age-varying states with period-t nodes.

    For a regime with `AgeSpecializedGrid` states, replace the representative base
    axis with this period's grid nodes so period-t's value function is tabulated on
    period-t's grid (consistent with the continuation interpolation, which reads
    V_{t+1} on period-(t+1)'s grid). Same shape as the base, so the shared compiled
    kernel is not retraced. Age-invariant regimes return the base axis unchanged.
    """
    # getattr (not direct access) so a duck-typed mock regime without the field works.
    axes = getattr(regime.solution, "period_state_axes", None)
    if axes is not None and period in axes:
        return {**state_action_space.states, **axes[period]}
    return state_action_space.states


def solve(
    *,
    flat_params: FlatParams,
    ages: AgeGrid,
    regimes: MappingProxyType[RegimeName, Regime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
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
        Immutable mapping of periods to regime value function arrays.

    """
    # Compute V array shapes (and their device shardings, if any) and build
    # a consistent next_regime_to_V_arr template. Using the same pytree
    # structure (keys and shapes) across all periods avoids JIT re-
    # compilation from pytree mismatches.
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

    # AOT-compile all unique max_Q_over_a functions in parallel.
    compiled_functions = _compile_all_functions(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
    )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}

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
    diagnostic_rows: list[_DiagnosticRow] = []
    diagnostic_min: list[FloatND] = []
    diagnostic_max: list[FloatND] = []
    diagnostic_mean: list[FloatND] = []
    running_any_nan: BoolND = jnp.zeros((), dtype=bool)
    running_any_inf: BoolND = jnp.zeros((), dtype=bool)

    logger.info("Starting solution")
    total_start = time.monotonic()

    # backwards induction loop
    base_state_action_spaces = _build_base_state_action_spaces(
        regimes=regimes, flat_params=flat_params
    )

    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}

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
            state_action_space = base_state_action_spaces[regime_name]

            # evaluate Q-function on states and actions, and maximize over
            # actions (the compiled function is the period's max_Q_over_a).
            # Pass period/age as JAX arrays (not Python scalars) so the shared
            # jax.jit function is traced once with abstract shapes, not
            # recompiled for every distinct (period, age) pair. Age-varying
            # (`AgeSpecializedGrid`) states get this period's grid nodes.
            V_arr = compiled_functions[(regime_name, period)](
                **_states_for_period(regime, state_action_space, period),
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **flat_params[regime_name],
                period=jnp.int32(period),
                age=ages.values[period],
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

        # Maintain consistent pytree structure: keep all regime keys,
        # update active regimes with solved V arrays.
        next_regime_to_V_arr = MappingProxyType(
            {
                regime_name: period_solution.get(
                    regime_name, next_regime_to_V_arr[regime_name]
                )
                for regime_name in regimes
            }
        )
        solution[period] = MappingProxyType(period_solution)

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

    return MappingProxyType(solution)


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
    enable_jit: bool,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> dict[tuple[RegimeName, int], Callable]:
    """AOT-compile all unique max_Q_over_a functions in parallel.

    With shared-JIT, many periods share the same `jax.jit`-wrapped function
    object. This function deduplicates by object identity, traces each unique
    function once (sequential), then compiles the XLA programs in parallel
    via a thread pool (XLA releases the GIL during compilation).

    When JIT is disabled (`enable_jit=False`), returns the raw functions
    without compilation.

    Args:
        regimes: The internal regimes containing solve functions.
        flat_params: Regime parameters for constructing lowering args.
        ages: Age grid for the model.
        next_regime_to_V_arr: Template with consistent keys and V array shapes
            for constructing lowering arguments.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        max_compilation_workers: Maximum threads for parallel compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.

    Returns:
        Dict of (regime_name, period) to callable (compiled or raw) functions.

    """
    # Collect all (regime, period) -> function mappings.
    all_functions: dict[tuple[RegimeName, int], Callable] = {}
    for regime_name, regime in regimes.items():
        for period in regime.active_periods:
            all_functions[(regime_name, period)] = regime.solution.max_Q_over_a[period]

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
        state_action_space = regimes[regime_name].solution.state_action_space(
            regime_params=flat_params[regime_name],
        )
        lower_args = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(flat_params[regime_name]),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
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
    compute_intermediates = regime.solution.compute_intermediates.get(row.period)
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
