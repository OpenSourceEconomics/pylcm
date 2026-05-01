import functools
import logging
import os
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.interfaces import InternalRegime
from lcm.typing import FloatND, InternalParams, RegimeName
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_period_header,
    log_period_timing,
)


def solve(
    *,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        max_compilation_workers: Maximum number of threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    """
    # Compute V array shapes and build a consistent next_regime_to_V_arr
    # template.  Using the same pytree structure (keys and shapes) across
    # all periods avoids JIT re-compilation from pytree mismatches.
    regime_V_shapes = _get_regime_V_shapes(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
    )
    next_regime_to_V_arr = MappingProxyType(
        {
            regime_name: jnp.zeros(shape)
            for regime_name, shape in regime_V_shapes.items()
        }
    )

    # AOT-compile all unique max_Q_over_a functions in parallel.
    compiled_functions = _compile_all_functions(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
    )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}

    # Async diagnostics accumulators: per-period `jnp.any(isnan)` /
    # `jnp.any(isinf)` (and the debug min/max/mean trio) live here as
    # device-side scalars during the hot loop. The two NaN/Inf flags
    # are folded into single running scalars; the per-period min/max/
    # mean trio is appended to a list (only emitted at debug, where
    # we genuinely want every number on host).
    #
    # Per-period `block_until_ready()` after the running update forces
    # the device kernel to finish before the next period dispatches.
    # This frees the per-period `isnan(V_arr)` / `isinf(V_arr)`
    # intermediate buffers (~2 MB each at production grid sizes) so
    # they don't stack up. `block_until_ready` is a *device-only* sync
    # — no host transfer, no PCIe round-trip — so it doesn't
    # re-introduce the per-period host stalls that #334 removed; if
    # `max_Q_over_a` (the dominant per-period kernel) is in flight,
    # the call returns immediately when the small reduction is done.
    #
    # One host transfer per stat at end of solve (`.item()` on the
    # running scalars) decides whether to enter the failure-path
    # localisation. On a healthy solve no per-row materialisation
    # happens.
    #
    # Gate falls out of the public log level: `"off"` ⇒ nothing,
    # `"warning"` / `"progress"` ⇒ NaN/Inf only, `"debug"` ⇒ adds the
    # min/max/mean trio. `"off"` skips even the NaN fail-fast — that
    # is the documented contract of `"off"` (suppress all output) and
    # is what makes the level useful for tight estimation loops.
    diagnostics_enabled = logger.isEnabledFor(logging.WARNING)
    stats_enabled = logger.isEnabledFor(logging.DEBUG)
    diagnostic_rows: list[_DiagnosticRow] = []
    diagnostic_min: list[FloatND] = []
    diagnostic_max: list[FloatND] = []
    diagnostic_mean: list[FloatND] = []
    running_any_nan: FloatND = jnp.zeros((), dtype=bool)
    running_any_inf: FloatND = jnp.zeros((), dtype=bool)

    logger.info("Starting solution")
    total_start = time.monotonic()

    # backwards induction loop
    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        log_period_header(
            logger=logger,
            age=ages.values[period],
            n_active_regimes=len(active_regimes),
        )

        for regime_name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_space(
                regime_params=internal_params[regime_name],
            )
            max_Q_over_a = compiled_functions[(regime_name, period)]

            # evaluate Q-function on states and actions, and maximize over actions
            # Pass period/age as JAX arrays (not Python scalars) so the shared
            # jax.jit function is traced once with abstract shapes, not recompiled
            # for every distinct (period, age) pair.
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **internal_params[regime_name],
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
                running_any_nan = running_any_nan | jnp.any(jnp.isnan(V_arr))
                running_any_inf = running_any_inf | jnp.any(jnp.isinf(V_arr))
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
                for regime_name in internal_regimes
            }
        )
        solution[period] = MappingProxyType(period_solution)

        elapsed = time.monotonic() - period_start
        log_period_timing(logger=logger, elapsed=elapsed)

    if diagnostics_enabled:
        _emit_post_loop_diagnostics(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            solution=MappingProxyType(solution),
            internal_regimes=internal_regimes,
            internal_params=internal_params,
            running_any_nan=running_any_nan,
            running_any_inf=running_any_inf,
            diagnostic_min=diagnostic_min if stats_enabled else None,
            diagnostic_max=diagnostic_max if stats_enabled else None,
            diagnostic_mean=diagnostic_mean if stats_enabled else None,
        )

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return MappingProxyType(solution)


def _compile_all_functions(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
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
        internal_regimes: The internal regimes containing solve functions.
        internal_params: Regime parameters for constructing lowering args.
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
    for regime_name, regime in internal_regimes.items():
        for period in regime.active_periods:
            all_functions[(regime_name, period)] = regime.solve_functions.max_Q_over_a[
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
        state_action_space = internal_regimes[regime_name].state_action_space(
            regime_params=internal_params[regime_name],
        )
        lower_args = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(internal_params[regime_name]),
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


def _get_regime_V_shapes(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
) -> dict[RegimeName, tuple[int, ...]]:
    """Compute value function array shapes for all regimes.

    The V array has one dimension per state variable, with size equal to
    the number of grid points for that state.

    Args:
        internal_regimes: The internal regimes.
        internal_params: Regime parameters (needed for runtime grid shapes).

    Returns:
        Dict of regime names to V array shapes.

    """
    shapes: dict[RegimeName, tuple[int, ...]] = {}
    for regime_name, regime in internal_regimes.items():
        state_action_space = regime.state_action_space(
            regime_params=internal_params[regime_name],
        )
        shapes[regime_name] = tuple(len(v) for v in state_action_space.states.values())
    return shapes


@dataclass(frozen=True)
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata — no device-array references — so
    every (regime, period) row stays at a few bytes. The expensive bits
    (state-action space, next-period V mapping, params, the
    `compute_intermediates` closure) are reconstructed lazily on the
    failure path from `internal_regimes`, `internal_params`, and the
    partial `solution` that has been built up to that point.

    The earlier design captured those device-backed objects directly on
    each row, which pinned every period's V template in device memory
    until the post-loop flush — at production grid sizes that hits OOM
    well before the loop completes.
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
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    running_any_nan: FloatND,
    running_any_inf: FloatND,
    diagnostic_min: list[FloatND] | None,
    diagnostic_max: list[FloatND] | None,
    diagnostic_mean: list[FloatND] | None,
) -> None:
    """Flush async diagnostics: raise on NaN, warn on Inf, log debug stats.

    Two host transfers (the `.item()` calls on the running scalars)
    decide whether we enter the per-row failure-path localisation. On
    a healthy solve neither inner walk runs and no per-row scalar is
    materialised — the property that lets a production-sized solve at
    `log_level="warning"` fit on a 16 GB device that was OOMing on the
    previous stack-and-flush pattern.
    """
    if running_any_nan.item():
        _raise_first_nan_row(
            diagnostic_rows=diagnostic_rows,
            solution=solution,
            internal_regimes=internal_regimes,
            internal_params=internal_params,
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
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
) -> None:
    """Find the first NaN-bearing (regime, period) and raise.

    Only invoked on the failure path (`running_any_nan` was True).
    Materialises one host-side bool per row until the first hit; on
    a healthy solve this function is never called.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isnan(V_arr)).item():
            _raise_at(
                row=row,
                solution=solution,
                internal_regimes=internal_regimes,
                internal_params=internal_params,
            )


def _raise_at(
    *,
    row: _DiagnosticRow,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
) -> None:
    """Run the enriched NaN diagnostic on a single offending row and raise."""
    internal_regime = internal_regimes[row.regime_name]
    regime_params = internal_params[row.regime_name]
    state_action_space = internal_regime.state_action_space(regime_params=regime_params)
    next_regime_to_V_arr = _reconstruct_next_regime_to_V_arr(
        period=row.period,
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        solution=solution,
    )
    compute_intermediates = internal_regime.solve_functions.compute_intermediates.get(
        row.period
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
        internal_params=regime_params,
        period=row.period,
    )


def _reconstruct_next_regime_to_V_arr(
    *,
    period: int,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> MappingProxyType[RegimeName, FloatND]:
    """Recreate the rolling `next_regime_to_V_arr` that was used at `period`.

    The hot loop rolls the per-regime V forward via `period_solution.get(name,
    next_regime_to_V_arr[name])`, so at iteration `period` each regime's slot
    holds its V from the smallest later period where it was active, falling
    back to a zeros template otherwise.

    We rebuild the same mapping post-hoc from `solution`. The shapes come from
    the regime's state-action space at the supplied params — identical to what
    `_get_regime_V_shapes` saw during solve setup.
    """
    regime_V_shapes = _get_regime_V_shapes(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
    )
    later_periods = sorted(p for p in solution if p > period)
    result: dict[RegimeName, FloatND] = {}
    for regime_name, shape in regime_V_shapes.items():
        rolled: FloatND | None = None
        for q in later_periods:
            if regime_name in solution[q]:
                rolled = solution[q][regime_name]
                break
        result[regime_name] = rolled if rolled is not None else jnp.zeros(shape)
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
    for row, v_min, v_max, v_mean in zip(
        diagnostic_rows, mins.tolist(), maxs.tolist(), means.tolist(), strict=True
    ):
        logger.debug(
            "  %s  age %s   V min=%.3g  max=%.3g  mean=%.3g",
            row.regime_name,
            row.age,
            v_min,
            v_max,
            v_mean,
        )
