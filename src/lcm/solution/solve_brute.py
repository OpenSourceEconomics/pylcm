import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.interfaces import InternalRegime
from lcm.typing import FloatND, InternalParams, RegimeName
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_header,
    log_period_timing,
    log_V_stats,
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
            Defaults to the number of physical CPU cores.

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
        {name: jnp.zeros(shape) for name, shape in regime_V_shapes.items()}
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

        for name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_space(
                regime_params=internal_params[name],
            )
            max_Q_over_a = compiled_functions[(name, period)]

            # evaluate Q-function on states and actions, and maximize over actions
            # Pass period/age as JAX arrays (not Python scalars) so the shared
            # jax.jit function is traced once with abstract shapes, not recompiled
            # for every distinct (period, age) pair.
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **internal_params[name],
                period=jnp.int32(period),
                age=ages.values[period],
            )

            log_nan_in_V(
                logger=logger,
                regime_name=name,
                age=ages.values[period],
                V_arr=V_arr,
            )
            log_V_stats(logger=logger, regime_name=name, V_arr=V_arr)

            validate_V(
                V_arr=V_arr,
                age=float(ages.values[period]),
                regime_name=name,
                partial_solution=MappingProxyType(solution),
                compute_intermediates=internal_regime.solve_functions.compute_intermediates.get(
                    period
                ),
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                internal_params=internal_params[name],
                period=period,
            )
            period_solution[name] = V_arr

        # Maintain consistent pytree structure: keep all regime keys,
        # update active regimes with solved V arrays.
        next_regime_to_V_arr = MappingProxyType(
            {
                name: period_solution.get(name, next_regime_to_V_arr[name])
                for name in internal_regimes
            }
        )
        solution[period] = MappingProxyType(period_solution)

        elapsed = time.monotonic() - period_start
        log_period_timing(logger=logger, elapsed=elapsed)

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
        Mapping of (regime_name, period) to callable (compiled or raw) functions.

    """
    # Collect all (regime, period) -> function mappings.
    all_functions: dict[tuple[RegimeName, int], Callable] = {}
    for name, regime in internal_regimes.items():
        for period in regime.active_periods:
            all_functions[(name, period)] = regime.solve_functions.max_Q_over_a[period]

    # If JIT is disabled, return raw functions directly.
    if not enable_jit:
        return all_functions

    # Deduplicate by identity (or by underlying function for partials).
    unique: dict[Hashable, tuple[Callable, RegimeName, int]] = {}
    for (name, period), func in all_functions.items():
        func_id = _func_dedup_key(func)
        if func_id not in unique:
            unique[func_id] = (func, name, period)

    n_workers = max_compilation_workers or _get_physical_core_count()
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
    for i, (func_id, (func, name, period)) in enumerate(unique.items(), 1):
        state_action_space = internal_regimes[name].state_action_space(
            regime_params=internal_params[name],
        )
        lower_args = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(internal_params[name]),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
        label = f"{name} (age {ages.values[period]})"
        labels[func_id] = label
        logger.info("%d/%d  %s", i, n_unique, label)
        logger.info("  lowering ...")
        start = time.monotonic()
        lowered[func_id] = jax.jit(func).lower(**lower_args)
        elapsed = time.monotonic() - start
        logger.info("  lowered in %s", format_duration(seconds=elapsed))

    # Phase 2: Compile all lowered programs in parallel (XLA releases the GIL).
    compiled = _compile_lowered_programs(
        lowered=lowered,
        labels=labels,
        n_workers=n_workers,
        logger=logger,
    )

    # Map back to (regime, period) keys.
    return {key: compiled[_func_dedup_key(func)] for key, func in all_functions.items()}


def _compile_lowered_programs(
    *,
    lowered: dict[Hashable, jax.stages.Lowered],
    labels: dict[Hashable, str],
    n_workers: int,
    logger: logging.Logger,
) -> dict[Hashable, jax.stages.Compiled]:
    """Compile lowered programs in parallel with memory-aware throttling.

    Use HLO text size as a proxy for compilation memory. A condition variable
    enforces a memory budget: large compilations limit concurrency while small
    ones run in parallel.

    """
    hlo_sizes: dict[Hashable, int] = {
        fid: len(low.as_text()) for fid, low in lowered.items()
    }
    budget = _get_compilation_memory_budget()
    largest_cost = max(hlo_sizes.values()) * _HLO_TO_MEMORY_SCALE
    logger.info(
        "Compilation memory budget: %s  (largest estimated: %s)",
        format_bytes(budget),
        format_bytes(largest_cost),
    )

    compiled: dict[Hashable, jax.stages.Compiled] = {}

    def _compile_and_log(
        func_id: Hashable,
        low: jax.stages.Lowered,
        label: str,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        logger.info("  compiling %s ...", label)
        rss_before = _get_rss()
        start = time.monotonic()
        result = low.compile()
        elapsed = time.monotonic() - start
        rss_after = _get_rss()
        rss_delta = rss_after - rss_before
        hlo_size = hlo_sizes[func_id]
        actual_scale = rss_delta / hlo_size if hlo_size > 0 else 0
        logger.info(
            "  compiled  %s  %s  (RSS delta: %s, HLO: %s, scale: %.0f)",
            label,
            format_duration(seconds=elapsed),
            format_bytes(rss_delta),
            format_bytes(hlo_size),
            actual_scale,
        )
        return func_id, result

    # Submit largest-first so big compilations start immediately and
    # small ones fill the gaps as memory frees up.
    ordered = sorted(lowered, key=lambda k: hlo_sizes[k], reverse=True)

    active_cost = 0
    budget_freed = threading.Condition(threading.Lock())

    def _compile_with_budget(
        func_id: Hashable,
        low: jax.stages.Lowered,
        label: str,
        cost: int,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        try:
            return _compile_and_log(func_id, low, label)
        finally:
            nonlocal active_cost
            with budget_freed:
                active_cost -= cost
                budget_freed.notify_all()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures: list = []
        for func_id in ordered:
            cost = hlo_sizes[func_id] * _HLO_TO_MEMORY_SCALE
            with budget_freed:
                while active_cost + cost > budget and active_cost > 0:
                    budget_freed.wait()
                active_cost += cost
            futures.append(
                pool.submit(
                    _compile_with_budget,
                    func_id,
                    lowered[func_id],
                    labels[func_id],
                    cost,
                )
            )
        for future in as_completed(futures):
            func_id, comp = future.result()
            compiled[func_id] = comp

    return compiled


def _func_dedup_key(func: Callable) -> Hashable:
    """Return a hashable deduplication key for a callable.

    For `functools.partial` objects wrapping shared JIT functions, deduplicate
    by the underlying function's identity and the keyword argument names.
    For plain callables, use object identity.
    """
    if isinstance(func, functools.partial):
        return (id(func.func), tuple(sorted(func.keywords)))
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
        Mapping of regime names to V array shapes.

    """
    shapes: dict[RegimeName, tuple[int, ...]] = {}
    for name, regime in internal_regimes.items():
        state_action_space = regime.state_action_space(
            regime_params=internal_params[name],
        )
        shapes[name] = tuple(len(v) for v in state_action_space.states.values())
    return shapes


# Estimated bytes of RAM per byte of HLO text during XLA compilation.
_HLO_TO_MEMORY_SCALE = 100


def _get_compilation_memory_budget() -> int:
    """Return memory budget for parallel XLA compilation in bytes.

    Use 70% of total physical RAM. Fall back to 8 GB on non-POSIX platforms.

    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size * 0.7)
    except ValueError, OSError:
        return 8 * 1024**3


def _get_physical_core_count() -> int:
    """Return the number of physical CPU cores.

    Count unique (package, core) pairs via sysfs topology on Linux.
    Fall back to `os.cpu_count()` on other platforms.

    """
    cpu_dir = Path("/sys/devices/system/cpu")
    try:
        physical: set[tuple[str, str]] = set()
        for entry in cpu_dir.iterdir():
            if entry.name.startswith("cpu") and entry.name[3:].isdigit():
                pkg = (entry / "topology" / "physical_package_id").read_text().strip()
                core = (entry / "topology" / "core_id").read_text().strip()
                physical.add((pkg, core))
        if physical:
            return len(physical)
    except OSError:
        pass
    return os.cpu_count() or 1


_BYTES_PER_KB = 1024


def format_bytes(n: float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < _BYTES_PER_KB:
            return f"{n:.1f} {unit}"
        n /= _BYTES_PER_KB
    return f"{n:.1f} PB"


def _get_rss() -> int:
    """Return current process RSS in bytes via /proc/self/status.

    Fall back to 0 on non-Linux platforms.

    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * _BYTES_PER_KB
    except OSError:
        pass
    return 0
