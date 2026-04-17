import functools
import logging
import os
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
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

            # Include sibling regimes already solved this period (and the
            # current regime's V_arr, even though it is NaN-bearing — users
            # debugging the snapshot want to see all of it).
            partial = MappingProxyType(
                {
                    **solution,
                    period: MappingProxyType({**period_solution, name: V_arr}),
                }
            )
            validate_V(
                V_arr=V_arr,
                age=float(ages.values[period]),
                regime_name=name,
                partial_solution=partial,
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
        Dict of (regime_name, period) to callable (compiled or raw) functions.

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
        func_id = _func_dedup_key(func=func)
        if func_id not in unique:
            unique[func_id] = (func, name, period)

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
        label = f"{name} (age {ages.values[period].item()})"
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
    for name, regime in internal_regimes.items():
        state_action_space = regime.state_action_space(
            regime_params=internal_params[name],
        )
        shapes[name] = tuple(len(v) for v in state_action_space.states.values())
    return shapes
