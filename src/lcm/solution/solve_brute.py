import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import MappingProxyType

import cloudpickle
import jax
import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.interfaces import InternalRegime
from lcm.solution.cache_warming import (
    lower_functions_in_subprocess,
    reconstruct_and_compile,
)
from lcm.typing import FloatND, InternalParams, RegimeName, UserParams
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
    max_compilation_workers: int | None = None,
    regimes: MappingProxyType | None = None,
    regime_id_class: type | None = None,
    fixed_params: UserParams | None = None,
    enable_jit: bool = True,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.
        max_compilation_workers: Maximum number of parallel lowering processes.
            Defaults to `os.cpu_count()`.
        regimes: User regimes (for subprocess Model reconstruction).
        regime_id_class: Regime ID dataclass (for subprocess Model reconstruction).
        fixed_params: Fixed params (for subprocess Model reconstruction).
        enable_jit: Whether JIT is enabled.

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
        max_compilation_workers=max_compilation_workers,
        logger=logger,
        regimes=regimes,
        regime_id_class=regime_id_class,
        fixed_params=fixed_params,
        enable_jit=enable_jit,
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
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **internal_params[name],
                period=period,
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
    max_compilation_workers: int | None,
    logger: logging.Logger,
    regimes: MappingProxyType | None,
    regime_id_class: type | None,
    fixed_params: UserParams | None,
    enable_jit: bool,
) -> dict[tuple[RegimeName, int], Callable]:
    """AOT-compile all unique max_Q_over_a functions in parallel.

    With shared-JIT, many periods share the same `jax.jit`-wrapped function
    object. This function deduplicates by object identity, then lowers each
    unique function in a subprocess (parallelizing the expensive tracing
    step). The serialized HLO is sent back to the main process, which
    reconstructs and compiles the executable in milliseconds.

    When JIT is disabled (`enable_jit=False`) or model ingredients are not
    provided, falls back to sequential in-process lowering.

    Args:
        internal_regimes: The internal regimes containing solve functions.
        internal_params: Regime parameters for constructing lowering args.
        ages: Age grid for the model.
        next_regime_to_V_arr: Template with consistent keys and V array shapes
            for constructing lowering arguments.
        max_compilation_workers: Maximum number of parallel lowering processes.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.
        regimes: User regimes for subprocess Model reconstruction.
        regime_id_class: Regime ID dataclass for subprocess Model reconstruction.
        fixed_params: Fixed params for subprocess Model reconstruction.
        enable_jit: Whether JIT is enabled.

    Returns:
        Mapping of (regime_name, period) to callable (compiled or raw) functions.

    """
    # Collect all (regime, period) -> function mappings.
    all_functions: dict[tuple[RegimeName, int], Callable] = {}
    for name, regime in internal_regimes.items():
        for period in regime.active_periods:
            all_functions[(name, period)] = regime.solve_functions.max_Q_over_a[period]

    # If JIT is disabled, return raw functions directly.
    sample_func = next(iter(all_functions.values()))
    if not hasattr(sample_func, "lower"):
        return all_functions

    # Deduplicate by object identity — get one representative (regime, period)
    # per unique function.
    unique_repr: dict[int, tuple[RegimeName, int]] = {}
    for (name, period), func in all_functions.items():
        func_id = id(func)
        if func_id not in unique_repr:
            unique_repr[func_id] = (name, period)

    n_workers = max_compilation_workers or os.cpu_count() or 1
    n_unique = len(unique_repr)

    logger.info(
        "AOT compilation: %d unique functions (%d regime-period pairs, %d workers)",
        n_unique,
        len(all_functions),
        n_workers,
    )

    can_parallel = regimes is not None and regime_id_class is not None
    if can_parallel and n_workers > 1:
        assert regimes is not None  # noqa: S101
        assert regime_id_class is not None  # noqa: S101
        compiled = _compile_parallel(
            unique_repr=unique_repr,
            internal_params=internal_params,
            ages=ages,
            next_regime_to_V_arr=next_regime_to_V_arr,
            n_workers=n_workers,
            logger=logger,
            regimes=regimes,
            regime_id_class=regime_id_class,
            fixed_params=fixed_params or MappingProxyType({}),
            enable_jit=enable_jit,
        )
    else:
        compiled = _compile_sequential(
            unique_repr=unique_repr,
            internal_regimes=internal_regimes,
            internal_params=internal_params,
            ages=ages,
            next_regime_to_V_arr=next_regime_to_V_arr,
            logger=logger,
        )

    # Map back to (regime, period) keys.
    return {key: compiled[id(func)] for key, func in all_functions.items()}


def _compile_sequential(
    *,
    unique_repr: dict[int, tuple[RegimeName, int]],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    logger: logging.Logger,
) -> dict[int, Callable]:
    """Lower and compile functions sequentially in the main process."""
    compiled: dict[int, Callable] = {}
    n_unique = len(unique_repr)

    for i, (func_id, (name, period)) in enumerate(unique_repr.items(), 1):
        func = internal_regimes[name].solve_functions.max_Q_over_a[period]
        state_action_space = internal_regimes[name].state_action_space(
            regime_params=internal_params[name],
        )
        lower_args = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(internal_params[name]),
            "period": period,
            "age": ages.values[period],
        }
        label = f"{name} (age {ages.values[period]})"
        logger.info("%d/%d  %s", i, n_unique, label)
        logger.info("  lowering ...")
        start = time.monotonic()
        lowered = func.lower(**lower_args)  # ty: ignore[unresolved-attribute]
        logger.info(
            "  lowered in %s", format_duration(seconds=time.monotonic() - start)
        )
        start = time.monotonic()
        comp = lowered.compile()
        logger.info(
            "  compiled in %s", format_duration(seconds=time.monotonic() - start)
        )
        compiled[func_id] = comp

    return compiled


def _compile_parallel(
    *,
    unique_repr: dict[int, tuple[RegimeName, int]],
    internal_params: InternalParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    n_workers: int,
    logger: logging.Logger,
    regimes: MappingProxyType,
    regime_id_class: type,
    fixed_params: UserParams,
    enable_jit: bool,
) -> dict[int, Callable]:
    """Lower functions in parallel subprocesses, compile in main process.

    Each subprocess rebuilds the Model from cloudpickled Regimes, lowers its
    assigned functions, serializes the HLO bytes, and returns them. The main
    process reconstructs compiled executables from HLO in milliseconds.
    """
    import pickle  # noqa: PLC0415

    # Assign unique functions round-robin to workers.
    assignments_per_worker: dict[int, list[tuple[RegimeName, int]]] = {
        i: [] for i in range(n_workers)
    }
    func_id_to_assignment: dict[int, tuple[RegimeName, int]] = {}
    for i, (func_id, (name, period)) in enumerate(unique_repr.items()):
        worker_idx = i % n_workers
        assignments_per_worker[worker_idx].append((name, period))
        func_id_to_assignment[func_id] = (name, period)

    # Cloudpickle the shared model ingredients once.
    shared_data = (
        dict(regimes),
        regime_id_class,
        ages,
        fixed_params,
        enable_jit,
        internal_params,
        next_regime_to_V_arr,
    )
    enable_x64: bool = jax.config.jax_enable_x64  # ty: ignore[unresolved-attribute]

    logger.info(
        "Lowering %d functions in %d parallel workers ...", len(unique_repr), n_workers
    )
    lower_start = time.monotonic()

    ctx = mp.get_context("spawn")
    n_done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {}
        for worker_idx, assignments in assignments_per_worker.items():
            if not assignments:
                continue
            pickled = cloudpickle.dumps((*shared_data, assignments))
            futures[pool.submit(lower_functions_in_subprocess, pickled, enable_x64)] = (
                worker_idx
            )

        all_results: list[tuple[str, int, bytes, dict]] = []
        for future in as_completed(futures):
            worker_results = pickle.loads(future.result())  # noqa: S301
            all_results.extend(worker_results)
            n_done += len(worker_results)
            elapsed = time.monotonic() - lower_start
            logger.info(
                "  lowered %d/%d  (%s elapsed)",
                n_done,
                len(unique_repr),
                format_duration(seconds=elapsed),
            )

    lower_elapsed = time.monotonic() - lower_start
    logger.info("Lowering complete  (%s)", format_duration(seconds=lower_elapsed))

    # Phase 2: Reconstruct + compile in main process (fast, ~50ms each).
    compile_start = time.monotonic()

    # Build lookup from (regime_name, period) -> func_id
    assignment_to_func_id = {v: k for k, v in func_id_to_assignment.items()}

    compiled: dict[int, Callable] = {}
    for regime_name, period, hlo_bytes, metadata in all_results:
        func_id = assignment_to_func_id[(regime_name, period)]
        compiled[func_id] = reconstruct_and_compile(hlo_bytes, metadata)

    compile_elapsed = time.monotonic() - compile_start
    logger.info(
        "Compiled %d functions  (%s)",
        len(compiled),
        format_duration(seconds=compile_elapsed),
    )

    return compiled


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
