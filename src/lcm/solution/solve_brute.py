import functools
import logging
import math
import os
import time
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid
from lcm.interfaces import InternalRegime
from lcm.typing import FlatRegimeParams, FloatND, InternalParams, RegimeName
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_header,
    log_period_timing,
    log_V_stats,
)


@dataclass(frozen=True)
class CompiledSolve:
    """AOT-compiled solve kernels plus the shape-consistent V template.

    Reused across partition-point sweeps so `Model.solve` pays the compile
    cost once, not once per point.
    """

    compiled_functions: MappingProxyType[tuple[RegimeName, int], Callable]
    """Immutable mapping of `(regime_name, period)` to the compiled
    `max_Q_over_a` callable."""
    next_regime_to_V_arr_template: MappingProxyType[RegimeName, FloatND]
    """Zero-initialised V-arrays keyed by regime name; used as starting
    value for backward induction on every call to `run_compiled_solve`."""
    partition_shape: tuple[int, ...] = ()
    """Shape of the partition-grid Cartesian product the kernel was
    wrapped for. Empty tuple means no partition sweep; otherwise
    `prod(partition_shape)` is the leading axis of every returned
    V-array. Consumed by `run_compiled_solve` for per-partition-point
    diagnostic logging."""


def solve(
    *,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search (compile + run in one call).

    Thin wrapper around `compile_solve` + `run_compiled_solve`. Callers
    that sweep partition points should split these two steps themselves
    so the compile cost is paid exactly once.

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
    compiled = compile_solve(
        internal_params=internal_params,
        ages=ages,
        internal_regimes=internal_regimes,
        logger=logger,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
    )
    return run_compiled_solve(
        compiled=compiled,
        internal_params=internal_params,
        ages=ages,
        internal_regimes=internal_regimes,
        logger=logger,
    )


def compile_solve(
    *,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
    partition_shape: tuple[int, ...] = (),
    regime_partitions: Mapping[
        RegimeName, Mapping[str, DiscreteGrid]
    ] = MappingProxyType({}),
) -> CompiledSolve:
    """AOT-compile all max_Q_over_a functions and build the V template.

    Separated from `run_compiled_solve` so callers can sweep partition
    points (or similar scalar-param variations) over the same compiled
    kernels, paying the compile cost exactly once.

    Args:
        internal_params: Regime params used to construct state-action
            spaces for lowering (only shapes/dtypes are consulted). When
            `partition_shape` is non-empty, the partition-valued leaves
            must already carry a leading axis of length
            `prod(partition_shape)` (see `stack_partition_scalars`).
        ages: Age grid for the model.
        internal_regimes: The internal regimes.
        logger: Logger for compilation progress.
        enable_jit: Whether to JIT-compile. When `False`, returns raw
            functions bundled in a `CompiledSolve` unchanged.
        max_compilation_workers: Maximum threads for parallel compilation.
        partition_shape: Shape of the partition-grid Cartesian product.
            `()` means no partition sweep (default). A non-empty shape
            switches on a `jax.vmap` wrap over the leading partition axis
            inside each compiled kernel, so a single dispatch produces
            all partition points' value functions at once.
        regime_partitions: Per-regime mapping of partition names → grid.
            Only consulted when `partition_shape` is non-empty; tells the
            wrap which kwargs carry the leading partition axis.

    Returns:
        A `CompiledSolve` holding the compiled per-period kernels and
        the zero-initialised V template. When `partition_shape` is
        non-empty the template's V-arrays carry a leading axis of
        length `prod(partition_shape)`.

    """
    regime_V_shapes = _get_regime_V_shapes(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
    )
    partition_size = math.prod(partition_shape)
    leading = (partition_size,) if partition_shape else ()
    next_regime_to_V_arr = MappingProxyType(
        {name: jnp.zeros(leading + shape) for name, shape in regime_V_shapes.items()}
    )

    compiled_functions = _compile_all_functions(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
        partition_shape=partition_shape,
        regime_partitions=regime_partitions,
    )
    return CompiledSolve(
        compiled_functions=MappingProxyType(dict(compiled_functions)),
        next_regime_to_V_arr_template=next_regime_to_V_arr,
        partition_shape=partition_shape,
    )


def run_compiled_solve(
    *,
    compiled: CompiledSolve,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Run backward induction using an already-compiled solve.

    Args:
        compiled: Result of `compile_solve`. Reused across partition
            points so no recompilation happens between calls.
        internal_params: Immutable mapping of regime names to flat
            parameter mappings. Partition scalars flow in here and are
            picked up at dispatch time — no new compile.
        ages: Age grid.
        internal_regimes: The internal regimes (needed for active-period
            bookkeeping and diagnostic machinery).
        logger: Logger.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    """
    next_regime_to_V_arr = compiled.next_regime_to_V_arr_template
    compiled_functions = compiled.compiled_functions
    partition_shape = compiled.partition_shape

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

            # Diagnostic logging: when the kernel was vmap'd over a
            # partition axis the leading axis of `V_arr` is the flat
            # partition index. Loop over it so NaN warnings and stats
            # attribute to a specific point rather than being aggregated.
            if partition_shape:
                for point_idx in range(V_arr.shape[0]):
                    V_slice = V_arr[point_idx]
                    label = f"{name}[partition point {point_idx}]"
                    log_nan_in_V(
                        logger=logger,
                        regime_name=label,
                        age=ages.values[period],
                        V_arr=V_slice,
                    )
                    log_V_stats(logger=logger, regime_name=label, V_arr=V_slice)
            else:
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
    partition_shape: tuple[int, ...] = (),
    regime_partitions: Mapping[
        RegimeName, Mapping[str, DiscreteGrid]
    ] = MappingProxyType({}),
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
        partition_shape: Shape of the partition-grid Cartesian product. When
            non-empty, every kernel is wrapped in `jax.vmap` over the leading
            partition axis.
        regime_partitions: Per-regime partition names; consulted only when
            `partition_shape` is non-empty.

    Returns:
        dict of (regime_name, period) to callable (compiled or raw) functions.

    """
    # Collect all (regime, period) -> function mappings.
    all_functions: dict[tuple[RegimeName, int], Callable] = {}
    for name, regime in internal_regimes.items():
        for period in regime.active_periods:
            all_functions[(name, period)] = regime.solve_functions.max_Q_over_a[period]

    if partition_shape:
        all_functions = _vmap_raw_functions_over_partition_axis(
            all_functions=all_functions,
            internal_regimes=internal_regimes,
            internal_params=internal_params,
            regime_partitions=regime_partitions,
        )

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


def _vmap_raw_functions_over_partition_axis(
    *,
    all_functions: dict[tuple[RegimeName, int], Callable],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    regime_partitions: Mapping[RegimeName, Mapping[str, DiscreteGrid]],
) -> dict[tuple[RegimeName, int], Callable]:
    """Wrap each unique raw function once in vmap over the leading partition axis.

    Within a regime, multiple periods share a raw function (shared-JIT), so
    the per-(regime, raw-func) cache keeps the wrapper object stable across
    those periods — the existing id()-based dedup inside the lowering loop
    then collapses them as before.
    """
    # Cache is keyed per regime: the wrapper closes over state/action names
    # and `partition_names_for_regime`, so even two regimes sharing the same
    # raw `max_Q_over_a` object cannot safely share a wrapper. Dedup within
    # a regime across its periods still collapses — `raw_func` identity
    # matches across periods via shared-JIT.
    wrap_cache: dict[tuple[RegimeName, Hashable], Callable] = {}
    wrapped_functions: dict[tuple[RegimeName, int], Callable] = {}
    for (name, period), raw_func in all_functions.items():
        cache_key = (name, _func_dedup_key(func=raw_func))
        if cache_key not in wrap_cache:
            wrap_cache[cache_key] = _wrap_with_partition_vmap(
                func=raw_func,
                internal_regime=internal_regimes[name],
                internal_params_for_regime=internal_params[name],
                partition_names_for_regime=frozenset(regime_partitions.get(name, ())),
            )
        wrapped_functions[(name, period)] = wrap_cache[cache_key]
    return wrapped_functions


def _wrap_with_partition_vmap(
    *,
    func: Callable,
    internal_regime: InternalRegime,
    internal_params_for_regime: FlatRegimeParams,
    partition_names_for_regime: frozenset[str],
) -> Callable:
    """Wrap `max_Q_over_a` in `jax.vmap` over the leading partition axis.

    The wrap uses a dict-shaped `in_axes` that marks:

    - partition-valued entries of `internal_params` → axis 0
    - every entry of `next_regime_to_V_arr` → axis 0 (leading partition axis)
    - everything else (state/action grids, non-partition params, period, age)
      → None

    Args:
        func: Raw `max_Q_over_a` callable (typically a shared-JIT function).
        internal_regime: Owning internal regime; used only to build the
            state-action space so the wrap can precompute the list of
            state/action kwarg names.
        internal_params_for_regime: The regime's params dict; its keys
            define the param kwargs the wrap will forward.
        partition_names_for_regime: Subset of `internal_params` keys that
            carry a leading partition axis.

    Returns:
        A kw-only callable with the same external signature as `func`, but
        internally vmap'd over the leading partition axis.

    """
    state_action_space = internal_regime.state_action_space(
        regime_params=internal_params_for_regime
    )
    # Partition states are supposed to be lifted out of the regime's
    # state-action space by `lift_partitions_from_regime`. If one ever leaks
    # back in, `top_level_axes[name] = None` below would silently broadcast
    # where it should vmap, so we stop here with a loud error.
    state_action_names = set(state_action_space.states) | set(
        state_action_space.actions
    )
    leaked = partition_names_for_regime & state_action_names
    if leaked:
        msg = (
            f"Partition names {sorted(leaked)} appear in the regime's "
            f"state-action space; `lift_partitions_from_regime` was expected "
            f"to remove them before this wrap."
        )
        raise AssertionError(msg)
    # Classify each top-level kwarg: 0 for vmap'd partition leaves, None for
    # broadcast. `_broadcast_in_axes_like` then expands None into a matching
    # all-None pytree for nested kwargs so JAX's in_axes structure aligns
    # exactly with the input pytree.
    top_level_axes: dict[str, object] = {}
    for name in state_action_space.states:
        top_level_axes[name] = None
    for name in state_action_space.actions:
        top_level_axes[name] = None
    for name in internal_params_for_regime:
        top_level_axes[name] = 0 if name in partition_names_for_regime else None
    top_level_axes["next_regime_to_V_arr"] = 0
    top_level_axes["period"] = None
    top_level_axes["age"] = None

    def _call_with_kwargs_dict(kwargs_dict: Mapping[str, object]) -> FloatND:
        return func(**kwargs_dict)

    def call_as_kwargs(**kwargs: object) -> FloatND:
        # jax.jit canonicalises kwargs to alphabetical order; sorting here
        # keeps in_axes and the value pytree byte-identical in structure.
        sorted_kwargs = MappingProxyType(
            {name: kwargs[name] for name in sorted(kwargs)}
        )
        in_axes = MappingProxyType(
            {
                name: _broadcast_in_axes_like(
                    axis=top_level_axes[name], value=sorted_kwargs[name]
                )
                for name in sorted_kwargs
            }
        )
        return jax.vmap(_call_with_kwargs_dict, in_axes=(in_axes,))(sorted_kwargs)

    return call_as_kwargs


def _broadcast_in_axes_like(*, axis: object, value: object) -> object:
    """Expand a scalar `axis` spec into a pytree matching `value`'s structure.

    `jax.vmap`'s `in_axes` supports a prefix-tree form where a scalar (e.g.
    `None`) is interpreted as "apply to every leaf below". In practice some
    JAX versions are stricter and require the `in_axes` pytree to match the
    value structure exactly, so we expand here instead of relying on the
    prefix semantics.
    """
    # Leaves get the axis as-is.
    leaves_and_tree = jax.tree_util.tree_flatten(value)
    leaves, treedef = leaves_and_tree
    return jax.tree_util.tree_unflatten(treedef, [axis] * len(leaves))


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
