"""AOT-compile simulate functions for a fixed batch size.

When `Model(n_subjects=N)` is set, `compile_all_simulate_functions(...)` returns
an `internal_regimes` mapping with each regime's `simulate_functions` callables
swapped for AOT-compiled programs sized for batch shape `N`. The existing
simulate call sites then pick them up transparently — no signature changes
downstream.

Mirrors the pattern in `solve_brute._compile_all_functions`: deduplicate by
callable identity, sequentially lower (tracing is not thread-safe), then
parallel-compile via `ThreadPoolExecutor` (XLA releases the GIL).
"""

import dataclasses
import logging
import time
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType

import jax
import jax.numpy as jnp
from dags.tree import tree_path_from_qname
from jax import Array

from lcm.ages import AgeGrid
from lcm.interfaces import InternalRegime
from lcm.simulation.random import generate_simulation_keys
from lcm.solution.solve_brute import (
    _func_dedup_key,
    _resolve_compilation_workers,
)
from lcm.typing import (
    FlatRegimeParams,
    InternalParams,
    RegimeName,
)
from lcm.utils.logging import format_duration
from lcm.utils.namespace import flatten_regime_namespace


def compile_all_simulate_functions(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
    n_subjects: int,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """AOT-compile every unique simulate function for batch shape `n_subjects`.

    Args:
        internal_regimes: Original internal regimes from the Model.
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.
        n_subjects: Batch size for which to compile.
        max_compilation_workers: Maximum threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger.

    Returns:
        Immutable mapping of regime names to InternalRegime where each
        regime's `simulate_functions` has its callables replaced by
        AOT-compiled programs.

    """
    # Per-regime V-shape lookup for building period-specific templates that
    # match the *sparse* mapping `simulate.simulate(...)` actually dispatches:
    # `period_to_regime_to_V_arr.get(P+1, {})` — only regimes active at P+1.
    regime_V_shapes = _get_regime_V_shapes(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
    )

    unique, func_keys = _collect_unique_simulate_functions(
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        ages=ages,
        n_subjects=n_subjects,
        regime_V_shapes=regime_V_shapes,
    )

    n_workers = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    n_unique = len(unique)
    logger.info(
        "Simulate AOT compilation: %d unique functions (%d workers)",
        n_unique,
        n_workers,
    )

    lowered: dict[Hashable, jax.stages.Lowered] = {}
    for i, (key, (func, args, label)) in enumerate(unique.items(), 1):
        logger.info("%d/%d  %s", i, n_unique, label)
        logger.info("  lowering ...")
        start = time.monotonic()
        # `func` is a `jax.jit`-wrapped callable; ty sees only the abstract
        # Callable type, so it can't see `.lower(...)`.
        lowered[key] = func.lower(**args)  # ty: ignore[unresolved-attribute]
        logger.info(
            "  lowered in %s", format_duration(seconds=time.monotonic() - start)
        )

    compiled: dict[Hashable, jax.stages.Compiled] = {}

    def _compile_and_log(
        *,
        key: Hashable,
        low: jax.stages.Lowered,
        label: str,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        logger.info("  compiling %s ...", label)
        start = time.monotonic()
        result = low.compile()
        logger.info(
            "  compiled  %s  %s",
            label,
            format_duration(seconds=time.monotonic() - start),
        )
        return key, result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_compile_and_log, key=key, low=low, label=unique[key][2])
            for key, low in lowered.items()
        ]
        for future in as_completed(futures):
            k, c = future.result()
            compiled[k] = c

    return _swap_in_compiled(
        internal_regimes=internal_regimes,
        compiled=compiled,
        func_keys=func_keys,
    )


def _collect_unique_simulate_functions(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
    n_subjects: int,
    regime_V_shapes: dict[RegimeName, tuple[int, ...]],
) -> tuple[
    dict[Hashable, tuple[Callable, dict, str]],
    dict[tuple[RegimeName, str, int | None], Hashable],
]:
    """Walk every regime/period and dedup the simulate functions to compile.

    `argmax_and_max_Q_over_a` dedup keys on `(func_id, active_at_next_period)`
    so two periods that share the same argmax callable but see a different
    `next_regime_to_V_arr` pytree (different active-regime set at P+1) get
    separate compiled programs whose signature matches what runtime actually
    dispatches.
    """
    unique: dict[Hashable, tuple[Callable, dict, str]] = {}
    func_keys: dict[tuple[RegimeName, str, int | None], Hashable] = {}

    for regime_name, regime in internal_regimes.items():
        regime_params = internal_params.get(regime_name, MappingProxyType({}))
        sf = regime.simulate_functions

        # `sf.argmax_and_max_Q_over_a` has entries for *every* period
        # (pylcm builds them across the full age grid), but the regime is
        # only dispatched at runtime for periods in `regime.active_periods`.
        # The unused entries can carry a stale `complete_targets` set
        # whose shape doesn't match the regime's actual transitions
        # (e.g. a forced-canwork regime's argmax for a pre-FRA period
        # has choose targets in scope, even though the regime never
        # reaches that period at runtime). Tracing those would surface
        # `next_<state>` bookkeeping inconsistencies that the lazy path
        # never trips. Restrict AOT to active periods to mirror runtime.
        for period in regime.active_periods:
            argmax_func = sf.argmax_and_max_Q_over_a[period]
            active_next = _active_regimes_at_period(
                internal_regimes=internal_regimes, period=period + 1
            )
            next_regime_to_V_arr = MappingProxyType(
                {name: jnp.zeros(regime_V_shapes[name]) for name in active_next}
            )
            args = _build_argmax_args(
                internal_regime=regime,
                regime_params=regime_params,
                ages=ages,
                period=period,
                n_subjects=n_subjects,
                next_regime_to_V_arr=next_regime_to_V_arr,
            )
            key = ("argmax", _func_dedup_key(func=argmax_func), active_next)
            func_keys[(regime_name, "argmax", period)] = key
            if key not in unique:
                label = (
                    f"{regime_name}/argmax_and_max_Q_over_a "
                    f"(age {ages.values[period].item()})"
                )
                unique[key] = (jax.jit(argmax_func), args, label)

        if not regime.terminal:
            args = _build_next_state_args(
                internal_regime=regime,
                regime_params=regime_params,
                ages=ages,
                n_subjects=n_subjects,
            )
            key = ("next_state", _func_dedup_key(func=sf.next_state))
            func_keys[(regime_name, "next_state", None)] = key
            if key not in unique:
                # Re-wrap with `jax.jit`: when `fixed_params` are partialled
                # into the regime, `sf.next_state` is a `functools.partial`
                # (no `.lower()`); plain jit objects are also fine to re-jit.
                unique[key] = (
                    jax.jit(sf.next_state),
                    args,
                    f"{regime_name}/next_state",
                )

        if sf.compute_regime_transition_probs is not None:
            args = _build_crtp_args(
                internal_regime=regime,
                regime_params=regime_params,
                ages=ages,
                n_subjects=n_subjects,
            )
            key = ("crtp", _func_dedup_key(func=sf.compute_regime_transition_probs))
            func_keys[(regime_name, "crtp", None)] = key
            if key not in unique:
                unique[key] = (
                    jax.jit(sf.compute_regime_transition_probs),
                    args,
                    f"{regime_name}/compute_regime_transition_probs",
                )

    return unique, func_keys


def _swap_in_compiled(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    compiled: dict[Hashable, jax.stages.Compiled],
    func_keys: dict[tuple[RegimeName, str, int | None], Hashable],
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Swap compiled programs into each regime's `simulate_functions`."""
    new_regimes: dict[RegimeName, InternalRegime] = {}
    for regime_name, regime in internal_regimes.items():
        sf = regime.simulate_functions
        # Only active periods are AOT-compiled (see
        # `_collect_unique_simulate_functions`); leave inactive-period
        # entries untouched so the existing closure stays in place — they
        # are never dispatched at runtime anyway.
        argmax_compiled_for_active = {
            period: compiled[func_keys[(regime_name, "argmax", period)]]
            for period in regime.active_periods
        }
        argmax_compiled = MappingProxyType(
            {
                period: argmax_compiled_for_active.get(period, original_func)
                for period, original_func in sf.argmax_and_max_Q_over_a.items()
            }
        )
        if regime.terminal:
            next_state_compiled = sf.next_state
        else:
            next_state_compiled = compiled[func_keys[(regime_name, "next_state", None)]]
        if sf.compute_regime_transition_probs is None:
            crtp_compiled = None
        else:
            crtp_compiled = compiled[func_keys[(regime_name, "crtp", None)]]

        new_sf = dataclasses.replace(
            sf,
            argmax_and_max_Q_over_a=argmax_compiled,
            next_state=next_state_compiled,
            compute_regime_transition_probs=crtp_compiled,
        )
        new_regimes[regime_name] = dataclasses.replace(
            regime, simulate_functions=new_sf
        )

    return MappingProxyType(new_regimes)


def _get_regime_V_shapes(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
) -> dict[RegimeName, tuple[int, ...]]:
    shapes: dict[RegimeName, tuple[int, ...]] = {}
    for regime_name, regime in internal_regimes.items():
        space = regime.state_action_space(
            regime_params=internal_params.get(regime_name, MappingProxyType({}))
        )
        shapes[regime_name] = tuple(len(v) for v in space.states.values())
    return shapes


def _active_regimes_at_period(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    period: int,
) -> tuple[RegimeName, ...]:
    """Tuple of regime names active at `period`, in `internal_regimes` order.

    Returned as a `tuple` so it's hashable and pytree-key-stable. An empty
    tuple matches the runtime fallback for periods past the last (`{}`).
    """
    return tuple(
        regime_name
        for regime_name, regime in internal_regimes.items()
        if period in regime.active_periods
    )


def _build_argmax_args(
    *,
    internal_regime: InternalRegime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    period: int,
    n_subjects: int,
    next_regime_to_V_arr: MappingProxyType[RegimeName, Array],
) -> dict[str, object]:
    base = internal_regime.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(base.states, n_subjects=n_subjects)
    return {
        **subject_states,
        **base.discrete_actions,
        **base.continuous_actions,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **regime_params,
        "period": jnp.int32(period),
        "age": ages.values[period],
    }


def _build_next_state_args(
    *,
    internal_regime: InternalRegime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    n_subjects: int,
) -> dict[str, object]:
    base = internal_regime.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(base.states, n_subjects=n_subjects)
    subject_actions = _subject_shape_arrays(
        {**base.discrete_actions, **base.continuous_actions},
        n_subjects=n_subjects,
    )

    stoch_transition_names = (
        internal_regime.simulate_functions.stochastic_transition_names
    )
    stoch_next_func_names = sorted(
        next_func_name
        for next_func_name in flatten_regime_namespace(
            internal_regime.simulate_functions.transitions
        )
        if tree_path_from_qname(next_func_name)[-1] in stoch_transition_names
    )
    _, stoch_keys = generate_simulation_keys(
        key=jax.random.key(0),
        names=stoch_next_func_names,
        n_initial_states=n_subjects,
    )

    # `period` is passed as a plain Python int by `calculate_next_states`
    # (transitions.py), which traces as the default-precision int. Match that
    # here so the lowered shape signature lines up with the runtime call.
    return {
        **subject_states,
        **subject_actions,
        **stoch_keys,
        "period": 0,
        "age": ages.values[0],
        **regime_params,
    }


def _build_crtp_args(
    *,
    internal_regime: InternalRegime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    n_subjects: int,
) -> dict[str, object]:
    base = internal_regime.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(base.states, n_subjects=n_subjects)
    subject_actions = _subject_shape_arrays(
        {**base.discrete_actions, **base.continuous_actions},
        n_subjects=n_subjects,
    )
    return {
        **subject_states,
        **subject_actions,
        "period": 0,
        "age": ages.values[0],
        **regime_params,
    }


def _subject_shape_arrays(
    base_arrays: Mapping[str, Array],
    *,
    n_subjects: int,
) -> dict[str, Array]:
    """Return zeros of shape `(n_subjects,)` mirroring each base array's dtype."""
    return {
        name: jnp.zeros((n_subjects,), dtype=arr.dtype)
        for name, arr in base_arrays.items()
    }
