"""AOT-compile simulate functions for a fixed batch size.

When `Model(n_subjects=N)` is set, `compile_all_simulation_phases(...)` returns
an `regimes` mapping with each regime's `simulation` callables
swapped for AOT-compiled programs sized for batch shape `N`. The existing
simulate call sites then pick them up transparently — no signature changes
downstream.

A regime declaring `gated_edges` runs one further program per edge per period:
the gate evaluator the router re-evaluates at the realized candidate target
state. Those are compiled here too and installed in the router's own
population-call table, so no gated model pays a trace and a compile in its
first routed period.

Compilation deduplicates callables by identity (only one program per unique
callable), lowers them sequentially (JAX tracing is not thread-safe), then
parallel-compiles them via a `ThreadPoolExecutor` (XLA releases the GIL).
"""

import dataclasses
import logging
import time
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType

import jax
import jax.numpy as jnp
from dags.tree import qname_from_tree_path

from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.gated_edges import (
    ResolvedGatedEdge,
    bind_edge_period_context,
    build_reference_params_mapping_for_fold,
    build_same_period_mapping_for_fold,
    unsupplied_dissolution_flag,
)
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG, SAME_PERIOD_V_ARG
from _lcm.simulation.gated_routing import (
    bind_provenance_params,
    install_population_call,
    population_call,
    split_population_call_args,
)
from _lcm.simulation.initial_conditions import subject_array_sharding
from _lcm.simulation.random import generate_simulation_keys
from _lcm.solution.backward_induction import (
    _func_dedup_key,
    _iter_edge_topologies,
    _resolve_compilation_workers,
)
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
    _RegimeVTopology,
)
from _lcm.typing import FlatParams, FlatRegimeParams, RegimeName
from _lcm.utils.logging import format_duration
from lcm.ages import AgeGrid
from lcm.typing import FloatND, IntND, ScalarFloat, ScalarInt


def compile_all_simulation_phases(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    n_subjects: int,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> MappingProxyType[RegimeName, Regime]:
    """AOT-compile every unique simulate function for batch shape `n_subjects`.

    Args:
        regimes: Original internal regimes from the Model.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.
        n_subjects: Batch size for which to compile.
        max_compilation_workers: Maximum threads for parallel XLA compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger.

    Returns:
        Immutable mapping of regime names to Regime where each regime's
        `simulation` phase has its callables replaced by AOT-compiled programs.

    """
    # Per-regime V-shape and -sharding lookup for building period-specific
    # templates that match the *sparse* mapping `simulate.simulate(...)`
    # actually dispatches: `period_to_regime_to_V_arr.get(P+1, {})` — only
    # regimes active at P+1. The sharding must match what `solve(...)`
    # produces, so the AOT program accepts the runtime value-function arrays.
    regime_V_topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
    )

    # One model-wide subject sharding: subjects propagate across regimes, so
    # every AOT program must be lowered with the same per-subject sharding.
    subject_sharding = subject_array_sharding(regimes=regimes, n_subjects=n_subjects)

    unique, func_keys, gate_calls = _collect_unique_simulation_callables(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        n_subjects=n_subjects,
        regime_V_topology=regime_V_topology,
        subject_sharding=subject_sharding,
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
        # A gate evaluator's population call takes its two kwarg pools
        # POSITIONALLY (`vmap`'s `in_axes` addresses positional arguments
        # only), so its lower-args arrive as a tuple; every other program
        # here is lowered by keyword.
        lowered[key] = (
            func.lower(*args)  # ty: ignore[unresolved-attribute]
            if isinstance(args, tuple)
            else func.lower(**args)  # ty: ignore[unresolved-attribute, invalid-argument-type]
        )
        # Drop the concrete lower-args once the `Lowered` object has captured
        # its abstract values. This releases V-shaped templates, per-regime
        # subject-state/action zeros, and the regime-params view before the
        # parallel compile pool starts piling Compiled kernels onto the heap.
        unique[key] = (func, None, label)
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
            # Release the HLO module held by the `Lowered` object now that
            # its `Compiled` counterpart is in `compiled`; otherwise every
            # lowered intermediate stays resident until the slowest compile
            # finishes.
            del lowered[k]

    # The router looks a gate evaluator's population call up on the evaluator
    # itself, so the compiled program is installed there rather than swapped
    # into a regime field.
    for key, (evaluator, axis_size) in gate_calls.items():
        install_population_call(evaluator, axis_size=axis_size, call=compiled[key])

    return _swap_in_compiled(
        regimes=regimes,
        compiled=compiled,
        func_keys=func_keys,
    )


def _collect_unique_simulation_callables(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    n_subjects: int,
    regime_V_topology: dict[RegimeName, _RegimeVTopology],
    subject_sharding: jax.NamedSharding | None,
) -> tuple[
    dict[Hashable, tuple[Callable, dict | tuple | None, str]],
    dict[tuple[RegimeName, str, int | None], Hashable],
    dict[Hashable, tuple[Callable, int]],
]:
    """Walk every regime/period and dedup the simulate functions to compile.

    `argmax_and_max_Q_over_a` dedup keys on `(func_id, continuation_targets)`
    so two periods that share the same argmax callable but see a different
    `next_regime_to_V_arr` pytree (different graph target set) get
    separate compiled programs whose signature matches what runtime actually
    dispatches.

    Returns:
        Tuple of the unique programs to lower keyed by dedup key, the
        regime/slot/period lookup into them, and the gate evaluators to
        install in the router's population-call table keyed by the same dedup
        key.

    """
    unique: dict[Hashable, tuple[Callable, dict | tuple | None, str]] = {}
    func_keys: dict[tuple[RegimeName, str, int | None], Hashable] = {}
    gate_calls: dict[Hashable, tuple[Callable, int]] = {}

    # One zero `Wbar` per declared gated edge, shaped like the target's state
    # grid plus the source's stakeholder axis — the same templates the solve
    # side lowers its source kernels against. A source regime's own decision
    # reads `Wbar` in place of the raw target V, so its continuation slot is
    # lowered against this and not against the target's own topology.
    edge_to_V_arr = MappingProxyType(
        {
            (source_name, target_name): _build_zero_V_arr(topology=topology)
            for source_name, target_name, topology in _iter_edge_topologies(
                regimes=regimes, flat_params=flat_params
            )
        }
    )

    for regime_name, regime in regimes.items():
        regime_params = flat_params.get(regime_name, MappingProxyType({}))
        sf = regime.simulation

        # `sf.argmax_and_max_Q_over_a` has entries for *every* period
        # (pylcm builds them across the full age grid), but the regime is
        # only dispatched at runtime for periods in `regime.active_periods`.
        # Inactive-period entries can carry a continuation-target set whose
        # shape doesn't match the regime's actual transitions for that
        # period; tracing them would surface `next_<state>` bookkeeping
        # mismatches the lazy path never reaches. Restrict AOT to active
        # periods to mirror runtime.
        for period in regime.active_periods:
            argmax_func = sf.argmax_and_max_Q_over_a[period]
            continuation_targets = (
                ()
                if period == ages.n_periods - 1
                else regime.solution.reachability.targets(
                    period=period, source=regime_name
                )
            )
            next_regime_to_V_arr = _with_edge_substitution(
                regime=regime,
                regime_name=regime_name,
                next_regime_to_V_arr=MappingProxyType(
                    {
                        name: _build_zero_V_arr(topology=regime_V_topology[name])
                        for name in continuation_targets
                    }
                ),
                edge_to_V_arr=edge_to_V_arr,
            )
            args = _build_argmax_args(
                regime=regime,
                regime_params=regime_params,
                ages=ages,
                period=period,
                n_subjects=n_subjects,
                next_regime_to_V_arr=next_regime_to_V_arr,
                regime_V_topology=regime_V_topology,
                flat_params=flat_params,
                subject_sharding=subject_sharding,
            )
            key = ("argmax", _func_dedup_key(func=argmax_func), continuation_targets)
            func_keys[(regime_name, "argmax", period)] = key
            if key not in unique:
                label = (
                    f"{regime_name}/argmax_and_max_Q_over_a "
                    f"(age {ages.values[period].item()})"
                )
                unique[key] = (jax.jit(argmax_func), args, label)

        # `next_state` / `crtp` are keyed per-regime: each regime's lower-args
        # depend on its own state-action shapes, so even when two regimes
        # share a callable identity, their compiled programs are distinct.
        if not regime.terminal:
            # `next_state` lower-args are period-independent, so build them once;
            # periods whose specialized functions resolve to the same closures
            # share one compiled program via the callable dedup key.
            args = _build_next_state_args(
                regime=regime,
                regime_params=regime_params,
                ages=ages,
                n_subjects=n_subjects,
                subject_sharding=subject_sharding,
            )
            for period in regime.active_periods:
                next_state_func = sf.next_state[period]
                key = (
                    "next_state",
                    regime_name,
                    _func_dedup_key(func=next_state_func),
                )
                func_keys[(regime_name, "next_state", period)] = key
                if key not in unique:
                    # Re-wrap with `jax.jit`: when `fixed_params` are partialled
                    # into the regime, `next_state_func` is a `functools.partial`
                    # (no `.lower()`); plain jit objects are also fine to re-jit.
                    label = (
                        f"{regime_name}/next_state (age {ages.values[period].item()})"
                    )
                    unique[key] = (jax.jit(next_state_func), args, label)

        if sf.compute_regime_transition_probs is not None:
            args = _build_crtp_args(
                regime=regime,
                regime_params=regime_params,
                ages=ages,
                n_subjects=n_subjects,
                subject_sharding=subject_sharding,
            )
            key = (
                "crtp",
                regime_name,
                _func_dedup_key(func=sf.compute_regime_transition_probs),
            )
            func_keys[(regime_name, "crtp", None)] = key
            if key not in unique:
                unique[key] = (
                    jax.jit(sf.compute_regime_transition_probs),
                    args,
                    f"{regime_name}/compute_regime_transition_probs",
                )

        _collect_edge_gate_evaluators(
            regime=regime,
            regime_name=regime_name,
            regimes=regimes,
            flat_params=flat_params,
            ages=ages,
            n_subjects=n_subjects,
            regime_V_topology=regime_V_topology,
            subject_sharding=subject_sharding,
            unique=unique,
            gate_calls=gate_calls,
        )

    return unique, func_keys, gate_calls


def _collect_edge_gate_evaluators(
    *,
    regime: Regime,
    regime_name: RegimeName,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    n_subjects: int,
    regime_V_topology: dict[RegimeName, _RegimeVTopology],
    subject_sharding: jax.NamedSharding | None,
    unique: dict[Hashable, tuple[Callable, dict | tuple | None, str]],
    gate_calls: dict[Hashable, tuple[Callable, int]],
) -> None:
    """Add one program per unique gate evaluator this regime's edges declare.

    The router recomputes each edge's gate at the realized candidate target
    state, once per edge per period, through a population call it memoizes on
    the evaluator. Lowering that same population call here — against the same
    two kwarg pools, split by the same function the router splits them with —
    is what lets the compiled program be installed under the key the router
    then looks up.

    An evaluator is shared across the periods that group onto it, and its
    period and age enter as traced scalars rather than as static arguments,
    so those periods share one program.
    """
    for target_name, edge in regime.gated_edges.items():
        for fold_period in _edge_fold_periods(regime=regime, ages=ages):
            evaluator = edge.simulate_gate_evaluator_at(period=fold_period)
            key = ("gate", regime_name, target_name, _func_dedup_key(func=evaluator))
            if key in gate_calls:
                continue
            args = _build_gate_evaluator_args(
                edge=edge,
                evaluator=evaluator,
                source_name=regime_name,
                target_regime=regimes[target_name],
                target_name=target_name,
                fold_period=fold_period,
                fold_age=ages.period_to_age(fold_period),
                flat_params=flat_params,
                n_subjects=n_subjects,
                regime_V_topology=regime_V_topology,
                subject_sharding=subject_sharding,
            )
            gate_calls[key] = (evaluator, n_subjects)
            unique[key] = (
                population_call(evaluator, axis_size=n_subjects),
                args,
                (
                    f"{regime_name}/gate into {target_name} "
                    f"(age {ages.values[fold_period].item()})"
                ),
            )


def _edge_fold_periods(*, regime: Regime, ages: AgeGrid) -> tuple[int, ...]:
    """Return the fold periods this regime's edges are routed at.

    A gate is decided on the value the subject would enter NEXT period, so the
    router passes `period + 1`; the source's last active period has no
    successor and routes nothing.
    """
    return tuple(
        period + 1 for period in regime.active_periods if period + 1 < ages.n_periods
    )


def _build_gate_evaluator_args(
    *,
    edge: ResolvedGatedEdge,
    evaluator: Callable,
    source_name: RegimeName,
    target_regime: Regime,
    target_name: RegimeName,
    fold_period: int,
    fold_age: float | ScalarFloat | ScalarInt,
    flat_params: FlatParams,
    n_subjects: int,
    regime_V_topology: dict[RegimeName, _RegimeVTopology],
    subject_sharding: jax.NamedSharding | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the positional pair one gate evaluator's population call takes.

    Every element mirrors what `simulation.gated_routing.route_gated_edges`
    dispatches:

    - the candidate target states, per-subject arrays carrying the TARGET
      regime's own simulate state names and grid dtypes, because the router
      hands the evaluator that regime's slice of the state carrier;
    - each argument bound from the regime that owns it, as the evaluator's own
      `arg_provenance` records;
    - the fold's period and age, for an evaluator that declares them;
    - the same-period value mapping and the reference regimes' own params,
      under the two reserved keys.

    The dissolution flag is supplied rather than left out: an edge whose gate
    reads `D_target` refuses a mapping without one, and where the gate does
    not read it the stand-in has the same shape and dtype either way, so the
    lowered signature matches whichever the run supplies.
    """
    candidate_target_states = _subject_state_carrier_template(
        regime=target_regime,
        n_subjects=n_subjects,
        sharding=subject_sharding,
    )
    target_V = _build_zero_V_arr(topology=regime_V_topology[edge.target])
    period_solution = {
        name: _build_zero_V_arr(topology=regime_V_topology[name])
        for name in dict.fromkeys((edge.target, *edge.reference_regimes))
    }
    static_kwargs: dict[str, object] = {
        **bind_provenance_params(
            evaluator.arg_provenance,  # ty: ignore[unresolved-attribute]
            flat_params=flat_params,
            source_name=source_name,
            target_name=target_name,
        ),
        **bind_edge_period_context(
            evaluator, fold_period=fold_period, fold_age=fold_age
        ),
        SAME_PERIOD_V_ARG: build_same_period_mapping_for_fold(
            edge=edge,
            period_solution=period_solution,
            period_dissolution_flags={
                # Shaped by the same rule a target that publishes no flag is
                # stood in for, so the two are one signature.
                edge.target: jnp.zeros(
                    unsupplied_dissolution_flag(edge=edge, target_V=target_V).shape,
                    dtype=bool,
                )
            },
        ),
        SAME_PERIOD_PARAMS_ARG: build_reference_params_mapping_for_fold(
            edge=edge, flat_params=flat_params
        ),
    }
    return split_population_call_args(
        evaluator,
        batched_kwargs=candidate_target_states,
        static_kwargs=static_kwargs,
    )


def _subject_state_carrier_template(
    *,
    regime: Regime,
    n_subjects: int,
    sharding: jax.NamedSharding | None,
) -> dict[str, FloatND | IntND]:
    """Return zeros shaped like one regime's slice of the simulate carrier.

    `build_initial_states` gives every simulate state a `(n_subjects,)` array
    at the grid's own dtype — the discrete grid's index dtype, the canonical
    float dtype otherwise — and the router hands the target regime's slice of
    that carrier straight to the gate evaluator.
    """
    arrays: dict[str, FloatND | IntND] = {}
    for state_name in regime.simulation.state_names:
        grid = regime.simulation.grids[state_name]
        dtype = (
            grid.to_jax().dtype
            if isinstance(grid, DiscreteGrid)
            else canonical_float_dtype()
        )
        zeros = jnp.zeros((n_subjects,), dtype=dtype)
        arrays[state_name] = (
            zeros if sharding is None else jax.device_put(zeros, sharding)
        )
    return arrays


def _swap_in_compiled(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    compiled: dict[Hashable, jax.stages.Compiled],
    func_keys: dict[tuple[RegimeName, str, int | None], Hashable],
) -> MappingProxyType[RegimeName, Regime]:
    """Swap compiled programs into each regime's `simulation` phase."""
    new_regimes: dict[RegimeName, Regime] = {}
    for regime_name, regime in regimes.items():
        sf = regime.simulation
        # Only active periods are AOT-compiled (see
        # `_collect_unique_simulation_callables`); leave inactive-period
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
            next_state_compiled_for_active = {
                period: compiled[func_keys[(regime_name, "next_state", period)]]
                for period in regime.active_periods
            }
            next_state_compiled = MappingProxyType(
                {
                    period: next_state_compiled_for_active.get(period, original_func)
                    for period, original_func in sf.next_state.items()
                }
            )
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
        new_regimes[regime_name] = dataclasses.replace(regime, simulation=new_sf)

    return MappingProxyType(new_regimes)


def _with_edge_substitution(
    *,
    regime: Regime,
    regime_name: RegimeName,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    edge_to_V_arr: Mapping[tuple[RegimeName, RegimeName], FloatND],
) -> MappingProxyType[RegimeName, FloatND]:
    """Replace each gated-edge target's raw V template with its edge continuation.

    A regime declaring `gated_edges` chooses its own action against the edge's
    operand channels — the target's value components, the gate's references and
    each leg's fallback, on the target's grid under one trailing channel axis —
    which `simulation.gated_routing.substitute_gated_edge_continuations` swaps
    into the continuation mapping before the decision. Lowering against the
    target's own V would size that slot by the target's topology instead, so
    the compiled program would reject the array it is invoked with.

    Returns `next_regime_to_V_arr` unchanged for a regime without gated edges.
    """
    if not regime.gated_edges:
        return next_regime_to_V_arr
    return MappingProxyType(
        {
            name: (
                edge_to_V_arr[(regime_name, name)]
                if name in regime.gated_edges
                else arr
            )
            for name, arr in next_regime_to_V_arr.items()
        }
    )


def _build_argmax_args(
    *,
    regime: Regime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    period: int,
    n_subjects: int,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    regime_V_topology: dict[RegimeName, _RegimeVTopology],
    flat_params: FlatParams,
    subject_sharding: jax.NamedSharding | None,
) -> dict[str, object]:
    """Build the argmax program's lowering arguments.

    A regime declaring `same_period_refs` reads each reference regime's
    THIS-period value inside its value-aware feasibility mask, and simulate
    dispatches those arrays together with each reference regime's OWN flat
    params (the reference V is interpolated over the reference regime's grid,
    whose runtime grid points are that regime's parameters). Both ride along
    here so the compiled program accepts them.

    The same-period templates come from each reference regime's V topology, so
    a reference regime that is ALSO a gated-edge target is lowered against its
    own value function — not against the `Wbar` substituted into the
    continuation slot, which simulate never passes here.
    """
    base = regime.solution.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(
        base.states, n_subjects=n_subjects, sharding=subject_sharding
    )
    same_period_args: dict[str, object] = {}
    if regime.same_period_ref_regimes:
        same_period_args[SAME_PERIOD_V_ARG] = MappingProxyType(
            {
                ref: _build_zero_V_arr(topology=regime_V_topology[ref])
                for ref in regime.same_period_ref_regimes
            }
        )
        same_period_args[SAME_PERIOD_PARAMS_ARG] = MappingProxyType(
            {ref: flat_params[ref] for ref in regime.same_period_ref_regimes}
        )
    return {
        **subject_states,
        **base.discrete_actions,
        **base.continuous_actions,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **same_period_args,
        **regime_params,
        "period": jnp.int32(period),
        "age": ages.values[period],
    }


def _build_next_state_args(
    *,
    regime: Regime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    n_subjects: int,
    subject_sharding: jax.NamedSharding | None,
) -> dict[str, object]:
    base = regime.solution.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(
        base.states, n_subjects=n_subjects, sharding=subject_sharding
    )
    # Simulate-only states (carried states declared via `Phased`)
    # are not solve grid axes, so they are absent from `state_action_space`. The
    # simulate `next_state` program carries and reads them, so seed each one.
    subject_states.update(
        _simulate_only_subject_states(
            regime, n_subjects=n_subjects, sharding=subject_sharding
        )
    )
    subject_actions = _subject_shape_arrays(
        {**base.discrete_actions, **base.continuous_actions},
        n_subjects=n_subjects,
        sharding=subject_sharding,
    )

    transition_plans = regime.simulation.transition_plans
    stoch_next_func_names = sorted(
        qname_from_tree_path((target_regime_name, transition_name))
        for target_regime_name, bundle in (regime.simulation.transitions.items())
        for transition_name in bundle
        if transition_plans[target_regime_name].is_lottery(transition_name)
    )
    _, stoch_keys = generate_simulation_keys(
        key=jax.random.key(0),
        names=stoch_next_func_names,
        n_initial_states=n_subjects,
    )

    return {
        **subject_states,
        **subject_actions,
        **stoch_keys,
        "period": jnp.int32(0),
        "age": ages.values[0],
        **regime_params,
    }


def _build_crtp_args(
    *,
    regime: Regime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
    n_subjects: int,
    subject_sharding: jax.NamedSharding | None,
) -> dict[str, object]:
    base = regime.solution.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(
        base.states, n_subjects=n_subjects, sharding=subject_sharding
    )
    # The realized draw reads carried states as leaves, so the lower-args
    # must seed them like the next_state program's.
    simulate_only_states = _simulate_only_subject_states(
        regime, n_subjects=n_subjects, sharding=subject_sharding
    )
    subject_actions = _subject_shape_arrays(
        {**base.discrete_actions, **base.continuous_actions},
        n_subjects=n_subjects,
        sharding=subject_sharding,
    )
    return {
        **subject_states,
        **simulate_only_states,
        **subject_actions,
        "period": jnp.int32(0),
        "age": ages.values[0],
        **regime_params,
    }


def _simulate_only_subject_states(
    regime: Regime,
    *,
    n_subjects: int,
    sharding: jax.NamedSharding | None,
) -> dict[str, FloatND | IntND]:
    """Return `(n_subjects,)` zeros for the regime's simulate-only states.

    Simulate-only states are the carried states (declared via
    `Phased(solve=..., simulate=Grid)`); they are carried per subject in
    simulate but are not solve grid axes. Each is seeded with a zero array of
    its grid's dtype.
    """
    arrays: dict[str, FloatND | IntND] = {}
    for name, grid in regime.simulation.carried_grids.items():
        zeros = jnp.zeros((n_subjects,), dtype=grid.to_jax().dtype)
        arrays[name] = zeros if sharding is None else jax.device_put(zeros, sharding)
    return arrays


def _subject_shape_arrays(
    base_arrays: Mapping[str, FloatND | IntND],
    *,
    n_subjects: int,
    sharding: jax.NamedSharding | None,
) -> dict[str, FloatND | IntND]:
    """Return zeros of shape `(n_subjects,)` mirroring each base array's dtype.

    With `build_initial_states` casting discrete states to the grid dtype,
    runtime states (initial + post-transition) share the grid's dtype, so
    using `arr.dtype` from the regime's grid here matches runtime.

    When the regime distributes its grids, `sharding` scatters the zeros
    across the device mesh exactly as `build_initial_states` scatters the
    runtime per-subject arrays, so the AOT-compiled program is lowered for
    the device layout it is dispatched with.
    """
    arrays: dict[str, FloatND | IntND] = {}
    for name, arr in base_arrays.items():
        zeros = jnp.zeros((n_subjects,), dtype=arr.dtype)
        arrays[name] = zeros if sharding is None else jax.device_put(zeros, sharding)
    return arrays
