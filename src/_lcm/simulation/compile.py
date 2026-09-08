"""Prewarm the declared simulation programs and host-owned gate evaluators."""

import dataclasses
import logging
import time
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from dags.tree import qname_from_tree_path

from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.core_program import CoreProgram
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.gated_edges import (
    ResolvedGatedEdge,
    bind_edge_period_context,
    build_reference_params_mapping_for_fold,
    build_same_period_mapping_for_fold,
    unsupplied_dissolution_flag,
)
from _lcm.regime_building.Q_and_F import (
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
)
from _lcm.simulation.gated_routing import (
    bind_provenance_params,
    install_population_call,
    population_call,
    split_population_call_args,
)
from _lcm.simulation.initial_conditions import subject_array_sharding
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.random import generate_simulation_keys
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.value_placement import simulation_value_sharding
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


def bind_simulation_runtime(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    execution: ResolvedExecution,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, Regime]:
    """Attach one shared executor to a call-local copy of the program bundles."""
    executor = SimulationRuntime(
        execution=execution,
        enable_jit=enable_jit,
        subject_devices=_subject_devices(
            regimes=regimes, device_ids=execution.device_ids
        ),
    )
    return MappingProxyType(
        {
            name: dataclasses.replace(
                regime,
                simulation=dataclasses.replace(
                    regime.simulation,
                    programs=dataclasses.replace(
                        regime.simulation.programs, executor=executor
                    ),
                ),
            )
            for name, regime in regimes.items()
        }
    )


def _subject_devices(
    *, regimes: Mapping[RegimeName, Regime], device_ids: tuple[int, ...]
) -> tuple[jax.Device, ...]:
    """Resolve the actual population devices from the canonical regime axes."""
    devices = placed_devices_for_ids(
        submesh_device_ids=(), visible_device_ids=device_ids
    )
    return (
        devices
        if any(regime.solution.sharded_state_names for regime in regimes.values())
        else devices[:1]
    )


def lower_simulation_programs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    n_subjects: int,
    max_compilation_workers: int | None,
    logger: logging.Logger,
    device_ids: tuple[int, ...] = (),
) -> MappingProxyType[RegimeName, Regime]:
    """Prewarm the same program cache that lazy simulation dispatches.

    Arguments mirror the current population's sparse continuation topology,
    carried states, globally addressed random keys and subject placement.
    Template arrays are released after each program has been prepared.
    """
    if any(
        isinstance(executor := regime.simulation.programs.executor, SimulationRuntime)
        and executor.execution.device_memory_bytes is not None
        for regime in regimes.values()
    ):
        logger.info(
            "Deferring budgeted simulation compilation until live residency is "
            "available at the first dispatch."
        )
        return regimes
    regime_V_topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
        phase="simulate",
        device_ids=device_ids,
    )
    subject_sharding = subject_array_sharding(
        regimes=regimes, n_subjects=n_subjects, device_ids=device_ids
    )
    subject_devices = _subject_devices(regimes=regimes, device_ids=device_ids)
    edge_topologies = MappingProxyType(
        {
            (source_name, target_name): dataclasses.replace(
                topology,
                sharding=simulation_value_sharding(
                    stored_sharding=topology.sharding, devices=subject_devices
                ),
            )
            for source_name, target_name, topology in _iter_edge_topologies(
                regimes=regimes, flat_params=flat_params
            )
        }
    )
    prepared_gates: set[Hashable] = set()
    worker_count = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures: set[Future[None]] = set()
        for regime_name, regime in regimes.items():
            regime_params = flat_params.get(regime_name, MappingProxyType({}))
            programs = regime.simulation.programs
            executor = programs.executor
            if not isinstance(executor, SimulationRuntime):
                raise TypeError("Simulation prewarming requires the runtime executor.")
            for period, program in programs.decision.items():
                continuation_targets = (
                    ()
                    if period == ages.n_periods - 1
                    else regime.solution.reachability.targets(
                        period=period, source=regime_name
                    )
                )
                values = _with_edge_substitution(
                    regime=regime,
                    regime_name=regime_name,
                    next_regime_to_V_arr=MappingProxyType(
                        {
                            name: _build_zero_V_arr(topology=regime_V_topology[name])
                            for name in continuation_targets
                        }
                    ),
                    edge_to_V_arr=MappingProxyType(
                        {
                            (regime_name, target): _build_zero_V_arr(
                                topology=edge_topologies[regime_name, target]
                            )
                            for target in continuation_targets
                            if (regime_name, target) in edge_topologies
                        }
                    ),
                )
                arguments = _build_argmax_args(
                    regime=regime,
                    regime_params=regime_params,
                    ages=ages,
                    period=period,
                    n_subjects=n_subjects,
                    next_regime_to_V_arr=values,
                    regime_V_topology=regime_V_topology,
                    flat_params=flat_params,
                    subject_sharding=subject_sharding,
                )
                futures.add(
                    pool.submit(
                        _prepare_and_log,
                        executor=executor,
                        logger=logger,
                        program=program,
                        arguments=arguments,
                        period=period,
                        n_subjects=n_subjects,
                    )
                )
                del arguments, values
                _drain_compilations(futures=futures, max_pending=worker_count - 1)
            for family, builder in (
                (programs.transition, _build_next_state_args),
                (programs.route, _build_crtp_args),
            ):
                for period, program in family.items():
                    arguments = builder(
                        regime=regime,
                        regime_params=regime_params,
                        ages=ages,
                        n_subjects=n_subjects,
                        subject_sharding=subject_sharding,
                    )
                    arguments.update(period=jnp.int32(period), age=ages.values[period])
                    futures.add(
                        pool.submit(
                            _prepare_and_log,
                            executor=executor,
                            logger=logger,
                            program=program,
                            arguments=arguments,
                            period=period,
                            n_subjects=n_subjects,
                        )
                    )
                    del arguments
                    _drain_compilations(futures=futures, max_pending=worker_count - 1)
            _prepare_edge_gate_evaluators(
                regime=regime,
                regime_name=regime_name,
                regimes=regimes,
                flat_params=flat_params,
                ages=ages,
                n_subjects=n_subjects,
                regime_V_topology=regime_V_topology,
                subject_sharding=subject_sharding,
                subject_devices=subject_devices,
                prepared=prepared_gates,
                pool=pool,
                futures=futures,
                max_pending=worker_count - 1,
                logger=logger,
            )
        _drain_compilations(futures=futures)
    return regimes


def _prepare_and_log(
    *,
    executor: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[str, object],
    period: int,
    n_subjects: int,
    logger: logging.Logger,
) -> None:
    """Prepare one program in the bounded worker pool using transient arguments."""
    logger.info("  preparing %s ...", program.name)
    executor.prepare(
        program=program, arguments=arguments, period=period, n_subjects=n_subjects
    )


def _compile_and_install_gate(
    *,
    key: Hashable,
    low: jax.stages.Lowered,
    label: str,
    logger: logging.Logger,
    evaluator: Callable,
    axis_size: int,
) -> None:
    """Compile one host-owned gate and install its population specialization."""
    _, compiled = _compile_and_log(key=key, low=low, label=label, logger=logger)
    install_population_call(func=evaluator, axis_size=axis_size, call=compiled)


def _drain_compilations(*, futures: set[Future[None]], max_pending: int = 0) -> None:
    """Bound queued templates, draining every task before propagating a failure."""
    first_error: Exception | None = None
    while len(futures) > max_pending:
        future = next(as_completed(futures))
        try:
            future.result()
        except Exception as error:  # noqa: BLE001 - drain every compiler failure
            if first_error is None:
                first_error = error
            max_pending = 0
        finally:
            futures.remove(future)
    if first_error is not None:
        raise first_error


def _compile_and_log(
    *,
    key: Hashable,
    low: jax.stages.Lowered,
    label: str,
    logger: logging.Logger,
) -> tuple[Hashable, jax.stages.Compiled]:
    """Compile one lowered program, logging its duration."""
    logger.info("  compiling %s ...", label)
    start = time.monotonic()
    result = low.compile()
    logger.info(
        "  compiled  %s  %s",
        label,
        format_duration(seconds=time.monotonic() - start),
    )
    return key, result


def _prepare_edge_gate_evaluators(
    *,
    regime: Regime,
    regime_name: RegimeName,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    n_subjects: int,
    regime_V_topology: dict[RegimeName, _RegimeVTopology],
    subject_sharding: jax.sharding.Sharding,
    subject_devices: tuple[jax.Device, ...],
    prepared: set[Hashable],
    pool: ThreadPoolExecutor,
    futures: set[Future[None]],
    max_pending: int,
    logger: logging.Logger,
) -> None:
    """Lower each distinct gate, dropping its templates before bounded compilation.

    The router recomputes each edge's gate at the realized candidate target
    state, once per edge per period, through a population call it memoizes on
    the evaluator. Lowering that same population call here — against the same
    two kwarg pools, split by the same function the router splits them with —
    is what lets the compiled program be installed under the key the router
    then looks up.

    An evaluator is shared across the periods that group onto it, and its
    period and age enter as traced scalars rather than as static arguments,
    so those periods share one program.

    Only the fold periods the edge carries an evaluator for are lowered. An
    edge whose target is active over a narrower window than its source has no
    evaluator for the source's other landing periods, and the router never asks
    for one there: it skips an edge whose target is not folded at the period it
    routes against. Lowering what the router never calls would demand a program
    the edge has no value to build.
    """
    for target_name, edge in regime.gated_edges.items():
        by_period = edge.simulate_gate_evaluators_by_period
        for fold_period in _edge_fold_periods(regime=regime, ages=ages):
            # An empty mapping is a staging mistake, not a narrow window, and
            # the lookup below reports it as such; only a populated mapping
            # decides which periods to skip.
            if by_period and fold_period not in by_period:
                continue
            evaluator = edge.simulate_gate_evaluator_at(period=fold_period)
            key = ("gate", regime_name, target_name, _func_dedup_key(func=evaluator))
            if key in prepared:
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
            batched, shared = args
            args = (
                dict(
                    place_simulation_arguments(
                        arguments=batched,
                        subject_arg_names=tuple(batched),
                        value_reads=(),
                        devices=subject_devices,
                    )
                ),
                dict(
                    place_simulation_arguments(
                        arguments=shared,
                        subject_arg_names=(),
                        value_reads=(),
                        devices=subject_devices,
                    )
                ),
            )
            function = population_call(func=evaluator, axis_size=n_subjects)
            lowered = cast("jax.stages.Wrapped", function).lower(*args)
            del args, batched, shared
            futures.add(
                pool.submit(
                    _compile_and_install_gate,
                    key=key,
                    low=lowered,
                    label=(
                        f"{regime_name}/gate into {target_name} "
                        f"(age {ages.values[fold_period].item()})"
                    ),
                    logger=logger,
                    evaluator=evaluator,
                    axis_size=n_subjects,
                )
            )
            del lowered
            prepared.add(key)
            _drain_compilations(futures=futures, max_pending=max_pending)


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
    subject_sharding: jax.sharding.Sharding,
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
            provenance=evaluator.arg_provenance,  # ty: ignore[unresolved-attribute]
            flat_params=flat_params,
            source_name=source_name,
            target_name=target_name,
        ),
        **bind_edge_period_context(
            func=evaluator, fold_period=fold_period, fold_age=fold_age
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
        func=evaluator,
        batched_kwargs=candidate_target_states,
        static_kwargs=static_kwargs,
    )


def _subject_state_carrier_template(
    *,
    regime: Regime,
    n_subjects: int,
    sharding: jax.sharding.Sharding,
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
        arrays[state_name] = jax.device_put(zeros, sharding)
    return arrays


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
    subject_sharding: jax.sharding.Sharding,
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
        base_arrays=base.states, n_subjects=n_subjects, sharding=subject_sharding
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
    # A gated edge's projected operands, where this period's decision program
    # declares them. Zero templates suffice: lowering captures the abstract
    # value, and the runtime call supplies the solved arrays.
    edge_reference_regimes = regime.simulation.edge_reference_regimes_by_period.get(
        period
    )
    if edge_reference_regimes is not None:
        same_period_args[EDGE_REF_V_ARG] = MappingProxyType(
            {
                ref: _build_zero_V_arr(topology=regime_V_topology[ref])
                for ref in edge_reference_regimes
            }
        )
        same_period_args[EDGE_REF_PARAMS_ARG] = MappingProxyType(
            {ref: flat_params[ref] for ref in edge_reference_regimes}
        )
    taste_shock_kwargs = {}
    if regime.has_taste_shocks:
        _, taste_shock_keys = generate_simulation_keys(
            key=jax.random.key(0),
            names=["taste_shock"],
            n_initial_states=n_subjects,
        )
        taste_shock_kwargs = {"taste_shock_key": taste_shock_keys["key_taste_shock"]}
    return {
        **subject_states,
        **base.discrete_actions,
        **base.continuous_actions,
        **taste_shock_kwargs,
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
    subject_sharding: jax.sharding.Sharding,
) -> dict[str, object]:
    base = regime.solution.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(
        base_arrays=base.states, n_subjects=n_subjects, sharding=subject_sharding
    )
    # Simulate-only states (carried states declared via `Phased`)
    # are not solve grid axes, so they are absent from `state_action_space`. The
    # simulate `next_state` program carries and reads them, so seed each one.
    subject_states.update(
        _simulate_only_subject_states(
            regime=regime, n_subjects=n_subjects, sharding=subject_sharding
        )
    )
    subject_actions = _subject_shape_arrays(
        base_arrays={**base.discrete_actions, **base.continuous_actions},
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
    subject_sharding: jax.sharding.Sharding,
) -> dict[str, object]:
    base = regime.solution.state_action_space(regime_params=regime_params)
    subject_states = _subject_shape_arrays(
        base_arrays=base.states, n_subjects=n_subjects, sharding=subject_sharding
    )
    # The realized draw reads carried states as leaves, so the lower-args
    # must seed them like the next_state program's.
    simulate_only_states = _simulate_only_subject_states(
        regime=regime, n_subjects=n_subjects, sharding=subject_sharding
    )
    subject_actions = _subject_shape_arrays(
        base_arrays={**base.discrete_actions, **base.continuous_actions},
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
    *, regime: Regime, n_subjects: int, sharding: jax.sharding.Sharding
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
        arrays[name] = jax.device_put(zeros, sharding)
    return arrays


def _subject_shape_arrays(
    *,
    base_arrays: Mapping[str, FloatND | IntND],
    n_subjects: int,
    sharding: jax.sharding.Sharding,
) -> dict[str, FloatND | IntND]:
    """Return zeros of shape `(n_subjects,)` mirroring each base array's dtype.

    With `build_initial_states` casting discrete states to the grid dtype,
    runtime states (initial + post-transition) share the grid's dtype, so
    using `arr.dtype` from the regime's grid here matches runtime.

    `sharding` places the zeros exactly as `build_initial_states` places the
    runtime per-subject arrays — scattered across the device mesh when the
    regime distributes its grids, on the model's first device otherwise — so
    the AOT-compiled program is lowered for the device layout it is dispatched
    with.
    """
    arrays: dict[str, FloatND | IntND] = {}
    for name, arr in base_arrays.items():
        zeros = jnp.zeros((n_subjects,), dtype=arr.dtype)
        arrays[name] = jax.device_put(zeros, sharding)
    return arrays
