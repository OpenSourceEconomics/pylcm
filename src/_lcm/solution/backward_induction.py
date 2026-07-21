import functools
import gc
import inspect
import logging
import os
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.engine import Regime, StateActionSpace
from _lcm.regime_building.gated_edges import (
    build_reference_params_mapping_for_fold,
    build_same_period_mapping_for_fold,
)
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG, SAME_PERIOD_V_ARG
from _lcm.solution.contract import (
    BackwardInductionResult,
    ContinuationPayload,
    KernelResult,
    SimulationPolicy,
)
from _lcm.solution.diagnostics import (
    _emit_post_loop_diagnostics,
    _fold_period_diagnostics,
    _init_diagnostic_accumulators,
)
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
)
from _lcm.typing import FlatParams, RegimeName
from _lcm.utils.logging import (
    format_duration,
    log_period_header,
    log_period_timing,
    raise_or_warn,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND


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


def solve(  # noqa: C901, PLR0915
    *,
    flat_params: FlatParams,
    ages: AgeGrid,
    regimes: MappingProxyType[RegimeName, Regime],
    logger: logging.Logger,
    enable_jit: bool,
    max_compilation_workers: int | None = None,
) -> BackwardInductionResult:
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
        The named backward-induction outputs: the immutable mapping of periods
        to regime value-function arrays, the immutable mapping of periods to each
        regime's published simulation policy (the off-grid policy artifact
        simulation can interpolate; regimes whose kernels publish none have no
        entry), and the immutable mapping of periods to each COLLECTIVE regime's
        dissolution flag `D` — `True` on the state cells whose action mask is
        empty (E2), distinct from a numeric `-inf` value; empty inner mappings
        for models without collective regimes, so the default path only gains an
        empty dissolution mapping.

    """
    next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr = (
        _build_continuation_templates(regimes=regimes, flat_params=flat_params)
    )

    # AOT-compile all unique solve kernels in parallel.
    compiled_functions = _compile_all_functions(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        next_edge_to_V_arr=next_edge_to_V_arr,
        enable_jit=enable_jit,
        max_compilation_workers=max_compilation_workers,
        logger=logger,
    )

    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    simulation_policies: dict[int, MappingProxyType[RegimeName, SimulationPolicy]] = {}
    dissolution_flags: dict[int, MappingProxyType[RegimeName, BoolND]] = {}

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
    _reject_edge_fold_state_param_collisions(
        regimes=regimes,
        base_state_action_spaces=base_state_action_spaces,
        flat_params=flat_params,
    )

    # A published simulation policy is a solve *output*, accumulated for
    # every period; no backward step reads it. Its buffers can alias the
    # period's continuation buffer, so leaving policies on device pins one
    # continuation-sized buffer per period for the whole induction. Evict each
    # period's policies to host as they are produced; simulation
    # re-materializes them on device.
    host_device = jax.devices("cpu")[0]

    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}
        period_continuations: dict[RegimeName, ContinuationPayload] = {}
        period_simulation_policies: dict[RegimeName, SimulationPolicy] = {}
        period_dissolution_flags: dict[RegimeName, BoolND] = {}

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

        # COLLECTIVE-REGIMES (E2): regimes declaring `same_period_refs` read
        # other regimes' V of THIS period, so those references must be solved
        # first — order the period's active regimes topologically by the
        # reference edges (stable: dict order among independent regimes).
        # Models without references keep the plain dict order.
        for regime_name in _order_regime_names_by_same_period_refs(
            active_regimes=active_regimes
        ):
            regime = active_regimes[regime_name]
            result = _run_period_kernel(
                regime=regime,
                regime_name=regime_name,
                period=period,
                compiled_cores=compiled_functions[(regime_name, period)],
                state_action_space=base_state_action_spaces[regime_name],
                flat_params=flat_params,
                ages=ages,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                next_edge_to_V_arr=next_edge_to_V_arr,
                period_solution=period_solution,
            )
            V_arr = result.V_arr
            # The published V mapping is the calling convention for every
            # downstream consumer — the parents' cores and the AOT-lowered
            # simulate programs are both compiled against the per-regime V
            # topology — so a kernel output arriving with a different
            # sharding (the compiled program's output sharding is the
            # backend's choice) is placed back on the template's mesh here.
            V_arr = _match_leaf_template_sharding(
                leaf=V_arr,
                template_leaf=next_regime_to_V_arr[regime_name],
            )
            _fail_if_continuation_publisher_returned_none(
                result=result,
                regime_name=regime_name,
                period=period,
                continuation_publishers=next_regime_to_continuation,
            )
            if result.continuation is not None:
                period_continuations[regime_name] = result.continuation
            if result.simulation_policy is not None:
                period_simulation_policies[regime_name] = result.simulation_policy
            # COLLECTIVE-REGIMES (E2): a collective regime publishes its
            # empty-mask dissolution flag D alongside V; singleton regimes
            # leave it None and never touch this mapping.
            if result.dissolution is not None:
                period_dissolution_flags[regime_name] = result.dissolution
            running_any_nan, running_any_inf = _fold_period_diagnostics(
                V_arr=V_arr,
                regime_name=regime_name,
                period=period,
                ages=ages,
                diagnostics_enabled=diagnostics_enabled,
                stats_enabled=stats_enabled,
                diagnostic_rows=diagnostic_rows,
                diagnostic_min=diagnostic_min,
                diagnostic_max=diagnostic_max,
                diagnostic_mean=diagnostic_mean,
                running_any_nan=running_any_nan,
                running_any_inf=running_any_inf,
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

        # COLLECTIVE-REGIMES (E3'): fold each declared gated edge whose target
        # was solved this period onto the target grid, and roll the resulting
        # Wbar into the edge continuation the source reads next period. Reads
        # only the still-live period-t arrays (`period_solution`,
        # `period_dissolution_flags`). The node fold is streamed to cap peak
        # memory; parents then read Wbar in place of the raw target V via the
        # existing next_regime_to_V_arr threading.
        next_edge_to_V_arr = _roll_gated_edges(
            regimes=regimes,
            period_solution=period_solution,
            period_dissolution_flags=period_dissolution_flags,
            base_state_action_spaces=base_state_action_spaces,
            flat_params=flat_params,
            next_edge_to_V_arr=next_edge_to_V_arr,
        )
        next_regime_to_V_arr, next_regime_to_continuation = _roll_continuation_inputs(
            regimes=regimes,
            period_solution=period_solution,
            period_continuations=period_continuations,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
        )
        solution[period] = MappingProxyType(period_solution)
        # COLLECTIVE-REGIMES (E2): publish each collective regime's dissolution
        # flag D alongside V. Kept as a plain per-period mapping (not rolled
        # like `next_regime_to_V_arr`): nothing consumes a NEXT-period D — the
        # E3' gates read the still-live per-period flags at each period's end,
        # before the roll (above).
        dissolution_flags[period] = MappingProxyType(period_dissolution_flags)
        simulation_policies[period] = MappingProxyType(
            {
                regime_name: jax.block_until_ready(
                    jax.device_put(simulation_policy, host_device)
                )
                for regime_name, simulation_policy in period_simulation_policies.items()
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

        _release_rolled_continuations(period_continuations=period_continuations)

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

    _drain_V_arr_shards(solution=solution, dissolution_flags=dissolution_flags)

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return BackwardInductionResult(
        value_functions=MappingProxyType(solution),
        simulation_policies=MappingProxyType(simulation_policies),
        dissolution_flags=MappingProxyType(dissolution_flags),
    )


def _release_rolled_continuations(
    *, period_continuations: dict[RegimeName, ContinuationPayload]
) -> None:
    """Free the device buffers rolled off the period just solved.

    The superseded continuation inputs and the period's transient working set
    are unreferenced once the period rolls, but a rolled continuation payload
    sits in a registered pytree that CPython's cyclic collector frees only when
    it next runs — forcing a collection here frees the device pool promptly,
    capping peak resident across the loop (mirrors the forward-sim memory
    rework in `result.py`).

    Gated on whether this period actually produced a continuation (the generic
    per-period kernel output the loop already tracks), not on the solver type:
    a period whose kernels publish none rolls no such buffer, so the collection
    — which otherwise dominates small warm solves with no memory gain — is
    skipped for it.
    """
    if period_continuations:
        gc.collect()


def _run_period_kernel(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    compiled_cores: MappingProxyType[str, Callable],
    state_action_space: StateActionSpace,
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    period_solution: Mapping[RegimeName, FloatND],
) -> KernelResult:
    """Invoke one regime's period adapter for one period.

    Every regime exposes the same kind of adapter; the loop never branches on
    solver type. The adapter wraps the regime's shared jitted core(s) (passed in
    AOT-compiled as `compiled_cores`), calls them with the solver's own argument
    layout, and returns a `KernelResult` — the value-function array plus the
    optional generic outputs (`continuation`, `simulation_policy`, and the
    collective `dissolution` flag D), which the backward-induction loop
    accumulates.

    COLLECTIVE-REGIMES (E2): a regime declaring `same_period_refs` additionally
    receives the referenced regimes' V arrays of THIS period, read off
    `period_solution` — the within-period topological order guarantees they were
    solved earlier in this period's loop. COLLECTIVE-REGIMES (E3'): a source
    declaring `gated_edges` receives its own rolled Wbar arrays, keyed by target
    regime name, which the grid-search kernel substitutes for the raw target V in
    `next_regime_to_V_arr`. Every other regime's adapter is called with the
    unchanged uniform signature.

    `period`/`age` are passed as JAX arrays (not Python scalars) so a shared
    `jax.jit` function is traced once with abstract shapes, not recompiled
    for every distinct (period, age) pair.

    The adapter is handed its full per-key compiled-core map (`compiled_cores`):
    a single-core kernel reads `["main"]`, a multi-core kernel reads each of its
    own core keys.

    Returns:
        The kernel's result for this regime-period.

    """
    period_kernel = regime.solution.period_kernels[period]
    same_period_kwargs: dict[str, object] = {}
    if regime.same_period_ref_regimes:
        same_period_kwargs["same_period_regime_to_V_arr"] = MappingProxyType(
            {
                ref_regime_name: period_solution[ref_regime_name]
                for ref_regime_name in regime.same_period_ref_regimes
            }
        )
    if regime.gated_edges:
        # COLLECTIVE-REGIMES (E3'): hand the source its own rolled Wbar arrays,
        # keyed by target regime name; the grid-search kernel substitutes them
        # for the raw target V in `next_regime_to_V_arr`.
        same_period_kwargs["edge_regime_to_V_arr"] = MappingProxyType(
            {
                target_name: next_edge_to_V_arr[(regime_name, target_name)]
                for target_name in regime.gated_edges
            }
        )
    return period_kernel(
        compiled_cores=compiled_cores,
        state_action_space=state_action_space,
        next_regime_to_V_arr=next_regime_to_V_arr,
        next_regime_to_continuation=next_regime_to_continuation,
        flat_params=flat_params,
        period=period,
        ages=ages,
        **same_period_kwargs,
    )


def _order_regime_names_by_same_period_refs(
    *,
    active_regimes: dict[RegimeName, Regime],
) -> tuple[RegimeName, ...]:
    """Topologically order one period's active regimes by `same_period_refs`.

    COLLECTIVE-REGIMES (E2). A regime reading another regime's same-period V
    must be solved after it. Stable Kahn ordering: at each step the first (in
    dict order) not-yet-placed regime whose active references are all placed is
    emitted, so models without references keep the plain dict order exactly. A
    cycle is rejected at model build (`_fail_if_same_period_ref_cycle`); the
    raise here is a defensive backstop for direct engine callers.
    """
    if not any(regime.same_period_ref_regimes for regime in active_regimes.values()):
        return tuple(active_regimes)
    placed: dict[RegimeName, None] = {}
    remaining = dict(active_regimes)
    while remaining:
        ready = next(
            (
                regime_name
                for regime_name, regime in remaining.items()
                if all(ref not in remaining for ref in regime.same_period_ref_regimes)
            ),
            None,
        )
        if ready is None:
            msg = (
                "same_period_refs form a cycle among the period's active "
                f"regimes: {sorted(remaining)}. This should have been "
                "rejected at model build."
            )
            raise RuntimeError(msg)
        placed[ready] = None
        del remaining[ready]
    return tuple(placed)


def _roll_continuation_inputs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    period_solution: dict[RegimeName, FloatND],
    period_continuations: dict[RegimeName, ContinuationPayload],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
]:
    """Roll the per-period continuation mappings forward by one period.

    Both mappings keep their full template key sets — V for every regime,
    carries for every carry-producing regime — and update only the entries
    solved this period, so the pytree structure stays JIT-stable.

    The `.get(..., prior)` fallback is for regimes *inactive* this period: they
    keep the prior period's entry. It relies on the invariant that every
    continuation-publishing regime publishes on each of its active periods — the
    solve loop enforces this before rolling, so an active publisher can never
    fall through to the stale prior carry here.

    Returns:
        Tuple of the rolled V mapping and the rolled carry mapping.

    """
    rolled_V_arr = MappingProxyType(
        {
            regime_name: _match_leaf_template_sharding(
                leaf=period_solution[regime_name],
                template_leaf=next_regime_to_V_arr[regime_name],
            )
            if regime_name in period_solution
            else next_regime_to_V_arr[regime_name]
            for regime_name in regimes
        }
    )
    rolled_continuation = MappingProxyType(
        {
            regime_name: _match_continuation_template_sharding(
                continuation=period_continuations[regime_name],
                template=next_regime_to_continuation[regime_name],
            )
            if regime_name in period_continuations
            else next_regime_to_continuation[regime_name]
            for regime_name in next_regime_to_continuation
        }
    )
    return rolled_V_arr, rolled_continuation


# COLLECTIVE-REGIMES (E3'): a gated edge's continuation slot is keyed by the
# (source regime, target regime) pair — a source has at most one edge per target,
# and the same target is read raw by other regimes, so the edge cannot share the
# plain regime-keyed V slot.
type _EdgeKey = tuple[RegimeName, RegimeName]


def _roll_gated_edges(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    period_solution: dict[RegimeName, FloatND],
    period_dissolution_flags: dict[RegimeName, BoolND],
    base_state_action_spaces: dict[RegimeName, StateActionSpace],
    flat_params: FlatParams,
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
) -> MappingProxyType[_EdgeKey, FloatND]:
    """Fold every gated edge whose target was solved this period; roll the rest.

    COLLECTIVE-REGIMES (E3'). For each declared edge whose target regime (and
    every reference regime it reads) was solved in the period just completed,
    evaluate its ``Wbar`` producer on the still-live period-``t`` arrays and
    store it; edges whose target is inactive this period keep their previous
    ``Wbar`` (the roll semantics of `next_regime_to_V_arr`). Keeps the full key
    set so the pytree structure stays JIT-stable.
    """
    if not next_edge_to_V_arr:
        return next_edge_to_V_arr
    rolled: dict[_EdgeKey, FloatND] = dict(next_edge_to_V_arr)
    for source_name, source in regimes.items():
        for target_name, edge in source.gated_edges.items():
            if target_name not in period_solution:
                continue
            if any(ref not in period_solution for ref in edge.reference_regimes):
                # COLLECTIVE-REGIMES (E3', F4). The target was solved this period
                # but a reference regime it reads was not — so its same-period V
                # does not exist and this edge's Wbar cannot be folded here. This
                # is reached ONLY at an unconsumed boundary: the construction-time
                # guard `_fail_if_gated_edge_references_inactive` rejects any model
                # where a reference is absent in a period whose Wbar is actually
                # consumed (target active at t AND source active at t-1). What
                # survives to here is a target-active period with no source one
                # period earlier (e.g. a self-loop edge at the target's earliest
                # active period), whose rolled Wbar is never read — so keeping the
                # previous value is correct, not a stale-value bug. (Simulate
                # tolerates the same boundary; see
                # test_repeating_self_loop_gated_edge_simulates_past_activity_boundary.)
                continue
            fold = source.gated_edge_folds[target_name]
            same_period_mapping = build_same_period_mapping_for_fold(
                edge=edge,
                period_solution=period_solution,
                period_dissolution_flags=period_dissolution_flags,
            )
            wbar = _evaluate_edge_fold(
                fold=fold,
                target_states=base_state_action_spaces[target_name].states,
                same_period_mapping=same_period_mapping,
                source_flat_params=flat_params[source_name],
                reference_flat_params=build_reference_params_mapping_for_fold(
                    edge=edge, flat_params=flat_params
                ),
            )
            rolled[(source_name, target_name)] = wbar
    return MappingProxyType(rolled)


def _reject_edge_fold_state_param_collisions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace],
    flat_params: FlatParams,
) -> None:
    """Reject a gated edge whose fold binds one leaf as BOTH a target state and a
    source param (simulate-round8 F1).

    A gate / gate-ref projection / fallback projection declares its arguments by
    bare name. `get_edge_fold` exposes the target's state grids and the source's
    gate/projection params in ONE flat signature, so a name that is simultaneously
    a TARGET STATE of the target regime and a key of `flat_params[source]` occupies
    a single leaf that two binders both claim: `_evaluate_edge_fold` (below)
    overwrites the state grid with the source param, so the SOLVE-side ``Wbar``
    reads the param, while the simulate evaluator's ``_expose``
    (`get_edge_simulate_gate_evaluator`) classifies the same name as a state
    BEFORE it would record a source param, so the SIMULATE-side gate reads the
    realized target state. Solve and simulate then evaluate DIFFERENT predicates
    for the same edge -- the gate flips, ``Wbar`` changes, or a fallback
    coordinate is written from the wrong value, all silently.

    Why this is a solve-time (not construction-time) fence: a gate/projection
    param is bound from a BARE key the user adds to `flat_params[source]`, never
    from the function-qualified regime params template, so it is absent from
    `regime_to_flat_param_names[source]` and the collision is only visible once
    `flat_params` is in hand. A LEGITIMATE direct target-state read (a gate that
    reads a target state the source never supplies as a param -- e.g. a reused
    state NAME across two regimes) is untouched, because that name is not a key of
    `flat_params[source]`.
    """
    for source_name, source in regimes.items():
        if not source.gated_edges:
            continue
        source_param_names = set(flat_params[source_name])
        for target_name in source.gated_edges:
            fold = source.gated_edge_folds[target_name]
            sig_params = set(inspect.signature(fold).parameters)
            target_state_names = set(base_state_action_spaces[target_name].states)
            collisions = sorted(sig_params & target_state_names & source_param_names)
            if collisions:
                msg = (
                    f"The gated edge '{source_name}' -> '{target_name}' has a gate "
                    f"or projection argument {collisions} that is simultaneously a "
                    f"TARGET state of '{target_name}' and a source parameter in "
                    f"`flat_params['{source_name}']`. The fold's single leaf for "
                    "each such name is bound as the source param on the solve side "
                    "(`_evaluate_edge_fold`) but as the realized target state on the "
                    "simulate side (`get_edge_simulate_gate_evaluator`), so the "
                    "solved `Wbar` and the simulate router would evaluate different "
                    "gates. Rename the source parameter (or the target state) so the "
                    "two namespaces are disjoint."
                )
                raise ModelInitializationError(msg)
            # A source flat-param key (or target state) that shadows one of the
            # internal ENGINE argument names is a second solve/simulate divergence
            # of the same class (simulate-round9 F1): on the solve side
            # `_evaluate_edge_fold` binds `SAME_PERIOD_V_ARG` to the value mapping
            # and `SAME_PERIOD_PARAMS_ARG` to the reference params, then overwrites
            # those slots with any same-named source flat-param; on the simulate
            # side `_expose` classifies the identical spelling as the engine arg
            # BEFORE it can be recorded as a source param. So a source scalar named
            # `same_period_regime_to_params` opens the gate on solve (scalar) but
            # closes it on simulate (mapping), and a source
            # `same_period_regime_to_V_arr` overwrites the value mapping outright.
            # Reserve the engine names against
            # both source params and target states, restricted to the names actually
            # present in this fold's signature.
            engine_args = {SAME_PERIOD_V_ARG, SAME_PERIOD_PARAMS_ARG}
            engine_collisions = sorted(
                sig_params & engine_args & (source_param_names | target_state_names)
            )
            if engine_collisions:
                msg = (
                    f"The gated edge '{source_name}' -> '{target_name}' has a gate "
                    f"or projection argument {engine_collisions} that shadows a "
                    "reserved internal engine argument name "
                    f"({sorted(engine_args)}). Such a name is bound as the source "
                    "parameter / target state on one side of the solve/simulate seam "
                    "but as the engine's value/params mapping on the other, so the "
                    "solved `Wbar` and the simulate router would evaluate different "
                    "gates (or crash when a source value overwrites the value "
                    "mapping). Rename the source parameter (or the target state) so "
                    "it does not collide with a reserved engine argument."
                )
                raise ModelInitializationError(msg)


def _evaluate_edge_fold(
    *,
    fold: Callable,
    target_states: Mapping[str, ContinuousState | DiscreteState],
    same_period_mapping: Mapping[RegimeName, FloatND],
    source_flat_params: Mapping[str, object],
    reference_flat_params: Mapping[RegimeName, Mapping[str, object]],
) -> FloatND:
    """Call one edge's fold with exactly the arguments its signature declares.

    Every parameter the fold needs is bound from the SOURCE regime — the fold is
    the source's own continuation object, and its gate / projections are declared
    on the source, so this is the namespace they are written against. (It is also
    the contract the simulate-side gate evaluator and leg projectors must match
    argument for argument; see `_lcm.regime_building.gated_edges
    .EdgeArgProvenance`.) The one exception is a REFERENCE regime's own
    interpolation grid, which belongs to neither the source nor the target:
    those params ride in `reference_flat_params` under
    `Q_and_F.SAME_PERIOD_PARAMS_ARG`, keyed by regime, and the reference readers
    resolve them internally (F4).

    COLLECTIVE-REGIMES (E3', slice 5): the target regime's grid may carry
    DISCRETE state axes (e.g. EKL's encoded spouse-type categorical, or any
    other `DiscreteGrid` state) alongside continuous ones — `target_states`
    is `base_state_action_spaces[target_name].states`, whose value type is
    `ContinuousState | DiscreteState` at the source (`_lcm.engine.
    StateActionSpace.states`), not float-only. Slices 1-4's gated-edge tests
    never exercised a discrete target-grid axis, so this signature under-
    typed the mapping to `FloatND` only; discrete states hit a
    `BeartypeCallHintParamViolation` at the `int32`-vs-float check inside
    `fold` (`get_edge_fold`'s `jnp.meshgrid` state broadcast tolerates either
    dtype — the guard was purely a stale type hint).
    """
    sig_params = set(inspect.signature(fold).parameters)
    kwargs: dict[str, object] = {
        name: arr for name, arr in target_states.items() if name in sig_params
    }
    kwargs[SAME_PERIOD_V_ARG] = same_period_mapping
    if SAME_PERIOD_PARAMS_ARG in sig_params:
        kwargs[SAME_PERIOD_PARAMS_ARG] = reference_flat_params
    kwargs.update(
        {
            name: value
            for name, value in source_flat_params.items()
            if name in sig_params
        }
    )
    return fold(**kwargs)


def _match_continuation_template_sharding(
    *, continuation: ContinuationPayload, template: ContinuationPayload
) -> ContinuationPayload:
    """Place a solved period's continuation on its template's device sharding.

    The parent's cores are AOT-compiled against the continuation template, so
    the template's per-leaf sharding is the calling convention. A producer can
    emit mixed-sharding leaves (value rows derived from the sharded value
    array, endogenous-grid rows broadcast replicated from the asset grid);
    every leaf is placed onto its template counterpart's sharding, a no-op
    where they already agree. Assumes the template of a distributed regime is
    itself sharded — an unsharded template under a distributed state would
    pull the continuation onto one device.
    """
    return jax.tree.map(
        lambda leaf, template_leaf: _match_leaf_template_sharding(
            leaf=leaf, template_leaf=template_leaf
        ),
        continuation,
        template,
    )


def _match_leaf_template_sharding(*, leaf: FloatND, template_leaf: FloatND) -> FloatND:
    """Place one solved array on its template's device sharding (no-op on match).

    Applied where a solved value array is published and where the continuation
    mappings roll forward, for the same reason as the continuations: a compiled
    kernel's output sharding is the backend's choice, so a value array can
    arrive replicated while the templates every consumer (parent cores and the
    AOT-lowered simulate programs) was lowered against are sharded.
    """
    if leaf.sharding == template_leaf.sharding:
        return leaf
    return jax.device_put(leaf, template_leaf.sharding)


def _fail_if_continuation_publisher_returned_none(
    *,
    result: KernelResult,
    regime_name: RegimeName,
    period: int,
    continuation_publishers: Mapping[RegimeName, ContinuationPayload],
) -> None:
    """Fail loud if a continuation-publishing regime published nothing.

    A regime with a continuation template MUST publish a continuation on every
    active period. If its kernel returns None, `_roll_continuation_inputs` would
    silently roll the stale prior period's carry forward — wrong numbers, not a
    crash — so surface the offending (regime, period) instead.
    """
    if result.continuation is None and regime_name in continuation_publishers:
        msg = (
            f"Regime '{regime_name}' declares a continuation template but its "
            f"kernel returned no continuation in active period {period}. A "
            f"continuation-based solver must publish a continuation on every "
            f"active period."
        )
        raise RuntimeError(msg)


def _build_continuation_templates(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, ContinuationPayload],
    MappingProxyType[_EdgeKey, FloatND],
]:
    """Build the period-invariant continuation-input templates.

    All mappings keep the same pytree structure (keys and shapes) across all
    periods, avoiding JIT re-compilation from pytree mismatches:

    - the V template holds a zero array per regime, shaped (and sharded) like
      the regime's V array;
    - the continuation template holds entries only for continuation-publishing
      regimes, in the key order reused every period;
    - the gated-edge (E3') template holds a zero ``Wbar`` per declared edge,
      shaped like the target regime's V state grid plus the source regime's
      stakeholder axis (a singleton source: the target grid alone). Empty for
      models without gated edges, so the default path only gains an empty third
      mapping.
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
    next_regime_to_continuation = MappingProxyType(
        {
            regime_name: regime.solution.continuation_template
            for regime_name, regime in regimes.items()
            if regime.solution.continuation_template is not None
        }
    )
    next_edge_to_V_arr = MappingProxyType(
        {
            (source_name, target_name): jnp.zeros(shape)
            for source_name, target_name, shape in _iter_edge_shapes(
                regimes=regimes, flat_params=flat_params
            )
        }
    )
    return next_regime_to_V_arr, next_regime_to_continuation, next_edge_to_V_arr


def _edge_lower_kwargs(
    *,
    regime: Regime,
    regime_name: RegimeName,
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
) -> dict[str, object]:
    """Lowering kwargs for a source kernel's gated-edge Wbar templates (E3')."""
    if not regime.gated_edges:
        return {}
    return {
        "edge_regime_to_V_arr": MappingProxyType(
            {
                target_name: next_edge_to_V_arr[(regime_name, target_name)]
                for target_name in regime.gated_edges
            }
        )
    }


def _iter_edge_shapes(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> Iterator[tuple[RegimeName, RegimeName, tuple[int, ...]]]:
    """Yield ``(source, target, Wbar_shape)`` for every declared gated edge (E3')."""
    for source_name, source in regimes.items():
        if not source.gated_edges:
            continue
        for target_name in source.gated_edges:
            target = regimes[target_name]
            target_states = target.solution.state_action_space(
                regime_params=flat_params[target_name]
            ).states
            shape = tuple(len(v) for v in target_states.values())
            if source.stakeholders is not None:
                shape = (*shape, len(source.stakeholders))
            yield source_name, target_name, shape


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
    dissolution_flags: dict[int, MappingProxyType[RegimeName, BoolND]] | None = None,
) -> None:
    """Block until every V_arr (and dissolution-flag) shard is materialised.

    Solve → simulate barrier: backward induction returns sharded V_arrs,
    but the simulate phase must consume materialised arrays rather than
    in-flight kernels. `jax.block_until_ready` walks the pytree of V_arrs
    and blocks per-shard (no host transfer, no cross-device collective);
    free when kernels are already done, the minimum necessary sync when
    they are not. V stays sharded across devices. The collective dissolution
    flags (E2) ride along in the same barrier.
    """
    jax.block_until_ready((solution, dissolution_flags))


def _compile_all_functions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
    next_edge_to_V_arr: MappingProxyType[_EdgeKey, FloatND],
    enable_jit: bool,
    max_compilation_workers: int | None,
    logger: logging.Logger,
) -> dict[tuple[RegimeName, int], MappingProxyType[str, Callable]]:
    """AOT-compile all unique solve cores in parallel.

    Each regime exposes one period adapter per period; the adapter wraps one or
    more shared jitted cores, keyed by a stable per-kernel name (`cores()`).
    Most kernels carry a single `"main"` core; a multi-core kernel carries
    several named cores, each a distinct traced program. Many
    periods share the same core object, so this deduplicates the cores by
    identity, lowers each unique core once (sequential — tracing is
    single-threaded) with the adapter's per-key lowering arguments, then compiles
    the XLA programs in parallel via a thread pool (XLA releases the GIL during
    compilation). The loop stays free of any solver-type fork.

    When JIT is disabled (`enable_jit=False`), returns the raw cores without
    compilation.

    Args:
        regimes: The internal regimes containing the period adapters.
        flat_params: Regime parameters for constructing lowering args.
        ages: Age grid for the model.
        next_regime_to_V_arr: Template with consistent keys and V array shapes
            for constructing lowering arguments.
        next_regime_to_continuation: Template with consistent keys and carry
            shapes for constructing lowering arguments.
        next_edge_to_V_arr: Template with consistent keys and ``Wbar`` shapes
            for constructing a source kernel's gated-edge lowering arguments
            (E3'); empty for models without gated edges.
        enable_jit: Whether to JIT-compile the functions of the internal regimes.
        max_compilation_workers: Maximum threads for parallel compilation.
            Defaults to `os.cpu_count()`.
        logger: Logger for compilation progress.

    Returns:
        Dict of (regime_name, period) to the immutable mapping of core key to
        compiled (or raw) core.

    """
    # Collect all (regime, period, core_key) -> shared jitted core mappings off
    # the period adapters.
    all_functions: dict[tuple[RegimeName, int, str], Callable] = {}
    for regime_name, regime in regimes.items():
        for period in regime.active_periods:
            cores = regime.solution.period_kernels[period].cores()
            for core_key, core in cores.items():
                all_functions[(regime_name, period, core_key)] = core

    # If JIT is disabled, return the raw cores keyed by core_key per (regime,
    # period).
    if not enable_jit:
        return _group_cores_by_regime_period(all_functions)

    # Deduplicate by identity (or by underlying function for partials), keeping
    # one representative (regime, period, core_key) per unique core so its
    # adapter can build the lowering arguments for that key.
    unique: dict[Hashable, tuple[Callable, RegimeName, int, str]] = {}
    for (regime_name, period, core_key), func in all_functions.items():
        func_id = _func_dedup_key(func=func)
        if func_id not in unique:
            unique[func_id] = (func, regime_name, period, core_key)

    n_workers = _resolve_compilation_workers(
        max_compilation_workers=max_compilation_workers
    )
    n_unique = len(unique)

    logger.info(
        "AOT compilation: %d unique functions (%d regime-period-core triples, "
        "%d workers)",
        n_unique,
        len(all_functions),
        n_workers,
    )

    # Phase 1: Lower all unique cores (sequential — tracing is not thread-safe
    # and must happen on the main thread). Each adapter builds the named core's
    # lowering arguments off a fresh, params-completed state-action space.
    lowered: dict[Hashable, jax.stages.Lowered] = {}
    labels: dict[Hashable, str] = {}
    for i, (func_id, (func, regime_name, period, core_key)) in enumerate(
        unique.items(), 1
    ):
        regime = regimes[regime_name]
        edge_kwargs = _edge_lower_kwargs(
            regime=regime,
            regime_name=regime_name,
            next_edge_to_V_arr=next_edge_to_V_arr,
        )
        lower_args = regime.solution.period_kernels[period].build_lower_args(
            core_key=core_key,
            state_action_space=regime.solution.state_action_space(
                regime_params=flat_params[regime_name],
            ),
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            **edge_kwargs,
        )
        label = f"{regime_name} {core_key} (age {ages.values[period].item()})"
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

    # Map back to (regime, period) keys, grouping the compiled cores by core key.
    return _group_cores_by_regime_period(
        {
            key: compiled[_func_dedup_key(func=func)]
            for key, func in all_functions.items()
        }
    )


def _group_cores_by_regime_period(
    cores_by_triple: dict[tuple[RegimeName, int, str], Callable],
) -> dict[tuple[RegimeName, int], MappingProxyType[str, Callable]]:
    """Group (regime, period, core_key) -> core into (regime, period) -> {key: core}.

    The solve loop dispatches each period adapter with its full per-key core map,
    so a multi-core kernel receives all its compiled cores while a single-core
    kernel receives `{"main": ...}`.
    """
    grouped: dict[tuple[RegimeName, int], dict[str, Callable]] = {}
    for (regime_name, period, core_key), core in cores_by_triple.items():
        grouped.setdefault((regime_name, period), {})[core_key] = core
    return {key: MappingProxyType(cores) for key, cores in grouped.items()}


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
