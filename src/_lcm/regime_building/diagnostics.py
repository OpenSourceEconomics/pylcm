"""Per-period diagnostic closures and feasibility-conditional reductions.

Cold-path machinery used only when `validate_V` detects NaN in a solved
value-function array. `_build_compute_intermediates_per_period` produces
one JIT-compiled closure per period that productmaps
`get_compute_intermediates` over the full state-action space and fuses
the compute step with on-device reductions (`_wrap_with_reduction`).
The fused output is consumed by `_enrich_with_diagnostics` in
`_lcm.utils.error_handling`.
"""

from collections.abc import Callable, Hashable
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.engine import StateActionSpace
from _lcm.grids import Grid
from _lcm.reachability import PhaseReachability
from _lcm.regime_building.age_normalization import (
    AgeGridSchedule,
    continuation_group_key,
    continuation_info_lookup,
    expand_groups_to_periods,
    group_periods_by_key,
    resolve_periodized_nodes,
)
from _lcm.regime_building.Q_and_F import get_compute_intermediates
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.dispatchers import productmap
from lcm.typing import BoolND, FloatND, IntND


def _build_compute_intermediates_per_period(
    *,
    active_periods: tuple[int, ...],
    flat_param_names: frozenset[str],
    phase_reachability: PhaseReachability,
    source_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    grids: MappingProxyType[StateOrActionName, Grid],
    enable_jit: bool,
    certainty_equivalent: CertaintyEquivalent | None,
    grid_schedule: AgeGridSchedule | None = None,
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
) -> MappingProxyType[int, Callable]:
    """Build diagnostic intermediate closures for each period of a non-terminal regime.

    Each closure fuses a productmap over the full state-action space with
    on-device reductions (matching the `max_Q_over_a` productmap pattern)
    and is JIT-compiled. Periods sharing the same target configuration
    reuse a single scalar closure. The caller is responsible for handling
    terminal regimes. Used in the error path when `validate_V` detects NaN.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.
        functions: Immutable mapping of internal user functions.
        constraints: Immutable mapping of constraint functions.
        transitions: Immutable mapping of regime-to-regime transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for the current regime.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        state_action_space: State-action space used for productmap sizing.
        grids: Immutable mapping of state/action names to grid specs; used
            for per-state batch sizes.
        enable_jit: Whether to JIT-compile the fused closure.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.

    Returns:
        Immutable mapping of period index to fused closure.

    """
    state_batch_sizes = {
        name: grid.batch_size
        for name, grid in grids.items()
        if name in state_action_space.state_names
    }

    # `continuation_info` mirrors `_build_Q_and_F_per_period.continuation_info` so a
    # NaN diagnostic recomputes intermediates on the *same* period-specific target
    # grid the primary solve used, not the representative grid. `group_key` mirrors
    # `_build_Q_and_F_per_period.group_key`'s grouping.
    continuation_info = continuation_info_lookup(
        period_to_regime_v_interp=period_to_regime_v_interp,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    group_key = continuation_group_key(
        phase_reachability=phase_reachability,
        source_regime_name=source_regime_name,
        functions=functions,
        constraints=constraints,
        grid_schedule=grid_schedule,
    )

    configs = group_periods_by_key(active_periods, group_key)

    variable_names = (
        *state_action_space.state_names,
        *state_action_space.action_names,
    )
    built: dict[tuple[tuple[RegimeName, ...], Hashable], Callable] = {}
    for key, periods in configs.items():
        period_targets = key[0]
        representative_period = periods[0]
        scalar = get_compute_intermediates(
            flat_param_names=flat_param_names,
            functions=cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(functions, representative_period),
            ),
            constraints=cast(
                "ConstraintFunctionsMapping",
                resolve_periodized_nodes(constraints, representative_period),
            ),
            period_targets=period_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=continuation_info(representative_period),
            certainty_equivalent=certainty_equivalent,
            # The diagnostics are handed the full value arrays and map over
            # every state, so none of the solve kernel's co-mapped axes have
            # been sliced off here.
            co_map_state_names=(),
        )
        mapped = _productmap_over_state_action_space(
            func=scalar,
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
            state_batch_sizes=state_batch_sizes,
        )
        fused = _wrap_with_reduction(func=mapped, variable_names=variable_names)
        built[key] = jax.jit(fused) if enable_jit else fused

    return expand_groups_to_periods(configs, built)


def _wrap_with_reduction(
    *,
    func: Callable,
    variable_names: tuple[str, ...],
) -> Callable:
    """Fuse a productmap'd intermediates function with on-device reductions.

    The wrapped function returns a flat pytree of scalars and per-dimension
    vectors instead of full state-action-shaped arrays. When JIT-compiled,
    XLA can often fuse the compute and reduce steps so the full-shape
    intermediates never materialise.

    Args:
        func: Productmap'd closure returning
            `(U_arr, F_arr, CE, Q_arr, regime_probs)`. `regime_probs`
            is a mapping of target regime names to per-point probability
            arrays.
        variable_names: Tuple of state + action names in the order that
            matches the productmap axes of `func`. Used to label the
            `{metric}_by_{name}` reductions.

    Returns:
        Callable taking the same kwargs as `func` and returning a dict with
        `{Y}_overall` scalars and `{Y}_by_{name}` vectors for `Y` in
        {`U_nan`, `CE_nan`, `Q_nan`, `F_feasible`}, plus `regime_probs` as
        a dict of per-target scalar means. The `{U,CE,Q}_nan_*` fractions
        are conditional on feasibility (numerator restricted to feasible
        cells, denominator is the feasible-cell count); `F_feasible_*`
        is the plain mean over all cells.

    """

    # `kwargs` carries the wrapped function's full input map: the
    # `next_regime_to_V_arr` mapping alongside the Float/Int/Bool-valued
    # state/action inputs.
    def reduced(
        **kwargs: MappingProxyType[RegimeName, FloatND] | FloatND | IntND | BoolND,
    ) -> dict[str, Any]:
        U_arr, F_arr, CE, Q_arr, regime_probs = func(**kwargs)
        F_float = F_arr.astype(float)
        # NaN-count arrays are masked by feasibility: only feasible cells
        # contribute to numerators. Infeasible cells are zeroed out because
        # the solver masks them before the max, so a NaN there never
        # propagates to V_arr — reporting it would conflate causes.
        nan_arrays: dict[str, FloatND] = {
            "U_nan": jnp.isnan(U_arr).astype(float) * F_float,
            "CE_nan": jnp.isnan(CE).astype(float) * F_float,
            "Q_nan": jnp.isnan(Q_arr).astype(float) * F_float,
        }

        out: dict[str, Any] = {}
        F_total = jnp.maximum(jnp.sum(F_float), 1.0)
        for key, arr in nan_arrays.items():
            out[f"{key}_overall"] = jnp.sum(arr) / F_total
            for i, name in enumerate(variable_names):
                if i < arr.ndim:
                    axes = tuple(j for j in range(arr.ndim) if j != i)
                    F_slice = jnp.maximum(jnp.sum(F_float, axis=axes), 1.0)
                    out[f"{key}_by_{name}"] = jnp.sum(arr, axis=axes) / F_slice

        # F itself is a plain mean over all cells — it is the denominator's
        # source, not a conditional metric.
        out["F_feasible_overall"] = jnp.mean(F_float)
        for i, name in enumerate(variable_names):
            if i < F_float.ndim:
                axes = tuple(j for j in range(F_float.ndim) if j != i)
                out[f"F_feasible_by_{name}"] = jnp.mean(F_float, axis=axes)

        out["regime_probs"] = {k: jnp.mean(v) for k, v in regime_probs.items()}
        return out

    return reduced


def _productmap_over_state_action_space(
    *,
    func: Callable,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    state_batch_sizes: dict[StateName, int],
) -> Callable:
    """Wrap a scalar state-action function with productmap over actions then states.

    Matches the pattern used by `get_max_Q_over_a`: actions form the inner
    Cartesian product (unbatched), states form the outer loop (with batching).

    Args:
        func: Scalar function taking state and action values as keyword
            arguments.
        action_names: Tuple of action variable names; becomes the inner
            productmap (unbatched).
        state_names: Tuple of state variable names; becomes the outer
            productmap.
        state_batch_sizes: Mapping of state name to productmap batch size.

    Returns:
        Callable taking the same kwargs as `func` but expecting grid arrays
        instead of scalars for state and action variables. Output axes are
        ordered as `(*state_names, *action_names)`.

    """
    inner = productmap(
        func=func,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )
    return productmap(
        func=inner,
        variables=state_names,
        batch_sizes=state_batch_sizes,
    )
