"""Per-period diagnostic closures and feasibility-conditional reductions.

Cold-path machinery used only when `validate_V` detects NaN in a solved
value-function array. `_build_compute_intermediates_per_period` produces
one JIT-compiled closure per period that productmaps
`get_compute_intermediates` over the full state-action space and fuses
the compute step with on-device reductions (`_wrap_with_reduction`).
The fused output is consumed by `_enrich_with_diagnostics` in
`lcm.utils.error_handling`.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from lcm.ages import AgeGrid
from lcm.grids import Grid
from lcm.interfaces import StateActionSpace
from lcm.regime_building.Q_and_F import get_complete_targets, get_compute_intermediates
from lcm.regime_building.V import VInterpolationInfo
from lcm.typing import (
    ActionName,
    FunctionsMapping,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.utils.dispatchers import productmap


def _build_compute_intermediates_per_period(
    *,
    flat_param_names: frozenset[str],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    state_action_space: StateActionSpace,
    grids: MappingProxyType[StateOrActionName, Grid],
    ages: AgeGrid,
    enable_jit: bool,
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
        ages: Age grid for the model.
        enable_jit: Whether to JIT-compile the fused closure.

    Returns:
        Immutable mapping of period index to fused closure.

    """
    state_batch_sizes = {
        name: grid.batch_size
        for name, grid in grids.items()
        if name in state_action_space.state_names
    }

    configs: dict[tuple[str, ...], list[int]] = {}
    for period in range(ages.n_periods):
        complete = get_complete_targets(
            period=period,
            transitions=transitions,
            regimes_to_active_periods=regimes_to_active_periods,
            stochastic_transition_names=stochastic_transition_names,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        configs.setdefault(complete, []).append(period)

    variable_names = (
        *state_action_space.state_names,
        *state_action_space.action_names,
    )
    built: dict[tuple[str, ...], Callable] = {}
    for complete_targets in configs:
        scalar = get_compute_intermediates(
            flat_param_names=flat_param_names,
            functions=functions,
            constraints=constraints,
            complete_targets=complete_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        mapped = _productmap_over_state_action_space(
            func=scalar,
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
            state_batch_sizes=state_batch_sizes,
        )
        fused = _wrap_with_reduction(func=mapped, variable_names=variable_names)
        built[complete_targets] = jax.jit(fused) if enable_jit else fused

    result: dict[int, Callable] = {}
    for key, periods in configs.items():
        for period in periods:
            result[period] = built[key]

    return MappingProxyType(result)


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
            `(U_arr, F_arr, E_next_V, Q_arr, regime_probs)`. `regime_probs`
            is a mapping of target regime names to per-point probability
            arrays.
        variable_names: Tuple of state + action names in the order that
            matches the productmap axes of `func`. Used to label the
            `{metric}_by_{name}` reductions.

    Returns:
        Callable taking the same kwargs as `func` and returning a dict with
        `{Y}_overall` scalars and `{Y}_by_{name}` vectors for `Y` in
        {`U_nan`, `E_nan`, `Q_nan`, `F_feasible`}, plus `regime_probs` as
        a dict of per-target scalar means. The `{U,E,Q}_nan_*` fractions
        are conditional on feasibility (numerator restricted to feasible
        cells, denominator is the feasible-cell count); `F_feasible_*`
        is the plain mean over all cells.

    """

    def reduced(**kwargs: Array) -> dict[str, Any]:
        U_arr, F_arr, E_next_V, Q_arr, regime_probs = func(**kwargs)
        F_float = F_arr.astype(float)
        # NaN-count arrays are masked by feasibility: only feasible cells
        # contribute to numerators. Infeasible cells are zeroed out because
        # the solver masks them before the max, so a NaN there never
        # propagates to V_arr — reporting it would conflate causes.
        nan_arrays: dict[str, Array] = {
            "U_nan": jnp.isnan(U_arr).astype(float) * F_float,
            "E_nan": jnp.isnan(E_next_V).astype(float) * F_float,
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
