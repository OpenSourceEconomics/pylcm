"""Helper module for regime class initialization and utilities."""

import functools
import inspect
from types import MappingProxyType
from typing import Any

import jax
import pandas as pd
from dags import concatenate_functions, get_annotations
from dags.signature import with_signature
from jax import Array

from lcm.ages import AgeGrid
from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.grids import Grid
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import (
    InternalFunctions,
    PhaseVariantContainer,
    StateActionSpace,
    StateSpaceInfo,
)
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.next_state import get_next_state_function_for_simulation
from lcm.Q_and_F import get_Q_and_F, get_Q_and_F_terminal
from lcm.regime import Regime
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)
from lcm.typing import (
    ArgmaxQOverAFunction,
    GridsDict,
    InternalRegimeParams,
    InternalUserFunction,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeTransitionFunction,
    VmappedRegimeTransitionFunction,
)
from lcm.utils import flatten_regime_namespace


def build_state_space_info(regime: Regime) -> StateSpaceInfo:
    return create_state_space_info(
        regime=regime,
        is_last_period=False,
    )


def build_state_action_space(
    regime: Regime,
) -> StateActionSpace:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)
    return create_state_action_space(
        variable_info=variable_info,
        grids=grids,
    )


def build_Q_and_F_functions(
    regime_name: str,
    regime: Regime,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    internal_functions: InternalFunctions,
    state_space_infos: MappingProxyType[RegimeName, StateSpaceInfo],
    ages: AgeGrid,
) -> MappingProxyType[int, QAndFFunction]:
    Q_and_F_functions = {}
    for period, age in enumerate(ages.values):
        if regime.terminal:
            Q_and_F = get_Q_and_F_terminal(
                internal_functions=internal_functions,
                period=period,
                age=age,
            )
        else:
            Q_and_F = get_Q_and_F(
                regime_name=regime_name,
                regimes_to_active_periods=regimes_to_active_periods,
                period=period,
                age=age,
                next_state_space_infos=state_space_infos,
                internal_functions=internal_functions,
            )
        Q_and_F_functions[period] = Q_and_F

    return MappingProxyType(Q_and_F_functions)


def build_max_Q_over_a_functions(
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    *,
    enable_jit: bool,
) -> MappingProxyType[int, MaxQOverAFunction]:
    max_Q_over_a_functions = {}
    for period, Q_and_F in Q_and_F_functions.items():
        max_Q_over_a_functions[period] = _build_max_Q_over_a_function(
            state_action_space=state_action_space,
            Q_and_F=Q_and_F,
            enable_jit=enable_jit,
        )
    return MappingProxyType(max_Q_over_a_functions)


def _build_max_Q_over_a_function(
    state_action_space: StateActionSpace,
    Q_and_F: QAndFFunction,
    enable_jit: bool,  # noqa: FBT001
) -> MaxQOverAFunction:
    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=state_action_space.actions_names,
        states_names=state_action_space.states_names,
    )

    if enable_jit:
        max_Q_over_a = jax.jit(max_Q_over_a)

    return max_Q_over_a


def build_argmax_and_max_Q_over_a_functions(
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    *,
    enable_jit: bool,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    argmax_and_max_Q_over_a_functions = {}
    for period, Q_and_F in Q_and_F_functions.items():
        fn = _build_argmax_and_max_Q_over_a_function(
            state_action_space=state_action_space,
            Q_and_F=Q_and_F,
            enable_jit=enable_jit,
        )
        argmax_and_max_Q_over_a_functions[period] = simulation_spacemap(
            fn,
            actions_names=(),
            states_names=tuple(state_action_space.states),
        )
    return MappingProxyType(argmax_and_max_Q_over_a_functions)


def _build_argmax_and_max_Q_over_a_function(
    state_action_space: StateActionSpace,
    Q_and_F: QAndFFunction,
    enable_jit: bool,  # noqa: FBT001
) -> ArgmaxQOverAFunction:
    argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=state_action_space.actions_names,
    )

    if enable_jit:
        argmax_and_max_Q_over_a = jax.jit(argmax_and_max_Q_over_a)

    return argmax_and_max_Q_over_a


def build_next_state_simulation_functions(
    internal_functions: InternalFunctions,
    grids: GridsDict,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    *,
    enable_jit: bool,
) -> NextStateSimulationFunction:
    next_state = get_next_state_function_for_simulation(
        transitions=flatten_regime_namespace(internal_functions.transitions),
        functions=internal_functions.functions,
        variable_info=variable_info,
        grids=grids,
        gridspecs=gridspecs,
    )
    signature = inspect.signature(next_state)
    parameters = list(signature.parameters)

    next_state_vmapped = vmap_1d(
        func=next_state,
        variables=tuple(
            parameter
            for parameter in parameters
            if parameter not in ("period", "age", "internal_regime_params")
        ),
    )

    next_state_vmapped = with_signature(
        next_state_vmapped, kwargs=parameters, enforce=False
    )

    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped


def build_regime_transition_probs_functions(
    internal_functions: MappingProxyType[str, InternalUserFunction],
    regime_transition_probs: InternalUserFunction,
    grids: MappingProxyType[str, Array],
    regime_names_to_ids: RegimeNamesToIds,
    *,
    is_stochastic: bool,
    enable_jit: bool,
) -> PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]:
    # Wrap deterministic next_regime to return one-hot probability array
    if is_stochastic:
        probs_fn = regime_transition_probs
    else:
        probs_fn = _wrap_deterministic_regime_transition(
            regime_transition_probs, regime_names_to_ids
        )

    # Wrap to convert array output to dict format
    wrapped_regime_transition_probs = _wrap_regime_transition_probs(
        probs_fn, regime_names_to_ids
    )

    functions_pool = dict(internal_functions) | {
        "regime_transition_probs": wrapped_regime_transition_probs
    }

    next_regime = concatenate_functions(
        functions=functions_pool,
        targets="regime_transition_probs",
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    signature = inspect.signature(next_regime)
    parameters = list(signature.parameters)

    # We do this because a transition function without any parameters will throw
    # an error with vmap
    next_regime_accepting_all = with_signature(
        next_regime,
        args=parameters + [state for state in grids if state not in parameters],
    )

    signature = inspect.signature(next_regime_accepting_all)
    parameters = list(signature.parameters)

    next_regime_vmapped = vmap_1d(
        func=next_regime_accepting_all,
        variables=tuple(
            parameter
            for parameter in parameters
            if parameter not in ("period", "age", "internal_regime_params")
        ),
    )

    return PhaseVariantContainer(
        solve=jax.jit(next_regime) if enable_jit else next_regime,
        simulate=jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped,
    )


def _wrap_regime_transition_probs(
    fn: InternalUserFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> InternalUserFunction:
    """Wrap next_regime function to convert array output to dict format.

    The next_regime function returns a JAX array of probabilities indexed by
    the regime's id. This wrapper converts the array to dict format for internal
    processing.

    Args:
        fn: The user's next_regime function (already wrapped with
            internal_regime_params).
        regime_names_to_ids: Mapping from regime names to integer indices.

    Returns:
        A wrapped function that returns MappingProxyType[str, float|Array].

    """
    # Get regime names in index order from regime_names_to_ids
    regime_names_by_id: list[tuple[int, str]] = sorted(
        [(idx, name) for name, idx in regime_names_to_ids.items()],
        key=lambda x: x[0],
    )
    regime_names = [name for _, name in regime_names_by_id]

    # Preserve original annotations
    annotations = get_annotations(fn)
    annotations_with_internal_regime_params = annotations.copy()
    return_annotation = annotations_with_internal_regime_params.pop(
        "return", "dict[str, Any]"
    )

    @with_signature(
        args=annotations_with_internal_regime_params,
        return_annotation=return_annotation,
    )
    @functools.wraps(fn)
    def wrapped(
        *args: Array | int,
        internal_regime_params: InternalRegimeParams,
        **kwargs: Array | int,
    ) -> MappingProxyType[str, Any]:
        result = fn(*args, internal_regime_params=internal_regime_params, **kwargs)
        # Convert array to dict using ordering by regime id
        return MappingProxyType(
            {name: result[idx] for idx, name in enumerate(regime_names)}
        )

    return wrapped


def _wrap_deterministic_regime_transition(
    fn: InternalUserFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> InternalUserFunction:
    """Wrap deterministic next_regime to return one-hot probability array.

    Converts a deterministic regime transition function that returns an integer
    regime ID to a function that returns a one-hot probability array, matching
    the interface of stochastic regime transitions.

    Args:
        fn: The user's deterministic next_regime function (returns int).
        regime_names_to_ids: Mapping from regime names to integer indices.

    Returns:
        A wrapped function that returns a one-hot probability array.

    """
    n_regimes = len(regime_names_to_ids)

    # Preserve original annotations but update return type
    annotations = get_annotations(fn)
    annotations_with_internal_regime_params = annotations.copy()
    annotations_with_internal_regime_params.pop("return", None)

    @with_signature(
        args=annotations_with_internal_regime_params, return_annotation="Array"
    )
    @functools.wraps(fn)
    def wrapped(
        *args: Array | int,
        internal_regime_params: InternalRegimeParams,
        **kwargs: Array | int,
    ) -> Array:
        regime_idx = fn(*args, internal_regime_params=internal_regime_params, **kwargs)
        return jax.nn.one_hot(regime_idx, n_regimes)

    return wrapped
