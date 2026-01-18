"""Helper module for regime class initialization and utilities."""

import functools
import inspect
from dataclasses import fields
from typing import Any

import jax
from dags import concatenate_functions, get_annotations
from dags.signature import with_signature
from jax import Array

from lcm.ages import AgeGrid
from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import (
    InternalFunctions,
    PhaseVariantContainer,
    StateActionSpace,
    StateSpaceInfo,
    Target,
)
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F, get_Q_and_F_terminal
from lcm.regime import Regime
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)
from lcm.typing import (
    ArgmaxQOverAFunction,
    GridsDict,
    InternalUserFunction,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
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
    regime: Regime,
    regimes_to_active_periods: dict[RegimeName, list[int]],
    internal_functions: InternalFunctions,
    state_space_infos: dict[RegimeName, StateSpaceInfo],
    grids: GridsDict,
    ages: AgeGrid,
    regime_id: object,
) -> dict[int, QAndFFunction]:
    Q_and_F_functions = {}
    for period, age in enumerate(ages.values):
        if regime.terminal:
            Q_and_F = get_Q_and_F_terminal(
                regime=regime,
                internal_functions=internal_functions,
                period=period,
                age=age,
            )
        else:
            Q_and_F = get_Q_and_F(
                regime=regime,
                regimes_to_active_periods=regimes_to_active_periods,
                period=period,
                age=age,
                next_state_space_infos=state_space_infos,
                grids=grids,
                internal_functions=internal_functions,
                regime_id=regime_id,
            )
        Q_and_F_functions[period] = Q_and_F

    return Q_and_F_functions


def build_max_Q_over_a_functions(
    regime: Regime,
    Q_and_F_functions: dict[int, QAndFFunction],
    *,
    enable_jit: bool,
) -> dict[int, MaxQOverAFunction]:
    state_action_space = build_state_action_space(regime)
    max_Q_over_a_functions = {}
    for period, Q_and_F in Q_and_F_functions.items():
        max_Q_over_a_functions[period] = _build_max_Q_over_a_function(
            state_action_space=state_action_space,
            Q_and_F=Q_and_F,
            enable_jit=enable_jit,
        )
    return max_Q_over_a_functions


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
    regime: Regime,
    Q_and_F_functions: dict[int, QAndFFunction],
    *,
    enable_jit: bool,
) -> dict[int, ArgmaxQOverAFunction]:
    state_action_space = build_state_action_space(regime)
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
    return argmax_and_max_Q_over_a_functions


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
    *,
    enable_jit: bool,
) -> NextStateSimulationFunction:
    next_state = get_next_state_function(
        transitions=flatten_regime_namespace(internal_functions.transitions),
        functions=internal_functions.functions,
        grids=grids,
        target=Target.SIMULATE,
    )
    signature = inspect.signature(next_state)
    parameters = list(signature.parameters)

    next_state_vmapped = vmap_1d(
        func=next_state,
        variables=tuple(
            parameter
            for parameter in parameters
            if parameter not in ("period", "age", "params")
        ),
    )

    next_state_vmapped = with_signature(
        next_state_vmapped, kwargs=parameters, enforce=False
    )

    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped


def build_regime_transition_probs_functions(
    internal_functions: dict[str, InternalUserFunction],
    regime_transition_probs: InternalUserFunction,
    grids: dict[str, Array],
    regime_id: object,
    *,
    is_stochastic: bool,
    enable_jit: bool,
) -> PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]:
    # Wrap deterministic next_regime to return one-hot probability array
    if is_stochastic:
        probs_fn = regime_transition_probs
    else:
        probs_fn = _wrap_deterministic_regime_transition(
            regime_transition_probs, regime_id
        )

    # Keep array format - regime_transition_probs returns Array indexed by regime ID
    functions_pool = internal_functions | {"regime_transition_probs": probs_fn}

    next_regime = concatenate_functions(
        functions=functions_pool,
        targets="regime_transition_probs",
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
            if parameter not in ("period", "age", "params")
        ),
    )

    return PhaseVariantContainer(
        solve=jax.jit(next_regime) if enable_jit else next_regime,
        simulate=jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped,
    )


def _wrap_deterministic_regime_transition(
    fn: InternalUserFunction,
    regime_id: object,
) -> InternalUserFunction:
    """Wrap deterministic next_regime to return one-hot probability array.

    Converts a deterministic regime transition function that returns an integer
    regime ID to a function that returns a one-hot probability array, matching
    the interface of stochastic regime transitions.

    Args:
        fn: The user's deterministic next_regime function (returns int).
        regime_id: Instance mapping regime names to integer indices.

    Returns:
        A wrapped function that returns a one-hot probability array.

    """
    n_regimes = len(fields(regime_id))  # ty: ignore[invalid-argument-type]

    # Preserve original annotations but update return type
    annotations = get_annotations(fn)
    annotations_with_params = annotations.copy()
    annotations_with_params.pop("return", None)

    @with_signature(args=annotations_with_params, return_annotation="Array")
    @functools.wraps(fn)
    def wrapped(
        *args: Array | int, params: dict[str, Any], **kwargs: Array | int
    ) -> Array:
        regime_id = fn(*args, params=params, **kwargs)
        return jax.nn.one_hot(regime_id, n_regimes)

    return wrapped
