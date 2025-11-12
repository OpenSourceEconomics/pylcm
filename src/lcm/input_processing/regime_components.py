"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax
from dags import concatenate_functions
from dags.signature import with_signature
from dags.tree import flatten_to_qnames

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import (
    InternalFunctions,
    PeriodVariantContainer,
    StateActionSpace,
    StateSpaceInfo,
    Target,
)
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)

if TYPE_CHECKING:
    from jax import Array

    from lcm.regime import Regime
    from lcm.typing import (
        ArgmaxQOverAFunction,
        GridsDict,
        InternalUserFunction,
        MaxQOverAFunction,
        NextStateSimulationFunction,
        QAndFFunction,
        RegimeTransitionFunction,
        VmappedRegimeTransitionFunction,
    )


def build_state_space_infos(regime: Regime) -> PeriodVariantContainer[StateSpaceInfo]:
    terminal_ssi = create_state_space_info(
        regime=regime,
        is_last_period=True,
    )

    non_terminal_ssi = create_state_space_info(
        regime=regime,
        is_last_period=False,
    )

    return PeriodVariantContainer(terminal=terminal_ssi, non_terminal=non_terminal_ssi)


def build_state_action_spaces(
    regime: Regime,
) -> PeriodVariantContainer[StateActionSpace]:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)

    terminal_sas = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        is_last_period=True,
    )

    non_terminal_sas = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        is_last_period=False,
    )

    return PeriodVariantContainer(terminal=terminal_sas, non_terminal=non_terminal_sas)


def build_Q_and_F_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
    state_space_infos: dict[str, PeriodVariantContainer[StateSpaceInfo]],
    grids: GridsDict,
) -> PeriodVariantContainer[QAndFFunction]:
    Q_and_F_terminal = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_infos={
            name: info.terminal for name, info in state_space_infos.items()
        },
        grids=grids,
        is_last_period=True,
    )
    Q_and_F_before_terminal = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_infos={
            name: info.terminal for name, info in state_space_infos.items()
        },
        grids=grids,
        is_last_period=False,
    )
    Q_and_F_non_terminal = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_infos={
            name: info.non_terminal for name, info in state_space_infos.items()
        },
        grids=grids,
        is_last_period=False,
    )
    return PeriodVariantContainer(
        terminal=Q_and_F_terminal,
        non_terminal=Q_and_F_non_terminal,
        before_terminal=Q_and_F_before_terminal,
    )


def build_max_Q_over_a_functions(
    regime: Regime,
    Q_and_F_functions: PeriodVariantContainer[QAndFFunction],
    *,
    enable_jit: bool,
) -> PeriodVariantContainer[MaxQOverAFunction]:
    state_action_spaces = build_state_action_spaces(regime)

    max_Q_over_a_functions = {}

    for attr in ("terminal", "non_terminal", "before_terminal"):
        fn = _build_max_Q_over_a_function(
            state_action_space=getattr(state_action_spaces, attr),
            Q_and_F=getattr(Q_and_F_functions, attr),
            enable_jit=enable_jit,
        )
        max_Q_over_a_functions[attr] = fn

    return PeriodVariantContainer(**max_Q_over_a_functions)


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
    Q_and_F_functions: PeriodVariantContainer[QAndFFunction],
    *,
    enable_jit: bool,
) -> PeriodVariantContainer[ArgmaxQOverAFunction]:
    state_action_spaces = build_state_action_spaces(regime)

    argmax_and_max_Q_over_a_functions = {}

    for attr in ("terminal", "non_terminal", "before_terminal"):
        state_action_space = getattr(state_action_spaces, attr)

        fn = _build_argmax_and_max_Q_over_a_function(
            state_action_space=state_action_space,
            Q_and_F=getattr(Q_and_F_functions, attr),
            enable_jit=enable_jit,
        )
        fn_spacemapped = simulation_spacemap(
            fn,
            actions_names=(),
            states_names=tuple(state_action_space.states),
        )
        argmax_and_max_Q_over_a_functions[attr] = fn_spacemapped

    return PeriodVariantContainer(**argmax_and_max_Q_over_a_functions)


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
        transitions=flatten_to_qnames(internal_functions.transitions),
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
            if parameter not in ("period", "params")
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
    *,
    enable_jit: bool,
) -> dict[str, RegimeTransitionFunction | VmappedRegimeTransitionFunction]:
    next_regime = concatenate_functions(
        functions=internal_functions
        | {"regime_transition_probs": regime_transition_probs},
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
            if parameter not in ("period", "params")
        ),
    )

    return {
        "solve": next_regime,
        "simulate": jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped,
    }
