"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import jax

from lcm.dispatchers import vmap_1d
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import (
    ArgmaxQOverAFunctions,
    InternalFunctions,
    MaxQOverAFunctions,
    QAndFFunctions,
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
        QAndFFunction,
    )


def build_state_space_infos(regime: Regime) -> StateSpaceInfo:
    return create_state_space_info(
        regime=regime,
        is_last_period=False,
    )


def build_state_action_spaces(
    regime: Regime,
) -> StateActionSpace:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)
    return create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        is_last_period=False,
    )


def build_Q_and_F_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
) -> QAndFFunctions:
    state_space_info = build_state_space_infos(regime)

    # Create Q and F functions
    q_and_f_terminal: QAndFFunction = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_info=state_space_info,
        is_last_period=True,
    )
    q_and_f_non_terminal: QAndFFunction = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_info=state_space_info,
        is_last_period=False,
    )
    return QAndFFunctions(terminal=q_and_f_terminal, non_terminal=q_and_f_non_terminal)


def build_max_Q_over_a_functions(
    regime: Regime, Q_and_F_functions: QAndFFunctions, *, enable_jit: bool
) -> MaxQOverAFunctions:
    state_action_space = build_state_action_spaces(regime)

    action_names = tuple(state_action_space.continuous_actions) + tuple(
        state_action_space.discrete_actions
    )
    max_Q_over_a_terminal = get_max_Q_over_a(
        Q_and_F=Q_and_F_functions.terminal,
        actions_names=action_names,
        states_names=tuple(state_action_space.states),
    )
    max_Q_over_a_non_terminal = get_max_Q_over_a(
        Q_and_F=Q_and_F_functions.non_terminal,
        actions_names=action_names,
        states_names=tuple(state_action_space.states),
    )

    return MaxQOverAFunctions(
        terminal=jax.jit(max_Q_over_a_terminal)
        if enable_jit
        else max_Q_over_a_terminal,
        non_terminal=jax.jit(max_Q_over_a_non_terminal)
        if enable_jit
        else max_Q_over_a_non_terminal,
    )


def build_argmax_and_max_Q_over_a_functions(
    regime: Regime, Q_and_F_functions: QAndFFunctions, *, enable_jit: bool
) -> dict[int, ArgmaxQOverAFunction]:
    state_action_space = build_state_action_spaces(regime)

    action_names = tuple(state_action_space.discrete_actions) + tuple(
        state_action_space.continuous_actions
    )
    argmax_and_max_Q_over_a_terminal = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F_functions.terminal, actions_names=action_names
    )
    argmax_and_max_Q_over_a_non_terminal = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F_functions.non_terminal, actions_names=action_names
    )

    return ArgmaxQOverAFunctions(
        terminal=jax.jit(argmax_and_max_Q_over_a_terminal)
        if enable_jit
        else argmax_and_max_Q_over_a_terminal,
        non_terminal=jax.jit(argmax_and_max_Q_over_a_non_terminal)
        if enable_jit
        else argmax_and_max_Q_over_a_non_terminal,
    )


def build_next_state_simulation_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
    grids: dict[str, Array],
    *,
    enable_jit: bool,
) -> Any:
    state_action_space = build_state_action_spaces(regime)
    next_state = get_next_state_function(
        internal_functions=internal_functions,
        grids=grids,
        next_states=tuple(state_action_space.states),
        target=Target.SIMULATE,
    )
    signature = inspect.signature(next_state)
    parameters = list(signature.parameters)

    next_state_vmapped = vmap_1d(
        func=next_state,
        variables=tuple(
            parameter
            for parameter in parameters
            if parameter not in ["_period", "params"]
        ),
    )
    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped
