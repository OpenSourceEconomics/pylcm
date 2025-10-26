"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import jax

from lcm.dispatchers import vmap_1d
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import InternalFunctions, StateActionSpace, StateSpaceInfo, Target
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
        MaxQOverAFunction,
        QAndFFunction,
    )


def build_state_space_infos(regime: Regime) -> dict[int, StateSpaceInfo]:
    state_space_infos = {}
    for period in regime.active:
        state_space_infos[period] = create_state_space_info(
            regime=regime,
            is_last_period=(period == regime.active[-1]),
        )
    return state_space_infos


def build_state_action_spaces(
    regime: Regime,
) -> dict[int, StateActionSpace]:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)
    state_action_spaces = {}
    for period in regime.active:
        state_action_spaces[period] = create_state_action_space(
            variable_info=variable_info,
            grids=grids,
            is_last_period=(period == regime.active[-1]),
        )
    return state_action_spaces


def build_Q_and_F_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
) -> dict[int, Any]:
    state_space_infos = build_state_space_infos(regime)
    # Create last period's next state space info
    last_periods_next_state_space_info = StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )

    Q_and_F_functions = {}
    # Importantly, for Q_and_F, we have to go in reversed order, because the
    # next_state_space_info depends on the next period
    for period in reversed(regime.active):
        is_last_period = period == regime.active[-1]

        # Determine next state space info
        if is_last_period:
            next_state_space_info = last_periods_next_state_space_info
        else:
            next_state_space_info = state_space_infos[period + 1]

        # Create Q and F functions
        Q_and_F = get_Q_and_F(
            regime=regime,
            internal_functions=internal_functions,
            next_state_space_info=next_state_space_info,
            period=period,
            is_last_period=is_last_period,
        )
        Q_and_F_functions[period] = Q_and_F
    return Q_and_F_functions


def build_max_Q_over_a_functions(
    regime: Regime, Q_and_F_functions: dict[int, QAndFFunction], *, enable_jit: bool
) -> dict[int, MaxQOverAFunction]:
    state_action_space = build_state_action_spaces(regime)

    max_Q_over_a_functions = {}
    for period in regime.active:
        action_names = tuple(state_action_space[period].continuous_actions) + tuple(
            state_action_space[period].discrete_actions
        )
        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F_functions[period],
            actions_names=action_names,
            states_names=tuple(state_action_space[period].states),
        )
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if enable_jit else max_Q_over_a
        )
    return max_Q_over_a_functions


def build_argmax_and_max_Q_over_a_functions(
    regime: Regime, Q_and_F_functions: dict[int, QAndFFunction], *, enable_jit: bool
) -> dict[int, ArgmaxQOverAFunction]:
    state_action_space = build_state_action_spaces(regime)

    argmax_and_max_Q_over_a_functions = {}
    for period in regime.active:
        action_names = tuple(state_action_space[period].discrete_actions) + tuple(
            state_action_space[period].continuous_actions
        )
        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F_functions[period], actions_names=action_names
        )
        argmax_and_max_Q_over_a_functions[period] = (
            jax.jit(argmax_and_max_Q_over_a) if enable_jit else argmax_and_max_Q_over_a
        )
    return argmax_and_max_Q_over_a_functions


def build_next_state_simulation_functions(
    regime: Regime, internal_functions: InternalFunctions, grids: dict[str, Array]
) -> dict[int, Any]:
    state_action_spaces = build_state_action_spaces(regime)
    next_state_simulation_functions = {}
    for period in regime.active:
        next_state = get_next_state_function(
            internal_functions=internal_functions,
            grids=grids,
            next_states=tuple(state_action_spaces[period].states),
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
        next_state_simulation_functions[period] = next_state_vmapped
    return next_state_simulation_functions
