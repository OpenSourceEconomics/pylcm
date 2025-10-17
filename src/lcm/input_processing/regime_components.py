"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax

from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import InternalFunctions, StateActionSpace, StateSpaceInfo
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)

if TYPE_CHECKING:
    from lcm.regime import Regime
    from lcm.typing import ArgmaxQOverAFunction, MaxQOverAFunction, QAndFFunction


def build_state_space_infos(
    regime: Regime, n_periods: int
) -> dict[int, StateSpaceInfo]:
    state_space_infos = {}
    for period in range(n_periods):
        state_space_infos[period] = create_state_space_info(
            regime=regime,
            is_last_period=(period == n_periods - 1),
        )
    return state_space_infos


def build_state_action_spaces(
    regime: Regime, n_periods: int
) -> dict[int, StateActionSpace]:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)
    state_action_spaces = {}
    for period in range(n_periods):
        state_action_spaces[period] = create_state_action_space(
            variable_info=variable_info,
            grids=grids,
            is_last_period=(period == n_periods - 1),
        )
    return state_action_spaces


def build_Q_and_F_functions(
    regime: Regime, n_periods: int, internal_functions: InternalFunctions
) -> dict[int, Any]:
    state_space_infos = build_state_space_infos(regime, n_periods)
    # Create last period's next state space info
    last_periods_next_state_space_info = StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )

    Q_and_F_functions = {}
    # Importantly, for Q_and_F, we have to go in reversed order, because the
    # next_state_space_info depends on the next period
    for period in reversed(range(n_periods)):
        is_last_period = period == n_periods - 1

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
        Q_and_F_functions[period] = jax.jit(Q_and_F) if regime.enable_jit else Q_and_F
    return Q_and_F_functions


def build_max_Q_over_a_functions(
    regime: Regime,
    n_periods: int,
    Q_and_F_functions: dict[int, QAndFFunction],
) -> dict[int, MaxQOverAFunction]:
    state_action_space = build_state_action_spaces(regime, n_periods)

    max_Q_over_a_functions = {}
    for period in range(n_periods):
        action_names = tuple(state_action_space[period].continuous_actions) + tuple(
            state_action_space[period].discrete_actions
        )
        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F_functions[period],
            actions_names=action_names,
            states_names=tuple(state_action_space[period].states),
        )
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if regime.enable_jit else max_Q_over_a
        )
    return max_Q_over_a_functions


def build_argmax_and_max_Q_over_a_functions(
    regime: Regime,
    n_periods: int,
    Q_and_F_functions: dict[int, QAndFFunction],
) -> dict[int, ArgmaxQOverAFunction]:
    state_action_space = build_state_action_spaces(regime, n_periods)

    argmax_and_max_Q_over_a_functions = {}
    for period in range(n_periods):
        action_names = tuple(state_action_space[period].discrete_actions) + tuple(
            state_action_space[period].continuous_actions
        )
        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F_functions[period], actions_names=action_names
        )
        argmax_and_max_Q_over_a_functions[period] = (
            jax.jit(argmax_and_max_Q_over_a)
            if regime.enable_jit
            else argmax_and_max_Q_over_a
        )
    return argmax_and_max_Q_over_a_functions
