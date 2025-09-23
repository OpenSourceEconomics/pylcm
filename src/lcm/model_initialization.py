"""Helper module for Regime class initialization and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax

from lcm.exceptions import ModelInitializationError
from lcm.input_processing import process_model
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)

if TYPE_CHECKING:
    from lcm.input_processing.regime_processing import InternalRegime
    from lcm.typing import ArgmaxQOverAFunction, MaxQOverAFunction


def get_components(internal_regimes: list[InternalRegime], n_periods: int, jit: bool):
    last_period = n_periods - 1

    # Terminal period related components
    # ----------------------------------------------------------------------------------
    state_action_spaces_terminal: dict[str, StateActionSpace] = {}
    state_space_infos_terminal: dict[str, StateSpaceInfo] = {}
    max_Q_over_a_terminal: dict[str, MaxQOverAFunction] = {}
    argmax_and_max_Q_over_a_terminal: dict[str, ArgmaxQOverAFunction] = {}

    last_periods_next_state_space_info = StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )

    # Non-terminal periods related components
    # ----------------------------------------------------------------------------------
    state_action_spaces: dict[str, StateActionSpace] = {}
    state_space_infos: dict[str, dict[int, StateSpaceInfo]] = {}
    max_Q_over_a: dict[str, dict[int, MaxQOverAFunction]] = {}
    argmax_and_max_Q_over_a: dict[str, dict[int, ArgmaxQOverAFunction]] = {}

    for period in reversed(range(n_periods)):

        active_regimes = [
            ir for ir in internal_regimes if period in ir.active
        ]

        for internal_regime in active_regimes:

            state_action_space = create_state_action_space(
                internal_model=internal_regime
                is_last_period=False,
            )

            state_action_space_terminal = create_state_action_space(
                internal_model=internal_regime,
                is_last_period=True,
            )

            state_space_info = create_state_space_info(
                internal_model=internal_regime,
                is_last_period=False,
            )

            state_space_info_terminal = create_state_space_info(
                internal_model=internal_regime,
                is_last_period=True,
            )

            if period == last_period:
                next_state_space_info = last_periods_next_state_space_info
            else:
                next_state_space_info = state_space_infos[period + 1]

            Q_and_F = get_Q_and_F(
                internal_model=internal_regime,
                next_state_space_info=next_state_space_info,
                period=period,
            )



def get_regime_components(internal_regime: InternalRegime) -> dict[str, Any]:
    last_period = internal_model.n_periods - 1
    state_action_spaces: dict[int, StateActionSpace] = {}
    state_space_infos: dict[int, StateSpaceInfo] = {}
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = {}
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = {}

    # Create last period's next state space info

    # Create functions for each period (reversed order following Backward induction)
    for period in reversed(range(internal_model.n_periods)):
        is_last_period = period == last_period

        # Determine next state space info
        if is_last_period:
            next_state_space_info = last_periods_next_state_space_info
        else:
            next_state_space_info = state_space_infos[period + 1]

        # Create Q and F functions
        Q_and_F = get_Q_and_F(
            internal_model=internal_model,
            next_state_space_info=next_state_space_info,
            period=period,
        )

        # Create optimization functions
        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.continuous_actions)
            + tuple(state_action_space.discrete_actions),
            states_names=tuple(state_action_space.states),
        )

        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.discrete_actions)
            + tuple(state_action_space.continuous_actions),
        )

        # Store results
        state_action_spaces[period] = state_action_space
        state_space_infos[period] = state_space_info
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if model.enable_jit else max_Q_over_a
        )
        argmax_and_max_Q_over_a_functions[period] = (
            jax.jit(argmax_and_max_Q_over_a)
            if model.enable_jit
            else argmax_and_max_Q_over_a
        )

    return {
        "internal_model": internal_model,
        "params_template": internal_model.params,
        "state_action_spaces": state_action_spaces,
        "state_space_infos": state_space_infos,
        "max_Q_over_a_functions": max_Q_over_a_functions,
        "argmax_and_max_Q_over_a_functions": argmax_and_max_Q_over_a_functions,
    }
