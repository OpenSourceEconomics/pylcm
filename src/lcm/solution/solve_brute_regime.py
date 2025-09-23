from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.error_handling import validate_value_function_array

if TYPE_CHECKING:
    import logging

    from lcm.interfaces import StateActionSpace
    from lcm.typing import FloatND, MaxQOverAFunction, ParamsDict
    from lcm.model import Model
    from lcm.input_processing.regime_processing import InternalRegime


def solve(
    params: ParamsDict,
    model: Model,
) -> dict[int, FloatND]:

    n_periods = len(model.state_action_spaces)
    solution: dict[int, dict[str, FloatND]] = {}

    # Terminal period
    # ----------------------------------------------------------------------------------
    for internal_regime in _get_active_regimes(model.internal_regimes, period=n_periods - 1):

        state_action_space = model.state_action_spaces_terminal[internal_regime.name]
        max_Q_over_a_terminal = model.max_Q_over_a_terminal[internal_regime.name]

        # evaluate Q-function on states and actions, and maximize over actions
        V_arr = max_Q_over_a_terminal(
            **state_action_space.states,
            **state_action_space.discrete_actions,
            **state_action_space.continuous_actions,
            params=params,
        )

        validate_value_function_array(
            V_arr=V_arr,
            period=n_periods - 1,
        )

        solution[n_periods - 1][internal_regime.name] = V_arr

    # Non-terminal periods
    # ----------------------------------------------------------------------------------
    for period in reversed(range(n_periods - 1)):

        for internal_regime in _get_active_regimes(model.internal_regimes, period):
            
            state_action_space = model.state_action_spaces[internal_regime.name]
            max_Q_over_a = model.max_Q_over_a[internal_regime.name]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.discrete_actions,
                **state_action_space.continuous_actions,
                next_V_arr_dict=solution[period + 1],
                params=params,
            )

            validate_value_function_array(
                V_arr=V_arr,
                period=period,
            )

            solution[period][internal_regime.name] = V_arr

    return solution


def _get_active_regimes(internal_regimes: list[InternalRegime], period: int) -> list[InternalRegime]:
    return [ir for ir in internal_regimes if period in ir.active]