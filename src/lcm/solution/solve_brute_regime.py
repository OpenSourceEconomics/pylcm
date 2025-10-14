from __future__ import annotations

from typing import TYPE_CHECKING

from lcm.error_handling import validate_value_function_array

if TYPE_CHECKING:
    from lcm.interfaces import InternalRegime
    from lcm.model import Model
    from lcm.typing import FloatND, ParamsDict


type RegimeName = str


def solve(
    params: dict[RegimeName, ParamsDict],
    model: Model,
) -> dict[int, FloatND]:
    """Solve the model for the given parameters."""
    n_periods = len(model.state_action_spaces)
    solution: dict[int, dict[RegimeName, FloatND]] = {}

    # Terminal period
    # ----------------------------------------------------------------------------------
    for internal_regime in _get_active_regimes(
        model.internal_regimes, period=n_periods - 1
    ):
        state_action_space = model.state_action_spaces_terminal[internal_regime.name]
        max_Q_over_a_terminal = model.max_Q_over_a_terminal[internal_regime.name]

        # evaluate Q-function on states and actions, and maximize over actions
        V_arr = max_Q_over_a_terminal(
            **state_action_space.states,
            **state_action_space.discrete_actions,
            **state_action_space.continuous_actions,
            params=params.get(internal_regime.name, {}),
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
                params=params.get(internal_regime.name, {}),
            )

            validate_value_function_array(
                V_arr=V_arr,
                period=period,
            )

            solution[period][internal_regime.name] = V_arr

    return solution


def _get_active_regimes(
    internal_regimes: list[InternalRegime], period: int
) -> list[InternalRegime]:
    return [ir for ir in internal_regimes if period in ir.active]
