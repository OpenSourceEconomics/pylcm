from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.error_handling import validate_value_function_array
from lcm.interfaces import TerminalNonTerminal

if TYPE_CHECKING:
    import logging

    from lcm.interfaces import MaxQOverAFunction, StateActionSpace
    from lcm.typing import FloatND, ParamsDict


def solve(
    params: ParamsDict,
    n_periods: int,
    state_action_spaces: TerminalNonTerminal[StateActionSpace],
    max_Q_over_a_functions: TerminalNonTerminal[MaxQOverAFunction],
    logger: logging.Logger,
) -> dict[int, FloatND]:
    """Solve a model using grid search.

    Args:
        params: Dict of model parameters.
        n_periods: The number of periods in the model.
        state_action_space: The regimes state action spaces.
        max_Q_over_a_functions: The functions to calculate the maximum of the Q-function
            over all actions. The result corresponds to the Q-function of that period.
        logger: Logger that logs to stdout.

    Returns:
        Dict with one value function array per period.

    """
    solution = {}
    next_V_arr = jnp.empty(0)

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(n_periods)):
        max_Q_over_a = max_Q_over_a_functions(is_terminal=period == n_periods - 1)
        state_action_space = state_action_spaces(is_terminal=period == n_periods - 1)

        # evaluate Q-function on states and actions, and maximize over actions
        V_arr = max_Q_over_a(
            **state_action_space.states,
            **state_action_space.discrete_actions,
            **state_action_space.continuous_actions,
            period=period,
            next_V_arr=next_V_arr,
            params=params,
        )

        validate_value_function_array(
            V_arr=V_arr,
            period=period,
        )

        solution[period] = V_arr
        next_V_arr = V_arr
        logger.info("Period: %s", period)

    return solution
