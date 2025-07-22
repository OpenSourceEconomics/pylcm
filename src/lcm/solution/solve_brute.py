import logging

import jax.numpy as jnp
from jax import Array

from lcm.interfaces import StateActionSpace
from lcm.typing import MaxQcOverDFunction, MaxQOverCFunction, ParamsDict


def solve(
    params: ParamsDict,
    state_action_spaces: dict[int, StateActionSpace],
    max_Q_over_c_functions: dict[int, MaxQOverCFunction],
    max_Qc_over_d_functions: dict[int, MaxQcOverDFunction],
    logger: logging.Logger,
) -> dict[int, Array]:
    """Solve a model using grid search.

    Args:
        params: Dict of model parameters.
        state_action_spaces: Dict with one state_action_space per period.
        max_Q_over_c_functions: Dict with one function per period. The functions
            calculate the maximum of the Q-function over the continuous actions. The
            result corresponds to the Qc-function of that period.
        max_Qc_over_d_functions: Dict with one function per period. The functions
            calculate the the (expected) maximum of the Qc-function over the discrete
            actions. The result corresponds to the value function array of that period.
        logger: Logger that logs to stdout.

    Returns:
        Dict with one value function array per period.

    """
    n_periods = len(state_action_spaces)
    solution = {}
    next_V_arr = jnp.empty(0)

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(n_periods)):
        state_action_space = state_action_spaces[period]

        max_Qc_over_d = max_Qc_over_d_functions[period]
        max_Q_over_c = max_Q_over_c_functions[period]

        # evaluate Q-function on states and actions, and maximize over continuous
        # actions
        Qc_arr = max_Q_over_c(
            **state_action_space.states,
            **state_action_space.discrete_actions,
            **state_action_space.continuous_actions,
            next_V_arr=next_V_arr,
            params=params,
        )

        # maximize Qc-function evaluations over discrete actions
        V_arr = max_Qc_over_d(Qc_arr, params=params)

        solution[period] = V_arr
        next_V_arr = V_arr
        logger.info("Period: %s", period)

    return solution
