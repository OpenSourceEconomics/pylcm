from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.error_handling import validate_value_function_array
from lcm.shocks import fill_shock_grids, pre_compute_shock_probabilities

if TYPE_CHECKING:
    import logging

    from lcm.interfaces import (
        InternalRegime,
    )
    from lcm.typing import FloatND, ParamsDict, RegimeName


def solve(
    params: ParamsDict,
    n_periods: int,
    internal_regimes: dict[str, InternalRegime],
    logger: logging.Logger,
) -> dict[int, dict[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        params: Dict of model parameters.
        n_periods: The number of periods in the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.

    Returns:
        Dict with one value function array per period.

    """
    solution: dict[int, dict[RegimeName, FloatND]] = {}
    next_V_arr: dict[RegimeName, FloatND] = {
        name: jnp.empty(0) for name in internal_regimes
    }
    params = pre_compute_shock_probabilities(internal_regimes, params)
    internal_regimes = fill_shock_grids(internal_regimes, params)
    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(n_periods)):
        period_solution: dict[RegimeName, FloatND] = {}

        for name, internal_regime in internal_regimes.items():
            max_Q_over_a = internal_regime.max_Q_over_a_functions(
                period, n_periods=n_periods
            )
            state_action_space = internal_regime.state_action_spaces(
                period, n_periods=n_periods
            )
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
            period_solution[name] = V_arr
        next_V_arr = period_solution
        solution[period] = period_solution
        logger.info("Period: %s", period)

    return solution
