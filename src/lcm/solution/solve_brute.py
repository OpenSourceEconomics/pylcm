from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.error_handling import validate_value_function_array

if TYPE_CHECKING:
    import logging

    from lcm.ages import AgeGrid
    from lcm.interfaces import (
        InternalRegime,
    )
    from lcm.typing import FloatND, ParamsDict, RegimeName


def solve(
    params: ParamsDict,
    ages: AgeGrid,
    internal_regimes: dict[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> dict[int, dict[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        params: Dict of model parameters.
        ages: Age grid for the model.
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

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(ages.n_periods)):
        period_solution: dict[RegimeName, FloatND] = {}

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        for name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_spaces
            max_Q_over_a = internal_regime.max_Q_over_a_functions[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
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
        logger.info("Age: %s", ages.ages[period])

    return solution
