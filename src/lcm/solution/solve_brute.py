from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.error_handling import validate_value_function_array
from lcm.shocks import pre_compute_shock_probabilities, update_sas_with_shocks

if TYPE_CHECKING:
    import logging

    from lcm.ages import AgeGrid
    from lcm.interfaces import (
        InternalRegime,
    )
    from lcm.typing import FloatND, ParamsDict, RegimeName


def solve(
    user_params: ParamsDict,
    ages: AgeGrid,
    internal_regimes: dict[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> dict[int, dict[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        user_params: Dict of model parameters as provided by the user.
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

    # Augment the params provided by the user with transition probabilities
    # for all shocks and then fill the grids in the state action space with
    # the shock values calculated from the params
    params_with_precomputed_shocks = pre_compute_shock_probabilities(
        internal_regimes, user_params
    )
    internal_regimes_with_updated_sas = update_sas_with_shocks(
        internal_regimes, user_params
    )

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(ages.n_periods)):
        period_solution: dict[RegimeName, FloatND] = {}

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes_with_updated_sas.items()
            if period in regime.active_periods
        }

        for name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_space
            max_Q_over_a = internal_regime.max_Q_over_a_functions[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_V_arr=next_V_arr,
                params=params_with_precomputed_shocks,
            )

            validate_value_function_array(
                V_arr=V_arr,
                period=period,
            )
            period_solution[name] = V_arr

        next_V_arr = period_solution
        solution[period] = period_solution
        logger.info("Age: %s", ages.values[period])

    return solution
