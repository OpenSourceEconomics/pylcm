import logging
from types import MappingProxyType

import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.error_handling import validate_value_function_array
from lcm.interfaces import InternalRegime
from lcm.typing import FloatND, InternalParams, RegimeName


def solve(
    *,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        internal_params: Flat model-level parameter mapping with regime-prefixed keys.
        ages: Age grid for the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    """
    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    next_V_arr: MappingProxyType[RegimeName, FloatND] = MappingProxyType(
        {name: jnp.empty(0) for name in internal_regimes}
    )

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
            state_action_space = internal_regime.state_action_space(
                model_params=internal_params,
            )
            max_Q_over_a = internal_regime.max_Q_over_a_functions[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_V_arr=next_V_arr,
                **internal_params,
            )

            validate_value_function_array(V_arr=V_arr, age=ages.values[period])
            period_solution[name] = V_arr

        next_V_arr = MappingProxyType(period_solution)
        solution[period] = next_V_arr
        logger.info("Age: %s", ages.values[period])

    return MappingProxyType(solution)
