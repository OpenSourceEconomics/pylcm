import logging
import time
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
        internal_params: Immutable mapping of regime names to flat parameter mappings.
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
    has_multiple_regimes = len(internal_regimes) > 1
    total_start = time.monotonic()

    # backwards induction loop
    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        for name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_space(
                regime_params=internal_params[name],
            )
            max_Q_over_a = internal_regime.max_Q_over_a_functions[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_V_arr=next_V_arr,
                **internal_params[name],
            )

            if jnp.any(jnp.isnan(V_arr)) or jnp.any(jnp.isinf(V_arr)):
                logger.warning(
                    "NaN/Inf in V_arr for regime '%s' at age %s",
                    name,
                    ages.values[period],
                )

            logger.debug(
                "  regime '%s': V min=%.3g max=%.3g mean=%.3g",
                name,
                float(jnp.min(V_arr)),
                float(jnp.max(V_arr)),
                float(jnp.mean(V_arr)),
            )

            validate_value_function_array(
                V_arr=V_arr, age=ages.values[period], regime_name=name
            )
            period_solution[name] = V_arr

        next_V_arr = MappingProxyType(period_solution)
        solution[period] = next_V_arr

        elapsed = time.monotonic() - period_start
        if has_multiple_regimes:
            logger.info(
                "Age: %s  regimes=%d  (%.1fs)",
                ages.values[period],
                len(active_regimes),
                elapsed,
            )
        else:
            logger.info("Age: %s  (%.1fs)", ages.values[period], elapsed)

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%.1fs)", total_elapsed)

    return MappingProxyType(solution)
