import logging
import time
from types import MappingProxyType

import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.interfaces import InternalRegime
from lcm.typing import FloatND, InternalParams, RegimeName
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_timing,
    log_V_stats,
)


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
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND] = MappingProxyType(
        {name: jnp.empty(0) for name in internal_regimes}
    )

    logger.info("Starting solution")
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
            max_Q_over_a = internal_regime.solve_functions.max_Q_over_a[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **internal_params[name],
            )

            log_nan_in_V(
                logger=logger,
                regime_name=name,
                age=ages.values[period],
                V_arr=V_arr,
            )
            log_V_stats(logger=logger, regime_name=name, V_arr=V_arr)

            validate_V(
                V_arr=V_arr,
                age=float(ages.values[period]),
                regime_name=name,
                partial_solution=MappingProxyType(solution),
                diagnostic_funcs=internal_regime.solve_functions.diagnostic_Q_and_F.get(
                    period
                ),
                diagnostic_kwargs={
                    **state_action_space.states,
                    **state_action_space.actions,
                    "next_regime_to_V_arr": next_regime_to_V_arr,
                    **internal_params[name],
                },
                state_names=state_action_space.state_names,
            )
            period_solution[name] = V_arr

        next_regime_to_V_arr = MappingProxyType(period_solution)
        solution[period] = next_regime_to_V_arr

        elapsed = time.monotonic() - period_start
        log_period_timing(
            logger=logger,
            age=ages.values[period],
            n_active_regimes=len(active_regimes),
            elapsed=elapsed,
        )

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return MappingProxyType(solution)
