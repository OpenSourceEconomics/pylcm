import logging
from types import MappingProxyType

import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.error_handling import validate_value_function_array
from lcm.grids import Grid, IrregSpacedGrid, ShockGrid
from lcm.interfaces import (
    InternalRegime,
    StateActionSpace,
)
from lcm.shocks import SHOCK_GRIDPOINT_FUNCTIONS
from lcm.typing import FlatRegimeParams, FloatND, InternalParams, RegimeName


def solve(
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        internal_params: Dict of model parameters.
        ages: Age grid for the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.

    Returns:
        Dict with one value function array per period.

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
            state_action_space = _replace_dynamic_states(
                internal_regime.state_action_space,
                internal_params[name],
                internal_regime.gridspecs,
            )
            max_Q_over_a = internal_regime.max_Q_over_a_functions[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_V_arr=next_V_arr,
                **internal_params[name],
            )

            validate_value_function_array(V_arr, age=ages.values[period])
            period_solution[name] = V_arr

        next_V_arr = MappingProxyType(period_solution)
        solution[period] = next_V_arr
        logger.info("Age: %s", ages.values[period])

    return MappingProxyType(solution)


def _replace_dynamic_states(
    state_action_space: StateActionSpace,
    params: FlatRegimeParams,
    gridspecs: MappingProxyType[str, Grid],
) -> StateActionSpace:
    """Replace placeholder states with dynamic grid values from params.

    For dynamic IrregSpacedGrid, the grid points come from params as
    ``{state_name}__points``. For dynamic ShockGrid, the grid points are computed
    from shock params in the params dict.

    If dynamic params were already partialled via fixed_params (and thus not in
    ``params``), the grid is not replaced — it was already handled by the partialled
    compiled functions.

    """
    replacements: dict[str, object] = {}
    for state_name, spec in gridspecs.items():
        if state_name not in state_action_space.states:
            continue
        if isinstance(spec, IrregSpacedGrid) and spec.is_dynamic:
            points_key = f"{state_name}__points"
            if points_key not in params:
                continue
            replacements[state_name] = params[points_key]
        elif isinstance(spec, ShockGrid) and spec.dynamic_shock_params:
            # Check if all dynamic params are present (they might have been
            # partialled out via fixed_params)
            all_present = all(
                f"{state_name}__{p}" in params for p in spec.dynamic_shock_params
            )
            if not all_present:
                continue
            shock_kw = dict(spec.shock_params)
            for p in spec.dynamic_shock_params:
                shock_kw[p] = params[f"{state_name}__{p}"]
            replacements[state_name] = SHOCK_GRIDPOINT_FUNCTIONS[
                spec.distribution_type
            ](spec.n_points, **shock_kw)

    if not replacements:
        return state_action_space

    new_states = dict(state_action_space.states) | replacements
    return state_action_space.replace(states=MappingProxyType(new_states))
