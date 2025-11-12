from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, vmap

from lcm.error_handling import validate_value_function_array
from lcm.interfaces import (
    InternalRegime,
    SimulationResults,
)
from lcm.random import draw_random_seed
from lcm.simulation.processing import as_panel, process_simulated_data
from lcm.simulation.util import (
    calculate_next_regime_membership,
    calculate_next_states,
    create_regime_state_action_space,
    get_regime_name_to_id_mapping,
)
from lcm.utils import flatten_regime_namespace

if TYPE_CHECKING:
    import logging

    import pandas as pd

    from lcm.typing import FloatND, IntND, ParamsDict, RegimeName


def simulate(
    params: dict[RegimeName, ParamsDict],
    initial_states: dict[RegimeName, dict[str, Array]],
    initial_regimes: list[RegimeName],
    internal_regimes: dict[RegimeName, InternalRegime],
    logger: logging.Logger,
    V_arr_dict: dict[int, dict[RegimeName, FloatND]],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> dict[RegimeName, pd.DataFrame]:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: List of initial states to start from. Typically from the
            observed dataset.
        internal_regimes: Dict of internal regime instances.
        initial_regimes: List containing the names of the regimes the subjects start in.
        logger: Logger that logs to stdout.
        V_arr_dict: Dict of value function arrays of length n_periods.
        additional_targets: List of targets to compute. If provided, the targets
            are computed and added to the simulation results.
        seed: Random number seed; will be passed to `jax.random.key`. If not provided,
            a random seed will be generated.

    Returns:
        DataFrame with the simulation results.

    """
    if seed is None:
        seed = draw_random_seed()

    logger.info("Starting simulation")

    # Preparations
    # ----------------------------------------------------------------------------------
    regime_name_to_id = get_regime_name_to_id_mapping(internal_regimes)

    # The following variables are updated during the forward simulation
    states = flatten_regime_namespace(initial_states)
    subject_regime_ids = jnp.asarray(
        [regime_name_to_id[initial_regime] for initial_regime in initial_regimes]
    )

    n_periods = len(V_arr_dict)
    n_initial_subjects = subject_regime_ids.shape[0]
    key = jax.random.key(seed=seed)

    # Forward simulation
    # ----------------------------------------------------------------------------------
    simulation_results: dict[str, dict[int, SimulationResults]] = {
        regime_name: {} for regime_name in internal_regimes
    }
    for period in range(n_periods):
        logger.info("Period: %s", period)

        is_last_period = period == n_periods - 1

        new_subject_regime_ids = jnp.empty(n_initial_subjects)
        for regime_name, internal_regime in internal_regimes.items():
            # Select Subjects that are in the current regime
            subjects_in_regime = jnp.nonzero(
                regime_name_to_id[regime_name] == subject_regime_ids
            )[0]

            state_action_space = create_regime_state_action_space(
                regime_name=regime_name,
                states=states,
                internal_regime=internal_regime,
                subjects_in_regime=subjects_in_regime,
                is_last_period=is_last_period,
            )

            # Compute optimal actions indices and convert to values
            # --------------------------------------------------------------------------
            # We need to pass the value function array of the next period to the
            # argmax_and_max_Q_over_a function, as the current Q-function requires the
            # next periods's value funciton. In the last period, we pass an empty dict.
            next_V_arr = V_arr_dict.get(period + 1, {})

            # The Q-function values contain the information of how much value each
            # action combination is worth. To find the optimal discrete action, we
            # therefore only need to maximize the Q-function values over all actions.
            argmax_and_max_Q_over_a = internal_regime.argmax_and_max_Q_over_a_functions(
                period, n_periods=n_periods
            )

            indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
                **state_action_space.states,
                **state_action_space.discrete_actions,
                **state_action_space.continuous_actions,
                period=period,
                next_V_arr=next_V_arr,
                params=params,
            )

            validate_value_function_array(V_arr, period=period)

            optimal_actions = _lookup_values_from_indices(
                flat_indices=indices_optimal_actions,
                grids=state_action_space.actions,
            )

            # Store results
            # --------------------------------------------------------------------------
            regime_states = {
                sn: states[f"{regime_name}__{sn}"][subjects_in_regime]
                for sn in state_action_space.states
            }
            simulation_results[regime_name][period] = SimulationResults(
                V_arr=V_arr,
                actions=optimal_actions,
                states=regime_states,
                subject_ids=subjects_in_regime,
            )

            # Update states
            # --------------------------------------------------------------------------
            if not is_last_period:
                next_states_key, next_regime_key, key = jax.random.split(key, 3)

                # Calculate next states
                next_states = calculate_next_states(
                    internal_regime=internal_regime,
                    subjects_in_regime=subjects_in_regime,
                    optimal_actions=optimal_actions,
                    period=period,
                    params=params[regime_name],
                    states=states,
                    state_action_space=state_action_space,
                    key=next_states_key,
                )
                # Update states
                states = next_states

                # Calculate next regime membership
                next_regimes = calculate_next_regime_membership(
                    internal_regime=internal_regime,
                    subjects_in_regime=subjects_in_regime,
                    optimal_actions=optimal_actions,
                    period=period,
                    params=params[regime_name],
                    state_action_space=state_action_space,
                    new_subject_regime_ids=new_subject_regime_ids,
                    regime_name_to_id=regime_name_to_id,
                    key=next_regime_key,
                )
                new_subject_regime_ids = next_regimes

        subject_regime_ids = new_subject_regime_ids

    processed = {}
    for regime_name, regime_simulation_results in simulation_results.items():
        _processed = process_simulated_data(
            regime_simulation_results,
            internal_regime=internal_regime,
            params=params,
            additional_targets=additional_targets,
        )

        processed[regime_name] = as_panel(_processed)
    return processed


def _lookup_values_from_indices(
    flat_indices: IntND,
    grids: dict[str, Array],
) -> dict[str, Array]:
    """Retrieve values from indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Dictionary of grid values.

    Returns:
        Dictionary of values.

    """
    grids_shapes = tuple(len(grid) for grid in grids.values())

    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
    }


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))
