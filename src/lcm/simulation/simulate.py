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
from lcm.simulation.processing import process_simulated_data
from lcm.simulation.util import (
    calculate_next_regime_membership,
    calculate_next_states,
    convert_flat_to_nested_initial_states,
    create_regime_state_action_space,
    get_regime_name_to_id_mapping,
    validate_flat_initial_states,
)
from lcm.utils import flatten_regime_namespace

if TYPE_CHECKING:
    import logging

    import pandas as pd

    from lcm.typing import (
        FloatND,
        Int1D,
        IntND,
        ParamsDict,
        RegimeName,
    )


def simulate(
    params: dict[RegimeName, ParamsDict],
    initial_states: dict[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: dict[RegimeName, InternalRegime],
    regime_id_cls: type,
    logger: logging.Logger,
    V_arr_dict: dict[int, dict[RegimeName, FloatND]],
    *,
    additional_targets: dict[RegimeName, list[str]] | None = None,
    seed: int | None = None,
) -> dict[RegimeName, pd.DataFrame]:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: Flat dict mapping state names to arrays. All arrays must have
            the same length (number of subjects). Each state name should correspond to
            a state variable defined in at least one regime.
            Example: {"wealth": jnp.array([10.0, 50.0]), "health": jnp.array([0, 1])}
        internal_regimes: Dict of internal regime instances.
        regime_id_cls: Dataclass mapping regime names to integer indices.
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

    # Validate and convert flat initial_states to nested format
    # ----------------------------------------------------------------------------------
    validate_flat_initial_states(initial_states, internal_regimes)
    nested_initial_states = convert_flat_to_nested_initial_states(
        initial_states, internal_regimes
    )

    # Preparations
    # ----------------------------------------------------------------------------------
    regime_name_to_id = get_regime_name_to_id_mapping(regime_id_cls)

    # The following variables are updated during the forward simulation
    states = flatten_regime_namespace(nested_initial_states)
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

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        for regime_name, internal_regime in active_regimes.items():
            result, new_states, new_subject_regime_ids, key = (
                _simulate_regime_in_period(
                    regime_name=regime_name,
                    internal_regime=internal_regime,
                    period=period,
                    states=states,
                    subject_regime_ids=subject_regime_ids,
                    new_subject_regime_ids=new_subject_regime_ids,
                    V_arr_dict=V_arr_dict,
                    params=params,
                    regime_name_to_id=regime_name_to_id,
                    key=key,
                    is_last_period=is_last_period,
                )
            )
            states = new_states
            simulation_results[regime_name][period] = result

        subject_regime_ids = new_subject_regime_ids

    processed = {}
    for regime_name, regime_simulation_results in simulation_results.items():
        processed[regime_name] = process_simulated_data(
            regime_simulation_results,
            internal_regime=internal_regimes[regime_name],
            params=params,
            additional_targets=additional_targets,
            n_initial_subjects=n_initial_subjects,
        )

    return processed


def _simulate_regime_in_period(
    regime_name: RegimeName,
    internal_regime: InternalRegime,
    period: int,
    states: dict[str, Array],
    subject_regime_ids: Int1D,
    new_subject_regime_ids: Int1D,
    V_arr_dict: dict[int, dict[RegimeName, FloatND]],
    params: dict[RegimeName, ParamsDict],
    regime_name_to_id: dict[RegimeName, int],
    key: Array,
    *,
    is_last_period: bool,
) -> tuple[SimulationResults, dict[str, Array], Int1D, Array]:
    """Simulate one regime for one period.

    This function processes all subjects in a given regime for a single period,
    computing optimal actions, updating states, and determining next regime membership.

    Args:
        regime_name: Name of the current regime.
        internal_regime: Internal representation of the regime.
        period: Current period (0-indexed).
        states: Current states for all subjects (namespaced by regime).
        subject_regime_ids: Current regime membership for all subjects.
        new_subject_regime_ids: Array to populate with next period's regime memberships.
        V_arr_dict: Value function arrays for all periods and regimes.
        params: Model parameters for all regimes.
        regime_name_to_id: Mapping from regime names to integer IDs.
        key: JAX random key for stochastic operations.
        is_last_period: Whether this is the final period (no state updates needed).

    Returns:
        Tuple containing:
        - SimulationResults for this regime-period
        - Updated states dictionary
        - Updated new_subject_regime_ids array
        - Updated JAX random key

    """
    # Select subjects in the current regime
    # ---------------------------------------------------------------------------------
    subject_ids_in_regime = jnp.asarray(
        regime_name_to_id[regime_name] == subject_regime_ids
    )

    state_action_space = create_regime_state_action_space(
        internal_regime=internal_regime,
        states=states,
        is_last_period=is_last_period,
    )
    # Compute optimal actions
    # ---------------------------------------------------------------------------------
    # We need to pass the value function array of the next period to the
    # argmax_and_max_Q_over_a function, as the current Q-function requires the
    # next period's value function. In the last period, we pass an empty dict.
    next_V_arr = V_arr_dict.get(period + 1, {})

    # The Q-function values contain the information of how much value each
    # action combination is worth. To find the optimal discrete action, we
    # therefore only need to maximize the Q-function values over all actions.
    argmax_and_max_Q_over_a = internal_regime.argmax_and_max_Q_over_a_functions[period]

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=next_V_arr,
        params=params,
    )
    validate_value_function_array(V_arr, period=period)

    optimal_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
    )
    # Store results for this regime-period
    # ---------------------------------------------------------------------------------

    res = {
        state_name.removeprefix(f"{regime_name}__"): state
        for state_name, state in states.items()
        if state_name.startswith(f"{regime_name}__")
    }

    simulation_result = SimulationResults(
        V_arr=V_arr,
        actions=optimal_actions,
        states=res,
        in_regime=subject_ids_in_regime,
    )

    # Update states and regime membership for next period
    # ---------------------------------------------------------------------------------
    if not internal_regime.terminal:
        next_states_key, next_regime_key, key = jax.random.split(key, 3)

        next_states = calculate_next_states(
            internal_regime=internal_regime,
            subjects_in_regime=subject_ids_in_regime,
            optimal_actions=optimal_actions,
            period=period,
            params=params[regime_name],
            states=states,
            state_action_space=state_action_space,
            key=next_states_key,
        )
        states = next_states
        new_subject_regime_ids = calculate_next_regime_membership(
            internal_regime=internal_regime,
            subjects_in_regime=subject_ids_in_regime,
            optimal_actions=optimal_actions,
            period=period,
            params=params[regime_name],
            state_action_space=state_action_space,
            new_subject_regime_ids=new_subject_regime_ids,
            regime_name_to_id=regime_name_to_id,
            key=next_regime_key,
        )

    return simulation_result, states, new_subject_regime_ids, key


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
