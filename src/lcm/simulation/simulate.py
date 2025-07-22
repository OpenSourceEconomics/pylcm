import logging
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array, vmap

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.interfaces import (
    InternalModel,
    InternalSimulationPeriodResults,
    StateActionSpace,
)
from lcm.max_Qc_over_d import get_argmax_and_max_Qc_over_d
from lcm.random import draw_random_seed, generate_simulation_keys
from lcm.simulation.processing import as_panel, process_simulated_data
from lcm.state_action_space import create_state_action_space
from lcm.typing import ArgmaxQOverCFunction, ParamsDict


def solve_and_simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    argmax_and_max_Q_over_c_functions: dict[int, ArgmaxQOverCFunction],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    solve_model: Callable[..., dict[int, Array]],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """First solve the model and then simulate the model forward in time.

    Same docstring as `simulate` mutatis mutandis.

    """
    V_arr_dict = solve_model(params)
    return simulate(
        params=params,
        initial_states=initial_states,
        argmax_and_max_Q_over_c_functions=argmax_and_max_Q_over_c_functions,
        model=model,
        next_state=next_state,
        logger=logger,
        V_arr_dict=V_arr_dict,
        additional_targets=additional_targets,
        seed=seed,
    )


def simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    argmax_and_max_Q_over_c_functions: dict[int, ArgmaxQOverCFunction],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    V_arr_dict: dict[int, Array],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: List of initial states to start from. Typically from the
            observed dataset.
        argmax_and_max_Q_over_c_functions: Dict of functions of length n_periods. Each
            function calculates the argument maximizing Q over the continuous actions.
        next_state: Function that returns the next state given the current
            state and action variables. For stochastic variables, it returns a random
            draw from the distribution of the next state.
        model: Model instance.
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
    n_periods = len(V_arr_dict)
    n_initial_states = len(next(iter(initial_states.values())))

    state_action_space = create_state_action_space(
        model=model,
        initial_states=initial_states,
    )

    argmax_and_max_Qc_over_d = get_argmax_and_max_Qc_over_d(
        variable_info=model.variable_info
    )

    # The following variables are updated during the forward simulation
    states = initial_states
    key = jax.random.key(seed=seed)

    # Forward simulation
    # ----------------------------------------------------------------------------------
    simulation_results = {}

    for period in range(n_periods):
        state_action_space = state_action_space.replace(states)

        discrete_actions_grid_shape = tuple(
            len(grid) for grid in state_action_space.discrete_actions.values()
        )
        continuous_actions_grid_shape = tuple(
            len(grid) for grid in state_action_space.continuous_actions.values()
        )

        # Compute optimal continuous actions conditional on discrete actions
        # ------------------------------------------------------------------------------
        # We need to pass the value function array of the next period to the
        # argmax_and_max_Q_over_c function, as the current Q-function requires the next
        # periods's value funciton. In the last period, we pass an empty array.
        next_V_arr = V_arr_dict.get(period + 1, jnp.empty(0))

        argmax_and_max_Q_over_c = simulation_spacemap(
            argmax_and_max_Q_over_c_functions[period],
            actions_names=tuple(state_action_space.discrete_actions),
            states_names=tuple(state_action_space.states),
        )

        # Returns the optimal continuous action index conditional on the states and
        # discrete actions, as well as the maximum value.
        indices_argmax_Q_over_c, Qc_arr = argmax_and_max_Q_over_c(
            **state_action_space.states,
            **state_action_space.discrete_actions,
            **state_action_space.continuous_actions,
            next_V_arr=next_V_arr,
            params=params,
        )

        # The Qc-function values contain the information of how much value each discrete
        # action combination is worth, assuming the corresponding optimal continuous
        # actions are taken. To find the optimal discrete action, we therefore only need
        # to maximize the Qc-function values over the discrete actions.
        # ------------------------------------------------------------------------------
        indices_optimal_discrete_actions, V_arr = argmax_and_max_Qc_over_d(
            Qc_arr, params=params
        )

        # Look up the continuous actions index from the above set given the optimal
        # discrete actions.
        # ------------------------------------------------------------------------------
        indices_optimal_continuous_actions = _lookup_optimal_continuous_actions(
            indices_argmax_Q_over_c=indices_argmax_Q_over_c,
            discrete_argmax=indices_optimal_discrete_actions,
            discrete_actions_grid_shape=discrete_actions_grid_shape,
        )

        # Convert action indices to action values
        # ------------------------------------------------------------------------------
        optimal_actions = _lookup_actions_from_indices(
            indices_optimal_discrete_actions=indices_optimal_discrete_actions,
            indices_optimal_continuous_actions=indices_optimal_continuous_actions,
            discrete_actions_grid_shape=discrete_actions_grid_shape,
            continuous_actions_grid_shape=continuous_actions_grid_shape,
            state_action_space=state_action_space,
        )

        # Store results
        # ------------------------------------------------------------------------------
        simulation_results[period] = InternalSimulationPeriodResults(
            value=V_arr,
            actions=optimal_actions,
            states=states,
        )

        # Update states
        # ------------------------------------------------------------------------------
        key, stochastic_variables_keys = generate_simulation_keys(
            key=key,
            ids=model.function_info.query("is_stochastic_next").index.tolist(),
        )

        states_with_next_prefix = next_state(
            **states,
            **optimal_actions,
            _period=jnp.repeat(period, n_initial_states),
            params=params,
            keys=stochastic_variables_keys,
        )
        # 'next_' prefix is added by the next_state function, but needs to be removed
        # because in the next period, next states will be current states.
        states = {
            k.removeprefix("next_"): v for k, v in states_with_next_prefix.items()
        }

        logger.info("Period: %s", period)

    processed = process_simulated_data(
        simulation_results,
        model=model,
        params=params,
        additional_targets=additional_targets,
    )

    return as_panel(processed, n_periods=n_periods)


@partial(vmap_1d, variables=("indices_argmax_Q_over_c", "discrete_argmax"))
def _lookup_optimal_continuous_actions(
    indices_argmax_Q_over_c: Array,
    discrete_argmax: Array,
    discrete_actions_grid_shape: tuple[int, ...],
) -> Array:
    """Look up the optimal continuous action index given index of discrete action.

    Args:
        indices_argmax_Q_over_c: Index array of optimal continous actions conditional on
            discrete actions and states.
        discrete_argmax: Index array of optimal discrete actions.
        discrete_actions_grid_shape: Shape of the discrete actions grid.

    Returns:
        Index array of optimal continuous actions.

    """
    indices = jnp.unravel_index(discrete_argmax, shape=discrete_actions_grid_shape)
    return indices_argmax_Q_over_c[indices]


def _lookup_actions_from_indices(
    indices_optimal_discrete_actions: Array,
    indices_optimal_continuous_actions: Array,
    discrete_actions_grid_shape: tuple[int, ...],
    continuous_actions_grid_shape: tuple[int, ...],
    state_action_space: StateActionSpace,
) -> dict[str, Array]:
    """Lookup optimal actions from indices.

    Args:
        indices_optimal_discrete_actions: Indices of optimal discrete actions.
        indices_optimal_continuous_actions: Indices of optimal continuous actions.
        discrete_actions_grid_shape: Shape of the discrete actions grid.
        continuous_actions_grid_shape: Shape of the continuous actions grid.
        state_action_space: StateActionSpace instance.

    Returns:
        Dictionary of optimal actions.

    """
    optimal_discrete_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_discrete_actions,
        grids=state_action_space.discrete_actions,
        grids_shapes=discrete_actions_grid_shape,
    )

    optimal_continuous_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_continuous_actions,
        grids=state_action_space.continuous_actions,
        grids_shapes=continuous_actions_grid_shape,
    )

    return optimal_discrete_actions | optimal_continuous_actions


def _lookup_values_from_indices(
    flat_indices: Array,
    grids: dict[str, Array],
    grids_shapes: tuple[int, ...],
) -> dict[str, Array]:
    """Retrieve values from indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Dictionary of grid values.
        grids_shapes: Shape of the grids. Is used to unravel the index.

    Returns:
        Dictionary of values.

    """
    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
    }


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))
