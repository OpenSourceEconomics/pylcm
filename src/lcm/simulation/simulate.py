from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from dags.tree import flatten_to_qnames
from jax import Array, vmap

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.error_handling import validate_value_function_array
from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import (
    InternalRegime,
    InternalSimulationPeriodResults,
    StateActionSpace,
)
from lcm.random import draw_random_seed, generate_simulation_keys
from lcm.simulation.processing import as_panel, process_simulated_data
from lcm.state_action_space import create_state_action_space

if TYPE_CHECKING:
    import logging

    import pandas as pd

    from lcm.typing import (
        FloatND,
        IntND,
        ParamsDict,
        Period,
    )


def simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    initial_regimes: list[str],
    internal_regimes: dict[str, InternalRegime],
    logger: logging.Logger,
    V_arr_dict: dict[int, FloatND],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: List of initial states to start from. Typically from the
            observed dataset.
        internal_regimes: Dict of internal regime instances.
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

    # The following variables are updated during the forward simulation
    states = flatten_to_qnames(initial_states)
    regime_mapping = {
        regime.name: i
        for regime, i in zip(
            internal_regimes.values(), range(len(internal_regimes)), strict=False
        )
    }
    subject_regime_ids = jnp.asarray(
        [regime_mapping[initial_regime] for initial_regime in initial_regimes]
    )

    # Preparations
    # ----------------------------------------------------------------------------------
    n_periods = len(V_arr_dict)
    n_initial_states = subject_regime_ids.shape[0]
    key = jax.random.key(seed=seed)

    # Forward simulation
    # ----------------------------------------------------------------------------------
    simulation_results = {regime_name: {} for regime_name in internal_regimes}
    regime_results = {}
    for period in range(n_periods):
        logger.info("Period: %s", period)

        is_last_period = period == n_periods - 1

        if is_last_period:
            query = "is_state and enters_concurrent_valuation"
        else:
            query = "is_state and (enters_concurrent_valuation | enters_transition)"

        new_subject_regime_ids = jnp.zeros(n_initial_states)
        for regime_name, internal_regime in internal_regimes.items():
            # Select Subjects that are in the current regime
            subjects_in_curr_regime = jnp.nonzero(
                regime_mapping[regime_name] == subject_regime_ids
            )[0]

            # Create state action space with current subjects
            states_for_state_action_space = {
                state_name: states[f"{regime_name}__{state_name}"][
                    subjects_in_curr_regime
                ]
                for state_name in internal_regime.variable_info.query(query).index
            }
            state_action_space = create_state_action_space(
                variable_info=internal_regime.variable_info,
                grids=internal_regime.grids,
                states=states_for_state_action_space,
                is_last_period=is_last_period,
            )

            # Get action grids
            discrete_actions_grid_shape = tuple(
                len(grid) for grid in state_action_space.discrete_actions.values()
            )
            continuous_actions_grid_shape = tuple(
                len(grid) for grid in state_action_space.continuous_actions.values()
            )
            actions_grid_shape = (
                discrete_actions_grid_shape + continuous_actions_grid_shape
            )

            # Compute optimal actions
            # ------------------------------------------------------------------------------
            # We need to pass the value function array of the next period to the
            # argmax_and_max_Q_over_a function, as the current Q-function requires the next
            # periods's value funciton. In the last period, we pass an empty array.
            next_V_arr = V_arr_dict.get(period + 1, jnp.empty(0))

            argmax_and_max_Q_over_a = simulation_spacemap(
                internal_regime.argmax_and_max_Q_over_a_functions(
                    period, n_periods=n_periods
                ),
                actions_names=(),
                states_names=tuple(state_action_space.states),
            )

            # The Q-function values contain the information of how much value each action
            # combination is worth. To find the optimal discrete action, we therefore only
            # need to maximize the Q-function values over all actions.
            # ------------------------------------------------------------------------------
            indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
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

            # Convert action indices to action values
            # ------------------------------------------------------------------------------
            optimal_actions = _lookup_actions_from_indices(
                indices_optimal_actions=indices_optimal_actions,
                actions_grid_shape=actions_grid_shape,
                state_action_space=state_action_space,
            )
            # Store results
            # ------------------------------------------------------------------------------
            curr_state_action_space = (
                internal_regime.state_action_spaces.terminal
                if is_last_period
                else internal_regime.state_action_spaces.non_terminal
            )
            result_relevant_states = {
                k: states[f"{regime_name}__{k}"][subjects_in_curr_regime]
                for k in curr_state_action_space.states
            }
            simulation_results[regime_name][period] = InternalSimulationPeriodResults(
                value=V_arr,
                actions=optimal_actions,
                states=result_relevant_states,
                subject_ids=subjects_in_curr_regime,
            )
            # Update states
            # ------------------------------------------------------------------------------
            if not is_last_period:
                stochastic_next_function_names = [
                    next_fn_name
                    for next_fn_name, next_fn in flatten_to_qnames(
                        internal_regime.transitions
                    ).items()
                    if is_stochastic_transition(next_fn)
                ]
                key, stochastic_variables_keys = generate_simulation_keys(
                    key=key,
                    names=stochastic_next_function_names,
                    n_initial_states=subjects_in_curr_regime.shape[0],
                )
                next_state_vmapped = internal_regime.next_state_simulation_function
                signature = inspect.signature(next_state_vmapped)
                parameters = set(signature.parameters)
                next_vars: dict[str, Array | Period | ParamsDict] = (
                    state_action_space.states  # type: ignore[assignment]
                    | optimal_actions
                    | stochastic_variables_keys
                    | {"period": period, "params": params[regime_name]}
                )

                next_state_input = {
                    parameter: next_vars[parameter] for parameter in parameters
                }

                states_with_next_prefix = next_state_vmapped(**next_state_input)

                # 'next_' prefix is added by the next_state function, but needs to be
                # removed for the next iteration of the loop, where these will be the
                # current states.
                states = {
                    k.replace("next_", ""): states[k.replace("next_", "")]
                    .at[subjects_in_curr_regime]
                    .set(v)
                    for k, v in states_with_next_prefix.items()
                }

                # Update regime
                # --------------------------------------------------------------------------
                signature = inspect.signature(
                    internal_regime.regime_transition_probs["simulate"]
                )
                parameters = set(signature.parameters)
                next_regime_input = {
                    parameter: next_vars[parameter] for parameter in parameters
                }
                _regime_transition_probs = internal_regime.regime_transition_probs[
                    "simulate"
                ](**next_regime_input)

                key, regime_transition_key = generate_simulation_keys(
                    key=key,
                    names=["regime_transition"],
                    n_initial_states=subjects_in_curr_regime.shape[0],
                )
                new_regimes = draw_key_from_dict(
                    d=_regime_transition_probs,
                    keys=regime_transition_key["key_regime_transition"],
                    regime_mapping=regime_mapping,
                )
                print(new_regimes)
                print(new_subject_regime_ids)
                new_subject_regime_ids = new_subject_regime_ids.at[
                    subjects_in_curr_regime
                ].set(new_regimes)
                print(new_subject_regime_ids)

        regime_results[period] = new_subject_regime_ids
        subject_regime_ids = new_subject_regime_ids

    processed = {}
    for regime_name, regime_simulation_results in simulation_results.items():
        _processed = process_simulated_data(
            regime_simulation_results,
            internal_regime=internal_regime,
            params=params,
            additional_targets=additional_targets,
        )

        processed[regime_name] = as_panel(_processed, n_periods=n_periods)
    return processed


@partial(vmap_1d, variables=("indices_argmax_Q_over_c", "discrete_argmax"))
def _lookup_optimal_continuous_actions(
    indices_argmax_Q_over_c: IntND,
    discrete_argmax: IntND,
    discrete_actions_grid_shape: tuple[int, ...],
) -> IntND:
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
    indices_optimal_actions: IntND,
    actions_grid_shape: tuple[int, ...],
    state_action_space: StateActionSpace,
) -> dict[str, Array]:
    """Lookup optimal actions from indices.

    Args:
        indices_optimal_actions: Indices of optimal actions.
        actions_grid_shape: Shape of the actions grid.
        state_action_space: StateActionSpace instance.

    Returns:
        Dictionary of optimal actions.

    """
    return _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.discrete_actions
        | state_action_space.continuous_actions,
        grids_shapes=actions_grid_shape,
    )


def _lookup_values_from_indices(
    flat_indices: IntND,
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


def draw_key_from_dict(
    d: dict[str, Array], regime_mapping: dict[str, int], keys: Array
) -> list[str]:
    """Draw a random key from a dictionary of arrays.

    Args:
        d: Dictionary of arrays, all of the same length. The values in the arrays
            represent a probability distribution over the keys. That is, for the
            dictionary {'regime1': jnp.array([0.2, 0.5]), 'regime2': jnp.array([0.8, 0.5])},
            0.2 + 0.8 = 1.0 and 0.5 + 0.5 = 1.0.
        keys: JAX random keys.

    Returns:
        A random key from the dictionary for each entry in the arrays.

    """
    regime_ids = jnp.array([regime_mapping[key] for key in d])

    def draw_single_key(
        key: Array,
        p: Array,
    ) -> str:
        return jax.random.choice(
            key,
            regime_ids,
            p=p,
        )

    draw_key = vmap(draw_single_key, in_axes=(0, 0))
    draw = draw_key(keys, jnp.array(list(d.values())).T)
    return draw
