import logging
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax import Array, vmap

from lcm.ages import AgeGrid
from lcm.error_handling import validate_value_function_array
from lcm.interfaces import (
    InternalRegime,
    PeriodRegimeSimulationData,
    merge_cross_boundary_params,
)
from lcm.random import draw_random_seed
from lcm.simulation.result import SimulationResult
from lcm.simulation.utils import (
    calculate_next_regime_membership,
    calculate_next_states,
    convert_initial_states_to_nested,
    create_regime_state_action_space,
)
from lcm.typing import (
    FloatND,
    Int1D,
    InternalParams,
    IntND,
    RegimeName,
    RegimeNamesToIds,
)
from lcm.utils import flatten_regime_namespace


def simulate(
    *,
    internal_params: InternalParams,
    initial_states: Mapping[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    regime_names_to_ids: RegimeNamesToIds,
    logger: logging.Logger,
    V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    ages: AgeGrid,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        initial_states: Flat mapping of state names to arrays. All arrays must have
            the same length (number of subjects). Each state name should correspond to
            a state variable defined in at least one regime.
            Example: {"wealth": jnp.array([10.0, 50.0]), "health": jnp.array([0, 1])}
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        initial_regimes: List of regime names the subjects start in.
        logger: Logger that logs to stdout.
        V_arr_dict: Immutable mapping of periods to regime value function arrays.
        ages: AgeGrid for the model, used to convert periods to ages.
        seed: Random number seed; will be passed to `jax.random.key`. If not provided,
            a random seed will be generated.

    Returns:
        SimulationResult object. Call .to_dataframe() to get a pandas DataFrame.

    """
    if seed is None:
        seed = draw_random_seed()

    internal_params = merge_cross_boundary_params(internal_params, internal_regimes)

    logger.info("Starting simulation")

    # Convert flat initial_states to nested format
    # ----------------------------------------------------------------------------------
    nested_initial_states = convert_initial_states_to_nested(
        initial_states=initial_states, internal_regimes=internal_regimes
    )

    # Preparations
    # ----------------------------------------------------------------------------------
    key = jax.random.key(seed=seed)

    # The following variables are updated during the forward simulation
    states = MappingProxyType(flatten_regime_namespace(nested_initial_states))
    subject_regime_ids = jnp.asarray(
        [regime_names_to_ids[initial_regime] for initial_regime in initial_regimes]
    )

    # Forward simulation
    # ----------------------------------------------------------------------------------
    simulation_results: dict[RegimeName, dict[int, PeriodRegimeSimulationData]] = {
        regime_name: {} for regime_name in internal_regimes
    }
    for period, age in enumerate(ages.values):
        logger.info("Age: %s", age)

        new_subject_regime_ids = subject_regime_ids

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        active_regimes_next_period = tuple(
            regime_name
            for regime_name, regime in internal_regimes.items()
            if period + 1 in regime.active_periods
        )

        for regime_name, internal_regime in active_regimes.items():
            result, new_states, new_subject_regime_ids, key = (
                _simulate_regime_in_period(
                    regime_name=regime_name,
                    internal_regime=internal_regime,
                    period=period,
                    age=age,
                    states=states,
                    subject_regime_ids=subject_regime_ids,
                    new_subject_regime_ids=new_subject_regime_ids,
                    V_arr_dict=V_arr_dict,
                    internal_params=internal_params,
                    regime_names_to_ids=regime_names_to_ids,
                    active_regimes_next_period=active_regimes_next_period,
                    key=key,
                )
            )
            states = new_states
            simulation_results[regime_name][period] = result

        subject_regime_ids = new_subject_regime_ids

    # Wrap results in MappingProxyType for immutability
    wrapped_results = MappingProxyType(
        {
            regime_name: MappingProxyType(period_results)
            for regime_name, period_results in simulation_results.items()
        }
    )

    return SimulationResult(
        raw_results=wrapped_results,
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        V_arr_dict=V_arr_dict,
        ages=ages,
    )


def _simulate_regime_in_period(
    *,
    regime_name: RegimeName,
    internal_regime: InternalRegime,
    period: int,
    age: float,
    states: MappingProxyType[str, Array],
    subject_regime_ids: Int1D,
    new_subject_regime_ids: Int1D,
    V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    internal_params: InternalParams,
    regime_names_to_ids: MappingProxyType[RegimeName, int],
    active_regimes_next_period: tuple[RegimeName, ...],
    key: Array,
) -> tuple[PeriodRegimeSimulationData, MappingProxyType[str, Array], Int1D, Array]:
    """Simulate one regime for one period.

    This function processes all subjects in a given regime for a single period,
    computing optimal actions, updating states, and determining next regime membership.

    Args:
        regime_name: Name of the current regime.
        internal_regime: Internal representation of the regime.
        period: Current period (0-indexed).
        age: Age corresponding to current period.
        states: Current states for all subjects (namespaced by regime).
        subject_regime_ids: Current regime membership for all subjects.
        new_subject_regime_ids: Array to populate with next period's regime memberships.
        V_arr_dict: Value function arrays for all periods and regimes.
        internal_params: Model parameters for all regimes.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key for stochastic operations.

    Returns:
        Tuple containing:
        - PeriodRegimeData for this regime-period
        - Updated states dictionary
        - Updated new_subject_regime_ids array
        - Updated JAX random key

    """
    # Select subjects in the current regime
    # ---------------------------------------------------------------------------------
    subject_ids_in_regime = jnp.asarray(
        regime_names_to_ids[regime_name] == subject_regime_ids
    )

    state_action_space = create_regime_state_action_space(
        internal_regime=internal_regime,
        states=states,
    )
    # Compute optimal actions
    # ---------------------------------------------------------------------------------
    # We need to pass the value function array of the next period to the
    # argmax_and_max_Q_over_a function, as the current Q-function requires the
    # next period's value function. In the last period, we pass an empty dict.
    next_V_arr = V_arr_dict.get(period + 1, MappingProxyType({}))

    # The Q-function values contain the information of how much value each
    # action combination is worth. To find the optimal discrete action, we
    # therefore only need to maximize the Q-function values over all actions.
    argmax_and_max_Q_over_a = internal_regime.argmax_and_max_Q_over_a_functions[period]

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=next_V_arr,
        **internal_params[regime_name],
    )
    validate_value_function_array(V_arr=V_arr, age=age)

    optimal_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
    )
    # Store results for this regime-period
    # ---------------------------------------------------------------------------------
    # For state-less regimes (e.g., terminal regimes with no states), V_arr may be a
    # scalar. We need to broadcast it to match the number of subjects.
    n_subjects = subject_ids_in_regime.shape[0]
    if V_arr.ndim == 0:
        V_arr = jnp.broadcast_to(V_arr, (n_subjects,))

    res = {
        state_name.removeprefix(f"{regime_name}__"): state
        for state_name, state in states.items()
        if state_name.startswith(f"{regime_name}__")
    }

    simulation_result = PeriodRegimeSimulationData(
        V_arr=V_arr,
        actions=optimal_actions,
        states=MappingProxyType(res),
        in_regime=subject_ids_in_regime,
    )

    # Update states and regime membership for next period
    # ---------------------------------------------------------------------------------
    if not internal_regime.terminal:
        next_states_key, next_regime_key, key = jax.random.split(key, 3)

        next_states = calculate_next_states(
            internal_regime=internal_regime,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=internal_params[regime_name],
            states=states,
            state_action_space=state_action_space,
            key=next_states_key,
            subjects_in_regime=subject_ids_in_regime,
        )
        states = next_states
        new_subject_regime_ids = calculate_next_regime_membership(
            internal_regime=internal_regime,
            state_action_space=state_action_space,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=internal_params[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            new_subject_regime_ids=new_subject_regime_ids,
            active_regimes_next_period=active_regimes_next_period,
            key=next_regime_key,
            subjects_in_regime=subject_ids_in_regime,
        )

    return simulation_result, states, new_subject_regime_ids, key


def _lookup_values_from_indices(
    *,
    flat_indices: IntND,
    grids: MappingProxyType[str, Array],
) -> MappingProxyType[str, Array]:
    """Retrieve values from indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Immutable mapping of grid values.

    Returns:
        Read-only mapping of values.

    """
    # Handle empty grids case (no actions)
    if not grids:
        return MappingProxyType({})

    grids_shapes = tuple(len(grid) for grid in grids.values())

    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return MappingProxyType(
        {
            name: grid[index]
            for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
        }
    )


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))
