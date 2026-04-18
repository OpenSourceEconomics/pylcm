import logging
import time
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array, vmap

from lcm.ages import AgeGrid
from lcm.interfaces import (
    InternalRegime,
    PeriodRegimeSimulationData,
)
from lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
)
from lcm.simulation.random import draw_random_seed
from lcm.simulation.result import SimulationResult
from lcm.simulation.transitions import (
    calculate_next_regime_membership,
    calculate_next_states,
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
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_header,
    log_period_timing,
    log_regime_transitions,
)


def simulate(
    *,
    internal_params: InternalParams,
    initial_conditions: Mapping[str, Array],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    regime_names_to_ids: RegimeNamesToIds,
    logger: logging.Logger,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    ages: AgeGrid,
    simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    seed: int | None = None,
) -> SimulationResult:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        initial_conditions: Flat mapping of state names (plus `"regime"`) to arrays.
            All arrays must have the same length (number of subjects). The `"regime"`
            entry must contain integer regime codes.
            Example: {"wealth": jnp.array([10.0, 50.0]), "regime": jnp.array([0, 0])}
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        logger: Logger that logs to stdout.
        period_to_regime_to_V_arr: Immutable mapping of periods to regime
            value function arrays.
        ages: AgeGrid for the model, used to convert periods to ages.
        simulation_output_dtypes: Mapping of variable name to `pd.CategoricalDtype`,
            used for building simulation metadata.
        seed: Random number seed; will be passed to `jax.random.key`. If not provided,
            a random seed will be generated.

    Returns:
        SimulationResult object. Call .to_dataframe() to get a pandas DataFrame.

    """
    if seed is None:
        seed = draw_random_seed()

    logger.info("Starting simulation")
    total_start = time.monotonic()

    # Extract state arrays from initial conditions, which include the regime on top.
    initial_states = {k: v for k, v in initial_conditions.items() if k != "regime"}

    # Preparations
    key = jax.random.key(seed=seed)

    # The following variables are updated during the forward simulation
    states = build_initial_states(
        initial_states=initial_states, internal_regimes=internal_regimes
    )
    starting_periods = _compute_starting_periods(
        initial_ages=initial_states["age"], ages=ages
    )
    subject_regime_ids = jnp.full_like(initial_conditions["regime"], MISSING_CAT_CODE)
    subject_ids = jnp.arange(initial_conditions["regime"].shape[0], dtype=jnp.int32)

    # Forward simulation
    simulation_results: dict[RegimeName, dict[int, PeriodRegimeSimulationData]] = {
        regime_name: {} for regime_name in internal_regimes
    }
    # Build reverse lookup for regime transition logging
    ids_to_names: dict[int, str] = {v: k for k, v in regime_names_to_ids.items()}

    for period, age in enumerate(ages.values):
        period_start = time.monotonic()

        # Activate subjects whose starting period matches the current period
        subject_regime_ids = jnp.where(
            starting_periods == period,
            initial_conditions["regime"],
            subject_regime_ids,
        )

        prev_regime_ids = subject_regime_ids
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

        log_period_header(logger=logger, age=age, n_active_regimes=len(active_regimes))

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
                    subject_ids=subject_ids,
                    period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                    internal_params=internal_params,
                    regime_names_to_ids=regime_names_to_ids,
                    active_regimes_next_period=active_regimes_next_period,
                    key=key,
                )
            )
            states = new_states
            simulation_results[regime_name][period] = result

            log_nan_in_V(
                logger=logger, regime_name=regime_name, age=age, V_arr=result.V_arr
            )

        subject_regime_ids = new_subject_regime_ids

        log_regime_transitions(
            logger=logger,
            prev_regime_ids=prev_regime_ids,
            new_regime_ids=subject_regime_ids,
            ids_to_names=ids_to_names,
        )

        elapsed = time.monotonic() - period_start
        log_period_timing(logger=logger, elapsed=elapsed)

    total_elapsed = time.monotonic() - total_start
    logger.info("Simulation complete  (%s)", format_duration(seconds=total_elapsed))

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
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        ages=ages,
        simulation_output_dtypes=simulation_output_dtypes,
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
    subject_ids: Int1D,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
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
        subject_ids: Global subject-id array, stored verbatim on the returned
            `PeriodRegimeSimulationData` so downstream concatenation (across
            partition groups) can restore subject ordering.
        period_to_regime_to_V_arr: Value function arrays for all periods and regimes.
        internal_params: Model parameters for all regimes.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key for stochastic operations.

    Returns:
        Tuple containing:
        - PeriodRegimeSimulationData for this regime-period
        - Updated states dictionary
        - Updated new_subject_regime_ids array
        - Updated JAX random key

    """
    # Select subjects in the current regime
    subject_ids_in_regime = jnp.asarray(
        regime_names_to_ids[regime_name] == subject_regime_ids
    )

    state_action_space = create_regime_state_action_space(
        internal_regime=internal_regime,
        states=states,
    )
    # Compute optimal actions
    # We need to pass the value function array of the next period to the
    # argmax_and_max_Q_over_a function, as the current Q-function requires the
    # next period's value function. In the last period, we pass an empty dict.
    next_regime_to_V_arr = period_to_regime_to_V_arr.get(
        period + 1, MappingProxyType({})
    )

    # The Q-function values contain the information of how much value each
    # action combination is worth. To find the optimal discrete action, we
    # therefore only need to maximize the Q-function values over all actions.
    argmax_and_max_Q_over_a = (
        internal_regime.simulate_functions.argmax_and_max_Q_over_a[period]
    )

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **internal_params[regime_name],
        period=jnp.int32(period),
        age=age,
    )
    validate_V(V_arr=V_arr, age=age, regime_name=regime_name)

    optimal_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
    )
    # Store results for this regime-period
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
        subject_ids=subject_ids,
    )

    # Update states and regime membership for next period
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


def _compute_starting_periods(
    *,
    initial_ages: Array,
    ages: AgeGrid,
) -> Int1D:
    """Convert per-subject initial ages to starting period indices.

    Args:
        initial_ages: Array of initial ages for each subject.
        ages: AgeGrid defining the lifecycle.

    Returns:
        Array of starting period indices (one per subject).

    Raises:
        ValueError: If any initial age is not a valid age grid point.

    """
    age_values = jnp.asarray(ages.values)
    starting_periods = jnp.searchsorted(age_values, initial_ages)

    # Clamp indices to valid range before accessing age_values. searchsorted can
    # return len(age_values) for ages beyond the grid maximum.
    safe_idx = jnp.clip(starting_periods, 0, len(age_values) - 1)

    # Validate that all initial ages are actual grid points. Use isclose instead
    # of strict equality to handle floating-point representation of sub-annual ages.
    in_bounds = starting_periods < len(age_values)
    valid = in_bounds & jnp.isclose(age_values[safe_idx], initial_ages)
    if not jnp.all(valid):
        invalid_ages = initial_ages[~valid]
        msg = (
            f"Initial ages {invalid_ages.tolist()} are not valid age grid points. "
            f"Valid ages: {ages.values}."  # noqa: PD011
        )
        raise ValueError(msg)

    return starting_periods
