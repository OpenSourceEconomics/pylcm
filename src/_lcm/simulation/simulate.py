import logging
import time
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pandas as pd
from jax import vmap

from _lcm.engine import (
    PeriodRegimeSimulationData,
    Regime,
)
from _lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
    trim_pad_from_raw_results,
)
from _lcm.simulation.random import draw_random_seed
from _lcm.simulation.transitions import (
    calculate_next_regime_membership,
    calculate_next_states,
    create_regime_state_action_space,
)
from _lcm.solution.validate_V import validate_V
from _lcm.typing import (
    FlatParams,
    InitialConditions,
    PRNGKeyND,
    RegimeIdsToNames,
    RegimeName,
    RegimeNamesToIds,
    StateOrActionName,
    StatesPerRegime,
)
from _lcm.utils.containers import invert_regime_ids
from _lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_header,
    log_period_timing,
    log_regime_transitions,
    raise_or_warn,
    validation_enabled,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.result import SimulationResult
from lcm.typing import Float1D, FloatND, Int1D, IntND, ScalarFloat, ScalarInt


def simulate(
    *,
    flat_params: FlatParams,
    initial_conditions: InitialConditions,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    logger: logging.Logger,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    ages: AgeGrid,
    simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    seed: int | None = None,
    subject_batch_size: int = 0,
    original_n_subjects: int | None = None,
) -> SimulationResult:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        initial_conditions: Flat mapping of state names (plus `"regime_id"`) to
            arrays. All arrays must have the same length (number of subjects).
            The `"regime_id"` entry must contain integer regime codes.
            Example:
            {"wealth": jnp.array([10.0, 50.0]), "regime_id": jnp.array([0, 0])}
        regimes: Immutable mapping of regime names to internal regime
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
        subject_batch_size: Concrete subject chunk size, already resolved by the
            caller (`Model.simulate` maps the user-facing `0`/`>0` knob to
            an int here). `0` or a value `>= n_subjects` simulates the whole
            population in a single pass; a smaller value chunks the subjects,
            bounding the per-period device workspace at the cost of re-running the
            period loop per chunk. Results are invariant to this knob: per-subject
            RNG keys are generated for the full population and sliced by global
            index.
        original_n_subjects: Subject count before any per-device padding applied by
            `pad_initial_conditions_for_devices`. When set, RNG keys are sized to it
            and the trailing pad rows are trimmed from the results before they are
            returned. `None` means no padding was applied.

    Returns:
        SimulationResult object. Call .to_dataframe() to get a pandas DataFrame.

    """
    if seed is None:
        seed = draw_random_seed()

    logger.info("Starting simulation")
    total_start = time.monotonic()

    # Extract state arrays from initial conditions, which include the regime on top.
    initial_states = {k: v for k, v in initial_conditions.items() if k != "regime_id"}

    # Forward-simulate one subject chunk at a time. Subjects are independent across
    # the forward path (no cross-subject reduction), so each chunk runs the full
    # period loop on its own slice and the per-chunk results are concatenated on the
    # subject axis. Chunking bounds the per-period device workspace; the chunk size
    # is `subject_batch_size` (the whole population in one pass when `None`).
    n_subjects = int(initial_conditions["regime_id"].shape[0])
    batch_size = (
        n_subjects if subject_batch_size == 0 else min(subject_batch_size, n_subjects)
    )

    starting_periods = _compute_starting_periods(
        initial_ages=initial_states["age"], ages=ages
    )
    # Build reverse lookup for regime transition logging. `regime_names_to_ids`
    # values are `ScalarInt` (jax 0-d arrays), which can't serve as dict keys
    # directly; `invert_regime_ids` coerces them to Python `int`.
    regime_ids_to_names = invert_regime_ids(regime_names_to_ids)

    # When chunking, offload each chunk's results to host as it finishes so the
    # device frees them before the next chunk's period loop allocates — bounding
    # device residency to a single chunk. A single pass (batch_size == n_subjects)
    # keeps results on the compute device (no memory pressure, and no host
    # round-trip for downstream targets).
    host_device = jax.devices("cpu")[0] if batch_size < n_subjects else None

    chunk_results: list[dict[RegimeName, dict[int, PeriodRegimeSimulationData]]] = []
    for chunk_start in range(0, n_subjects, batch_size):
        # Every chunk runs exactly `batch_size` subjects so they all match the
        # program compiled for that shape. A short final chunk slides its window
        # back to end at the population tail; the overlapped subjects were already
        # simulated in the previous chunk and are recomputed bit-identically
        # (per-subject keys depend only on the global index), so trimming the
        # leading overlap leaves each subject represented once, in global order.
        slice_start = min(chunk_start, n_subjects - batch_size)
        n_overlap = chunk_start - slice_start
        subject_slice = slice(slice_start, slice_start + batch_size)
        chunk = _simulate_subject_chunk(
            initial_states={
                name: array[subject_slice] for name, array in initial_states.items()
            },
            initial_regime_ids=initial_conditions["regime_id"][subject_slice],
            starting_periods=starting_periods[subject_slice],
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_ids_to_names=regime_ids_to_names,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            flat_params=flat_params,
            ages=ages,
            seed=seed,
            logger=logger,
        )
        if n_overlap:
            chunk = _trim_chunk_subjects(chunk=chunk, n_overlap=n_overlap)
        if host_device is not None:
            # `block_until_ready` forces the D2H copy to complete before the loop
            # continues, so the chunk's device buffers become free for the next
            # chunk; the host-resident copies stay `jax.Array` (CPU-backed).
            chunk = jax.block_until_ready(jax.device_put(chunk, host_device))
        chunk_results.append(chunk)

    simulation_results = _concatenate_chunk_results(
        chunk_results=chunk_results, regimes=regimes
    )

    # Drain the per-period compute graph before returning. Mirrors solve's
    # `_drain_V_arr_shards`: simulation_results carries per (regime, period)
    # V_arrs / states / actions whose kernels may still be in flight when
    # the Python loop exits, especially at `log_level="off"` where no
    # per-period diagnostics force materialisation. `jax.block_until_ready`
    # walks the pytree and blocks per-shard (no host transfer, no cross-
    # device collective).
    jax.block_until_ready(simulation_results)

    total_elapsed = time.monotonic() - total_start
    logger.info("Simulation complete  (%s)", format_duration(seconds=total_elapsed))

    # Wrap results in MappingProxyType for immutability
    wrapped_results = MappingProxyType(
        {
            regime_name: MappingProxyType(period_results)
            for regime_name, period_results in simulation_results.items()
        }
    )

    # Drop any device-alignment pad rows so `SimulationResult` exposes only the
    # user's real subjects (see `pad_initial_conditions_for_devices`). No-op when
    # `original_n_subjects` already matched the dispatched leading axis.
    if original_n_subjects is not None:
        wrapped_results = trim_pad_from_raw_results(
            raw_results=wrapped_results,
            original_n_subjects=original_n_subjects,
        )

    return SimulationResult(
        raw_results=wrapped_results,
        regimes=regimes,
        flat_params=flat_params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        ages=ages,
        simulation_output_dtypes=simulation_output_dtypes,
        subject_batch_size=subject_batch_size,
    )


def _simulate_subject_chunk(
    *,
    initial_states: dict[StateOrActionName, Float1D | IntND],
    initial_regime_ids: Int1D,
    starting_periods: Int1D,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None = None,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_ids_to_names: RegimeIdsToNames,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    flat_params: FlatParams,
    ages: AgeGrid,
    seed: int,
    logger: logging.Logger,
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Run the full period loop for one chunk of subjects.

    `initial_states`, `initial_regime_ids`, and `starting_periods` are already
    sliced to this chunk's subjects; `n_subjects` and `subject_slice` describe the
    chunk's position in the full population so RNG keys stay full-population and are
    sliced by global index. The key stream is re-derived from `seed` here so the
    per-period carry is identical across chunks (it is subject-count-independent).

    Returns the per-(regime, period) results for this chunk's subjects.
    """
    key = jax.random.key(seed=seed)
    states = build_initial_states(initial_states=initial_states, regimes=regimes)
    subject_regime_ids = jnp.full_like(
        initial_regime_ids, MISSING_CAT_CODE, dtype=jnp.int32
    )

    simulation_results: dict[RegimeName, dict[int, PeriodRegimeSimulationData]] = {
        regime_name: {} for regime_name in regimes
    }

    for period, age in enumerate(ages.values):
        period_start = time.monotonic()

        # Activate subjects whose starting period matches the current period
        subject_regime_ids = jnp.where(
            starting_periods == period,
            initial_regime_ids,
            subject_regime_ids,
        )

        prev_regime_ids = subject_regime_ids
        new_subject_regime_ids = subject_regime_ids

        active_regimes = {
            regime_name: regime
            for regime_name, regime in regimes.items()
            if period in regime.active_periods
        }

        active_regimes_next_period = tuple(
            regime_name
            for regime_name, regime in regimes.items()
            if period + 1 in regime.active_periods
        )

        log_period_header(logger=logger, age=age, n_active_regimes=len(active_regimes))

        for regime_name, regime in active_regimes.items():
            result, new_states, new_subject_regime_ids, key = (
                _simulate_regime_in_period(
                    regime_name=regime_name,
                    regime=regime,
                    period=period,
                    age=age,
                    states=states,
                    subject_regime_ids=subject_regime_ids,
                    new_subject_regime_ids=new_subject_regime_ids,
                    period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                    flat_params=flat_params,
                    regime_names_to_ids=regime_names_to_ids,
                    active_regimes_next_period=active_regimes_next_period,
                    key=key,
                    logger=logger,
                    n_subjects=n_subjects,
                    subject_slice=subject_slice,
                    original_n_subjects=original_n_subjects,
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
            regime_ids_to_names=regime_ids_to_names,
        )

        elapsed = time.monotonic() - period_start
        log_period_timing(logger=logger, elapsed=elapsed)

    return simulation_results


def _trim_chunk_subjects(
    *,
    chunk: dict[RegimeName, dict[int, PeriodRegimeSimulationData]],
    n_overlap: int,
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Drop the first `n_overlap` subjects from every leaf of a chunk's results.

    A short final chunk overlaps the tail of the previous chunk to stay the
    compiled batch size. The overlapped subjects at the front duplicate that tail;
    dropping them leaves each subject represented once, in global order.
    """
    return {
        regime_name: {
            period: PeriodRegimeSimulationData(
                V_arr=data.V_arr[n_overlap:],
                actions=MappingProxyType(
                    {name: arr[n_overlap:] for name, arr in data.actions.items()}
                ),
                states=MappingProxyType(
                    {name: arr[n_overlap:] for name, arr in data.states.items()}
                ),
                in_regime=data.in_regime[n_overlap:],
            )
            for period, data in period_data.items()
        }
        for regime_name, period_data in chunk.items()
    }


def _concatenate_chunk_results(
    *,
    chunk_results: list[dict[RegimeName, dict[int, PeriodRegimeSimulationData]]],
    regimes: MappingProxyType[RegimeName, Regime],
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Concatenate per-chunk simulation results along the subject axis.

    Every chunk runs the full period loop, so each populates the same
    (regime, period) slots with arrays over its own subjects. Concatenating on
    axis 0 in chunk order reassembles the full population in global subject order.
    A single chunk (the unbatched case) is returned untouched.
    """
    if len(chunk_results) == 1:
        return chunk_results[0]

    combined: dict[RegimeName, dict[int, PeriodRegimeSimulationData]] = {
        regime_name: {} for regime_name in regimes
    }
    for regime_name, period_data in chunk_results[0].items():
        for period in period_data:
            per_chunk = [chunk[regime_name][period] for chunk in chunk_results]
            combined[regime_name][period] = PeriodRegimeSimulationData(
                V_arr=jnp.concatenate([data.V_arr for data in per_chunk]),
                actions=MappingProxyType(
                    {
                        name: jnp.concatenate(
                            [data.actions[name] for data in per_chunk]
                        )
                        for name in per_chunk[0].actions
                    }
                ),
                states=MappingProxyType(
                    {
                        name: jnp.concatenate([data.states[name] for data in per_chunk])
                        for name in per_chunk[0].states
                    }
                ),
                in_regime=jnp.concatenate([data.in_regime for data in per_chunk]),
            )
    return combined


def _simulate_regime_in_period(
    *,
    regime_name: RegimeName,
    regime: Regime,
    period: int,
    age: ScalarInt | ScalarFloat,
    states: StatesPerRegime,
    subject_regime_ids: Int1D,
    new_subject_regime_ids: Int1D,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    flat_params: FlatParams,
    regime_names_to_ids: RegimeNamesToIds,
    active_regimes_next_period: tuple[RegimeName, ...],
    key: PRNGKeyND,
    logger: logging.Logger,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None = None,
) -> tuple[PeriodRegimeSimulationData, StatesPerRegime, Int1D, PRNGKeyND]:
    """Simulate one regime for one period.

    This function processes all subjects in a given regime for a single period,
    computing optimal actions, updating states, and determining next regime membership.

    Args:
        regime_name: Name of the current regime.
        regime: Internal representation of the regime.
        period: Current period (0-indexed).
        age: Age corresponding to current period.
        states: Carrier of current-period state arrays for every regime and
            state.
        subject_regime_ids: Current regime membership for all subjects.
        new_subject_regime_ids: Array to populate with next period's regime memberships.
        period_to_regime_to_V_arr: Value function arrays for all periods and regimes.
        flat_params: Model parameters for all regimes.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key for stochastic operations.
        n_subjects: Total number of subjects (the full population), used to keep RNG
            key generation independent of how subjects are chunked.
        subject_slice: Global-index slice of the subjects in this chunk.

    Returns:
        Tuple containing:
        - PeriodRegimeSimulationData for this regime-period
        - Updated state carrier
        - Updated new_subject_regime_ids array
        - Updated JAX random key

    """
    # Select subjects in the current regime
    subject_ids_in_regime = jnp.asarray(
        regime_names_to_ids[regime_name] == subject_regime_ids
    )

    state_action_space = create_regime_state_action_space(
        regime=regime,
        regime_states=states[regime_name],
        regime_params=flat_params[regime_name],
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
    argmax_and_max_Q_over_a = regime.simulate_functions.argmax_and_max_Q_over_a[period]

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **flat_params[regime_name],
        period=jnp.int32(period),
        age=age,
    )
    if validation_enabled(logger):
        try:
            validate_V(V_arr=V_arr, age=age, regime_name=regime_name)
        except InvalidValueFunctionError as error:
            raise_or_warn(logger=logger, error=error)

    optimal_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
    )
    # Store results for this regime-period
    # For state-less regimes (e.g., terminal regimes with no states), V_arr may be a
    # scalar. We need to broadcast it to match this chunk's subject count.
    n_chunk_subjects = subject_ids_in_regime.shape[0]
    if V_arr.ndim == 0:
        V_arr = jnp.broadcast_to(V_arr, (n_chunk_subjects,))

    simulation_result = PeriodRegimeSimulationData(
        V_arr=V_arr,
        actions=optimal_actions,
        states=states[regime_name],
        in_regime=subject_ids_in_regime,
    )

    # Update states and regime membership for next period
    if not regime.terminal:
        next_states_key, next_regime_key, key = jax.random.split(key, 3)

        next_states = calculate_next_states(
            regime=regime,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=flat_params[regime_name],
            states_per_regime=states,
            state_action_space=state_action_space,
            key=next_states_key,
            subjects_in_regime=subject_ids_in_regime,
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
        )
        states = next_states
        new_subject_regime_ids = calculate_next_regime_membership(
            regime=regime,
            state_action_space=state_action_space,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=flat_params[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            new_subject_regime_ids=new_subject_regime_ids,
            active_regimes_next_period=active_regimes_next_period,
            key=next_regime_key,
            subjects_in_regime=subject_ids_in_regime,
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
        )

    return simulation_result, states, new_subject_regime_ids, key


def _lookup_values_from_indices(
    *,
    flat_indices: IntND,
    grids: MappingProxyType[StateOrActionName, FloatND | IntND],
) -> MappingProxyType[StateOrActionName, FloatND | IntND]:
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
    initial_ages: Float1D,
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
