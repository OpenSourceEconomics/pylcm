import itertools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap

from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import safeguarded_continuous_argmax
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import (
    EGMPolicyRead,
    PeriodRegimeSimulationData,
    Regime,
    StateActionSpace,
)
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG
from _lcm.simulation.additional_targets import _compute_targets
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    build_initial_states,
    trim_pad_from_raw_results,
)
from _lcm.simulation.random import draw_random_seed, generate_simulation_keys
from _lcm.simulation.transitions import (
    calculate_next_regime_membership,
    calculate_next_states,
    create_regime_state_action_space,
)
from _lcm.solution.validate_V import validate_V
from _lcm.typing import (
    ActionName,
    FlatParams,
    FlatRegimeParams,
    InitialConditions,
    PeriodToRegimeToSimulationPolicy,
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
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarFloat, ScalarInt


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
    period_to_regime_to_sim_policy: PeriodToRegimeToSimulationPolicy | None = None,
    seed: int | None = None,
    subject_batch_size: int = 0,
    original_n_subjects: int | None = None,
    period_to_regime_to_dissolution_flags: MappingProxyType[
        int, MappingProxyType[RegimeName, BoolND]
    ] = MappingProxyType({}),
    own_stakeholder: str | None = None,
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
        period_to_regime_to_sim_policy: Immutable mapping of periods to each
            EGM regime's published off-grid simulation policy, or `None`
            (user-supplied V arrays carry no policy). Sparse over regimes.
            Where a regime qualifies (`SimulationPhase.egm_policy_read`), the
            continuous action is interpolated from the policy at the
            subject's resources instead of argmaxed over the action grid.
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
        original_n_subjects: Subject count before any subject-axis padding applied
            by `pad_initial_conditions_to_multiple`. When set, RNG keys are sized to it
            and the trailing pad rows are trimmed from the results before they are
            returned. `None` means no padding was applied.
        period_to_regime_to_dissolution_flags: Immutable mapping of periods to each
            COLLECTIVE regime's dissolution flag `D` (E2/E4), as returned by
            `backward_induction.solve`'s third element. Empty (the default)
            for models without gated edges reading `D_target`, or when the
            caller does not have it at hand — not yet surfaced through the
            public `Model.solve`/`Model.simulate` API (mirrors solve's own
            internal-only status; a public accessor is a follow-up).
        own_stakeholder: ROW-SPLIT (synthetic mode). This simulate() call's
            fixed own-role for dissolution routing on a COLLECTIVE source's
            gated edge — e.g. "f" for an all-women population tracking
            synthetic male partners, "m" for an all-men population. A
            single value for the WHOLE call, not a per-subject array (see
            `_lcm.simulation.gated_routing._select_own_leg`). `None`
            (default) preserves the original "first declared leg" routing
            convention exactly — byte-identical for any caller that does
            not pass it (e.g. the public `Model.simulate()`, not yet
            surfaced here either).

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
        # `n_subjects` is padded up to a multiple of `batch_size` upstream (see
        # `pad_initial_conditions_to_multiple`), so every chunk — including the
        # last — is exactly `batch_size` rows; the trailing pad rows are dropped
        # once, after the loop, by `trim_pad_from_raw_results`.
        subject_slice = slice(chunk_start, chunk_start + batch_size)
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
            period_to_regime_to_dissolution_flags=period_to_regime_to_dissolution_flags,
            period_to_regime_to_sim_policy=period_to_regime_to_sim_policy,
            flat_params=flat_params,
            ages=ages,
            seed=seed,
            logger=logger,
            own_stakeholder=own_stakeholder,
        )
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

    # Drop any subject-axis alignment pad rows so `SimulationResult` exposes only
    # the user's real subjects (see `pad_initial_conditions_to_multiple`). No-op when
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
    period_to_regime_to_dissolution_flags: MappingProxyType[
        int, MappingProxyType[RegimeName, BoolND]
    ],
    flat_params: FlatParams,
    ages: AgeGrid,
    seed: int,
    logger: logging.Logger,
    own_stakeholder: str | None = None,
    period_to_regime_to_sim_policy: PeriodToRegimeToSimulationPolicy | None = None,
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Run the full period loop for one chunk of subjects.

    `initial_states`, `initial_regime_ids`, and `starting_periods` are already
    sliced to this chunk's subjects; `n_subjects` and `subject_slice` describe the
    chunk's position in the full population so RNG keys stay full-population and are
    sliced by global index. The key stream is re-derived from `seed` here so the
    per-period carry is identical across chunks (it is subject-count-independent).
    `own_stakeholder`: see `simulate()`'s docstring (ROW-SPLIT synthetic mode).

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

    # The params-completed base space is period-invariant within one simulate
    # call (params are fixed), so build it once per regime — runtime-grid
    # completion (e.g. process gridpoint computation) rides on it and would
    # otherwise rerun every period.
    base_state_action_spaces = {
        regime_name: regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
        )
        for regime_name, regime in regimes.items()
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
                    base_state_action_space=base_state_action_spaces[regime_name],
                    period=period,
                    age=age,
                    states=states,
                    subject_regime_ids=subject_regime_ids,
                    new_subject_regime_ids=new_subject_regime_ids,
                    period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                    period_to_regime_to_dissolution_flags=period_to_regime_to_dissolution_flags,
                    base_state_action_spaces=base_state_action_spaces,
                    sim_policy=(period_to_regime_to_sim_policy or {})
                    .get(period, {})
                    .get(regime_name),
                    flat_params=flat_params,
                    regime_names_to_ids=regime_names_to_ids,
                    active_regimes_next_period=active_regimes_next_period,
                    key=key,
                    logger=logger,
                    n_subjects=n_subjects,
                    subject_slice=subject_slice,
                    original_n_subjects=original_n_subjects,
                    own_stakeholder=own_stakeholder,
                )
            )
            states = new_states
            simulation_results[regime_name][period] = result

            # Out-of-regime subjects carry placeholder entries (possibly -inf,
            # when their state is infeasible under this regime's problem);
            # validate only the subjects simulated in this regime.
            #
            # COLLECTIVE-REGIMES (E4): a collective regime's `V_arr` carries a
            # trailing stakeholder axis (`(n_subjects, n_stakeholders)`), so
            # `in_regime` (always `(n_subjects,)`) needs trailing singleton
            # axes to broadcast against it; a singleton regime's `V_arr` is
            # already `(n_subjects,)` and this is a no-op reshape.
            in_regime_broadcast = result.in_regime.reshape(
                result.in_regime.shape
                + (1,) * (result.V_arr.ndim - result.in_regime.ndim)
            )
            log_nan_in_V(
                logger=logger,
                regime_name=regime_name,
                age=age,
                V_arr=jnp.where(in_regime_broadcast, result.V_arr, 0.0),
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
    base_state_action_space: StateActionSpace,
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace],
    period: int,
    age: ScalarInt | ScalarFloat,
    states: StatesPerRegime,
    subject_regime_ids: Int1D,
    new_subject_regime_ids: Int1D,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    period_to_regime_to_dissolution_flags: MappingProxyType[
        int, MappingProxyType[RegimeName, BoolND]
    ],
    flat_params: FlatParams,
    regime_names_to_ids: RegimeNamesToIds,
    active_regimes_next_period: tuple[RegimeName, ...],
    key: PRNGKeyND,
    logger: logging.Logger,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None = None,
    own_stakeholder: str | None = None,
    sim_policy: EGMSimPolicy | NestedEGMSimPolicy | None = None,
) -> tuple[PeriodRegimeSimulationData, StatesPerRegime, Int1D, PRNGKeyND]:
    """Simulate one regime for one period.

    This function processes all subjects in a given regime for a single period,
    computing optimal actions, updating states, and determining next regime membership.

    Args:
        regime_name: Name of the current regime.
        regime: Internal representation of the regime.
        base_state_action_space: The regime's params-completed state-action
            space, built once per simulate call.
        base_state_action_spaces: Every regime's params-completed state-action
            space (COLLECTIVE-REGIMES E4: a gated edge's value/gate fold
            needs its TARGET regime's own state grids, not just this
            regime's).
        period: Current period (0-indexed).
        age: Age corresponding to current period.
        states: Carrier of current-period state arrays for every regime and
            state.
        subject_regime_ids: Current regime membership for all subjects.
        new_subject_regime_ids: Array to populate with next period's regime memberships.
        period_to_regime_to_V_arr: Value function arrays for all periods and regimes.
        period_to_regime_to_dissolution_flags: Each COLLECTIVE regime's dissolution
            flag `D` per period (E2/E4); empty for models without one.
        flat_params: Model parameters for all regimes.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key for stochastic operations.
        n_subjects: Total number of subjects (the full population), used to keep RNG
            key generation independent of how subjects are chunked.
        subject_slice: Global-index slice of the subjects in this chunk.
        own_stakeholder: See `simulate()`'s docstring (ROW-SPLIT synthetic mode);
            threaded down to `route_gated_edges`.
        sim_policy: The regime's published off-grid simulation policy for this
            period, or `None`. Consumed only where the regime qualifies
            (`regime.simulation.egm_policy_read`).

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
        base=base_state_action_space,
    )
    # Compute optimal actions
    # We need to pass the value function array of the next period to the
    # argmax_and_max_Q_over_a function, as the current Q-function requires the
    # next period's value function. In the last period, we pass an empty dict.
    next_regime_to_V_arr = period_to_regime_to_V_arr.get(
        period + 1, MappingProxyType({})
    )

    # COLLECTIVE-REGIMES (E4): the simulate value router. A regime declaring
    # `gated_edges` must have its OWN action choice informed by the gated
    # continuation `Wbar`, not the target's raw (ungated) value — substitute
    # it into `next_regime_to_V_arr` exactly like the solve-side kernel does
    # (`_with_edge_substitution`), computed here from the already-solved
    # next-period arrays. `same_period_mappings` (the target V / `D` /
    # reference-V arrays each firing edge was folded on, per target) feeds
    # the REGIME-ROUTING step below, after the action and candidate
    # next-states are known — `route_gated_edges` RECOMPUTES the gate from
    # these (simulate F1 fix) rather than interpolating a baked boolean.
    # No-op (returns the inputs unchanged) for a regime without
    # `gated_edges`. See design doc §2 (E4) / §3.
    next_regime_to_V_arr, same_period_mappings = substitute_gated_edge_continuations(
        regime=regime,
        regime_name=regime_name,
        period=period,
        next_regime_to_V_arr=next_regime_to_V_arr,
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        period_to_regime_to_dissolution_flags=period_to_regime_to_dissolution_flags,
        flat_params=flat_params,
    )

    # The Q-function values contain the information of how much value each
    # action combination is worth. To find the optimal discrete action, we
    # therefore only need to maximize the Q-function values over all actions.
    argmax_and_max_Q_over_a = regime.simulation.argmax_and_max_Q_over_a[period]

    # COLLECTIVE-REGIMES (E2, simulate side): a regime declaring
    # `same_period_refs` reads other regimes' THIS-period V (not the
    # next-period continuation) inside its value-aware feasibility mask —
    # exactly like the solve kernel's `same_period_regime_to_V_arr`, sourced
    # here from the already-solved solution instead of the live backward-
    # induction loop. Empty for every regime without same-period references.
    #
    # Each reference regime's OWN flat params ride alongside under
    # `SAME_PERIOD_PARAMS_ARG` (F4): the reader interpolates the REFERENCE
    # regime's V over the REFERENCE regime's grid, whose runtime grid points are
    # that regime's parameters, not this one's — mirrors the identical pairing in
    # `solution.solvers._GridSearchPeriodKernel`.
    same_period_kwargs: dict[str, object] = {}
    if regime.same_period_ref_regimes:
        this_period_V = period_to_regime_to_V_arr.get(period, MappingProxyType({}))
        same_period_kwargs["same_period_regime_to_V_arr"] = MappingProxyType(
            {ref: this_period_V[ref] for ref in regime.same_period_ref_regimes}
        )
        same_period_kwargs[SAME_PERIOD_PARAMS_ARG] = MappingProxyType(
            {ref: flat_params[ref] for ref in regime.same_period_ref_regimes}
        )

    taste_shock_kwargs = {}
    if regime.has_taste_shocks:
        # Per-subject Gumbel keys are generated for the full population and
        # sliced by global subject index, so a subject's draw is invariant to
        # how the population is chunked.
        key, gumbel_keys = generate_simulation_keys(
            key=key,
            names=["taste_shock"],
            n_initial_states=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
        )
        taste_shock_kwargs = {"taste_shock_key": gumbel_keys["key_taste_shock"]}

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        **taste_shock_kwargs,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **same_period_kwargs,
        **flat_params[regime_name],
        period=jnp.int32(period),
        age=age,
    )
    if validation_enabled(logger):
        try:
            # Out-of-regime subjects carry placeholder entries (their state is
            # meaningless under this regime's problem); validate only the
            # subjects simulated in this regime.
            #
            # COLLECTIVE-REGIMES (E4): `V_arr` carries a trailing stakeholder
            # axis for a collective regime; broadcast the mask the same way
            # as the diagnostic logger below.
            in_regime_mask = subject_ids_in_regime.reshape(
                subject_ids_in_regime.shape
                + (1,) * (V_arr.ndim - subject_ids_in_regime.ndim)
            )
            validate_V(
                V_arr=jnp.where(in_regime_mask, V_arr, 0.0),
                age=age,
                regime_name=regime_name,
            )
        except InvalidValueFunctionError as error:
            raise_or_warn(logger=logger, error=error)

    # COLLECTIVE-REGIMES (E4, F7 guard). A STATELESS collective regime (no
    # declared states, e.g. a terminal `Regime(stakeholders=(...), ...)`
    # with only actions and/or constant utilities) has no per-subject state
    # array for the dispatcher to `vmap` the household argmax over, so
    # `argmax_and_max_Q_over_a` returns a single household-wide result: a
    # 0-d `indices_optimal_actions` and a `V_arr` of shape `(n_stakeholders,)`
    # — identical for every subject, but missing the leading subject axis
    # every OTHER path (stateless singleton, stateful collective) carries.
    # Left alone, the 0-d index reaches `_lookup_values_from_indices` ->
    # `vmapped_unravel_index`, which `vmap`s over axis 0 and requires
    # `flat_indices.ndim >= 1` — a `ValueError` for any subject count. And
    # even when there are no actions to look up (so that call is a no-op),
    # the un-broadcast `(n_stakeholders,)` V_arr would silently mismatch
    # every downstream `(n_chunk_subjects, ...)` shape (the `in_regime`
    # broadcast below, `to_dataframe`'s per-row extraction). Detected from
    # the regime's OWN declaration (`stakeholders` set, no state names) —
    # not from array shapes, which could coincide with `n_chunk_subjects`
    # by chance. A stateFUL collective regime (`state_action_space.states`
    # non-empty) already carries its own subject axis from the state
    # arrays and is untouched by this branch.
    n_chunk_subjects = subject_ids_in_regime.shape[0]
    if regime.stakeholders is not None and not state_action_space.states:
        indices_optimal_actions = jnp.broadcast_to(
            jnp.asarray(indices_optimal_actions), (n_chunk_subjects,)
        )
        V_arr = jnp.broadcast_to(V_arr[None, ...], (n_chunk_subjects, *V_arr.shape))

    optimal_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
    )
    optimal_actions = _replace_continuous_action_with_policy_read(
        optimal_actions=optimal_actions,
        regime=regime,
        sim_policy=sim_policy,
        states=states[regime_name],
        flat_params=flat_params[regime_name],
        period=period,
        age=age,
    )
    # Store results for this regime-period
    # For state-less regimes (e.g., terminal regimes with no states), V_arr may be a
    # scalar. We need to broadcast it to match this chunk's subject count.
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
        # The realized regime draw reads current-period carried values, so it
        # runs against the pre-advance carrier; only then do the next-period
        # states replace it.
        new_subject_regime_ids = calculate_next_regime_membership(
            regime=regime,
            state_action_space=state_action_space,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=flat_params[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            states_per_regime=states,
            new_subject_regime_ids=new_subject_regime_ids,
            active_regimes_next_period=active_regimes_next_period,
            key=next_regime_key,
            subjects_in_regime=subject_ids_in_regime,
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
        )
        # COLLECTIVE-REGIMES (E4): the value router's routing half. A gated
        # edge's target is always ALSO an ordinary declared transition
        # target, so `calculate_next_states` already computed candidate
        # target states above and `calculate_next_regime_membership` already
        # drew a (gate-blind) next regime id; `route_gated_edges` now
        # RECOMPUTES the gate at those candidate states (simulate F1 fix)
        # and OVERRIDES both — the target when open, a leg's fallback (with
        # its own projected states) when closed — for every subject in this
        # regime. No-op for a regime without `gated_edges`.
        next_states, new_subject_regime_ids = route_gated_edges(
            regime=regime,
            same_period_mappings=same_period_mappings,
            next_states=next_states,
            regime_names_to_ids=regime_names_to_ids,
            new_subject_regime_ids=new_subject_regime_ids,
            subjects_in_regime=subject_ids_in_regime,
            flat_params=flat_params,
            own_stakeholder=own_stakeholder,
        )
        states = next_states

    return simulation_result, states, new_subject_regime_ids, key


def _replace_continuous_action_with_policy_read(
    *,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: EGMSimPolicy | NestedEGMSimPolicy | None,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
) -> MappingProxyType[ActionName, FloatND | IntND]:
    """Interpolate the published EGM policy at each subject's resources.

    Replaces the grid-argmax value of the EGM continuous action with the
    off-grid solve-phase optimum. Discrete-state axes index the row at the
    subject's state (positions located on the variable's grid), and the row is
    interpolated at the subject's resources. For a regime with discrete
    actions the discrete branch is re-decided first: each branch's conditional
    value row is interpolated at that branch's own resources, discrete-only
    user constraints exclude infeasible branches, and the winner's policy row
    supplies the continuous action while the recorded discrete actions switch
    to the winning branch.
    Applies only where the regime qualifies (`regime.simulation.egm_policy_read`
    is set). Kept on the grid pair:
    - rows with a passive continuous-state axis (the build-time gate excludes
      passive regimes; the runtime guard is defensive) — each row is the
      envelope policy conditional on one passive node, so blending across a
      passive-dimension branch switch would read neither branch;
    - subjects with any feasible branch out of its live row support (an
      incomplete value comparison cannot pick the winner), or whose winning
      read is non-finite, non-positive, or outside the intrinsic budget
      (`action <= resources - savings_lower_bound`).
    """
    if sim_policy is None:
        return optimal_actions
    if isinstance(sim_policy, NestedEGMSimPolicy):
        # The nested (continuous-outer) payload is self-describing (it names
        # both actions, the liquid state, and the search settings), so it
        # needs no build-time `egm_policy_read` qualification of its own.
        return _read_nested_policy(
            payload=sim_policy,
            optimal_actions=optimal_actions,
            regime=regime,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
        )
    read = regime.simulation.egm_policy_read
    if read is None:
        return optimal_actions
    if sim_policy.row_passive_state_names:
        return optimal_actions

    n_subjects = next(iter(states.values())).shape[0]

    def grid_position(name: StateOrActionName, values: FloatND | IntND) -> IntND:
        grid_values = jnp.asarray(regime.simulation.grids[name].to_jax())
        return jnp.clip(
            jnp.searchsorted(grid_values, values), 0, grid_values.shape[0] - 1
        )

    state_positions = tuple(
        grid_position(name, jnp.asarray(states[name]))
        for name in sim_policy.row_discrete_state_names
    )

    if sim_policy.row_discrete_action_names:
        return _redecide_branch_and_read_policy(
            optimal_actions=optimal_actions,
            regime=regime,
            sim_policy=sim_policy,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
            read=read,
            n_subjects=n_subjects,
            state_positions=state_positions,
        )

    resources = _resources_at_subjects(
        read=read,
        regime=regime,
        sim_policy=sim_policy,
        states=states,
        optimal_actions=optimal_actions,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    off_grid_action, in_support = _interp_rows_with_support(
        sim_policy=sim_policy,
        field="policy",
        index=state_positions,
        resources=resources,
        n_subjects=n_subjects,
    )

    # Post-read acceptance: the replacement must be an in-support
    # interpolation (outside the live rows the read is an edge extension,
    # not the policy), finite, positive, and within the intrinsic budget;
    # any other read keeps that subject's grid-argmax value.
    grid_action = jnp.asarray(optimal_actions[read.action_name])
    accepted = (
        in_support
        & jnp.isfinite(off_grid_action)
        & (off_grid_action > 0.0)
        & (off_grid_action <= resources - read.savings_lower_bound)
    )
    off_grid_action = jnp.where(accepted, off_grid_action, grid_action)

    return MappingProxyType({**optimal_actions, read.action_name: off_grid_action})


def _redecide_branch_and_read_policy(
    *,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: EGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    read: EGMPolicyRead,
    n_subjects: int,
    state_positions: tuple[IntND, ...],
) -> MappingProxyType[ActionName, FloatND | IntND]:
    """Re-decide the discrete branch off-grid and read the winner's policy.

    Each discrete-action combo's published rows are the solve-phase optimum
    conditional on that branch, and the branch's resources follow from the
    regime DAG at the candidate action values. The comparison mirrors the
    decision problem the solve encodes:
    - a branch failing a discrete-only user constraint at the subject's state
      is excluded (its value reads as `-inf`), exactly as the constraint masks
      the grid argmax;
    - among feasible branches the interpolated conditional values — each at
      its own resources — pick the winner;
    - the winner's policy row is interpolated at the winner's resources, and
      the recorded discrete actions switch to the winning branch.
    A subject keeps the grid-argmax pair (discrete and continuous alike) when
    any feasible branch's resources fall outside that branch's live row
    support (the value comparison would be incomplete), no branch is feasible,
    or the winning read is non-finite, non-positive, or outside the intrinsic
    budget.
    """
    action_names = sim_policy.row_discrete_action_names
    action_grids = tuple(
        jnp.asarray(regime.simulation.grids[name].to_jax()) for name in action_names
    )
    combo_shape = tuple(int(grid.shape[0]) for grid in action_grids)
    # The synthesized intrinsic budget constraint reads the continuous action,
    # which is not decided yet at branch-comparison time; the post-read
    # acceptance enforces the budget instead. DC-EGM validation restricts all
    # user constraints to discrete variables, so they evaluate per branch.
    discrete_constraint_names = [
        name
        for name in regime.simulation.constraints
        if name != DCEGM_BUDGET_CONSTRAINT_NAME
    ]

    values_per_combo = []
    policies_per_combo = []
    resources_per_combo = []
    feasible_per_combo = []
    in_support_per_combo = []
    for combo in itertools.product(*(range(size) for size in combo_shape)):
        combo_actions = {
            name: jnp.full((n_subjects,), grid[position], dtype=grid.dtype)
            for name, grid, position in zip(
                action_names, action_grids, combo, strict=True
            )
        }
        data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = {
            **dict(states),
            **combo_actions,
            "period": jnp.full(n_subjects, period, dtype=jnp.int32),
            "age": jnp.full(n_subjects, age),
        }
        targets = _compute_targets(
            data=data,
            targets=[read.resources_target, *discrete_constraint_names],
            regime=regime,
            regime_params=flat_params,
        )
        resources = jnp.reshape(
            jnp.asarray(targets[read.resources_target]), (n_subjects,)
        )
        feasible = jnp.ones(n_subjects, dtype=bool)
        for constraint_name in discrete_constraint_names:
            feasible &= jnp.reshape(
                jnp.asarray(targets[constraint_name]), (n_subjects,)
            ).astype(bool)
        index = (*state_positions, *combo)
        value, in_support = _interp_rows_with_support(
            sim_policy=sim_policy,
            field="value",
            index=index,
            resources=resources,
            n_subjects=n_subjects,
        )
        policy, _ = _interp_rows_with_support(
            sim_policy=sim_policy,
            field="policy",
            index=index,
            resources=resources,
            n_subjects=n_subjects,
        )
        values_per_combo.append(jnp.where(feasible, value, -jnp.inf))
        policies_per_combo.append(policy)
        resources_per_combo.append(resources)
        feasible_per_combo.append(feasible)
        in_support_per_combo.append(in_support)

    values = jnp.stack(values_per_combo)
    policies = jnp.stack(policies_per_combo)
    resources = jnp.stack(resources_per_combo)
    feasible = jnp.stack(feasible_per_combo)
    in_support = jnp.stack(in_support_per_combo)

    winner = jnp.argmax(values, axis=0)

    def at_winner(stacked: FloatND) -> FloatND:
        return jnp.take_along_axis(stacked, winner[None, :], axis=0)[0]

    off_grid_action = at_winner(policies)
    winner_resources = at_winner(resources)
    winner_value = at_winner(values)
    accepted = (
        jnp.any(feasible, axis=0)
        & jnp.all(~feasible | in_support, axis=0)
        & jnp.isfinite(winner_value)
        & jnp.isfinite(off_grid_action)
        & (off_grid_action > 0.0)
        & (off_grid_action <= winner_resources - read.savings_lower_bound)
    )

    new_actions = dict(optimal_actions)
    grid_action = jnp.asarray(optimal_actions[read.action_name])
    new_actions[read.action_name] = jnp.where(accepted, off_grid_action, grid_action)
    winner_positions = jnp.unravel_index(winner, combo_shape)
    for name, grid, positions in zip(
        action_names, action_grids, winner_positions, strict=True
    ):
        current = jnp.asarray(optimal_actions[name])
        new_actions[name] = jnp.where(
            accepted, grid[positions].astype(current.dtype), current
        )
    return MappingProxyType(new_actions)


# Numerical tolerance certifying the outer transition's unit action slope
# (the affine inversion contract of the nested policy read).
_UNIT_SLOPE_ATOL = 1e-8


def _interp_across_outer_axis(
    *, nodes: Float1D, values: FloatND, query: FloatND
) -> FloatND:
    """Linear read of per-candidate subject values `(C, n)` at `query` `(n,)`."""
    hi = jnp.clip(jnp.searchsorted(nodes, query), 1, nodes.shape[0] - 1)
    lo = hi - 1
    weight = jnp.clip((query - nodes[lo]) / (nodes[hi] - nodes[lo]), 0.0, 1.0)
    subject_axis = jnp.arange(query.shape[0])
    return values[lo, subject_axis] * (1.0 - weight) + values[hi, subject_axis] * weight


def _read_nested_policy(
    *,
    payload: NestedEGMSimPolicy,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
) -> MappingProxyType[ActionName, FloatND | IntND]:
    """Replay the continuous-outer keeper/adjuster decision off-grid.

    Reconstructs, per subject, exactly the solve's decision problem in the
    solve's own policy class: read every branch's conditional value row at
    the subject's liquid state (cubic Hermite with the published marginal as
    slope — the solve's read convention), rebuild the outer-value surrogate
    across the shared mesh (`LocalCubicOuterInterpolant`), refine it with the
    same globally safeguarded search the solve used (exact nodes always
    compete; golden section only inside brackets around node-local maxima),
    and let the exact keeper win ties. The winning candidate's inner
    consumption is the per-candidate policy read interpolated across the
    outer axis at the refined optimum; the outer *action* is recovered from
    the winning post-decision through the transition's affine-unit-slope
    inversion (verified numerically, refused otherwise).

    Subjects off the passive (durable) grid — the normal case after a
    continuous adjustment — blend the two bracketing rows linearly, on both
    sides of the comparison.

    Falls back to the grid-argmax pair (both actions) for the whole regime
    when the payload shape is outside the v1 scope (discrete-action rows,
    more than one passive axis, a non-affine or unresolvable outer
    transition), and per subject when any branch read leaves its live row
    support, or the winning consumption is non-finite, non-positive, or
    outside the intrinsic budget (`c <= resources - savings_lower_bound`).
    """
    keeper_pol = payload.keeper
    bank = payload.adjuster
    if (
        keeper_pol.row_discrete_action_names
        or bank.policies.row_discrete_action_names
        or len(keeper_pol.row_passive_state_names) > 1
    ):
        return optimal_actions
    n_subjects = next(iter(states.values())).shape[0]
    liquid = jnp.asarray(states[payload.liquid_state_name])

    def grid_position(name: StateOrActionName) -> IntND:
        grid_values = jnp.asarray(regime.simulation.grids[name].to_jax())
        return jnp.clip(
            jnp.searchsorted(grid_values, jnp.asarray(states[name])),
            0,
            grid_values.shape[0] - 1,
        )

    discrete_idx = tuple(
        grid_position(name) for name in keeper_pol.row_discrete_state_names
    )

    passive_bracket: tuple[IntND, IntND, FloatND] | None = None
    if keeper_pol.row_passive_state_names:
        name = keeper_pol.row_passive_state_names[0]
        grid_values = jnp.asarray(regime.simulation.grids[name].to_jax())
        x = jnp.asarray(states[name])
        hi = jnp.clip(jnp.searchsorted(grid_values, x), 1, grid_values.shape[0] - 1)
        lo = hi - 1
        weight = jnp.clip(
            (x - grid_values[lo]) / (grid_values[hi] - grid_values[lo]), 0.0, 1.0
        )
        passive_bracket = (lo, hi, weight)

    def blended_read(
        pol: EGMSimPolicy, field: Literal["policy", "value"]
    ) -> tuple[FloatND, BoolND]:
        if passive_bracket is None:
            return _interp_rows_with_support(
                sim_policy=pol,
                field=field,
                index=discrete_idx,
                resources=liquid,
                n_subjects=n_subjects,
            )
        lo, hi, weight = passive_bracket
        value_lo, support_lo = _interp_rows_with_support(
            sim_policy=pol,
            field=field,
            index=(*discrete_idx, lo),
            resources=liquid,
            n_subjects=n_subjects,
        )
        value_hi, support_hi = _interp_rows_with_support(
            sim_policy=pol,
            field=field,
            index=(*discrete_idx, hi),
            resources=liquid,
            n_subjects=n_subjects,
        )
        return (
            value_lo * (1.0 - weight) + value_hi * weight,
            support_lo & support_hi,
        )

    keeper_value, keeper_support = blended_read(keeper_pol, "value")
    keeper_action, _ = blended_read(keeper_pol, "policy")

    def candidate_read(pol: EGMSimPolicy) -> tuple[FloatND, BoolND, FloatND]:
        value, support = blended_read(pol, "value")
        action, _ = blended_read(pol, "policy")
        return value, support, action

    candidate_values, candidate_support, candidate_actions = vmap(candidate_read)(
        bank.policies
    )

    # The same surrogate class and safeguarded search as the solve's collapse:
    # exact node values compete directly, golden section refines only inside
    # brackets around node-local maxima of the interpolated profile.
    outer_nodes = bank.outer_nodes
    profile = jnp.where(candidate_support, candidate_values, -jnp.inf)
    interpolant = LocalCubicOuterInterpolant()
    search = safeguarded_continuous_argmax(
        lambda query: interpolant.evaluate(
            nodes=outer_nodes, values=profile, query=query
        ),
        nodes=outer_nodes,
        node_values=profile,
        golden_iterations=payload.golden_iterations,
    )
    adjust = search.value > keeper_value  # exact keeper wins ties

    # Inner action of the adjusting branch: the per-candidate policy reads,
    # interpolated linearly across the outer axis at the refined optimum
    # (policy rows carry no slope data — matching the flat read's convention).
    adjuster_action = _interp_across_outer_axis(
        nodes=outer_nodes, values=candidate_actions, query=search.x
    )

    outer_offset_slope = _outer_transition_offset_and_slope(
        payload=payload,
        regime=regime,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    if outer_offset_slope is None:
        return optimal_actions
    offset, slope_is_unit = outer_offset_slope
    keep_value = _keeper_post_decision(
        payload=payload,
        regime=regime,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    if keep_value is None:
        return optimal_actions

    chosen_post_decision = jnp.where(adjust, search.x, keep_value)
    outer_action = chosen_post_decision - offset
    inner_action = jnp.where(adjust, adjuster_action, keeper_action)
    winner_value = jnp.where(adjust, search.value, keeper_value)

    resources = _nested_resources(
        payload=payload,
        regime=regime,
        states=states,
        outer_action=outer_action,
        outer_post_decision=chosen_post_decision,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )

    accepted = (
        slope_is_unit
        & keeper_support
        & jnp.all(candidate_support, axis=0)
        & jnp.isfinite(winner_value)
        & jnp.isfinite(outer_action)
        & jnp.isfinite(inner_action)
        & (inner_action > 0.0)
        & (inner_action <= resources - payload.savings_lower_bound)
    )

    new_actions = dict(optimal_actions)
    new_actions[payload.inner_action_name] = jnp.where(
        accepted, inner_action, jnp.asarray(optimal_actions[payload.inner_action_name])
    )
    new_actions[payload.outer_action_name] = jnp.where(
        accepted, outer_action, jnp.asarray(optimal_actions[payload.outer_action_name])
    )
    return MappingProxyType(new_actions)


def _nested_resources(
    *,
    payload: NestedEGMSimPolicy,
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    outer_action: FloatND,
    outer_post_decision: FloatND,
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> FloatND:
    """The chosen branch's resources through the simulate DAG (subject axis kept).

    The outer post-decision is bound as data: it is a *transition* in the
    simulate DAG, so the target computation cannot derive it from the action.
    """
    data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = {
        **dict(states),
        payload.outer_action_name: outer_action,
        payload.outer_post_decision_name: outer_post_decision,
        "period": jnp.full(n_subjects, period, dtype=jnp.int32),
        "age": jnp.full(n_subjects, age),
    }
    return jnp.reshape(
        jnp.asarray(
            _compute_targets(
                data=data,
                targets=[payload.resources_target_name],
                regime=regime,
                regime_params=flat_params,
            )[payload.resources_target_name]
        ),
        (n_subjects,),
    )


def _resolve_function_kwargs(
    func: Callable[..., FloatND],
    *,
    states: Mapping[StateOrActionName, FloatND | IntND],
    bindings: Mapping[str, FloatND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> dict[str, FloatND | IntND] | None:
    """Bind a simulate-phase function's parameters, or `None` if any is unmet."""
    import inspect  # noqa: PLC0415

    kwargs: dict[str, FloatND | IntND] = {}
    for name in inspect.signature(func).parameters:
        if name in bindings:
            kwargs[name] = jnp.asarray(bindings[name])
        elif name in states:
            kwargs[name] = jnp.asarray(states[name])
        elif name == "period":
            kwargs[name] = jnp.full(n_subjects, period, dtype=jnp.int32)
        elif name == "age":
            kwargs[name] = jnp.full(n_subjects, age)
        elif name in flat_params:
            kwargs[name] = jnp.asarray(flat_params[name])
        else:
            return None
    return kwargs


def _outer_transition_offset_and_slope(
    *,
    payload: NestedEGMSimPolicy,
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> tuple[FloatND, BoolND] | None:
    """Per-subject offset of the outer transition, with a unit-slope check.

    The winning outer post-decision `s'` is a *transition value*; the
    recorded action must invert `s' = T(states, a)`. The nested v1 scope
    supports the affine-unit-slope contract `T(states, a) = offset(states)
    + a`, verified numerically per subject by probing `T` at `a = 0` and
    `a = 1`. Returns `None` (whole-regime fallback) when the transition is
    absent or its arguments cannot be resolved.
    """
    # Transitions are keyed by target regime; the outer post-decision must
    # resolve to one shared callable across every target that declares it.
    found = [
        per_target[payload.outer_post_decision_name]
        for per_target in regime.simulation.transitions.values()
        if payload.outer_post_decision_name in per_target
    ]
    if not found or any(func is not found[0] for func in found[1:]):
        return None
    transition = found[0]
    zeros = jnp.zeros(n_subjects)
    probes = []
    for action_value in (zeros, zeros + 1.0):
        kwargs = _resolve_function_kwargs(
            transition,
            states=states,
            bindings={payload.outer_action_name: action_value},
            flat_params=flat_params,
            period=period,
            age=age,
            n_subjects=n_subjects,
        )
        if kwargs is None:
            return None
        probes.append(jnp.reshape(jnp.asarray(transition(**kwargs)), (n_subjects,)))
    offset, at_one = probes
    slope_is_unit = jnp.abs((at_one - offset) - 1.0) <= _UNIT_SLOPE_ATOL
    return offset, slope_is_unit


def _keeper_post_decision(
    *,
    payload: NestedEGMSimPolicy,
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> FloatND | None:
    """The keeper branch's outer post-decision `s' = keep(Z)` per subject.

    Uses the solver's declared no-adjustment candidate; without one, keeping
    means holding the current durable (the state the outer post-decision is
    the next-period value of) unchanged. Returns `None` (whole-regime
    fallback) when the candidate's arguments cannot be resolved.
    """
    if payload.outer_no_adjustment_name is None:
        durable_name = payload.outer_post_decision_name.removeprefix("next_")
        durable = states.get(durable_name)
        if durable is None:
            return None
        return jnp.asarray(durable)
    keep_func = regime.simulation.functions.get(payload.outer_no_adjustment_name)
    if keep_func is None:
        return None
    kwargs = _resolve_function_kwargs(
        keep_func,
        states=states,
        bindings={},
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    if kwargs is None:
        return None
    return jnp.reshape(jnp.asarray(keep_func(**kwargs)), (n_subjects,))


def _interp_rows_with_support(
    *,
    sim_policy: EGMSimPolicy,
    field: Literal["policy", "value"],
    index: tuple[IntND | int, ...],
    resources: FloatND,
    n_subjects: int,
) -> tuple[FloatND, BoolND]:
    """Interpolate one published row field per subject with its support flag.

    The live support runs from the row's first abscissa to its last finite one
    (rows are NaN-padded in the tail). Outside it the interpolant extends an
    edge secant (below) or clamps (above) — feasible values that need not
    approximate the published function. Reads by field:
    - `"value"` ⇒ cubic Hermite with the marginal-utility row as the node-slope
      input (Fritsch-Carlson-limited inside the interpolant) — the convention
      the solve publishes values under, so the branch ranking the re-decision
      sees is the solve convention's ranking;
    - `"policy"` ⇒ piecewise linear (the policy row carries no slope data).
    """
    rows_f = getattr(sim_policy, field)
    rows_x = sim_policy.endog_grid
    rows_slope = sim_policy.marginal_utility if field == "value" else None
    if index:
        rows_x = rows_x[index]
        rows_f = rows_f[index]
        rows_slope = rows_slope[index] if rows_slope is not None else None
    if rows_x.ndim == 1:
        rows_x = jnp.broadcast_to(rows_x, (n_subjects, *rows_x.shape))
        rows_f = jnp.broadcast_to(rows_f, (n_subjects, *rows_f.shape))
        if rows_slope is not None:
            rows_slope = jnp.broadcast_to(rows_slope, (n_subjects, *rows_slope.shape))
    if rows_slope is None:
        values = vmap(
            lambda x_query, xp, fp: interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)
        )(resources, rows_x, rows_f)
    else:
        values = vmap(
            lambda x_query, xp, fp, fp_slopes: interp_on_padded_grid(
                x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
            )
        )(resources, rows_x, rows_f, rows_slope)
    valid_length = jnp.sum(jnp.isfinite(rows_x), axis=-1)
    last_live = jnp.take_along_axis(rows_x, (valid_length - 1)[:, None], axis=-1)[:, 0]
    in_support = (resources >= rows_x[:, 0]) & (resources <= last_live)
    return values, in_support


def _resources_at_subjects(
    *,
    read: EGMPolicyRead,
    regime: Regime,
    sim_policy: EGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> FloatND:
    """Compute each subject's endogenous resources through the simulate DAG.

    The chosen discrete actions enter the data: a budget whose income depends
    on a discrete action (work choice) reads the simulated choice. The result
    keeps the subject axis (`_compute_targets` squeezes its outputs, which
    would collapse a single-subject vector to 0-d before the next-state vmap).
    """
    data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = {
        **dict(states),
        **{
            name: jnp.asarray(optimal_actions[name])
            for name in sim_policy.row_discrete_action_names
        },
        "period": jnp.full(n_subjects, period, dtype=jnp.int32),
        "age": jnp.full(n_subjects, age),
    }
    return jnp.reshape(
        jnp.asarray(
            _compute_targets(
                data=data,
                targets=[read.resources_target],
                regime=regime,
                regime_params=flat_params,
            )[read.resources_target]
        ),
        (n_subjects,),
    )


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
# the `shape` argument constant (in_axes = (0, None)). Jitted with the shape
# static so the vmap is traced once per (subject-count, grid-shape) pair
# instead of on every period-regime call in the simulation's inner loop.
vmapped_unravel_index = jax.jit(
    vmap(jnp.unravel_index, in_axes=(0, None)), static_argnums=1
)


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
