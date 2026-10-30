"""The forward-simulation loop over periods, regimes and subjects.

`simulate` seeds the population from the validated initial conditions and then,
for each period, decides every subject's actions — replaying a solver's
published policy where one exists, recomputing on the grid otherwise — applies
the state and regime transitions, and collects one
`PeriodRegimeSimulationData` per (regime, period) pair.
"""

import functools
import inspect
import itertools
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_inversion import (
    invert_declared_outer_target,
    outer_candidate_is_admissible,
)
from _lcm.egm.outer_refinement import safeguarded_continuous_argmax
from _lcm.egm.published_policy import (
    EGMSimPolicy,
    NBEGMGridPolicy,
    NNBEGMSimPolicy,
)
from _lcm.engine import (
    EGMPolicyRead,
    NBEGMPolicyRead,
    NNBEGMPolicyRead,
    PeriodRegimeSimulationData,
    Regime,
    StateActionSpace,
    placed_devices_for_ids,
)
from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.off_grid import count_off_grid
from _lcm.params.edges import regime_kernel_params
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.Q_and_F import (
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
    _get_feasibility,
)
from _lcm.simulation import chunk_operations, population_operations
from _lcm.simulation.additional_targets import _compute_targets
from _lcm.simulation.assembly import concatenate_arrays, take_rows
from _lcm.simulation.chunk_admission import PreparedSimulationChunks
from _lcm.simulation.chunk_inputs import (
    SimulationCallInputs,
    prepare_simulation_call_inputs,
    prepare_simulation_chunk_inputs,
)
from _lcm.simulation.chunk_offload import chunk_host_device, offload_chunk
from _lcm.simulation.current_iid import prepare_current_iid_keys
from _lcm.simulation.current_iid_overrides import (
    CurrentIIDOverrides,
    gather_current_iid_overrides,
)
from _lcm.simulation.diagnostic_operations import (
    owned_value_nan_count,
    period_value_flags,
    profiled_transition_counts,
)
from _lcm.simulation.dissolution_route import (
    DissolutionFallback,
    RerouteOutcome,
    empty_gated_entries,
    log_dissolution_fallbacks,
    record_open_entries,
    reroute_rows_without_feasible_action,
    rows_of,
)
from _lcm.simulation.exit_replay import prepare_exit_period
from _lcm.simulation.exits import PreparedExitDecision
from _lcm.simulation.gated_routing import (
    GatedRouteDelta,
    commit_gated_route_delta,
    gated_route_candidates,
)
from _lcm.simulation.host_operations import _HeldObject
from _lcm.simulation.initial_conditions import (
    _initial_own_stakeholder,
    build_initial_states,
    physical_state_at_row,
    trim_pad_from_raw_results,
)
from _lcm.simulation.membership import (
    activate_subject_membership,
    initialize_subject_membership,
)
from _lcm.simulation.memory import (
    SimulationMemory,
    run_simulation_operation,
    run_simulation_producer,
)
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.participation_routing import record_participation_entries
from _lcm.simulation.period_inputs import (
    GATE_FOLD,
    GATE_ROUTE,
    POLICY_SCORE,
    acquire_decision_inputs,
    acquire_gate_inputs,
    decision_reads,
    gate_reads,
    unit_value_reads,
)
from _lcm.simulation.plan_summary import (
    SimulationPlanSummary,
    build_simulation_plan_summary,
)
from _lcm.simulation.policy_programs import ReplayPayload
from _lcm.simulation.program_arguments import (
    ACTIVE_SUBJECTS_ARGUMENT,
    current_iid_arguments,
    decision_arguments,
    declares_active_subjects,
    gate_fold_arguments,
    gate_route_arguments,
    policy_prepare_arguments,
    policy_rank_arguments,
    process_node_arguments,
)
from _lcm.simulation.programs import (
    budgeted_simulation_programs_ready,
    forward_regimes_by_period,
)
from _lcm.simulation.random import (
    create_simulation_key,
    draw_random_seed,
)
from _lcm.simulation.random_address import RANDOM_ADDRESS_VERSION
from _lcm.simulation.replay_inputs import PreparedReplayReader, place_replay_payload
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resolve_budget_devices,
    union_buffer_footprints,
)
from _lcm.simulation.runtime import SimulationRuntime, execute_simulation_program
from _lcm.simulation.shock_paths import (
    fail_if_a_held_process_code_is_off_node,
    shock_path_values,
    supplied_shock_argument_names,
)
from _lcm.simulation.subject_groups import (
    ComponentValueSource,
    SubjectGroupPlan,
    SubjectRows,
    plan_subject_groups,
)
from _lcm.simulation.taste_stream import (
    build_taste_stream_addresses,
    create_taste_shock_key,
    prepare_decision_taste_keys,
)
from _lcm.simulation.transitions import (
    calculate_next_regime_membership,
    calculate_next_states,
    clip_next_states,
    create_regime_state_action_space,
    fail_if_drawn_codes_leave_their_grids,
)
from _lcm.simulation.unit_executor import SimulationUnitExecutor
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.backward_induction import _states_for_period
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.solution.feasibility import (
    BoundedResult,
    consume_runtime_bound_outputs,
    validate_runtime_bound_status,
)
from _lcm.solution.solve_phase_records import CallId, solve_phase
from _lcm.solution.validate_V import validate_V, value_function_nan_error
from _lcm.time import TimeAxis, coordinate_at, coordinate_kind
from _lcm.typing import (
    ActionName,
    ArrayTree,
    EconFunctionArg,
    FlatParams,
    FlatRegimeParams,
    InitialConditions,
    PeriodToRegimeToParticipationSurfaces,
    PeriodToRegimeToSimulationPolicy,
    PRNGKeyND,
    QAndFArg,
    QAndFFunction,
    QAndFKwargs,
    RegimeIdsToNames,
    RegimeName,
    RegimeNamesToIds,
    StateName,
    StateOrActionName,
    StatesPerRegime,
)
from _lcm.utils.containers import invert_regime_ids
from _lcm.utils.logging import (
    format_duration,
    log_non_finite_values,
    log_period_header,
    log_period_timing,
    log_regime_transition_counts,
    non_finite_by_regime,
    raise_or_warn,
    validation_enabled,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidSimulationInputError,
    InvalidValueFunctionError,
    NoFeasibleActionError,
    OffGridStateError,
    ShockDeclarationError,
)
from lcm.result import SimulationResult
from lcm.solver_api import SIMULATION_POLICY, ActionOutput, ReplayReader
from lcm.typing import (
    Bool1D,
    BoolND,
    Float1D,
    FloatND,
    FunctionName,
    Int1D,
    IntND,
    PRNGKey,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
)

type _PeriodToRegimeToReplayReader = Mapping[
    int, Mapping[RegimeName, PreparedReplayReader | ReplayReader]
]


def simulate(  # noqa: C901, PLR0912, PLR0915
    *,
    flat_params: FlatParams,
    initial_conditions: InitialConditions,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    logger: logging.Logger,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    ages: TimeAxis,
    simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    period_to_regime_to_sim_policy: PeriodToRegimeToSimulationPolicy | None = None,
    participation_surfaces: PeriodToRegimeToParticipationSurfaces = MappingProxyType(
        {}
    ),
    period_to_regime_to_replay_reader: _PeriodToRegimeToReplayReader = (
        MappingProxyType({})
    ),
    seed: int | PRNGKey | None = None,
    taste_shock_seed: int | PRNGKey | None = None,
    subject_batch_size: int = 0,
    original_n_subjects: int | None = None,
    device_ids: tuple[int, ...] = (),
    retained_footprint: DeviceBufferFootprint | None = None,
    prepared_chunks: PreparedSimulationChunks | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
    shock_paths: Mapping[StateName, np.ndarray] = MappingProxyType({}),
    current_iid_overrides: CurrentIIDOverrides | None = None,
    process_code_reads: Mapping[
        RegimeName, Mapping[StateName, _ContinuousStochasticProcess]
    ] = MappingProxyType({}),
    call_id: CallId | None = None,
    component_values: ComponentValueSource | None = None,
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
        ages: TimeAxis for the model, used to convert periods to ages.
        period_to_regime_to_sim_policy: Immutable mapping of periods to each
            EGM regime's published off-grid simulation policy, or `None` when
            the caller supplies no compatible policy mapping. Sparse over
            regimes.
            Where a regime qualifies (`SimulationPhase.egm_policy_read`), the
            continuous action is interpolated from the policy at the
            subject's resources instead of argmaxed over the action grid.
        period_to_regime_to_replay_reader: Model-preflighted external replay
            readers, independently addressed by period and regime. Empty for
            built-in solver routes.
        simulation_output_dtypes: Mapping of variable name to `pd.CategoricalDtype`,
            used for building simulation metadata.
        seed: Integer seed or scalar Threefry key of the ordinary stream. An integer
            becomes a Threefry key. If not provided, a random seed is generated.
        taste_shock_seed: Independent taste seed, or None for the ordinary stream.
        subject_batch_size: Concrete outer subject chunk extent, resolved by
            `Model.simulate` from ExecutionConfig and whole-chunk admission.
            `0` or a value `>= n_subjects` simulates the whole
            population in a single pass; a smaller value chunks the subjects,
            bounding the per-period device workspace at the cost of re-running the
            period loop per chunk. Results are invariant to this knob: per-subject
            RNG keys are generated for the full population and sliced by global
            index.
        original_n_subjects: Subject count before any subject-axis padding applied
            by `pad_initial_conditions_to_multiple`. When set, RNG keys are sized to it
            and the trailing pad rows are trimmed from the results before they are
            returned. `None` means no padding was applied.
        device_ids: The model's device ids, ascending. Empty names every
            device JAX reports.
        process_code_reads: Per regime, the process states whose node position
            its readers take. A household holding one off the regime's nodes is
            refused before the regime decides or a gate reads it.
        call_id: Identifier of the public call whose phase records the setup,
            each chunk, the completion wait and the result assembly are
            written under, or `None` to write none.
        component_values: Where a block-major solution's values are read from,
            one code at a time, or `None` to read `period_to_regime_to_V_arr`.
            Requires the grouped route: every code is acquired in grid order,
            its subjects are simulated from its values alone, and it is
            released before the next code is acquired. A code no subject holds
            is acquired and released too. The result then holds the source's
            complete value store. A source whose `subject_codes` selects codes
            simulates those codes' subjects alone, on the chunks the whole
            population's plan cuts for them: the result holds their rows in
            original order, names them in `_subject_rows`, and holds no values.

    Returns:
        SimulationResult object. Call .to_dataframe() to get a pandas DataFrame.

    Raises:
        ExecutionPlanningError: `component_values` is given without a grouping
            route or with a budgeted simulation.

    """
    with solve_phase(name="simulation_setup", logger=logger, call_id=call_id):
        if seed is None:
            seed = draw_random_seed()

        taste_addresses = build_taste_stream_addresses(
            ages=ages,
            discrete_actions_by_regime={
                name: {
                    action: grid
                    for action in regime.solution.action_names
                    if isinstance(grid := regime.solution.grids[action], DiscreteGrid)
                }
                for name, regime in regimes.items()
                if regime.has_taste_shocks
            },
        )

        logger.info("Starting simulation")
        total_start = time.monotonic()

        # Extract state arrays from initial conditions, which include the regime
        # and each subject's role on top. Neither is a state of any regime.
        initial_states = {
            k: v
            for k, v in initial_conditions.items()
            if k not in {"regime_id", "own_stakeholder"}
        }

        # Forward-simulate one subject chunk at a time. Subjects are independent across
        # the forward path (no cross-subject reduction), so each chunk runs the full
        # period loop on its own slice and the per-chunk results are concatenated on the
        # subject axis. Chunking bounds the per-period device workspace; the chunk size
        # is `subject_batch_size` (the whole population in one pass when zero).
        n_subjects = int(initial_conditions["regime_id"].shape[0])
        batch_size = (
            n_subjects
            if subject_batch_size == 0
            else min(subject_batch_size, n_subjects)
        )

        grouping = next(iter(regimes.values())).simulation.programs.grouping
        subject_codes = (
            None if component_values is None else component_values.subject_codes
        )
        group_plan = (
            None
            if grouping is None
            else plan_subject_groups(
                route=grouping,
                codes=(
                    np.asarray(jax.device_get(initial_conditions[grouping.state_name]))
                    if grouping.state_name in initial_conditions
                    else None
                ),
                n_real=(
                    n_subjects if original_n_subjects is None else original_n_subjects
                ),
                width=batch_size,
                selected=subject_codes,
            )
        )

        if component_values is not None and group_plan is None:
            raise ExecutionPlanningError(
                "Values retained one code at a time can only be simulated with "
                "subjects grouped by that code."
            )

        runtime = next(iter(regimes.values())).simulation.programs.executor
        memory = None
        if (
            isinstance(runtime, SimulationRuntime)
            and runtime.execution.device_memory_bytes is not None
        ):
            if retained_footprint is None:
                raise ExecutionPlanningError(
                    "Budgeted simulation requires retained solution residency."
                )
            if not budgeted_simulation_programs_ready(
                regimes=regimes,
                n_periods=ages.n_periods,
                enable_jit=runtime.enable_jit,
            ):
                raise ExecutionPlanningError(
                    "Budgeted simulation currently requires compiled decision "
                    "programs; host gated/replay adapters require their own "
                    "workspace accounting."
                    f"{runtime.execution.device_memory_cap_note()}"
                )
            inputs = union_buffer_footprints(
                footprints=(
                    retained_footprint,
                    measure_buffer_footprint(
                        tree=(
                            initial_conditions,
                            flat_params,
                            ages.values,  # noqa: PD011
                            period_to_regime_to_V_arr,
                            period_to_regime_to_sim_policy,
                        )
                    ),
                )
            )
            memory = SimulationMemory(
                budget_bytes=runtime.execution.device_memory_bytes,
                budget_note=runtime.execution.device_memory_cap_note(),
                axis_widths=MappingProxyType({})
                if prepared_chunks is None
                else prepared_chunks.plan.profile.axis_widths,
                subject_devices=runtime.subject_devices,
                operations=runtime.operations,
                devices=resolve_budget_devices(
                    execution_devices=placed_devices_for_ids(
                        submesh_device_ids=(), visible_device_ids=device_ids
                    ),
                    live=inputs,
                ),
                inputs=inputs,
            )
            memory.check_resident()
            if component_values is not None:
                raise ExecutionPlanningError(
                    "Values retained one code at a time are simulated without a "
                    "device budget; set ExecutionConfig.device_memory_bytes=None."
                )

        call_inputs = (
            prepare_simulation_call_inputs(
                flat_params=flat_params,
                regimes=regimes,
                device_ids=device_ids,
                memory=memory,
                process_grid_resolver=process_grid_resolver,
            )
            if prepared_chunks is None
            else prepared_chunks.call_inputs
        )
        if memory is not None:
            memory.inputs = union_buffer_footprints(
                footprints=(
                    memory.inputs,
                    measure_buffer_footprint(tree=call_inputs.array_roots),
                )
            )
            memory.close_unit()

        initial_own_stakeholder = _initial_own_stakeholder(
            initial_conditions=initial_conditions,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            memory=memory,
        )
        starting_periods = _compute_starting_periods(
            initial_ages=initial_states["age"], ages=ages, memory=memory
        )
        if memory is not None:
            # These arrays survive every chunk. Transfer their metadata to the
            # permanent inventory before releasing setup's temporary summaries.
            memory.inputs = union_buffer_footprints(
                footprints=(
                    memory.inputs,
                    measure_buffer_footprint(
                        tree=(initial_own_stakeholder, starting_periods)
                    ),
                )
            )
            memory.close_unit()
        # Build reverse lookup for regime transition logging. `regime_names_to_ids`
        # values are `ScalarInt` (jax 0-d arrays), which can't serve as dict keys
        # directly; `invert_regime_ids` coerces them to Python `int`.
        regime_ids_to_names = invert_regime_ids(regime_names_to_ids)

        # Diagnostic only: report the resolved execution plan once per call, so a
        # clean run carries evidence of which route (legacy/subjects), devices,
        # widths and chunking actually engaged. Never consulted for dispatch — it
        # is built from what the runtime already resolved above.
        plan_summary = build_simulation_plan_summary(
            regimes=regimes,
            subject_devices=call_inputs.devices,
            n_subjects=n_subjects,
            batch_size=batch_size,
            prepared_chunks=prepared_chunks,
            group_plan=group_plan,
        )
        _log_simulation_plan(logger=logger, plan_summary=plan_summary)

        # When chunking, offload each chunk's results to host as it finishes so the
        # device frees them before the next chunk's period loop allocates — bounding
        # device residency to a single chunk. A single pass (batch_size == n_subjects)
        # keeps results on the compute device (no memory pressure, and no host
        # round-trip for downstream targets).
        windows: tuple[tuple[slice | SubjectRows, int | None], ...] = (
            tuple(
                (slice(start, start + batch_size), None)
                for start in range(0, n_subjects, batch_size)
            )
            if group_plan is None
            else tuple(
                (SubjectRows(rows=chunk.rows), chunk.code)
                for chunk in group_plan.chunks
            )
        )
        host_device = (
            chunk_host_device(subject_devices=call_inputs.devices)
            if len(windows) > 1
            else None
        )

        chunk_results: list[
            dict[RegimeName, dict[int, PeriodRegimeSimulationData]]
        ] = []
        dissolution_records: list[DissolutionFallback] = []
        completed_setup = (
            None
            if prepared_chunks is None
            else measure_buffer_footprint(
                tree=(initial_conditions, initial_own_stakeholder, starting_periods)
            )
        )
    for group_code, group_windows in _window_groups(
        windows=windows,
        codes=None if component_values is None else component_values.codes,
    ):
        values = (
            period_to_regime_to_V_arr
            if component_values is None
            else component_values.acquire(code=cast("int", group_code))
        )
        try:
            for subject_slice, code in group_windows:
                with solve_phase(
                    name="simulation_chunk", logger=logger, call_id=call_id
                ):
                    if prepared_chunks is not None:
                        if memory is None:
                            raise ExecutionPlanningError(  # noqa: TRY301
                                "A prepared chunk requires its live memory owner."
                            )
                        prepared_chunks.require_chunk(
                            memory=memory, completed_setup=completed_setup
                        )
                    # Ungrouped, `n_subjects` is padded up to a multiple of
                    # `batch_size` upstream (see `pad_initial_conditions_to_multiple`),
                    # so every chunk — including the last — is exactly `batch_size`
                    # rows; the trailing pad rows are dropped once, after the loop,
                    # by `trim_pad_from_raw_results`. Grouped, every chunk gathers
                    # `batch_size` original rows of one code.
                    chunk = _simulate_subject_chunk(
                        initial_states={
                            name: _population_window(
                                array=array, window=subject_slice, memory=memory
                            )
                            for name, array in initial_states.items()
                        },
                        initial_regime_ids=_population_window(
                            array=initial_conditions["regime_id"],
                            window=subject_slice,
                            memory=memory,
                        ),
                        initial_own_stakeholder=_population_window(
                            array=initial_own_stakeholder,
                            window=subject_slice,
                            memory=memory,
                        ),
                        starting_periods=_population_window(
                            array=starting_periods,
                            window=subject_slice,
                            memory=memory,
                        ),
                        n_subjects=n_subjects,
                        subject_slice=subject_slice,
                        code=code,
                        stored_codes=(
                            None
                            if component_values is None
                            else (cast("int", group_code),)
                        ),
                        original_n_subjects=original_n_subjects,
                        regimes=regimes,
                        regime_names_to_ids=regime_names_to_ids,
                        regime_ids_to_names=regime_ids_to_names,
                        period_to_regime_to_V_arr=values,
                        period_to_regime_to_sim_policy=period_to_regime_to_sim_policy,
                        participation_surfaces=participation_surfaces,
                        period_to_regime_to_replay_reader=(
                            period_to_regime_to_replay_reader
                        ),
                        flat_params=flat_params,
                        ages=ages,
                        seed=seed,
                        taste_shock_seed=taste_shock_seed,
                        taste_addresses=taste_addresses,
                        logger=logger,
                        device_ids=device_ids,
                        memory=memory,
                        call_inputs=call_inputs,
                        process_grid_resolver=process_grid_resolver,
                        shock_paths=shock_paths,
                        current_iid_overrides=current_iid_overrides,
                        process_code_reads=process_code_reads,
                        dissolution_records=dissolution_records,
                    )
                    if host_device is not None:
                        # `block_until_ready` forces the D2H copy to complete
                        # before the loop continues, so the chunk's device buffers
                        # become free for the next chunk; the host-resident copies
                        # stay `jax.Array` (CPU-backed).
                        chunk = offload_chunk(
                            tree=chunk, host_device=host_device, memory=memory
                        )
                    chunk_results.append(chunk)
                    if memory is not None:
                        memory.close_unit()
                        memory.set_chunk_inputs(tree=())
                        memory.replace_outputs(tree=chunk_results)
            if component_values is not None and group_windows:
                # Every chunk of the code has finished reading its values
                # before they are released.
                jax.block_until_ready(chunk_results[-len(group_windows) :])
        except BaseException:
            if component_values is not None:
                component_values.abandon(code=cast("int", group_code))
            raise
        if component_values is not None:
            component_values.release(code=cast("int", group_code))

    with solve_phase(name="simulation_completion", logger=logger, call_id=call_id):
        # Only a selection of codes no subject holds runs no chunk at all.
        simulation_results = (
            _concatenate_chunk_results(
                chunk_results=chunk_results, regimes=regimes, memory=memory
            )
            if chunk_results
            else {}
        )
        if group_plan is not None:
            simulation_results = _restore_subject_order(
                results=simulation_results, plan=group_plan, memory=memory
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

    with solve_phase(name="simulation_result", logger=logger, call_id=call_id):
        # Wrap results in MappingProxyType for immutability
        wrapped_results = MappingProxyType(
            {
                regime_name: MappingProxyType(period_results)
                for regime_name, period_results in simulation_results.items()
            }
        )

        # Drop any subject-axis alignment pad rows so `SimulationResult` exposes
        # only the user's real subjects (see `pad_initial_conditions_to_multiple`).
        # No-op when `original_n_subjects` already matched the dispatched leading
        # axis.
        if original_n_subjects is not None:
            wrapped_results = trim_pad_from_raw_results(
                raw_results=wrapped_results,
                original_n_subjects=original_n_subjects,
                memory=memory,
            )

        # Which regimes can publish a nested continuous-outer read. Their declared
        # route says so, so the `nested_policy_fallback` column it gates is a
        # property of the model rather than of the payloads a solve retained.
        nested_policy_regimes = frozenset(
            regime_name
            for regime_name, regime in regimes.items()
            if regime.simulation.replay_route.consumer_route == "nnbegm_nested"
        )

        result = SimulationResult(
            raw_results=wrapped_results,
            regimes=regimes,
            flat_params=flat_params,
            period_to_regime_to_V_arr=(
                period_to_regime_to_V_arr
                if component_values is None
                else MappingProxyType({})
                if subject_codes is not None
                else component_values.values()
            ),
            ages=ages,
            simulation_output_dtypes=simulation_output_dtypes,
            subject_batch_size=subject_batch_size,
            nested_policy_regimes=nested_policy_regimes,
            randomness={
                "address_version": RANDOM_ADDRESS_VERSION,
                "prng_impl": str(jax.random.key_impl(seed))
                if isinstance(seed, jax.Array)
                else "threefry2x32",
                "taste_prng_impl": str(jax.random.key_impl(taste_shock_seed))
                if isinstance(taste_shock_seed, jax.Array)
                else "threefry2x32",
                "precision": "float64" if jax.config.jax_enable_x64 else "float32",
            },
        )
        # Diagnostic-only attribute, mirroring how `Model.simulate` attaches
        # `_solution`: never part of the constructor's persisted fields, so it is
        # not written by `SimulationResult.save` and is `None` after `load`.
        result._plan_summary = plan_summary  # noqa: SLF001
        result._dissolution_fallbacks = tuple(dissolution_records)  # noqa: SLF001
        if subject_codes is not None and group_plan is not None:
            result._subject_rows = group_plan.rows  # noqa: SLF001
    return result


def _log_simulation_plan(
    *, logger: logging.Logger, plan_summary: SimulationPlanSummary
) -> None:
    """Emit the plan lines only at the tiers that show them.

    Both lines are formatted from the record on demand, so a run at the warning
    or off tier pays nothing for them.
    """
    if logger.isEnabledFor(logging.INFO):
        logger.info(plan_summary.summary())
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(plan_summary.details())


def _window_groups(
    *,
    windows: tuple[tuple[slice | SubjectRows, int | None], ...],
    codes: tuple[int, ...] | None,
) -> tuple[tuple[int | None, tuple[tuple[slice | SubjectRows, int | None], ...]], ...]:
    """Return the windows as one group per code, or as one group without codes.

    With `codes`, every code gets a group in that order, empty when no window
    holds it; a window's code is the code of its group.
    """
    if codes is None:
        return ((None, windows),)
    return tuple(
        (code, tuple(window for window in windows if window[1] == code))
        for code in codes
    )


def _initialize_chunk_state(
    *,
    initial_states: dict[StateOrActionName, Float1D | IntND],
    initial_regime_ids: Int1D,
    initial_own_stakeholder: Int1D,
    regimes: MappingProxyType[RegimeName, Regime],
    device_ids: tuple[int, ...],
    seed: int | PRNGKey,
    memory: SimulationMemory | None,
) -> tuple[StatesPerRegime, Int1D, Int1D, PRNGKeyND]:
    """Build a chunk's carriers, enrolling completed arrays before the next step."""
    key = create_simulation_key(seed=seed, memory=memory)
    states = build_initial_states(
        initial_states=initial_states,
        regimes=regimes,
        device_ids=device_ids,
        memory=memory,
    )
    if memory is not None:
        memory.hold(tree=(states, key))
    regime_ids, own_stakeholder = initialize_subject_membership(
        initial_regime_ids=initial_regime_ids,
        initial_own_stakeholder=initial_own_stakeholder,
        memory=memory,
    )
    return states, regime_ids, own_stakeholder, key


def _simulate_subject_chunk(  # noqa: PLR0915
    *,
    initial_states: dict[StateOrActionName, Float1D | IntND],
    initial_regime_ids: Int1D,
    starting_periods: Int1D,
    n_subjects: int,
    subject_slice: slice | SubjectRows,
    code: int | None = None,
    stored_codes: tuple[int, ...] | None = None,
    original_n_subjects: int | None = None,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_ids_to_names: RegimeIdsToNames,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    flat_params: FlatParams,
    ages: TimeAxis,
    seed: int | PRNGKey,
    logger: logging.Logger,
    initial_own_stakeholder: Int1D,
    period_to_regime_to_sim_policy: PeriodToRegimeToSimulationPolicy | None = None,
    participation_surfaces: PeriodToRegimeToParticipationSurfaces = MappingProxyType(
        {}
    ),
    period_to_regime_to_replay_reader: _PeriodToRegimeToReplayReader = (
        MappingProxyType({})
    ),
    device_ids: tuple[int, ...] = (),
    memory: SimulationMemory | None = None,
    call_inputs: SimulationCallInputs | None = None,
    taste_shock_seed: int | PRNGKey | None = None,
    taste_addresses: Mapping[
        tuple[int, RegimeName], tuple[int, ...]
    ] = MappingProxyType({}),
    process_grid_resolver: ProcessGridResolver | None = None,
    shock_paths: Mapping[StateName, np.ndarray] = MappingProxyType({}),
    current_iid_overrides: CurrentIIDOverrides | None = None,
    process_code_reads: Mapping[
        RegimeName, Mapping[StateName, _ContinuousStochasticProcess]
    ] = MappingProxyType({}),
    dissolution_records: list[DissolutionFallback] | None = None,
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Run the full period loop for one chunk of subjects.

    `initial_states`, `initial_regime_ids`, and `starting_periods` are already
    sliced to this chunk's subjects; `n_subjects` and `subject_slice` describe the
    chunk's position in the full population so RNG keys stay full-population and are
    sliced, or for a grouped chunk gathered, by original row. `code` is the code of
    the grouping state every subject of a grouped chunk holds, `None` ungrouped;
    `stored_codes` are the codes the stored values hold along that state's
    axis, `None` for every code.
    The key stream is re-derived from `seed` here so the per-period carry is
    identical across chunks (it is subject-count-independent).
    `initial_own_stakeholder`: each subject's seeded role, already sliced to
    this chunk. It is carried through the period loop and updated wherever a
    gated edge routes a row, so a dissolution follows the row's own leg.
    `device_ids`: the model's device ids, ascending; empty names every device
    JAX reports.

    Returns the per-(regime, period) results for this chunk's subjects.
    """
    chunk_inputs = prepare_simulation_chunk_inputs(
        initial_states=initial_states,
        initial_regime_ids=initial_regime_ids,
        initial_own_stakeholder=initial_own_stakeholder,
        starting_periods=starting_periods,
        flat_params=flat_params,
        regimes=regimes,
        device_ids=device_ids,
        memory=memory,
        call_inputs=call_inputs,
        process_grid_resolver=process_grid_resolver,
    )
    devices = chunk_inputs.devices
    flat_params = chunk_inputs.flat_params
    initial_regime_ids = chunk_inputs.initial_regime_ids
    initial_own_stakeholder = chunk_inputs.initial_own_stakeholder
    starting_periods = chunk_inputs.starting_periods
    base_state_action_spaces = chunk_inputs.base_state_action_spaces
    states, subject_regime_ids, own_stakeholder, key = _initialize_chunk_state(
        initial_states=initial_states,
        initial_regime_ids=initial_regime_ids,
        initial_own_stakeholder=initial_own_stakeholder,
        regimes=regimes,
        device_ids=device_ids,
        seed=seed,
        memory=memory,
    )

    taste_key = create_taste_shock_key(
        seed=taste_shock_seed if taste_addresses else None,
        memory=memory,
        live_inputs=(states, subject_regime_ids, own_stakeholder, key),
    )

    simulation_results: dict[RegimeName, dict[int, PeriodRegimeSimulationData]] = {
        regime_name: {} for regime_name in regimes
    }
    # Each row's role in the previous period and the source whose gate it
    # passed open into this period: a household without a feasible action takes
    # the dissolution route of that gate, once.
    source_roles: Int1D | None = None
    entered_through: Int1D | None = None

    for period, active_regimes in enumerate(
        forward_regimes_by_period(regimes=regimes, n_periods=ages.n_periods)
    ):
        period_start = time.monotonic()
        if memory is not None:
            # The previous period's roles and open-branch entries are read by
            # this period's dissolution routes, so they stay roots until then.
            memory.unit_inputs = (
                states,
                subject_regime_ids,
                own_stakeholder,
                key,
                taste_key,
                source_roles,
                entered_through,
            )
        original_age = chunk_operations.period_age(
            values=ages.values, period=period, memory=memory
        )
        age = place_simulation_arguments(
            arguments={"age": original_age},
            subject_arg_names=(),
            value_reads=(),
            devices=devices,
            budget_bytes=None if memory is None else memory.budget_bytes,
            live_footprint=None if memory is None else memory.budget_snapshot(),
            budget_devices=() if memory is None else memory.devices,
        )["age"]
        if memory is not None:
            memory.hold(tree=age)

        subject_regime_ids, own_stakeholder = activate_subject_membership(
            period=period,
            starting_periods=starting_periods,
            initial_regime_ids=initial_regime_ids,
            initial_own_stakeholder=initial_own_stakeholder,
            regime_ids=subject_regime_ids,
            own_stakeholder=own_stakeholder,
            memory=memory,
        )

        new_subject_regime_ids = subject_regime_ids
        new_own_stakeholder = own_stakeholder
        new_entered_through = run_simulation_operation(
            memory=memory,
            function=empty_gated_entries,
            arguments={"regime_ids": subject_regime_ids},
            subject_arg_names=("regime_ids",),
            subject_outputs=True,
        )

        log_period_header(
            logger=logger,
            age=age,
            n_active_regimes=len(active_regimes),
            time_kind=coordinate_kind(ages),
        )

        period_policies = (period_to_regime_to_sim_policy or {}).get(period, {})
        period_readers = period_to_regime_to_replay_reader.get(period, {})
        executor = next(iter(regimes.values())).simulation.programs.executor
        owner = PeriodSimulationReads(
            period=period,
            devices=devices,
            reads_by_unit={
                name: unit_value_reads(
                    regime=active,
                    name=name,
                    period=period,
                    values=period_to_regime_to_V_arr,
                    policy=period_policies.get(name),
                    reader=period_readers.get(name),
                )
                for name, active in active_regimes.items()
            },
            release_enabled=isinstance(executor, SimulationRuntime)
            and executor.enable_jit,
            before_transfer=None if memory is None else memory.before_transfer,
            grouping=None
            if code is None
            else next(iter(regimes.values())).simulation.programs.grouping,
            code=code,
            stored_codes=stored_codes,
        )
        if memory is not None:
            memory.period_owner = owner

        states, subject_regime_ids, own_stakeholder, prepared_exits = (
            prepare_exit_period(
                regimes=regimes,
                period=period,
                states=states,
                regime_ids=subject_regime_ids,
                roles=own_stakeholder,
                entered_through=entered_through,
                source_roles=source_roles,
                regime_names_to_ids=regime_names_to_ids,
                flat_params=flat_params,
                ages=ages,
                values=period_to_regime_to_V_arr,
                surfaces=participation_surfaces,
                owner=owner,
                key=key,
                subject_slice=subject_slice,
                original_n_subjects=n_subjects
                if original_n_subjects is None
                else original_n_subjects,
                memory=memory,
                taste_key=taste_key,
                taste_addresses=taste_addresses,
                current_iid_overrides=current_iid_overrides,
                process_code_reads=process_code_reads,
            )
        )
        # Current exits are simultaneous. The scheduled transition accumulators
        # start from final acting membership, after every original choice froze.
        new_subject_regime_ids = subject_regime_ids
        new_own_stakeholder = own_stakeholder

        ran_this_period: list[RegimeName] = []
        for regime_name, regime in active_regimes.items():
            dispatched_regime = _bind_unit_executor(
                regime=regime,
                memory=memory,
                inputs=(
                    states,
                    subject_regime_ids,
                    new_subject_regime_ids,
                    own_stakeholder,
                    new_own_stakeholder,
                    key,
                    taste_key,
                    age,
                    source_roles,
                    entered_through,
                    new_entered_through,
                    prepared_exits,
                ),
            )
            (
                result,
                new_states,
                new_subject_regime_ids,
                new_own_stakeholder,
                subject_regime_ids,
                own_stakeholder,
                entered_through,
                new_entered_through,
            ) = _simulate_regime_in_period(
                regime_name=regime_name,
                regime=dispatched_regime,
                regimes=regimes,
                value_owner=owner,
                subject_devices=devices,
                memory=memory,
                base_state_action_space=base_state_action_spaces[regime_name],
                period=period,
                age=age,
                states=states,
                subject_regime_ids=subject_regime_ids,
                new_subject_regime_ids=new_subject_regime_ids,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                base_state_action_spaces=base_state_action_spaces,
                sim_policy=(period_to_regime_to_sim_policy or {})
                .get(period, {})
                .get(regime_name),
                replay_reader=period_to_regime_to_replay_reader.get(period, {}).get(
                    regime_name
                ),
                flat_params=flat_params,
                regime_names_to_ids=regime_names_to_ids,
                decision_targets_next_period=(
                    ()
                    if period == ages.n_periods - 1
                    else regime.solution.reachability.targets(
                        period=period, source=regime_name
                    )
                ),
                realization_targets_next_period=(
                    ()
                    if period == ages.n_periods - 1
                    else regime.simulation.reachability.targets(
                        period=period, source=regime_name
                    )
                ),
                key=key,
                taste_key=taste_key,
                taste_address=taste_addresses.get((period, regime_name)),
                current_iid_overrides=current_iid_overrides,
                logger=logger,
                n_subjects=n_subjects,
                subject_slice=subject_slice,
                original_n_subjects=original_n_subjects,
                supplied_shocks=shock_path_values(
                    paths=shock_paths,
                    argument_names=supplied_shock_argument_names(
                        regime=regime, period=period
                    ),
                    period=period + 1,
                    subject_slice=subject_slice,
                    n_subjects=len(subject_regime_ids),
                )
                if shock_paths
                else MappingProxyType({}),
                own_stakeholder=own_stakeholder,
                new_own_stakeholder=new_own_stakeholder,
                gated_edge_fold_age=(
                    coordinate_at(ages=ages, period=period + 1)
                    if regime.gated_edges and period + 1 < ages.n_periods
                    else None
                ),
                source_roles=source_roles,
                entered_through=entered_through,
                new_entered_through=new_entered_through,
                process_code_reads=process_code_reads,
                dissolution_records=dissolution_records,
                ran_this_period=tuple(ran_this_period),
                prepared_decision=prepared_exits.get(regime_name),
            )
            ran_this_period.append(regime_name)
            states = new_states
            simulation_results[regime_name][period] = result
            if memory is not None:
                memory.publish(tree=result)
            owner.commit(
                unit=regime_name,
                outputs=(
                    result,
                    states,
                    new_subject_regime_ids,
                    new_own_stakeholder,
                    key,
                ),
            )
            if memory is not None:
                cast(
                    "SimulationUnitExecutor",
                    dispatched_regime.simulation.programs.executor,
                ).close()
                memory.close_unit()

        del prepared_exits
        owner.finish()
        if memory is not None:
            memory.period_owner = None
            # Unit-local owners were released after publication. The period's
            # current carry, both memberships and both entry carriers still live
            # through diagnostics.
            memory.unit_inputs = (
                states,
                subject_regime_ids,
                new_subject_regime_ids,
                own_stakeholder,
                new_own_stakeholder,
                key,
                taste_key,
                original_age,
                age,
                source_roles,
                entered_through,
                new_entered_through,
            )

        _validate_period_values(
            logger=logger,
            age=age,
            time_kind=coordinate_kind(ages),
            memory=memory,
            period_results=tuple(
                (regime_name, simulation_results[regime_name][period])
                for regime_name in active_regimes
            ),
        )

        # The period's final membership, after dissolution routes moved rows
        # into the regime they acted in, is where each transition starts.
        prev_regime_ids = subject_regime_ids
        source_roles = own_stakeholder
        entered_through = new_entered_through
        subject_regime_ids = new_subject_regime_ids
        own_stakeholder = new_own_stakeholder

        log_regime_transition_counts(
            logger=logger,
            prev_regime_ids=prev_regime_ids,
            new_regime_ids=subject_regime_ids,
            regime_ids_to_names=regime_ids_to_names,
            counts_factory=(
                None
                if memory is None
                else functools.partial(
                    profiled_transition_counts,
                    memory=memory,
                    prev_regime_ids=prev_regime_ids,
                    new_regime_ids=subject_regime_ids,
                    sorted_ids=tuple(sorted(regime_ids_to_names)),
                )
            ),
        )

        elapsed = time.monotonic() - period_start
        log_period_timing(logger=logger, elapsed=elapsed)

    return simulation_results


def _bind_unit_executor(
    *, regime: Regime, memory: SimulationMemory | None, inputs: ArrayTree
) -> Regime:
    """Bind live unit residency without changing any persistent regime bundle."""
    if memory is None:
        return regime
    memory.unit_inputs = inputs
    return replace(
        regime,
        simulation=replace(
            regime.simulation,
            programs=replace(
                regime.simulation.programs,
                executor=SimulationUnitExecutor(
                    runtime=cast(
                        "SimulationRuntime", regime.simulation.programs.executor
                    ),
                    live_footprint=memory.budget_snapshot,
                    budget_devices=memory.devices,
                    axis_widths=memory.axis_widths,
                    on_output=memory.hold,
                ),
            ),
        ),
    )


def _concatenate_chunk_results(
    *,
    chunk_results: list[dict[RegimeName, dict[int, PeriodRegimeSimulationData]]],
    regimes: MappingProxyType[RegimeName, Regime],
    memory: SimulationMemory | None = None,
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
            merged = {
                field.name: jax.tree.map(
                    lambda *arrays: concatenate_arrays(arrays=arrays, memory=memory),
                    *(getattr(data, field.name) for data in per_chunk),
                )
                for field in fields(PeriodRegimeSimulationData)
            }
            combined[regime_name][period] = replace(per_chunk[0], **merged)
    return combined


def _population_window(
    *, array: jax.Array, window: slice | SubjectRows, memory: SimulationMemory | None
) -> jax.Array:
    """Take one chunk's rows of a population array: a window, or original rows."""
    if isinstance(window, SubjectRows):
        return chunk_operations.take_population(
            array=array, rows=window.rows, memory=memory
        )
    return chunk_operations.slice_population(
        array=array,
        start=window.start,
        width=window.stop - window.start,
        memory=memory,
    )


def _restore_subject_order(
    *,
    results: dict[RegimeName, dict[int, PeriodRegimeSimulationData]],
    plan: SubjectGroupPlan,
    memory: SimulationMemory | None = None,
) -> dict[RegimeName, dict[int, PeriodRegimeSimulationData]]:
    """Gather every grouped output back into original subject order.

    Each original row reads its own output; the outputs of a group's repeated
    tail rows are read by none. Gathering copies bytes, so values, signed zeros
    and NaN payloads are unchanged.
    """
    return {
        regime_name: {
            period: replace(
                data,
                **{
                    field.name: jax.tree.map(
                        lambda array: take_rows(
                            array=array, rows=plan.positions, memory=memory
                        ),
                        getattr(data, field.name),
                    )
                    for field in fields(PeriodRegimeSimulationData)
                },
            )
            for period, data in periods.items()
        }
        for regime_name, periods in results.items()
    }


def _require_next_period_values(
    *,
    next_period_values: Mapping[RegimeName, FloatND],
    required_targets: tuple[RegimeName, ...],
    source_regime_name: RegimeName,
    source_period: int,
) -> MappingProxyType[RegimeName, FloatND]:
    """Validate and select the solution-graph continuation values a decision needs.

    Args:
        next_period_values: The caller-supplied value-function arrays for the
            target period (`source_period + 1`), keyed by regime name.
        required_targets: This source's retained *solution*-graph targets —
            every one must have an entry in `next_period_values`.
        source_regime_name: The source regime making the decision.
        source_period: The source period (0-indexed).

    Returns:
        Immutable mapping of just the required targets to their value arrays.

    Raises:
        InvalidSimulationInputError: If any required target is missing from
            `next_period_values`.

    """
    missing = tuple(sorted(set(required_targets) - next_period_values.keys()))
    if missing:
        msg = (
            f"Missing continuation values for source regime '{source_regime_name}' "
            f"at period {source_period} (target period {source_period + 1}). "
            f"Required solution-graph targets: {required_targets}. "
            f"Missing: {missing}. Supply a solution produced by this model or "
            f"provide all required arrays."
        )
        raise InvalidSimulationInputError(msg)
    return MappingProxyType(
        {target: next_period_values[target] for target in required_targets}
    )


def _referenced_value_kwargs(
    *,
    regime: Regime,
    period_to_regime_to_V_arr: Mapping[int, Mapping[RegimeName, FloatND]],
    flat_params: FlatParams,
    period: int,
) -> dict[ReferenceName, QAndFArg]:
    """Build the value mappings one regime's kernel reads beside its own grids.

    Two channels, each empty for a regime that declares nothing on it, and each
    carrying the referenced value together with the reference regime's OWN
    params — the split the solve-side grid-search kernel makes, so both phases
    read one operator:

    - a value constraint's same-period reference, at THIS period's value;
    - a gated edge's leg fallbacks, read where the row
      lands and therefore at the value of the period it lands IN.

    Args:
        regime: The canonical regime being simulated.
        period_to_regime_to_V_arr: Solved values, by period and regime.
        flat_params: Immutable mapping of regime names to flat params.
        period: The period the row decides in; it lands in the next one.

    Returns:
        Mapping of engine argument names to their per-regime mappings.

    """
    kwargs: dict[ReferenceName, QAndFArg] = {}
    if regime.same_period_ref_regimes:
        this_period_V = period_to_regime_to_V_arr.get(period, MappingProxyType({}))
        kwargs[SAME_PERIOD_V_ARG] = MappingProxyType(
            {ref: this_period_V[ref] for ref in regime.same_period_ref_regimes}
        )
        kwargs[SAME_PERIOD_PARAMS_ARG] = MappingProxyType(
            {
                ref: regime_kernel_params(flat_params, regime_name=ref)
                for ref in regime.same_period_ref_regimes
            }
        )
    # The same per-period set the AOT lowering consults, so a compiled
    # program's pytree and this call's arguments name the same channels — and
    # only the regimes this period's edges land in, since an edge whose target
    # is not declared at `period + 1` reads nothing there.
    edge_reference_regimes = regime.simulation.edge_reference_regimes_by_period.get(
        period
    )
    if edge_reference_regimes is not None:
        landing_V = period_to_regime_to_V_arr.get(period + 1, MappingProxyType({}))
        kwargs[EDGE_REF_V_ARG] = MappingProxyType(
            {ref: landing_V[ref] for ref in edge_reference_regimes}
        )
        kwargs[EDGE_REF_PARAMS_ARG] = MappingProxyType(
            {
                ref: regime_kernel_params(flat_params, regime_name=ref)
                for ref in edge_reference_regimes
            }
        )
    return kwargs


def _read_external_replay(
    *,
    reader: ReplayReader,
    regime_name: RegimeName,
    regime: Regime,
    period: int,
    age: ScalarInt | ScalarFloat,
    states: Mapping[StateName, FloatND | IntND],
    state_action_space: StateActionSpace,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    referenced_value_kwargs: QAndFKwargs,
    subject_ids_in_regime: BoolND,
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]:
    """Run one preflighted external reader and verify its chosen actions."""
    n_subjects = subject_ids_in_regime.shape[0]
    fallback_actions = MappingProxyType(
        {
            name: jnp.broadcast_to(jnp.asarray(nodes)[0], (n_subjects,))
            for name, nodes in state_action_space.actions.items()
        }
    )
    try:
        replay_output = reader(
            states=MappingProxyType(dict(states)),
            fallback_actions=fallback_actions,
        )
    except Exception as error:
        raise InvalidSimulationInputError(
            f"External replay reader failed at ({period}, {regime_name!r}): {error}"
        ) from error
    if type(replay_output) is not ActionOutput:
        raise InvalidSimulationInputError(
            "External replay reader returned the wrong output type at "
            f"({period}, {regime_name!r}); expected ActionOutput."
        )
    expected_action_names = set(state_action_space.action_names)
    actual_action_names = set(replay_output.actions)
    if actual_action_names != expected_action_names:
        raise InvalidSimulationInputError(
            "External replay reader returned incompatible action names at "
            f"({period}, {regime_name!r}): expected "
            f"{tuple(state_action_space.action_names)!r}, got "
            f"{tuple(replay_output.actions)!r}."
        )
    optimal_actions = _canonicalize_external_replay_actions(
        replay_output=replay_output,
        state_action_space=state_action_space,
        subject_ids_in_regime=subject_ids_in_regime,
        n_subjects=n_subjects,
        regime_name=regime_name,
        period=period,
    )
    value, replay_feasible = _canonical_Q_at_actions(
        candidate_actions=optimal_actions,
        regime=regime,
        canonical_states=state_action_space.states,
        action_names=state_action_space.action_names,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=flat_params,
        referenced_value_kwargs=referenced_value_kwargs,
        period=period,
        age=age,
        active_subjects=subject_ids_in_regime,
    )
    if bool(jnp.any(subject_ids_in_regime & ~replay_feasible)):
        raise InvalidSimulationInputError(
            "External replay reader returned an infeasible action for at least "
            f"one subject at ({period}, {regime_name!r})."
        )
    return optimal_actions, value


def _canonicalize_external_replay_actions(
    *,
    replay_output: ActionOutput,
    state_action_space: StateActionSpace,
    subject_ids_in_regime: BoolND,
    n_subjects: int,
    regime_name: RegimeName,
    period: int,
) -> MappingProxyType[ActionName, FloatND | IntND]:
    """Broadcast reader actions and reject invalid values before canonical scoring."""
    actions: dict[ActionName, FloatND | IntND] = {}
    for name in state_action_space.action_names:
        try:
            action = jnp.broadcast_to(
                jnp.asarray(replay_output.actions[name]),
                (n_subjects,),
            )
        except (TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                "External replay reader returned an action that cannot be converted "
                f"and broadcast for {name!r} at ({period}, {regime_name!r}): {error}"
            ) from error

        if not (
            jnp.issubdtype(action.dtype, jnp.integer)
            or jnp.issubdtype(action.dtype, jnp.floating)
        ):
            raise InvalidSimulationInputError(
                "External replay reader returned a non-numeric action for "
                f"{name!r} at ({period}, {regime_name!r})."
            )
        if bool(jnp.any(subject_ids_in_regime & ~jnp.isfinite(action))):
            raise InvalidSimulationInputError(
                "External replay reader returned a non-finite action for "
                f"{name!r} at ({period}, {regime_name!r})."
            )

        discrete_domain = state_action_space.discrete_actions.get(name)
        if discrete_domain is not None:
            if not jnp.issubdtype(action.dtype, jnp.integer):
                raise InvalidSimulationInputError(
                    "External replay reader returned a non-integer categorical code "
                    f"for action {name!r} at ({period}, {regime_name!r})."
                )
            domain = jnp.asarray(discrete_domain)
            in_domain = jnp.any(action[:, None] == domain[None, :], axis=1)
            if bool(jnp.any(subject_ids_in_regime & ~in_domain)):
                raise InvalidSimulationInputError(
                    "External replay reader returned an action outside its declared "
                    f"categorical domain for {name!r} at "
                    f"({period}, {regime_name!r})."
                )
            action = action.astype(domain.dtype)

        actions[name] = action

    return MappingProxyType(actions)


def _validate_period_values(
    *,
    logger: logging.Logger,
    age: ScalarInt | ScalarFloat,
    period_results: tuple[tuple[RegimeName, PeriodRegimeSimulationData], ...],
    memory: SimulationMemory | None = None,
    time_kind: Literal["age", "period"] = "age",
) -> None:
    """Validate one period's simulated values for every regime active in it.

    The per-regime NaN and Inf reductions run as one compiled program over
    the period's value arrays, so a period costs a single device-to-host
    transfer whatever the number of regimes. Only a regime the NaN row accuses
    pays for the enriched value-function report, which reads the device again
    to describe what it found. With ``runtime_checks=False`` no reduction is
    issued.

    Every offending regime of the period is named first, then the enriched
    reports run. With runtime checks enabled the first enriched report raises, so
    naming the period's regimes first is what keeps the other offenders --- an
    Inf-only regime among them, which the enriched report never speaks about
    --- in the record of the aborted run.

    Args:
        logger: Logger carrying the runtime-validation policy.
        age: Age corresponding to the current period.
        period_results: Tuple of (regime name, that regime's simulated data for
            this period) pairs, in the order the period simulated them.
        memory: Optional current-period owner for profiled diagnostic admission.

    """
    if not validation_enabled(logger) or not period_results:
        return
    regime_names = tuple(regime_name for regime_name, _ in period_results)
    arguments = {
        "values": tuple(data.V_arr for _, data in period_results),
        "in_regime": tuple(data.in_regime for _, data in period_results),
    }
    flags = (
        non_finite_by_regime(**arguments)
        if memory is None
        else memory.run(
            function=period_value_flags,
            arguments=arguments,
            subject_arg_names=("values", "in_regime"),
        )
    )
    has_nan, has_non_finite = flags.tolist()
    log_non_finite_values(
        logger=logger,
        age=age,
        regime_names=regime_names,
        flags=tuple(has_non_finite),
        time_kind=time_kind,
    )
    for (regime_name, data), flag in zip(period_results, has_nan, strict=True):
        if flag:
            _validate_simulated_value(
                value=data.V_arr,
                subject_ids_in_regime=data.in_regime,
                age=age,
                regime_name=regime_name,
                logger=logger,
                memory=memory,
                time_kind=time_kind,
            )


def _validate_simulated_value(
    *,
    value: FloatND,
    subject_ids_in_regime: BoolND,
    age: ScalarInt | ScalarFloat,
    regime_name: RegimeName,
    logger: logging.Logger,
    memory: SimulationMemory | None = None,
    time_kind: Literal["age", "period"] = "age",
) -> None:
    """Validate values only for subjects whose current regime owns the rows."""
    if not validation_enabled(logger):
        return
    try:
        if memory is not None:
            n_nan = int(
                memory.run(
                    function=owned_value_nan_count,
                    arguments={"value": value, "in_regime": subject_ids_in_regime},
                    subject_arg_names=("value", "in_regime"),
                )
            )
            if n_nan:
                raise value_function_nan_error(
                    n_nan=n_nan,
                    total=int(value.size),
                    age=age,
                    regime_name=regime_name,
                    time_kind=time_kind,
                )
            return
        # Out-of-regime subjects carry placeholder entries (their state is
        # meaningless under this regime's problem). A collective value carries a
        # trailing stakeholder axis, so broadcast the ownership mask over it.
        in_regime_mask = subject_ids_in_regime.reshape(
            subject_ids_in_regime.shape
            + (1,) * (value.ndim - subject_ids_in_regime.ndim)
        )
        validate_V(
            logger=logger,
            V_arr=jnp.where(in_regime_mask, value, 0.0),
            age=age,
            regime_name=regime_name,
            time_kind=time_kind,
        )
    except InvalidValueFunctionError as error:
        raise_or_warn(logger=logger, error=error)


def _simulate_regime_in_period(  # noqa: C901, PLR0912, PLR0915
    *,
    regime_name: RegimeName,
    regime: Regime,
    regimes: Mapping[RegimeName, Regime],
    value_owner: PeriodSimulationReads,
    subject_devices: tuple[jax.Device, ...],
    memory: SimulationMemory | None,
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
    flat_params: FlatParams,
    regime_names_to_ids: RegimeNamesToIds,
    decision_targets_next_period: tuple[RegimeName, ...],
    realization_targets_next_period: tuple[RegimeName, ...],
    key: PRNGKeyND,
    logger: logging.Logger,
    n_subjects: int,
    subject_slice: slice | SubjectRows,
    original_n_subjects: int | None = None,
    own_stakeholder: Int1D,
    new_own_stakeholder: Int1D,
    gated_edge_fold_age: float | None = None,
    sim_policy: (
        EGMSimPolicy | NBEGMGridPolicy | NNBEGMSimPolicy | NestedEGMSimPolicy | None
    ) = None,
    replay_reader: PreparedReplayReader | ReplayReader | None = None,
    taste_key: PRNGKeyND | None = None,
    taste_address: tuple[int, ...] | None = None,
    supplied_shocks: Mapping[str, np.ndarray] = MappingProxyType({}),
    current_iid_overrides: CurrentIIDOverrides | None = None,
    source_roles: Int1D | None = None,
    entered_through: Int1D | None = None,
    new_entered_through: Int1D | None = None,
    process_code_reads: Mapping[
        RegimeName, Mapping[StateName, _ContinuousStochasticProcess]
    ] = MappingProxyType({}),
    dissolution_records: list[DissolutionFallback] | None = None,
    ran_this_period: tuple[RegimeName, ...] = (),
    prepared_decision: PreparedExitDecision | None = None,
) -> tuple[
    PeriodRegimeSimulationData,
    StatesPerRegime,
    Int1D,
    Int1D,
    Int1D,
    Int1D,
    Int1D | None,
    Int1D | None,
]:
    """Simulate one regime for one period.

    This function processes all subjects in a given regime for a single period,
    computing optimal actions, updating states, and determining next regime membership.

    Args:
        regime_name: Name of the current regime.
        regime: Internal representation of the regime.
        regimes: Every regime of the model, keyed by name. A gated edge folds
            its target's value on that target's own period-`t + 1` state axes,
            which are read off the target regime itself.
        base_state_action_space: The regime's params-completed state-action
            space, built once per simulate call.
        base_state_action_spaces: Every regime's params-completed state-action
            space — a gated edge's value-and-gate fold needs its TARGET
            regime's own state grids, not just this regime's.
        period: Current period (0-indexed).
        age: Age corresponding to current period.
        states: Carrier of current-period state arrays for every regime and
            state.
        subject_regime_ids: Current regime membership for all subjects.
        new_subject_regime_ids: Array to populate with next period's regime memberships.
        period_to_regime_to_V_arr: Value function arrays for all periods and regimes.
        flat_params: Model parameters for all regimes.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        decision_targets_next_period: This source's retained *solution*-graph
            targets next period — the continuation values the decision/Q
            evaluation reads. A caller-supplied `period_to_regime_to_V_arr`
            missing one of these raises `InvalidSimulationInputError`.
        realization_targets_next_period: This source's retained
            *simulation*-graph targets next period — the admissible set for
            the realized regime draw. May differ from
            `decision_targets_next_period` under a `Phased` regime
            transition.
        key: JAX random key for stochastic operations.
        n_subjects: Total number of subjects (the full population), used to keep RNG
            key generation independent of how subjects are chunked.
        subject_slice: Global-index slice, or original rows, of the subjects in
            this chunk.
        own_stakeholder: The role each subject occupies this period, as a code
            in the model's role vocabulary. Decides which leg of a gated edge
            a row follows, and is published beside the states.
        new_own_stakeholder: Array to populate with next period's roles, in the
            same accumulate-across-regimes manner as
            `new_subject_regime_ids`.
        gated_edge_fold_age: Age corresponding to `period + 1`; threaded to
            gated-edge folds, gate evaluators, and fallback projectors.
        sim_policy: The regime's published off-grid simulation policy for this
            period, or `None`. Consumed only where the regime qualifies
            (`regime.simulation.egm_policy_read`).
        replay_reader: Model-preflighted external policy reader for this exact
            result cell, or ``None`` for an engine-owned replay route.
        value_owner: This period's owner of addressed copies.
        subject_devices: Actual ordered devices evaluating the population.
        source_roles: Each row's role in the previous period, or `None` in a
            chunk's first period.
        entered_through: Each row's source regime id when it passed that
            source's gate into this period through the open branch and has not
            used the gate's route yet, else `NO_GATED_ENTRY`; `None` in a
            chunk's first period. A row of this regime with no feasible action
            and such an entry takes that gate's dissolution route this period
            instead of raising.
        new_entered_through: The next period's open-branch entries as
            accumulated by the regimes already routed this period, written into
            for this regime's rows.
        process_code_reads: Per regime, the process states whose node position
            its readers take.
        dissolution_records: Collects one record per rerouted subject when
            runtime checks are on.
        ran_this_period: The regimes that already ran this period. A row whose
            dissolution route leads into one of them cannot act this period.
        prepared_decision: Final actions already selected by the immutable
            original-subject exit pass, or `None` for an ordinary decision.

    Returns:
        Tuple containing:
        - PeriodRegimeSimulationData for this regime-period
        - Updated state carrier
        - Updated new_subject_regime_ids array
        - Updated new_own_stakeholder array
        - Current regime membership, after any dissolution route
        - Current roles, after any dissolution route
        - Current open-branch entries, with every route taken here used up
        - Updated next-period open-branch entries

    """
    # Select subjects in the current regime
    subject_ids_in_regime = chunk_operations.regime_mask(
        regime_ids=subject_regime_ids,
        regime_id=regime_names_to_ids[regime_name],
        memory=memory,
    )
    genuine = _genuine_rows(
        subject_slice=subject_slice,
        n_real=n_subjects if original_n_subjects is None else original_n_subjects,
    )
    # Whatever route brought a household here, the position it is read at must
    # be a node of this regime. A regime reading no position transfers nothing.
    coded = process_code_reads.get(regime_name, {})
    if coded:
        fail_if_a_held_process_code_is_off_node(
            states=states[regime_name],
            processes=coded,
            holders=subject_ids_in_regime & genuine,
            where=f"in regime '{regime_name}' at period {period}",
            memory=memory,
        )

    state_action_space = create_regime_state_action_space(
        regime=regime,
        regime_states=states[regime_name],
        base=base_state_action_space,
    )
    # The decision reads the next period's reachable values. In the last period
    # it receives an empty mapping because no continuation is consumed.
    next_period_values = period_to_regime_to_V_arr.get(period + 1, MappingProxyType({}))
    next_regime_to_V_arr = _require_next_period_values(
        next_period_values=next_period_values,
        required_targets=decision_targets_next_period,
        source_regime_name=regime_name,
        source_period=period,
    )

    # Fold only this period's addressed raw values. Copies are shared across
    # readers in the period and no fold survives into another subject chunk.
    raw_gate_reads = gate_reads(
        regime=regime,
        name=regime_name,
        period=period,
        values=period_to_regime_to_V_arr,
    )
    gate_values = acquire_gate_inputs(
        reads=tuple(
            read for read in raw_gate_reads if read.source.core_key == GATE_FOLD
        ),
        owner=value_owner,
        name=regime_name,
        values=period_to_regime_to_V_arr,
    )
    if period in regime.simulation.programs.gate_fold:
        folded_gate_values = cast(
            "Mapping[RegimeName, FloatND]",
            execute_simulation_program(
                programs=regime.simulation.programs,
                family="gate_fold",
                period=period,
                arguments={
                    **gate_fold_arguments(
                        edge_values=gate_values,
                        flat_params=flat_params,
                        fold_age=jnp.asarray(
                            0.0 if gated_edge_fold_age is None else gated_edge_fold_age
                        ),
                    ),
                    "next_regime_to_V_arr": next_regime_to_V_arr,
                    "target_states_by_target": MappingProxyType(
                        {
                            target: _states_for_period(
                                regime=regimes[target],
                                state_action_space=base_state_action_spaces[target],
                                period=period + 1,
                            )
                            for target in regime.gated_edges
                            if target in next_period_values
                        }
                    ),
                },
                n_subjects=n_subjects,
            ),
        )
        next_regime_to_V_arr = MappingProxyType(
            {**next_regime_to_V_arr, **folded_gate_values}
        )
    referenced_value_kwargs = _referenced_value_kwargs(
        regime=regime,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        flat_params=flat_params,
        period=period,
    )

    selected_reads = decision_reads(
        regime=regime, period=period, policy=sim_policy, reader=replay_reader
    )
    # All same-unit occurrences resolve from the same original source objects.
    policy_reads = (
        rekeyed_value_reads(
            reads=regime.simulation.programs.decision[period].requirements.value_reads,
            core_key=POLICY_SCORE,
        )
        if sim_policy is not None
        and replay_reader is None
        and regime.simulation.replay_route.consumer_route != "nnbegm_finite"
        else ()
    )
    next_regime_to_V_arr, referenced_value_kwargs = acquire_decision_inputs(
        reads=(*selected_reads, *policy_reads),
        owner=value_owner,
        name=regime_name,
        next_values=next_regime_to_V_arr,
        referenced=referenced_value_kwargs,
        stored_values=period_to_regime_to_V_arr,
    )
    if sim_policy is not None and replay_reader is None:
        sim_policy = place_replay_payload(
            payload=sim_policy,
            key=SIMULATION_POLICY,
            period=period,
            regime=regime_name,
            core="simulation_policy_replay",
            owner=value_owner,
        )
    built_reader = (
        replay_reader.build(owner=value_owner, devices=subject_devices)
        if isinstance(replay_reader, PreparedReplayReader)
        else replay_reader
    )

    n_chunk_subjects = subject_ids_in_regime.shape[0]
    replay_route = regime.simulation.replay_route
    direct_nnbegm_replay = replay_route.consumer_route == "nnbegm_finite" and (
        isinstance(sim_policy, NNBEGMSimPolicy)
    )
    current_shocks: Mapping[str, FloatND | IntND] = MappingProxyType({})
    if prepared_decision is not None:
        # The immutable original-source pass already made the one Full choice
        # and revealed a mover's final Participating action. Only this final
        # actor advances below; it must never make another Full decision here.
        current_shocks = prepared_decision.physical
        V_arr = prepared_decision.value
        optimal_actions = run_simulation_operation(
            memory=memory,
            function=_lookup_values_from_indices,
            arguments={
                "flat_indices": prepared_decision.indices,
                "grids": state_action_space.actions,
            },
            subject_arg_names=("flat_indices",),
        )
        nested_fallback = None
    elif built_reader is not None:
        optimal_actions, V_arr = _read_external_replay(
            reader=built_reader,
            regime_name=regime_name,
            regime=regime,
            period=period,
            age=age,
            states=states[regime_name],
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            flat_params=regime_kernel_params(flat_params, regime_name=regime_name),
            referenced_value_kwargs=referenced_value_kwargs,
            subject_ids_in_regime=run_simulation_operation(
                memory=memory,
                function=_active_subjects,
                arguments={"in_regime": subject_ids_in_regime, "genuine": genuine},
                subject_arg_names=("in_regime", "genuine"),
                subject_outputs=True,
            ),
        )
        nested_fallback = None
    elif direct_nnbegm_replay:
        _fail_if_a_finite_bank_state_is_off_grid(
            regime_name=regime_name,
            regime=regime,
            sim_policy=sim_policy,
            states=states[regime_name],
            subjects=run_simulation_operation(
                memory=memory,
                function=_active_subjects,
                arguments={"in_regime": subject_ids_in_regime, "genuine": genuine},
                subject_arg_names=("in_regime", "genuine"),
                subject_outputs=True,
            ),
            period=period,
            memory=memory,
        )
        # The payload names the complete finite candidate set, so the unrelated
        # Cartesian simulation argmax is neither compiled nor executed.
        optimal_actions, V_arr, nested_fallback = _execute_finite_replay(
            regime=regime,
            sim_policy=sim_policy,
            states=states[regime_name],
            canonical_states=state_action_space.states,
            flat_params=regime_kernel_params(flat_params, regime_name=regime_name),
            period=period,
            age=age,
            n_subjects=n_chunk_subjects,
            next_regime_to_V_arr=next_regime_to_V_arr,
            referenced_value_kwargs=referenced_value_kwargs,
        )
    else:
        taste_shock_kwargs = {}
        if regime.has_taste_shocks:
            if isinstance(subject_slice, SubjectRows):
                raise ExecutionPlanningError(
                    "Grouped simulation does not cover taste shocks."
                )
            key, gumbel_keys = prepare_decision_taste_keys(
                key=key,
                taste_key=taste_key,
                taste_address=taste_address,
                n_subjects=n_subjects,
                subject_slice=subject_slice,
                original_n_subjects=original_n_subjects,
                memory=memory,
                period=period,
            )
            taste_shock_kwargs = {"taste_shock_key": gumbel_keys}

        current_iid_keys = MappingProxyType({})
        if regime.simulation.current_iid_draw_plan is not None:
            if not isinstance(subject_slice, slice):
                raise ExecutionPlanningError(
                    "Current IID replay does not cover grouped subject rows."
                )
            current_iid_keys = prepare_current_iid_keys(
                plan=regime.simulation.current_iid_draw_plan,
                key=key,
                period=period,
                subject_slice=subject_slice,
                original_n_subjects=n_subjects
                if original_n_subjects is None
                else original_n_subjects,
                memory=memory,
            )

        def decide(
            *,
            space: StateActionSpace,
            in_regime: Bool1D,
            accepted: BoolND | None = None,
        ) -> tuple[IntND, FloatND, int]:
            nonlocal current_shocks
            active_subjects = (
                run_simulation_operation(
                    memory=memory,
                    function=_active_subjects,
                    arguments={"in_regime": in_regime, "genuine": genuine},
                    subject_arg_names=("in_regime", "genuine"),
                    subject_outputs=True,
                )
                if declares_active_subjects(
                    program=regime.simulation.programs.forward_decision[period]
                )
                else None
            )
            decision_outputs = execute_simulation_program(
                programs=regime.simulation.programs,
                family="decision",
                period=period,
                n_subjects=n_chunk_subjects,
                arguments=decision_arguments(
                    current_iid=current_iid_arguments(
                        regime=regime,
                        flat_params=flat_params,
                        keys=current_iid_keys,
                        overrides=gather_current_iid_overrides(
                            bank=current_iid_overrides,
                            names=()
                            if regime.simulation.current_iid_draw_plan is None
                            else regime.simulation.current_iid_draw_plan.physical_names,
                            period=period,
                            subject_slice=subject_slice,
                        ),
                    ),
                    states={
                        **space.states,
                        **(
                            {
                                name: states[regime_name][name]
                                for name in (
                                    regime.simulation.current_iid_draw_plan.state_inputs
                                )
                            }
                            if regime.simulation.current_iid_draw_plan is not None
                            else {}
                        ),
                    },
                    discrete_actions=space.discrete_actions,
                    continuous_actions=space.continuous_actions,
                    taste_keys=taste_shock_kwargs,
                    next_values=next_regime_to_V_arr,
                    references=referenced_value_kwargs,
                    params=regime_kernel_params(flat_params, regime_name=regime_name),
                    period=jnp.int32(period) if memory is None else np.int32(period),
                    age=age,
                    active_subjects=active_subjects,
                ),
            )
            if regime.simulation.current_iid_draw_plan is not None:
                # Runtime-bound records, when declared, follow the replay tuple.
                if not isinstance(decision_outputs, tuple) or len(
                    decision_outputs
                ) != 4 + 2 * sum(
                    bound.kind == "dag"
                    for bound in regime.simulation.post_decision_bounds
                ):
                    raise TypeError(
                        "Current IID replay must publish physical draws "
                        "and mandatory validity."
                    )
                indices, values, physical, invalid, *bound_records = decision_outputs
                current_shocks = cast("Mapping[str, FloatND | IntND]", physical)
                invalid = cast("BoolND", invalid)
                if invalid.shape != in_regime.shape or invalid.dtype != jnp.bool_:
                    raise TypeError(
                        "Current IID replay validity must be Boolean per subject."
                    )
                if bool(
                    run_simulation_operation(
                        memory=memory,
                        function=_current_iid_invalid,
                        arguments={"invalid": invalid, "in_regime": in_regime},
                    )
                ):
                    raise ShockDeclarationError(
                        "Current IID law is invalid or unresolved "
                        "during physical replay. Correlated physical overrides "
                        "require a complete physical frame on active cells; "
                        "otherwise supply named base innovations."
                    )
                decision_outputs = (indices, values, *bound_records)
            indices, values = consume_runtime_bound_outputs(
                outputs=cast("tuple[QAndFArg, ...]", decision_outputs),
                bound_specs=regime.simulation.post_decision_bounds,
                regime_name=regime_name,
                period=period,
            )
            indices = cast("IntND", indices)
            values = cast("FloatND", values)
            # A scalar stateless decision lacks the leading subject axis that
            # every downstream simulation operation expects.
            if not space.states and indices.ndim == 0:
                indices, values = chunk_operations.broadcast_collective(
                    indices=indices,
                    value=values,
                    n_subjects=n_chunk_subjects,
                    memory=memory,
                )
            if accepted is not None:
                # A subject the policy read decided is never an empty grid row.
                indices = jnp.where(accepted, jnp.maximum(indices, 0), indices)
            first_empty = run_simulation_operation(
                memory=memory,
                function=_first_empty_action_row,
                arguments={"indices": indices, "in_regime": in_regime},
                subject_arg_names=("indices", "in_regime"),
            ).item()
            return indices, values, int(first_empty)

        # Policy-first: a plain DC-EGM or NB-EGM regime decides from its
        # published policy, and the dense grid argmax runs only if some
        # subject's read is rejected (off support, non-finite, outside the
        # budget, infeasible).
        # ponytail: one host sync per regime-period decides whether the grid
        # argmax runs at all; when it does, it covers the whole chunk and only
        # the rejected subjects take its decision. Gathering the rejected rows
        # would recompile per subset size.
        policy_first = None
        if _reads_egm_policy_first(
            regime=regime,
            sim_policy=sim_policy,
            action_names=state_action_space.action_names,
        ):
            active_in_regime = run_simulation_operation(
                memory=memory,
                function=_active_subjects,
                arguments={"in_regime": subject_ids_in_regime, "genuine": genuine},
                subject_arg_names=("in_regime", "genuine"),
                subject_outputs=True,
            )
            policy_first = _read_egm_policy_first(
                regime=regime,
                sim_policy=cast("EGMSimPolicy | NBEGMGridPolicy", sim_policy),
                states=states[regime_name],
                canonical_states=state_action_space.states,
                action_names=state_action_space.action_names,
                action_dtypes={
                    name: jnp.asarray(grid).dtype
                    for name, grid in state_action_space.actions.items()
                },
                next_regime_to_V_arr=next_regime_to_V_arr,
                flat_params=regime_kernel_params(flat_params, regime_name=regime_name),
                period=period,
                age=age,
                in_regime=active_in_regime,
                referenced_value_kwargs=referenced_value_kwargs,
                memory=memory,
            )
        policy_accepted = None if policy_first is None else policy_first[2]
        grid_owns_empty = sim_policy is None or policy_first is not None
        if policy_first is not None and not bool(
            jnp.any(active_in_regime & ~policy_first[2])
        ):
            optimal_actions, V_arr, _ = policy_first
            nested_fallback = None
        else:
            indices_optimal_actions, V_arr, empty_row = decide(
                space=state_action_space,
                in_regime=subject_ids_in_regime,
                accepted=policy_accepted,
            )
            if empty_row >= 0 and grid_owns_empty and entered_through is not None:
                (
                    states,
                    subject_regime_ids,
                    own_stakeholder,
                    new_subject_regime_ids,
                    new_own_stakeholder,
                    entered_through,
                    subject_ids_in_regime,
                    empty_row,
                    redecide,
                ) = _take_dissolution_routes(
                    regime_name=regime_name,
                    regime=regime,
                    regimes=regimes,
                    base_spaces=base_state_action_spaces,
                    ran_this_period=ran_this_period,
                    indices=indices_optimal_actions,
                    in_regime=subject_ids_in_regime,
                    entered_through=entered_through,
                    source_roles=cast("Int1D", source_roles),
                    states=states,
                    canonical_states=state_action_space.states,
                    subject_regime_ids=subject_regime_ids,
                    own_stakeholder=own_stakeholder,
                    new_subject_regime_ids=new_subject_regime_ids,
                    new_own_stakeholder=new_own_stakeholder,
                    regime_names_to_ids=regime_names_to_ids,
                    flat_params=flat_params,
                    period=period,
                    age=age,
                    subject_slice=subject_slice,
                    n_real=n_subjects
                    if original_n_subjects is None
                    else original_n_subjects,
                    logger=logger,
                    memory=memory,
                    records=dissolution_records,
                )
                # Rows whose route leads back into this regime decide again at
                # their projected state; their route is used up, so a row still
                # without a feasible action is refused below.
                if redecide and coded:
                    fail_if_a_held_process_code_is_off_node(
                        states=states[regime_name],
                        processes=coded,
                        holders=subject_ids_in_regime & genuine,
                        where=f"projected by a dissolution route back into regime "
                        f"'{regime_name}' at period {period}",
                        memory=memory,
                    )
                if redecide:
                    state_action_space = create_regime_state_action_space(
                        regime=regime,
                        regime_states=states[regime_name],
                        base=base_state_action_space,
                    )
                    indices_optimal_actions, V_arr, empty_row = decide(
                        space=state_action_space,
                        in_regime=subject_ids_in_regime,
                        accepted=policy_accepted,
                    )
            if empty_row >= 0 and grid_owns_empty:
                state = physical_state_at_row(
                    regime=regime,
                    states=state_action_space.states,
                    row=empty_row,
                    regime_params={
                        **regime.resolved_fixed_params,
                        **regime_kernel_params(flat_params, regime_name=regime_name),
                    },
                    memory=memory,
                )
                raise NoFeasibleActionError(
                    f"Regime '{regime_name}', period {period}, state {state} "
                    "has no feasible action."
                )
            optimal_actions = run_simulation_operation(
                memory=memory,
                function=_lookup_values_from_indices,
                arguments={
                    "flat_indices": indices_optimal_actions,
                    "grids": state_action_space.actions,
                },
                subject_arg_names=("flat_indices",)
                if indices_optimal_actions.ndim
                else (),
            )
            if policy_first is None:
                optimal_actions, V_arr, nested_fallback = (
                    _replace_continuous_action_with_policy_read(
                        optimal_actions=optimal_actions,
                        regime=regime,
                        sim_policy=sim_policy,
                        states=states[regime_name],
                        flat_params=regime_kernel_params(
                            flat_params, regime_name=regime_name
                        ),
                        period=period,
                        age=age,
                        canonical_states=state_action_space.states,
                        action_names=state_action_space.action_names,
                        next_regime_to_V_arr=next_regime_to_V_arr,
                        referenced_value_kwargs=referenced_value_kwargs,
                        grid_values=V_arr,
                    )
                )
            else:
                policy_actions, policy_value, accepted = policy_first
                optimal_actions = MappingProxyType(
                    {
                        name: jnp.where(
                            accepted,
                            policy_actions[name],
                            jnp.asarray(optimal_actions[name]),
                        )
                        for name in optimal_actions
                    }
                )
                V_arr = jnp.where(accepted, policy_value, V_arr)
                nested_fallback = None

    # Store results for this regime-period
    # For state-less regimes (e.g., terminal regimes with no states), V_arr may be a
    # scalar. We need to broadcast it to match this chunk's subject count.
    if V_arr.ndim == 0:
        V_arr = chunk_operations.broadcast_value(
            value=V_arr, n_subjects=n_chunk_subjects, memory=memory
        )

    # `None` from the reader means no nested continuous-outer read ran for
    # this regime-period (no payload, the flat single-EGM path, passive rows,
    # the discrete-branch redecide), so no subject fell back on that path.
    nested_policy_fallback = (
        chunk_operations.empty_fallback(mask=subject_ids_in_regime, memory=memory)
        if nested_fallback is None
        else nested_fallback
    )

    simulation_result = PeriodRegimeSimulationData(
        shocks=current_shocks
        if regime.simulation.current_iid_draw_plan is not None
        else MappingProxyType({}),
        V_arr=V_arr,
        actions=optimal_actions,
        states=states[regime_name],
        in_regime=subject_ids_in_regime,
        own_stakeholder=own_stakeholder,
        nested_policy_fallback=nested_policy_fallback,
    )

    # Update states and regime membership for next period
    if not regime.terminal:
        next_states_key = next_regime_key = key
        if memory is not None:
            memory.hold(tree=(simulation_result, next_states_key, next_regime_key, key))

        next_states = calculate_next_states(
            current_shocks=simulation_result.shocks,
            regime=regime,
            regimes=regimes,
            process_nodes=process_node_arguments(
                regime=regime,
                regimes=regimes,
                base_spaces=base_state_action_spaces,
            ),
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=regime_kernel_params(flat_params, regime_name=regime_name),
            states_per_regime=states,
            state_action_space=state_action_space,
            key=next_states_key,
            subjects_in_regime=subject_ids_in_regime,
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            supplied_shocks=supplied_shocks,
            original_n_subjects=original_n_subjects,
            memory=memory,
        )
        if memory is not None:
            memory.hold(tree=next_states)
        # The realized regime draw reads current-period carried values, so it
        # runs against the pre-advance carrier; only then do the next-period
        # states replace it.
        new_subject_regime_ids = calculate_next_regime_membership(
            current_shocks=simulation_result.shocks,
            regime=regime,
            state_action_space=state_action_space,
            optimal_actions=optimal_actions,
            period=period,
            age=age,
            regime_params=regime_kernel_params(flat_params, regime_name=regime_name),
            regime_names_to_ids=regime_names_to_ids,
            states_per_regime=states,
            new_subject_regime_ids=new_subject_regime_ids,
            active_regimes_next_period=realization_targets_next_period,
            key=next_regime_key,
            subjects_in_regime=subject_ids_in_regime,
            n_subjects=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
            memory=memory,
            runtime_checks=validation_enabled(logger),
        )
        drawn_arrival_ids = new_subject_regime_ids
        # A subject's raw codes are checked against the regime it was drawn to
        # before a gate can read a value at them or reroute the subject.
        if validation_enabled(logger):
            fail_if_drawn_codes_leave_their_grids(
                next_states=next_states,
                regimes=regimes,
                targets=_drawn_code_targets(
                    targets=realization_targets_next_period,
                    regime_names_to_ids=regime_names_to_ids,
                ),
                source=regime_name,
                period=period,
                new_regime_ids=new_subject_regime_ids,
                regime_ids=regime_names_to_ids,
                subjects=subject_ids_in_regime
                & _genuine_rows(
                    subject_slice=subject_slice,
                    n_real=n_subjects
                    if original_n_subjects is None
                    else original_n_subjects,
                ),
                memory=memory,
            )
        # A gate reads a target's node positions at the drawn states.
        for target in realization_targets_next_period:
            if process_code_reads.get(target):
                fail_if_a_held_process_code_is_off_node(
                    states=next_states.get(target, {}),
                    processes=process_code_reads[target],
                    holders=subject_ids_in_regime
                    & genuine
                    & (new_subject_regime_ids == regime_names_to_ids[target]),
                    where=f"drawn from regime '{regime_name}' into regime "
                    f"'{target}' at period {period}",
                    memory=memory,
                )
        # The value router's routing half. A gated
        # edge's target is always ALSO an ordinary declared transition
        # target, so `calculate_next_states` already computed candidate
        # target states above and `calculate_next_regime_membership` already
        # drew a (gate-blind) next regime id; `route_gated_edges` now
        # RECOMPUTES the gate at those candidate states
        # and OVERRIDES both — the target when open, a leg's fallback (with
        # its own projected states) when closed — for every subject in this
        # regime. No-op for a regime without gated edges.
        if period in regime.simulation.programs.gate_route:
            route_values = acquire_gate_inputs(
                reads=tuple(
                    read
                    for read in raw_gate_reads
                    if read.source.core_key == GATE_ROUTE
                ),
                owner=value_owner,
                name=regime_name,
                values=period_to_regime_to_V_arr,
            )
            route_delta, new_subject_regime_ids, new_own_stakeholder = cast(
                "tuple[Mapping[RegimeName, GatedRouteDelta], Int1D, Int1D]",
                execute_simulation_program(
                    programs=regime.simulation.programs,
                    family="gate_route",
                    period=period,
                    arguments=gate_route_arguments(
                        edge_values=route_values,
                        next_states=gated_route_candidates(
                            regime=regime, next_states=next_states
                        ),
                        new_subject_regime_ids=new_subject_regime_ids,
                        subjects_in_regime=subject_ids_in_regime,
                        flat_params=flat_params,
                        own_stakeholder=own_stakeholder,
                        new_own_stakeholder=new_own_stakeholder,
                        fold_age=jnp.asarray(
                            0.0 if gated_edge_fold_age is None else gated_edge_fold_age
                        ),
                    ),
                    n_subjects=n_subjects,
                ),
            )
            next_states = commit_gated_route_delta(
                delta=route_delta,
                next_states=next_states,
            )
            if new_entered_through is not None:
                new_entered_through = run_simulation_operation(
                    memory=memory,
                    function=record_open_entries,
                    arguments={
                        "closed_masks": tuple(
                            cast("Bool1D", edge_delta["closed"])
                            for edge_delta in route_delta.values()
                        ),
                        "routed_ids": new_subject_regime_ids,
                        "subjects_in_regime": subject_ids_in_regime,
                        "new_entered_through": new_entered_through,
                    },
                    subject_arg_names=(
                        "closed_masks",
                        "routed_ids",
                        "subjects_in_regime",
                        "new_entered_through",
                    ),
                    static_arguments={
                        "source_id": int(regime_names_to_ids[regime_name]),
                        "target_ids": tuple(
                            int(regime_names_to_ids[target]) for target in route_delta
                        ),
                    },
                    subject_outputs=True,
                )
        participation_targets = tuple(
            target
            for target in realization_targets_next_period
            if period + 1 in regimes[target].simulation.exit_queries
            and regime_name
            in regimes[target].simulation.exit_queries[period + 1].participation
        )
        if participation_targets and new_entered_through is not None:
            new_entered_through = run_simulation_operation(
                memory=memory,
                function=record_participation_entries,
                arguments={
                    "drawn_ids": drawn_arrival_ids,
                    "routed_ids": new_subject_regime_ids,
                    "holders": subject_ids_in_regime,
                    "entries": new_entered_through,
                },
                static_arguments={
                    "source_id": int(regime_names_to_ids[regime_name]),
                    "target_ids": tuple(
                        int(regime_names_to_ids[target])
                        for target in participation_targets
                    ),
                },
                subject_arg_names=("drawn_ids", "routed_ids", "holders", "entries"),
                subject_outputs=True,
            )
        states = clip_next_states(
            states=next_states,
            regimes=regimes,
            base_spaces=base_state_action_spaces,
            targets=tuple(
                dict.fromkeys(
                    (
                        *realization_targets_next_period,
                        *(
                            leg.realized_fallback.regime
                            for edge in regime.gated_edges.values()
                            for leg in edge.legs
                        ),
                    )
                )
            ),
            next_period=period + 1,
            source=regime_name,
            new_regime_ids=new_subject_regime_ids,
            regime_ids=regime_names_to_ids,
            logger=logger,
            subjects=subject_ids_in_regime,
            genuine=_genuine_rows(
                subject_slice=subject_slice,
                n_real=n_subjects
                if original_n_subjects is None
                else original_n_subjects,
            ),
            memory=memory,
        )

    return (
        simulation_result,
        states,
        new_subject_regime_ids,
        new_own_stakeholder,
        subject_regime_ids,
        own_stakeholder,
        entered_through,
        new_entered_through,
    )


def _take_dissolution_routes(
    *,
    regime_name: RegimeName,
    regime: Regime,
    regimes: Mapping[RegimeName, Regime],
    base_spaces: Mapping[RegimeName, StateActionSpace],
    ran_this_period: tuple[RegimeName, ...],
    indices: IntND,
    in_regime: Bool1D,
    entered_through: Int1D,
    source_roles: Int1D,
    states: StatesPerRegime,
    canonical_states: Mapping[StateName, FloatND | IntND],
    subject_regime_ids: Int1D,
    own_stakeholder: Int1D,
    new_subject_regime_ids: Int1D,
    new_own_stakeholder: Int1D,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    period: int,
    age: ScalarInt | ScalarFloat,
    subject_slice: slice | SubjectRows,
    n_real: int,
    logger: logging.Logger,
    memory: SimulationMemory | None,
    records: list[DissolutionFallback] | None,
) -> tuple[StatesPerRegime, Int1D, Int1D, Int1D, Int1D, Int1D, Bool1D, int, bool]:
    """Send this regime's rows without a feasible action down their gate's route.

    A route back into this regime keeps the row here at its projected state.
    A projected state follows its fallback grid's `off_grid` rule, as a closed
    branch taken at the gate does. A route into a regime that already ran this
    period raises `NoFeasibleActionError`, since the row could not act there.
    Returns the updated carriers, the open-branch entries with every route taken
    here used up, this regime's remaining rows, whether any of them must decide
    again at a projected state, and the first
    row still without a feasible action (`-1` when none is left), which the
    caller refuses as before.
    """
    empty = in_regime & (indices < 0)
    outcome = reroute_rows_without_feasible_action(
        regime_name=regime_name,
        regimes=regimes,
        empty=empty,
        entered_through=entered_through,
        source_roles=source_roles,
        states=states,
        subject_regime_ids=subject_regime_ids,
        own_stakeholder=own_stakeholder,
        new_subject_regime_ids=new_subject_regime_ids,
        new_own_stakeholder=new_own_stakeholder,
        regime_names_to_ids=regime_names_to_ids,
        flat_params=flat_params,
        period=period,
        age=age,
    )
    if memory is not None:
        memory.hold(tree=outcome)
    genuine = _genuine_rows(subject_slice=subject_slice, n_real=n_real)
    _fail_if_rerouted_into_a_regime_that_ran(
        outcome=outcome,
        genuine=genuine,
        ran_this_period=ran_this_period,
        regime_name=regime_name,
        regime=regime,
        canonical_states=canonical_states,
        flat_params=flat_params,
        regime_names_to_ids=regime_names_to_ids,
        period=period,
        memory=memory,
    )
    states = clip_next_states(
        states=outcome.states,
        regimes=regimes,
        base_spaces=base_spaces,
        targets=tuple(
            dict.fromkeys(
                leg.realized_fallback.regime
                for source in regimes.values()
                if (edge := source.gated_edges.get(regime_name)) is not None
                for leg in edge.legs
            )
        ),
        next_period=period,
        reported_period=period,
        source=regime_name,
        new_regime_ids=outcome.subject_regime_ids,
        regime_ids=regime_names_to_ids,
        logger=logger,
        subjects=outcome.rerouted,
        genuine=genuine,
        memory=memory,
    )
    returned = outcome.rerouted & (
        outcome.fallback_ids == regime_names_to_ids[regime_name]
    )
    remaining = in_regime & ~(outcome.rerouted & ~returned)
    redecide = bool(np.asarray(returned).any())
    rerouted_rows = rows_of(outcome.rerouted & genuine)
    if rerouted_rows.size:
        new_records: tuple[DissolutionFallback, ...] = ()
        if validation_enabled(logger):
            regime_params = {
                **regime.resolved_fixed_params,
                **regime_kernel_params(flat_params, regime_name=regime_name),
            }
            ids_to_names = {
                int(code): name for name, code in regime_names_to_ids.items()
            }
            sources = np.asarray(entered_through)
            fallbacks = np.asarray(outcome.fallback_ids)
            new_records = tuple(
                DissolutionFallback(
                    subject_id=int(
                        subject_slice.rows[row]
                        if isinstance(subject_slice, SubjectRows)
                        else cast("int", subject_slice.start) + row
                    ),
                    period=period,
                    regime=regime_name,
                    state=MappingProxyType(
                        physical_state_at_row(
                            regime=regime,
                            states=canonical_states,
                            row=int(row),
                            regime_params=regime_params,
                            memory=memory,
                        )
                    ),
                    source=ids_to_names[int(sources[row])],
                    fallback=ids_to_names[int(fallbacks[row])],
                )
                for row in rerouted_rows
            )
            if records is not None:
                records.extend(new_records)
        log_dissolution_fallbacks(
            logger=logger,
            regime_name=regime_name,
            period=period,
            n_rows=int(rerouted_rows.size),
            recorded=records is not None and validation_enabled(logger),
        )
    return (
        states,
        outcome.subject_regime_ids,
        outcome.own_stakeholder,
        outcome.new_subject_regime_ids,
        outcome.new_own_stakeholder,
        outcome.entered_through,
        remaining,
        int(
            _first_empty_action_row(
                indices=indices, in_regime=in_regime & ~outcome.rerouted
            )
        ),
        redecide,
    )


def _fail_if_rerouted_into_a_regime_that_ran(
    *,
    outcome: RerouteOutcome,
    genuine: np.ndarray,
    ran_this_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    regime: Regime,
    canonical_states: Mapping[StateName, FloatND | IntND],
    flat_params: FlatParams,
    regime_names_to_ids: RegimeNamesToIds,
    period: int,
    memory: SimulationMemory | None,
) -> None:
    """Raise if a rerouted row's fallback already ran this period.

    Regimes run in dependency order, each gated target before its fallbacks.
    When gated targets and fallbacks form a cycle, no regime may be ready; the
    earliest-declared regime still waiting then runs next, so a fallback can
    precede a target it serves. A row routed into such a fallback would not act
    this period.
    """
    if not ran_this_period:
        return
    ran_ids = np.asarray([regime_names_to_ids[name] for name in ran_this_period])
    late = np.asarray(outcome.rerouted) & genuine
    late &= np.isin(np.asarray(outcome.fallback_ids), ran_ids)
    if not late.any():
        return
    row = int(np.flatnonzero(late)[0])
    fallback = next(
        name
        for name, code in regime_names_to_ids.items()
        if int(code) == int(np.asarray(outcome.fallback_ids)[row])
    )
    state = physical_state_at_row(
        regime=regime,
        states=canonical_states,
        row=row,
        regime_params={
            **regime.resolved_fixed_params,
            **regime_kernel_params(flat_params, regime_name=regime_name),
        },
        memory=memory,
    )
    raise NoFeasibleActionError(
        f"Regime '{regime_name}', period {period}, state {state} has no feasible "
        f"action, and its dissolution route leads to regime '{fallback}', which "
        "already ran this period: the gated targets and fallbacks form a cycle."
    )


def _fail_if_a_finite_bank_state_is_off_grid(
    *,
    regime_name: RegimeName,
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    subjects: BoolND,
    period: int,
    memory: SimulationMemory | None,
) -> None:
    """Refuse a subject whose continuous state lies outside the finite bank's grid.

    The finite NNBEGM bank holds the solve's candidates on the regime's own state
    grid. Off that grid a linear read ranks the candidates wrongly: a candidate
    infeasible at the grid's edge, such as an adjustment that only becomes
    affordable at higher wealth, cannot be extrapolated into view. Process
    states are exempt; their draws are read along the node coordinate. One host
    sync per regime-period decides whether any subject is off its grid.
    """
    checked = {
        name: jnp.asarray(grid.to_jax())
        for name in sim_policy.state_names
        if isinstance(grid := regime.simulation.grids[name], ContinuousGrid)
        and not isinstance(grid, _ContinuousStochasticProcess)
    }
    counts = [
        run_simulation_operation(
            memory=memory,
            function=count_off_grid,
            arguments={"values": states[name], "points": points, "feasible": subjects},
            subject_arg_names=("values", "feasible"),
        )
        for name, points in checked.items()
    ]
    if not counts or not int(functools.reduce(jnp.add, counts)):
        return
    subject_mask = np.asarray(subjects)
    found, off_names = [], []
    for name, points in checked.items():
        values = np.asarray(states[name])[subject_mask]
        low, high = float(points[0]), float(points[-1])
        off = values[(values < low) | (values > high)]
        if off.size:
            off_names.append(f"'{name}'")
            found.append(
                f"{off.size} subject(s) hold '{name}' in "
                f"[{off.min():.6g}, {off.max():.6g}], outside its grid "
                f"[{low:.6g}, {high:.6g}]"
            )
    names = ", ".join(off_names)
    raise OffGridStateError(
        f"Regime '{regime_name}', period {period}: " + "; ".join(found) + ". "
        f"The regime is solved by NNBEGM over a FiniteOuterGrid, whose simulation "
        "reads the solve's candidates on the regime's own state grid and does not "
        f"extrapolate beyond it. Widen the grid of {names} in regime "
        f"'{regime_name}' to cover these values. A start must lie inside it, and so "
        "must every state a predecessor regime's choices land on: a GridSearch or "
        "NBEGM predecessor can choose actions whose landings lie outside it."
    )


def _execute_finite_replay(
    *,
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    canonical_states: Mapping[StateName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    referenced_value_kwargs: QAndFKwargs,
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND, BoolND]:
    """Run the declared reconstruction and ranking programs."""
    payload = ReplayPayload.from_policy(sim_policy)
    bank = execute_simulation_program(
        programs=regime.simulation.programs,
        family="policy_prepare",
        period=period,
        n_subjects=n_subjects,
        arguments=policy_prepare_arguments(
            payload=payload,
            states=states,
            params=flat_params,
            age=age,
        ),
    )
    bank = cast("tuple[FloatND, FloatND, FloatND, BoolND]", bank)
    ranked = execute_simulation_program(
        programs=regime.simulation.programs,
        family="policy_rank",
        period=period,
        n_subjects=n_subjects,
        arguments=policy_rank_arguments(
            payload=payload,
            bank=bank,
            canonical_states=canonical_states,
            params=flat_params,
            age=age,
            next_values=next_regime_to_V_arr,
            references=referenced_value_kwargs,
        ),
    )
    return cast(
        "tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND, BoolND]",
        consume_runtime_bound_outputs(
            outputs=cast("tuple[QAndFArg, ...]", ranked),
            bound_specs=regime.simulation.post_decision_bounds,
            regime_name=regime.name,
            period=period,
        ),
    )


def _replace_continuous_action_with_policy_read(
    *,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: (
        EGMSimPolicy | NBEGMGridPolicy | NNBEGMSimPolicy | NestedEGMSimPolicy | None
    ),
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    grid_values: FloatND,
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND, BoolND | None]:
    """Apply a nested or finite NNBEGM policy read to the grid-argmax decision.

    A flat `EGMSimPolicy` is read policy-first before the grid argmax runs
    (`_read_egm_policy_first`); a flat payload that reaches this point (passive
    row axes, or actions its rows do not carry) keeps the grid pair, as does
    an `NBEGMGridPolicy`.

    Returns three things, and the second two are independent of each other:

    - the recovered actions;
    - the value to report for the emitted pair. Every path that keeps the grid
      pair returns `grid_values` unchanged; every accepted off-grid pair is
      scored by the canonical Q and returns that attained value. A rejected
      nested read keeps the grid pair and its grid value;
    - a nested-fallback flag: the per-subject Boolean array from
      `_read_nested_policy`, extended to include canonical infeasibility and
      non-finite scores; or `None` on every other path. The caller resolves
      `None` to all-False where the regime's subject count is known.
    """
    if sim_policy is None:
        return optimal_actions, grid_values, None
    if isinstance(sim_policy, NestedEGMSimPolicy):
        # The nested payload names the actions, liquid state, and search settings,
        # so this reader does not consult fields of `egm_policy_read`. The
        # model/result boundary still requires that marker to declare the cell's
        # consuming route before a labelled replay artifact reaches this point.
        nested_actions, nested_fallback = _read_nested_policy(
            payload=sim_policy,
            optimal_actions=optimal_actions,
            regime=regime,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
        )
        # The read's winner is emitted; one canonical-Q evaluation reports its
        # value and its user feasibility. A read that leaves support or budget,
        # or is infeasible, keeps the grid pair and the grid value.
        nested_value, nested_feasible = _canonical_Q_at_actions(
            candidate_actions=nested_actions,
            regime=regime,
            canonical_states=canonical_states,
            action_names=action_names,
            next_regime_to_V_arr=next_regime_to_V_arr,
            flat_params=flat_params,
            referenced_value_kwargs=referenced_value_kwargs,
            period=period,
            age=age,
        )
        accepted = ~nested_fallback & nested_feasible & jnp.isfinite(nested_value)
        emitted_actions = MappingProxyType(
            {
                name: jnp.where(
                    accepted,
                    jnp.asarray(nested_actions[name]),
                    jnp.asarray(optimal_actions[name]),
                )
                for name in optimal_actions
            }
        )
        return (
            emitted_actions,
            jnp.where(accepted, nested_value, grid_values),
            ~accepted,
        )
    route = regime.simulation.replay_route
    if route.payload_type is None:
        return optimal_actions, grid_values, None
    if route.consumer_route == "nnbegm_finite":
        replayed_actions, replayed_value = _replay_nnbegm_candidates(
            optimal_actions=optimal_actions,
            regime=regime,
            sim_policy=cast("NNBEGMSimPolicy", sim_policy),
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
            canonical_states=canonical_states,
            action_names=action_names,
            next_regime_to_V_arr=next_regime_to_V_arr,
            referenced_value_kwargs=referenced_value_kwargs,
        )
        return replayed_actions, replayed_value, None
    return optimal_actions, grid_values, None


def _reads_egm_policy_first(
    *,
    regime: Regime,
    sim_policy: (
        EGMSimPolicy | NBEGMGridPolicy | NNBEGMSimPolicy | NestedEGMSimPolicy | None
    ),
    action_names: tuple[ActionName, ...],
) -> bool:
    """Whether this regime-period decides from the flat EGM or NB-EGM policy.

    The policy must carry every action the regime declares (the EGM continuous
    action plus the discrete actions). Flat EGM rows must carry no passive
    continuous-state axis: each passive row is the envelope policy conditional
    on one passive node, so blending across a passive-dimension branch switch
    would read neither branch.
    """
    read = regime.simulation.egm_policy_read
    if regime.has_taste_shocks or regime.simulation.current_iid_draw_plan is not None:
        return False
    if isinstance(sim_policy, NBEGMGridPolicy):
        return (
            isinstance(read, NBEGMPolicyRead)
            and regime.simulation.replay_route.consumer_route == "nbegm_grid"
            and set(action_names) == {read.action_name, *read.discrete_action_names}
        )
    return (
        isinstance(sim_policy, EGMSimPolicy)
        and isinstance(read, EGMPolicyRead)
        and regime.simulation.replay_route.consumer_route == "egm_off_grid"
        and not sim_policy.row_passive_state_names
        and set(action_names)
        == {read.action_name, *sim_policy.row_discrete_action_names}
    )


def _read_egm_policy_first(
    *,
    regime: Regime,
    sim_policy: EGMSimPolicy | NBEGMGridPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    action_dtypes: Mapping[ActionName, jnp.dtype],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    in_regime: BoolND,
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
    memory: SimulationMemory | None = None,
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND, BoolND]:
    """Decide from the published EGM rows instead of the action-grid argmax.

    Each discrete-action branch's row is the solve-phase optimum conditional on
    that branch. Its conditional value (cubic Hermite with the
    marginal-utility slopes, the solve's convention) is read at the branch's
    own resources; discrete-only constraints have already set infeasible
    branch rows to `-inf`. The branch of largest read value wins, its policy
    row (piecewise linear) supplies the continuous action, and its read value
    is the reported value, so value, actions and transitions all belong to the
    same (branch, action) pair. A process state between its nodes reads each
    branch from the blend of its two bracketing node rows (`_row_selections`).
    An NB-EGM regime reads its state-grid policy instead
    (`_read_nbegm_grid_policy`) and reports the canonical value of the emitted
    actions.

    Returns the actions, the reported value, and the per-subject acceptance.
    A subject is accepted when every live branch read lies inside its row
    support (otherwise the comparison is incomplete) and the winner's value
    and action are finite, the action is positive, within the intrinsic budget
    (`action <= resources - savings_lower_bound`, up to rounding, to which the
    emitted action is clipped) and canonically feasible.
    The caller replaces every rejected subject's decision by the grid argmax.

    The read runs as one compiled program, not operation by operation. The
    period is traced, so the periods that share one `Q_and_F` kernel and the
    same input shapes share that program.
    """
    dtypes = tuple((name, jnp.dtype(action_dtypes[name]).name) for name in action_names)
    operands = {
        "sim_policy": sim_policy,
        "states": MappingProxyType(dict(states)),
        "canonical_states": MappingProxyType(dict(canonical_states)),
        "next_regime_to_V_arr": next_regime_to_V_arr,
        "flat_params": flat_params,
        "period": jnp.asarray(period, dtype=jnp.int32),
        "age": age,
        "in_regime": in_regime,
        "referenced_value_kwargs": MappingProxyType(dict(referenced_value_kwargs)),
    }
    if memory is None:
        output = _policy_first_program(
            held_regime=_HeldObject(regime),
            held_Q_and_F=_HeldObject(regime.simulation.Q_and_F[period]),
            action_names=action_names,
            action_dtypes=dtypes,
            **operands,
        )
    else:
        # Budgeted: admitted against current residency like every producer.
        # Chunk admission profiles the same producer for each candidate width.
        output = run_simulation_producer(
            memory=memory,
            function=policy_first_operation(
                regime=regime,
                period=period,
                action_names=action_names,
                action_dtypes=dtypes,
            ),
            arguments=operands,
            owner=memory.operations,
        )
    actions, winner_value, read_ok, evaluated = cast("_PolicyFirstOutput", output)
    _, feasible = _validated_canonical_Q(
        evaluated=evaluated, regime=regime, period=period
    )
    return actions, winner_value, read_ok & feasible


# Actions, winner value, read-side acceptance and the unvalidated canonical
# evaluation of the emitted actions.
type _PolicyFirstOutput = tuple[
    MappingProxyType[ActionName, FloatND | IntND],
    FloatND,
    BoolND,
    tuple[FloatND, BoolND] | BoundedResult[tuple[FloatND, BoolND]],
]


@functools.partial(
    jax.jit,
    static_argnames=("held_regime", "held_Q_and_F", "action_names", "action_dtypes"),
)
def _policy_first_program(
    *,
    held_regime: _HeldObject,
    held_Q_and_F: _HeldObject,
    sim_policy: EGMSimPolicy | NBEGMGridPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    action_dtypes: tuple[tuple[ActionName, str], ...],
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    period: IntND,
    age: ScalarFloat | ScalarInt,
    in_regime: BoolND,
    referenced_value_kwargs: QAndFKwargs,
) -> _PolicyFirstOutput:
    """Unbudgeted `_policy_first_body`, jitted per regime and `Q_and_F` kernel."""
    return _policy_first_body(
        regime=cast("Regime", held_regime.value),
        Q_and_F=cast("QAndFFunction", held_Q_and_F.value),
        sim_policy=sim_policy,
        states=states,
        canonical_states=canonical_states,
        action_names=action_names,
        action_dtypes=action_dtypes,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=flat_params,
        period=period,
        age=age,
        in_regime=in_regime,
        referenced_value_kwargs=referenced_value_kwargs,
    )


def policy_first_operation(
    *,
    regime: Regime,
    period: int,
    action_names: tuple[ActionName, ...],
    action_dtypes: tuple[tuple[ActionName, str], ...],
) -> Callable[..., _PolicyFirstOutput]:
    """Bind the policy-first read to its model-owned inputs, as one producer.

    Equal bindings resolve to one producer identity, so the runtime's
    operations reuse one executable per regime, kernel and operand shape.
    """
    return functools.partial(
        _policy_first_body,
        regime=regime,
        Q_and_F=cast("QAndFFunction", regime.simulation.Q_and_F[period]),
        action_names=action_names,
        action_dtypes=action_dtypes,
    )


def _policy_first_body(
    *,
    regime: Regime,
    Q_and_F: QAndFFunction,
    sim_policy: EGMSimPolicy | NBEGMGridPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    action_dtypes: tuple[tuple[ActionName, str], ...],
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    period: IntND,
    age: ScalarFloat | ScalarInt,
    in_regime: BoolND,
    referenced_value_kwargs: QAndFKwargs,
) -> _PolicyFirstOutput:
    """Body of `_read_egm_policy_first`, before canonical feasibility.

    Returns the actions, the winner's value, the read-side acceptance and the
    unvalidated canonical evaluation of the emitted actions. An NB-EGM policy
    publishes no value at the realized state, so its winner's value is the
    canonical value of the emitted actions.
    """
    if isinstance(sim_policy, NBEGMGridPolicy):
        actions, winner_value, read_ok = _read_nbegm_grid_policy(
            regime=regime,
            sim_policy=sim_policy,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
        )
    else:
        actions, winner_value, read_ok = _read_egm_policy_rows(
            regime=regime,
            sim_policy=sim_policy,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
        )
    dtypes = dict(action_dtypes)
    actions = {name: actions[name].astype(dtypes[name]) for name in action_names}
    evaluated = _evaluate_canonical_Q_at_actions(
        candidate_actions=actions,
        regime=regime,
        canonical_states=canonical_states,
        action_names=action_names,
        next_regime_to_V_arr=MappingProxyType(dict(next_regime_to_V_arr)),
        flat_params=flat_params,
        referenced_value_kwargs=referenced_value_kwargs,
        period=period,
        age=age,
        active_subjects=in_regime,
        Q_and_F=Q_and_F,
    )
    if winner_value is None:
        winner_value, _ = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        read_ok &= jnp.isfinite(winner_value)
    return MappingProxyType(actions), winner_value, read_ok, evaluated


def _read_egm_policy_rows(
    *,
    regime: Regime,
    sim_policy: EGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: IntND,
    age: ScalarFloat | ScalarInt,
) -> tuple[dict[ActionName, FloatND | IntND], FloatND, BoolND]:
    """Read the flat EGM rows: the winning branch's actions, value and acceptance."""
    read = cast("EGMPolicyRead", regime.simulation.egm_policy_read)
    n_subjects = next(iter(states.values())).shape[0]
    selections = _row_selections(
        regime=regime, names=sim_policy.row_discrete_state_names, states=states
    )
    branch_names = sim_policy.row_discrete_action_names
    branch_grids = tuple(
        jnp.asarray(regime.simulation.grids[name].to_jax()) for name in branch_names
    )
    branch_shape = tuple(int(grid.shape[0]) for grid in branch_grids)
    values, policies, resources, in_support = [], [], [], []
    for branch in itertools.product(*(range(size) for size in branch_shape)):
        data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = {
            **dict(states),
            **{
                name: jnp.full((n_subjects,), grid[position], dtype=grid.dtype)
                for name, grid, position in zip(
                    branch_names, branch_grids, branch, strict=True
                )
            },
            "period": jnp.full(n_subjects, period, dtype=jnp.int32),
            "age": jnp.full(n_subjects, age),
        }
        branch_resources = jnp.reshape(
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
        value, support = _blended_row_read(
            sim_policy=sim_policy,
            field="value",
            selections=selections,
            branch=branch,
            resources=branch_resources,
            n_subjects=n_subjects,
        )
        policy, _ = _blended_row_read(
            sim_policy=sim_policy,
            field="policy",
            selections=selections,
            branch=branch,
            resources=branch_resources,
            n_subjects=n_subjects,
        )
        values.append(value)
        policies.append(policy)
        resources.append(branch_resources)
        in_support.append(support)

    stacked_values = jnp.stack(values)
    # A dead (constraint-masked) row reads `-inf`, a row without read support
    # reads NaN; neither competes, and only a live branch must be in support.
    live = jnp.isfinite(stacked_values)
    ranked = jnp.where(live, stacked_values, -jnp.inf)
    winner = jnp.argmax(ranked, axis=0).astype(jnp.int32)
    winner_value = _take_at_winner(stacked=ranked, winner=winner)
    action = _take_at_winner(stacked=jnp.stack(policies), winner=winner)
    winner_resources = _take_at_winner(stacked=jnp.stack(resources), winner=winner)
    budget = winner_resources - read.savings_lower_bound
    # A read on the constrained segment (`action = resources - floor`) can exceed
    # the budget by rounding only; that read is the budget itself.
    rounding = 8 * jnp.finfo(action.dtype).eps * jnp.maximum(1.0, jnp.abs(budget))
    within_budget = action <= budget + rounding
    action = jnp.minimum(action, budget)
    winner_positions = jnp.unravel_index(winner, branch_shape)
    actions = {
        name: grid[positions]
        for name, grid, positions in zip(
            branch_names, branch_grids, winner_positions, strict=True
        )
    }
    actions[read.action_name] = action
    read_ok = (
        jnp.all(~live | jnp.stack(in_support), axis=0)
        & jnp.isfinite(winner_value)
        & jnp.isfinite(action)
        & (action > 0.0)
        & within_budget
    )
    return actions, winner_value, read_ok


def _read_nbegm_grid_policy(
    *,
    regime: Regime,
    sim_policy: NBEGMGridPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: IntND,
    age: ScalarFloat | ScalarInt,
) -> tuple[dict[ActionName, FloatND | IntND], None, BoolND]:
    """Read the published NB-EGM state-grid policy at each subject's state.

    The liquid state reads linearly between its two bracketing grid nodes. Every
    other state axis selects its rows as `_row_selections` does, so a process
    state between its nodes blends its two bracketing node rows. With discrete
    actions, the branch whose conditional value reads largest wins and its
    conditional policy supplies the continuous action; a smooth solve publishes
    the collapsed policy alone.

    Returns the actions, no read value, and the read-side acceptance: the liquid
    state lies inside its grid, a branch is live, and the action is finite,
    positive and within the intrinsic budget
    (`action <= resources - savings_lower_bound`, up to rounding, to which the
    emitted action is clipped).
    """
    read = cast("NBEGMPolicyRead", regime.simulation.egm_policy_read)
    n_subjects = next(iter(states.values())).shape[0]
    liquid = jnp.asarray(states[read.liquid_state_name])
    nodes = jnp.asarray(regime.simulation.grids[read.liquid_state_name].to_jax())
    lo, weight = _node_bracket(nodes=nodes, values=liquid)
    surface_read = functools.partial(
        _state_grid_read,
        liquid_axis=sim_policy.state_names.index(read.liquid_state_name),
        selections=_row_selections(
            regime=regime,
            names=tuple(
                name
                for name in sim_policy.state_names
                if name != read.liquid_state_name
            ),
            states=states,
        ),
        lo=lo,
        weight=weight,
    )
    actions: dict[ActionName, FloatND | IntND] = {}
    live = jnp.ones((n_subjects,), dtype=bool)
    if sim_policy.branch_value is None or sim_policy.branch_inner_action is None:
        action = surface_read(surface=sim_policy.action)
    else:
        values = jax.vmap(lambda surface: surface_read(surface=surface))(
            sim_policy.branch_value
        )
        ranked = jnp.where(jnp.isfinite(values), values, -jnp.inf)
        winner = jnp.argmax(ranked, axis=0).astype(jnp.int32)
        live = jnp.isfinite(_take_at_winner(stacked=ranked, winner=winner))
        action = _take_at_winner(
            stacked=jax.vmap(lambda surface: surface_read(surface=surface))(
                sim_policy.branch_inner_action
            ),
            winner=winner,
        )
        codes = jnp.asarray(sim_policy.branch_discrete_actions)[winner]
        actions = {
            name: codes[:, column]
            for column, name in enumerate(sim_policy.discrete_action_names)
        }
    resources = jnp.reshape(
        jnp.asarray(
            _compute_targets(
                data={
                    **dict(states),
                    **actions,
                    "period": jnp.full(n_subjects, period, dtype=jnp.int32),
                    "age": jnp.full(n_subjects, age),
                },
                targets=[read.resources_target],
                regime=regime,
                regime_params=flat_params,
            )[read.resources_target]
        ),
        (n_subjects,),
    )
    budget = resources - read.savings_lower_bound
    rounding = 8 * jnp.finfo(action.dtype).eps * jnp.maximum(1.0, jnp.abs(budget))
    within_budget = action <= budget + rounding
    actions[read.action_name] = jnp.minimum(action, budget)
    read_ok = (
        jnp.isfinite(liquid)
        & (liquid >= nodes[0])
        & (liquid <= nodes[-1])
        & live
        & jnp.isfinite(action)
        & (action > 0.0)
        & within_budget
    )
    return actions, None, read_ok


def _state_grid_read(
    *,
    surface: FloatND,
    liquid_axis: int,
    selections: tuple[_RowSelection, ...],
    lo: IntND,
    weight: FloatND,
) -> FloatND:
    """Read one state-grid surface at the subjects' liquid brackets and rows.

    Each selected row is read linearly between the liquid nodes `lo` and
    `lo + 1` and blended by its coefficient. A term whose coefficient is zero
    contributes nothing, not even a non-finite surface value.
    """
    rows = jnp.moveaxis(surface, liquid_axis, -1)
    reads = []
    for index, coefficient in selections:
        row_read = _weighted(coefficient=1.0 - weight, value=rows[(*index, lo)]) + (
            _weighted(coefficient=weight, value=rows[(*index, lo + 1)])
        )
        reads.append(
            row_read
            if coefficient is None
            else _weighted(coefficient=coefficient, value=row_read)
        )
    return functools.reduce(jnp.add, reads)


def _weighted(*, coefficient: FloatND, value: FloatND) -> FloatND:
    """Weight a read, letting a zero weight discard even a non-finite read."""
    return jnp.where(coefficient == 0.0, 0.0, coefficient * value)


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
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], BoolND]:
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

    Falls back to the grid-argmax pair per subject when any branch read leaves
    its live row support, or the winning consumption is non-finite,
    non-positive, or outside the intrinsic budget
    (`c <= resources - savings_lower_bound`). A payload whose declared outer
    post-decision or keeper candidate does not resolve against the simulate
    function pool is refused rather than answered with the action-grid winner.

    Returns the recovered actions and a per-subject Boolean fallback flag.

    Inference must refuse whenever any fallback entry is True; the flag reaches
    the user as the `nested_policy_fallback` column.
    """
    _fail_if_the_published_capability_is_not_replayable(payload=payload)
    keeper_pol = payload.keeper
    bank = payload.adjuster
    n_subjects = next(iter(states.values())).shape[0]
    liquid = jnp.asarray(states[payload.liquid_state_name])

    discrete_idx = tuple(
        _grid_position(regime=regime, name=name, values=jnp.asarray(states[name]))
        for name in keeper_pol.row_discrete_state_names
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

    row_read = functools.partial(
        _blended_nested_read,
        passive_bracket=passive_bracket,
        discrete_idx=discrete_idx,
        liquid=liquid,
        n_subjects=n_subjects,
    )
    keeper_value, keeper_support = row_read(pol=keeper_pol, field="value")
    keeper_action, _ = row_read(pol=keeper_pol, field="policy")

    candidate_values, candidate_support, candidate_actions = vmap(
        functools.partial(
            _read_nested_candidate,
            passive_bracket=passive_bracket,
            discrete_idx=discrete_idx,
            liquid=liquid,
            n_subjects=n_subjects,
        )
    )(bank.policies)

    # The same surrogate class and safeguarded search as the solve's collapse:
    # exact node values compete directly, golden section refines only inside
    # brackets around node-local maxima of the interpolated profile.
    outer_nodes = bank.outer_nodes
    profile = jnp.where(candidate_support, candidate_values, -jnp.inf)
    search = safeguarded_continuous_argmax(
        objective=_InterpolatedProfileObjective(
            interpolant=LocalCubicOuterInterpolant(),
            nodes=outer_nodes,
            values=profile,
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

    offset, transition_at = _outer_transition_offset_and_forward(
        payload=payload,
        regime=regime,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    keep_value = _keeper_post_decision(
        payload=payload,
        regime=regime,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )

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

    # The keeper is a separately solved exact branch and may lie outside a legal,
    # narrower adjuster mesh. Validate it against the state domain carried by the
    # inverse; validate an adjusting proposal against the mesh it was solved on.
    # Both branches use the same endpoint-and-containment predicate.
    transition_image = transition_at(outer_action)
    pointwise_replay_ok = jnp.where(
        adjust,
        outer_candidate_is_admissible(
            image=transition_image,
            target=chosen_post_decision,
            low=jnp.asarray(outer_nodes[0], dtype=transition_image.dtype),
            high=jnp.asarray(outer_nodes[-1], dtype=transition_image.dtype),
        ),
        outer_candidate_is_admissible(
            image=transition_image,
            target=chosen_post_decision,
            low=jnp.asarray(
                payload.replay_capability.inverse.low, dtype=transition_image.dtype
            ),
            high=jnp.asarray(
                payload.replay_capability.inverse.high, dtype=transition_image.dtype
            ),
        ),
    )
    accepted = (
        pointwise_replay_ok
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
    return MappingProxyType(new_actions), ~accepted


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
    *,
    func: Callable[..., FloatND],
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


def _outer_transition_offset_and_forward(
    *,
    payload: NestedEGMSimPolicy,
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> tuple[FloatND, Callable[[FloatND], FloatND]]:
    """Per-subject offset of the outer post-decision, and the map itself.

    The winning outer post-decision `s'` is the value of the regime's
    `outer_post_decision` function; the recorded action must invert
    `s' = P(states, a)`. The continuous-outer scope supports the
    affine-unit-slope contract `P(states, a) = offset(states) + a`. Whether the
    declared map meets it was settled by the solve and rides on the payload, so
    this reads `payload.replay_capability` instead of certifying `P` again: a
    reader that re-certified could reach a verdict the solve it is replaying did
    not, and would then publish a decision the solve never ranked.

    What is evaluated here is only the numbers a certificate cannot carry — the
    per-subject offset, and the map itself at the recovered action for the
    per-subject affine residual.

    `P` is an ordinary regime FUNCTION of this period's states and actions --
    the stock chosen NOW -- not a state transition. The durable's law reads it
    and carries it forward, so the law itself does not mention the outer action
    and inverting the law could not recover one.
    """
    transition_at = _OuterTransitionAt(
        transition_func=_replay_function(
            regime=regime, name=payload.outer_post_decision_name
        ),
        outer_action_name=payload.outer_action_name,
        outer_post_decision_name=payload.outer_post_decision_name,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    offset = transition_at(jnp.zeros(n_subjects))
    return offset, transition_at


@dataclass(frozen=True, eq=False)
class _OuterTransitionAt:
    """The declared outer post-decision map evaluated at one action per subject."""

    transition_func: Callable[..., FloatND]
    """The regime's outer post-decision function, resolved for replay."""
    outer_action_name: ActionName
    """Name under which the evaluated action is bound into the map."""
    outer_post_decision_name: FunctionName
    """Name of the map, for the capability check's message."""
    states: Mapping[StateOrActionName, FloatND | IntND]
    """Per-subject state values the map reads."""
    flat_params: FlatRegimeParams
    """The regime's flat parameters."""
    period: int
    """Period index of the evaluation."""
    age: ScalarFloat | ScalarInt
    """Age of the evaluation."""
    n_subjects: int
    """Number of subjects; the image is reshaped to one value per subject."""

    def resolve(self, *, action_value: FloatND) -> dict[str, FloatND | IntND]:
        """Bind the map's arguments at `action_value`, refusing an unmet capability."""
        bound = _resolve_function_kwargs(
            func=self.transition_func,
            states=self.states,
            bindings={self.outer_action_name: action_value},
            flat_params=self.flat_params,
            period=self.period,
            age=self.age,
            n_subjects=self.n_subjects,
        )
        _fail_if_the_published_capability_does_not_hold(
            bound=bound, name=self.outer_post_decision_name
        )
        return cast("dict[str, FloatND | IntND]", bound)

    def __call__(self, action: FloatND) -> FloatND:
        """Return the map's image at `action`, one value per subject."""
        return jnp.reshape(
            jnp.asarray(self.transition_func(**self.resolve(action_value=action))),
            (self.n_subjects,),
        )


def _fail_if_the_published_capability_is_not_replayable(
    *, payload: NestedEGMSimPolicy
) -> None:
    """Refuse a continuous-outer payload whose own verdict says it is unreplayable.

    Structural support is settled before publication, so a payload carrying an
    unsupported verdict did not come from a publication that honoured that gate.
    The alternative to refusing is a replay that emits the action-grid winner
    under the name of the refined method, which is the outcome the verdict
    exists to prevent. Structural failure raises here; a per-subject shortfall
    is a different thing and stays with the pointwise mask below.

    Raises:
        InvalidSimulationInputError: If the payload's capability is unsupported.
    """
    if payload.replay_capability.continuous_replay_is_supported:
        return
    msg = (
        "Continuous-outer replay was handed a policy whose published replay "
        "capability reports the declaration as unreplayable: "
        f"{payload.replay_capability}. A solve that honoured its publication "
        "gate never returns such a policy, so this result did not come from one. "
        "Simulate with the `SolutionResult` this model's own `solve()` "
        "returned."
    )
    raise InvalidSimulationInputError(msg)


def _replay_function(*, regime: Regime, name: FunctionName) -> Callable[..., FloatND]:
    """Return a declared replay function, refusing a payload that outlived it.

    The published capability certified that the regime declares this function,
    so a lookup that misses names a payload replayed against a different regime
    than the one that published it.

    Raises:
        InvalidSimulationInputError: If the regime does not declare `name`.
    """
    functions = regime.simulation.functions
    if name in functions:
        return functions[name]
    msg = (
        f"Continuous-outer replay needs the declared function {name!r}, which "
        "this regime does not publish for simulation. The solve certified it "
        "before publishing the replay policy, so the policy and the regime it "
        "is being replayed against do not belong to the same model. Simulate "
        "with the `SolutionResult` this model's own `solve()` returned."
    )
    raise InvalidSimulationInputError(msg)


def _fail_if_the_published_capability_does_not_hold(
    *, bound: dict[str, FloatND | IntND] | None, name: FunctionName
) -> None:
    """Refuse a replay whose certified argument binding does not reproduce.

    The published capability certified that every argument of `name` can be
    supplied from a simulated state, a parameter, `period`, or `age`. Reaching
    an unbindable argument here means the policy was published against a
    different declaration than the one being replayed, and the alternative to
    refusing is emitting the action-grid winner for a regime that asked for the
    refined method.

    Raises:
        InvalidSimulationInputError: If the binding could not be built.
    """
    if bound is not None:
        return
    msg = (
        f"Continuous-outer replay could not bind the arguments of {name!r}, "
        "which the published replay capability certified as bindable. The "
        "replay policy and the regime it is being replayed against do not "
        "belong to the same model. Simulate with the `SolutionResult` "
        "this model's own `solve()` returned."
    )
    raise InvalidSimulationInputError(msg)


def _keeper_post_decision(
    *,
    payload: NestedEGMSimPolicy,
    regime: Regime,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    n_subjects: int,
) -> FloatND:
    """The keeper branch's outer post-decision `s' = keep(Z)` per subject.

    Uses the solver's declared no-adjustment candidate; without one, keeping
    means holding the current durable (the state the outer post-decision is
    the next-period value of) unchanged. The published capability certified
    that whichever of the two this regime declares can be evaluated here, so
    this evaluates rather than tests, and a binding that nonetheless fails is
    refused as a payload/regime mismatch.
    """
    if payload.outer_no_adjustment_name is None:
        durable = states.get(payload.outer_state_name)
        _fail_if_the_published_capability_does_not_hold(
            bound=None if durable is None else {payload.outer_state_name: durable},
            name=payload.outer_state_name,
        )
        return jnp.asarray(durable)
    keep_func = _replay_function(regime=regime, name=payload.outer_no_adjustment_name)
    kwargs = _resolve_function_kwargs(
        func=keep_func,
        states=states,
        bindings={},
        flat_params=flat_params,
        period=period,
        age=age,
        n_subjects=n_subjects,
    )
    _fail_if_the_published_capability_does_not_hold(
        bound=kwargs, name=payload.outer_no_adjustment_name
    )
    return jnp.reshape(
        jnp.asarray(keep_func(**cast("dict[str, FloatND | IntND]", kwargs))),
        (n_subjects,),
    )


# A row index over the discrete-state row axes, with the per-subject coefficient
# it enters the read with (`None`: the subject's only row).
type _RowSelection = tuple[tuple[IntND | int, ...], FloatND | None]


def _row_selections(
    *,
    regime: Regime,
    names: tuple[StateName, ...],
    states: Mapping[StateOrActionName, FloatND | IntND],
) -> tuple[_RowSelection, ...]:
    """Select each subject's discrete-state rows and their blend coefficients.

    An ordinary discrete state selects its node's row. A process state's value
    may lie between nodes; it reads the two bracketing node rows blended linearly
    in the node coordinate (extrapolating along the edge segment), the rule its
    transition weights follow off node. A node value takes its own row with
    coefficient exactly one. Where every row axis is a process, every node's row
    is shared by all subjects and is read once for all of them, rather than
    gathered per subject.
    """
    process_names = {
        name
        for name in names
        if isinstance(regime.simulation.grids[name], _ContinuousStochasticProcess)
    }
    shared = set(names) == process_names
    per_axis: list[tuple[tuple[IntND | int, FloatND | None], ...]] = []
    for name in names:
        values = jnp.asarray(states[name])
        nodes = jnp.asarray(regime.simulation.grids[name].to_jax())
        if name not in process_names:
            per_axis.append(
                ((_grid_position(regime=regime, name=name, values=values), None),)
            )
        elif nodes.shape[0] == 1:
            per_axis.append(((0, None),))
        else:
            lo, weight = _node_bracket(nodes=nodes, values=values)
            per_axis.append(
                tuple(
                    (
                        node,
                        jnp.where(node == lo, 1.0 - weight, 0.0)
                        + jnp.where(node == lo + 1, weight, 0.0),
                    )
                    for node in range(int(nodes.shape[0]))
                )
                if shared
                else ((lo, 1.0 - weight), (lo + 1, weight))
            )
    selections = []
    for combo in itertools.product(*per_axis):
        coefficients = [
            coefficient for _, coefficient in combo if coefficient is not None
        ]
        selections.append(
            (
                tuple(position for position, _ in combo),
                functools.reduce(jnp.multiply, coefficients) if coefficients else None,
            )
        )
    return tuple(selections)


def _node_bracket(*, nodes: Float1D, values: FloatND) -> tuple[IntND, FloatND]:
    """Lower bracketing node and linear weight of each value, as in the weights.

    Mirrors `get_irreg_coordinate`: the lower node is clipped to the first or
    last segment, so the weight extrapolates beyond the outer nodes, and a node
    value has weight exactly 0 (or exactly 1 at the last node).
    """
    lo = jnp.clip(
        jnp.searchsorted(nodes, values, side="right") - 1, 0, nodes.shape[0] - 2
    )
    return lo, (values - nodes[lo]) / (nodes[lo + 1] - nodes[lo])


def _blended_row_read(
    *,
    sim_policy: EGMSimPolicy,
    field: Literal["policy", "value"],
    selections: tuple[_RowSelection, ...],
    branch: tuple[int, ...],
    resources: FloatND,
    n_subjects: int,
) -> tuple[FloatND, BoolND]:
    """Read one field of the selected rows and blend them by their coefficients.

    A row with coefficient zero contributes nothing, not even a `-inf` or NaN
    read; the blend is in support where every contributing row is. A row
    without a coefficient is the subject's only row and is returned as read.
    """
    values, supports = [], []
    for index, coefficient in selections:
        read, in_support = _interp_rows_with_support(
            sim_policy=sim_policy,
            field=field,
            index=(*index, *branch),
            resources=resources,
            n_subjects=n_subjects,
        )
        if coefficient is None:
            values.append(read)
            supports.append(in_support)
        else:
            values.append(jnp.where(coefficient == 0.0, 0.0, coefficient * read))
            supports.append((coefficient == 0.0) | in_support)
    return (
        functools.reduce(jnp.add, values),
        functools.reduce(jnp.logical_and, supports),
    )


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
    values, in_support = _interp_padded_rows(
        rows_x=rows_x, rows_f=rows_f, rows_slope=rows_slope, resources=resources
    )
    return jnp.broadcast_to(values, (n_subjects,)), jnp.broadcast_to(
        in_support, (n_subjects,)
    )


@jax.jit
def _interp_padded_rows(
    *,
    rows_x: FloatND,
    rows_f: FloatND,
    rows_slope: FloatND | None,
    resources: FloatND,
) -> tuple[FloatND, BoolND]:
    """Read one padded row per subject at its resources, with the support flag.

    Compiled once per row shape: simulation reads these rows every
    regime-period, and the eager per-operation dispatch dominated the read. A
    single row shared by every subject (`rows_x.ndim == 1`) is read once at
    all resources rather than copied per subject.
    """
    if rows_x.ndim == 1:
        values = interp_on_padded_grid(
            x_query=resources, xp=rows_x, fp=rows_f, fp_slopes=rows_slope
        )
        last_live = rows_x[jnp.sum(jnp.isfinite(rows_x)) - 1]
        return values, (resources >= rows_x[0]) & (resources <= last_live)
    if rows_slope is None:
        values = vmap(interp_on_padded_grid)(x_query=resources, xp=rows_x, fp=rows_f)
    else:
        values = vmap(interp_on_padded_grid)(
            x_query=resources, xp=rows_x, fp=rows_f, fp_slopes=rows_slope
        )
    valid_length = jnp.sum(jnp.isfinite(rows_x), axis=-1)
    last_live = jnp.take_along_axis(rows_x, (valid_length - 1)[:, None], axis=-1)[:, 0]
    in_support = (resources >= rows_x[:, 0]) & (resources <= last_live)
    return values, in_support


def _select_nnbegm_discrete_actions(
    *,
    sim_policy: NNBEGMSimPolicy,
    winner: IntND,
    n_candidates: int,
) -> tuple[IntND | None, dict[ActionName, IntND]]:
    """Select exact discrete codes from the winning solve candidates."""
    if not sim_policy.discrete_action_names:
        if sim_policy.candidate_discrete_actions is not None:
            raise ValueError(
                "NNBEGM smooth replay received an unexpected discrete code bank."
            )
        return None, {}

    candidate_codes = sim_policy.candidate_discrete_actions
    if candidate_codes is None:
        raise ValueError("NNBEGM discrete replay payload is missing exact codes.")
    if candidate_codes.shape != (
        n_candidates,
        len(sim_policy.discrete_action_names),
    ):
        raise ValueError("NNBEGM discrete replay code bank is misaligned.")
    selected_codes = jnp.take(candidate_codes, winner, axis=0)
    selected_actions = {
        name: selected_codes[:, position]
        for position, name in enumerate(sim_policy.discrete_action_names)
    }
    return selected_codes, selected_actions


def _nnbegm_bank_feasibility(
    *,
    candidate_inner: FloatND,
    candidate_outer: FloatND,
    optimal_actions: Mapping[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
) -> BoolND:
    """Evaluate the period's constraints at every candidate tuple of the bank."""
    functions, constraints = regime.simulation.feasibility_pools_by_period[period]
    dag_bounds = tuple(
        bound for bound in regime.simulation.post_decision_bounds if bound.kind == "dag"
    )
    feasibility = _get_feasibility(
        functions=functions,
        constraints=constraints,
        bound_specs=dag_bounds,
        bound_periods=(period,) if dag_bounds else (),
    )
    n_candidates = candidate_inner.shape[0]
    codes = sim_policy.candidate_discrete_actions
    bank = {
        sim_policy.inner_action_name: candidate_inner,
        sim_policy.outer_action_name: candidate_outer,
        **{
            name: jnp.broadcast_to(codes[:, position, None], candidate_inner.shape)
            for position, name in enumerate(sim_policy.discrete_action_names)
            if codes is not None
        },
    }
    shared = {**flat_params, "period": jnp.int32(period), "age": age}
    per_subject = {
        **canonical_states,
        **{name: optimal_actions[name] for name in action_names if name not in bank},
    }
    names = inspect.signature(feasibility).parameters

    # Constraints are written for one state-action point, so the bank is mapped
    # over candidates and subjects alike.
    def at_point(point: Mapping[str, FloatND | IntND]) -> BoolND:
        pool = {**shared, **point}
        evaluated = feasibility(**{name: pool[name] for name in names})
        output = evaluated.output if type(evaluated) is BoundedResult else evaluated
        return jnp.asarray(output, dtype=bool)

    def at_candidate(candidate: Mapping[ActionName, FloatND | IntND]) -> BoolND:
        return jax.vmap(at_point)({**per_subject, **candidate})

    return jax.vmap(at_candidate)(bank).reshape(n_candidates, -1)


def _score_nnbegm_winner(
    *,
    winner_actions: Mapping[ActionName, FloatND | IntND],
    regime: Regime,
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
    bound_summary: bool = False,
) -> tuple[FloatND, BoolND] | BoundedResult[tuple[FloatND, BoolND]]:
    """Score the winning complete action tuple once under canonical Q."""
    scored = _evaluate_canonical_Q_at_actions(
        candidate_actions=winner_actions,
        regime=regime,
        canonical_states=canonical_states,
        action_names=action_names,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=flat_params,
        referenced_value_kwargs=referenced_value_kwargs,
        period=period,
        age=age,
    )
    if type(scored) is not BoundedResult or bound_summary:
        return scored
    validate_runtime_bound_status(
        statuses=scored.bound_status,
        bound_specs=tuple(
            bound
            for bound in regime.simulation.post_decision_bounds
            if bound.kind == "dag"
        ),
        regime_name=regime.name,
        period=period,
    )
    return cast("tuple[FloatND, BoolND]", scored.output)


def _invert_nnbegm_outer_targets(
    *,
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    candidate_inner: FloatND,
    candidate_target: FloatND,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
) -> tuple[FloatND, BoolND]:
    """Invert retained targets at realized states and verify the round trip."""
    policy_read = regime.simulation.egm_policy_read
    if not isinstance(policy_read, NNBEGMPolicyRead):
        raise TypeError("NNBEGM target inversion requires its replay adapter.")
    target_function = policy_read.outer_target_function_by_period[period]
    accepted = inspect.signature(target_function).parameters
    n_candidates, n_subjects = candidate_inner.shape
    candidate_shape = (n_candidates, n_subjects)
    state_inputs = {
        name: jnp.broadcast_to(jnp.asarray(value)[None, :], candidate_shape)
        for name, value in states.items()
    }
    if sim_policy.discrete_action_names:
        if sim_policy.candidate_discrete_actions is None:
            raise ValueError("NNBEGM discrete replay is missing candidate codes.")
        discrete_inputs = {
            name: jnp.broadcast_to(
                sim_policy.candidate_discrete_actions[:, position, None],
                candidate_shape,
            )
            for position, name in enumerate(sim_policy.discrete_action_names)
        }
    else:
        discrete_inputs = {}

    forward = _NNBEGMOuterForward(
        target_function=target_function,
        accepted_names=frozenset(accepted),
        flat_params=flat_params,
        state_inputs=state_inputs,
        discrete_inputs=discrete_inputs,
        inner_action_name=sim_policy.inner_action_name,
        outer_action_name=sim_policy.outer_action_name,
        candidate_inner=candidate_inner,
        period=period,
        age=age,
        outer_post_decision=policy_read.outer_post_decision,
        candidate_shape=candidate_shape,
    )

    zeros = jnp.zeros_like(candidate_inner)
    bound_at_zero = forward.bind(outer_action=zeros)
    at_zero_results = target_function(**bound_at_zero)
    at_zero = jnp.broadcast_to(
        jnp.asarray(at_zero_results[policy_read.outer_post_decision]), candidate_shape
    )
    # The certificate the solve took, carried on the policy it published rather
    # than taken again here. Both phases must recover the action identically and
    # admit the same stocks, so simulation reads the solve's answer -- including
    # the period's own outer domain, which an age-varying outer grid would
    # otherwise leave the two phases disagreeing about.
    inverse = sim_policy.replay_capability.inverse

    if policy_read.outer_no_adjustment_target is None:
        keeper_targets = jnp.broadcast_to(
            jnp.asarray(states[policy_read.outer_state_name])[None, :],
            (sim_policy.n_keeper_candidates, n_subjects),
        )
    else:
        keeper_targets = jnp.broadcast_to(
            jnp.asarray(at_zero_results[policy_read.outer_no_adjustment_target]),
            candidate_shape,
        )[: sim_policy.n_keeper_candidates]
    # The adjuster candidates target declared search nodes, so the node values
    # are taken from the policy rather than read off the interpolated bank: the
    # bank's adjuster rows are constant along the state axes, but interpolating
    # a constant surface reproduces the node only to within a rounding, and a
    # target a rounding away from a domain endpoint is not that endpoint. The
    # bank is still read for its finiteness, which is how the solve records a
    # candidate it dropped itself.
    interpolated_adjuster = candidate_target[sim_policy.n_keeper_candidates :]
    n_adjuster_candidates = interpolated_adjuster.shape[0]
    n_outer_nodes = sim_policy.outer_grid_values.shape[0]
    branches_per_node = n_adjuster_candidates // n_outer_nodes
    if branches_per_node * n_outer_nodes != n_adjuster_candidates:
        raise ValueError(
            "NNBEGM adjuster candidates do not divide evenly over the declared "
            f"outer nodes: {n_adjuster_candidates} candidates, "
            f"{n_outer_nodes} nodes."
        )
    declared_adjuster = jnp.broadcast_to(
        jnp.repeat(sim_policy.outer_grid_values, repeats=branches_per_node)[:, None],
        interpolated_adjuster.shape,
    ).astype(interpolated_adjuster.dtype)
    adjuster_targets = jnp.where(
        jnp.isfinite(interpolated_adjuster), declared_adjuster, jnp.nan
    )
    realized_target = jnp.concatenate((keeper_targets, adjuster_targets), axis=0)

    inversion = invert_declared_outer_target(
        inverse=inverse,
        target=realized_target,
        at_zero=at_zero,
        forward=forward,
    )
    represented = jnp.isfinite(realized_target) & inversion.admissible
    # An unrepresented candidate carries an action that reaches a stock outside
    # the outer state's domain. It is dropped rather than published, and the
    # action itself is blanked so no downstream reader can score it: a candidate
    # the solve never ranked must not be able to win by arriving at the canonical
    # Q with a finite number attached.
    candidate_outer = jnp.where(represented, inversion.action, jnp.nan)
    return candidate_outer, represented


@dataclass(frozen=True, eq=False)
class _NNBEGMOuterForward:
    """The declared outer target map over one candidate bank, per subject."""

    target_function: Callable[..., Mapping[str, FloatND]]
    """The period's outer target function from the replay adapter."""
    accepted_names: frozenset[str]
    """Parameter names `target_function` accepts; every other input is dropped."""
    flat_params: FlatRegimeParams
    """The regime's flat parameters."""
    state_inputs: Mapping[str, FloatND | IntND]
    """Per-subject states broadcast to the candidate bank's shape."""
    discrete_inputs: Mapping[str, IntND]
    """Candidate discrete action codes broadcast to the bank's shape, if any."""
    inner_action_name: ActionName
    """Name under which the candidate inner actions are bound."""
    outer_action_name: ActionName
    """Name under which the evaluated outer action is bound."""
    candidate_inner: FloatND
    """Candidate inner actions, shaped `(n_candidates, n_subjects)`."""
    period: int
    """Period index of the evaluation."""
    age: ScalarFloat | ScalarInt
    """Age of the evaluation."""
    outer_post_decision: str
    """Name of the outer post-decision output read from the map's results."""
    candidate_shape: tuple[int, int]
    """`(n_candidates, n_subjects)`; every image is broadcast to it."""

    def bind(self, *, outer_action: FloatND) -> dict[str, EconFunctionArg]:
        """Assemble the map's accepted inputs at `outer_action`."""
        pool = {
            **dict(self.flat_params),
            **self.state_inputs,
            **self.discrete_inputs,
            self.inner_action_name: self.candidate_inner,
            self.outer_action_name: outer_action,
            "period": jnp.int32(self.period),
            "age": self.age,
        }
        return {
            name: value for name, value in pool.items() if name in self.accepted_names
        }

    def evaluate(self, *, outer_action: FloatND) -> Mapping[str, FloatND]:
        """Evaluate every output of the map at `outer_action`."""
        return self.target_function(**self.bind(outer_action=outer_action))

    def __call__(self, outer_action: FloatND) -> FloatND:
        """Return the outer post-decision image at `outer_action`, bank-shaped."""
        return jnp.broadcast_to(
            jnp.asarray(
                self.evaluate(outer_action=outer_action)[self.outer_post_decision]
            ),
            self.candidate_shape,
        )


def _replay_nnbegm_candidates(
    *,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
) -> tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]:
    """Reconstruct the solve candidate product, then rank it by published value."""
    candidate_inner, candidate_outer, candidate_value, represented = (
        _prepare_nnbegm_candidate_bank(
            regime=regime,
            sim_policy=sim_policy,
            states=states,
            flat_params=flat_params,
            period=period,
            age=age,
        )
    )
    return cast(
        "tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]",
        _rank_nnbegm_candidate_bank(
            candidate_inner=candidate_inner,
            candidate_outer=candidate_outer,
            candidate_value=candidate_value,
            represented=represented,
            optimal_actions=optimal_actions,
            regime=regime,
            sim_policy=sim_policy,
            flat_params=flat_params,
            period=period,
            age=age,
            canonical_states=canonical_states,
            action_names=action_names,
            next_regime_to_V_arr=next_regime_to_V_arr,
            referenced_value_kwargs=referenced_value_kwargs,
        ),
    )


def _prepare_nnbegm_candidate_bank(
    *,
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    states: Mapping[StateOrActionName, FloatND | IntND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
) -> tuple[FloatND, FloatND, FloatND, BoolND]:
    """Pure reconstruction of the candidate bank and its represented mask.

    Returns the inner actions, outer actions and published values of every
    candidate at the subjects' states, and whether each candidate is live (in
    support, finite) and its outer action reaches a stock inside the outer
    state's domain. Support is the grid's extent for an ordinary continuous
    state; a process state's draw is in support wherever it is finite.
    """
    n_subjects = next(iter(states.values())).shape[0]
    coordinates = []
    in_support = jnp.ones((n_subjects,), dtype=bool)
    for name in sim_policy.state_names:
        values = jnp.asarray(states[name])
        grid = regime.simulation.grids[name]
        points = jnp.asarray(grid.to_jax())
        if isinstance(grid, DiscreteGrid):
            coordinate = jnp.clip(
                jnp.searchsorted(points, values), 0, points.shape[0] - 1
            )
            in_support &= points[coordinate] == values
        elif isinstance(grid, _ContinuousStochasticProcess):
            # A process value off its nodes is read linearly in the node
            # coordinate, extrapolating along the edge segment beyond the outer
            # nodes, the rule its transition weights follow off node.
            coordinate = grid.get_coordinate(values)
            in_support &= jnp.isfinite(values)
        else:
            continuous_grid = cast("ContinuousGrid", grid)
            coordinate = continuous_grid.get_coordinate(values)
            in_support &= (
                jnp.isfinite(values) & (values >= points[0]) & (values <= points[-1])
            )
        coordinates.append(coordinate)

    read_bank = functools.partial(_read_candidate_bank, coordinates=coordinates)
    candidate_inner = read_bank(bank=sim_policy.candidate_inner_action)
    candidate_target = read_bank(bank=sim_policy.candidate_outer_target)
    candidate_value = read_bank(bank=sim_policy.candidate_value)
    candidate_outer, outer_represented = _invert_nnbegm_outer_targets(
        regime=regime,
        sim_policy=sim_policy,
        candidate_inner=candidate_inner,
        candidate_target=candidate_target,
        states=states,
        flat_params=flat_params,
        period=period,
        age=age,
    )
    represented = (
        in_support[None, :]
        & jnp.isfinite(candidate_inner)
        & jnp.isfinite(candidate_value)
        & outer_represented
    )
    return candidate_inner, candidate_outer, candidate_value, represented


def _rank_nnbegm_candidate_bank(
    *,
    candidate_inner: FloatND,
    candidate_outer: FloatND,
    candidate_value: FloatND,
    represented: BoolND,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    regime: Regime,
    sim_policy: NNBEGMSimPolicy,
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    referenced_value_kwargs: QAndFKwargs,
    bound_summary: bool = False,
) -> (
    tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]
    | BoundedResult[tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]]
):
    """Pick the feasible represented candidate with the highest published value.

    Feasibility is the regime's constraints at the subject's state, evaluated
    over the bank; the continuation is not. JAX argmax returns the first
    maximum, preserving keeper/outer-major then declared discrete-product order
    exactly. Canonical Q scores the one winning action tuple for the reported
    value; a winner it scores non-finite publishes no action, as does a subject
    with no feasible represented candidate.
    """
    n_candidates = candidate_inner.shape[0]
    feasible = _nnbegm_bank_feasibility(
        candidate_inner=candidate_inner,
        candidate_outer=candidate_outer,
        optimal_actions=optimal_actions,
        regime=regime,
        sim_policy=sim_policy,
        flat_params=flat_params,
        period=period,
        age=age,
        canonical_states=canonical_states,
        action_names=action_names,
    )
    ranking_values = jnp.where(represented & feasible, candidate_value, -jnp.inf)
    winner = jnp.asarray(jnp.argmax(ranking_values, axis=0), dtype=jnp.int32)
    selected_inner = _take_at_winner(stacked=candidate_inner, winner=winner)
    selected_outer = _take_at_winner(stacked=candidate_outer, winner=winner)
    selected_codes, selected_discrete_actions = _select_nnbegm_discrete_actions(
        sim_policy=sim_policy,
        winner=winner,
        n_candidates=n_candidates,
    )
    scored = _score_nnbegm_winner(
        winner_actions={
            **{
                name: optimal_actions[name]
                for name in action_names
                if name in optimal_actions
            },
            sim_policy.inner_action_name: selected_inner,
            sim_policy.outer_action_name: selected_outer,
            **selected_discrete_actions,
        },
        regime=regime,
        flat_params=flat_params,
        period=period,
        age=age,
        canonical_states=canonical_states,
        action_names=action_names,
        next_regime_to_V_arr=next_regime_to_V_arr,
        referenced_value_kwargs=referenced_value_kwargs,
        bound_summary=bound_summary,
    )
    selected_value, selected_feasible = cast(
        "tuple[FloatND, BoolND]",
        scored.output if type(scored) is BoundedResult else scored,
    )
    any_valid = (
        jnp.any(represented & feasible, axis=0)
        & selected_feasible
        & jnp.isfinite(selected_value)
    )
    chosen_inner = jnp.where(any_valid, selected_inner, jnp.nan)
    chosen_outer = jnp.where(any_valid, selected_outer, jnp.nan)
    chosen_value = jnp.where(any_valid, selected_value, -jnp.inf)
    if selected_codes is None:
        chosen_discrete_actions = {}
    else:
        chosen_discrete_actions = {
            name: jnp.where(any_valid, selected_discrete_actions[name], -1)
            for name in sim_policy.discrete_action_names
        }
    output = (
        MappingProxyType(
            {
                **optimal_actions,
                sim_policy.inner_action_name: chosen_inner,
                sim_policy.outer_action_name: chosen_outer,
                **chosen_discrete_actions,
            }
        ),
        chosen_value,
    )
    return (
        BoundedResult[tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]](
            output=output, bound_status=scored.bound_status
        )
        if type(scored) is BoundedResult
        else output
    )


def _grid_position(
    *, regime: Regime, name: StateOrActionName, values: FloatND | IntND
) -> IntND:
    """Return each value's clipped insertion position on the named simulation grid."""
    grid_values = jnp.asarray(regime.simulation.grids[name].to_jax())
    return jnp.clip(jnp.searchsorted(grid_values, values), 0, grid_values.shape[0] - 1)


def _take_at_winner(*, stacked: FloatND | BoolND, winner: IntND) -> FloatND | BoolND:
    """Select the winning candidate's row from a candidate-major stack per subject."""
    return jnp.take_along_axis(stacked, winner[None, :], axis=0)[0]


def _blended_nested_read(
    *,
    pol: EGMSimPolicy,
    field: Literal["policy", "value"],
    passive_bracket: tuple[IntND, IntND, FloatND] | None,
    discrete_idx: tuple[IntND, ...],
    liquid: FloatND,
    n_subjects: int,
) -> tuple[FloatND, BoolND]:
    """Read one policy field at the subjects' liquid values and discrete rows.

    With a passive bracket, the two enclosing rows on the passive state's axis
    are read and interpolated linearly with the bracket's weight; support is the
    conjunction of both rows' support.
    """
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


# keyword-only-exempt: library-callback=jax.vmap
def _read_nested_candidate(
    pol: EGMSimPolicy,
    *,
    passive_bracket: tuple[IntND, IntND, FloatND] | None,
    discrete_idx: tuple[IntND, ...],
    liquid: FloatND,
    n_subjects: int,
) -> tuple[FloatND, BoolND, FloatND]:
    """Read one candidate policy's value, support, and action; mapped over the bank."""
    value, support = _blended_nested_read(
        pol=pol,
        field="value",
        passive_bracket=passive_bracket,
        discrete_idx=discrete_idx,
        liquid=liquid,
        n_subjects=n_subjects,
    )
    action, _ = _blended_nested_read(
        pol=pol,
        field="policy",
        passive_bracket=passive_bracket,
        discrete_idx=discrete_idx,
        liquid=liquid,
        n_subjects=n_subjects,
    )
    return value, support, action


@dataclass(frozen=True, eq=False)
class _InterpolatedProfileObjective:
    """The interpolated candidate-value profile as a continuous search objective."""

    interpolant: LocalCubicOuterInterpolant
    """The surrogate that interpolates the profile between outer nodes."""
    nodes: FloatND
    """Outer nodes the profile is known at."""
    values: FloatND
    """Profile values at `nodes`, `-inf` where a candidate is unsupported."""

    def __call__(self, query: FloatND) -> FloatND:
        """Evaluate the interpolated profile at `query`."""
        return self.interpolant.evaluate(
            nodes=self.nodes, values=self.values, query=query
        )


def _read_candidate_bank(
    *, bank: FloatND, coordinates: list[FloatND | IntND]
) -> FloatND:
    """Interpolate every surface of a candidate bank at the subjects' coordinates."""
    return jax.vmap(functools.partial(map_coordinates, coordinates=coordinates))(
        input=bank
    )


def _canonical_Q_at_actions(
    *,
    candidate_actions: Mapping[ActionName, FloatND | IntND],
    regime: Regime,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    period: int,
    age: ScalarFloat | ScalarInt,
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
    active_subjects: BoolND | None = None,
) -> tuple[FloatND, BoolND]:
    """Score one action value per subject with the canonical state-action value.

    Evaluates `Q` and its feasibility at exactly the supplied (state, action)
    pairs through `regime.simulation.Q_and_F` — the same objective, transitions,
    constraints, aggregators, params, and next-period value arrays the grid
    maximization uses. A value it returns is therefore directly comparable with
    the grid winner's value, and the flag it returns is the canonical
    feasibility (user constraints and the intrinsic budget alike).

    Returns:
        Tuple of the per-subject canonical value and feasibility.

    """
    evaluated = _evaluate_canonical_Q_at_actions(
        candidate_actions=candidate_actions,
        regime=regime,
        canonical_states=canonical_states,
        action_names=action_names,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=flat_params,
        period=period,
        age=age,
        referenced_value_kwargs=referenced_value_kwargs,
        active_subjects=active_subjects,
    )
    return _validated_canonical_Q(evaluated=evaluated, regime=regime, period=period)


def _validated_canonical_Q(
    *,
    evaluated: tuple[FloatND, BoolND] | BoundedResult[tuple[FloatND, BoolND]],
    regime: Regime,
    period: int,
) -> tuple[FloatND, BoolND]:
    """Refuse a canonical evaluation whose integrity records failed, else unwrap it."""
    if type(evaluated) is BoundedResult:
        validate_runtime_bound_status(
            statuses=evaluated.bound_status,
            bound_specs=tuple(
                bound
                for bound in regime.simulation.post_decision_bounds
                if bound.kind == "dag"
            ),
            regime_name=regime.name,
            period=period,
        )
        return cast("tuple[FloatND, BoolND]", evaluated.output)
    return cast("tuple[FloatND, BoolND]", evaluated)


def _evaluate_canonical_Q_at_actions(
    *,
    candidate_actions: Mapping[ActionName, FloatND | IntND],
    regime: Regime,
    canonical_states: Mapping[StateName, FloatND | IntND],
    action_names: tuple[ActionName, ...],
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
    flat_params: FlatRegimeParams,
    period: int | IntND,
    age: ScalarFloat | ScalarInt,
    referenced_value_kwargs: QAndFKwargs = MappingProxyType({}),
    active_subjects: BoolND | None = None,
    Q_and_F: QAndFFunction | None = None,
) -> tuple[FloatND, BoolND] | BoundedResult[tuple[FloatND, BoolND]]:
    """Evaluate canonical actions once, retaining integrity through transformations.

    `Q_and_F` is the period's `regime.simulation.Q_and_F` entry; a caller that
    traces `period` passes it, since a traced period cannot index the mapping.
    """
    from _lcm.regime_building.processing import (  # noqa: PLC0415 - completed model's pointwise owner
        _SubjectPointwiseQAndF,
    )

    function = (
        regime.simulation.Q_and_F[cast("int", period)] if Q_and_F is None else Q_and_F
    )
    owner = function
    while isinstance(owner, functools.partial):
        if owner.args:
            raise TypeError(
                "Pointwise integrity requires keyword-only parameter bindings."
            )
        owner = owner.func
    evaluated = function(
        **canonical_states,
        **{name: jnp.asarray(candidate_actions[name]) for name in action_names},
        next_regime_to_V_arr=next_regime_to_V_arr,
        **referenced_value_kwargs,
        **flat_params,
        period=jnp.asarray(period, dtype=jnp.int32),
        age=age,
        **(
            {ACTIVE_SUBJECTS_ARGUMENT: active_subjects}
            if active_subjects is not None and isinstance(owner, _SubjectPointwiseQAndF)
            else {}
        ),
    )
    if type(evaluated) is BoundedResult:
        if len(evaluated.bound_status) != sum(
            bound.kind == "dag" for bound in regime.simulation.post_decision_bounds
        ):
            raise TypeError(
                "Runtime bound status must match model declaration authority."
            )
        values, feasible = cast("tuple[FloatND, BoolND]", evaluated.output)
        return BoundedResult(
            output=(jnp.asarray(values), jnp.asarray(feasible).astype(bool)),
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
        )
    if any(bound.kind == "dag" for bound in regime.simulation.post_decision_bounds):
        raise TypeError(
            "Canonical actual-action evaluation must produce mandatory integrity."
        )
    values, feasible = cast("tuple[FloatND, BoolND]", evaluated)
    return jnp.asarray(values), jnp.asarray(feasible).astype(bool)


def _drawn_code_targets(
    *,
    targets: Iterable[RegimeName],
    regime_names_to_ids: RegimeNamesToIds,
) -> tuple[RegimeName, ...]:
    """Order the drawn targets by regime id, as the memory profile records them."""
    return tuple(
        sorted(dict.fromkeys(targets), key=lambda name: int(regime_names_to_ids[name]))
    )


def _genuine_rows(*, subject_slice: slice | SubjectRows, n_real: int) -> np.ndarray:
    """Mark the chunk rows that stand for a distinct real subject.

    An ungrouped chunk's rows at or past `n_real` are alignment copies of the
    last subject; a grouped chunk's short tail repeats its last original row.
    """
    if isinstance(subject_slice, SubjectRows):
        rows = subject_slice.rows
        return np.concatenate(([True], rows[1:] != rows[:-1]))
    return np.arange(subject_slice.start, subject_slice.stop) < n_real


def _current_iid_invalid(*, invalid: BoolND, in_regime: BoolND) -> BoolND:
    """Consume native law status only for occupant rows, independently of logging."""
    return jnp.any(invalid & in_regime)


def _active_subjects(
    *,
    in_regime: BoolND,
    genuine: BoolND | np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> BoolND:
    """Own integrity only for real subjects acting in this source regime."""
    return in_regime & genuine


def _first_empty_action_row(*, indices: IntND, in_regime: BoolND) -> IntND:
    """Find an owned empty decision, excluding other regimes and padding."""
    rows = jnp.arange(in_regime.shape[0], dtype=jnp.int32)
    first = jnp.min(jnp.where(in_regime & (indices < 0), rows, in_regime.shape[0]))
    return jnp.where(first < in_regime.shape[0], first, -1)


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
    initial_ages: Float1D | Int1D,
    ages: TimeAxis,
    memory: SimulationMemory | None = None,
) -> Int1D:
    """Convert per-subject initial ages to starting period indices.

    Args:
        initial_ages: Array of initial ages for each subject.
        ages: TimeAxis defining the lifecycle.

    Returns:
        Array of starting period indices (one per subject).

    Raises:
        ValueError: If any initial age is not a valid age grid point.

    """
    starting_periods, valid, all_valid = run_simulation_operation(
        memory=memory,
        function=population_operations.starting_periods,
        arguments={"initial_ages": initial_ages, "age_values": ages.values},  # noqa: PD011
        subject_arg_names=("initial_ages",),
    )
    if not bool(all_valid):
        # Error-detail selection stays on the host after the admitted predicate;
        # no data-dependent device gather can bypass the operation's profile.
        invalid_ages = np.asarray(initial_ages)[~np.asarray(valid)]
        msg = (
            f"Initial ages {invalid_ages.tolist()} are not valid age grid points. "
            f"Valid ages: {ages.values}."  # noqa: PD011
        )
        raise ValueError(msg)

    return starting_periods
