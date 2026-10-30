"""Compile forward chunk stages and reserve their explicitly retained owners.

The profile executes no numerical stage. Real completed grids and retained solution
arrays supply only descriptors. Logical future slots are conservative reservations,
separate from both actual entry-buffer residency and raw compiler peaks.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_replay_inputs import CurrentIIDReplayInputs
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueTransferKind,
    classify_value_transfer,
    resolve_value_transfer,
)
from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.projected_forward import ProjectedForwardInputs
from _lcm.simulation import (
    population_operations,
)
from _lcm.simulation.arrival_routing import record_arrival_entries
from _lcm.simulation.arrival_values import is_arrival_surface, stored_arrival_surface
from _lcm.simulation.assembly import (
    _concatenate_arrays,
    _slice_array,
    _take_rows,
)
from _lcm.simulation.chunk_offload import chunk_host_device
from _lcm.simulation.chunk_operations import (
    _broadcast_collective,
    _broadcast_value,
    _empty_fallback,
    _period_age,
    _regime_mask,
    _slice_population,
    _take_population,
)
from _lcm.simulation.chunk_planning import (
    SimulationChunkProfile,
    SimulationStageProfile,
)
from _lcm.simulation.chunk_profile_inventory import (
    ChunkProfileInventory,
    abstract_tree,
    add_bytes,
    maximum_bytes,
    payload_bytes,
)
from _lcm.simulation.current_iid_overrides import CurrentIIDOverrideSchema
from _lcm.simulation.diagnostic_operations import diagnostic_bindings
from _lcm.simulation.dissolution_route import empty_gated_entries, record_open_entries
from _lcm.simulation.entry_allocations import _pad_initial_leaf
from _lcm.simulation.exit_profiles import profile_exit_period
from _lcm.simulation.forward_program_profiles import (
    ForwardProgramProfile,
    _profile_program,
    _stochastic_keys,
    profile_forward_unit,
)
from _lcm.simulation.gated_routing import (
    commit_gated_route_delta,
    gated_route_candidates,
)
from _lcm.simulation.host_operations import StaticArgument
from _lcm.simulation.initial_conditions import (
    MISSING_CAT_CODE,
    _cast_carrier,
    _fill_carrier,
)
from _lcm.simulation.membership import (
    _activate_subject_membership,
    _empty_subject_membership,
)
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.program_arguments import gate_route_arguments
from _lcm.simulation.programs import forward_regimes_by_period
from _lcm.simulation.random import (
    _create_simulation_key,
    _generate_rowed_simulation_keys,
    _generate_windowed_simulation_keys,
)
from _lcm.simulation.random_address import RandomAddress, draw_window_keys
from _lcm.simulation.replay_inputs import replay_payload_reads
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.subject_groups import grouped_rows, type_local_transfer
from _lcm.simulation.taste_stream import (
    draw_taste_shock_keys,
)
from _lcm.simulation.transitions import (
    _advance_states_for_subjects,
    _clip_next_states,
    _count_drawn_off_grid_codes,
    _count_realized_grid_state,
    _draw_random_regime_ids_from_scalars,
    _update_regime_ids,
    drawn_code_checks,
    transition_random_addresses,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.solution.backward_induction import (
    CompilationWave,
    _resolve_compilation_workers,
)
from _lcm.time import TimeAxis
from _lcm.transition_checks import (
    regime_probability_flags,
    regime_probability_inactive_indices,
)
from _lcm.typing import (
    FlatParams,
    PeriodToRegimeToArrivalSurfaces,
    RegimeName,
    RegimeNamesToIds,
    ShapeDtypePytree,
    SimulationPolicy,
)
from _lcm.utils.logging import LogLevel, get_logger
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import SIMULATION_POLICY

type _FiniteRankOutput = tuple[
    Mapping[str, jax.ShapeDtypeStruct], jax.ShapeDtypeStruct, jax.ShapeDtypeStruct
]

# Output descriptors of a gate route: the per-regime state delta, the routed regime
# ids and the routed stakeholder roles.
type _GateRouteOutput = tuple[
    Mapping[str, ShapeDtypePytree], jax.ShapeDtypeStruct, jax.ShapeDtypeStruct
]


# Keep setup, per-unit publication and period cleanup in their lifetime order.
def profile_simulation_chunk(  # noqa: C901, PLR0912, PLR0915
    *,
    runtime: SimulationRuntime,
    regimes: Mapping[str, Regime],
    flat_params: FlatParams,
    base_spaces: Mapping[str, StateActionSpace],
    values: Mapping[int, Mapping[str, jax.Array]],
    ages: TimeAxis,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    n_subjects: int,
    population: int,
    original_population: int,
    widths: Mapping[str, int],
    independent_taste: bool,
    prng_impl: str | None = None,
    taste_prng_impl: str = "threefry2x32",
    log_level: LogLevel,
    runtime_checks: bool = True,
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None = None,
    current_iid_replay_inputs: CurrentIIDReplayInputs = MappingProxyType({}),
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
    arrival_surfaces: PeriodToRegimeToArrivalSurfaces = MappingProxyType({}),
    max_compilation_workers: int | None = None,
    group_sizes: tuple[int, ...] | None = None,
    current_iid_override_schema: CurrentIIDOverrideSchema = (),
) -> SimulationChunkProfile:
    """Prepare actual compiled stages for one proposed outer population extent.

    The current route covers declared grid and finite-policy decisions without
    host gated or other replay adapters. Diagnostics, retained storage and outer
    assembly profiles feed the selector before any candidate chunk is allocated.

    Every forward unit is first lowered from the entry carrier on the calling
    thread into one compilation wave, which compiles each program on the worker
    pool; the ordered walk then finds each executable in the runtime's shared
    cache under its exact abstract key and compiles only what the wave did not.

    With `group_sizes`, the subjects of each code of the grouping state form
    chunks of their own: the population is not padded to a chunk multiple, each
    chunk gathers its original rows and their keys, reads every value carrying
    the state through one code's block, and the outputs are gathered back into
    original order.
    """
    # `simulate` reaches this module through chunk admission, so import lazily.
    from _lcm.simulation.simulate import (  # noqa: PLC0415
        _active_subjects,
        _current_iid_invalid,
        _first_empty_action_row,
    )

    if population < original_population or original_population <= 0 or n_subjects <= 0:
        raise ExecutionPlanningError("Chunk profiles need a valid positive population.")
    padded = (
        -(-population // n_subjects) * n_subjects if group_sizes is None else population
    )
    inventory = ChunkProfileInventory(runtime=runtime)
    devices = runtime.subject_devices
    subject = subject_operand_sharding(devices=devices)
    shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
    scalar_int = jax.ShapeDtypeStruct((), np.dtype(np.int32), sharding=shared)
    subject_int = jax.ShapeDtypeStruct(
        (n_subjects,), np.dtype(np.int32), sharding=subject
    )
    full_int = jax.ShapeDtypeStruct((padded,), np.dtype(np.int32), sharding=subject)
    rows = (
        None
        if group_sizes is None
        else jax.ShapeDtypeStruct((n_subjects,), np.dtype(np.int32), sharding=shared)
    )
    initial = {
        name: jax.ShapeDtypeStruct(
            (n_subjects, *array.shape[1:]), array.dtype, sharding=subject
        )
        for name, array in initial_conditions.items()
        if name not in {"regime_id", "own_stakeholder"}
    }
    # These full-population products outlive every chunk. Setup's compiler peak
    # still includes its temporary age-validity summaries.
    roles = _profile_population_roles(
        inventory=inventory,
        regimes=regimes,
        initial_conditions=initial_conditions,
        full_int=full_int,
        scalar_int=scalar_int,
    )
    entry_periods, _, _ = cast(
        "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
        inventory.operation(
            function=population_operations.starting_periods,
            arguments={
                "initial_ages": jax.ShapeDtypeStruct(
                    (padded,), initial_conditions["age"].dtype, sharding=shared
                ),
                "age_values": ages.values,
            },
            subject_arg_names=("initial_ages",),
        ),
    )
    permanent = payload_bytes(tree=(roles, entry_periods))
    setup = dict(permanent)
    key = _profile_entry_key(
        inventory=inventory,
        shared=shared,
        impl=prng_impl or jax.config.jax_default_prng_impl,
    )
    taste_key = (
        _profile_entry_key(inventory=inventory, shared=shared, impl=taste_prng_impl)
        if independent_taste
        else None
    )
    carrier = _profile_initial_carrier(
        inventory=inventory, initial=initial, regimes=regimes
    )
    forward_regimes = forward_regimes_by_period(
        regimes=regimes, n_periods=ages.n_periods
    )
    with CompilationWave(
        n_workers=_resolve_compilation_workers(
            max_compilation_workers=max_compilation_workers
        ),
        logger=get_logger(log_level=log_level, runtime_checks=runtime_checks),
        log_kernel_memory=False,
        compiled={},
        compile_phase="simulation_compilation",
    ) as wave:
        for period in range(ages.n_periods):
            for name, regime in forward_regimes[period].items():
                if period in regime.simulation.exit_queries:
                    continue
                profile_forward_unit(
                    runtime=runtime,
                    regimes=regimes,
                    regime=regime,
                    name=name,
                    period=period,
                    flat_params=flat_params,
                    base=base_spaces[name],
                    base_spaces=base_spaces,
                    values=values,
                    ages=ages,
                    n_subjects=n_subjects,
                    widths=widths,
                    columns=carrier[name],
                    ordinary_key=key,
                    taste_key=taste_key,
                    policy=(policies or {}).get(period, {}).get(name),
                    projected_forward_inputs=projected_forward_inputs,
                    current_iid_replay_input=current_iid_replay_inputs.get(
                        period, {}
                    ).get(name),
                    wave=wave,
                    current_iid_override_schema=current_iid_override_schema,
                )
    regime_ids, own_roles = cast(
        "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
        inventory.operation(
            function=_empty_subject_membership,
            arguments={
                "initial_regime_ids": subject_int,
                "initial_own_stakeholder": subject_int,
            },
            subject_arg_names=("initial_regime_ids", "initial_own_stakeholder"),
            subject_outputs=True,
        ),
    )
    add_bytes(
        target=permanent,
        source=payload_bytes(tree=(initial, subject_int, subject_int, subject_int)),
    )
    inventory.close_unit()
    published: dict[jax.Device, int] = {}
    records: list[ShapeDtypePytree] = []
    maximum_period: dict[jax.Device, int] = {}
    for period in range(ages.n_periods):
        period_values: list[jax.ShapeDtypeStruct] = []
        period_masks: list[jax.ShapeDtypeStruct] = []
        inventory.operation(
            function=_period_age,
            arguments={"values": ages.values, "period": scalar_int},
        )
        regime_ids, own_roles = cast(
            "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
            inventory.operation(
                function=_activate_subject_membership,
                arguments={
                    "period": scalar_int,
                    "starting_periods": subject_int,
                    "initial_regime_ids": subject_int,
                    "initial_own_stakeholder": subject_int,
                    "regime_ids": regime_ids,
                    "own_stakeholder": own_roles,
                },
                subject_arg_names=(
                    "starting_periods",
                    "initial_regime_ids",
                    "initial_own_stakeholder",
                    "regime_ids",
                    "own_stakeholder",
                ),
                subject_outputs=True,
            ),
        )
        entries = inventory.operation(
            function=empty_gated_entries,
            arguments={"regime_ids": regime_ids},
            subject_arg_names=("regime_ids",),
            subject_outputs=True,
        )
        carrier, regime_ids, own_roles, prepared_exits = profile_exit_period(
            inventory=inventory,
            regimes=regimes,
            period=period,
            states=carrier,
            regime_ids=regime_ids,
            roles=own_roles,
            entered_through=regime_ids,
            source_roles=own_roles,
            regime_names_to_ids=regime_names_to_ids,
            flat_params=flat_params,
            values=values,
            surfaces=arrival_surfaces,
            ages=ages,
            key=key,
            taste_key=taste_key,
            n_subjects=n_subjects,
            widths=widths,
            current_iid_override_schema=current_iid_override_schema,
        )
        pending = payload_bytes(
            tree=(
                carrier,
                regime_ids,
                own_roles,
                entries,
                key,
                taste_key,
                prepared_exits,
            )
        )
        inventory.close_unit()
        new_ids = regime_ids
        for name, regime in forward_regimes[period].items():
            cores = profile_forward_unit(
                runtime=runtime,
                regimes=regimes,
                regime=regime,
                name=name,
                period=period,
                flat_params=flat_params,
                base=base_spaces[name],
                base_spaces=base_spaces,
                values=values,
                ages=ages,
                n_subjects=n_subjects,
                widths=widths,
                columns=carrier[name],
                ordinary_key=key,
                taste_key=taste_key,
                policy=(policies or {}).get(period, {}).get(name),
                projected_forward_inputs=projected_forward_inputs,
                current_iid_replay_input=current_iid_replay_inputs.get(period, {}).get(
                    name
                ),
                current_iid_override_schema=current_iid_override_schema,
                prepared_decision=prepared_exits.get(name),
            )
            mask = inventory.operation(
                function=_regime_mask,
                arguments={"regime_ids": regime_ids, "regime_id": scalar_int},
                subject_arg_names=("regime_ids",),
                subject_outputs=True,
            )
            if name not in prepared_exits and regime.has_taste_shocks:
                _profile_taste(
                    inventory=inventory,
                    key=key,
                    taste_key=taste_key,
                    width=n_subjects,
                    scalar_int=scalar_int,
                )
            if (
                name not in prepared_exits
                and regime.simulation.current_iid_draw_plan is not None
            ):
                for _ in regime.simulation.current_iid_draw_plan.physical_names:
                    inventory.operation(
                        function=draw_window_keys,
                        arguments={
                            "key": key,
                            "address_words": jax.ShapeDtypeStruct(
                                (8,), np.uint32, sharding=shared
                            ),
                            "subject_start": jax.ShapeDtypeStruct(
                                (2,), np.uint32, sharding=shared
                            ),
                            "last_real_subject": jax.ShapeDtypeStruct(
                                (2,), np.uint32, sharding=shared
                            ),
                            "period": jax.ShapeDtypeStruct(
                                (), np.uint32, sharding=shared
                            ),
                        },
                        static_arguments={"subject_count": n_subjects},
                        subject_outputs=True,
                    )
            physical = {}
            if name in prepared_exits:
                indices, value, physical = prepared_exits[name]
                inventory.operation(
                    function=_first_empty_action_row,
                    arguments={"indices": indices, "in_regime": mask},
                    subject_arg_names=("indices", "in_regime"),
                )
                decoder = cores["action_decode"]
                actions = inventory.compiled(
                    name="_lookup_values_from_indices",
                    executable=decoder.executable,
                    memory=decoder.memory,
                    arguments=decoder.arguments,
                )
                fallback = inventory.operation(
                    function=_empty_fallback,
                    arguments={"mask": mask},
                    subject_arg_names=("mask",),
                    subject_outputs=True,
                )
            elif regime.simulation.uses_case_u_reconstruction:
                from _lcm.simulation.case_u_replay import (  # noqa: PLC0415
                    case_u_profile_output,
                    case_u_status,
                )

                replayed = case_u_profile_output(
                    output=_record_core(
                        inventory=inventory,
                        profile=cores["decision"],
                        family="decision",
                    ),
                )
                inventory.operation(
                    function=case_u_status,
                    arguments={
                        "law_invalid": replayed.law_invalid,
                        "invalid": replayed.invalid,
                        "feasible": replayed.feasible,
                        "in_regime": mask,
                    },
                    subject_arg_names=(
                        "law_invalid",
                        "invalid",
                        "feasible",
                        "in_regime",
                    ),
                )
                actions, value, physical = (
                    replayed.actions,
                    replayed.value,
                    replayed.physical,
                )
                fallback = inventory.operation(
                    function=_empty_fallback,
                    arguments={"mask": mask},
                    subject_arg_names=("mask",),
                    subject_outputs=True,
                )
            elif regime.simulation.case_r_replay_spec is not None:
                from _lcm.simulation.case_r_replay import (  # noqa: PLC0415
                    case_r_profile_output,
                    case_r_status,
                )

                replayed = case_r_profile_output(
                    output=_record_core(
                        inventory=inventory,
                        profile=cores["decision"],
                        family="decision",
                    ),
                )
                inventory.operation(
                    function=case_r_status,
                    arguments={
                        "law_invalid": replayed.law_invalid,
                        "invalid": replayed.invalid,
                        "feasible": replayed.feasible,
                        "in_regime": mask,
                    },
                    subject_arg_names=(
                        "law_invalid",
                        "invalid",
                        "feasible",
                        "in_regime",
                    ),
                )
                actions, value, physical = (
                    replayed.actions,
                    replayed.value,
                    replayed.physical,
                )
                fallback = inventory.operation(
                    function=_empty_fallback,
                    arguments={"mask": mask},
                    subject_arg_names=("mask",),
                    subject_outputs=True,
                )
            elif regime.simulation.replay_route.consumer_route == "nnbegm_finite":
                _record_core(
                    inventory=inventory,
                    profile=cores["policy_prepare"],
                    family="policy_prepare",
                )
                rank_outputs = cast(
                    "tuple[ShapeDtypePytree, ...]",
                    _record_core(
                        inventory=inventory,
                        profile=cores["decision"],
                        family="policy_rank",
                    ),
                )
                actions, value, fallback = cast("_FiniteRankOutput", rank_outputs[:3])
            else:
                if "policy_first" in cores:
                    inventory.operation(
                        function=_active_subjects,
                        arguments={"in_regime": mask, "genuine": mask},
                        subject_arg_names=("in_regime", "genuine"),
                        subject_outputs=True,
                    )
                    _record_core(
                        inventory=inventory,
                        profile=cores["policy_first"],
                        family="policy_first",
                    )
                decision = cores["decision"]
                decision_outputs = cast(
                    "tuple[jax.ShapeDtypeStruct, ...]",
                    _record_core(
                        inventory=inventory, profile=decision, family="decision"
                    ),
                )
                indices, value = cast(
                    "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
                    decision_outputs[:2],
                )
                if regime.simulation.current_iid_draw_plan is not None:
                    physical = decision_outputs[2]
                    inventory.operation(
                        function=_current_iid_invalid,
                        arguments={"invalid": decision_outputs[3], "in_regime": mask},
                        subject_arg_names=("invalid", "in_regime"),
                    )
                if not base_spaces[name].states and indices.ndim == 0:
                    indices, value = cast(
                        "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
                        inventory.operation(
                            function=_broadcast_collective,
                            arguments={"indices": indices, "value": value},
                            static_arguments={"n_subjects": n_subjects},
                            subject_outputs=True,
                        ),
                    )
                inventory.operation(
                    function=_first_empty_action_row,
                    arguments={"indices": indices, "in_regime": mask},
                    subject_arg_names=("indices", "in_regime"),
                )
                decoder = decision.action_decoder
                if decoder is None:
                    raise ExecutionPlanningError(
                        "A decision chunk profile omitted its action decoder."
                    )
                actions = inventory.compiled(
                    name="_lookup_values_from_indices",
                    executable=decoder.executable,
                    memory=decoder.memory,
                    arguments=decoder.arguments,
                )
                if not value.shape:
                    value = cast(
                        "jax.ShapeDtypeStruct",
                        inventory.operation(
                            function=_broadcast_value,
                            arguments={"value": value},
                            static_arguments={"n_subjects": n_subjects},
                            subject_outputs=True,
                        ),
                    )
                fallback = inventory.operation(
                    function=_empty_fallback,
                    arguments={"mask": mask},
                    subject_arg_names=("mask",),
                    subject_outputs=True,
                )
            record = (
                value,
                actions,
                carrier[name],
                mask,
                own_roles,
                fallback,
                physical,
            )
            records.append(record)
            period_values.append(value)
            period_masks.append(cast("jax.ShapeDtypeStruct", mask))
            add_bytes(target=published, source=payload_bytes(tree=record))
            if not regime.terminal:
                carrier, key, new_ids, own_roles = _profile_next_subjects(
                    inventory=inventory,
                    regime=regime,
                    name=name,
                    period=period,
                    cores=cores,
                    carrier=carrier,
                    mask=cast("jax.ShapeDtypeStruct", mask),
                    key=key,
                    new_ids=new_ids,
                    own_roles=own_roles,
                    flat_params=flat_params,
                    values=values,
                    age=jax.ShapeDtypeStruct((), ages.values.dtype, sharding=shared),
                    widths=widths,
                    scalar_int=scalar_int,
                    regime_names_to_ids=regime_names_to_ids,
                    population=padded,
                    original_population=original_population,
                    width=n_subjects,
                    subject_rows=rows,
                    runtime_checks=runtime_checks,
                    regimes=regimes,
                    base_spaces=base_spaces,
                )
            # The real period owner retains every committed carry until finish.
            add_bytes(
                target=pending,
                source=payload_bytes(tree=(carrier, new_ids, own_roles, key)),
            )
            inventory.close_unit()
        for binding in diagnostic_bindings(
            values=tuple(period_values),
            in_regime=tuple(period_masks),
            prev_regime_ids=regime_ids,
            new_regime_ids=new_ids,
            sorted_ids=tuple(
                sorted(int(code) for code in regime_names_to_ids.values())
            ),
            log_level=log_level,
            runtime_checks=runtime_checks,
        ):
            inventory.operation(
                function=binding.function,
                arguments=binding.arguments,
                subject_arg_names=binding.subject_arg_names,
                static_arguments=binding.static_arguments,
                subject_outputs=binding.subject_outputs,
            )
        inventory.close_unit()
        add_bytes(
            target=pending,
            source=_period_copy_reservation(
                regimes=forward_regimes[period],
                values=values,
                policies=policies,
                current_iid_replay_inputs=current_iid_replay_inputs,
                projected_forward_inputs=projected_forward_inputs,
                arrival_surfaces=arrival_surfaces,
                period=period,
                devices=devices,
            ),
        )
        maximum_bytes(target=maximum_period, source=pending)
        regime_ids = new_ids
    add_bytes(target=permanent, source=maximum_period)
    host_stages, output_bank, padded_inputs = _profile_outer_storage(
        inventory=inventory,
        initial_conditions=initial_conditions,
        records=records,
        published=published,
        population=padded,
        original_population=original_population,
        width=n_subjects,
        roles=roles,
        entry_periods=entry_periods,
        group_sizes=group_sizes,
    )
    add_bytes(target=permanent, source=padded_inputs)
    add_bytes(target=setup, source=padded_inputs)
    add_bytes(target=permanent, source=inventory.maximum_unit)
    return SimulationChunkProfile(
        n_subjects=n_subjects,
        padded_population=(
            padded
            if group_sizes is None
            else grouped_rows(sizes=group_sizes, width=n_subjects)
        ),
        stages=tuple(inventory.stages),
        fixed_reservation=MappingProxyType(permanent),
        output_reservation=MappingProxyType(output_bank),
        host_stages=host_stages,
        axis_widths=MappingProxyType(dict(widths)),
        setup_reservation=MappingProxyType(setup),
    )


def _profile_next_subjects(
    *,
    inventory: ChunkProfileInventory,
    regime: Regime,
    name: str,
    period: int,
    cores: Mapping[str, ForwardProgramProfile],
    carrier: Mapping[str, Mapping[str, jax.ShapeDtypeStruct]],
    mask: jax.ShapeDtypeStruct,
    key: jax.ShapeDtypeStruct,
    new_ids: jax.ShapeDtypeStruct,
    own_roles: jax.ShapeDtypeStruct,
    flat_params: FlatParams,
    values: Mapping[int, Mapping[str, jax.Array]],
    age: jax.ShapeDtypeStruct,
    widths: Mapping[str, int],
    scalar_int: jax.ShapeDtypeStruct,
    regime_names_to_ids: RegimeNamesToIds,
    population: int,
    original_population: int,
    width: int,
    subject_rows: jax.ShapeDtypeStruct | None = None,
    runtime_checks: bool = True,
    regimes: Mapping[str, Regime] = MappingProxyType({}),
    base_spaces: Mapping[str, StateActionSpace] = MappingProxyType({}),
) -> tuple[
    Mapping[str, Mapping[str, jax.ShapeDtypeStruct]],
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
]:
    """Advance state, membership and ordinary-key metadata in actual dispatch order."""
    stochastic_names = tuple(
        binding.removeprefix("key_")
        for binding in _stochastic_keys(regime=regime, key=key)
    )
    _profile_keys(
        inventory=inventory,
        key=key,
        names=stochastic_names,
        population=population,
        original_population=original_population,
        width=width,
        scalar_int=scalar_int,
        rows=subject_rows,
        addresses=transition_random_addresses(
            regime=regime, regimes=regimes, names=stochastic_names
        ),
    )
    raw = (
        _record_core(
            inventory=inventory,
            profile=cores["transition"],
            family="transition",
        )
        if "transition" in cores
        else {}
    )
    next_states = {
        target: {label.removeprefix("next_"): leaf for label, leaf in outputs.items()}
        for target, outputs in cast(
            "Mapping[str, Mapping[str, ShapeDtypePytree]]", raw
        ).items()
    }
    advanced = inventory.operation(
        function=_advance_states_for_subjects,
        arguments={
            "states_per_regime": carrier,
            "next_states_per_regime": next_states,
            "subject_indices": mask,
        },
        subject_arg_names=(
            "states_per_regime",
            "next_states_per_regime",
            "subject_indices",
        ),
    )
    route = cast(
        "Mapping[str, jax.ShapeDtypeStruct]",
        _record_core(inventory=inventory, profile=cores["route"], family="route"),
    )
    targets = regime.simulation.reachability.targets(period=period, source=name)
    if runtime_checks:
        inventory.operation(
            function=regime_probability_flags,
            arguments={"probabilities": tuple(route.values()), "rows": mask},
            static_arguments={
                "inactive_indices": regime_probability_inactive_indices(
                    names=tuple(route), active_regimes_next_period=targets
                )
            },
        )
    names = sorted(
        (target for target in targets if target in route),
        key=lambda target: int(regime_names_to_ids[target]),
    )
    if not names:
        raise ExecutionPlanningError(
            "A simulation chunk profile has no active regime draw target."
        )
    _, draw_keys = _profile_keys(
        inventory=inventory,
        key=key,
        names=("regime_transition",),
        population=population,
        original_population=original_population,
        width=width,
        scalar_int=scalar_int,
        rows=subject_rows,
        addresses={"regime_transition": RandomAddress(kind="regime", name="regime")},
    )
    rows = tuple(route[target] for target in names)
    drawn = inventory.operation(
        function=_draw_random_regime_ids_from_scalars,
        arguments={
            "keys": draw_keys["key_regime_transition"],
            "prob_rows": rows,
            "regime_id_scalars": tuple(scalar_int for _ in names),
        },
        subject_arg_names=("keys", "prob_rows") if rows[0].shape else ("keys",),
    )
    new_ids = cast(
        "jax.ShapeDtypeStruct",
        inventory.operation(
            function=_update_regime_ids,
            arguments={
                "subjects_in_regime": mask,
                "next_regime_ids": drawn,
                "new_subject_regime_ids": new_ids,
            },
            subject_arg_names=(
                "subjects_in_regime",
                "next_regime_ids",
                "new_subject_regime_ids",
            ),
        ),
    )
    if runtime_checks:
        checks = drawn_code_checks(
            next_state_names={
                target: tuple(states) for target, states in next_states.items()
            },
            regimes=regimes,
            targets=tuple(names),
        )
        if checks:
            inventory.operation(
                function=_count_drawn_off_grid_codes,
                arguments={
                    "codes": tuple(
                        next_states[target][state] for target, state, _ in checks
                    ),
                    "n_codes": tuple(scalar_int for _ in checks),
                    "target_ids": tuple(scalar_int for _ in checks),
                    "subjects": mask,
                    "new_regime_ids": new_ids,
                },
                subject_arg_names=("codes", "subjects", "new_regime_ids"),
            )
    drawn_arrival_ids = new_ids
    carrier = cast("Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]", advanced)
    if period in regime.simulation.programs.gate_route:
        shared = simulation_value_sharding(
            stored_sharding=mask.sharding,
            devices=inventory.runtime.subject_devices,
        )
        edge_values = MappingProxyType(
            {
                target: MappingProxyType(
                    {
                        reference: values[period + 1][reference]
                        for reference in dict.fromkeys(
                            (target, *edge.reference_regimes)
                        )
                    }
                )
                for target, edge in regime.gated_edges.items()
                if target in values.get(period + 1, {})
            }
        )
        gate_profile = _profile_program(
            runtime=inventory.runtime,
            program=regime.simulation.programs.gate_route[period],
            arguments=gate_route_arguments(
                edge_values=abstract_tree(tree=edge_values, sharding=shared),
                next_states=gated_route_candidates(regime=regime, next_states=carrier),
                new_subject_regime_ids=new_ids,
                subjects_in_regime=mask,
                flat_params=abstract_tree(tree=flat_params, sharding=shared),
                own_stakeholder=own_roles,
                new_own_stakeholder=own_roles,
                fold_age=age,
            ),
            period=period,
            n_subjects=mask.shape[0],
            widths=widths,
        )
        route_delta, new_ids, own_roles = cast(
            "_GateRouteOutput",
            _record_core(
                inventory=inventory,
                profile=gate_profile,
                family="gate_route",
            ),
        )
        carrier = cast(
            "Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]",
            inventory.operation(
                function=commit_gated_route_delta,
                arguments={"delta": route_delta, "next_states": carrier},
                subject_arg_names=("delta", "next_states"),
                subject_outputs=True,
            ),
        )
        inventory.operation(
            function=record_open_entries,
            arguments={
                "closed_masks": tuple(
                    cast("Mapping[str, jax.ShapeDtypeStruct]", edge_delta)["closed"]
                    for edge_delta in route_delta.values()
                ),
                "routed_ids": new_ids,
                "subjects_in_regime": mask,
                "new_entered_through": new_ids,
            },
            subject_arg_names=(
                "closed_masks",
                "routed_ids",
                "subjects_in_regime",
                "new_entered_through",
            ),
            static_arguments={
                "source_id": int(regime_names_to_ids[name]),
                "target_ids": tuple(
                    int(regime_names_to_ids[target]) for target in route_delta
                ),
            },
            subject_outputs=True,
        )
    arrival_targets = tuple(
        target
        for target in names
        if period + 1 in regimes[target].simulation.exit_queries
        and name in regimes[target].simulation.exit_queries[period + 1].arrivals
    )
    if arrival_targets:
        inventory.operation(
            function=record_arrival_entries,
            arguments={
                "drawn_ids": drawn_arrival_ids,
                "routed_ids": new_ids,
                "holders": mask,
                "entries": new_ids,
            },
            static_arguments={
                "source_id": int(regime_names_to_ids[name]),
                "target_ids": tuple(
                    int(regime_names_to_ids[target]) for target in arrival_targets
                ),
            },
            subject_arg_names=("drawn_ids", "routed_ids", "holders", "entries"),
            subject_outputs=True,
        )
    carrier = _profile_clip_next_states(
        inventory=inventory,
        regime=regime,
        regimes=regimes,
        base_spaces=base_spaces,
        targets=targets,
        carrier=carrier,
        mask=mask,
        new_ids=new_ids,
        scalar_int=scalar_int,
        regime_names_to_ids=regime_names_to_ids,
        next_period=period + 1,
        runtime_checks=runtime_checks,
    )
    return carrier, key, new_ids, own_roles


def _profile_clip_next_states(
    *,
    inventory: ChunkProfileInventory,
    regime: Regime,
    regimes: Mapping[str, Regime],
    base_spaces: Mapping[str, StateActionSpace],
    targets: tuple[RegimeName, ...],
    carrier: Mapping[str, Mapping[str, jax.ShapeDtypeStruct]],
    mask: jax.ShapeDtypeStruct,
    new_ids: jax.ShapeDtypeStruct,
    scalar_int: jax.ShapeDtypeStruct,
    regime_names_to_ids: RegimeNamesToIds,
    next_period: int,
    runtime_checks: bool,
) -> Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]:
    """Profile `clip_next_states`: realized off-grid counts, then economic clipping.

    The targets are the realized draw's and every gated fallback's, as the
    simulation passes them; a count runs where runtime checks or a raise rule
    ask for one, and the clip where any grid declares `off_grid="clip_state"`.
    """
    shared = simulation_value_sharding(
        stored_sharding=mask.sharding, devices=inventory.runtime.subject_devices
    )
    points: dict[str, dict[str, ShapeDtypePytree]] = {}
    for target in dict.fromkeys(
        (
            *targets,
            *(
                leg.realized_fallback.regime
                for edge in regime.gated_edges.values()
                for leg in edge.legs
            ),
        )
    ):
        if target not in carrier or target not in regimes:
            continue
        target_regime = regimes[target]
        axes = target_regime.solution.period_state_axes
        state_points = {
            **(base_spaces[target].states if target in base_spaces else {}),
            **({} if axes is None else axes.get(next_period, {})),
        }
        for state, values in carrier[target].items():
            grid = target_regime.simulation.grids.get(state)
            if not isinstance(grid, ContinuousGrid) or isinstance(
                grid, _ContinuousStochasticProcess
            ):
                continue
            check = runtime_checks or grid.off_grid == "raise"
            if not check and grid.off_grid != "clip_state":
                continue
            nodes = abstract_tree(
                tree=state_points[state] if state in state_points else grid.to_jax(),
                sharding=shared,
            )
            if check:
                inventory.operation(
                    function=_count_realized_grid_state,
                    arguments={
                        "values": values,
                        "points": nodes,
                        "subjects": mask,
                        "new_regime_ids": new_ids,
                        "target_id": scalar_int,
                    },
                    subject_arg_names=("values", "subjects", "new_regime_ids"),
                )
            if grid.off_grid == "clip_state":
                points.setdefault(target, {})[state] = nodes
    if not points:
        return carrier
    return cast(
        "Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]",
        inventory.operation(
            function=_clip_next_states,
            arguments={
                "states": carrier,
                "points": points,
                "subjects": mask,
                "new_regime_ids": new_ids,
                "regime_ids": regime_names_to_ids,
            },
            subject_arg_names=("states", "subjects", "new_regime_ids"),
            subject_outputs=True,
        ),
    )


def _profile_population_roles(
    *,
    inventory: ChunkProfileInventory,
    regimes: Mapping[str, Regime],
    initial_conditions: Mapping[str, jax.Array],
    full_int: jax.ShapeDtypeStruct,
    scalar_int: jax.ShapeDtypeStruct,
) -> jax.ShapeDtypeStruct:
    """Profile the existing supplied-role vocabulary and collective-role checks."""
    if "own_stakeholder" not in initial_conditions:
        return cast(
            "jax.ShapeDtypeStruct",
            inventory.operation(
                function=population_operations.default_roles,
                arguments={"regime_ids": full_int},
                subject_arg_names=("regime_ids",),
                subject_outputs=True,
            ),
        )
    role_ids = next(iter(regimes.values())).stakeholder_names_to_ids
    roles, _ = cast(
        "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
        inventory.operation(
            function=population_operations.canonical_roles,
            arguments={
                "declared": full_int,
                "known_role_ids": tuple(scalar_int for _ in role_ids),
            },
            subject_arg_names=("declared",),
        ),
    )
    for regime in regimes.values():
        if regime.stakeholders is not None:
            inventory.operation(
                function=population_operations.role_mismatch,
                arguments={
                    "roles": roles,
                    "regime_ids": full_int,
                    "regime_id": scalar_int,
                    "role_ids": tuple(scalar_int for _ in regime.stakeholders),
                },
                subject_arg_names=("roles", "regime_ids"),
            )
    return roles


# CPU retention and GPU offload have distinct assembly/trim ownership branches.
def _profile_outer_storage(  # noqa: C901, PLR0912
    *,
    inventory: ChunkProfileInventory,
    initial_conditions: Mapping[str, jax.Array],
    records: list[ShapeDtypePytree],
    published: Mapping[jax.Device, int],
    population: int,
    original_population: int,
    width: int,
    roles: jax.ShapeDtypeStruct,
    entry_periods: jax.ShapeDtypeStruct,
    group_sizes: tuple[int, ...] | None = None,
) -> tuple[
    tuple[SimulationStageProfile, ...], dict[jax.Device, int], dict[jax.Device, int]
]:
    """Profile exact entry padding/windows and every final record concatenate/trim.

    Grouped by an invariant state, every chunk gathers its original rows, every
    group's chunks are concatenated, and one gather restores original order.

    CPU execution keeps all completed chunks and their assembled/trimmed outputs.
    GPU execution keeps one compute chunk while CPU assembly is separately profiled
    and explicitly outside its compute-device ceiling. Neither route assumes future
    output aliases from abstract descriptor identity.
    """
    devices = inventory.runtime.subject_devices
    scalar = jax.ShapeDtypeStruct(
        (), np.dtype(np.int32), sharding=jax.sharding.SingleDeviceSharding(devices[0])
    )
    population_columns = {}
    padded_inputs: dict[jax.Device, int] = {}
    for name, array in initial_conditions.items():
        descriptor = array
        if array.shape[0] < population:
            descriptor = inventory.operation(
                function=_pad_initial_leaf,
                arguments={"array": array},
                static_arguments={"pad": population - array.shape[0]},
                devices=(devices[0],),
            )
            add_bytes(target=padded_inputs, source=payload_bytes(tree=descriptor))
        population_columns[name] = descriptor
    population_columns.update({"__roles__": roles, "__entry_periods__": entry_periods})
    if width < population or group_sizes is not None:
        for column in population_columns.values():
            if (
                not isinstance(column, jax.Array | jax.ShapeDtypeStruct)
                or column.sharding is None
            ):
                raise ExecutionPlanningError(
                    "Population windows need actual source layouts."
                )
            source_devices = tuple(
                device for device in devices if device in column.sharding.device_set
            )
            inventory.operation(
                function=_slice_population if group_sizes is None else _take_population,
                arguments=(
                    {"array": column, "start": scalar}
                    if group_sizes is None
                    else {
                        "array": column,
                        "rows": jax.ShapeDtypeStruct(
                            (width,), np.dtype(np.int32), sharding=scalar.sharding
                        ),
                    }
                ),
                static_arguments={"width": width} if group_sizes is None else {},
                subject_arg_names=("array",) if len(source_devices) > 1 else (),
                subject_outputs=len(source_devices) > 1,
                devices=source_devices,
            )
    inventory.close_unit()
    chunks = (
        population // width
        if group_sizes is None
        else grouped_rows(sizes=group_sizes, width=width) // width
    )
    multiple = chunks > 1
    host = chunk_host_device(subject_devices=devices)
    gpu_offload = multiple and all(device.platform == "gpu" for device in devices)
    final_inventory = (
        ChunkProfileInventory(runtime=inventory.runtime) if gpu_offload else inventory
    )
    output_bank: dict[jax.Device, int] = {}
    if multiple:
        # This destination bank survives every later chunk on CPU. The source
        # chunk and one payload scratch per copy remain charged on compute devices.
        if not gpu_offload:
            total = sum(
                leaf.dtype.itemsize * math.prod(leaf.shape)
                for record in records
                for leaf in jax.tree.leaves(record)
            )
            output_bank[host] = chunks * total
            add_bytes(target=inventory.maximum_unit, source=published)
        else:
            add_bytes(target=output_bank, source=published)
        add_bytes(target=inventory.maximum_unit, source=published)
    else:
        add_bytes(target=output_bank, source=published)
    for record in records:
        for record_leaf in jax.tree.leaves(record):
            leaf = record_leaf
            if multiple:
                local = cast(
                    "jax.ShapeDtypeStruct",
                    abstract_tree(
                        tree=leaf, sharding=jax.sharding.SingleDeviceSharding(host)
                    ),
                )
                leaf = final_inventory.operation(
                    function=_concatenate_arrays,
                    arguments={"arrays": tuple(local for _ in range(chunks))},
                    subject_arg_names=("arrays",),
                    devices=(host,),
                )
            if group_sizes is not None:
                executing = (host,) if multiple else devices
                final_inventory.operation(
                    function=_take_rows,
                    arguments={
                        "array": leaf,
                        "rows": jax.ShapeDtypeStruct(
                            (original_population,),
                            np.dtype(np.int32),
                            sharding=jax.sharding.SingleDeviceSharding(executing[0]),
                        ),
                    },
                    subject_arg_names=("array",),
                    devices=executing,
                )
            elif original_population < population:
                executing = (host,) if multiple else devices
                final_inventory.operation(
                    function=_slice_array,
                    arguments={"array": leaf},
                    subject_arg_names=("array",),
                    static_arguments={"start": 0, "stop": original_population},
                    devices=executing,
                )
    final_inventory.close_unit()
    return (
        (tuple(final_inventory.stages) if gpu_offload else ()),
        output_bank,
        padded_inputs,
    )


def _record_core(
    *, inventory: ChunkProfileInventory, profile: ForwardProgramProfile, family: str
) -> ShapeDtypePytree:
    """Use the actual resolved core's compiler output metadata."""
    return inventory.compiled(
        name=f"core:{family}",
        executable=profile.executable,
        memory=profile.memory,
        arguments=profile.arguments,
    )


def _profile_initial_carrier(
    *,
    inventory: ChunkProfileInventory,
    initial: Mapping[str, jax.ShapeDtypeStruct],
    regimes: Mapping[str, Regime],
) -> Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]:
    """Profile the same cast/fill branches as the canonical carrier writer."""
    template = next(iter(initial.values()))
    result = {}
    for name, regime in regimes.items():
        columns = {}
        for state in regime.simulation.state_names:
            discrete = isinstance(regime.simulation.grids[state], DiscreteGrid)
            dtype = np.dtype(jnp.int32 if discrete else canonical_float_dtype())
            if state not in initial:
                column = inventory.operation(
                    function=_fill_carrier,
                    arguments={"template": template},
                    subject_arg_names=("template",),
                    static_arguments={
                        "dtype": dtype.str,
                        "fill_value": MISSING_CAT_CODE if discrete else float("nan"),
                    },
                    subject_outputs=True,
                )
            else:
                column = inventory.operation(
                    function=_cast_carrier,
                    arguments={"value": initial[state]},
                    subject_arg_names=("value",),
                    static_arguments={"dtype": dtype.str},
                    subject_outputs=True,
                )
            columns[state] = cast("jax.ShapeDtypeStruct", column)
        result[name] = MappingProxyType(columns)
    return MappingProxyType(result)


def _profile_keys(
    *,
    inventory: ChunkProfileInventory,
    key: jax.ShapeDtypeStruct,
    names: tuple[str, ...],
    population: int,
    original_population: int,
    width: int,
    scalar_int: jax.ShapeDtypeStruct,
    rows: jax.ShapeDtypeStruct | None = None,
    addresses: Mapping[str, RandomAddress] = MappingProxyType({}),
) -> tuple[jax.ShapeDtypeStruct, Mapping[str, jax.ShapeDtypeStruct]]:
    """Profile the addressed innovations at a contiguous or grouped row window."""
    static_arguments: dict[str, StaticArgument] = {
        "names": names,
        "n_initial_states": population,
        "original_n_subjects": original_population,
        "partitionable": jax.config.jax_threefry_partitionable,
        "addresses": tuple(
            addresses.get(name, RandomAddress(kind="state", name=name)).words()
            for name in names
        ),
    }
    period = jax.ShapeDtypeStruct((), np.uint32, sharding=scalar_int.sharding)
    if rows is None:
        function = _generate_windowed_simulation_keys
        arguments = {
            "key": key,
            "start": jax.ShapeDtypeStruct(
                (2,), np.uint32, sharding=scalar_int.sharding
            ),
            "period": period,
        }
        static_arguments["width"] = width
    else:
        function = _generate_rowed_simulation_keys
        static_arguments.pop("n_initial_states")
        static_arguments.pop("original_n_subjects")
        arguments = {
            "key": key,
            "rows": jax.ShapeDtypeStruct((width, 2), np.uint32, sharding=rows.sharding),
            "period": period,
        }
    return key, cast(
        "Mapping[str, jax.ShapeDtypeStruct]",
        inventory.operation(
            function=function,
            arguments=arguments,
            static_arguments=static_arguments,
            subject_arg_names=() if rows is None else ("rows",),
            subject_outputs=True,
        ),
    )


def _profile_taste(
    *,
    inventory: ChunkProfileInventory,
    key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
    width: int,
    scalar_int: jax.ShapeDtypeStruct,
) -> None:
    """Include the exact ordinary or independent taste-key body and carry."""
    shared = scalar_int.sharding
    inventory.operation(
        function=draw_taste_shock_keys,
        arguments={
            "key": key if taste_key is None else taste_key,
            "address_words": jax.ShapeDtypeStruct(
                (8,), np.dtype(np.uint32), sharding=shared
            ),
            "subject_start": jax.ShapeDtypeStruct(
                (2,), np.dtype(np.uint32), sharding=shared
            ),
            "last_real_subject": jax.ShapeDtypeStruct(
                (2,), np.dtype(np.uint32), sharding=shared
            ),
            "period": jax.ShapeDtypeStruct((), np.uint32, sharding=shared),
        },
        static_arguments={"subject_count": width},
        subject_outputs=True,
    )


def _period_copy_reservation(
    *,
    regimes: Mapping[str, Regime],
    values: Mapping[int, Mapping[str, jax.Array]],
    period: int,
    devices: tuple[jax.Device, ...],
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None = None,
    current_iid_replay_inputs: CurrentIIDReplayInputs = MappingProxyType({}),
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
    arrival_surfaces: PeriodToRegimeToArrivalSurfaces = MappingProxyType({}),
) -> dict[jax.Device, int]:
    """Reserve declared nonaligned copies once per period/address/ordered layout."""
    policy_sources = _policy_read_sources(
        policies=(policies or {}).get(period, {}), period=period
    )
    from _lcm.egm.current_iid_replay_inputs import (  # noqa: PLC0415
        native_replay_payload,
        native_replay_program,
    )

    for replay in current_iid_replay_inputs.get(period, {}).values():
        leaves = jax.tree.leaves(native_replay_payload(replay=replay))
        if not all(isinstance(leaf, jax.Array) for leaf in leaves):
            raise ExecutionPlanningError(
                "Case-R copy profiles require canonical JAX rows."
            )
        policy_sources.update(
            (read.target, leaf)
            for read, leaf in zip(
                replay.reads(core=native_replay_program(replay=replay)),
                leaves,
                strict=True,
            )
        )
    reads = tuple(
        read
        for regime in regimes.values()
        for family in (
            regime.simulation.programs.policy_prepare,
            regime.simulation.programs.forward_decision,
            regime.simulation.programs.gate_fold,
            regime.simulation.programs.gate_route,
            regime.simulation.programs.exit_full,
            regime.simulation.programs.exit_participating,
            *regime.simulation.programs.arrival_families,
        )
        if period in family
        for read in family[period].requirements.value_reads
        if read.target.kind is not ValueArtifactKind.GATED_CONTINUATION
    )
    # A grouped chunk reads one code's block of each value carrying the state;
    # every code's block has the same size.
    grouping = next(
        (regime.simulation.programs.grouping for regime in regimes.values()), None
    )
    seen = set()
    result: dict[jax.Device, int] = {}
    host = _host_outside_device_ceiling(devices=devices)
    for read in reads:
        source = _retained_read_source(
            read=read,
            values=values,
            policy_sources=policy_sources,
            projected_forward_inputs=projected_forward_inputs,
            arrival_surfaces=arrival_surfaces,
        )
        required = simulation_value_sharding(
            stored_sharding=source.sharding, devices=devices
        )
        identity = (read.target, required)
        if identity in seen:
            continue
        seen.add(identity)
        if grouping is not None:
            transfer = type_local_transfer(
                route=grouping,
                code=grouping.codes[0],
                read=read,
                stored=source,
                required_sharding=required,
            )
            if transfer.delivers_stored_buffer:
                continue
        else:
            kind = classify_value_transfer(
                stored_sharding=source.sharding, required_sharding=required
            )
            if kind is ValueTransferKind.ALIGNED_LOCAL:
                continue
            transfer = resolve_value_transfer(
                target=read.target,
                source=read.source,
                kind=kind,
                stored_template=source,
                source_sharding=required,
            )
        add_bytes(
            target=result,
            source=dict.fromkeys(required.device_set, transfer.cost.per_device_bytes),
        )
        add_bytes(
            target=result,
            source=dict.fromkeys(
                (required.device_set | source.sharding.device_set) - {host},
                transfer.cost.temporary_bytes,
            ),
        )
    return result


def _host_outside_device_ceiling(
    *, devices: tuple[jax.Device, ...]
) -> jax.Device | None:
    """Return the host CPU whose RAM stays outside these chunk devices' ceiling.

    GPU execution keeps host RAM out of the device budget, as completed-chunk
    offload does, so a source the solve retained on the host (a finite policy)
    stages its copy scratch outside it. A CPU chunk budgets its own host: `None`.
    """
    if all(device.platform == "gpu" for device in devices):
        return chunk_host_device(subject_devices=devices)
    return None


def _policy_read_sources(
    *, policies: Mapping[RegimeName, SimulationPolicy], period: int
) -> dict[ValueArtifactAddress, jax.Array]:
    """Index canonical retained finite leaves by the actual period-owner addresses."""
    sources = {}
    for name, policy in policies.items():
        if not isinstance(policy, NNBEGMSimPolicy | EGMSimPolicy):
            raise ExecutionPlanningError(
                "Chunk policy copies require a finite or flat EGM payload schema."
            )
        leaves = jax.tree.leaves(policy)
        if not all(isinstance(leaf, jax.Array) for leaf in leaves):
            raise ExecutionPlanningError(
                "Chunk policy copies require canonical JAX leaves."
            )
        reads = replay_payload_reads(
            payload=policy,
            key=SIMULATION_POLICY,
            period=period,
            regime=name,
            core="simulation_policy_replay",
        )
        sources.update(
            (read.target, leaf) for read, leaf in zip(reads, leaves, strict=True)
        )
    return sources


def _retained_read_source(
    *,
    read: ValueRead,
    values: Mapping[int, Mapping[str, jax.Array]],
    policy_sources: Mapping[ValueArtifactAddress, jax.Array],
    projected_forward_inputs: ProjectedForwardInputs = MappingProxyType({}),
    arrival_surfaces: PeriodToRegimeToArrivalSurfaces = MappingProxyType({}),
) -> jax.Array:
    """Resolve only the declared value and finite-policy storage classes."""
    if is_arrival_surface(address=read.target):
        return cast(
            "jax.Array",
            stored_arrival_surface(address=read.target, surfaces=arrival_surfaces),
        )
    if read.target.kind is ValueArtifactKind.CONTINUATION_LEAF:
        if read.target.regime not in projected_forward_inputs.get(
            read.target.period, {}
        ):
            raise ExecutionPlanningError(
                "Projected copy profile lacks its original source."
            )
        leaf = projected_forward_inputs[read.target.period][
            read.target.regime
        ].leaves()[read.target.leaf_path]
        if not isinstance(leaf, jax.Array):
            raise ExecutionPlanningError(
                "Projected copy profile requires actual retained arrays."
            )
        return leaf
    if read.target.kind is ValueArtifactKind.REGIME_VALUE:
        return values[read.target.period][read.target.regime]
    if (
        read.target.kind
        in (ValueArtifactKind.REPLAY_ARTIFACT_LEAF, ValueArtifactKind.CONTINUATION_LEAF)
        and read.target in policy_sources
    ):
        return policy_sources[read.target]
    raise ExecutionPlanningError(
        "Chunk copies require an explicit retained artifact schema for "
        f"{read.target!r}."
    )


def _profile_entry_key(
    *,
    inventory: ChunkProfileInventory,
    shared: jax.sharding.Sharding,
    impl: str,
) -> jax.ShapeDtypeStruct:
    """Describe the actual placed host seed and the selected key implementation."""
    return cast(
        "jax.ShapeDtypeStruct",
        inventory.operation(
            function=_create_simulation_key,
            arguments={
                "seed": jax.ShapeDtypeStruct(
                    (),
                    jax.dtypes.canonicalize_dtype(np.dtype(np.int64)),
                    sharding=shared,
                )
            },
            static_arguments={
                "impl": impl,
                "seed_offset": jax.config.jax_random_seed_offset,
            },
        ),
    )
