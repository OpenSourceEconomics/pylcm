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
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueTransferKind,
    classify_value_transfer,
    resolve_value_transfer,
)
from _lcm.grids import DiscreteGrid
from _lcm.simulation import (
    population_operations,
)
from _lcm.simulation.assembly import (
    _concatenate_arrays,
    _slice_array,
)
from _lcm.simulation.chunk_offload import chunk_host_device
from _lcm.simulation.chunk_operations import (
    _broadcast_collective,
    _broadcast_value,
    _empty_fallback,
    _period_age,
    _regime_mask,
    _slice_population,
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
from _lcm.simulation.diagnostic_operations import diagnostic_bindings
from _lcm.simulation.entry_allocations import _pad_initial_leaf
from _lcm.simulation.forward_program_profiles import (
    ForwardProgramProfile,
    _stochastic_keys,
    profile_forward_unit,
)
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
from _lcm.simulation.policy_diagnostics import dropped_candidate_counts
from _lcm.simulation.random import (
    _create_simulation_key,
    _generate_windowed_simulation_keys,
    _split_simulation_key,
)
from _lcm.simulation.replay_inputs import replay_payload_reads
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.taste_stream import (
    _advance_simulation_taste_key,
    draw_taste_shock_keys,
)
from _lcm.simulation.transitions import (
    _advance_states_for_subjects,
    _draw_random_regime_ids_from_scalars,
    _update_regime_ids,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.typing import FlatParams, RegimeNamesToIds
from _lcm.utils.logging import LogLevel
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import SIMULATION_POLICY

type _FiniteRankOutput = tuple[
    Mapping[str, jax.ShapeDtypeStruct], jax.ShapeDtypeStruct, jax.ShapeDtypeStruct
]


# Keep setup, per-unit publication and period cleanup in their lifetime order.
def profile_simulation_chunk(  # noqa: C901, PLR0912, PLR0915
    *,
    runtime: SimulationRuntime,
    regimes: Mapping[str, Regime],
    flat_params: FlatParams,
    base_spaces: Mapping[str, StateActionSpace],
    values: Mapping[int, Mapping[str, jax.Array]],
    ages: AgeGrid,
    initial_conditions: Mapping[str, jax.Array],
    regime_names_to_ids: RegimeNamesToIds,
    n_subjects: int,
    population: int,
    original_population: int,
    widths: Mapping[str, int],
    independent_taste: bool,
    log_level: LogLevel,
    policies: Mapping[int, Mapping[str, object]] | None = None,
) -> SimulationChunkProfile:
    """Prepare actual compiled stages for one proposed outer population extent.

    The current route covers declared grid and finite-policy decisions without
    host gated or other replay adapters. Diagnostics, retained storage and outer
    assembly profiles feed the selector before any candidate chunk is allocated.
    """
    if population < original_population or original_population <= 0 or n_subjects <= 0:
        raise ExecutionPlanningError("Chunk profiles need a valid positive population.")
    padded = -(-population // n_subjects) * n_subjects
    inventory = ChunkProfileInventory(runtime=runtime)
    devices = runtime.subject_devices
    subject = subject_operand_sharding(devices=devices)
    shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
    scalar_int = jax.ShapeDtypeStruct((), np.dtype(np.int32), sharding=shared)
    subject_int = jax.ShapeDtypeStruct(
        (n_subjects,), np.dtype(np.int32), sharding=subject
    )
    full_int = jax.ShapeDtypeStruct((padded,), np.dtype(np.int32), sharding=subject)
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
        inventory=inventory, shared=shared, impl=jax.config.jax_default_prng_impl
    )
    taste_key = (
        _profile_entry_key(inventory=inventory, shared=shared, impl="threefry2x32")
        if independent_taste
        else None
    )
    carrier = _profile_initial_carrier(
        inventory=inventory, initial=initial, regimes=regimes
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
    records: list[object] = []
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
        pending = payload_bytes(tree=(carrier, regime_ids, own_roles, key, taste_key))
        inventory.close_unit()
        new_ids = regime_ids
        for name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            cores = profile_forward_unit(
                runtime=runtime,
                regime=regime,
                name=name,
                period=period,
                flat_params=flat_params,
                base=base_spaces[name],
                values=values,
                ages=ages,
                n_subjects=n_subjects,
                widths=widths,
                columns=carrier[name],
                ordinary_key=key,
                taste_key=taste_key,
                policy=(policies or {}).get(period, {}).get(name),
            )
            mask = inventory.operation(
                function=_regime_mask,
                arguments={"regime_ids": regime_ids, "regime_id": scalar_int},
                subject_arg_names=("regime_ids",),
                subject_outputs=True,
            )
            if regime.has_taste_shocks:
                _profile_taste(
                    inventory=inventory,
                    key=key,
                    taste_key=taste_key,
                    population=padded,
                    original_population=original_population,
                    width=n_subjects,
                    scalar_int=scalar_int,
                )
            if regime.simulation.replay_route.consumer_route == "nnbegm_finite":
                bank = cast(
                    "tuple[jax.ShapeDtypeStruct, ...]",
                    _record_core(
                        inventory=inventory,
                        profile=cores["policy_prepare"],
                        family="policy_prepare",
                    ),
                )
                if log_level != "off":
                    inventory.operation(
                        function=dropped_candidate_counts,
                        arguments={"live": bank[2], "represented": bank[3]},
                        subject_arg_names=("live", "represented"),
                    )
                actions, value, fallback = cast(
                    "_FiniteRankOutput",
                    _record_core(
                        inventory=inventory,
                        profile=cores["decision"],
                        family="policy_rank",
                    ),
                )
            else:
                decision = cores["decision"]
                indices, value = cast(
                    "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
                    _record_core(
                        inventory=inventory, profile=decision, family="decision"
                    ),
                )
                if regime.stakeholders is not None and not base_spaces[name].states:
                    indices, value = cast(
                        "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
                        inventory.operation(
                            function=_broadcast_collective,
                            arguments={"indices": indices, "value": value},
                            static_arguments={"n_subjects": n_subjects},
                            subject_outputs=True,
                        ),
                    )
                decoder = decision.action_decoder
                if decoder is None:
                    raise ExecutionPlanningError(
                        "A decision chunk profile omitted its action decoder."
                    )
                actions = inventory.compiled(
                    name="_lookup_values_from_indices",
                    executable=decoder.executable,
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
            record = (value, actions, carrier[name], mask, own_roles, fallback)
            records.append(record)
            period_values.append(value)
            period_masks.append(cast("jax.ShapeDtypeStruct", mask))
            add_bytes(target=published, source=payload_bytes(tree=record))
            if not regime.terminal:
                carrier, key, new_ids = _profile_next_subjects(
                    inventory=inventory,
                    regime=regime,
                    name=name,
                    period=period,
                    cores=cores,
                    carrier=carrier,
                    mask=cast("jax.ShapeDtypeStruct", mask),
                    key=key,
                    new_ids=new_ids,
                    scalar_int=scalar_int,
                    regime_names_to_ids=regime_names_to_ids,
                    population=padded,
                    original_population=original_population,
                    width=n_subjects,
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
                regimes=regimes,
                values=values,
                policies=policies,
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
    )
    add_bytes(target=permanent, source=padded_inputs)
    add_bytes(target=setup, source=padded_inputs)
    add_bytes(target=permanent, source=inventory.maximum_unit)
    return SimulationChunkProfile(
        n_subjects=n_subjects,
        padded_population=padded,
        stages=tuple(inventory.stages),
        fixed_reservation=permanent,
        output_reservation=output_bank,
        host_stages=host_stages,
        axis_widths=widths,
        setup_reservation=setup,
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
    scalar_int: jax.ShapeDtypeStruct,
    regime_names_to_ids: RegimeNamesToIds,
    population: int,
    original_population: int,
    width: int,
) -> tuple[
    Mapping[str, Mapping[str, jax.ShapeDtypeStruct]],
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
]:
    """Advance state, membership and ordinary-key metadata in actual dispatch order."""
    split = cast(
        "tuple[jax.ShapeDtypeStruct, ...]",
        inventory.operation(
            function=_split_simulation_key,
            arguments={"key": key},
            static_arguments={"partitionable": jax.config.jax_threefry_partitionable},
        ),
    )
    _profile_keys(
        inventory=inventory,
        key=split[0],
        names=tuple(
            name.removeprefix("key_")
            for name in _stochastic_keys(regime=regime, key=key)
        ),
        population=population,
        original_population=original_population,
        width=width,
        scalar_int=scalar_int,
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
        for target, outputs in cast("Mapping[str, Mapping[str, object]]", raw).items()
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
        key=split[1],
        names=("regime_transition",),
        population=population,
        original_population=original_population,
        width=width,
        scalar_int=scalar_int,
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
    carrier = cast("Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]", advanced)
    key = split[2]
    return carrier, key, new_ids


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
    records: list[object],
    published: Mapping[jax.Device, int],
    population: int,
    original_population: int,
    width: int,
    roles: jax.ShapeDtypeStruct,
    entry_periods: jax.ShapeDtypeStruct,
) -> tuple[
    tuple[SimulationStageProfile, ...], dict[jax.Device, int], dict[jax.Device, int]
]:
    """Profile exact entry padding/windows and every final record concatenate/trim.

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
    if width < population:
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
                function=_slice_population,
                arguments={"array": column, "start": scalar},
                static_arguments={"width": width},
                subject_arg_names=("array",) if len(source_devices) > 1 else (),
                subject_outputs=len(source_devices) > 1,
                devices=source_devices,
            )
    inventory.close_unit()
    chunks = population // width
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
            if original_population < population:
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
) -> object:
    """Use the actual resolved core's compiler output metadata."""
    return inventory.compiled(
        name=f"core:{family}",
        executable=profile.executable,
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
) -> tuple[jax.ShapeDtypeStruct, Mapping[str, jax.ShapeDtypeStruct]]:
    """Profile the original full split with dynamic window start and fixed width."""
    return cast(
        "tuple[jax.ShapeDtypeStruct, Mapping[str, jax.ShapeDtypeStruct]]",
        inventory.operation(
            function=_generate_windowed_simulation_keys,
            arguments={"key": key, "start": scalar_int},
            static_arguments={
                "names": names,
                "n_initial_states": population,
                "original_n_subjects": original_population,
                "partitionable": jax.config.jax_threefry_partitionable,
                "width": width,
            },
        ),
    )


def _profile_taste(
    *,
    inventory: ChunkProfileInventory,
    key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
    population: int,
    original_population: int,
    width: int,
    scalar_int: jax.ShapeDtypeStruct,
) -> None:
    """Include the exact ordinary or independent taste-key body and carry."""
    if taste_key is None:
        _profile_keys(
            inventory=inventory,
            key=key,
            names=("taste_shock",),
            population=population,
            original_population=original_population,
            width=width,
            scalar_int=scalar_int,
        )
        return
    inventory.operation(
        function=_advance_simulation_taste_key,
        arguments={"key": key},
        static_arguments={
            "original_n_subjects": original_population,
            "partitionable": jax.config.jax_threefry_partitionable,
        },
    )
    shared = scalar_int.sharding
    inventory.operation(
        function=draw_taste_shock_keys,
        arguments={
            "key": taste_key,
            "address_words": jax.ShapeDtypeStruct(
                (8,), np.dtype(np.uint32), sharding=shared
            ),
            "subject_start": jax.ShapeDtypeStruct(
                (2,), np.dtype(np.uint32), sharding=shared
            ),
            "last_real_subject": jax.ShapeDtypeStruct(
                (2,), np.dtype(np.uint32), sharding=shared
            ),
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
    policies: Mapping[int, Mapping[str, object]] | None = None,
) -> dict[jax.Device, int]:
    """Reserve declared nonaligned copies once per period/address/ordered layout."""
    policy_sources = _policy_read_sources(
        policies=(policies or {}).get(period, {}), period=period
    )
    reads = tuple(
        read
        for regime in regimes.values()
        for family in (
            regime.simulation.programs.policy_prepare,
            regime.simulation.programs.decision,
        )
        if period in family
        for read in family[period].requirements.value_reads
    )
    seen = set()
    result: dict[jax.Device, int] = {}
    for read in reads:
        source = _retained_read_source(
            read=read, values=values, policy_sources=policy_sources
        )
        required = simulation_value_sharding(
            stored_sharding=source.sharding, devices=devices
        )
        identity = (read.target, required)
        if identity in seen:
            continue
        seen.add(identity)
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
                required.device_set | source.sharding.device_set,
                transfer.cost.temporary_bytes,
            ),
        )
    return result


def _policy_read_sources(
    *, policies: Mapping[str, object], period: int
) -> dict[ValueArtifactAddress, jax.Array]:
    """Index canonical retained finite leaves by the actual period-owner addresses."""
    sources = {}
    for name, policy in policies.items():
        if not isinstance(policy, NNBEGMSimPolicy):
            raise ExecutionPlanningError(
                "Chunk policy copies require a finite retained payload schema."
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
) -> jax.Array:
    """Resolve only the declared value and finite-policy storage classes."""
    if read.target.kind is ValueArtifactKind.REGIME_VALUE:
        return values[read.target.period][read.target.regime]
    if (
        read.target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF
        and read.target in policy_sources
    ):
        return policy_sources[read.target]
    raise ExecutionPlanningError(
        "Chunk copies require an explicit retained artifact schema."
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
