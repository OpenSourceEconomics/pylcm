"""Inventory the actual original-choice, route and revealed-action programs."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax
import numpy as np

from _lcm.engine import Regime
from _lcm.execution.core_program import CoreBuildContext
from _lcm.simulation.arrival_routing import (
    arrival_frame_bases,
    arrival_route_codes,
    ordinary_exit_rows,
    pending_arrival_rows,
)
from _lcm.simulation.arrival_values import ArrivalSurfaces, arrival_replay_context
from _lcm.simulation.chunk_operations import _regime_mask
from _lcm.simulation.chunk_profile_inventory import ChunkProfileInventory, abstract_tree
from _lcm.simulation.current_iid_overrides import (
    CurrentIIDOverrideSchema,
    current_iid_override_prototype,
)
from _lcm.simulation.exit_replay import (
    _answer_status,
    _final_decision,
    _moving_actor_rows,
    _original_bases,
    _original_produced,
    _query_reached_landing_totals,
    exit_bindings,
)
from _lcm.simulation.exits import arrival_entry_routes, route_original_subjects
from _lcm.simulation.forward_program_profiles import (
    PreparedDecisionProfile,
    _profile_program,
)
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.random_address import draw_window_keys
from _lcm.simulation.taste_stream import draw_taste_shock_keys
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue, RegimeNamesToIds, ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import RegimeName

type _Carrier = Mapping[str, Mapping[str, jax.ShapeDtypeStruct]]
type _Answer = tuple[jax.ShapeDtypeStruct, ...]
type _Frame = tuple[ShapeDtypePytree, ...]
type _RouteOutput = tuple[
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    _Carrier,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
]


def profile_exit_period(  # noqa: C901, PLR0915 - retain replay's ordered owner walk
    *,
    inventory: ChunkProfileInventory,
    regimes: Mapping[RegimeName, Regime],
    period: int,
    states: _Carrier,
    regime_ids: jax.ShapeDtypeStruct,
    roles: jax.ShapeDtypeStruct,
    entered_through: jax.ShapeDtypeStruct,
    source_roles: jax.ShapeDtypeStruct,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    values: Mapping[int, Mapping[str, jax.Array]],
    surfaces: ArrivalSurfaces = MappingProxyType({}),
    ages: TimeAxis,
    key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
    n_subjects: int,
    widths: Mapping[str, int],
    current_iid_override_schema: CurrentIIDOverrideSchema = (),
) -> tuple[
    _Carrier,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    Mapping[str, PreparedDecisionProfile],
]:
    """Compile the same bodies as replay and keep their returned owner metadata."""
    actors = {
        name: regime
        for name, regime in regimes.items()
        if period in regime.simulation.exit_queries
    }
    if not actors:
        return states, regime_ids, roles, {}
    bindings = exit_bindings(actors=actors, period=period)
    devices = inventory.runtime.subject_devices
    subject = subject_operand_sharding(devices=devices)
    shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
    codes = {
        name: jax.ShapeDtypeStruct(
            jax.typeof(code).shape,
            jax.typeof(code).dtype,
            weak_type=jax.typeof(code).weak_type,
            sharding=shared,
        )
        for name, code in regime_names_to_ids.items()
    }

    def window(
        *, function: Callable[..., PytreeValue], root: jax.ShapeDtypeStruct
    ) -> jax.ShapeDtypeStruct:
        return cast(
            "jax.ShapeDtypeStruct",
            inventory.operation(
                function=function,
                arguments={
                    "key": root,
                    "address_words": jax.ShapeDtypeStruct(
                        (8,), np.uint32, sharding=shared
                    ),
                    "subject_start": jax.ShapeDtypeStruct(
                        (2,), np.uint32, sharding=shared
                    ),
                    "last_real_subject": jax.ShapeDtypeStruct(
                        (2,), np.uint32, sharding=shared
                    ),
                    "period": jax.ShapeDtypeStruct((), np.uint32, sharding=shared),
                },
                static_arguments={"subject_count": n_subjects},
                subject_outputs=True,
            ),
        )

    keys = {
        shock: window(function=draw_window_keys, root=key)
        for binding in bindings.values()
        for shock in binding.group.shocks
    }
    tastes = {
        name: window(
            function=draw_taste_shock_keys, root=key if taste_key is None else taste_key
        )
        for name, regime in actors.items()
        if regime.has_taste_shocks
    }
    params = {name: dict(flat_params.get(name, {})) for name in bindings}
    overrides = current_iid_override_prototype(
        schema=current_iid_override_schema,
        names=tuple(
            shock for binding in bindings.values() for shock in binding.group.shocks
        ),
        n_subjects=n_subjects,
        sharding=subject,
    )

    def native(
        *,
        name: str,
        family: str,
        columns: _Carrier,
        holders: jax.ShapeDtypeStruct,
        bases: ShapeDtypePytree = None,
        produced: ShapeDtypePytree = None,
    ) -> tuple[ShapeDtypePytree, ...]:
        regime = actors[name]
        plan = regime.simulation.exit_queries[period]
        queries = (
            plan.arrivals[family.removeprefix("arrival_entry:")].queries
            if family.startswith("arrival_entry:")
            else plan.full
            if family == "exit_full"
            else plan.participating[name]
        )
        if queries is None:
            raise ExecutionPlanningError("An incoming-only actor has no Full program.")
        context = queries.definitions[queries.root].context
        arguments = queries.arguments(
            context=arrival_replay_context(
                queries=queries,
                values=values,
                surfaces=surfaces,
                context=CoreBuildContext(
                    state_action_space=context.state_action_space,
                    next_regime_to_V_arr=values.get(period + 1, {}),
                    next_regime_to_continuation={},
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                    same_period_regime_to_V_arr=values.get(period, {}),
                    edge_regime_to_V_arr=values.get(period + 1, {}),
                ),
            ),
            fixed_flat_params=plan.fixed_flat_params,
        )
        inputs = {
            "states": columns[name],
            "keys": keys,
            "overrides": overrides,
            "_holders": holders,
            "taste_keys": {queries.root: tastes[name]} if name in tastes else {},
            "query_arguments": {
                query: {
                    key: abstract_tree(tree=value, sharding=shared)
                    for key, value in row.items()
                }
                for query, row in arguments.items()
            },
            "group_params": abstract_tree(tree=params, sharding=shared),
            **({"held_bases": bases} if bases is not None else {}),
            **({"produced": produced} if produced is not None else {}),
        }
        profile = _profile_program(
            runtime=inventory.runtime,
            program=regime.simulation.programs.arrival_program(
                family=family, period=period
            )
            if family.startswith("arrival_entry:")
            else getattr(regime.simulation.programs, family)[period],
            arguments=inputs,
            period=period,
            n_subjects=n_subjects,
            widths=widths,
        )
        return cast(
            "tuple[ShapeDtypePytree, ...]",
            inventory.compiled(
                name=f"core:{family}",
                executable=profile.executable,
                memory=profile.memory,
                arguments=profile.arguments,
            ),
        )

    def check(
        *, answer: _Answer, frame: _Frame, holders: jax.ShapeDtypeStruct, landings: bool
    ) -> None:
        inventory.operation(
            function=_answer_status,
            arguments={
                "valid": answer[3],
                "support": answer[2],
                "law_valid": frame[6],
                "holders": holders,
            },
        )
        if landings:
            inventory.operation(
                function=_query_reached_landing_totals,
                arguments={"counts": answer[4], "holders": holders},
            )

    originals: dict[str, tuple[_Answer, _Frame]] = {}
    projections = {}
    adjustments = {}
    routes = []
    pending_entries = {
        (name, source): inventory.operation(
            function=pending_arrival_rows,
            arguments={
                "regime_ids": regime_ids,
                "entered_through": entered_through,
                "source_id": codes[source],
                "target_id": codes[name],
            },
            subject_arg_names=("regime_ids", "entered_through"),
            subject_outputs=True,
        )
        for name, regime in actors.items()
        for source in regime.simulation.exit_queries[period].arrivals
    }
    for name, regime in actors.items():
        queries = regime.simulation.exit_queries[period].full
        if queries is None:
            continue
        holders = cast(
            "jax.ShapeDtypeStruct",
            inventory.operation(
                function=_regime_mask,
                arguments={"regime_ids": regime_ids, "regime_id": codes[name]},
                subject_arg_names=("regime_ids",),
                subject_outputs=True,
            ),
        )
        incoming = tuple(
            pending
            for (target, _), pending in pending_entries.items()
            if target == name
        )
        if incoming:
            holders = cast(
                "jax.ShapeDtypeStruct",
                inventory.operation(
                    function=ordinary_exit_rows,
                    arguments={"holders": holders, "pending": incoming},
                    subject_arg_names=("holders", "pending"),
                    subject_outputs=True,
                ),
            )
        output = native(name=name, family="exit_full", columns=states, holders=holders)
        answer, frame = cast("_Answer", output[0]), cast("_Frame", output[1])
        originals[name] = (answer, frame)
        check(
            answer=answer,
            frame=frame,
            holders=holders,
            landings=bool(queries.landing_checks),
        )
        projected = cast("Mapping[str, ShapeDtypePytree]", output[2])
        paid = cast("Mapping[str, ShapeDtypePytree]", output[3])
        for identifier, members in queries.definitions[
            queries.root
        ].exit_targets.items():
            for member, query in members.items():
                target = queries.definitions[query].read
                address = f"{name}:{query}"
                projections[address], adjustments[address] = (
                    projected[query],
                    paid[query],
                )
                destination = regimes[target.target]
                routes.append(
                    (
                        name,
                        target.target,
                        identifier,
                        None
                        if member == "self"
                        else regime.stakeholder_names_to_ids[member],
                        -1
                        if destination.stakeholders is None
                        else destination.stakeholder_names_to_ids[target.role],
                        address,
                    )
                )
    decisions = {name: answer[1] for name, (answer, _) in originals.items()}
    arrival_frames, arrival_pending = {}, {}
    for (name, source), holders in pending_entries.items():
        entry = actors[name].simulation.exit_queries[period].arrivals[source]
        output = native(
            name=name,
            family=f"arrival_entry:{source}",
            columns=states,
            holders=cast("jax.ShapeDtypeStruct", holders),
        )
        answer, frame = cast("_Answer", output[0]), cast("_Frame", output[1])
        check(
            answer=answer,
            frame=frame,
            holders=cast("jax.ShapeDtypeStruct", holders),
            landings=bool(entry.queries.landing_checks),
        )
        edge = f"{name}:{source}"
        arrival_frames[edge], arrival_pending[edge] = frame, holders
        entry_routes, route_codes, addresses = arrival_entry_routes(
            regimes=regimes,
            target=name,
            source=source,
            entry=entry,
            first_code=min((-1, *(route[2] for route in routes if route[0] == name)))
            - 1,
        )
        routes.extend(tuple(route) for route in entry_routes)
        projected = cast("Mapping[str, ShapeDtypePytree]", output[2])
        adjustments_for_entry = cast("Mapping[str, ShapeDtypePytree]", output[3])
        for address, query in addresses.items():
            projections[address], adjustments[address] = (
                projected[query],
                adjustments_for_entry[query],
            )
        decisions[name] = cast(
            "jax.ShapeDtypeStruct",
            inventory.operation(
                function=arrival_route_codes,
                arguments={
                    "original": decisions[name],
                    "incoming": answer[1],
                    "pending": holders,
                    "source_roles": source_roles,
                },
                static_arguments={"codes": route_codes},
                subject_arg_names=("original", "incoming", "pending", "source_roles"),
                subject_outputs=True,
            ),
        )
    final_ids, final_roles, final_states, moved, paid = cast(
        "_RouteOutput",
        inventory.operation(
            function=route_original_subjects,
            arguments={
                "original_regime_ids": regime_ids,
                "original_roles": roles,
                "original_states": states,
                "decisions": decisions,
                "projections": projections,
                "adjustments": adjustments,
                "regime_ids": codes,
            },
            static_arguments={"routes": tuple(routes)},
            subject_arg_names=(
                "original_regime_ids",
                "original_roles",
                "original_states",
                "decisions",
                "projections",
                "adjustments",
            ),
            subject_outputs=True,
        ),
    )
    bases = inventory.operation(
        function=_original_bases,
        arguments={
            "regime_ids": regime_ids,
            "bases": {name: frame[2] for name, (_, frame) in originals.items()},
            "regime_names_to_ids": codes,
        },
        subject_arg_names=("regime_ids", "bases"),
        subject_outputs=True,
    )
    produced = inventory.operation(
        function=_original_produced,
        arguments={
            "regime_ids": regime_ids,
            "availability": {name: frame[7] for name, (_, frame) in originals.items()},
            "regime_names_to_ids": codes,
        },
        static_arguments={
            "physical_names": tuple(
                (name, tuple(cast("Mapping[str, jax.ShapeDtypeStruct]", frame[0])))
                for name, (_, frame) in originals.items()
            ),
        },
        subject_arg_names=("regime_ids", "availability"),
        subject_outputs=True,
    )
    if arrival_frames:
        bases, produced = cast(
            "tuple[ShapeDtypePytree, ShapeDtypePytree]",
            inventory.operation(
                function=arrival_frame_bases,
                arguments={
                    "bases": bases,
                    "produced": produced,
                    "incoming_bases": {
                        edge: frame[2] for edge, frame in arrival_frames.items()
                    },
                    "incoming_available": {
                        edge: frame[7] for edge, frame in arrival_frames.items()
                    },
                    "pending": arrival_pending,
                },
                static_arguments={
                    "physical_names": tuple(
                        (
                            edge,
                            tuple(cast("Mapping[str, jax.ShapeDtypeStruct]", frame[0])),
                        )
                        for edge, frame in arrival_frames.items()
                    )
                },
                subject_arg_names=(
                    "bases",
                    "produced",
                    "incoming_bases",
                    "incoming_available",
                    "pending",
                ),
                subject_outputs=True,
            ),
        )
    prepared = {}
    for name, regime in actors.items():
        holders = cast(
            "jax.ShapeDtypeStruct",
            inventory.operation(
                function=_moving_actor_rows,
                arguments={
                    "moved": moved,
                    "regime_ids": final_ids,
                    "regime_id": codes[name],
                },
                subject_arg_names=("moved", "regime_ids"),
                subject_outputs=True,
            ),
        )
        output = native(
            name=name,
            family="exit_participating",
            columns=final_states,
            holders=holders,
            bases=bases,
            produced=produced,
        )
        answer, frame = cast("_Answer", output[0]), cast("_Frame", output[1])
        check(
            answer=answer,
            frame=frame,
            holders=holders,
            landings=bool(
                regime.simulation.exit_queries[period]
                .participating[name]
                .landing_checks
            ),
        )
        source_answer, source_frame = originals.get(name, (answer, frame))
        names = regime.simulation.exit_queries[period].physical_names
        source_physical = cast("Mapping[str, jax.ShapeDtypeStruct]", source_frame[0])
        target_physical = cast("Mapping[str, jax.ShapeDtypeStruct]", frame[0])
        prepared[name] = cast(
            "PreparedDecisionProfile",
            inventory.operation(
                function=_final_decision,
                arguments={
                    "moved": moved,
                    "roles": final_roles,
                    "paid": paid,
                    "source_indices": source_answer[1],
                    "source_value": source_answer[0],
                    "source_physical": {key: source_physical[key] for key in names},
                    "target_indices": answer[1],
                    "target_value": answer[0],
                    "target_physical": {key: target_physical[key] for key in names},
                },
                static_arguments={
                    "stakeholder_ids": tuple(
                        regime.stakeholder_names_to_ids[member]
                        for member in regime.stakeholders
                    )
                    if regime.stakeholders is not None
                    else ()
                },
                subject_arg_names=(
                    "moved",
                    "roles",
                    "paid",
                    "source_indices",
                    "source_value",
                    "source_physical",
                    "target_indices",
                    "target_value",
                    "target_physical",
                ),
                subject_outputs=True,
            ),
        )
    return final_states, final_ids, final_roles, prepared
