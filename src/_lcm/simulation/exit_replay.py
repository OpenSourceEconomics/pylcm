"""Once-only current exits for an immutable chunk of original subject rows."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.current_iid import CurrentIIDDrawFrame, CurrentIIDDrawPlan
from _lcm.engine import Regime
from _lcm.execution.core_program import CoreBuildContext
from _lcm.execution.value_transfer import ValueArtifactKind
from _lcm.grids.continuous import ContinuousGrid
from _lcm.off_grid import count_off_grid, off_grid_message
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.collective import NO_ROLE
from _lcm.shock_registry import ShockBinding
from _lcm.simulation.arrival_routing import (
    arrival_frame_bases,
    arrival_route_codes,
    ordinary_exit_rows,
    pending_arrival_rows,
)
from _lcm.simulation.arrival_values import (
    ArrivalSurfaces,
    arrival_replay_context,
    is_arrival_surface,
    stored_arrival_surface,
)
from _lcm.simulation.chunk_operations import regime_mask
from _lcm.simulation.current_iid import prepare_current_iid_keys
from _lcm.simulation.current_iid_overrides import (
    CurrentIIDOverrides,
    gather_current_iid_overrides,
)
from _lcm.simulation.exits import (
    ExitRoute,
    ExitSimulationPlan,
    OriginalExitOutput,
    ParticipatingExitOutput,
    PreparedExitDecision,
    arrival_entry_routes,
    route_original_subjects,
)
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.runtime import SimulationRuntime, execute_simulation_program
from _lcm.simulation.shock_paths import fail_if_a_held_process_code_is_off_node
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.simulation.taste_stream import prepare_decision_taste_keys
from _lcm.simulation.unit_executor import SimulationUnitExecutor
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.landing_counts import fail_on_query_landings
from _lcm.time import TimeAxis
from _lcm.typing import (
    ArrayTree,
    ArrivalQueryArguments,
    FlatParams,
    PRNGKeyND,
    PytreeValue,
    QAndFKwargs,
    RegimeNamesToIds,
    StatesPerRegime,
)
from _lcm.utils.counts import sum_counts
from lcm.exceptions import (
    ExecutionPlanningError,
    NoFeasibleActionError,
    OffGridStateError,
    ShockDeclarationError,
)
from lcm.shocks import ShockName
from lcm.typing import BoolND, FloatND, IntND, RegimeName, StateName

if TYPE_CHECKING:
    from _lcm.policy_queries import PolicyAnswer, ScalarPolicyQueries
else:
    type PolicyAnswer = tuple[FloatND, IntND, BoolND, BoolND, IntND]
    # Query construction imports the engine, which imports forward program types.
    type ScalarPolicyQueries = object

type _OriginalAnswers = Mapping[RegimeName, tuple[PolicyAnswer, CurrentIIDDrawFrame]]
type _QueryArguments = dict[str, dict[str, PytreeValue]]
type _FinalDecisionOutput = tuple[IntND, FloatND, Mapping[str, FloatND | IntND]]


def prepare_exit_period(  # noqa: C901, PLR0912, PLR0915 - retain replay's ordered owner walk
    *,
    regimes: Mapping[RegimeName, Regime],
    period: int,
    states: StatesPerRegime,
    regime_ids: IntND,
    roles: IntND,
    entered_through: IntND | None = None,
    source_roles: IntND | None = None,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    ages: TimeAxis,
    values: Mapping[int, Mapping[RegimeName, FloatND]],
    owner: PeriodSimulationReads,
    key: PRNGKeyND,
    surfaces: ArrivalSurfaces = MappingProxyType({}),
    subject_slice: slice | SubjectRows,
    original_n_subjects: int,
    memory: SimulationMemory | None,
    current_iid_overrides: CurrentIIDOverrides | None = None,
    taste_key: PRNGKeyND | None = None,
    taste_addresses: Mapping[
        tuple[int, RegimeName], tuple[int, ...]
    ] = MappingProxyType({}),
    process_code_reads: Mapping[
        RegimeName, Mapping[StateName, _ContinuousStochasticProcess]
    ] = MappingProxyType({}),
) -> tuple[StatesPerRegime, IntND, IntND, Mapping[RegimeName, PreparedExitDecision]]:
    """Choose at entry, route simultaneously, reveal, then prepare final actions."""
    actors = {
        name: regime
        for name, regime in regimes.items()
        if period in regime.simulation.exit_queries
    }
    if not actors:
        return states, regime_ids, roles, MappingProxyType({})
    # The union permission binding is used only to enumerate named key sites.
    # Physical laws are evaluated solely by their actual source/target producer.
    bindings = exit_bindings(actors=actors, period=period)
    if isinstance(subject_slice, SubjectRows) and (
        bindings or any(regime.has_taste_shocks for regime in actors.values())
    ):
        raise ExecutionPlanningError(
            "Current exit innovations do not cover grouped subject rows."
        )
    keys = (
        prepare_current_iid_keys(
            plan=CurrentIIDDrawPlan(bindings=bindings),
            key=key,
            period=period,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
            memory=memory,
        )
        if isinstance(subject_slice, slice)
        else MappingProxyType({})
    )
    taste_banks = {}
    overrides = gather_current_iid_overrides(
        bank=current_iid_overrides,
        names=tuple(
            shock for binding in bindings.values() for shock in binding.group.shocks
        ),
        period=period,
        subject_slice=subject_slice,
    )
    for name, regime in actors.items():
        if regime.has_taste_shocks and isinstance(subject_slice, slice):
            _, taste_banks[name] = prepare_decision_taste_keys(
                key=key,
                taste_key=taste_key,
                taste_address=taste_addresses.get((period, name)),
                n_subjects=original_n_subjects,
                subject_slice=subject_slice,
                original_n_subjects=original_n_subjects,
                memory=memory,
                period=period,
            )
    originals: dict[RegimeName, tuple[PolicyAnswer, CurrentIIDDrawFrame]] = {}
    projected = {}
    adjustments = {}
    routes = []
    pending_entries = {}
    for name, regime in actors.items():
        for source in regime.simulation.exit_queries[period].arrivals:
            if entered_through is None or source_roles is None:
                continue
            pending_entries[name, source] = run_simulation_operation(
                memory=memory,
                function=pending_arrival_rows,
                arguments={
                    "regime_ids": regime_ids,
                    "entered_through": entered_through,
                    "source_id": regime_names_to_ids[source],
                    "target_id": regime_names_to_ids[name],
                },
                subject_arg_names=("regime_ids", "entered_through"),
                subject_outputs=True,
            )
    for name, regime in actors.items():
        plan = regime.simulation.exit_queries[period]
        if plan.full is None:
            continue
        holders = regime_mask(
            regime_ids=regime_ids, regime_id=regime_names_to_ids[name], memory=memory
        )
        incoming = tuple(
            pending
            for (target, _), pending in pending_entries.items()
            if target == name
        )
        if incoming:
            holders = run_simulation_operation(
                memory=memory,
                function=ordinary_exit_rows,
                arguments={"holders": holders, "pending": incoming},
                subject_arg_names=("holders", "pending"),
                subject_outputs=True,
            )
        arguments = _query_arguments(
            plan=plan,
            queries=plan.full,
            states=states[name],
            flat_params=flat_params,
            period=period,
            ages=ages,
            values=values,
            surfaces=surfaces,
            owner=owner,
            unit=name,
            family="exit_full",
            regime=regime,
        )
        output = _dispatch(
            regime=regime,
            family="exit_full",
            period=period,
            arguments={
                "states": states[name],
                "keys": keys,
                "overrides": overrides,
                "query_arguments": arguments,
                "_holders": holders,
                "taste_keys": {plan.full.root: taste_banks[name]}
                if name in taste_banks
                else {},
                "group_params": {
                    group: dict(flat_params.get(group, {})) for group in bindings
                },
            },
            n_subjects=regime_ids.size,
            memory=memory,
            roots=(
                states,
                regime_ids,
                roles,
                originals,
                keys,
                taste_banks,
                projected,
                adjustments,
                overrides,
            ),
        )
        answer, frame_payload, source_projections, source_adjustments = cast(
            "OriginalExitOutput", output
        )
        frame = CurrentIIDDrawFrame(*frame_payload)
        originals[name] = (answer, frame)
        _check_answer(
            queries=plan.full,
            answer=answer,
            frame=frame,
            holders=holders,
            memory=memory,
            regime_name=name,
            period=period,
            states=states[name],
        )
        root = plan.full.definitions[plan.full.root]
        for identifier, members in root.exit_targets.items():
            for member, query in members.items():
                target = plan.full.definitions[query].read
                address = f"{name}:{query}"
                projected[address] = source_projections[query]
                adjustments[address] = source_adjustments[query]
                destination = regimes[target.target]
                routes.append(
                    ExitRoute(
                        source=name,
                        target=target.target,
                        identifier=identifier,
                        source_role=None
                        if member == "self"
                        else regime.stakeholder_names_to_ids[member],
                        target_role=NO_ROLE
                        if destination.stakeholders is None
                        else destination.stakeholder_names_to_ids[target.role],
                        projection=address,
                    )
                )
    decisions = {name: record[0][1] for name, record in originals.items()}
    arrival_frames = {}
    arrival_pending = {}
    for (name, source), holders in pending_entries.items():
        regime = actors[name]
        plan = regime.simulation.exit_queries[period]
        entry = plan.arrivals[source]
        queries = entry.queries
        family = f"arrival_entry:{source}"
        output = _dispatch(
            regime=regime,
            family=family,
            period=period,
            arguments={
                "states": states[name],
                "keys": keys,
                "overrides": overrides,
                "query_arguments": _query_arguments(
                    plan=plan,
                    queries=queries,
                    states=states[name],
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                    values=values,
                    surfaces=surfaces,
                    owner=owner,
                    unit=name,
                    family=family,
                    regime=regime,
                ),
                "_holders": holders,
                "taste_keys": {queries.root: taste_banks[name]}
                if name in taste_banks
                else {},
                "group_params": {
                    group: dict(flat_params.get(group, {})) for group in bindings
                },
            },
            n_subjects=regime_ids.size,
            memory=memory,
            roots=(
                states,
                regime_ids,
                roles,
                originals,
                keys,
                taste_banks,
                projected,
                adjustments,
                overrides,
                decisions,
                arrival_frames,
            ),
        )
        answer, frame_payload, entry_projections, entry_adjustments = cast(
            "OriginalExitOutput", output
        )
        frame = CurrentIIDDrawFrame(*frame_payload)
        # Every fallback is priced for every arriving row, so a fallback regime
        # reading a process's node position must receive a node from each one.
        root = queries.definitions[queries.root]
        for member, query in root.exit_targets.get(1, {}).items():
            fallback = queries.definitions[query].read.target
            _fail_if_a_fallback_lands_off_a_raise_grid(
                fallback=regimes[fallback],
                fallback_name=fallback,
                regime_name=name,
                period=period,
                states=entry_projections[query],
                holders=holders,
            )
            if process_code_reads.get(fallback):
                fail_if_a_held_process_code_is_off_node(
                    states=entry_projections[query],
                    processes=process_code_reads[fallback],
                    holders=holders,
                    where=f"projected by route '{member}' of the arrival from "
                    f"regime '{source}' into regime '{name}' onto its fallback "
                    f"regime '{fallback}' at period {period}",
                    memory=memory,
                )
        _check_answer(
            queries=queries,
            answer=answer,
            frame=frame,
            holders=holders,
            memory=memory,
            regime_name=name,
            period=period,
            states=states[name],
        )
        edge = f"{name}:{source}"
        arrival_frames[edge], arrival_pending[edge] = frame, holders
        next_code = (
            min((-1, *(route.identifier for route in routes if route.source == name)))
            - 1
        )
        entry_routes, codes, addresses = arrival_entry_routes(
            regimes=regimes,
            target=name,
            source=source,
            entry=entry,
            first_code=next_code,
        )
        routes.extend(entry_routes)
        for address, query in addresses.items():
            projected[address], adjustments[address] = (
                entry_projections[query],
                entry_adjustments[query],
            )
        decisions[name] = run_simulation_operation(
            memory=memory,
            function=arrival_route_codes,
            arguments={
                "original": decisions[name],
                "incoming": answer[1],
                "pending": holders,
                "source_roles": source_roles,
            },
            static_arguments={"codes": tuple(codes)},
            subject_arg_names=("original", "incoming", "pending", "source_roles"),
            subject_outputs=True,
        )
    final_ids, final_roles, final_states, moved, paid = run_simulation_operation(
        memory=memory,
        function=route_original_subjects,
        arguments={
            "original_regime_ids": regime_ids,
            "original_roles": roles,
            "original_states": states,
            "decisions": decisions,
            "projections": projected,
            "adjustments": adjustments,
            "regime_ids": regime_names_to_ids,
        },
        static_arguments={
            "routes": tuple(
                (
                    route.source,
                    route.target,
                    route.identifier,
                    route.source_role,
                    route.target_role,
                    route.projection,
                )
                for route in routes
            )
        },
        subject_arg_names=(
            "original_regime_ids",
            "original_roles",
            "original_states",
            "decisions",
            "projections",
            "adjustments",
        ),
        subject_outputs=True,
    )
    bases = _merge_bases(
        originals=originals,
        regime_ids=regime_ids,
        regime_names_to_ids=regime_names_to_ids,
        memory=memory,
    )
    produced = run_simulation_operation(
        memory=memory,
        function=_original_produced,
        arguments={
            "regime_ids": regime_ids,
            "availability": {
                name: frame.information_available
                for name, (_, frame) in originals.items()
            },
            "regime_names_to_ids": regime_names_to_ids,
        },
        static_arguments={
            "physical_names": tuple(
                (name, tuple(frame.physical)) for name, (_, frame) in originals.items()
            ),
        },
        subject_arg_names=("regime_ids", "availability"),
        subject_outputs=True,
    )
    prepared = {}
    if arrival_frames:
        bases, produced = run_simulation_operation(
            memory=memory,
            function=arrival_frame_bases,
            arguments={
                "bases": bases,
                "produced": produced,
                "incoming_bases": {
                    edge: frame.held_bases for edge, frame in arrival_frames.items()
                },
                "incoming_available": {
                    edge: frame.information_available
                    for edge, frame in arrival_frames.items()
                },
                "pending": arrival_pending,
            },
            static_arguments={
                "physical_names": tuple(
                    (edge, tuple(frame.physical))
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
        )
    for name, regime in actors.items():
        plan = regime.simulation.exit_queries[period]
        queries = plan.participating[name]
        # The source Full draw chose the exit or the exact stay action once.
        # Only movers reveal and choose a destination Participating action.
        holders = run_simulation_operation(
            memory=memory,
            function=_moving_actor_rows,
            arguments={
                "moved": moved,
                "regime_ids": final_ids,
                "regime_id": regime_names_to_ids[name],
            },
            subject_arg_names=("moved", "regime_ids"),
            subject_outputs=True,
        )
        arguments = _query_arguments(
            plan=plan,
            queries=queries,
            states=final_states[name],
            flat_params=flat_params,
            period=period,
            ages=ages,
            values=values,
            surfaces=surfaces,
            owner=owner,
            unit=name,
            family="exit_participating",
            regime=regime,
        )
        query_taste_keys = (
            {queries.root: taste_banks[name]} if name in taste_banks else {}
        )
        output = _dispatch(
            regime=regime,
            family="exit_participating",
            period=period,
            arguments={
                "states": final_states[name],
                "keys": keys,
                "held_bases": bases,
                "produced": produced,
                "overrides": overrides,
                "_holders": holders,
                "taste_keys": query_taste_keys,
                "query_arguments": arguments,
                "group_params": {
                    group: dict(flat_params.get(group, {})) for group in bindings
                },
            },
            n_subjects=regime_ids.size,
            memory=memory,
            roots=(
                states,
                regime_ids,
                roles,
                originals,
                final_states,
                final_ids,
                final_roles,
                prepared,
                keys,
                taste_banks,
                bases,
                moved,
                paid,
                projected,
                adjustments,
                overrides,
                produced,
            ),
        )
        answer, frame_payload = cast("ParticipatingExitOutput", output)
        frame = CurrentIIDDrawFrame(*frame_payload)
        _check_answer(
            queries=queries,
            answer=answer,
            frame=frame,
            holders=holders,
            memory=memory,
            regime_name=name,
            period=period,
            states=final_states[name],
        )
        source_answer, source_frame = originals.get(name, (answer, frame))
        physical_names = plan.physical_names
        selected = run_simulation_operation(
            memory=memory,
            function=_final_decision,
            static_arguments={
                "stakeholder_ids": tuple(
                    regime.stakeholder_names_to_ids[member]
                    for member in regime.stakeholders
                )
                if regime.stakeholders is not None
                else (),
            },
            arguments={
                "moved": moved,
                "roles": final_roles,
                "paid": paid,
                "source_indices": source_answer[1],
                "source_value": source_answer[0],
                "source_physical": {
                    shock: source_frame.physical[shock] for shock in physical_names
                },
                "target_indices": answer[1],
                "target_value": answer[0],
                "target_physical": {
                    shock: frame.physical[shock] for shock in physical_names
                },
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
        )
        prepared[name] = PreparedExitDecision(*selected)
    return (
        MappingProxyType(
            {name: MappingProxyType(columns) for name, columns in final_states.items()}
        ),
        final_ids,
        final_roles,
        MappingProxyType(prepared),
    )


def exit_bindings(
    *, actors: Mapping[RegimeName, Regime], period: int
) -> Mapping[str, ShockBinding]:
    """Enumerate only demanded named base sites, retaining each joint group."""
    return {
        group: binding
        for regime in actors.values()
        for queries in (
            regime.simulation.exit_queries[period].full,
            *regime.simulation.exit_queries[period].participating.values(),
            *(
                entry.queries
                for entry in regime.simulation.exit_queries[period].arrivals.values()
            ),
        )
        if queries is not None
        for node in queries.definitions.values()
        if node.context.current_iid_draw_plan is not None
        for group, binding in node.context.current_iid_draw_plan.bindings.items()
        if node.read.policy_information & binding.group.shocks.keys()
    }


def _query_arguments(
    *,
    plan: ExitSimulationPlan,
    queries: ScalarPolicyQueries,
    states: Mapping[str, FloatND | IntND],
    flat_params: FlatParams,
    period: int,
    ages: TimeAxis,
    values: Mapping[int, Mapping[RegimeName, FloatND]],
    owner: PeriodSimulationReads,
    unit: RegimeName,
    surfaces: ArrivalSurfaces = MappingProxyType({}),
    family: str,
    regime: Regime,
) -> _QueryArguments:
    context = queries.definitions[queries.root].context
    arguments = cast(
        "_QueryArguments",
        {
            name: dict(row)
            for name, row in queries.arguments(
                context=arrival_replay_context(
                    queries=queries,
                    values=values,
                    surfaces=surfaces,
                    context=CoreBuildContext(
                        state_action_space=replace(
                            context.state_action_space,
                            states=MappingProxyType(dict(states)),
                        ),
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
            ).items()
        },
    )
    program = (
        regime.simulation.programs.arrival_program(family=family, period=period)
        if family.startswith("arrival_entry:")
        else getattr(regime.simulation.programs, family)[period]
    )
    for read in program.requirements.value_reads:
        if is_arrival_surface(address=read.target):
            stored = stored_arrival_surface(address=read.target, surfaces=surfaces)
        elif read.target.kind is ValueArtifactKind.REGIME_VALUE:
            stored = values[read.target.period][read.target.regime]
        else:
            raise ExecutionPlanningError(
                "Exit replay needs canonical regime-value query operands."
            )
        array = owner.read(unit=unit, read=read, value=cast("jax.Array", stored))
        _set_read(arguments=arguments, path=read.source.path, array=array)
    return arguments


def _set_read(
    *, arguments: _QueryArguments, path: tuple[str | int, ...], array: jax.Array
) -> None:
    """Replace the exact leaf, without flattening or dropping nested query paths."""
    replaced = _replace_query_read(node=arguments, path=path, array=array)
    arguments.clear()
    arguments.update(cast("_QueryArguments", replaced))


def _replace_query_read(
    *, node: PytreeValue, path: tuple[str | int, ...], array: jax.Array
) -> PytreeValue:
    if not path:
        return array
    step, *remaining = path
    if isinstance(node, ArrivalQueryArguments) and step == "by_target":
        return ArrivalQueryArguments(
            by_target=cast(
                "Mapping[str, Mapping[str, QAndFKwargs]]",
                _replace_query_read(
                    node=node.by_target, path=tuple(remaining), array=array
                ),
            )
        )
    if isinstance(node, Mapping) and isinstance(step, str) and step in node:
        return {
            **node,
            step: _replace_query_read(
                node=node[step], path=tuple(remaining), array=array
            ),
        }
    raise ExecutionPlanningError(
        "A GridSearch query value read needs its declared mapping/arrival field path."
    )


def _dispatch(
    *,
    regime: Regime,
    family: str,
    period: int,
    arguments: Mapping[str, PytreeValue],
    n_subjects: int,
    memory: SimulationMemory | None,
    roots: PytreeValue,
) -> PytreeValue:
    programs = regime.simulation.programs
    if memory is None:
        return execute_simulation_program(
            programs=programs,
            family=family,
            period=period,
            arguments=arguments,
            n_subjects=n_subjects,
        )
    # Native query outputs have array leaves, including resolved law operands.
    memory.unit_inputs = cast("ArrayTree", roots)
    executor = SimulationUnitExecutor(
        runtime=cast("SimulationRuntime", programs.executor),
        live_footprint=memory.budget_snapshot,
        budget_devices=memory.devices,
        axis_widths=memory.axis_widths,
        on_output=memory.hold,
    )
    try:
        return execute_simulation_program(
            programs=replace(programs, executor=executor),
            family=family,
            period=period,
            arguments=arguments,
            n_subjects=n_subjects,
        )
    finally:
        executor.close()


def _fail_if_a_fallback_lands_off_a_raise_grid(
    *,
    fallback: Regime,
    fallback_name: RegimeName,
    regime_name: RegimeName,
    period: int,
    states: Mapping[str, FloatND | IntND],
    holders: BoolND,
) -> None:
    """Refuse an arrival fallback landing off a grid declared `off_grid="raise"`.

    A `clip_state` grid's landing is clipped where the fallback is projected.
    """
    axes = fallback.solution.period_state_axes
    for state, values in states.items():
        grid = fallback.simulation.grids.get(state)
        if (
            not isinstance(grid, ContinuousGrid)
            or isinstance(grid, _ContinuousStochasticProcess)
            or grid.off_grid != "raise"
        ):
            continue
        points = None if axes is None else axes.get(period, {}).get(state)
        count = int(
            count_off_grid(
                values=jnp.asarray(values),
                points=grid.to_jax() if points is None else jnp.asarray(points),
                feasible=holders,
            )
        )
        if count:
            raise OffGridStateError(
                off_grid_message(
                    count=count,
                    regime=regime_name,
                    target=fallback_name,
                    period=period,
                    state=state,
                    phase="arrival fallback",
                )
            )


def _answer_status(
    *, valid: BoolND, support: BoolND, law_valid: BoolND, holders: BoolND
) -> tuple[BoolND, BoolND]:
    return jnp.any(holders & ~(valid & law_valid)), jnp.any(holders & ~support)


def _query_reached_landing_totals(*, counts: IntND, holders: BoolND) -> IntND:
    """Count only actual actor rows; numerical integrity is checked separately."""
    return sum_counts(counts=jnp.where(holders[..., None], counts, 0), axis=0)


def _moving_actor_rows(
    *, moved: BoolND, regime_ids: IntND, regime_id: IntND | int
) -> BoolND:
    """Select only original movers acting in this one final regime."""
    return moved & (regime_ids == regime_id)


def _check_answer(
    *,
    queries: ScalarPolicyQueries,
    answer: PolicyAnswer,
    frame: CurrentIIDDrawFrame,
    holders: BoolND,
    memory: SimulationMemory | None,
    regime_name: RegimeName,
    period: int,
    states: Mapping[str, FloatND | IntND],
) -> None:
    invalid, empty = run_simulation_operation(
        memory=memory,
        function=_answer_status,
        arguments={
            "valid": answer[3],
            "support": answer[2],
            "law_valid": frame.valid,
            "holders": holders,
        },
    )
    if bool(invalid):
        raise ShockDeclarationError(
            "Current exit law or query is invalid or unresolved during replay."
        )
    if bool(empty):
        row = int(np.flatnonzero(np.asarray(holders & ~answer[2]))[0])
        state = {name: np.asarray(value)[row].item() for name, value in states.items()}
        raise NoFeasibleActionError(
            f"Regime '{regime_name}', period {period}, state {state}: the final "
            "current-period policy has no feasible action."
        )
    if queries.landing_checks:
        totals = run_simulation_operation(
            memory=memory,
            function=_query_reached_landing_totals,
            arguments={"counts": answer[4], "holders": holders},
        )
        fail_on_query_landings(
            totals=totals, checks=queries.landing_checks, phase="simulate"
        )


def _merge_bases(
    *,
    originals: _OriginalAnswers,
    regime_ids: IntND,
    regime_names_to_ids: RegimeNamesToIds,
    memory: SimulationMemory | None,
) -> Mapping[str, FloatND]:
    return run_simulation_operation(
        memory=memory,
        function=_original_bases,
        arguments={
            "regime_ids": regime_ids,
            "bases": {name: frame.held_bases for name, (_, frame) in originals.items()},
            "regime_names_to_ids": regime_names_to_ids,
        },
        subject_arg_names=("regime_ids", "bases"),
        subject_outputs=True,
    )


def _original_bases(
    *,
    regime_ids: IntND,
    bases: Mapping[RegimeName, Mapping[str, FloatND]],
    regime_names_to_ids: RegimeNamesToIds,
) -> dict[str, FloatND]:
    combined = {
        shock: value for source in bases.values() for shock, value in source.items()
    }
    for name, source in bases.items():
        for shock, value in source.items():
            combined[shock] = jnp.where(
                regime_ids == regime_names_to_ids[name], value, combined[shock]
            )
    return combined


def _original_produced(
    *,
    regime_ids: IntND,
    availability: Mapping[RegimeName, Mapping[ShockName, BoolND]],
    regime_names_to_ids: RegimeNamesToIds,
    physical_names: tuple[tuple[RegimeName, tuple[ShockName, ...]], ...],
) -> dict[ShockName, BoolND]:
    """Retain actual producing contexts in immutable original subject order."""
    produced = {
        shock: jnp.zeros_like(regime_ids, dtype=jnp.bool_)
        for _, names in physical_names
        for shock in names
    }
    for owner, names in physical_names:
        for shock in names:
            produced[shock] |= (
                regime_ids == regime_names_to_ids[owner]
            ) & availability[owner].get(shock, True)
    return produced


def _final_decision(
    *,
    moved: BoolND,
    roles: IntND,
    paid: FloatND,
    source_indices: IntND,
    source_value: FloatND,
    source_physical: Mapping[str, FloatND | IntND],
    target_indices: IntND,
    target_value: FloatND,
    target_physical: Mapping[str, FloatND | IntND],
    stakeholder_ids: tuple[int, ...],
) -> _FinalDecisionOutput:
    mask = moved[:, None] if source_value.ndim > 1 else moved
    if stakeholder_ids:
        adjustment = jnp.stack(
            tuple(jnp.where(roles == role, paid, 0.0) for role in stakeholder_ids),
            axis=-1,
        )
    else:
        adjustment = paid
    # Compiler metadata reconstructs this output with sharding/abstract leaves.
    # The runtime creates its array-only record after dispatch returns arrays.
    return (
        jnp.where(moved, target_indices, source_indices),
        jnp.where(mask, target_value + adjustment, source_value),
        {
            name: jnp.where(moved, target_physical[name], physical)
            for name, physical in source_physical.items()
        },
    )
