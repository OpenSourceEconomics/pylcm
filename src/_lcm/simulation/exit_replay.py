"""Once-only current exits for an immutable chunk of original subject rows."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from _lcm.current_iid import CurrentIIDDrawFrame, CurrentIIDDrawPlan
from _lcm.engine import Regime
from _lcm.execution.core_program import CoreBuildContext
from _lcm.execution.value_transfer import ValueArtifactKind
from _lcm.regime_building.collective import NO_ROLE
from _lcm.shock_registry import ShockBinding
from _lcm.simulation.chunk_operations import regime_mask
from _lcm.simulation.current_iid import prepare_current_iid_keys
from _lcm.simulation.exits import (
    ExitRoute,
    ExitSimulationPlan,
    OriginalExitOutput,
    ParticipatingExitOutput,
    PreparedExitDecision,
    route_original_subjects,
)
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.runtime import SimulationRuntime, execute_simulation_program
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.simulation.taste_stream import prepare_decision_taste_keys
from _lcm.simulation.unit_executor import SimulationUnitExecutor
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.time import TimeAxis
from _lcm.typing import (
    ArrayTree,
    FlatParams,
    PRNGKeyND,
    PytreeValue,
    RegimeNamesToIds,
    StatesPerRegime,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    NoFeasibleActionError,
    ShockDeclarationError,
)
from lcm.typing import BoolND, FloatND, IntND, RegimeName

if TYPE_CHECKING:
    from _lcm.policy_queries import PolicyAnswer, ScalarPolicyQueries
else:
    type PolicyAnswer = tuple[FloatND, IntND, BoolND, BoolND]
    # Query construction imports the engine, which imports forward program types.
    type ScalarPolicyQueries = object

type _OriginalAnswers = Mapping[RegimeName, tuple[PolicyAnswer, CurrentIIDDrawFrame]]
type _QueryArguments = dict[str, dict[str, PytreeValue]]
type _FinalDecisionOutput = tuple[IntND, FloatND, Mapping[str, FloatND | IntND]]


def prepare_exit_period(
    *,
    regimes: Mapping[RegimeName, Regime],
    period: int,
    states: StatesPerRegime,
    regime_ids: IntND,
    roles: IntND,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    ages: TimeAxis,
    values: Mapping[int, Mapping[RegimeName, FloatND]],
    owner: PeriodSimulationReads,
    key: PRNGKeyND,
    subject_slice: slice | SubjectRows,
    original_n_subjects: int,
    memory: SimulationMemory | None,
    taste_key: PRNGKeyND | None = None,
    taste_addresses: Mapping[
        tuple[int, RegimeName], tuple[int, ...]
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
    for name, regime in actors.items():
        plan = regime.simulation.exit_queries[period]
        if plan.full is None:
            continue
        holders = regime_mask(
            regime_ids=regime_ids, regime_id=regime_names_to_ids[name], memory=memory
        )
        arguments = _query_arguments(
            plan=plan,
            queries=plan.full,
            states=states[name],
            flat_params=flat_params,
            period=period,
            ages=ages,
            values=values,
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
            ),
        )
        answer, frame_payload, source_projections, source_adjustments = cast(
            "OriginalExitOutput", output
        )
        frame = CurrentIIDDrawFrame(*frame_payload)
        originals[name] = (answer, frame)
        _check_answer(
            answer=answer,
            frame=frame,
            holders=holders,
            memory=memory,
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
    final_ids, final_roles, final_states, moved, paid = run_simulation_operation(
        memory=memory,
        function=route_original_subjects,
        arguments={
            "original_regime_ids": regime_ids,
            "original_roles": roles,
            "original_states": states,
            "decisions": {name: record[0][1] for name, record in originals.items()},
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
    prepared = {}
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
            ),
        )
        answer, frame_payload = cast("ParticipatingExitOutput", output)
        frame = CurrentIIDDrawFrame(*frame_payload)
        _check_answer(
            answer=answer,
            frame=frame,
            holders=holders,
            memory=memory,
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
    family: str,
    regime: Regime,
) -> _QueryArguments:
    context = queries.definitions[queries.root].context
    arguments = cast(
        "_QueryArguments",
        {
            name: dict(row)
            for name, row in queries.arguments(
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
                fixed_flat_params=plan.fixed_flat_params,
            ).items()
        },
    )
    for read in getattr(regime.simulation.programs, family)[
        period
    ].requirements.value_reads:
        if read.target.kind is not ValueArtifactKind.REGIME_VALUE:
            raise ExecutionPlanningError(
                "Exit replay needs canonical regime-value query operands."
            )
        array = owner.read(
            unit=unit, read=read, value=values[read.target.period][read.target.regime]
        )
        _set_read(arguments=arguments, path=read.source.path, array=array)
    return arguments


def _set_read(
    *, arguments: _QueryArguments, path: tuple[str | int, ...], array: jax.Array
) -> None:
    """Replace the exact leaf, without flattening or dropping nested query paths."""
    current = cast("dict[str, PytreeValue]", arguments)
    for step in path[:-1]:
        if not isinstance(step, str) or not isinstance(current[step], Mapping):
            raise ExecutionPlanningError(
                "A GridSearch query value read needs its declared mapping path."
            )
        nested = dict(cast("Mapping[str, PytreeValue]", current[step]))
        current[step] = nested
        current = nested
    final = path[-1]
    if not isinstance(final, str):
        raise ExecutionPlanningError("A GridSearch query value leaf needs a name.")
    current[final] = array


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


def _answer_status(
    *, valid: BoolND, support: BoolND, law_valid: BoolND, holders: BoolND
) -> tuple[BoolND, BoolND]:
    return jnp.any(holders & ~(valid & law_valid)), jnp.any(holders & ~support)


def _moving_actor_rows(
    *, moved: BoolND, regime_ids: IntND, regime_id: IntND | int
) -> BoolND:
    """Select only original movers acting in this one final regime."""
    return moved & (regime_ids == regime_id)


def _check_answer(
    *,
    answer: PolicyAnswer,
    frame: CurrentIIDDrawFrame,
    holders: BoolND,
    memory: SimulationMemory | None,
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
        raise NoFeasibleActionError(
            "The final current-period policy has no feasible action."
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
