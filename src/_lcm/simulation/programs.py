"""Forward simulation's per-regime work, declared as planner-owned programs.

Each regime publishes three families of `CoreProgram`, keyed by the period they
serve:

- `decision`: the argmax over the action product and the value it attains, at
  every subject's own state cell.
- `transition`: the laws of motion carrying each subject into the next period.
- `route`: the regime-transition probabilities the realized draw reads.

A singleton regime also publishes `action_values`, the dense decision that
additionally returns, from the one kernel evaluation it maximizes over, the
action values, their feasibility, the flow utility and the certainty-equivalent
continuation. With taste shocks it publishes the EV1 decision's conditional
values and choice probabilities in place of a drawn action. Only policy lookup
dispatches it, so it compiles only when requested.

Every family tiles the subject axis, so the engine owns the width each body runs
at. A decision whose solve counterpart streams its action product declares that
product as a reduced axis too, with the same canonical order and the same exact
hard-max reduction; a decision whose solve counterpart keeps the canonical dense
reducer declares only the subject axis, and the absence of the reduced axis is
what says the action product stays materialized.

Every family declares an `output_roles` tree of the same structure its body
returns: a pair for the decision, one role per next-state leaf under its target
regime for the transition, and one role per drawable regime for the route.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import ClassVar, TypedDict, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.current_iid import CURRENT_IID_KEYS_ARGUMENT, CURRENT_IID_OVERRIDES_ARGUMENT
from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ReducedAxis,
    ValueRead,
)
from _lcm.execution.output_layout import OutputRoleLeaf, StateAxesLeading
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.decoupled import (
    DECOUPLED_LOOKUP_OUTPUT_ROLES,
    DECOUPLED_TASTE_SHOCK_LOOKUP_OUTPUT_ROLES,
    DecoupledTasteShockLookup,
    DecoupledWithActionParts,
)
from _lcm.regime_building.Q_and_F import (
    BoundedQAndF,
    QAndFParts,
    get_Q_and_F_parts,
    has_action_integrity,
    has_objective_profile,
    runtime_bound_specs,
)
from _lcm.simulation.gated_routing import (
    simulation_gate_fold,
    simulation_gate_route_delta,
)
from _lcm.simulation.program_arguments import ACTIVE_SUBJECTS_ARGUMENT
from _lcm.simulation.program_types import (
    ACTION_INDEX,
    ACTION_PARTS_ROLES,
    DECISION_PROGRAM,
    DECISION_VALUE,
    GATE_FOLD_PROGRAM,
    GATE_ROUTE_PROGRAM,
    GATED_CONTINUATION,
    GATED_NEXT_STATE,
    GATED_REGIME_ID,
    GATED_ROLE,
    GATED_ROUTE_MASK,
    LOOKUP_PART_ROLES,
    ROUTE_PROGRAM,
    SUBJECT_AXIS,
    SUBJECT_WIDTH_KEYWORD,
    TRANSITION_PROGRAM,
    SimulationBuildContext,
    SimulationPrograms,
    _PerSubjectFunction,
    subject_axis,
)
from _lcm.simulation.taste_shock_lookup import (
    TASTE_SHOCK_LOOKUP_OUTPUT_ROLES,
    TasteShockLookup,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION, HardMaxResult
from _lcm.solution.action_streaming import build_streaming_max_Q_over_a
from _lcm.solution.backward_induction import FoldAge
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.feasibility import (
    BoundedResult,
    reduce_bound_input_status,
    reduce_query_input_status,
)
from _lcm.solution.grid_search import (
    ACTION_PRODUCT_AXIS,
    _edge_reference_regimes_for_targets,
    _select_action_width_keyword,
    _supports_action_streaming,
    _value_reads,
)
from _lcm.typing import (
    ActionName,
    FlatParams,
    PytreeValue,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    ShapeDtypePytree,
    StateOrActionName,
    StatesPerRegime,
)
from _lcm.utils.dispatchers import map_over_leading_axis, productmap
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import Discrete
from lcm.typing import (
    Bool1D,
    BoolND,
    ContinuousState,
    DiscreteState,
    FloatND,
    Int1D,
    IntND,
    ReferenceName,
    StateName,
)

type ActionPartsDecisionOutput = tuple[
    IntND, FloatND, FloatND, BoolND, FloatND, FloatND
]


# Why a regime whose routing the host drives cedes its own width.
def build_simulation_programs(
    *,
    context: SolverBuildContext,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    per_subject_decisions: MappingProxyType[int, Callable[..., PytreeValue]],
    per_subject_transitions: MappingProxyType[int, _PerSubjectFunction],
    per_subject_route: _PerSubjectFunction | None,
    simulation_state_names: tuple[StateOrActionName, ...],
    active_periods: tuple[int, ...],
    has_gated_edges: bool,
    type_local_Q_and_F_functions: MappingProxyType[int, QAndFFunction] = (
        MappingProxyType({})
    ),
    type_local_per_subject_decisions: MappingProxyType[
        int, Callable[..., PytreeValue]
    ] = MappingProxyType({}),
) -> SimulationPrograms:
    """Declare one regime's decision, transition, and route programs.

    Args:
        context: The context the regime's solve programs were built under, read
            for the action product's shape, its streaming classification, and
            the value leaves each period's decision reads.
        Q_and_F_functions: The simulate phase's per-period action value and
            feasibility, whose object identity groups the periods that share a
            decision body.
        per_subject_decisions: Period to the dense canonical argmax reducer for
            that period, at one subject's state cell. Used where the solve
            classification keeps the action product materialized.
        per_subject_transitions: Period to the regime's law of motion at one
            subject's state cell, with the arguments that vary per subject.
        per_subject_route: The regime-transition probabilities at one subject's
            state cell, or `None` for a terminal regime.
        simulation_state_names: Per-subject state names, which is what the
            subject axis runs over.
        active_periods: Periods the regime is dispatched at.
        has_gated_edges: Whether the regime declares gated edges, whose routing
            a host loop drives.
        type_local_Q_and_F_functions: The simulate phase's per-period action
            value and feasibility reading every continuation that carries the
            blocked state without that axis; empty when no blocked state is
            carried.
        type_local_per_subject_decisions: The dense reducers over
            `type_local_Q_and_F_functions`, keyed like `per_subject_decisions`.

    Returns:
        The regime's declared programs.

    """
    del has_gated_edges
    streams_actions = _supports_action_streaming(context=context)
    if streams_actions:
        _fail_if_the_streamed_reduction_is_wrong(
            regime_name=context.regime_name,
            has_taste_shocks=context.has_taste_shocks,
            stakeholders=context.stakeholders,
        )
    action_width_keyword = _select_action_width_keyword(context=context)

    decision = _decision_programs(
        context=context,
        Q_and_F_functions=Q_and_F_functions,
        per_subject_decisions=per_subject_decisions,
        simulation_state_names=simulation_state_names,
        active_periods=active_periods,
        streams_actions=streams_actions,
        action_width_keyword=action_width_keyword,
    )
    type_local_decision = (
        _decision_programs(
            context=context,
            Q_and_F_functions=type_local_Q_and_F_functions,
            per_subject_decisions=type_local_per_subject_decisions,
            simulation_state_names=simulation_state_names,
            active_periods=active_periods,
            streams_actions=streams_actions,
            action_width_keyword=action_width_keyword,
        )
        if type_local_Q_and_F_functions
        else {}
    )

    # The dense decision with its parts, even where the decision streams: same
    # reads and subject tiling, no reduced action axis. A household argmax
    # publishes no such parts. A taste-shock regime applies the solve's EV1
    # reduction to its simulation decision, which draws nothing.
    action_value_bodies: dict[int, Callable[..., PytreeValue]] = {}
    action_value_roles: dict[int, tuple[str, ...]] = {}
    action_values: dict[int, CoreProgram] = {}
    space = context.state_action_space
    lookup_subject_arg_names = tuple(space.states)
    for period in decision if context.stakeholders is None else ():
        kernel = Q_and_F_functions[period]
        declared_lookup_subject_names = (
            *lookup_subject_arg_names,
            *(
                (ACTIVE_SUBJECTS_ARGUMENT,)
                if has_action_integrity(function=kernel)
                else ()
            ),
        )
        group = id(kernel)
        if group not in action_value_bodies:
            body, action_value_roles[group] = _lookup_body_and_roles(
                Q_and_F=kernel,
                has_taste_shocks=context.has_taste_shocks,
                n_discrete_action_axes=len(space.discrete_actions),
            )
            action_value_bodies[group] = _SubjectTiled(
                func=_decision_with_action_parts(
                    Q_and_F=kernel, action_names=space.action_names, body=body
                ),
                subject_arg_names=declared_lookup_subject_names,
                n_bound_records=len(runtime_bound_specs(function=kernel)),
            )
        action_values[period] = dataclasses.replace(
            decision[period],
            function=action_value_bodies[group],
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=DECISION_PROGRAM,
                subject_arg_names=declared_lookup_subject_names,
            ),
            requirements=dataclasses.replace(
                decision[period].requirements, reduced_axes=()
            ),
            output_roles=(
                *action_value_roles[group],
                *_bound_output_roles(function=kernel),
            ),
        )

    transition_bodies: dict[int, Callable[..., PytreeValue]] = {}
    transition: dict[int, CoreProgram] = {}
    for period in active_periods:
        built = per_subject_transitions.get(period)
        # A period whose targets carry no state has a law of motion that writes
        # nothing, so the regime publishes no program for it.
        if built is None or not jax.tree.leaves(built.output_roles):
            continue
        group = id(built.function)
        if group not in transition_bodies:
            transition_bodies[group] = _SubjectTiled(
                func=built.function,
                subject_arg_names=built.subject_arg_names,
            )
        transition[period] = CoreProgram(
            name=TRANSITION_PROGRAM,
            function=transition_bodies[group],
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=TRANSITION_PROGRAM,
                subject_arg_names=built.subject_arg_names,
            ),
            requirements=CoreExecutionRequirements(
                tiled_axes=(subject_axis(state_names=simulation_state_names),)
            ),
            output_roles=built.output_roles,
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
        )

    route: dict[int, CoreProgram] = {}
    if per_subject_route is not None:
        route_body = _SubjectTiled(
            func=per_subject_route.function,
            subject_arg_names=per_subject_route.subject_arg_names,
        )
        for period in active_periods:
            route[period] = CoreProgram(
                name=ROUTE_PROGRAM,
                function=route_body,
                argument_builder=_ArgumentsBoundAtDispatch(
                    program_name=ROUTE_PROGRAM,
                    subject_arg_names=per_subject_route.subject_arg_names,
                ),
                requirements=CoreExecutionRequirements(
                    tiled_axes=(subject_axis(state_names=simulation_state_names),),
                ),
                output_roles=per_subject_route.output_roles,
                disposition=CoreExecutionDisposition.PLANNED,
                donation_candidates=(),
            )

    return SimulationPrograms(
        decision=MappingProxyType(decision),
        transition=MappingProxyType(transition),
        route=MappingProxyType(route),
        type_local_decision=MappingProxyType(type_local_decision),
        action_values=MappingProxyType(action_values),
    )


def _decision_programs(
    *,
    context: SolverBuildContext,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    per_subject_decisions: MappingProxyType[int, Callable[..., PytreeValue]],
    simulation_state_names: tuple[StateOrActionName, ...],
    active_periods: tuple[int, ...],
    streams_actions: bool,
    action_width_keyword: str,
) -> dict[int, CoreProgram]:
    """Declare one decision program per period, sharing a body per period group."""
    action_names = context.state_action_space.action_names
    action_extents = context.state_action_space.actions_grid_shapes
    decision_bodies: dict[int, Callable[..., PytreeValue]] = {}
    decision: dict[int, CoreProgram] = {}
    for period in active_periods:
        subject_arg_names = (
            *_decision_subject_arg_names(context=context),
            *(
                (ACTIVE_SUBJECTS_ARGUMENT,)
                if has_action_integrity(function=Q_and_F_functions[period])
                else ()
            ),
        )
        group = id(Q_and_F_functions[period])
        if group not in decision_bodies:
            decision_bodies[group] = _decision_body(
                Q_and_F=Q_and_F_functions[period],
                dense_reducer=per_subject_decisions[period],
                context=context,
                streams_actions=streams_actions,
                action_width_keyword=action_width_keyword,
                subject_arg_names=subject_arg_names,
            )
        decision[period] = CoreProgram(
            name=DECISION_PROGRAM,
            function=decision_bodies[group],
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=DECISION_PROGRAM,
                subject_arg_names=subject_arg_names,
            ),
            requirements=CoreExecutionRequirements(
                reduced_axes=(
                    (
                        ReducedAxis(
                            name=ACTION_PRODUCT_AXIS,
                            coordinate_names=action_names,
                            coordinate_extents=action_extents,
                            canonical_order="c",
                            reduction=HARD_MAX_REDUCTION,
                            width_keyword=action_width_keyword,
                        ),
                    )
                    if streams_actions
                    else ()
                ),
                tiled_axes=(subject_axis(state_names=simulation_state_names),),
                value_reads=rekeyed_value_reads(
                    reads=_decision_value_reads(context=context, period=period),
                    core_key=DECISION_PROGRAM,
                ),
            ),
            output_roles=(
                ACTION_INDEX,
                DECISION_VALUE,
                *(
                    (
                        {
                            shock: StateAxesLeading(
                                state_names=(),
                                dtype=jnp.int32
                                if isinstance(law, Discrete)
                                else canonical_float_dtype(),
                                shape=(),
                            )
                            for binding in (
                                context.current_iid_draw_plan.bindings.values()
                            )
                            for shock, law in binding.group.shocks.items()
                        },
                        StateAxesLeading(
                            state_names=(),
                            n_free_leading_axes=1,
                            dtype=jnp.bool_,
                        ),
                    )
                    if context.current_iid_draw_plan is not None
                    else ()
                ),
                *_bound_output_roles(function=Q_and_F_functions[period]),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
        )
    return decision


def attach_gated_simulation_programs(
    *,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> dict[RegimeName, Regime]:
    """Attach the two planned gated families after every edge is resolved."""
    attached = dict(regimes)
    n_periods = (
        max(period for regime in regimes.values() for period in regime.active_periods)
        + 1
    )
    for name, regime in regimes.items():
        if not regime.gated_edges:
            continue
        fold = {}
        route = {}
        for period in regime.active_periods:
            if period + 1 >= n_periods:
                continue
            folded_targets = tuple(
                target
                for target, edge in regime.gated_edges.items()
                if period + 1 in edge.folds_by_period
            )
            if not folded_targets:
                continue
            fold[period] = CoreProgram(
                name=GATE_FOLD_PROGRAM,
                function=_GateFoldBody(
                    regime=regime, name=name, regimes=regimes, period=period
                ),
                argument_builder=_ArgumentsBoundAtDispatch(
                    program_name=GATE_FOLD_PROGRAM
                ),
                requirements=CoreExecutionRequirements(),
                output_roles=MappingProxyType(
                    dict.fromkeys(folded_targets, GATED_CONTINUATION)
                ),
                disposition=CoreExecutionDisposition.PLANNED,
                donation_candidates=(),
            )
            route[period] = CoreProgram(
                name=GATE_ROUTE_PROGRAM,
                function=_GateRouteBody(
                    regime=regime,
                    regime_names_to_ids=regime_names_to_ids,
                    fold_period=period + 1,
                ),
                argument_builder=_ArgumentsBoundAtDispatch(
                    program_name=GATE_ROUTE_PROGRAM,
                    subject_arg_names=_GATE_ROUTE_SUBJECT_ARG_NAMES,
                ),
                requirements=CoreExecutionRequirements(
                    tiled_axes=(
                        subject_axis(state_names=regime.simulation.state_names),
                    )
                ),
                output_roles=(
                    MappingProxyType(
                        {
                            target: MappingProxyType(
                                {
                                    "closed": GATED_ROUTE_MASK,
                                    "projected": MappingProxyType(
                                        {
                                            fallback: MappingProxyType(
                                                dict.fromkeys(
                                                    regimes[
                                                        fallback
                                                    ].simulation.state_names,
                                                    GATED_NEXT_STATE,
                                                )
                                            )
                                            for fallback in dict.fromkeys(
                                                leg.realized_fallback.regime
                                                for leg in regime.gated_edges[
                                                    target
                                                ].legs
                                            )
                                        }
                                    ),
                                }
                            )
                            for target in folded_targets
                        }
                    ),
                    GATED_REGIME_ID,
                    GATED_ROLE,
                ),
                disposition=CoreExecutionDisposition.PLANNED,
                donation_candidates=(),
            )
        attached[name] = dataclasses.replace(
            regime,
            simulation=dataclasses.replace(
                regime.simulation,
                programs=dataclasses.replace(
                    regime.simulation.programs,
                    gate_fold=MappingProxyType(fold),
                    gate_route=MappingProxyType(route),
                ),
            ),
        )
    return attached


# Payload consumer routes whose forward reads chunk admission profiles: the finite
# NNBEGM bank runs as declared programs, and the flat EGM and NB-EGM state-grid
# policy-first reads (`egm_off_grid`, `nbegm_grid`) as one compiled producer
# beside the grid-argmax fallback.
PROFILED_POLICY_ROUTES = frozenset({"nnbegm_finite", "egm_off_grid", "nbegm_grid"})


def budgeted_simulation_programs_ready(
    *, regimes: Mapping[RegimeName, Regime], n_periods: int, enable_jit: bool
) -> bool:
    """Return whether budgeted simulation can profile every forward unit it runs.

    Requirements derive from `forward_regimes_by_period`, the inventory profiling
    and dispatch walk. A unit needs compiled programs, an in-graph decision route
    and the gate stages it folds; a registered regime no subject can occupy owes
    nothing, so its declaration never vetoes admission. Chunk admission and the
    simulation loop both ask this one question.
    """
    return enable_jit and all(
        gated_simulation_programs_ready(
            regime=regime, period=period, n_periods=n_periods
        )
        and not (
            regime.simulation.replay_route.policy_applicable
            and regime.simulation.replay_route.consumer_route
            not in PROFILED_POLICY_ROUTES
        )
        and regime.simulation.external_replay_route is None
        for period, active in enumerate(
            forward_regimes_by_period(regimes=regimes, n_periods=n_periods)
        )
        for regime in active.values()
    )


def gated_simulation_programs_ready(
    *, regime: Regime, period: int, n_periods: int
) -> bool:
    """Return whether a forward unit's gate stages have planned programs.

    The unit owes a fold and a route program exactly when a gated edge folds at
    the next period. Otherwise its required gate domain is empty and satisfied.
    """
    if period + 1 >= n_periods or not any(
        period + 1 in edge.folds_by_period for edge in regime.gated_edges.values()
    ):
        return True
    programs = regime.simulation.programs
    return all(
        period in family
        and family[period].disposition is CoreExecutionDisposition.PLANNED
        for family in (programs.gate_fold, programs.gate_route)
    )


def forward_regimes_by_period(
    *, regimes: Mapping[RegimeName, Regime], n_periods: int
) -> tuple[MappingProxyType[RegimeName, Regime], ...]:
    """Return, per period, the regimes whose forward programs simulation runs.

    A `(period, regime)` pair is forward work only when a subject can occupy it:
    the regime is active at the period and the simulation graph reaches it. A
    pair solved only for its value (a perceived target no subject physically
    enters) has no decision program and is absent here, while its solved value
    stays available to the decisions that read it. Each period keeps the
    insertion order of `regimes`, so traversal order and random-site identity
    follow the regime mapping, except that a gated edge's target runs before
    its fallbacks (`order_targets_before_fallbacks`). Chunk
    profiling (the compilation wave and the
    sequential carrier walk) and dispatch all read this one inventory.
    """
    return tuple(
        order_targets_before_fallbacks(
            active={
                name: regime
                for name, regime in regimes.items()
                if period in regime.simulation.programs.decision
            },
            regimes=regimes,
        )
        for period in range(n_periods)
    )


def order_targets_before_fallbacks(
    *, active: Mapping[RegimeName, Regime], regimes: Mapping[RegimeName, Regime]
) -> MappingProxyType[RegimeName, Regime]:
    """Order a period's regimes so each gated target runs before its fallbacks.

    A household rerouted out of a gated target acts in a fallback regime in the
    same period, so that fallback must not have run yet. The order is the
    declaration order wherever that already satisfies this, and otherwise the
    earliest-declared regime whose targets have all run comes next. Rows are
    disjoint across regimes and random keys are addressed by regime id, so the
    order does not change any other result. A gate whose fallback is its own
    target imposes no order: that regime decides again for its rerouted rows.

    Where the targets and fallbacks active in the period form a cycle, no order
    lets every fallback run after its targets, and the earliest-declared regime
    still waiting comes next. Simulation then refuses only a household actually
    rerouted into a regime that already ran (`NoFeasibleActionError`).
    """
    must_follow: dict[RegimeName, set[RegimeName]] = {name: set() for name in active}
    for regime in regimes.values():
        for target, edge in regime.gated_edges.items():
            for leg in edge.legs:
                fallback = leg.realized_fallback.regime
                if target in active and fallback in active and fallback != target:
                    must_follow[fallback].add(target)
    ordered: dict[RegimeName, Regime] = {}
    while len(ordered) < len(active):
        waiting = [name for name in active if name not in ordered]
        ready = next(
            (name for name in waiting if must_follow[name] <= ordered.keys()),
            waiting[0],
        )
        ordered[ready] = active[ready]
    return MappingProxyType(ordered)


class _GateFoldArguments(TypedDict):
    """The arguments a gate-fold program hands its body."""

    next_regime_to_V_arr: Mapping[RegimeName, FloatND]
    target_states_by_target: (
        Mapping[RegimeName, Mapping[StateName, ContinuousState | DiscreteState]] | None
    )
    edge_values: Mapping[RegimeName, Mapping[RegimeName, FloatND]]
    flat_params: FlatParams
    fold_age: FoldAge


class _GateRouteArguments(TypedDict):
    """The arguments a gate-route program hands its body, besides the width."""

    edge_values: Mapping[RegimeName, Mapping[RegimeName, FloatND]]
    next_states: StatesPerRegime
    new_subject_regime_ids: Int1D
    subjects_in_regime: Bool1D
    flat_params: FlatParams
    own_stakeholder: Int1D
    new_own_stakeholder: Int1D
    fold_age: FoldAge


#: Gate-route operands partitioned across subjects; every other one is shared.
_GATE_ROUTE_SUBJECT_ARG_NAMES = (
    "next_states",
    "new_subject_regime_ids",
    "subjects_in_regime",
    "own_stakeholder",
    "new_own_stakeholder",
)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _GateFoldBody:
    """One period's gated continuations, folded on the regime-level grids.

    The fold reads the next period's value arrays over each
    target's own grid and writes the substituted continuation over that same
    grid. No operand and no output carries a subject axis, so the body is
    replicated rather than partitioned wherever the population is spread over
    several devices, and the grids it publishes stay shared operands of the
    gate route that consumes them.
    """

    regime: Regime
    name: RegimeName
    regimes: Mapping[RegimeName, Regime]
    period: int

    @property
    def subject_shard_arg_names(self) -> tuple[str, ...]:
        """Declare that the fold partitions nothing: it has no subject axis."""
        return ()

    def __call__(self, **kwargs: PytreeValue) -> PytreeValue:
        arguments = cast("_GateFoldArguments", kwargs)
        folded = simulation_gate_fold(
            regime=self.regime,
            regime_name=self.name,
            regimes=self.regimes,
            period=self.period,
            next_regime_to_V_arr=arguments["next_regime_to_V_arr"],
            base_state_action_spaces=MappingProxyType({}),
            target_states_by_target=arguments["target_states_by_target"],
            edge_values=arguments["edge_values"],
            flat_params=arguments["flat_params"],
            fold_age=arguments["fold_age"],
        )
        return MappingProxyType(
            {
                target: folded[target]
                for target in self.regime.gated_edges
                if target in folded
            }
        )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _GateRouteBody:
    """Each subject's own passage through its regime's gated edges.

    Every step is taken at the subject's own row: the gate is recomputed at
    that row's realized candidate state, its leg's destinations follow the role
    the row carries, and the fallback coordinates are written under an
    elementwise mask. Nothing reduces, sorts or scatters across rows, so the
    population may be partitioned over devices while the folded continuation
    grids and the params stay shared.
    """

    regime: Regime
    regime_names_to_ids: RegimeNamesToIds
    fold_period: int

    @property
    def subject_shard_arg_names(self) -> tuple[str, ...]:
        """Name the operands whose leaves carry the independent subject axis."""
        return _GATE_ROUTE_SUBJECT_ARG_NAMES

    def __call__(self, **kwargs: PytreeValue) -> PytreeValue:
        arguments = cast("_GateRouteArguments", kwargs)
        return simulation_gate_route_delta(
            regime=self.regime,
            fold_period=self.fold_period,
            edge_values=arguments["edge_values"],
            candidate_states=arguments["next_states"],
            regime_names_to_ids=self.regime_names_to_ids,
            new_subject_regime_ids=arguments["new_subject_regime_ids"],
            subjects_in_regime=arguments["subjects_in_regime"],
            flat_params=arguments["flat_params"],
            own_stakeholder=arguments["own_stakeholder"],
            new_own_stakeholder=arguments["new_own_stakeholder"],
            fold_age=arguments["fold_age"],
            subject_width=cast("int | None", kwargs[SUBJECT_WIDTH_KEYWORD]),
        )


def _fail_if_the_streamed_reduction_is_wrong(
    *,
    regime_name: RegimeName,
    has_taste_shocks: bool,
    stakeholders: tuple[str, ...] | None,
) -> None:
    """Refuse a streamed decision whose reduction is not the exact hard max.

    The simulate decision mirrors the solve: it streams exactly where the solve
    kernel streams, so both reduce the same canonical product with the same
    exact hard max and a period's two phases cannot disagree about the winner.
    That mirror is sound only for a regime whose action reduction *is* that hard
    max. A taste shock adds a per-subject Gumbel term to the choice and a
    collective utility reduces the household over its stakeholders; either one
    makes the streamed body answer a different question than the regime asks, so
    the mismatch is refused here rather than left to the classifier's ordering.
    """
    if not (has_taste_shocks or stakeholders is not None):
        return
    reduction = (
        "taste shocks draw a per-subject Gumbel term"
        if has_taste_shocks
        else "a collective utility reduces over stakeholders"
    )
    msg = (
        f"Regime {regime_name!r} would declare a streamed simulation "
        f"decision, but {reduction}, so its action reduction is not the exact "
        "hard max the streamed body folds. Declare a dense action product for "
        "this regime instead."
    )
    raise ExecutionPlanningError(msg)


def _decision_subject_arg_names(*, context: SolverBuildContext) -> tuple[str, ...]:
    """Return the decision arguments carrying a per-subject leading axis."""
    names = tuple(context.state_action_space.states)
    if context.has_taste_shocks:
        names = (*names, "taste_shock_key")
    if context.current_iid_draw_plan is not None:
        names = tuple(
            dict.fromkeys(
                (
                    *names,
                    *sorted(context.current_iid_draw_plan.state_inputs),
                    CURRENT_IID_KEYS_ARGUMENT,
                    CURRENT_IID_OVERRIDES_ARGUMENT,
                )
            )
        )
    return names


def _decision_value_reads(
    *, context: SolverBuildContext, period: int
) -> tuple[ValueRead, ...]:
    """Declare every stored value leaf one period's decision reads."""
    target_regimes = (
        ()
        if period == context.solution_reachability.n_periods - 1
        else context.solution_reachability.targets(
            period=period, source=context.regime_name
        )
    )
    return _value_reads(
        regime_name=context.regime_name,
        period=period,
        target_regimes=target_regimes,
        same_period_ref_regimes=context.same_period_ref_regimes,
        edge_reference_regimes=_edge_reference_regimes_for_targets(
            context=context, target_regimes=target_regimes
        ),
        edge_target_regimes=context.edge_target_regimes,
    )


def _bound_output_roles(*, function: QAndFFunction) -> tuple[StateAxesLeading, ...]:
    """Declare one scalar Boolean leaf per mandatory canonical integrity flag."""
    return tuple(
        StateAxesLeading(state_names=(), dtype=jnp.bool_, shape=())
        for _ in range(2 * len(runtime_bound_specs(function=function)))
    )


def _flatten_subject_integrity[Output: PytreeValue](
    *, result: BoundedResult[Output], n_records: int, n_query_records: int
) -> tuple[PytreeValue, ...]:
    """Require scalar Boolean summaries before exposing the subject core's outputs."""
    if (
        type(result) is not BoundedResult
        or len(result.bound_status) != n_records
        or len(result.query_status or ()) != n_query_records
    ):
        raise TypeError("Subject summaries must match model-owned declarations.")
    if any(
        flag.shape != () or flag.dtype != jnp.bool_
        for status in (*result.bound_status, *(result.query_status or ()))
        for flag in status
    ):
        raise TypeError("Subject integrity outputs must be scalar Boolean arrays.")
    outputs = cast("tuple[PytreeValue, ...]", result.output)
    return (
        *outputs,
        *(
            flag
            for status in (*result.bound_status, *(result.query_status or ()))
            for flag in status
        ),
    )


def _decision_body(
    *,
    Q_and_F: QAndFFunction,
    dense_reducer: Callable[..., PytreeValue],
    context: SolverBuildContext,
    streams_actions: bool,
    action_width_keyword: str,
    subject_arg_names: tuple[ReferenceName, ...],
) -> Callable[..., PytreeValue]:
    """Build one period group's decision body, tiled over the subject axis."""
    n_bound_records = len(runtime_bound_specs(function=Q_and_F))
    if not streams_actions:
        return _SubjectTiled(
            func=dense_reducer,
            subject_arg_names=subject_arg_names,
            n_bound_records=n_bound_records,
        )
    from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
        _get_extra_param_names,
    )

    action_names = context.state_action_space.action_names
    state_names = context.state_action_space.state_names
    q_and_f_arg_names = frozenset(inspect.signature(Q_and_F).parameters)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    cell = with_signature(
        _StreamedArgmaxQOverA(
            Q_and_F=Q_and_F,
            action_names=action_names,
            q_and_f_arg_names=q_and_f_arg_names,
            action_width_keyword=action_width_keyword,
        ),
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            action_width_keyword,
        ],
        return_annotation="tuple[IntND, FloatND]",
        enforce=False,
    )
    if context.current_iid_draw_plan is not None:
        cell = context.current_iid_draw_plan.wrap(cell)
    return _SubjectTiled(
        func=cell,
        subject_arg_names=subject_arg_names,
        n_bound_records=n_bound_records,
    )


# What the dense decision with its parts publishes, in the order its body
# returns them: the decision's own pair, then each part over the action product.
ACTION_VALUES_OUTPUT_ROLES = (ACTION_INDEX, DECISION_VALUE, *ACTION_PARTS_ROLES)


def _lookup_body_and_roles(
    *, Q_and_F: QAndFFunction, has_taste_shocks: bool, n_discrete_action_axes: int
) -> tuple[Callable[..., Callable[..., PytreeValue]], tuple[str, ...]]:
    """Return the lookup body for one kernel and the roles it publishes, in order."""
    match has_objective_profile(Q_and_F=Q_and_F), has_taste_shocks:
        case True, True:
            return (
                partial(
                    DecoupledTasteShockLookup,
                    n_discrete_action_axes=n_discrete_action_axes,
                ),
                DECOUPLED_TASTE_SHOCK_LOOKUP_OUTPUT_ROLES,
            )
        case True, False:
            return DecoupledWithActionParts, DECOUPLED_LOOKUP_OUTPUT_ROLES
        case False, True:
            return (
                partial(
                    TasteShockLookup, n_discrete_action_axes=n_discrete_action_axes
                ),
                TASTE_SHOCK_LOOKUP_OUTPUT_ROLES,
            )
        case _:
            return _DecisionWithActionParts, ACTION_VALUES_OUTPUT_ROLES


# Derived programs keyed by the source program's identity and the kept parts. The
# source is held alongside so its identity cannot be reused while the entry lives.
type KeptPartsPrograms = dict[
    tuple[int, frozenset[str]], tuple[CoreProgram, CoreProgram]
]


def with_kept_action_parts(
    *, program: CoreProgram, roles: frozenset[str], cache: KeptPartsPrograms
) -> CoreProgram:
    """Return a lookup decision with its parts that publishes only the parts in `roles`.

    The decision's own outputs are always published, followed by the kept parts in
    the order the program publishes them. The selection is made at each state cell, so
    a dropped part is never stacked over the subjects. Keeping every part returns
    `program` unchanged; one object per kept subset keeps the executor's
    compiled-route cache warm. `cache` belongs to the owner of `program`, so the
    derived programs live exactly as long as it does; the owner serializes calls.
    """
    unknown = roles - set(LOOKUP_PART_ROLES)
    if unknown:
        msg = (
            f"Unknown action part roles {sorted(unknown)}; known: {LOOKUP_PART_ROLES}."
        )
        raise ValueError(msg)
    published = tuple(
        role
        for role in cast("tuple[str | OutputRoleLeaf, ...]", program.output_roles)
        if isinstance(role, str)
    )
    if roles >= {role for role in published if role in LOOKUP_PART_ROLES}:
        return program
    key = (id(program), roles)
    if key not in cache:
        kept = tuple(
            role for role in published if role not in LOOKUP_PART_ROLES or role in roles
        )
        cache[key] = (
            program,
            dataclasses.replace(
                program,
                function=_select_cell_outputs(
                    function=program.function,
                    positions=tuple(published.index(role) for role in kept),
                ),
                output_roles=(
                    *kept,
                    *(
                        role
                        for role in cast(
                            "tuple[str | OutputRoleLeaf, ...]", program.output_roles
                        )
                        if not isinstance(role, str)
                    ),
                ),
            ),
        )
    return cache[key][1]


def _select_cell_outputs(
    *, function: Callable[..., PytreeValue], positions: tuple[int, ...]
) -> Callable[..., PytreeValue]:
    """Return `function` with its tiled cell body publishing only `positions`.

    `function` is a subject-tiled body, possibly under the keyword bindings the
    model layer wraps around it; the bindings are kept around the new body.
    """
    bindings: list[Mapping[str, PytreeValue]] = []
    inner = function
    while isinstance(inner, partial):
        if inner.args:
            msg = "A simulation body takes keyword bindings only."
            raise TypeError(msg)
        bindings.append(inner.keywords)
        inner = inner.func
    tiled = cast("_SubjectTiled", inner)
    selected: Callable[..., PytreeValue] = dataclasses.replace(
        tiled,
        func=with_signature(
            _SelectedOutputs(body=tiled.func, positions=positions),
            args=list(inspect.signature(tiled.func).parameters),
            enforce=False,
        ),
    )
    for keywords in reversed(bindings):
        selected = partial(selected, **keywords)
    return selected


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _SelectedOutputs:
    """A cell body that publishes only some of the outputs of another."""

    __name__: ClassVar[str] = "argmax_and_max_Q_over_a_with_kept_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    body: Callable[..., PytreeValue]
    """The cell body whose output tuple is selected from."""

    positions: tuple[int, ...]
    """Positions of the published outputs in the output tuple of `body`."""

    def __call__(
        self, **kwargs: PytreeValue
    ) -> tuple[PytreeValue, ...] | BoundedResult[tuple[PytreeValue, ...]]:
        """Return the outputs of `body` at `positions`."""
        result = self.body(**kwargs)
        outputs = result.output if type(result) is BoundedResult else result
        selected = tuple(
            cast("tuple[PytreeValue, ...]", outputs)[position]
            for position in self.positions
        )
        return (
            BoundedResult[tuple[PytreeValue, ...]](
                output=selected,
                bound_status=result.bound_status,
                query_status=result.query_status,
            )
            if type(result) is BoundedResult
            else selected
        )


def _decision_with_action_parts(
    *,
    Q_and_F: QAndFFunction,
    action_names: tuple[ActionName, ...],
    body: Callable[..., Callable[..., PytreeValue]],
) -> Callable[..., PytreeValue]:
    """Return the dense decision extended by the parts it reduces.

    The result takes the arguments of the parts form of `Q_and_F` mapped once
    over the action product, and calls `body` built from it: the hard max over
    that `Q` and `F`, or the EV1 reduction the solve applies, both without a
    taste-shock draw.
    """
    bounded = has_action_integrity(function=Q_and_F)
    parts = productmap(
        func=BoundedQAndF(function=Q_and_F, parts=True)
        if bounded
        else get_Q_and_F_parts(Q_and_F=Q_and_F),
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
        bound_summary=bounded,
    )
    return with_signature(
        (
            _BoundActionPartsBody(
                parts=cast("Callable[..., BoundedResult[QAndFParts]]", parts),
                body=body,
            )
            if bounded and body is not _DecisionWithActionParts
            else body(parts=parts)
        ),
        args=list(inspect.signature(parts).parameters),
        return_annotation="tuple[FloatND | IntND | BoolND, ...]",
        enforce=False,
    )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _BoundActionPartsBody:
    """Keep bound integrity beside a readout of the same evaluated action parts."""

    __name__: ClassVar[str] = "action_parts_with_bound_summary"
    """Name advertised by the canonical action readout wrapper."""

    parts: Callable[..., BoundedResult[QAndFParts]]
    """Canonical action evaluation with its integrity records."""
    body: Callable[..., Callable[..., PytreeValue]]
    """Readout constructor consuming already evaluated ordinary parts."""

    def __call__(self, **kwargs: PytreeValue) -> BoundedResult[PytreeValue]:
        evaluated = self.parts(**kwargs)
        readout = self.body(
            parts=partial(_evaluated_action_parts, parts=evaluated.output)
        )
        return BoundedResult(
            output=readout(**kwargs),
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
        )


def _evaluated_action_parts(*, parts: QAndFParts, **_kwargs: PytreeValue) -> QAndFParts:
    """Return the ordinary parts already evaluated by the canonical action owner."""
    return parts


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _DecisionWithActionParts:
    """The dense hard-max decision at one state cell, with the parts it maximizes.

    Returns the chosen flat index and its value, followed by the action values
    `Q`, their feasibility `F`, the flow utility `U` and the certainty-equivalent
    continuation `CE` over the action product, all from one evaluation of the
    regime's kernel. A cell without a feasible action publishes index -1, as the
    dense decision does. `Q`, `U` and `CE` are unmasked: an infeasible entry holds
    what the action would have; the objective masks `Q` with `F`.
    """

    __name__: ClassVar[str] = "argmax_and_max_Q_over_a_with_action_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts | BoundedResult[QAndFParts]]
    """The kernel's parts, mapped over the action product."""

    def __call__(
        self,
        **kwargs: PytreeValue,
    ) -> ActionPartsDecisionOutput | BoundedResult[ActionPartsDecisionOutput]:
        """Return the chosen flat index, its value, and `Q`, `F`, `U` and `CE`."""
        evaluated = self.parts(**kwargs)
        parts = cast(
            "QAndFParts",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        index, value = argmax_and_max(a=parts.Q, where=parts.F, initial=-jnp.inf)
        index = jnp.where(jnp.any(parts.F), index, -1)
        output = index, value, parts.Q, parts.F, parts.U, parts.CE
        return (
            BoundedResult[ActionPartsDecisionOutput](
                output=output,
                bound_status=evaluated.bound_status,
                query_status=evaluated.query_status,
            )
            if type(evaluated) is BoundedResult
            else output
        )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _StreamedArgmaxQOverA:
    """The argmax of Q over a streamed action product, at one state cell.

    Publishes the flat identity of the winning action in the canonical C order
    of the action product, and the value it attains — the same pair the dense
    canonical reducer publishes, obtained by the exact hard-max fold the solve
    kernel streams with. An all-infeasible cell publishes identity -1, as the
    dense reducer does; no action grid decodes it, so a caller refuses the cell
    before reading its action.
    """

    __name__: ClassVar[str] = "streamed_argmax_and_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: QAndFFunction
    """The regime's action value and feasibility, evaluated per action block."""

    action_names: tuple[ActionName, ...]
    """Action variable names, spanning the canonical streamed product."""

    q_and_f_arg_names: frozenset[ReferenceName]
    """The argument names `Q_and_F` declares, which select what it is handed."""

    action_width_keyword: str
    """Name of the planner-bound static action-block width in the call."""

    folds: dict[int, Callable[..., HardMaxResult | BoundedResult[HardMaxResult]]] = (
        dataclasses.field(default_factory=dict, repr=False)
    )
    """Streamed folds already built, by the block width each streams at."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: PytreeValue,
    ) -> tuple[IntND, FloatND] | BoundedResult[tuple[IntND, FloatND]]:
        """Return the chosen action's flat identity and the value it attains."""
        block_width = cast("int", states_actions_params[self.action_width_keyword])
        q_and_f_params = {
            name: value
            for name, value in states_actions_params.items()
            if name in self.q_and_f_arg_names
        }
        evaluated = self._fold(block_width=block_width)(
            next_regime_to_V_arr=next_regime_to_V_arr, **q_and_f_params
        )
        result = cast(
            "HardMaxResult",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        output = (
            result.best_global_action_id.astype(jnp.int32),
            result.best_value,
        )
        return (
            BoundedResult[tuple[IntND, FloatND]](
                output=output,
                bound_status=evaluated.bound_status,
                query_status=evaluated.query_status,
            )
            if type(evaluated) is BoundedResult
            else output
        )

    def _fold(
        self, *, block_width: int
    ) -> Callable[..., HardMaxResult | BoundedResult[HardMaxResult]]:
        """Return the streamed fold for one block width, building it once."""
        fold = self.folds.get(block_width)
        if fold is None:
            fold = build_streaming_max_Q_over_a(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                block_width=block_width,
            )
            self.folds[block_width] = fold
        return fold


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _SubjectTiled:
    """One per-subject body evaluated over tiles of the subject axis.

    The tile width arrives as a planner-bound keyword and sets the compiled
    evaluation window; the tiles are concatenated, never folded, so the result
    is the same array the whole population would produce. A body with no
    per-subject argument has no axis to tile and is called once.
    """

    __name__: ClassVar[str] = "subject_tiled"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    func: Callable[..., PytreeValue]
    """The body, at one subject's state cell."""

    subject_arg_names: tuple[ReferenceName, ...]
    """Arguments carrying the per-subject leading axis this splits into tiles."""
    n_bound_records: int = dataclasses.field(
        default=0,
        metadata={"fingerprint_omit_if_default": True},
    )
    """Declared mandatory records reduced separately before row concatenation."""
    n_query_records: int = dataclasses.field(
        default=0, metadata={"fingerprint_omit_if_default": True}
    )
    """Actual projected paired-read slots kept through the same subject reduction."""

    @property
    def subject_shard_arg_names(self) -> tuple[str, ...]:
        """Declare independent ordinary rows beside explicitly declared scalar roles."""
        if self.subject_arg_names == (ACTIVE_SUBJECTS_ARGUMENT,):
            return ()
        return self.subject_arg_names

    def __call__(self, **kwargs: PytreeValue) -> PytreeValue:
        """Return the body's output for every subject, evaluated in tiles."""
        width = cast("int", kwargs.pop(SUBJECT_WIDTH_KEYWORD))
        active = cast("BoolND | None", kwargs.pop(ACTIVE_SUBJECTS_ARGUMENT, None))
        if active is not None and (active.ndim != 1 or active.dtype != jnp.bool_):
            raise TypeError("Subject activity must be a Boolean row mask.")
        accepted = inspect.signature(self.func).parameters
        subject_arg_names = tuple(
            name for name in self.subject_arg_names if name != ACTIVE_SUBJECTS_ARGUMENT
        )
        if not subject_arg_names:
            result = self.func(
                **{name: value for name, value in kwargs.items() if name in accepted}
            )
            if active is not None and (self.n_bound_records or self.n_query_records):
                bounded = cast("BoundedResult[PytreeValue]", result)
                result = dataclasses.replace(
                    bounded,
                    bound_status=reduce_bound_input_status(
                        statuses=bounded.bound_status, active=jnp.any(active)
                    ),
                    query_status=reduce_query_input_status(
                        statuses=bounded.query_status, active=jnp.any(active)
                    ),
                )
            return (
                _flatten_subject_integrity(
                    result=cast("BoundedResult[PytreeValue]", result),
                    n_records=self.n_bound_records,
                    n_query_records=self.n_query_records,
                )
                if self.n_bound_records or self.n_query_records
                else result
            )
        tiles = {name: kwargs.pop(name) for name in subject_arg_names}
        shared = {name: value for name, value in kwargs.items() if name in accepted}
        if self.n_bound_records or self.n_query_records:
            result = map_over_leading_axis(
                func=partial(_evaluate_subject_tile, func=self.func, shared=shared),
                xs=tiles,
                batch_size=width,
                bound_summary=True,
                active=active,
            )
            return _flatten_subject_integrity(
                result=cast("BoundedResult[PytreeValue]", result),
                n_records=self.n_bound_records,
                n_query_records=self.n_query_records,
            )
        # A remainder shorter than the width would be traced and compiled as a
        # second program, whose rounding need not match the full tiles'. Pad it
        # with copies of the last subject so every tile has the width's shape,
        # then drop the copies. Only the remainder is padded: padding the whole
        # subject axis would give it an extent the subject mesh cannot divide,
        # and the partitioner would then replicate the result. Full tiles and
        # the padded remainder call one jitted tile, so the body is traced once.
        tile = jax.jit(
            jax.vmap(partial(_evaluate_subject_tile, func=self.func, shared=shared))
        )
        n_subjects = jax.tree.leaves(tiles)[0].shape[0]
        n_rest = n_subjects % width
        n_full = n_subjects - n_rest
        parts = []
        if n_full:
            head = jax.tree.map(
                lambda leaf: leaf[:n_full].reshape(-1, width, *leaf.shape[1:]),
                tiles,
            )
            full = jax.lax.map(tile, head)
            parts.append(
                jax.tree.map(
                    lambda leaf: leaf.reshape(
                        leaf.shape[0] * leaf.shape[1], *leaf.shape[2:]
                    ),
                    full,
                )
            )
        if n_rest:
            rest = jax.tree.map(
                lambda leaf: jnp.concatenate(
                    [leaf[n_full:], jnp.repeat(leaf[-1:], width - n_rest, axis=0)]
                ),
                tiles,
            )
            parts.append(jax.tree.map(lambda leaf: leaf[:n_rest], tile(rest)))
        if len(parts) == 1:
            return parts[0]
        return jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *parts)


# keyword-only-exempt: library-callback=jax.lax.map
def _evaluate_subject_tile(
    subject: Mapping[ReferenceName, PytreeValue],
    *,
    func: Callable[..., PytreeValue],
    shared: Mapping[ReferenceName, PytreeValue],
) -> PytreeValue:
    """Evaluate one subject's cell of a tiled simulation body."""
    return func(**subject, **shared)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ArgumentsBoundAtDispatch:
    """The argument contract of a simulation program, refused until it runs.

    A simulation program's arguments are a per-call fact: the subject population
    arrives with the initial conditions, and the states it carries are the ones
    the previous period wrote. Nothing binds them at model build, so asking for
    them there is a defect in the caller, and this says so rather than handing
    back an argument mapping the body would silently disagree with.
    """

    program_name: str
    """Name of the program whose arguments the caller asked to bind."""

    subject_arg_names: tuple[ReferenceName, ...] = ()
    """Arguments partitioned across subjects; every other operand is shared."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Bind complete forward arguments, refusing a model-build context."""
        if isinstance(context, SimulationBuildContext):
            return context.call_arguments
        msg = (
            f"Simulation program {self.program_name!r} binds its arguments at "
            f"dispatch, from the simulated population of period {context.period}, "
            "not at model build."
        )
        raise ExecutionPlanningError(msg)


__all__ = [
    "SUBJECT_AXIS",
    "SUBJECT_WIDTH_KEYWORD",
    "SimulationPrograms",
    "build_simulation_programs",
]
