"""Forward simulation's per-regime work, declared as planner-owned programs.

Each regime publishes three families of `CoreProgram`, keyed by the period they
serve:

- `decision`: the argmax over the action product and the value it attains, at
  every subject's own state cell.
- `transition`: the laws of motion carrying each subject into the next period.
- `route`: the regime-transition probabilities the realized draw reads.

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
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ReducedAxis,
    ValueRead,
)
from _lcm.simulation.program_types import (
    ACTION_INDEX,
    DECISION_PROGRAM,
    DECISION_VALUE,
    ROUTE_PROGRAM,
    SUBJECT_AXIS,
    SUBJECT_WIDTH_KEYWORD,
    TRANSITION_PROGRAM,
    SimulationBuildContext,
    SimulationPrograms,
    _PerSubjectFunction,
    subject_axis,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.action_streaming import build_streaming_max_Q_over_a
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import (
    ACTION_PRODUCT_AXIS,
    _edge_reference_regimes_for_targets,
    _select_action_width_keyword,
    _supports_action_streaming,
    _value_reads,
)
from _lcm.typing import (
    ActionName,
    QAndFFunction,
    RegimeName,
    StateOrActionName,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, IntND

# Why a regime whose routing the host drives cedes its own width.
_GATED_ROUTE_REASON = "host_driven:gated_edge_adapters"


def build_simulation_programs(
    *,
    context: SolverBuildContext,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    per_subject_decisions: MappingProxyType[int, Callable[..., object]],
    per_subject_transitions: MappingProxyType[int, _PerSubjectFunction],
    per_subject_route: _PerSubjectFunction | None,
    simulation_state_names: tuple[StateOrActionName, ...],
    active_periods: tuple[int, ...],
    has_gated_edges: bool,
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

    Returns:
        The regime's declared programs.

    """
    streams_actions = _supports_action_streaming(context=context)
    if streams_actions:
        _fail_if_the_streamed_reduction_is_wrong(
            regime_name=context.regime_name,
            has_taste_shocks=context.has_taste_shocks,
            stakeholders=context.stakeholders,
        )
    action_width_keyword = _select_action_width_keyword(context=context)
    action_names = context.state_action_space.action_names
    action_extents = context.state_action_space.actions_grid_shapes

    decision_bodies: dict[int, Callable[..., object]] = {}
    decision: dict[int, CoreProgram] = {}
    for period in active_periods:
        group = id(Q_and_F_functions[period])
        if group not in decision_bodies:
            decision_bodies[group] = _decision_body(
                Q_and_F=Q_and_F_functions[period],
                dense_reducer=per_subject_decisions[period],
                context=context,
                streams_actions=streams_actions,
                action_width_keyword=action_width_keyword,
                subject_arg_names=_decision_subject_arg_names(context=context),
            )
        decision[period] = CoreProgram(
            name=DECISION_PROGRAM,
            function=decision_bodies[group],
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=DECISION_PROGRAM,
                subject_arg_names=_decision_subject_arg_names(context=context),
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
            output_roles=(ACTION_INDEX, DECISION_VALUE),
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
        )

    transition_bodies: dict[int, Callable[..., object]] = {}
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
                    tiled_axes=(
                        ()
                        if has_gated_edges
                        else (subject_axis(state_names=simulation_state_names),)
                    ),
                ),
                output_roles=per_subject_route.output_roles,
                disposition=(
                    CoreExecutionDisposition.HOST_DRIVEN
                    if has_gated_edges
                    else CoreExecutionDisposition.PLANNED
                ),
                disposition_reason=_GATED_ROUTE_REASON if has_gated_edges else None,
                donation_candidates=(),
            )

    return SimulationPrograms(
        decision=MappingProxyType(decision),
        transition=MappingProxyType(transition),
        route=MappingProxyType(route),
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


def _decision_body(
    *,
    Q_and_F: QAndFFunction,
    dense_reducer: Callable[..., object],
    context: SolverBuildContext,
    streams_actions: bool,
    action_width_keyword: str,
    subject_arg_names: tuple[str, ...],
) -> Callable[..., object]:
    """Build one period group's decision body, tiled over the subject axis."""
    if not streams_actions:
        return _SubjectTiled(func=dense_reducer, subject_arg_names=subject_arg_names)
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
    return _SubjectTiled(func=cell, subject_arg_names=subject_arg_names)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _StreamedArgmaxQOverA:
    """The argmax of Q over a streamed action product, at one state cell.

    Publishes the flat identity of the winning action in the canonical C order
    of the action product, and the value it attains — the same pair the dense
    canonical reducer publishes, obtained by the exact hard-max fold the solve
    kernel streams with. An all-infeasible cell publishes identity zero, which
    is what the dense reducer's masked `argmax` publishes there.
    """

    __name__: ClassVar[str] = "streamed_argmax_and_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: QAndFFunction
    """The regime's action value and feasibility, evaluated per action block."""

    action_names: tuple[ActionName, ...]
    """Action variable names, spanning the canonical streamed product."""

    q_and_f_arg_names: frozenset[str]
    """The argument names `Q_and_F` declares, which select what it is handed."""

    action_width_keyword: str
    """Name of the planner-bound static action-block width in the call."""

    folds: dict[int, Callable[..., Any]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    """Streamed folds already built, by the block width each streams at."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: Any,  # noqa: ANN401
    ) -> tuple[IntND, FloatND]:
        """Return the chosen action's flat identity and the value it attains."""
        block_width = cast("int", states_actions_params[self.action_width_keyword])
        q_and_f_params = {
            name: value
            for name, value in states_actions_params.items()
            if name in self.q_and_f_arg_names
        }
        result = self._fold(block_width=block_width)(
            next_regime_to_V_arr=next_regime_to_V_arr, **q_and_f_params
        )
        return (
            jnp.maximum(result.best_global_action_id, 0).astype(jnp.int32),
            result.best_value,
        )

    def _fold(self, *, block_width: int) -> Callable[..., Any]:
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

    func: Callable[..., object]
    """The body, at one subject's state cell."""

    subject_arg_names: tuple[str, ...]
    """Arguments carrying the per-subject leading axis this splits into tiles."""

    def __call__(self, **kwargs: Any) -> object:  # noqa: ANN401
        """Return the body's output for every subject, evaluated in tiles."""
        width = cast("int", kwargs.pop(SUBJECT_WIDTH_KEYWORD))
        accepted = inspect.signature(self.func).parameters
        if not self.subject_arg_names:
            return self.func(
                **{name: value for name, value in kwargs.items() if name in accepted}
            )
        tiles = {name: kwargs.pop(name) for name in self.subject_arg_names}
        shared = {name: value for name, value in kwargs.items() if name in accepted}
        return jax.lax.map(
            partial(_evaluate_subject_tile, func=self.func, shared=shared),
            tiles,
            batch_size=width,
        )


# keyword-only-exempt: library-callback=jax.lax.map
def _evaluate_subject_tile(
    subject: Mapping[str, object],
    *,
    func: Callable[..., object],
    shared: Mapping[str, object],
) -> object:
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

    subject_arg_names: tuple[str, ...] = ()
    """Arguments partitioned across subjects; every other operand is shared."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
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
