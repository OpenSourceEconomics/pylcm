"""Bound current-period query metadata for immutable original-subject replay."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from inspect import signature
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.current_iid import CurrentIIDDrawFrame, CurrentIIDDrawPlan, _draw_base_value
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.simulation.program_types import subject_axis
from _lcm.typing import FlatParams, PytreeValue
from lcm.shocks import BaseInnovations, Discrete, LawValue, Uniform
from lcm.typing import BoolND, FloatND, IntND, RegimeName

if TYPE_CHECKING:
    from _lcm.engine import Regime
    from _lcm.policy_queries import ScalarPolicyQueries
    from _lcm.solution.contract import SolverBuildContext
else:
    # Policy queries import engine, which owns this forward declaration slot.
    type ScalarPolicyQueries = object
    type Regime = object
    type SolverBuildContext = object

type ExitFramePayload = tuple[
    Mapping[str, FloatND | IntND],
    Mapping[str, FloatND],
    Mapping[str, FloatND],
    Mapping[str, Mapping[str, LawValue | BoolND]],
    Mapping[str, Mapping[str, LawValue]],
    Mapping[str, FloatND],
    BoolND,
    Mapping[str, BoolND],
]
type OriginalExitOutput = tuple[
    tuple[FloatND, IntND, BoolND, BoolND],
    ExitFramePayload,
    Mapping[str, Mapping[str, FloatND | IntND]],
    Mapping[str, FloatND],
]
type ParticipatingExitOutput = tuple[
    tuple[FloatND, IntND, BoolND, BoolND], ExitFramePayload
]


@dataclass(frozen=True, kw_only=True)
class ExitSimulationPlan:
    """Original Full choice and revealed destination Participating policies."""

    full: ScalarPolicyQueries | None
    """Original-source choice, priced with its prescribed caller information."""
    participating: Mapping[RegimeName, ScalarPolicyQueries]
    """Final-regime policies after revelation; no second Full decision."""
    physical_names: tuple[str, ...] = ()
    """Own physical columns consumed by this actor's Participating policy."""
    fixed_flat_params: FlatParams = MappingProxyType({})
    """All bound query-context operands, retaining their canonical namespace."""

    def __post_init__(self) -> None:
        """Retain a stable destination roster independent of caller mutation."""
        object.__setattr__(
            self, "participating", MappingProxyType(dict(self.participating))
        )

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> Self:
        """Bind target operands without assuming they are ordinary edge targets."""
        return replace(
            self,
            fixed_flat_params=MappingProxyType(
                {
                    owner: MappingProxyType(dict(values))
                    for owner, values in fixed_flat_params.items()
                }
            ),
        )


def install_exit_simulation_queries(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
    original_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> MappingProxyType[RegimeName, Regime]:
    """Bind phase-correct subject queries after every canonical context exists.

    This builds genuine forward programs beside the ordinary actor programs. A
    query-only counterparty remains a transient reader, not a simulated occupant.
    """
    from _lcm.current_iid import CurrentIIDPlan  # noqa: PLC0415
    from _lcm.exit_queries import bind_exit_query_graph  # noqa: PLC0415
    from _lcm.policy_queries import QueryKind, ScalarPolicyQueries  # noqa: PLC0415
    from _lcm.time import coordinate_kind  # noqa: PLC0415

    contexts = {
        owner: replace(
            context,
            current_iid_plans=MappingProxyType(
                {
                    period: CurrentIIDPlan.create(
                        bindings=context.current_iid_draw_plan.bindings,
                        age=context.ages.exact_values[period],
                        coordinate_kind=coordinate_kind(context.ages),
                    )
                    for period in context.Q_and_F_functions
                }
            )
            if context.current_iid_draw_plan is not None
            else MappingProxyType({}),
        )
        for owner, context in contexts.items()
    }
    original_actors = {
        (owner, period)
        for owner, context in contexts.items()
        for period in context.Q_and_F_functions
        if period in canonical_regimes[owner].simulation.programs.decision
        and period in original_periods_by_regime.get(owner, ())
    }
    roots = {
        (owner, period)
        for owner, period in original_actors
        if contexts[owner].user_regimes_at(period=period)[owner].exits
    }
    full_queries = {}
    actors = set(roots)
    for owner, period in sorted(roots):
        definitions, root = bind_exit_query_graph(
            contexts=contexts,
            canonical_regimes=canonical_regimes,
            root=owner,
            period=period,
            phase="simulate",
        )
        full_queries[owner, period] = ScalarPolicyQueries(
            definitions=definitions, root=root, period=period
        )
        actors.update(
            (definitions[query].read.target, period)
            for targets in definitions[root].exit_targets.values()
            for query in targets.values()
        )
    for owner, period in sorted((actors & original_actors) - full_queries.keys()):
        definitions, root = bind_exit_query_graph(
            contexts=contexts,
            canonical_regimes=canonical_regimes,
            root=owner,
            period=period,
            phase="simulate",
        )
        full_queries[owner, period] = ScalarPolicyQueries(
            definitions=definitions, root=root, period=period
        )
    participating = {}
    for owner, period in sorted(actors):
        definitions, root = bind_exit_query_graph(
            contexts=contexts,
            canonical_regimes=canonical_regimes,
            root=owner,
            period=period,
            phase="simulate",
            kind=QueryKind.PARTICIPATING,
        )
        participating[owner, period] = ScalarPolicyQueries(
            definitions=definitions, root=root, period=period
        )
    updated = dict(canonical_regimes)
    for owner, period in sorted(actors):
        queries = full_queries.get((owner, period))
        actor_queries = participating[owner, period]
        draw = contexts[owner].current_iid_draw_plan or CurrentIIDDrawPlan(bindings={})
        uniform_bases = {
            name: isinstance(law, Uniform | Discrete)
            for node in (queries or actor_queries).definitions.values()
            for binding in (
                ()
                if node.context.current_iid_draw_plan is None
                else node.context.current_iid_draw_plan.bindings.values()
            )
            if node.read.policy_information & binding.group.shocks.keys()
            for name, law in binding.group.shocks.items()
        }
        regime = updated[owner]
        programs = regime.simulation.programs
        plan = ExitSimulationPlan(
            full=queries,
            participating={
                actor: query
                for (actor, p), query in participating.items()
                if p == period
            },
            physical_names=_query_draw(queries=actor_queries, draw=draw).physical_names,
        )
        updated[owner] = replace(
            regime,
            simulation=replace(
                regime.simulation,
                exit_queries=MappingProxyType(
                    {**regime.simulation.exit_queries, period: plan}
                ),
                programs=replace(
                    programs,
                    exit_full=MappingProxyType(
                        {
                            **programs.exit_full,
                            **(
                                {
                                    period: exit_policy_program(
                                        queries=queries,
                                        draw=draw,
                                        uniform_bases=uniform_bases,
                                        original=True,
                                    )
                                }
                                if queries is not None
                                else {}
                            ),
                        }
                    ),
                    exit_participating=MappingProxyType(
                        {
                            **programs.exit_participating,
                            period: exit_policy_program(
                                queries=participating[owner, period],
                                draw=draw,
                                uniform_bases=uniform_bases,
                                original=False,
                            ),
                        }
                    ),
                ),
            ),
        )
    return MappingProxyType(updated)


def _query_draw(
    *, queries: ScalarPolicyQueries, draw: CurrentIIDDrawPlan
) -> CurrentIIDDrawPlan:
    """Keep joint groups only when this actual policy demands their physical law."""
    information = queries.definitions[queries.root].read.policy_information
    return CurrentIIDDrawPlan(
        bindings={
            name: binding
            for name, binding in draw.bindings.items()
            if information & binding.group.shocks.keys()
        }
    )


class PreparedExitDecision(NamedTuple):
    """The final actor's decision, already selected before any transition."""

    indices: IntND
    value: FloatND
    physical: Mapping[str, FloatND | IntND]


@dataclass(frozen=True, kw_only=True)
class OriginalExitDecision:
    """Evaluate one original occupant without changing its entry context.

    Named keys include target-only innovations, but their physical transforms
    remain owned by the actual mapped query. Custody alone conveys no information.
    """

    queries: ScalarPolicyQueries
    draw: CurrentIIDDrawPlan
    uniform_bases: Mapping[str, bool]

    def __call__(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        keys: Mapping[str, jax.Array],
        group_params: Mapping[str, Mapping[str, LawValue | BoolND]],
        query_arguments: Mapping[str, Mapping[str, PytreeValue]],
        _holders: BoolND,
        taste_keys: Mapping[str, jax.Array],
        overrides: Mapping[str, BaseInnovations] = MappingProxyType({}),
    ) -> OriginalExitOutput:
        """Publish source price/frame and selected destination coordinates."""
        query_arguments = {
            name: {
                **row,
                **({"taste_shock_key": taste_keys[name]} if name in taste_keys else {}),
            }
            for name, row in query_arguments.items()
        }
        fixed = query_arguments[self.queries.root]
        source = {**states, "period": fixed["period"], "age": fixed["age"]}
        frame = self.draw.realise(
            keys=keys,
            group_params=group_params,
            arguments=source,
            overrides=overrides,
        )
        # Every extra innovation uses the very same scalar producer as realise.
        # No target law is evaluated against an original source's state pool.
        extra = {
            name: _draw_base_value(key=keys[name], uniform=uniform)
            for name, uniform in self.uniform_bases.items()
            if name not in frame.addressed_bases
        }
        frame = frame._replace(
            addressed_bases={**extra, **frame.addressed_bases},
            held_bases={**extra, **frame.held_bases},
        )
        root = self.queries.definitions[self.queries.root]
        frame = _observe_query_frame(
            queries=self.queries,
            states=states,
            query_arguments=query_arguments,
            frame=frame,
            keys=keys,
            group_params=group_params,
        )
        answer = self.queries(states=states, arguments=query_arguments, frame=frame)
        projections = {}
        adjustments = {}
        for identifier, targets in root.exit_targets.items():
            for member, query in targets.items():
                node = self.queries.definitions[query]
                selected = answer[1] == identifier
                # Keep unavailable and unselected projections behind a scalar
                # condition; vmap's select conversion must not be used here.
                projections[query] = jax.lax.cond(
                    selected,
                    lambda query=query: dict(
                        self.queries.project(
                            query=query,
                            states=states,
                            arguments=query_arguments,
                            frame=frame,
                        )
                    ),
                    lambda node=node: {
                        name: jnp.zeros((), dtype=grid.dtype)
                        for name, grid in node.context.state_action_space.states.items()
                    },
                )
                helper = root.exit_adjustments.get(identifier, {}).get(member)
                operands = {
                    **fixed,
                    **cast("Mapping[str, PytreeValue]", fixed["_lcm_query_projection"]),
                    **states,
                    **frame.physical,
                }
                adjustments[query] = (
                    jnp.asarray(0.0)
                    if helper is None
                    else jax.lax.cond(
                        selected,
                        lambda helper=helper, operands=operands: jnp.asarray(
                            helper(
                                **{
                                    name: operands[name]
                                    for name in signature(helper).parameters
                                }
                            )
                        ),
                        lambda: jnp.asarray(0.0),
                    )
                )
        return answer, _frame_output(frame=frame), projections, adjustments


@dataclass(frozen=True, kw_only=True)
class RevealedExitDecision:
    """Act in the final regime using held original-row innovations."""

    queries: ScalarPolicyQueries
    draw: CurrentIIDDrawPlan
    uniform_bases: Mapping[str, bool]

    def __call__(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        keys: Mapping[str, jax.Array],
        held_bases: Mapping[str, FloatND],
        group_params: Mapping[str, Mapping[str, LawValue | BoolND]],
        query_arguments: Mapping[str, Mapping[str, PytreeValue]],
        _holders: BoolND,
        taste_keys: Mapping[str, jax.Array],
    ) -> ParticipatingExitOutput:
        """Reveal target physical values, then run Participating exactly once."""
        query_arguments = {
            name: {
                **row,
                **({"taste_shock_key": taste_keys[name]} if name in taste_keys else {}),
            }
            for name, row in query_arguments.items()
        }
        fixed = query_arguments[self.queries.root]
        frame = self.draw.realise(
            keys=keys,
            group_params=group_params,
            arguments={**states, "period": fixed["period"], "age": fixed["age"]},
            overrides={
                name: BaseInnovations(values=held_bases[name])
                for name in self.draw.physical_names
            },
        )
        frame = frame._replace(
            addressed_bases={
                name: _draw_base_value(key=keys[name], uniform=uniform)
                for name, uniform in self.uniform_bases.items()
            },
            held_bases={name: held_bases[name] for name in self.uniform_bases},
        )
        frame = _observe_query_frame(
            queries=self.queries,
            states=states,
            query_arguments=query_arguments,
            frame=frame,
            keys=keys,
            group_params=group_params,
        )
        return (
            self.queries(states=states, arguments=query_arguments, frame=frame),
            _frame_output(frame=frame),
        )


def _observe_query_frame(
    *,
    queries: ScalarPolicyQueries,
    states: Mapping[str, FloatND | IntND],
    query_arguments: Mapping[str, Mapping[str, PytreeValue]],
    frame: CurrentIIDDrawFrame,
    keys: Mapping[str, jax.Array],
    group_params: Mapping[str, Mapping[str, LawValue | BoolND]],
) -> CurrentIIDDrawFrame:
    """Transform only explicit observations in their exact producing context."""
    root = queries.definitions[queries.root]
    for group, producer in root.observation_sources.items():
        if producer == queries.root:
            continue
        active = queries.observation_available(
            query=queries.root,
            group=group,
            states=states,
            arguments=query_arguments,
            frame=frame,
        )
        produce = partial(
            _produce_observed_group,
            queries=queries,
            states=states,
            query_arguments=query_arguments,
            frame=frame,
            keys=keys,
            group_params=group_params,
            group=group,
            producer=producer,
        )
        # Abstract tracing fixes the exact output tree without executing an
        # inactive projection or physical law. The false payload is never an
        # observation: the independent availability leaves below say so.
        prototype = jax.eval_shape(produce)
        neutral = CurrentIIDDrawFrame(
            *jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), prototype)
        )._replace(valid=jnp.asarray(a=True))
        neutral_payload = _frame_output(frame=neutral)
        produced = CurrentIIDDrawFrame(
            *jax.lax.cond(active, produce, lambda neutral=neutral_payload: neutral)
        )
        observed = {
            name: value
            for name, value in produced.physical.items()
            if name in root.read.policy_information
        }
        frame = frame._replace(
            physical={**frame.physical, **observed},
            source_params={**frame.source_params, **produced.source_params},
            source_inputs={**frame.source_inputs, **produced.source_inputs},
            information_available={
                **frame.information_available,
                **dict.fromkeys(observed, active),
            },
            valid=frame.valid & produced.valid,
        )
    return frame


def _produce_observed_group(
    *,
    queries: ScalarPolicyQueries,
    states: Mapping[str, FloatND | IntND],
    query_arguments: Mapping[str, Mapping[str, PytreeValue]],
    frame: CurrentIIDDrawFrame,
    keys: Mapping[str, jax.Array],
    group_params: Mapping[str, Mapping[str, LawValue | BoolND]],
    group: str,
    producer: str,
) -> ExitFramePayload:
    """Return the producer's numerical leaves across JAX's metadata boundary."""
    node = queries.definitions[producer]
    producing_states = queries.observation_inputs(
        query=queries.root,
        group=group,
        states=states,
        arguments=query_arguments,
        frame=frame,
    )
    fixed = query_arguments[producer]
    draw = node.context.current_iid_draw_plan
    if draw is None:
        raise ValueError("An observation producer must own its declared shock law.")
    binding = draw.bindings[group]
    return _frame_output(
        frame=CurrentIIDDrawPlan(bindings={group: binding}).realise(
            keys=keys,
            group_params=group_params,
            arguments={
                **producing_states,
                "period": fixed["period"],
                "age": fixed["age"],
            },
            overrides={
                name: BaseInnovations(values=frame.held_bases[name])
                for name in binding.group.shocks
            },
        )
    )


def _frame_output(*, frame: CurrentIIDDrawFrame) -> ExitFramePayload:
    """Keep one declared leaf roster for ordinary, base and physical inputs."""
    return (
        dict(frame.physical),
        dict(frame.addressed_bases),
        dict(frame.held_bases),
        {name: dict(params) for name, params in frame.source_params.items()},
        {name: dict(inputs) for name, inputs in frame.source_inputs.items()},
        {
            name: frame.recovery_error.get(name, jnp.asarray(0.0))
            for name in frame.physical
        },
        frame.valid,
        dict(frame.information_available),
    )


@dataclass(frozen=True, kw_only=True)
class _SequentialExitDecision:
    """Give JAX's positional batching boundary one explicit argument tree."""

    body: Callable[..., OriginalExitOutput | ParticipatingExitOutput]

    def __call__(
        self, **arguments: PytreeValue
    ) -> OriginalExitOutput | ParticipatingExitOutput:
        """Retain scalar conditions through the outer subject vmap."""
        return jax.custom_batching.sequential_vmap(
            partial(_evaluate_exit_decision, body=self.body)
        )(arguments)


# keyword-only-exempt: library-callback=jax.custom_batching.sequential_vmap
def _evaluate_exit_decision(
    arguments: Mapping[str, PytreeValue],
    *,
    body: Callable[..., OriginalExitOutput | ParticipatingExitOutput],
) -> OriginalExitOutput | ParticipatingExitOutput:
    """Evaluate one canonical subject body after sequential batching."""
    return body(**arguments)


def exit_policy_program(
    *,
    queries: ScalarPolicyQueries,
    draw: CurrentIIDDrawPlan,
    uniform_bases: Mapping[str, bool],
    original: bool,
) -> CoreProgram:
    """Declare the native subject program and every retained draw-frame leaf."""
    # Imports stay below engine construction: programs imports the engine's
    # forward declarations, while this builder runs only after contexts exist.
    from _lcm.current_iid import CORRELATION_SUPPORT_ARGUMENT  # noqa: PLC0415
    from _lcm.simulation.programs import (  # noqa: PLC0415
        _ArgumentsBoundAtDispatch,
        _SubjectTiled,
    )

    name = "simulate_exit_full" if original else "simulate_exit_participating"
    draw = _query_draw(queries=queries, draw=draw)
    body = (
        OriginalExitDecision(queries=queries, draw=draw, uniform_bases=uniform_bases)
        if original
        else RevealedExitDecision(
            queries=queries, draw=draw, uniform_bases=uniform_bases
        )
    )
    subject_inputs = ("states", "keys", "_holders", "taste_keys") + (
        () if original else ("held_bases",)
    )
    root = queries.definitions[queries.root]
    producing_bindings = dict(draw.bindings)
    for group, producer in root.observation_sources.items():
        producing_draw = queries.definitions[producer].context.current_iid_draw_plan
        if producing_draw is None:
            raise ValueError("An observation producer must own its declared shock law.")
        producing_bindings[group] = producing_draw.bindings[group]
    physical_names = tuple(
        dict.fromkeys(
            (
                *draw.physical_names,
                *(
                    name
                    for binding in producing_bindings.values()
                    for name in binding.group.shocks
                    if name in root.read.policy_information
                ),
            )
        )
    )
    frame_roles = (
        dict.fromkeys(physical_names, "physical_shock"),
        dict.fromkeys(uniform_bases, "addressed_base"),
        dict.fromkeys(uniform_bases, "held_base"),
        {
            name: dict.fromkeys(
                (
                    *binding.params_template,
                    *(
                        (CORRELATION_SUPPORT_ARGUMENT,)
                        if binding.group.correlation is None
                        else ()
                    ),
                ),
                "source_law_parameter",
            )
            for name, binding in producing_bindings.items()
        },
        {
            name: dict.fromkeys(
                (*binding.state_inputs, "period", "age", *binding.functions),
                "source_law_input",
            )
            for name, binding in producing_bindings.items()
        },
        dict.fromkeys(physical_names, "physical_recovery_error"),
        "law_validity",
        {
            name: "producing_context_available"
            for group, producer in root.observation_sources.items()
            if producer != queries.root
            for name in producing_bindings[group].group.shocks
            if name in root.read.policy_information
        },
    )
    answer_roles = ("decision_value", "action_index", "support", "integrity")
    targets = {
        query for members in root.exit_targets.values() for query in members.values()
    }
    roles = (
        (
            answer_roles,
            frame_roles,
            {
                query: dict.fromkeys(
                    queries.definitions[query].read.projection,
                    "projected_current_state",
                )
                for query in targets
            },
            dict.fromkeys(targets, "source_utility_adjustment"),
        )
        if original
        else (answer_roles, frame_roles)
    )
    return CoreProgram(
        name=name,
        # Sequential batching keeps unavailable benchmark/projection arithmetic
        # behind scalar conditions instead of converting them to eager selects.
        function=_SubjectTiled(
            func=with_signature(
                _SequentialExitDecision(body=body),
                args=list(signature(body).parameters),
                enforce=False,
            ),
            subject_arg_names=subject_inputs,
        ),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name=name, subject_arg_names=subject_inputs
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=()),),
            value_reads=queries.value_reads(core_key=name),
        ),
        output_roles=roles,
        disposition=CoreExecutionDisposition.PLANNED,
    )


class ExitRoute(NamedTuple):
    """One selected member's exact original-source to final-actor route."""

    source: RegimeName
    target: RegimeName
    identifier: int
    source_role: int | None
    target_role: int
    projection: str
    """Key of the exact producer query, retaining mapped-law provenance."""


type ExitRouteFields = tuple[RegimeName, RegimeName, int, int | None, int, str]


def route_original_subjects(
    *,
    original_regime_ids: IntND,
    original_roles: IntND,
    original_states: Mapping[RegimeName, Mapping[str, FloatND | IntND]],
    decisions: Mapping[RegimeName, IntND],
    projections: Mapping[str, Mapping[str, FloatND | IntND]],
    adjustments: Mapping[str, FloatND],
    routes: tuple[ExitRouteFields, ...],
    regime_ids: Mapping[RegimeName, IntND | int],
) -> tuple[
    IntND,
    IntND,
    dict[RegimeName, dict[str, FloatND | IntND]],
    BoolND,
    FloatND,
]:
    """Apply every route against immutable original membership and state.

    In particular, a row entering a regime whose original occupants leave cannot
    take that regime's exit. Its destination coordinates came from the original
    source's bound projection, before any row's state or membership was changed.
    """
    final_ids = original_regime_ids
    final_roles = original_roles
    states = {name: dict(values) for name, values in original_states.items()}
    moved = jnp.zeros_like(original_regime_ids, dtype=jnp.bool_)
    paid = jnp.zeros_like(original_regime_ids, dtype=jnp.asarray(0.0).dtype)
    for source, target, identifier, source_role, target_role, projection in routes:
        selected = (original_regime_ids == regime_ids[source]) & (
            decisions[source] == identifier
        )
        if source_role is not None:
            selected &= original_roles == source_role
        final_ids = jnp.where(selected, regime_ids[target], final_ids)
        final_roles = jnp.where(selected, target_role, final_roles)
        moved |= selected
        paid = jnp.where(selected, adjustments[projection], paid)
        for name, value in projections[projection].items():
            states[target][name] = jnp.where(selected, value, states[target][name])
    return final_ids, final_roles, states, moved, paid
