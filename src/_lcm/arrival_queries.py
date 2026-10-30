"""Bind edge-owned arrival comparisons to actual dated policy/value readers."""

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature
from dags.signature import rename_arguments
from dags.tree import qname_from_tree_path

from _lcm.arrival_continuation import (
    ARRIVAL_CONTINUATION,
    ArrivalContinuation,
    ArrivalOperand,
)
from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    CurrentIIDPlan,
)
from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime
from _lcm.execution.core_program import CoreArgumentBuilder, CoreBuildContext, ValueRead
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.exit_queries import (
    _bind_observation_sources,
    _compose,
    _QueryGraphBuilder,
    _validate_query_projection_bindings,
)
from _lcm.params.edges import regime_kernel_params
from _lcm.policy_queries import (
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    ScalarPolicyQueries,
    _admit_node,
)
from _lcm.regime_building.Q_and_F import _CollectiveQAndF, _ComputeCE, _QAndF
from _lcm.regime_building.V import get_V_interpolator
from _lcm.regime_building.value_adapter import (
    _AdaptedArrivalPredicate,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import (
    ArrivalQueryArguments,
    FlatParams,
    PytreeValue,
    QAndFKwargs,
    ShapeDtypePytree,
    ValueTemplate,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import NamedOperand, call_with_declared_arguments
from lcm.arrivals import ArrivalDecision, ArrivalRead
from lcm.exceptions import (
    ExecutionPlanningError,
    RegimeInitializationError,
    UnsupportedOperationError,
)
from lcm.shocks import Discrete
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    Phase,
    ReferenceName,
    RegimeName,
    UserFunction,
)


@dataclass(frozen=True, kw_only=True)
class ArrivalQueryPlan:
    """One incoming edge's exact-date read graph and original source roles."""

    source: RegimeName
    target: RegimeName
    period: int
    phase: Phase
    members: tuple[str, ...]
    queries: ScalarPolicyQueries
    transport_template: ArrivalContinuation | None = None
    transport_origins: Mapping[str, ValueArtifactAddress] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, kw_only=True)
class _ArrivalBuildContext(CoreBuildContext):
    """Call-local demanded leaves used when a future Q itself contains arrivals."""

    arrival_values: Mapping[ValueArtifactAddress, ValueTemplate] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _transport_name(*, origin: ValueArtifactAddress) -> str:
    key = origin.artifact_key
    identity = (
        origin.kind.value,
        origin.period,
        origin.regime,
        origin.target_regime,
        None if key is None else (key.type_id, key.schema_version),
        origin.leaf_path,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def _transport_alias(*, plan: ArrivalQueryPlan, name: str) -> ValueArtifactAddress:
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=plan.period,
        regime=plan.target,
        artifact_key=ARRIVAL_CONTINUATION,
        leaf_path=("operands", name),
    )


def _arrival_values(
    *, context: CoreBuildContext, plan: ArrivalQueryPlan
) -> Mapping[ValueArtifactAddress, ValueTemplate]:
    required = {
        read.target
        for read in plan.queries.value_reads(core_key="arrival")
        if read.target.period > plan.period
    }
    if not required:
        return MappingProxyType({})
    template = plan.transport_template
    if template is None:
        raise UnsupportedOperationError(
            "This arrival policy requires its declared later-value transport."
        )
    if isinstance(context, _ArrivalBuildContext):
        return MappingProxyType(
            {
                origin: context.arrival_values[
                    _transport_alias(plan=plan, name=_transport_name(origin=origin))
                ]
                for origin in required
            }
        )
    payload = context.next_regime_to_continuation.get(plan.target)
    if not isinstance(payload, ArrivalContinuation) or (
        payload.regime,
        payload.period,
        payload.schema,
        payload.names,
    ) != (template.regime, template.period, template.schema, template.names):
        raise ExecutionPlanningError(
            "An arrival requires its exact declared producer-date operand payload."
        )
    return MappingProxyType(
        {
            origin: payload.operands[_transport_name(origin=origin)]
            for origin in required
        }
    )


@dataclass(frozen=True, kw_only=True)
class ArrivalSourceArguments:
    """Actual arrival-date dynamic query operands beside the source's own inputs."""

    base: CoreArgumentBuilder
    plans: Mapping[RegimeName, ArrivalQueryPlan]

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        arguments = dict(self.base(context))
        arrivals = {}
        for target, plan in self.plans.items():
            transported = _arrival_values(context=context, plan=plan)
            arrival_context = _ArrivalBuildContext(
                **{
                    declaration.name: getattr(context, declaration.name)
                    for declaration in fields(CoreBuildContext)
                    if declaration.name
                    not in {
                        "period",
                        "next_regime_to_V_arr",
                        "same_period_regime_to_V_arr",
                        "next_regime_to_continuation",
                    }
                },
                period=plan.period,
                next_regime_to_V_arr={
                    origin.regime: value
                    for origin, value in transported.items()
                    if origin.kind is ValueArtifactKind.REGIME_VALUE
                    and origin.period == plan.period + 1
                },
                same_period_regime_to_V_arr=context.next_regime_to_V_arr,
                next_regime_to_continuation={},
                arrival_values=transported,
            )
            arrivals[target] = plan.queries.arguments(context=arrival_context)
        arguments["arrival_arguments"] = ArrivalQueryArguments(
            by_target=cast("Mapping[RegimeName, Mapping[str, QAndFKwargs]]", arrivals)
        )
        return MappingProxyType(arguments)


def arrival_source_value_reads(
    *, plans: Mapping[RegimeName, ArrivalQueryPlan], source: RegimeName, period: int
) -> tuple[ValueRead, ...]:
    """Name every stored comparison at its actual date in the source operand tree."""
    return tuple(
        replace(
            read,
            target=_transport_alias(plan=plan, name=_transport_name(origin=read.target))
            if read.target.period > plan.period
            else read.target,
            source=replace(
                read.source,
                source_period=period,
                source_regime=source,
                core_key="main",
                channel=ValueInputChannel.CONTINUATION_LEAF
                if read.target.period > plan.period
                else ValueInputChannel.NEXT_REGIME_VALUE,
                argument="arrival_arguments",
                path=("by_target", target, *read.source.path),
            ),
        )
        for target, plan in plans.items()
        for read in plan.queries.value_reads(core_key="arrival")
    )


@dataclass(frozen=True, kw_only=True)
class _ArrivalPolicyValue:
    """Price the bound Boolean routing at this action's actual target landing."""

    plan: ArrivalQueryPlan

    def __call__(
        self, **operands: PytreeValue
    ) -> tuple[FloatND, BoolND] | tuple[tuple[FloatND, BoolND], IntND]:
        arguments = cast("ArrivalQueryArguments", operands["arrival_arguments"])
        states = {
            name.removeprefix("next_"): value
            for name, value in operands.items()
            if name.startswith("next_") and name != "next_V_arr"
        }
        answer = self.plan.queries(
            states=cast("Mapping[str, FloatND]", states),
            arguments=arguments.by_target[self.plan.target],
        )
        value_and_integrity = (answer[0], answer[3])
        return (
            (value_and_integrity, answer[4])
            if self.plan.queries.landing_checks
            else value_and_integrity
        )


def _source_action_problem(
    *,
    context: SolverBuildContext,
    period: int,
    plans: Mapping[RegimeName, ArrivalQueryPlan],
) -> _QAndF | _CollectiveQAndF:
    """Replace only the actual continuation reader; keep canonical flow and laws."""
    function = context.Q_and_F_functions[period]
    if not isinstance(function, _QAndF | _CollectiveQAndF) or not isinstance(
        function.compute_CE, _ComputeCE
    ):
        raise UnsupportedOperationError(
            "Arrival pricing needs its canonical GridSearch Q/F continuation; "
            "endogenous-grid arrival reconstruction is an S10 dependency."
        )
    compute = function.compute_CE
    continuations = dict(compute.continuations)
    scalar_readers = dict(compute.arrival_scalar_readers)
    scalar_names = dict(compute.arrival_scalar_arg_names)
    scalar_checks = dict(compute.arrival_scalar_landing_checks)
    for target, plan in plans.items():
        target_names = plan.queries.definitions[plan.queries.root].read.projection
        names = [
            *(f"next_{name}" for name in target_names),
            "next_V_arr",
            "arrival_arguments",
        ]
        reader = with_signature(
            _ArrivalPolicyValue(plan=plan), args=names, enforce=False
        )
        if target in compute.scalar_targets:
            scalar_readers[target] = reader
            scalar_names[target] = ("arrival_arguments",)
            scalar_checks[target] = plan.queries.landing_checks
            continue
        original = continuations[target]
        if (
            original.integrated_weight_names
            or original.draw_dependent_names
            or original.landing_checks
        ):
            raise UnsupportedOperationError(
                "This arrival continuation needs its declared draw/landing "
                "reader adapter."
            )
        continuations[target] = replace(
            original,
            next_V=productmap(
                func=reader,
                variables=original.lottery_axis_names,
                batch_sizes=dict.fromkeys(original.lottery_axis_names, 0),
            ),
            extra_param_names=frozenset({"arrival_arguments"}),
            original_node_axes=original.original_node_axes
            or original.original_value_axes,
            original_value_axes=(),
            arrival_query=True,
            landing_checks=plan.queries.landing_checks,
        )
    compute = replace(
        compute,
        continuations=MappingProxyType(continuations),
        arrival_scalar_readers=MappingProxyType(scalar_readers),
        arrival_scalar_arg_names=MappingProxyType(scalar_names),
        arrival_scalar_landing_checks=tuple(scalar_checks.items()),
    )
    return replace(
        function,
        compute_CE=compute,
        landing_checks=compute.landing_checks,
        arg_names=(*function.arg_names, "arrival_arguments"),
        continuation_reads=None
        if function.continuation_reads is None
        else function.continuation_reads | {"arrival_arguments"},
    )


def _bind_transport_templates(
    *,
    plans_by_source: Mapping[RegimeName, Mapping[RegimeName, ArrivalQueryPlan]],
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    origin_templates: dict[ValueArtifactAddress, ArrivalOperand],
    dated_templates: dict[tuple[RegimeName, int], ArrivalContinuation],
) -> dict[RegimeName, dict[RegimeName, ArrivalQueryPlan]]:
    """Union the actual incoming readers of each one producer-date payload."""
    origins: dict[RegimeName, dict[str, ValueArtifactAddress]] = {}
    for plans in plans_by_source.values():
        for target, plan in plans.items():
            for read in plan.queries.value_reads(core_key="arrival"):
                if read.target.period <= plan.period:
                    continue
                if read.view is not None:
                    raise UnsupportedOperationError(
                        "Arrival transport requires a whole stored operand leaf."
                    )
                origins.setdefault(target, {})[_transport_name(origin=read.target)] = (
                    read.target
                )
    for target, unsorted in origins.items():
        declarations = dict(sorted(unsorted.items()))
        operands = {}
        for name, origin in declarations.items():
            if origin not in origin_templates:
                origin_templates[origin] = _origin_template(
                    origin=origin,
                    context=contexts[origin.regime],
                    regime=canonical_regimes[origin.regime],
                    dated_templates=dated_templates,
                )
            operands[name] = origin_templates[origin]
        period = next(
            plan.period
            for plans in plans_by_source.values()
            for name, plan in plans.items()
            if name == target
        )
        schema = hashlib.sha256(repr(tuple(declarations)).encode()).hexdigest()
        template = ArrivalContinuation(
            regime=target, period=period, schema=schema, operands=operands
        )
        dated_templates[target, period] = template
        origins[target] = declarations
    return {
        source: {
            target: replace(
                plan,
                transport_template=dated_templates.get((target, plan.period)),
                transport_origins=MappingProxyType(origins.get(target, {})),
            )
            for target, plan in plans.items()
        }
        for source, plans in plans_by_source.items()
    }


def _origin_template(
    *,
    origin: ValueArtifactAddress,
    context: SolverBuildContext,
    regime: Regime,
    dated_templates: Mapping[tuple[RegimeName, int], ArrivalContinuation],
) -> ArrivalOperand:
    if origin.kind is ValueArtifactKind.REGIME_VALUE:
        shape = tuple(
            nodes.size for nodes in context.state_action_space.states.values()
        )
        if context.stakeholders is not None:
            shape = (*shape, len(context.stakeholders))
        return context.place_on_regime_devices(
            template=jnp.zeros(shape, dtype=canonical_float_dtype())
        )
    if origin.kind is ValueArtifactKind.CONTINUATION_LEAF:
        payload = dated_templates.get((origin.regime, origin.period))
        if payload is not None:
            return payload.leaves()[origin.leaf_path]
        spec = regime.solution.continuation_spec
        if (
            spec is None
            or spec.artifact_key != ARRIVAL_CONTINUATION
            or spec.artifact_key != origin.artifact_key
        ):
            raise UnsupportedOperationError(
                "An arrival needs its declared future arrival schema; "
                "other continuation intersections are an S10 dependency."
            )
        payload = spec.template_at(period=origin.period)
        if not isinstance(payload, ArrivalContinuation):
            raise TypeError("An arrival schema needs its typed numeric payload.")
        return payload.leaves()[origin.leaf_path]
    raise UnsupportedOperationError(
        "Arrival custody requires a stored value or continuation leaf."
    )


def install_arrival_source_queries(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> MappingProxyType[RegimeName, SolverBuildContext]:
    """Bind future policy graphs after their actual later action problems exist."""
    updated = dict(contexts)
    origin_templates: dict[ValueArtifactAddress, ArrivalOperand] = {}
    dated_templates: dict[tuple[RegimeName, int], ArrivalContinuation] = {}
    periods = sorted(
        {
            period
            for context in contexts.values()
            for period in context.Q_and_F_functions
        },
        reverse=True,
    )
    for period in periods:
        plans_by_source = {}
        for source, context in tuple(updated.items()):
            if period not in context.Q_and_F_functions:
                continue
            law = context.laws[source]
            if (
                law.terminal
                or not law.arrivals
                or period >= context.solution_reachability.n_periods - 1
            ):
                continue
            targets = context.solution_reachability.targets(
                period=period, source=source
            )
            plans = {
                target: bind_arrival_query_graph(
                    contexts=updated,
                    canonical_regimes=canonical_regimes,
                    source=source,
                    target=target,
                    period=period + 1,
                    phase="solve",
                    declaration=law.arrivals[target],
                )
                for target in targets
                if target in law.arrivals
            }
            if plans:
                plans_by_source[source] = plans
        plans_by_source = _bind_transport_templates(
            plans_by_source=plans_by_source,
            contexts=updated,
            canonical_regimes=canonical_regimes,
            origin_templates=origin_templates,
            dated_templates=dated_templates,
        )
        for source, plans in plans_by_source.items():
            context = updated[source]
            updated[source] = replace(
                context,
                Q_and_F_functions=MappingProxyType(
                    {
                        **context.Q_and_F_functions,
                        period: _source_action_problem(
                            context=context, period=period, plans=plans
                        ),
                    }
                ),
                arrival_queries=MappingProxyType(
                    {
                        **context.arrival_queries,
                        period: MappingProxyType(plans),
                    }
                ),
            )
    return MappingProxyType(updated)


def install_arrival_replay_sources(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    pricing_contexts: Mapping[RegimeName, SolverBuildContext],
) -> MappingProxyType[RegimeName, SolverBuildContext]:
    """Keep actual current flow beside the source's perceived arrival continuation."""
    updated = dict(contexts)
    for source, context in contexts.items():
        plans_by_period = {
            period: plans
            for period, plans in pricing_contexts[source].arrival_queries.items()
            if period in context.Q_and_F_functions
        }
        if not plans_by_period:
            continue
        functions = dict(context.Q_and_F_functions)
        for period, plans in plans_by_period.items():
            functions[period] = _source_action_problem(
                context=context, period=period, plans=plans
            )
        updated[source] = replace(
            context,
            Q_and_F_functions=MappingProxyType(functions),
            arrival_queries=MappingProxyType(plans_by_period),
        )
    return MappingProxyType(updated)


@dataclass(frozen=True, kw_only=True)
class _ArrivalReducer:
    """Apply a Boolean comparison, then route each original source member once."""

    predicate: Callable[..., PytreeValue]
    reads: Mapping[str, tuple[str, str, int | None]]
    target_indices: tuple[int | None, ...]
    fallbacks: tuple[tuple[str, int | None], ...]
    collective: bool

    def __call__(self, **operands: PytreeValue) -> PolicyAnswer:
        values = dict(operands)
        valid = jnp.asarray(a=True)
        for name, (producer, read_field, index) in self.reads.items():
            answer = cast("PolicyAnswer", operands[producer])
            values[name] = (
                answer[2]
                if read_field == "support"
                else answer[0]
                if index is None
                else answer[0][index]
            )
            valid &= answer[3]
        accepted = jnp.asarray(
            call_with_declared_arguments(function=self.predicate, operands=values)
        )
        if accepted.shape != () or accepted.dtype != jnp.bool_:
            raise ValueError("An arrival predicate must return a scalar Boolean.")
        target = cast("PolicyAnswer", operands["_lcm_arrival_acting_target"])
        accepted &= target[2]
        valid &= target[3]
        routed = []
        support = jnp.asarray(a=True)
        for target_index, (producer, fallback_index) in zip(
            self.target_indices, self.fallbacks, strict=True
        ):
            fallback = cast("PolicyAnswer", operands[producer])
            target_value = (
                target[0] if target_index is None else target[0][target_index]
            )
            fallback_value = (
                fallback[0] if fallback_index is None else fallback[0][fallback_index]
            )
            routed.append(jnp.where(accepted, target_value, fallback_value))
            support &= accepted | fallback[2]
            valid &= accepted | fallback[3]
        value = jnp.stack(routed) if self.collective else routed[0]
        # This is a routing branch ID, never a maximizer over the comparisons.
        return (
            value,
            (~accepted).astype(jnp.int32),
            support,
            valid,
            jnp.zeros((0,), jnp.int32),
        )


@dataclass(frozen=True, kw_only=True)
class _ArrivalArguments:
    """Only current-law numeric operands, without an unrelated target Q bank."""

    period: int
    plan: CurrentIIDPlan | None

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        if context.ages is None:
            raise ValueError("An arrival reader needs its actual economic age.")
        arguments: dict[str, PytreeValue | ShapeDtypePytree] = {
            "period": jnp.int32(self.period),
            "age": context.ages.values[self.period],
        }
        if self.plan is not None:
            arguments[CURRENT_IID_ARGUMENT] = {
                name: dict(context.flat_params.get(name, {}))
                for name, _, _ in self.plan.bindings
            }
            arguments[CURRENT_IID_NODES_ARGUMENT] = self.plan.node_operands
            arguments[CURRENT_IID_ROWS_ARGUMENT] = self.plan.indices
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True)
class _IntegratedArguments:
    """The explicitly named stored Full array and its own interpolation params."""

    owner: RegimeName
    period: int
    helper_names: tuple[str, ...]

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        if context.same_period_regime_to_V_arr is None:
            raise ValueError("An IntegratedFull arrival read needs its dated stored V.")
        if context.ages is None:
            raise ValueError("An IntegratedFull arrival read needs its actual age.")
        params = regime_kernel_params(
            cast("FlatParams", context.flat_params), regime_name=self.owner
        )
        return MappingProxyType(
            {
                "_stored_full": context.same_period_regime_to_V_arr[self.owner],
                "period": jnp.int32(self.period),
                "age": context.ages.values[self.period],
                **{name: params[name] for name in self.helper_names},
            }
        )


@dataclass(frozen=True, kw_only=True)
class _IntegratedValue:
    """Read a represented integrated value without claiming policy support."""

    interpolate: Callable[..., PytreeValue]
    member: int | None

    def __call__(self, **operands: PytreeValue) -> PolicyAnswer:
        value_array = cast("FloatND", operands["_stored_full"])
        if self.member is not None:
            value_array = value_array[..., self.member]
        value = jnp.asarray(
            call_with_declared_arguments(
                function=self.interpolate,
                operands={**operands, "_stored_full": value_array},
            )
        )
        if value.shape != ():
            raise ValueError("An IntegratedFull comparison must read one named value.")
        return (
            value,
            jnp.asarray(-1, jnp.int32),
            jnp.asarray(a=True),
            ~jnp.isnan(value),
            jnp.zeros((0,), jnp.int32),
        )


def _make_integrated_read(
    *,
    builder: _QueryGraphBuilder,
    read: ArrivalRead,
    projection: Mapping[str, Callable[..., PytreeValue]],
    root: str,
) -> str:
    """Declare a shock-free stored representation with explicit native authority."""
    context = builder.contexts[read.regime]
    if (
        context.fold_state_names
        or context.co_map_state_names
        or context.invariant_bindings
    ):
        raise UnsupportedOperationError(
            "IntegratedFull arrival interpolation needs an explicit unblocked "
            "stored schema."
        )
    infos = context.regime_to_v_interpolation_info
    if context.period_to_regime_v_interp is not None:
        infos = context.period_to_regime_v_interp[builder.period]
    info = infos[read.regime]
    interpolate = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="",
        V_arr_name="_stored_full",
        interpolate_process_axes=True,
    )
    names = tuple(inspect.signature(interpolate).parameters)
    helper_names = tuple(
        name for name in names if name not in {*info.state_names, "_stored_full"}
    )
    key = f"query_{len(builder.keys)}"
    builder.keys.append(key)
    builder.nodes[key] = QueryNode(
        read=QueryRead(
            kind=QueryKind.INTEGRATED_FULL,
            target=read.regime,
            phase=builder.phase,
            period=builder.period,
            role=read.stakeholder or "",
            projection=cast("Mapping[str, UserFunction]", projection),
            state_inputs=frozenset(info.state_names),
            physical_inputs=frozenset(),
            operand_paths={
                name: path
                for name, path in builder.operand_paths.items()
                if any(
                    name in inspect.signature(function).parameters
                    for function in projection.values()
                )
            },
            policy_information=frozenset(),
            price_information=frozenset(),
        ),
        context=replace(
            context, current_iid_plans=MappingProxyType({builder.period: None})
        ),
        decision=with_signature(
            _IntegratedValue(
                interpolate=interpolate,
                member=_member_index(context=context, role=read.stakeholder),
            ),
            args=list(names),
            enforce=False,
        ),
        operand_builder=_IntegratedArguments(
            owner=read.regime, period=builder.period, helper_names=helper_names
        ),
        operand_value_reads=(
            ValueRead(
                target=ValueArtifactAddress(
                    kind=ValueArtifactKind.REGIME_VALUE,
                    period=builder.period,
                    regime=read.regime,
                ),
                source=ValueConsumerAddress(
                    source_period=builder.period,
                    source_regime=read.regime,
                    core_key=key,
                    channel=ValueInputChannel.SAME_PERIOD_VALUE,
                    argument="_stored_full",
                    path=(),
                ),
            ),
        ),
        ancestors=(root,),
        source_projection=cast("Mapping[str, UserFunction]", projection),
        answer_shape=(),
    )
    return key


def _member_index(*, context: SolverBuildContext, role: str | None) -> int | None:
    """Validate an explicitly declared destination role without picking a member."""
    members = context.stakeholders
    if members is None:
        if role is not None:
            raise RegimeInitializationError("A singleton arrival target has no role.")
        return None
    if role not in members:
        raise RegimeInitializationError(
            "A collective arrival read needs its exact role."
        )
    return members.index(role)


def _bound_edge_function(
    *,
    builder: _QueryGraphBuilder,
    source: RegimeName,
    target: RegimeName,
    path: tuple[str, ...],
    function: UserFunction,
    information: frozenset[str],
    read_names: Mapping[ReferenceName, ReferenceName] = MappingProxyType({}),
) -> Callable[..., PytreeValue]:
    """Bind source edge params while expanding helpers in landed target vocabulary."""
    if isinstance(function, _AdaptedArrivalPredicate):
        read_names = {
            function.read_aliases.get(name, name): alias
            for name, alias in read_names.items()
        }
        function = function.function
    branch = builder.canonical_regimes[source].edge_params_template
    for segment in path:
        child = branch.get(segment, {})
        branch = child if isinstance(child, Mapping) else {}
    _validate_arrival_information(
        builder=builder,
        target=target,
        function=function,
        edge_parameters=frozenset(
            name for name, value in branch.items() if isinstance(value, str)
        ),
        read_names=frozenset(read_names),
        information=information,
    )
    callback = rename_arguments(
        function,
        mapper={
            **{
                name: builder.aliases[source, qname_from_tree_path((*path, name))]
                for name in inspect.signature(function).parameters
                if isinstance(branch.get(name), str)
            },
            **read_names,
        },
    )
    combined = concatenate_functions(
        functions={**builder.contexts[target].functions, "_arrival_result": callback},
        targets="_arrival_result",
        enforce_signature=False,
        set_annotations=True,
    )
    return rename_arguments(
        combined,
        mapper={
            name: builder.aliases[target, name]
            for name in inspect.signature(combined).parameters
            if name not in read_names.values() and (target, name) in builder.aliases
        },
    )


def _validate_arrival_information(
    *,
    builder: _QueryGraphBuilder,
    target: RegimeName,
    function: UserFunction,
    edge_parameters: frozenset[str],
    read_names: frozenset[ReferenceName],
    information: frozenset[str],
) -> None:
    """Check primitive landed reads before an edge parameter can alias their names."""
    context = builder.contexts[target]
    plan = context.current_iid_plans.get(builder.period)
    physical = {
        name: law
        for _, binding, _ in (() if plan is None else plan.bindings)
        for name, law in binding.group.shocks.items()
    }
    continuous_codes = {
        f"{name}_code"
        for name, law in physical.items()
        if not isinstance(law, Discrete)
    }
    discrete_codes = {
        f"{name}_code": name
        for name, law in physical.items()
        if isinstance(law, Discrete)
    }
    pending = list(inspect.signature(function).parameters)
    visited = set()
    while pending:
        name = pending.pop()
        if name in read_names or name in visited:
            continue
        visited.add(name)
        if name in continuous_codes:
            raise UnsupportedOperationError(
                f"Arrival input {name!r} cannot read a continuous physical shock's "
                "code: a physical draw has no quadrature-node position."
            )
        physical_name = discrete_codes.get(name, name)
        if physical_name in physical:
            if physical_name not in information:
                raise RegimeInitializationError(
                    f"Arrival callback for {target!r} requires unobserved "
                    f"physical input {physical_name!r}."
                )
        elif name not in edge_parameters and name in context.functions:
            pending.extend(inspect.signature(context.functions[name]).parameters)


def bind_arrival_query_graph(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    source: RegimeName,
    target: RegimeName,
    period: int,
    phase: Phase,
    declaration: ArrivalDecision,
) -> ArrivalQueryPlan:
    """Bind a nonpublishing consumer at the actual target date and physical context."""
    aliases = {
        (owner, name): f"_lcm_arrival_operand_{index}"
        for index, (owner, name) in enumerate(
            (owner, name)
            for owner, context in contexts.items()
            for name in sorted(context.flat_param_names)
        )
    }
    builder = _QueryGraphBuilder(
        contexts=contexts,
        canonical_regimes=canonical_regimes,
        period=period,
        phase=phase,
        aliases=aliases,
        operand_paths={alias: path for path, alias in aliases.items()},
    )
    return _bind_arrival(
        builder=builder, source=source, target=target, declaration=declaration
    )


def _bind_arrival(
    *,
    builder: _QueryGraphBuilder,
    source: RegimeName,
    target: RegimeName,
    declaration: ArrivalDecision,
) -> ArrivalQueryPlan:
    """Keep comparison reads and actual acting policy separate in the same frame."""
    context = builder.contexts[target]
    source_members = builder.contexts[source].stakeholders
    members = source_members or tuple(declaration.routes)
    if source_members is None and len(members) != 1:
        raise RegimeInitializationError(
            "A singleton arrival needs its single source route."
        )
    if set(declaration.routes) != set(members):
        raise RegimeInitializationError("Arrival routes must name every source role.")
    root = "arrival"
    builder.keys.append(root)
    projection = {
        name: with_signature(NamedOperand(name=name), args=[name], enforce=False)
        for name in context.state_action_space.state_names
    }
    target_plan = context.current_iid_plans.get(builder.period)
    physical_names = frozenset() if target_plan is None else target_plan.physical_names
    information = frozenset(declaration.observes)
    inputs = {
        "_lcm_arrival_acting_target": builder.make(
            kind=QueryKind.PARTICIPATING,
            owner=target,
            projection=projection,
            guards=(),
            ancestry=(),
            ancestors=(root,),
            price_information=information,
        )
    }
    occupied_names = {
        *context.functions,
        *context.grids,
        *context.flat_param_names,
        *builder.aliases.values(),
        *declaration.reads,
        *inspect.signature(declaration.predicate).parameters,
        *(
            name
            for function in context.functions.values()
            for name in inspect.signature(function).parameters
        ),
    }
    read_names = {}
    for index, name in enumerate(declaration.reads):
        alias = f"_lcm_arrival_read_{index}"
        while alias in occupied_names:
            alias += "_"
        read_names[name] = alias
        occupied_names.add(alias)
    reads = {}
    for name, read in declaration.reads.items():
        mapped = {
            state: _bound_edge_function(
                builder=builder,
                source=source,
                target=target,
                path=(target, "references", name, state),
                function=function,
                information=information,
            )
            for state, function in read.projection.items()
        }
        projected = {
            state: _compose(
                projection=projection,
                function=function,
                source=root,
                physical_names=physical_names,
            )
            for state, function in mapped.items()
        }
        operand = f"_lcm_arrival_comparison_{len(reads)}"
        inputs[operand] = (
            _make_integrated_read(
                builder=builder, read=read, projection=projected, root=root
            )
            if read.kind == "IntegratedFull"
            else builder.make(
                kind=QueryKind(read.kind),
                owner=read.regime,
                projection=projected,
                guards=(),
                ancestry=(),
                role=read.stakeholder or "",
                ancestors=(root,),
                price_information=information | frozenset(read.observes),
                source_projection=cast("Mapping[str, UserFunction]", mapped),
            )
        )
        reads[read_names[name]] = (
            operand,
            read.field,
            None
            if read.kind == "IntegratedFull"
            else _member_index(
                context=builder.contexts[read.regime], role=read.stakeholder
            ),
        )
    target_indices = []
    fallbacks = []
    for member in members:
        route = declaration.routes[member]
        fallback = (
            route.solve_fallback
            if builder.phase == "solve"
            else route.simulate_fallback
        )
        mapped = {
            state: _bound_edge_function(
                builder=builder,
                source=source,
                target=target,
                path=(
                    target,
                    "routes",
                    member,
                    "fallback",
                    *((builder.phase,) if route.fallback_is_phased else ()),
                    state,
                ),
                function=function,
                information=information,
            )
            for state, function in fallback.projection.items()
        }
        projected = {
            state: _compose(
                projection=projection,
                function=function,
                source=root,
                physical_names=physical_names,
            )
            for state, function in mapped.items()
        }
        operand = f"_lcm_arrival_fallback_{len(fallbacks)}"
        inputs[operand] = builder.make(
            kind=QueryKind.PARTICIPATING,
            owner=fallback.regime,
            projection=projected,
            guards=(),
            ancestry=(),
            role=fallback.stakeholder or "",
            ancestors=(root,),
            price_information=information,
            source_projection=cast("Mapping[str, UserFunction]", mapped),
        )
        target_indices.append(
            _member_index(context=context, role=route.target_stakeholder)
        )
        fallbacks.append(
            (
                operand,
                _member_index(
                    context=builder.contexts[fallback.regime], role=fallback.stakeholder
                ),
            )
        )
    predicate = _bound_edge_function(
        builder=builder,
        source=source,
        target=target,
        path=(target, "predicate"),
        function=declaration.predicate,
        information=information,
        read_names=read_names,
    )
    reducer = _ArrivalReducer(
        predicate=predicate,
        reads=reads,
        target_indices=tuple(target_indices),
        fallbacks=tuple(fallbacks),
        collective=builder.contexts[source].stakeholders is not None,
    )
    names = tuple(
        dict.fromkeys(
            (
                *inputs,
                *(
                    name
                    for name in inspect.signature(predicate).parameters
                    if name not in reads
                ),
            )
        )
    )
    needed = set(names) | {
        name
        for function in projection.values()
        for name in inspect.signature(function).parameters
    }
    builder.nodes[root] = QueryNode(
        read=QueryRead(
            kind=QueryKind.ARRIVAL,
            target=target,
            phase=builder.phase,
            period=builder.period,
            projection=cast("Mapping[str, UserFunction]", projection),
            state_inputs=frozenset(context.state_action_space.state_names),
            physical_inputs=information,
            operand_paths={
                name: path
                for name, path in builder.operand_paths.items()
                if name in needed
            },
            policy_information=information,
            price_information=frozenset(),
        ),
        context=replace(
            context, current_iid_plans=MappingProxyType({builder.period: None})
        ),
        inputs=inputs,
        exit_targets={
            0: dict.fromkeys(members, inputs["_lcm_arrival_acting_target"]),
            1: {
                member: inputs[fallbacks[index][0]]
                for index, member in enumerate(members)
            },
        }
        if builder.phase == "simulate"
        else {},
        decision=with_signature(reducer, args=list(names), enforce=False),
        operand_builder=_ArrivalArguments(period=builder.period, plan=None),
        answer_shape=()
        if builder.contexts[source].stakeholders is None
        else (len(members),),
    )
    _bind_observation_sources(nodes=builder.nodes, period=builder.period)
    _validate_query_projection_bindings(nodes=builder.nodes, period=builder.period)
    if builder.phase == "solve":
        for node in builder.nodes.values():
            _admit_node(node=node, period=builder.period)
    return ArrivalQueryPlan(
        source=source,
        target=target,
        period=builder.period,
        phase=builder.phase,
        members=members,
        queries=ScalarPolicyQueries(
            definitions=MappingProxyType(builder.nodes),
            root=root,
            period=builder.period,
        ),
    )
