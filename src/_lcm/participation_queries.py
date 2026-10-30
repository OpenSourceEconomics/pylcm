"""Bind edge-owned participation comparisons to actual dated policy/value readers."""

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
from _lcm.grids.continuous import ContinuousGrid
from _lcm.params.edges import regime_kernel_params
from _lcm.participation_continuation import (
    PARTICIPATION_CONTINUATION,
    PARTICIPATION_SURFACE_PREFIX,
    ParticipationContinuation,
    ParticipationOperand,
    participation_surface_name,
)
from _lcm.policy_queries import (
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    ScalarPolicyQueries,
    _admit_node,
)
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.Q_and_F import _CollectiveQAndF, _ComputeCE, _QAndF
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.value_adapter import (
    _AdaptedParticipationPredicate,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import (
    FlatParams,
    ParticipationQueryArguments,
    PytreeValue,
    QAndFKwargs,
    ShapeDtypePytree,
    ValueTemplate,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import NamedOperand, call_with_declared_arguments
from lcm.exceptions import (
    ExecutionPlanningError,
    ModelInitializationError,
    RegimeInitializationError,
    UnsupportedOperationError,
)
from lcm.participation import ParticipationCheck, ParticipationRead
from lcm.shocks import Discrete
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    Phase,
    ReferenceName,
    RegimeName,
    StateName,
    UserFunction,
)


@dataclass(frozen=True, kw_only=True)
class ParticipationQueryPlan:
    """One incoming edge's exact-date read graph and original source roles."""

    source: RegimeName
    target: RegimeName
    period: int
    phase: Phase
    members: tuple[str, ...]
    queries: ScalarPolicyQueries
    transport_template: ParticipationContinuation | None = None
    transport_origins: Mapping[str, ValueArtifactAddress] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, kw_only=True)
class _ParticipationBuildContext(CoreBuildContext):
    """Call-local demanded leaves for a future Q that holds participation checks."""

    participation_values: Mapping[ValueArtifactAddress, ValueTemplate] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, kw_only=True)
class ParticipationPublication:
    """The dated participation payload one regime publishes, and how each leaf arises.

    `origins` are borrowed stored values a later participation reader needs; `surfaces`
    are computed by this regime's own participation plans: each names the target whose
    stored participation surface the leaf holds.
    """

    template: ParticipationContinuation
    origins: Mapping[str, ValueArtifactAddress] = field(
        default_factory=lambda: MappingProxyType({})
    )
    surfaces: Mapping[str, RegimeName] = field(
        default_factory=lambda: MappingProxyType({})
    )


def surface_address(
    *, source: RegimeName, target: RegimeName, period: int
) -> ValueArtifactAddress:
    """Address the stored participation surface `source` publishes at `period`.

    The surface holds what the participation check from `source` into `target`
    reads -- each branch's value and each operand's support -- tabulated on
    `target`'s grid one period later (see `surface_channels`). It is a leaf of
    `source`'s own dated participation payload, so it is transported, admitted and
    released like that payload.
    """
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=period,
        regime=source,
        artifact_key=PARTICIPATION_CONTINUATION,
        leaf_path=("operands", participation_surface_name(target=target)),
    )


def _is_surface(*, address: ValueArtifactAddress, period: int) -> bool:
    """Whether a plan dated `period` reads `address` as a stored surface."""
    return (
        address.kind is ValueArtifactKind.CONTINUATION_LEAF
        and address.period == period
        and address.artifact_key == PARTICIPATION_CONTINUATION
        and isinstance(name := address.leaf_path[-1], str)
        and name.startswith(PARTICIPATION_SURFACE_PREFIX)
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


def _transport_alias(
    *, plan: ParticipationQueryPlan, name: str
) -> ValueArtifactAddress:
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=plan.period,
        regime=plan.target,
        artifact_key=PARTICIPATION_CONTINUATION,
        leaf_path=("operands", name),
    )


def _participation_values(
    *, context: CoreBuildContext, plan: ParticipationQueryPlan
) -> Mapping[ValueArtifactAddress, ValueTemplate]:
    reads = plan.queries.value_reads(core_key="participation")
    surfaces = _surface_values(
        context=context,
        addresses={
            read.target
            for read in reads
            if _is_surface(address=read.target, period=plan.period)
        },
    )
    required = {read.target for read in reads if read.target.period > plan.period}
    if not required:
        return MappingProxyType(surfaces)
    template = plan.transport_template
    if template is None:
        raise UnsupportedOperationError(
            "This participation policy requires its declared later-value transport."
        )
    if isinstance(context, _ParticipationBuildContext):
        return MappingProxyType(
            {
                origin: context.participation_values[
                    _transport_alias(plan=plan, name=_transport_name(origin=origin))
                ]
                for origin in required
            }
            | surfaces
        )
    payload = context.next_regime_to_continuation.get(plan.target)
    if not isinstance(payload, ParticipationContinuation) or (
        payload.regime,
        payload.period,
        payload.schema,
        payload.names,
    ) != (template.regime, template.period, template.schema, template.names):
        raise ExecutionPlanningError(
            "A participation check requires its exact declared producer-date operand "
            "payload."
        )
    return MappingProxyType(
        {
            origin: payload.operands[_transport_name(origin=origin)]
            for origin in required
        }
        | surfaces
    )


def _surface_values(
    *, context: CoreBuildContext, addresses: set[ValueArtifactAddress]
) -> dict[ValueArtifactAddress, ValueTemplate]:
    """Read each stored participation surface from its publisher's dated payload."""
    if isinstance(context, _ParticipationBuildContext):
        return {address: context.participation_values[address] for address in addresses}
    values = {}
    for address in addresses:
        payload = context.next_regime_to_continuation.get(address.regime)
        if not isinstance(payload, ParticipationContinuation) or (
            payload.regime,
            payload.period,
        ) != (address.regime, address.period):
            raise ExecutionPlanningError(
                "A participation check requires its exact producer-date stored surface."
            )
        values[address] = payload.leaves()[address.leaf_path]
    return values


@dataclass(frozen=True, kw_only=True)
class ParticipationSourceArguments:
    """Actual participation-date query operands beside the source's own inputs."""

    base: CoreArgumentBuilder
    plans: Mapping[RegimeName, ParticipationQueryPlan]

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        arguments = dict(self.base(context))
        participation = {}
        for target, plan in self.plans.items():
            transported = _participation_values(context=context, plan=plan)
            participation_context = _ParticipationBuildContext(
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
                participation_values=transported,
            )
            participation[target] = plan.queries.arguments(
                context=participation_context
            )
        arguments["participation_arguments"] = ParticipationQueryArguments(
            by_target=cast(
                "Mapping[RegimeName, Mapping[str, QAndFKwargs]]", participation
            )
        )
        return MappingProxyType(arguments)


def participation_source_value_reads(
    *,
    plans: Mapping[RegimeName, ParticipationQueryPlan],
    source: RegimeName,
    period: int,
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
                or _is_surface(address=read.target, period=plan.period)
                else ValueInputChannel.NEXT_REGIME_VALUE,
                argument="participation_arguments",
                path=("by_target", target, *read.source.path),
            ),
        )
        for target, plan in plans.items()
        for read in plan.queries.value_reads(core_key="participation")
    )


@dataclass(frozen=True, kw_only=True)
class _ParticipationPolicyValue:
    """Price the bound Boolean routing at this action's actual target landing."""

    plan: ParticipationQueryPlan
    process_nodes: Mapping[StateName, Float1D] = field(
        default_factory=lambda: MappingProxyType({}),
        metadata={"fingerprint_omit_if_default": True},
    )
    """Nodes of each process state the continuation enumerates by position."""

    def __call__(
        self, **operands: PytreeValue
    ) -> tuple[FloatND, BoolND] | tuple[tuple[FloatND, BoolND], IntND]:
        arguments = cast(
            "ParticipationQueryArguments", operands["participation_arguments"]
        )
        states = {
            name.removeprefix("next_"): value
            for name, value in operands.items()
            if name.startswith("next_") and name != "next_V_arr"
        }
        # The continuation enumerates a process landing by its node position,
        # which indexes stored values; the participation reads the node's value.
        for name, nodes in self.process_nodes.items():
            states[name] = nodes[jnp.asarray(states[name]).astype(jnp.int32)]
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
    plans: Mapping[RegimeName, ParticipationQueryPlan],
) -> _QAndF | _CollectiveQAndF:
    """Replace only the actual continuation reader; keep canonical flow and laws."""
    function = context.Q_and_F_functions[period]
    if not isinstance(function, _QAndF | _CollectiveQAndF) or not isinstance(
        function.compute_CE, _ComputeCE
    ):
        raise UnsupportedOperationError(
            "Participation pricing needs its canonical GridSearch Q/F continuation; "
            "endogenous-grid participation reconstruction is an S10 dependency."
        )
    compute = function.compute_CE
    continuations = dict(compute.continuations)
    scalar_readers = dict(compute.participation_scalar_readers)
    scalar_names = dict(compute.participation_scalar_arg_names)
    scalar_checks = dict(compute.participation_scalar_landing_checks)
    for target, plan in plans.items():
        target_names = plan.queries.definitions[plan.queries.root].read.projection
        names = [
            *(f"next_{name}" for name in target_names),
            "next_V_arr",
            "participation_arguments",
        ]
        if target in compute.scalar_targets:
            scalar_readers[target] = with_signature(
                _ParticipationPolicyValue(plan=plan), args=names, enforce=False
            )
            scalar_names[target] = ("participation_arguments",)
            scalar_checks[target] = plan.queries.landing_checks
            continue
        original = continuations[target]
        reader = with_signature(
            _ParticipationPolicyValue(
                plan=plan,
                process_nodes=_landing_process_nodes(
                    target=target,
                    info=context.regime_to_v_interpolation_info[target],
                    lottery_axis_names=original.lottery_axis_names,
                ),
            ),
            args=names,
            enforce=False,
        )
        if (
            original.integrated_weight_names
            or original.draw_dependent_names
            or original.landing_checks
        ):
            raise UnsupportedOperationError(
                "This participation continuation needs its declared draw/landing "
                "reader adapter."
            )
        continuations[target] = replace(
            original,
            next_V=productmap(
                func=reader,
                variables=original.lottery_axis_names,
                batch_sizes=dict.fromkeys(original.lottery_axis_names, 0),
            ),
            extra_param_names=frozenset({"participation_arguments"}),
            original_node_axes=original.original_node_axes
            or original.original_value_axes,
            original_value_axes=(),
            participation_query=True,
            landing_checks=plan.queries.landing_checks,
        )
    compute = replace(
        compute,
        continuations=MappingProxyType(continuations),
        participation_scalar_readers=MappingProxyType(scalar_readers),
        participation_scalar_arg_names=MappingProxyType(scalar_names),
        participation_scalar_landing_checks=tuple(scalar_checks.items()),
    )
    return replace(
        function,
        compute_CE=compute,
        landing_checks=compute.landing_checks,
        arg_names=(*function.arg_names, "participation_arguments"),
        continuation_reads=None
        if function.continuation_reads is None
        else function.continuation_reads | {"participation_arguments"},
    )


def _landing_process_nodes(
    *,
    target: RegimeName,
    info: VInterpolationInfo,
    lottery_axis_names: tuple[str, ...],
) -> MappingProxyType[StateName, Float1D]:
    """Return the nodes of each process state the continuation lands on by position.

    Raises:
        ModelInitializationError: If such a process is parameterized at runtime,
            so its nodes are not known while the model builds.

    """
    nodes = {}
    for name, grid in info.discrete_states.items():
        if f"next_{name}" not in lottery_axis_names or not isinstance(
            grid, _ContinuousStochasticProcess
        ):
            continue
        if not grid.is_fully_specified:
            raise ModelInitializationError(
                f"A participation check into regime '{target}' reads the process "
                f"'{name}' at "
                "its landing, but that process is parameterized at runtime, so its "
                "nodes are not known while the model builds. Fix its parameters "
                "through the process constructor or `fixed_params`."
            )
        nodes[name] = grid.to_jax()
    return MappingProxyType(nodes)


# The operand carrying each surfaced participation check's own date and parameters.
PARTICIPATION_SURFACE_ARGUMENTS = "participation_surface_arguments"

_ACTING_TARGET = "_lcm_participation_acting_target"

_SUPPORT = "support"

# One component of a policy answer the participation check consumes: the answer's
# operand name and either `_SUPPORT` or the index of the value it reads (`None`
# for a scalar value).
type SurfaceChannel = tuple[str, str | int | None]


def _participation_reducer(*, plan: ParticipationQueryPlan) -> _ParticipationReducer:
    decision = plan.queries.definitions[plan.queries.root].decision
    reducer = None if decision is None else inspect.unwrap(decision)
    if not isinstance(reducer, _ParticipationReducer):
        raise ExecutionPlanningError("A participation plan lost its decision rule.")
    return reducer


def surface_channels(*, plan: ParticipationQueryPlan) -> tuple[SurfaceChannel, ...]:
    """The branch operands a stored participation surface holds, in trailing-axis order.

    Every policy answer the participation check consumes -- the acting target, each
    declared read and each member's fallback -- is stored through the components
    the decision reads: its support indicator, NaN where the answer fails its
    integrity check, and each value it reads, zero where the answer has no
    support.
    """
    reducer = _participation_reducer(plan=plan)
    channels: list[SurfaceChannel] = [
        (_ACTING_TARGET, _SUPPORT),
        *((_ACTING_TARGET, index) for index in reducer.target_indices),
    ]
    for producer, read_field, index in reducer.reads.values():
        channels.append((producer, _SUPPORT))
        if read_field != _SUPPORT:
            channels.append((producer, index))
    for producer, index in reducer.fallbacks:
        channels.extend(((producer, _SUPPORT), (producer, index)))
    return tuple(dict.fromkeys(channels))


@dataclass(frozen=True, kw_only=True)
class _ParticipationChannels:
    """Report the decision's operands at one node instead of deciding there."""

    channels: tuple[SurfaceChannel, ...]

    def __call__(self, **operands: PytreeValue) -> PolicyAnswer:
        dtype = canonical_float_dtype()
        values = []
        for producer, component in self.channels:
            answer = cast("PolicyAnswer", operands[producer])
            if component == _SUPPORT:
                values.append(jnp.where(answer[3], answer[2].astype(dtype), jnp.nan))
            else:
                value = answer[0] if component is None else answer[0][component]
                values.append(jnp.where(answer[2], value, 0).astype(dtype))
        return (
            jnp.stack(values),
            jnp.asarray(-1, jnp.int32),
            jnp.asarray(a=True),
            jnp.asarray(a=True),
            jnp.zeros((0,), jnp.int32),
        )


def surface_queries(*, plan: ParticipationQueryPlan) -> ScalarPolicyQueries:
    """The plan's query graph, answering with its surface channels at a landing."""
    channels = surface_channels(plan=plan)
    root = plan.queries.definitions[plan.queries.root]
    return replace(
        plan.queries,
        definitions=MappingProxyType(
            {
                **plan.queries.definitions,
                plan.queries.root: replace(
                    root,
                    decision=with_signature(
                        _ParticipationChannels(channels=channels),
                        args=list(dict.fromkeys(name for name, _ in channels)),
                        enforce=False,
                    ),
                    answer_shape=(len(channels),),
                ),
            }
        ),
    )


def plan_surface_address(*, plan: ParticipationQueryPlan) -> ValueArtifactAddress:
    """Address the stored surface tabulated from `plan`'s answers."""
    return surface_address(
        source=plan.source, target=plan.target, period=plan.period - 1
    )


def surface_reader_arguments(
    *, plans: Mapping[RegimeName, ParticipationQueryPlan], context: CoreBuildContext
) -> MappingProxyType[RegimeName, Mapping[str, PytreeValue]]:
    """Each surfaced decision's own date, age and parameters, as its root reads them."""
    from _lcm.policy_queries import _projection_operands  # noqa: PLC0415

    if context.ages is None:
        raise ValueError(
            "A participation surface reader needs its actual economic age."
        )
    ages = context.ages
    return MappingProxyType(
        {
            target: MappingProxyType(
                {
                    "period": jnp.int32(plan.period),
                    "age": ages.values[plan.period],
                    **_projection_operands(
                        node=plan.queries.definitions[plan.queries.root],
                        source_state_names=frozenset(),
                        flat_params=cast("FlatParams", context.flat_params),
                    ),
                }
            )
            for target, plan in plans.items()
            if decides_between_nodes(plan=plan)
        }
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ParticipationSurfaceValue:
    """Read a stored participation surface at a landing: interpolate, then decide there.

    Each channel is interpolated like a value function, by the target's own
    continuation reader. A support indicator reads as support where it is at
    least one half: exact on nodes, and inside a cell whose corners disagree the
    switch falls on the indicator's half level set. A value is the
    support-weighted mean of the supported corners, so a corner without support
    neither enters it nor makes it infinite. The participation check is then re-run
    on the interpolated operands at the landing, and each member takes the
    branch it selects. On a node every channel is that node's own, so the read
    equals the decision the plan takes there.
    """

    plan: ParticipationQueryPlan
    channels: tuple[SurfaceChannel, ...]
    interpolate: Callable[..., FloatND] | None
    """The target's continuation reader of one stored array; `None` for a target
    without states, whose surface is its one landing."""
    lottery_names: tuple[str, ...] = ()
    """Landing coordinates the reader maps over, one node per call here."""
    width: int | None = None
    """Trailing value axis the reader carries per call; `None` for none."""
    process_nodes: Mapping[StateName, Float1D] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Nodes of each process state the continuation enumerates by position."""

    def __call__(self, **operands: PytreeValue) -> tuple[FloatND, BoolND]:
        from _lcm.policy_queries import (  # noqa: PLC0415
            _mapped_query_states,
            _mapped_query_valid,
        )

        stored = cast("FloatND", operands["next_V_arr"])
        if stored.shape[-1:] != (len(self.channels),):
            raise ExecutionPlanningError(
                "A stored participation surface does not hold this participation "
                "check's branches."
            )
        read = (
            stored
            if self.interpolate is None
            else self._interpolated(stored=stored, operands=operands)
        )
        states = {
            name.removeprefix("next_"): value
            for name, value in operands.items()
            if name.startswith("next_") and name != "next_V_arr"
        }
        for name, nodes in self.process_nodes.items():
            states[name] = nodes[jnp.asarray(states[name]).astype(jnp.int32)]
        root = self.plan.queries.definitions[self.plan.queries.root]
        frame = cast(
            "Mapping[RegimeName, Mapping[str, PytreeValue]]",
            operands[PARTICIPATION_SURFACE_ARGUMENTS],
        )[self.plan.target]
        mapped = _mapped_query_states(node=root, available={**frame, **states})
        decision = cast("Callable[..., PolicyAnswer]", root.decision)
        available = {**frame, **mapped, **self._answers(read=read)}
        answer = decision(
            **{name: available[name] for name in inspect.signature(decision).parameters}
        )
        return answer[0], answer[3] & _mapped_query_valid(node=root, mapped=mapped)

    def _interpolated(
        self, *, stored: FloatND, operands: Mapping[str, PytreeValue]
    ) -> FloatND:
        """Every channel at this one landing, read like the target's value."""
        interpolate = cast("Callable[..., FloatND]", self.interpolate)
        coordinates = {
            name: operands[name]
            for name in inspect.signature(interpolate).parameters
            if name != "next_V_arr"
        }
        for name in self.lottery_names:
            coordinates[name] = jnp.reshape(
                cast("FloatND | IntND", coordinates[name]), (1,)
            )
        width = 1 if self.width is None else self.width
        if padding := -len(self.channels) % width:
            stored = jnp.concatenate(
                (stored, jnp.zeros((*stored.shape[:-1], padding), stored.dtype)),
                axis=-1,
            )
        blocks = []
        for start in range(0, stored.shape[-1], width):
            block = stored[..., start : start + width]
            values = interpolate(
                **coordinates,
                next_V_arr=block if self.width is not None else block[..., 0],
            )
            blocks.append(jnp.reshape(values, (width,)))
        return jnp.concatenate(blocks)[: len(self.channels)]

    def _answers(self, *, read: FloatND) -> dict[str, PolicyAnswer]:
        """Rebuild each consumed policy answer from its interpolated channels."""
        position = {channel: index for index, channel in enumerate(self.channels)}
        answers = {}
        for producer in dict.fromkeys(name for name, _ in self.channels):
            indicator = read[position[producer, _SUPPORT]]
            support = indicator >= 0.5  # noqa: PLR2004 - the indicator's half level
            weight = jnp.where(support, indicator, 1)
            components = {
                component: jnp.where(
                    support, read[position[producer, component]] / weight, -jnp.inf
                )
                for name, component in self.channels
                if name == producer and component != _SUPPORT
            }
            empty = jnp.asarray(-jnp.inf, read.dtype)
            if set(components) <= {None}:
                value = components.get(None, empty)
            else:
                indices = cast("dict[int, FloatND]", components)
                value = jnp.stack(
                    tuple(
                        indices.get(index, empty) for index in range(max(indices) + 1)
                    )
                )
            answers[producer] = (
                value,
                jnp.asarray(-1, jnp.int32),
                support,
                ~jnp.isnan(indicator),
                jnp.zeros((0,), jnp.int32),
            )
        return answers


def _surface_action_problem(
    *,
    context: SolverBuildContext,
    period: int,
    plans: Mapping[RegimeName, ParticipationQueryPlan],
) -> _QAndF | _CollectiveQAndF:
    """Read each surfaced participation target by its branches, decided on landing."""
    function = context.Q_and_F_functions[period]
    if not isinstance(function, _QAndF | _CollectiveQAndF) or not isinstance(
        function.compute_CE, _ComputeCE
    ):
        raise UnsupportedOperationError(
            "Participation pricing needs its canonical GridSearch Q/F continuation; "
            "endogenous-grid participation reconstruction is an S10 dependency."
        )
    compute = function.compute_CE
    continuations = dict(compute.continuations)
    scalar_readers = dict(compute.participation_scalar_readers)
    scalar_names = dict(compute.participation_scalar_arg_names)
    scalar_checks = dict(compute.participation_scalar_landing_checks)
    for target, plan in plans.items():
        if not decides_between_nodes(plan=plan):
            continue
        channels = surface_channels(plan=plan)
        root = plan.queries.definitions[plan.queries.root]
        if target in compute.scalar_targets:
            scalar_readers[target] = with_signature(
                _ParticipationSurfaceValue(
                    plan=plan, channels=channels, interpolate=None
                ),
                args=["next_V_arr", PARTICIPATION_SURFACE_ARGUMENTS],
                enforce=False,
            )
            scalar_names[target] = (PARTICIPATION_SURFACE_ARGUMENTS,)
            scalar_checks[target] = ()
            continue
        original = continuations[target]
        if (
            original.next_V is None
            or original.integrated_weight_names
            or original.draw_dependent_names
            or original.landing_checks
            or original.participation_query
        ):
            raise UnsupportedOperationError(
                "This participation continuation needs its declared draw/landing "
                "reader adapter."
            )
        names = list(
            dict.fromkeys(
                (
                    *inspect.signature(original.next_V).parameters,
                    *(f"next_{name}" for name in root.read.projection),
                    "next_V_arr",
                    PARTICIPATION_SURFACE_ARGUMENTS,
                )
            )
        )
        reader = with_signature(
            _ParticipationSurfaceValue(
                plan=plan,
                channels=channels,
                interpolate=cast("Callable[..., FloatND]", original.next_V),
                lottery_names=original.lottery_axis_names,
                width=root.value_shape[0] if root.value_shape else None,
                process_nodes=_landing_process_nodes(
                    target=target,
                    info=context.regime_to_v_interpolation_info[target],
                    lottery_axis_names=original.lottery_axis_names,
                ),
            ),
            args=names,
            enforce=False,
        )
        continuations[target] = replace(
            original,
            next_V=productmap(
                func=reader,
                variables=original.lottery_axis_names,
                batch_sizes=dict.fromkeys(original.lottery_axis_names, 0),
            ),
            extra_param_names=original.extra_param_names
            | {PARTICIPATION_SURFACE_ARGUMENTS},
            original_node_axes=original.original_node_axes
            or original.original_value_axes,
            original_value_axes=(),
            participation_query=True,
            landing_checks=(),
        )
    compute = replace(
        compute,
        continuations=MappingProxyType(continuations),
        participation_scalar_readers=MappingProxyType(scalar_readers),
        participation_scalar_arg_names=MappingProxyType(scalar_names),
        participation_scalar_landing_checks=tuple(scalar_checks.items()),
    )
    return replace(
        function,
        compute_CE=compute,
        landing_checks=compute.landing_checks,
        arg_names=(*function.arg_names, PARTICIPATION_SURFACE_ARGUMENTS),
        continuation_reads=None
        if function.continuation_reads is None
        else function.continuation_reads | {PARTICIPATION_SURFACE_ARGUMENTS},
    )


def decides_between_nodes(*, plan: ParticipationQueryPlan) -> bool:
    """Whether `plan`'s surface stores branches, decided again at each landing.

    Between nodes the decision is re-run on interpolated answers, the landing
    state, its date, its age and its parameters. A decision that observes
    realized physical information has no stored counterpart for it there: its
    surface stores the value after the decision, integrated over that
    information at each node, and is read like a value function.
    """
    root = plan.queries.definitions[plan.queries.root]
    decision = cast("Callable[..., PolicyAnswer]", root.decision)
    supplied = {
        "period",
        "age",
        *root.read.operand_paths,
        *root.read.projection,
        *root.inputs,
    }
    return (
        not root.read.physical_inputs
        and set(inspect.signature(decision).parameters) <= supplied
    )


def _bind_transport_templates(
    *,
    plans_by_source: Mapping[RegimeName, Mapping[RegimeName, ParticipationQueryPlan]],
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    origin_templates: dict[ValueArtifactAddress, ParticipationOperand],
    dated_templates: dict[tuple[RegimeName, int], ParticipationContinuation],
) -> tuple[
    dict[RegimeName, dict[RegimeName, ParticipationQueryPlan]],
    dict[RegimeName, ParticipationPublication],
]:
    """Union the actual incoming readers of each one producer-date payload.

    A plan borrows the later stored values its rebuilt action problems read
    through its target's payload, and reads each stored participation surface from the
    payload of the regime whose action problem it rebuilds.
    """
    origins, surfaces = _borrowed_reads(
        plans_by_source=plans_by_source, contexts=contexts
    )
    publications = {}
    for owner in sorted(origins.keys() | surfaces.keys()):
        declarations = dict(sorted(origins.get(owner, {}).items()))
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
            plan.period for plans in plans_by_source.values() for plan in plans.values()
        )
        surfaced = dict(sorted(surfaces.get(owner, {}).items()))
        for name, target in surfaced.items():
            operands[name] = _surface_template(
                context=contexts[owner],
                target_context=contexts[target],
                plan=contexts[owner].participation_queries[period][target],
            )
        schema = hashlib.sha256(repr(tuple(operands)).encode()).hexdigest()
        template = ParticipationContinuation(
            regime=owner, period=period, schema=schema, operands=operands
        )
        dated_templates[owner, period] = template
        origins[owner] = declarations
        publications[owner] = ParticipationPublication(
            template=template,
            origins=MappingProxyType(declarations),
            surfaces=MappingProxyType(surfaced),
        )
    return (
        {
            source: {
                target: replace(
                    plan,
                    transport_template=dated_templates.get((target, plan.period)),
                    transport_origins=MappingProxyType(origins.get(target, {})),
                )
                for target, plan in plans.items()
            }
            for source, plans in plans_by_source.items()
        },
        publications,
    )


def _borrowed_reads(
    *,
    plans_by_source: Mapping[RegimeName, Mapping[RegimeName, ParticipationQueryPlan]],
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> tuple[
    dict[RegimeName, dict[str, ValueArtifactAddress]],
    dict[RegimeName, dict[str, RegimeName]],
]:
    """Collect, per payload owner, the later values and the surfaces plans borrow."""
    origins: dict[RegimeName, dict[str, ValueArtifactAddress]] = {}
    surfaces: dict[RegimeName, dict[str, RegimeName]] = {}
    for plans in plans_by_source.values():
        for target, plan in plans.items():
            for read in plan.queries.value_reads(core_key="participation"):
                if _is_surface(address=read.target, period=plan.period):
                    owner = read.target.regime
                    surfaces.setdefault(owner, {})[read.target.leaf_path[-1]] = (
                        _surfaced_target(context=contexts[owner], address=read.target)
                    )
                    continue
                if read.target.period <= plan.period:
                    continue
                if read.view is not None:
                    raise UnsupportedOperationError(
                        "Participation check transport requires a whole stored operand "
                        "leaf."
                    )
                origins.setdefault(target, {})[_transport_name(origin=read.target)] = (
                    read.target
                )
    return origins, surfaces


def _surfaced_target(
    *, context: SolverBuildContext, address: ValueArtifactAddress
) -> RegimeName:
    """Recover the participation target whose stored surface `address` names."""
    for target in context.participation_queries.get(address.period, {}):
        if (
            surface_address(
                source=context.regime_name, target=target, period=address.period
            )
            == address
        ):
            return target
    raise ExecutionPlanningError(
        "A stored participation surface names no participation check of its publishing "
        "regime."
    )


def _surface_template(
    *,
    context: SolverBuildContext,
    target_context: SolverBuildContext,
    plan: ParticipationQueryPlan,
) -> ParticipationOperand:
    """Shape a stored surface on the target's grid, one channel vector per node.

    A surface read like a value function holds the plan's own answer per node.
    """
    shape = (
        *(
            nodes.size
            for nodes in landing_grid(
                context=target_context, period=plan.period
            ).values()
        ),
        *(
            (len(surface_channels(plan=plan)),)
            if decides_between_nodes(plan=plan)
            else plan.queries.definitions[plan.queries.root].value_shape
        ),
    )
    return context.place_on_regime_devices(
        template=jnp.zeros(shape, dtype=canonical_float_dtype())
    )


def landing_grid(
    *, context: SolverBuildContext, period: int
) -> MappingProxyType[str, FloatND | IntND]:
    """The nodes a target's value is stored on at `period`, in stored axis order."""
    nodes = {
        **context.state_action_space.states,
        **(context.period_to_state_nodes or {}).get(period, {}),
    }
    return MappingProxyType(
        {name: nodes[name] for name in context.state_action_space.states}
    )


def _origin_template(
    *,
    origin: ValueArtifactAddress,
    context: SolverBuildContext,
    regime: Regime,
    dated_templates: Mapping[tuple[RegimeName, int], ParticipationContinuation],
) -> ParticipationOperand:
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
            or spec.artifact_key != PARTICIPATION_CONTINUATION
            or spec.artifact_key != origin.artifact_key
        ):
            raise UnsupportedOperationError(
                "A participation check needs its declared future participation schema; "
                "other continuation intersections are an S10 dependency."
            )
        payload = spec.template_at(period=origin.period)
        if not isinstance(payload, ParticipationContinuation):
            raise TypeError("A participation schema needs its typed numeric payload.")
        return payload.leaves()[origin.leaf_path]
    raise UnsupportedOperationError(
        "Participation custody requires a stored value or continuation leaf."
    )


def install_participation_source_queries(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> MappingProxyType[RegimeName, SolverBuildContext]:
    """Bind each participation check to its target's action problem one level deep.

    A source's own action problem reads every participation check at its exact landing.
    The plan behind that read rebuilds the target's action problem there, from a pricing
    context: the canonical action problem, whose own next-date participation checks read
    the stored surface of that participation check. The surface holds the decision's
    branches and operands, interpolated separately and decided again at each landing. So
    no plan contains another plan, whatever the number of later participation dates.
    """
    updated = dict(contexts)
    pricing = dict(contexts)
    origin_templates: dict[ValueArtifactAddress, ParticipationOperand] = {}
    dated_templates: dict[tuple[RegimeName, int], ParticipationContinuation] = {}
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
                or not law.participation
                or period >= context.solution_reachability.n_periods - 1
            ):
                continue
            targets = context.solution_reachability.targets(
                period=period, source=source
            )
            plans = {
                target: bind_participation_query_graph(
                    contexts=pricing,
                    canonical_regimes=canonical_regimes,
                    source=source,
                    target=target,
                    period=period + 1,
                    phase="solve",
                    declaration=law.participation[target],
                )
                for target in targets
                if target in law.participation
            }
            if plans:
                plans_by_source[source] = plans
        if not plans_by_source:
            continue
        plans_by_source, publications = _bind_transport_templates(
            plans_by_source=plans_by_source,
            contexts=updated,
            canonical_regimes=canonical_regimes,
            origin_templates=origin_templates,
            dated_templates=dated_templates,
        )
        for owner, publication in publications.items():
            context = updated[owner]
            updated[owner] = replace(
                context,
                participation_publications=MappingProxyType(
                    {**context.participation_publications, period + 1: publication}
                ),
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
                participation_queries=MappingProxyType(
                    {
                        **context.participation_queries,
                        period: MappingProxyType(plans),
                    }
                ),
            )
            reader = pricing[source]
            pricing[source] = replace(
                reader,
                Q_and_F_functions=MappingProxyType(
                    {
                        **reader.Q_and_F_functions,
                        period: _surface_action_problem(
                            context=reader, period=period, plans=plans
                        ),
                    }
                ),
                participation_surfaces=MappingProxyType(
                    {**reader.participation_surfaces, period: MappingProxyType(plans)}
                ),
            )
    return MappingProxyType(updated)


def install_participation_replay_sources(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    pricing_contexts: Mapping[RegimeName, SolverBuildContext],
) -> MappingProxyType[RegimeName, SolverBuildContext]:
    """Keep actual current flow beside the source's perceived participation value."""
    updated = dict(contexts)
    for source, context in contexts.items():
        plans_by_period = {
            period: plans
            for period, plans in pricing_contexts[source].participation_queries.items()
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
            participation_queries=MappingProxyType(plans_by_period),
        )
    return MappingProxyType(updated)


@dataclass(frozen=True, kw_only=True)
class _ParticipationReducer:
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
            raise ValueError("A participation predicate must return a scalar Boolean.")
        target = cast("PolicyAnswer", operands[_ACTING_TARGET])
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
class _ParticipationArguments:
    """Only current-law numeric operands, without an unrelated target Q bank."""

    period: int
    plan: CurrentIIDPlan | None

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        if context.ages is None:
            raise ValueError("A participation reader needs its actual economic age.")
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
            raise ValueError(
                "An IntegratedFull participation read needs its dated stored V."
            )
        if context.ages is None:
            raise ValueError(
                "An IntegratedFull participation read needs its actual age."
            )
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
    read: ParticipationRead,
    projection: Mapping[str, Callable[..., PytreeValue]],
    root: str,
) -> str:
    """Declare a shock-free stored representation with explicit native authority."""
    context = builder.contexts[read.regime]
    if context.co_map_state_names or context.invariant_bindings:
        raise UnsupportedOperationError(
            "IntegratedFull participation check interpolation needs an explicit "
            "unblocked "
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


@dataclass(frozen=True, kw_only=True, eq=False)
class _ClippedProjection:
    """A projected landing coordinate clipped onto its `clip_state` grid."""

    function: Callable[..., PytreeValue]
    low: FloatND
    high: FloatND

    def __call__(self, **kwargs: PytreeValue) -> FloatND:
        return jnp.clip(
            jnp.asarray(self.function(**kwargs)), min=self.low, max=self.high
        )


def _clip_landing_states(
    *,
    projection: Mapping[StateName, Callable[..., PytreeValue]],
    context: SolverBuildContext,
    period: int,
) -> dict[StateName, Callable[..., PytreeValue]]:
    """Clip each landing coordinate a fallback writes onto a `clip_state` grid.

    A subject routed to the fallback lands there exactly as a realised state
    does, so it decides at the clipped state, never at the raw projection.
    """
    nodes = {
        **context.state_action_space.states,
        **(context.period_to_state_nodes or {}).get(period, {}),
    }
    clipped = dict(projection)
    for state, function in projection.items():
        grid = context.grids.get(state)
        if isinstance(grid, ContinuousGrid) and grid.off_grid == "clip_state":
            points = jnp.asarray(nodes[state])
            clipped[state] = with_signature(
                _ClippedProjection(function=function, low=points[0], high=points[-1]),
                args=list(inspect.signature(function).parameters),
                enforce=False,
            )
    return clipped


def _member_index(*, context: SolverBuildContext, role: str | None) -> int | None:
    """Validate an explicitly declared destination role without picking a member."""
    members = context.stakeholders
    if members is None:
        if role is not None:
            raise RegimeInitializationError(
                "A singleton participation target has no role."
            )
        return None
    if role not in members:
        raise RegimeInitializationError(
            "A collective participation read needs its exact role."
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
    if isinstance(function, _AdaptedParticipationPredicate):
        read_names = {
            function.read_aliases.get(name, name): alias
            for name, alias in read_names.items()
        }
        function = function.function
    branch = builder.canonical_regimes[source].edge_params_template
    for segment in path:
        child = branch.get(segment, {})
        branch = child if isinstance(child, Mapping) else {}
    _validate_participation_information(
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
        functions={
            **builder.contexts[target].functions,
            "_participation_result": callback,
        },
        targets="_participation_result",
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


def _validate_participation_information(
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
                f"Participation input {name!r} cannot read a continuous physical "
                "shock's "
                "code: a physical draw has no quadrature-node position."
            )
        physical_name = discrete_codes.get(name, name)
        if physical_name in physical:
            if physical_name not in information:
                raise RegimeInitializationError(
                    f"Participation callback for {target!r} requires unobserved "
                    f"physical input {physical_name!r}."
                )
        elif name not in edge_parameters and name in context.functions:
            pending.extend(inspect.signature(context.functions[name]).parameters)


def bind_participation_query_graph(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    source: RegimeName,
    target: RegimeName,
    period: int,
    phase: Phase,
    declaration: ParticipationCheck,
) -> ParticipationQueryPlan:
    """Bind a nonpublishing consumer at the actual target date and physical context."""
    aliases = {
        (owner, name): f"_lcm_participation_operand_{index}"
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
    return _bind_participation(
        builder=builder, source=source, target=target, declaration=declaration
    )


def _bind_participation(
    *,
    builder: _QueryGraphBuilder,
    source: RegimeName,
    target: RegimeName,
    declaration: ParticipationCheck,
) -> ParticipationQueryPlan:
    """Keep comparison reads and actual acting policy separate in the same frame."""
    context = builder.contexts[target]
    source_members = builder.contexts[source].stakeholders
    members = source_members or tuple(declaration.routes)
    if source_members is None and len(members) != 1:
        raise RegimeInitializationError(
            "A singleton participation check needs its single source route."
        )
    if set(declaration.routes) != set(members):
        raise RegimeInitializationError(
            "Participation routes must name every source role."
        )
    root = "participation"
    builder.keys.append(root)
    projection = {
        name: with_signature(NamedOperand(name=name), args=[name], enforce=False)
        for name in context.state_action_space.state_names
    }
    target_plan = context.current_iid_plans.get(builder.period)
    physical_names = frozenset() if target_plan is None else target_plan.physical_names
    information = frozenset(declaration.observes)
    inputs = {
        _ACTING_TARGET: builder.make(
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
        alias = f"_lcm_participation_read_{index}"
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
                path=(target, "reads", name, state),
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
        operand = f"_lcm_participation_comparison_{len(reads)}"
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
            # A fallback also projects the states its regime only carries in
            # simulation; the action problem it prices has no axis for them.
            if state in builder.contexts[fallback.regime].state_action_space.state_names
        }
        projected = _clip_landing_states(
            projection=projected,
            context=builder.contexts[fallback.regime],
            period=builder.period,
        )
        operand = f"_lcm_participation_fallback_{len(fallbacks)}"
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
    reducer = _ParticipationReducer(
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
            kind=QueryKind.PARTICIPATION,
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
            0: dict.fromkeys(members, inputs[_ACTING_TARGET]),
            1: {
                member: inputs[fallbacks[index][0]]
                for index, member in enumerate(members)
            },
        }
        if builder.phase == "simulate"
        else {},
        decision=with_signature(reducer, args=list(names), enforce=False),
        operand_builder=_ParticipationArguments(period=builder.period, plan=None),
        answer_shape=()
        if builder.contexts[source].stakeholders is None
        else (len(members),),
    )
    _bind_observation_sources(nodes=builder.nodes, period=builder.period)
    _validate_query_projection_bindings(nodes=builder.nodes, period=builder.period)
    if builder.phase == "solve":
        for node in builder.nodes.values():
            _admit_node(node=node, period=builder.period, same_period_values=True)
    return ParticipationQueryPlan(
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
