"""Private typed current-period policy reads and their native root closures."""

import functools
import inspect
import logging
import math
import weakref
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature
from jax.core import Tracer

from _lcm.current_iid import CurrentIIDDrawFrame
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalDispatch,
    InternalInputRef,
    InternalOutputSpec,
    ValueRead,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.grids import DiscreteGrid
from _lcm.params.edges import EDGES, flat_namespaces, regime_kernel_params
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import ParetoWeights, collective_argmax_and_readout
from _lcm.regime_building.max_Q_over_a import _evaluate_pareto_weights
from _lcm.regime_building.Q_and_F import (
    CanonicalActionKernel,
    EvaluatedQAndF,
    LandingCheck,
    has_objective_profile,
    has_projected_queries,
    runtime_bound_specs,
)
from _lcm.solution.contract import ContinuationPayload, PeriodKernel, SolverBuildContext
from _lcm.solution.grid_search import (
    GridSearch,
    _GridSearchArgumentBuilder,
    grid_search_query_inputs,
)
from _lcm.solution.landing_counts import (
    fail_on_query_landings,
    query_landing_check,
)
from _lcm.time import TimeAxis
from _lcm.typing import (
    FlatEdgeParams,
    FlatParams,
    FlatRegimeParams,
    PytreeValue,
    QAndFKwargs,
    QueryLandingCheck,
    ShapeDtypePytree,
)
from _lcm.utils.counts import sum_counts
from _lcm.utils.dispatchers import productmap, vary_carry_like_body
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.shocks import ShockGroupName
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, FloatND, IntND, RegimeName, StateName, UserFunction

_SOURCE = "_lcm_query_source"
_LIVE = "_lcm_query_live"
_PROJECTION = "_lcm_query_projection"
_PHYSICAL_CONTEXTS = "_lcm_query_physical_contexts"
_INFORMATION = "_lcm_query_information"
_ACTING = "_lcm_query_acting"
_TASTE = "_lcm_query_taste"
_FLAT_OPERAND_PATH_LENGTH = 2


class QueryKind(StrEnum):
    """Action problems and explicitly distinct nonpublishing arrival readers."""

    RAW = "Raw"
    PARTICIPATING = "Participating"
    FULL = "Full"
    ARRIVAL = "Arrival"
    INTEGRATED_FULL = "IntegratedFull"


@dataclass(frozen=True, kw_only=True)
class QueryRead:
    """One mapped policy context, with target and caller information separated."""

    kind: QueryKind
    """Action problem requested at this node."""
    target: RegimeName
    """Actual model regime, never a synthetic scheduler name."""
    phase: Literal["solve", "simulate"]
    """Physical-law and projection phase."""
    projection: Mapping[str, UserFunction]
    """Effective target-state helpers, including their canonical code adapters."""
    state_inputs: frozenset[str]
    """Actual state ancestors of the demanded action problem and laws."""
    physical_inputs: frozenset[str]
    """Demanded physical shocks; a regime's permission list is insufficient."""
    operand_paths: Mapping[str, tuple[str | int, ...]]
    """Current-call binding paths, distinct even when their values happen to agree."""
    policy_information: frozenset[str]
    """Information prescribed for the target's action choice."""
    price_information: frozenset[str]
    """Information available only to the caller pricing that prescribed policy."""
    period: int = 0
    """Computational period; an age is never inferred from this integer."""
    role: str = ""
    """Destination stakeholder role in the producing projection context."""
    law_identity: tuple[Hashable, ...] = ()
    """Canonical law, node and information-representation declaration identities."""
    signed_origins: frozenset[str] = frozenset()
    """Transitive signed channels, forbidden at a nonlinear policy read."""

    def __post_init__(self) -> None:
        """Snapshot caller containers before any closure is constructed."""
        object.__setattr__(self, "projection", MappingProxyType(dict(self.projection)))
        object.__setattr__(
            self, "operand_paths", MappingProxyType(dict(self.operand_paths))
        )
        for name in (
            "state_inputs",
            "physical_inputs",
            "policy_information",
            "price_information",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        object.__setattr__(self, "signed_origins", frozenset(self.signed_origins))
        object.__setattr__(self, "law_identity", tuple(self.law_identity))
        if not self.target or isinstance(self.period, bool) or self.period < 0:
            raise ValueError("A query needs an actual target and nonnegative period.")
        if self.signed_origins:
            raise ValueError("Signed ancestry cannot feed a policy query.")

    @property
    def identity(self) -> tuple[Hashable, ...]:
        """Return declaration/binding identity without reading device values."""
        return (
            self.kind,
            self.target,
            self.phase,
            self.period,
            self.role,
            tuple((name, id(helper)) for name, helper in self.projection.items()),
            tuple(self.operand_paths.items()),
            self.state_inputs,
            self.physical_inputs,
            self.policy_information,
            self.price_information,
            self.law_identity,
        )


def demanded_query_order(
    *,
    reads: Mapping[str, QueryRead],
    dependencies: Mapping[str, tuple[str, ...]],
    roots: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand demanded roots and reject typed cycles and Full publication reads."""
    graph: dict[str, tuple[str, ...]] = {}
    pending = list(reversed(roots))
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        if name not in reads or name not in dependencies:
            raise ValueError(f"Undeclared policy query {name!r}.")
        for upstream in dependencies[name]:
            if upstream not in reads:
                raise ValueError(f"Undeclared policy query {upstream!r}.")
            if reads[upstream].kind is QueryKind.FULL:
                raise ValueError(f"Query {name!r} cannot read Full query {upstream!r}.")
            if (
                reads[upstream].kind is QueryKind.INTEGRATED_FULL
                and reads[name].kind is not QueryKind.ARRIVAL
            ):
                raise ValueError("Only an arrival comparison may read IntegratedFull.")
        graph[name] = dependencies[name]
        pending.extend(reversed(dependencies[name]))
    try:
        return tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError(f"Policy query cycle: {error.args[1]!r}") from error


@dataclass(frozen=True, kw_only=True)
class QueryNode:
    """Canonical action problem and its explicitly named transient readers."""

    read: QueryRead
    """Mapped context of this action problem."""
    context: SolverBuildContext
    """Canonical target Q/F, grids, parameters and continuation declarations."""
    inputs: Mapping[str, str] = MappingProxyType({})
    """Predicate operand name to demanded producer key."""
    participation: Callable[..., BoolND] | None = None
    """Canonical kept-set extension; never replaces ordinary feasibility."""
    decision: Callable[..., PolicyAnswer] | None = None
    """Shared scalar Full/Participating rule after canonical stay selection."""
    available: Callable[..., BoolND] | None = None
    """Caller-context guard evaluated before a demanded benchmark."""
    exit_targets: Mapping[int, Mapping[str, str]] = MappingProxyType({})
    """Stable exit ID to source member and its exact destination query key."""
    exit_adjustments: Mapping[int, Mapping[str, Callable[..., PytreeValue]]] = (
        MappingProxyType({})
    )
    """The same bound once-paid source adjustments used by the decision."""
    ancestors: tuple[str, ...] = ()
    """Exact producing contexts along the mapped source path."""
    source_projection: Mapping[str, UserFunction] = MappingProxyType({})
    """Immediate caller-to-target binding for exact shared-law admission."""
    observation_sources: Mapping[ShockGroupName, str] = MappingProxyType({})
    """Target-only observed group to its exact producing mapped query context."""
    observation_aliases: Mapping[ShockGroupName, tuple[str, ...]] = MappingProxyType({})
    """Equivalent producing contexts whose guarded availability is combined."""
    observation_counts: tuple[int, ...] = ()
    """Declared original row count of each target-only observation group."""
    operand_builder: CoreArgumentBuilder | None = None
    """Explicit numeric operands for a nonpublishing arrival/value reader."""
    operand_value_reads: tuple[ValueRead, ...] = ()
    """Absolute native value authority of that reader's dynamic operands."""
    answer_shape: tuple[int, ...] | None = None
    """Source-owned payoff tuple when it differs from the physical context."""

    def __post_init__(self) -> None:
        """Freeze edges and require the actual canonical target authority."""
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "ancestors", tuple(self.ancestors))
        object.__setattr__(
            self, "source_projection", MappingProxyType(dict(self.source_projection))
        )
        object.__setattr__(
            self,
            "observation_sources",
            MappingProxyType(dict(self.observation_sources)),
        )
        object.__setattr__(self, "observation_counts", tuple(self.observation_counts))
        if len(self.observation_sources) != len(self.observation_counts):
            raise ValueError("Every observation source needs its declared row count.")
        aliases = {
            group: tuple(names) for group, names in self.observation_aliases.items()
        }
        if aliases.keys() != self.observation_sources.keys() or any(
            not names or self.observation_sources[group] not in names
            for group, names in aliases.items()
        ):
            raise ValueError("Every observation source needs its producing aliases.")
        object.__setattr__(self, "observation_aliases", MappingProxyType(aliases))
        object.__setattr__(
            self,
            "exit_targets",
            MappingProxyType(
                {
                    identifier: MappingProxyType(dict(targets))
                    for identifier, targets in self.exit_targets.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "exit_adjustments",
            MappingProxyType(
                {
                    identifier: MappingProxyType(dict(adjustments))
                    for identifier, adjustments in self.exit_adjustments.items()
                }
            ),
        )
        if self.read.target != self.context.regime_name:
            raise ValueError("A query's target differs from its canonical Q/F owner.")
        if self.read.kind is QueryKind.RAW and self.participation is not None:
            raise ValueError("Raw queries cannot impose dissolution participation.")
        self._validate_reader_contract()

    def _validate_reader_contract(self) -> None:
        """Keep nonpublishing numeric readers out of policy action entrypoints."""
        consumer = self.read.kind in {QueryKind.ARRIVAL, QueryKind.INTEGRATED_FULL}
        if consumer:
            if self.operand_builder is None or self.decision is None:
                raise ValueError(
                    "An arrival reader needs explicit operands and a rule."
                )
            if self.participation is not None:
                raise ValueError(
                    "An arrival reader cannot define action participation."
                )
            if self.read.kind is QueryKind.INTEGRATED_FULL and self.inputs:
                raise ValueError("An integrated value reader cannot read policies.")
        elif (
            self.operand_builder is not None
            or self.operand_value_reads
            or self.answer_shape is not None
        ):
            raise ValueError("Policy queries must use their canonical Q/F operands.")
        object.__setattr__(self, "operand_value_reads", tuple(self.operand_value_reads))
        if self.answer_shape is not None:
            shape = tuple(self.answer_shape)
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise ValueError("A reader answer shape needs nonnegative dimensions.")
            object.__setattr__(self, "answer_shape", shape)

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the answer owner's tuple shape, independent of physical laws."""
        if self.answer_shape is not None:
            return self.answer_shape
        stakeholders = self.context.stakeholders
        return () if stakeholders is None else (len(stakeholders),)


def _node_inputs(
    *, node: QueryNode, period: int
) -> tuple[CoreArgumentBuilder, tuple[ValueRead, ...]]:
    """Keep value-reader operands separate from canonical policy action inputs."""
    if node.operand_builder is not None:
        return node.operand_builder, node.operand_value_reads
    return grid_search_query_inputs(context=node.context, period=period)


# Values at one common action, stable ID, support, integrity, reached-landing counts.
type PolicyAnswer = tuple[FloatND, IntND, BoolND, BoolND, IntND]


def _query_landing_checks(*, node: QueryNode) -> tuple[LandingCheck, ...]:
    if node.read.kind in {QueryKind.ARRIVAL, QueryKind.INTEGRATED_FULL}:
        return ()
    return getattr(
        node.context.Q_and_F_functions[node.read.period], "landing_checks", ()
    )


def _evaluated_query_landings(*, node: QueryNode, evaluated: EvaluatedQAndF) -> IntND:
    """Require the original canonical count vector in declared check order."""
    checks = _query_landing_checks(node=node)
    if checks:
        if len(evaluated.landing_counts) != 1:
            raise TypeError("Canonical query lost its declared landing-count vector.")
        landings = evaluated.landing_counts[0]
        if landings.shape != (len(checks),) or landings.dtype != jnp.int32:
            raise TypeError("Canonical query landing counts changed shape or dtype.")
    else:
        if evaluated.landing_counts:
            raise TypeError("Canonical query returned undeclared landing checks.")
        landings = jnp.zeros((0,), dtype=jnp.int32)
    return landings


def _query_landing_totals(*, counts: IntND) -> IntND:
    """Reduce source/information axes, retaining the declared check axis."""
    return sum_counts(counts=counts, axis=tuple(range(counts.ndim - 1)))


@dataclass(frozen=True, kw_only=True, eq=False)
class ScalarPolicyQueries:
    """The same bound decision at one original subject or mapped destination.

    Simulation owns native argument placement, inventory and row custody. Every
    query's current parameters and canonical continuation enter explicitly.
    """

    definitions: Mapping[str, QueryNode]
    """Phase-resolved canonical query nodes, including complete mapped contexts."""
    root: str
    """Full for source choice, Participating for once-only destination action."""
    period: int
    """Exact computational period of the phase-specific Q/F declarations."""

    def project(
        self,
        *,
        query: str,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> Mapping[str, FloatND | IntND]:
        """Use the same canonical settlement at the immutable original source."""
        fixed = arguments[query]
        physical = {} if frame is None else frame.physical
        return _mapped_query_states(
            node=self.definitions[query],
            available={
                **cast("Mapping[str, PytreeValue]", fixed[_PROJECTION]),
                **states,
                **physical,
                "period": fixed["period"],
                "age": fixed["age"],
                _PHYSICAL_CONTEXTS: {self.root: physical},
            },
        )

    def available(
        self,
        *,
        query: str,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> BoolND:
        """Test the producing-context guard before its projection or physical law."""
        fixed = arguments[query]
        physical = {} if frame is None else frame.physical
        return _query_availability(
            node=self.definitions[query],
            available={
                **cast("Mapping[str, PytreeValue]", fixed[_PROJECTION]),
                **states,
                **physical,
                "period": fixed["period"],
                "age": fixed["age"],
                _PHYSICAL_CONTEXTS: {self.root: physical},
            },
        )

    def observation_available(
        self,
        *,
        query: str,
        group: ShockGroupName,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> BoolND:
        """Reveal one signal when any proved-equivalent producing alias is active."""
        physical = {} if frame is None else frame.physical
        return _observation_available(
            node=self.definitions[query],
            group=group,
            definitions=self.definitions,
            arguments=arguments,
            available={
                **states,
                **physical,
                _PHYSICAL_CONTEXTS: {self.root: physical},
            },
        )

    def observation_inputs(
        self,
        *,
        query: str,
        group: ShockGroupName,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> Mapping[str, FloatND | IntND]:
        """Map only the proved producing law inputs after its guard is admitted."""
        physical = {} if frame is None else frame.physical
        return _observation_inputs(
            node=self.definitions[query],
            group=group,
            definitions=self.definitions,
            arguments=arguments,
            period=self.period,
            available={
                **states,
                **physical,
                _PHYSICAL_CONTEXTS: {self.root: physical},
            },
        )

    def arguments(
        self,
        *,
        context: CoreBuildContext,
        fixed_flat_params: FlatParams = MappingProxyType({}),
    ) -> Mapping[str, Mapping[str, PytreeValue | ShapeDtypePytree]]:
        """Build canonical explicit operands for each demanded scalar query.

        Simulation supplies the immutable original state pool in ``context``;
        target action grids, law nodes and continuation operands retain their
        existing native argument builders.
        """
        return MappingProxyType(
            {
                name: _QueryArguments(
                    node=node,
                    base=_node_inputs(node=node, period=node.read.period)[0],
                    fixed_flat_params=fixed_flat_params,
                    information_nodes={
                        key: self.definitions[key]
                        for key in dict.fromkeys(
                            (
                                *node.ancestors,
                                *(
                                    producer
                                    for aliases in node.observation_aliases.values()
                                    for producer in aliases
                                ),
                                *(
                                    source
                                    for ancestor in node.ancestors
                                    for aliases in self.definitions[
                                        ancestor
                                    ].observation_aliases.values()
                                    for source in aliases
                                ),
                            )
                        )
                    },
                )(context)
                for name, node in self.definitions.items()
            }
        )

    def frame_arguments(
        self,
        *,
        source_state_names: frozenset[StateName],
        flat_params: FlatParams,
        ages: TimeAxis,
        fixed_flat_params: FlatParams = MappingProxyType({}),
    ) -> Mapping[str, Mapping[str, PytreeValue]]:
        """Bind only clock and settlement operands, without policy value inputs."""
        params = _query_flat_params(
            flat_params=flat_params, fixed_flat_params=fixed_flat_params
        )
        return MappingProxyType(
            {
                name: {
                    "period": jnp.int32(node.read.period),
                    "age": ages.values[node.read.period],
                    _PROJECTION: _projection_operands(
                        node=node,
                        source_state_names=source_state_names,
                        flat_params=params,
                    ),
                }
                for name, node in self.definitions.items()
            }
        )

    @property
    def landing_checks(self) -> tuple[QueryLandingCheck, ...]:
        """Match the reached-landing lane to each actual producing query/date."""
        order = demanded_query_order(
            reads={name: node.read for name, node in self.definitions.items()},
            dependencies={
                name: tuple(node.inputs.values())
                for name, node in self.definitions.items()
            },
            roots=(self.root,),
        )
        return tuple(
            query_landing_check(
                check=check, source=node.read.target, period=node.read.period
            )
            for name in order
            for node in (self.definitions[name],)
            for check in _query_landing_checks(node=node)
        )

    def value_reads(self, *, core_key: str) -> tuple[ValueRead, ...]:
        """Address every continuation at its exact nested simulation operand."""
        source_regime = self.definitions[self.root].read.target
        return tuple(
            replace(
                read,
                source=replace(
                    read.source,
                    source_regime=source_regime,
                    core_key=core_key,
                    argument="query_arguments",
                    path=(
                        name,
                        read.source.argument or read.source.channel.value,
                        *read.source.path,
                    ),
                ),
            )
            for name, node in self.definitions.items()
            for read in _node_inputs(node=node, period=node.read.period)[1]
        )

    def __call__(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> PolicyAnswer:
        """Resolve only demanded typed reads and invoke their shared scalar rule."""
        order = demanded_query_order(
            reads={name: node.read for name, node in self.definitions.items()},
            dependencies={
                name: tuple(node.inputs.values())
                for name, node in self.definitions.items()
            },
            roots=(self.root,),
        )
        answers = {}
        for name in order:
            node = self.definitions[name]
            fixed = {
                **{
                    key: value
                    for key, value in arguments[name].items()
                    if name == self.root or key != "taste_shock_key"
                },
                _ACTING: name == self.root
                and node.read.kind is QueryKind.PARTICIPATING,
            }
            from _lcm.query_information import (  # noqa: PLC0415
                InformationQueryCell,
                has_information,
            )

            if has_information(
                node=node, definitions=self.definitions, period=node.read.period
            ):
                cell = InformationQueryCell(
                    name=name,
                    node=node,
                    definitions=self.definitions,
                    period=node.read.period,
                    fixed={
                        key: value for key, value in fixed.items() if key != _PROJECTION
                    },
                    projection_operands=cast(
                        "Mapping[str, PytreeValue]", fixed[_PROJECTION]
                    ),
                    information_arguments=arguments,
                    realised_root=self.root if frame is not None else None,
                    root_frame=frame,
                )
                answer = cell(
                    live=jnp.asarray(a=True),
                    source=states,
                    dependencies={
                        operand: answers[producer]
                        for operand, producer in node.inputs.items()
                    },
                )
                answers[name] = (
                    *answer[:3],
                    jnp.all(answer[3]),
                    _query_landing_totals(counts=answer[4]),
                )
                continue
            answers[name] = _query_cell(
                jnp.asarray(a=True),
                states,
                {
                    operand: answers[producer]
                    for operand, producer in node.inputs.items()
                },
                node=node,
                period=node.read.period,
                fixed={
                    key: value for key, value in fixed.items() if key != _PROJECTION
                },
                projection_operands=cast(
                    "Mapping[str, PytreeValue]", fixed[_PROJECTION]
                ),
            )
        answer = answers[self.root]
        return (
            answer[0],
            answer[1][0] if answer[1].ndim else answer[1],
            answer[2],
            answer[3],
            jnp.concatenate(tuple(answers[name][4] for name in order)),
        )


def build_policy_query_kernel(
    *,
    root_context: SolverBuildContext,
    period: int,
    definitions: Mapping[str, QueryNode],
    root: str,
    chunk_size: int,
    recompute: bool = False,
) -> _PolicyQueryKernel:
    """Expand one genuine occupant root into a common-chunk native query graph."""
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("A native query chunk must be positive.")
    if {"_initialize", "_source", "_publish", "_finish"} & definitions.keys():
        raise ValueError("Query keys collide with native publication programs.")
    _admit_query_solver(context=root_context, period=period)
    root_space = root_context.state_action_space
    size = math.prod(value.size for value in root_space.states.values())
    chunk_size = min(chunk_size, size)
    stakeholders = root_context.stakeholders
    stakeholder_shape = () if stakeholders is None else (len(stakeholders),)
    value_shape = (
        tuple(value.size for value in root_space.states.values()) + stakeholder_shape
    )
    bank_shape = (math.ceil(size / chunk_size) * chunk_size, *stakeholder_shape)
    dtype = jnp.asarray(0.0).dtype
    host_buffers = MappingProxyType(
        {
            "query_banks": jax.ShapeDtypeStruct((2, *bank_shape), dtype),
            "query_source_live": jax.ShapeDtypeStruct((chunk_size,), jnp.bool_),
        }
        | (
            {
                "query_source_states": {
                    name: jax.ShapeDtypeStruct((chunk_size,), value.dtype)
                    for name, value in root_space.states.items()
                }
            }
            if root_space.states
            else {}
        )
    )
    order = demanded_query_order(
        reads={name: node.read for name, node in definitions.items()},
        dependencies={
            name: tuple(node.inputs.values()) for name, node in definitions.items()
        },
        roots=(root,),
    )
    if (
        definitions[root].read.kind is not QueryKind.FULL
        or definitions[root].read.target != root_context.regime_name
    ):
        raise ValueError("Only the occupant root's own Full query may publish.")
    programs = {
        "_initialize": CoreProgram(
            name="_initialize",
            function=_initialize_bank(shape=bank_shape, dtype=dtype),
            argument_builder=_EmptyArguments(),
            requirements=CoreExecutionRequirements(
                host_buffers=host_buffers, host_axis_names=("query_cell",)
            ),
            output_roles=StateAxesLeading(
                state_names=(), shape=bank_shape, dtype=dtype
            ),
            disposition=CoreExecutionDisposition.PLANNED,
        )
    }
    programs["_source"] = CoreProgram(
        name="_source",
        function=_source_chunk(width=chunk_size),
        argument_builder=_SourceArguments(regime=root_context.regime_name),
        requirements=CoreExecutionRequirements(
            host_buffers=host_buffers, host_axis_names=("query_cell",)
        ),
        output_roles=(
            {
                name: StateAxesLeading(
                    state_names=(), shape=(chunk_size,), dtype=value.dtype
                )
                for name, value in root_space.states.items()
            },
            StateAxesLeading(state_names=(), shape=(chunk_size,), dtype=jnp.bool_),
        ),
        internal_outputs=(
            InternalOutputSpec(label="states", path=(0,)),
            InternalOutputSpec(label="live", path=(1,)),
        ),
        disposition=CoreExecutionDisposition.PLANNED,
    )
    builders: dict[str, _QueryArguments] = {}
    for name in order:
        node = definitions[name]
        _admit_node(node=node, period=period)
        base, value_reads = _node_inputs(node=node, period=period)
        builder = _QueryArguments(
            node=node,
            base=base,
            information_nodes={
                key: definitions[key]
                for key in dict.fromkeys(
                    (
                        *node.ancestors,
                        *(
                            producer
                            for aliases in node.observation_aliases.values()
                            for producer in aliases
                        ),
                        *(
                            source
                            for ancestor in node.ancestors
                            for aliases in definitions[
                                ancestor
                            ].observation_aliases.values()
                            for source in aliases
                        ),
                    )
                )
            },
        )
        builders[name] = builder
        reads = tuple(
            replace(
                read,
                source=replace(
                    read.source, source_regime=root_context.regime_name, core_key=name
                ),
            )
            for read in value_reads
        )
        from _lcm.query_information import (  # noqa: PLC0415 - keep information policy beside query core
            has_information,
            support_shape,
        )

        information = has_information(node=node, definitions=definitions, period=period)
        information_shape = (
            support_shape(node=node, definitions=definitions, period=period)
            if information
            else ()
        )
        shape = (chunk_size, *information_shape, *node.value_shape)
        programs[name] = CoreProgram(
            name=name,
            function=_QueryCore(
                node=node, period=period, name=name, definitions=definitions
            ),
            argument_builder=builder,
            requirements=CoreExecutionRequirements(
                value_reads=reads,
                internal_inputs={
                    operand: InternalInputRef(producer=producer, label="answer")
                    for operand, producer in node.inputs.items()
                }
                | {
                    _SOURCE: InternalInputRef(producer="_source", label="states"),
                    _LIVE: InternalInputRef(producer="_source", label="live"),
                },
                host_axis_names=("query_cell",),
                host_buffers=host_buffers,
            ),
            output_roles=(
                StateAxesLeading(
                    state_names=(), shape=shape, dtype=jnp.asarray(0.0).dtype
                ),
                StateAxesLeading(
                    state_names=(),
                    shape=(chunk_size,),
                    dtype=jnp.int32,
                ),
                StateAxesLeading(
                    state_names=(),
                    shape=(chunk_size, *information_shape),
                    dtype=jnp.bool_,
                ),
                StateAxesLeading(state_names=(), shape=(), dtype=jnp.bool_),
                StateAxesLeading(
                    state_names=(),
                    shape=(len(_query_landing_checks(node=node)),),
                    dtype=jnp.int32,
                ),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            internal_outputs=(InternalOutputSpec(label="answer", path=()),),
        )
    bank_arguments = _BankArguments(
        regime=root_context.regime_name,
        shape=bank_shape,
        dtype=dtype,
    )
    programs["_publish"] = CoreProgram(
        name="_publish",
        function=_write_query_chunk,
        argument_builder=bank_arguments,
        requirements=CoreExecutionRequirements(
            internal_inputs={"answer": InternalInputRef(producer=root, label="answer")},
            host_buffers=host_buffers,
            host_axis_names=("query_cell",),
        ),
        output_roles=StateAxesLeading(state_names=(), shape=bank_shape, dtype=dtype),
        disposition=CoreExecutionDisposition.PLANNED,
    )
    programs["_finish"] = CoreProgram(
        name="_finish",
        function=_finish_bank(shape=value_shape, size=size),
        argument_builder=replace(bank_arguments, write=False),
        requirements=CoreExecutionRequirements(
            host_buffers=host_buffers, host_axis_names=("query_cell",)
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
    )
    schedule = _closed_query_schedule(
        definitions={name: definitions[name] for name in order},
        root=root,
        recompute=recompute,
    )
    programs = {
        name: replace(
            program,
            requirements=replace(program.requirements, internal_schedule=schedule),
        )
        for name, program in programs.items()
    }
    return _PolicyQueryKernel(
        programs=MappingProxyType(programs),
        builders=MappingProxyType(builders),
        definitions=MappingProxyType({name: definitions[name] for name in order}),
        root=root,
        chunk_size=chunk_size,
        recompute=recompute,
        root_context=root_context,
        period=period,
    )


@dataclass(frozen=True, kw_only=True)
class _EmptyArguments:
    """Native initialization has no materialized input owner."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, PytreeValue]:
        del context
        return MappingProxyType({})


@dataclass(frozen=True, kw_only=True, eq=False)
class _InitializeBank:
    """Allocate only after native compiler and owner admission has succeeded."""

    shape: tuple[int, ...]
    dtype: jnp.dtype

    def __call__(self) -> FloatND:
        return jnp.zeros(self.shape, dtype=self.dtype)


@functools.cache
def _initialize_bank(*, shape: tuple[int, ...], dtype: jnp.dtype) -> _InitializeBank:
    """Reuse the same shape-only callable when period compilation keys agree."""
    return _InitializeBank(shape=shape, dtype=dtype)


@dataclass(frozen=True, kw_only=True)
class _BankArguments:
    """Array-free compiler inputs for the host-owned publication carry."""

    regime: RegimeName
    shape: tuple[int, ...]
    dtype: jnp.dtype
    write: bool = True

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        sharding = context.next_regime_to_V_arr[self.regime].sharding
        if isinstance(sharding, jax.NamedSharding):
            sharding = jax.NamedSharding(sharding.mesh, jax.P())
        arguments = {
            "bank": jax.ShapeDtypeStruct(self.shape, self.dtype, sharding=sharding)
        }
        if self.write:
            arguments["start"] = jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding)
        return MappingProxyType(arguments)


def _write_query_chunk(*, bank: FloatND, start: IntND, answer: PolicyAnswer) -> FloatND:
    """Update a padded bank using the same admitted query tuple and chunk."""
    starts = (start, *(jnp.zeros_like(start) for _ in range(bank.ndim - 1)))
    return jax.lax.dynamic_update_slice(bank, answer[0], starts)


@dataclass(frozen=True, kw_only=True, eq=False)
class _FinishBank:
    """Remove only padding and restore the genuine durable state-axis order."""

    shape: tuple[int, ...]
    size: int

    def __call__(self, *, bank: FloatND) -> FloatND:
        return bank[: self.size].reshape(self.shape)


@functools.cache
def _finish_bank(*, shape: tuple[int, ...], size: int) -> _FinishBank:
    """Share only the identical static publication shape, without runtime owners."""
    return _FinishBank(shape=shape, size=size)


def _admit_node(
    *, node: QueryNode, period: int, same_period_values: bool = False
) -> None:
    """Refuse an unrepresented action/status or information intersection.

    `same_period_values` says the caller supplies the query date's stored values,
    which a target's value-dependent constraint reads.
    """
    context = node.context
    if node.read.phase != "solve" or node.read.period != period:
        raise UnsupportedOperationError(
            "A policy query needs its exact solve phase/period."
        )
    if node.read.kind is QueryKind.INTEGRATED_FULL:
        _fail_if_projection_incomplete(node=node, reader="A stored-value reader")
        return
    _admit_query_solver(context=context, period=period)
    if period not in context.Q_and_F_functions:
        raise ValueError("The target action problem is inactive at this period.")
    _admit_query_objective(context=context, period=period)
    _fail_if_projection_incomplete(node=node, reader="A query")
    if context.has_taste_shocks and context.stakeholders is not None:
        raise UnsupportedOperationError(
            "A collective query does not declare an EV1 stakeholder-taste law."
        )
    if (
        context.fold_state_names
        or context.co_map_state_names
        or context.invariant_bindings
        or context.action_partitions != 1
    ):
        raise UnsupportedOperationError(
            "This query closure needs an unblocked, unfolded action problem."
        )
    if (
        context.same_period_ref_regimes and not same_period_values
    ) or context.edge_reference_regimes:
        raise UnsupportedOperationError(
            "A query cannot price another root's stored Full value."
        )
    if runtime_bound_specs(function=context.Q_and_F_functions[period]):
        raise UnsupportedOperationError(
            "Query bound integrity must have a native status adapter."
        )


def _fail_if_projection_incomplete(*, node: QueryNode, reader: str) -> None:
    """Name the states a query's projection must cover and the ones it covers."""
    expected = sorted(node.context.state_action_space.state_names)
    if sorted(node.read.projection) != expected:
        raise ValueError(
            f"{reader} of regime {node.read.target!r} needs exactly one projection "
            f"per state of its value function ({expected}); got "
            f"{sorted(node.read.projection)}."
        )


def _admit_query_objective(*, context: SolverBuildContext, period: int) -> None:
    """Refuse a separate selection profile the query does not transport."""
    if has_objective_profile(Q_and_F=context.Q_and_F_functions[period]):
        raise UnsupportedOperationError(
            "Native policy query does not represent Decoupled objective selection; "
            "the query reduces reported values."
        )


def _admit_query_solver(*, context: SolverBuildContext, period: int) -> None:
    """Never replace an endogenous-grid policy by an action-grid maximization."""
    declaration = context.user_regimes_at(period=period)[context.regime_name]
    if not isinstance(declaration.solver, GridSearch):
        raise UnsupportedOperationError(
            f"Native policy query for {context.regime_name!r} requires its declared "
            "GridSearch solver; endogenous-grid query reconstruction is separate."
        )


@dataclass(frozen=True, kw_only=True)
class _SourceArguments:
    """Pass real source grids and an abstract scalar offset to native gathering."""

    regime: RegimeName

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        sharding = context.next_regime_to_V_arr[self.regime].sharding
        if isinstance(sharding, jax.NamedSharding):
            sharding = jax.NamedSharding(sharding.mesh, jax.P())
        return MappingProxyType(
            {
                "states": cast("StateActionSpace", context.state_action_space).states,
                "start": jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding),
            }
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SourceChunk:
    """Gather a padded source chunk under native compiler/placement admission."""

    width: int

    def __call__(
        self, *, states: Mapping[str, FloatND | IntND], start: IntND
    ) -> tuple[dict[str, FloatND | IntND], BoolND]:
        extents = tuple(value.size for value in states.values())
        size = math.prod(extents)
        positions = jnp.arange(self.width, dtype=jnp.int32) + start
        indices = (
            jnp.unravel_index(jnp.minimum(positions, size - 1), extents)
            if extents
            else ()
        )
        return (
            {
                name: value[index]
                for (name, value), index in zip(states.items(), indices, strict=True)
            },
            positions < size,
        )


@functools.cache
def _source_chunk(*, width: int) -> _SourceChunk:
    """Reuse the identical width-only gather without retaining source arrays."""
    return _SourceChunk(width=width)


@dataclass(frozen=True, kw_only=True)
class _QueryArguments:
    """Canonical target arguments plus one fixed source chunk prototype."""

    node: QueryNode
    base: Callable[..., Mapping[str, PytreeValue | ShapeDtypePytree]]
    fixed_flat_params: FlatParams = MappingProxyType({})
    information_nodes: Mapping[str, QueryNode] = MappingProxyType({})

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        """Build the exact prototype shared by lowering and every real chunk."""
        context = replace(
            context,
            flat_params=_query_flat_params(
                flat_params=cast("FlatParams", context.flat_params),
                fixed_flat_params=self.fixed_flat_params,
            ),
        )
        arguments = dict(
            self.base(
                replace(
                    context,
                    period=self.node.read.period,
                    state_action_space=self.node.context.state_action_space,
                )
            )
        )
        arguments[_PROJECTION] = _query_projection_operands(
            node=self.node, context=context
        )
        if self.information_nodes:
            from _lcm.current_iid import (  # noqa: PLC0415
                CURRENT_IID_ARGUMENT,
                CURRENT_IID_NODES_ARGUMENT,
                CURRENT_IID_ROWS_ARGUMENT,
            )

            keys = {
                "period",
                "age",
                CURRENT_IID_ARGUMENT,
                CURRENT_IID_NODES_ARGUMENT,
                CURRENT_IID_ROWS_ARGUMENT,
            }
            arguments[_INFORMATION] = cast(
                "ShapeDtypePytree",
                {
                    name: {
                        **{
                            key: value
                            for key, value in _node_inputs(
                                node=node, period=node.read.period
                            )[0](
                                replace(
                                    context,
                                    period=node.read.period,
                                    state_action_space=node.context.state_action_space,
                                )
                            ).items()
                            if key in keys
                        },
                        _PROJECTION: _query_projection_operands(
                            node=node, context=context
                        ),
                    }
                    for name, node in self.information_nodes.items()
                },
            )
        return MappingProxyType(arguments)


def query_parameter_namespaces(*, builder: CoreArgumentBuilder) -> frozenset[str]:
    """Name fixed parameter roots read by native queries, including arrivals."""
    from _lcm.arrival_publication import _PublicationArguments  # noqa: PLC0415
    from _lcm.arrival_queries import (  # noqa: PLC0415
        ArrivalSourceArguments,
        _ArrivalArguments,
        _IntegratedArguments,
    )

    while isinstance(builder, _PublicationArguments):
        builder = builder.base
    if isinstance(builder, _GridSearchArgumentBuilder):
        return frozenset(
            name
            for plan in builder.arrival_queries.values()
            for node in plan.queries.definitions.values()
            for name in _query_node_parameter_namespaces(node=node, information=False)
        ) | (
            frozenset(name for name, _, _ in builder.current_iid_plan.bindings)
            if builder.arrival_queries and builder.current_iid_plan is not None
            else frozenset()
        )
    if isinstance(builder, ArrivalSourceArguments):
        return query_parameter_namespaces(builder=builder.base) | frozenset(
            name
            for plan in builder.plans.values()
            for node in plan.queries.definitions.values()
            for name in _query_node_parameter_namespaces(node=node, information=False)
        )
    if isinstance(builder, _ArrivalArguments):
        return frozenset(
            name
            for name, _, _ in (() if builder.plan is None else builder.plan.bindings)
        )
    if isinstance(builder, _IntegratedArguments):
        return frozenset({builder.owner}) if builder.helper_names else frozenset()
    if not isinstance(builder, _QueryArguments):
        return frozenset()
    return _query_node_parameter_namespaces(
        node=builder.node, information=False
    ) | frozenset(
        name
        for node in builder.information_nodes.values()
        for name in _query_node_parameter_namespaces(node=node, information=True)
    )


def _query_node_parameter_namespaces(
    *, node: QueryNode, information: bool
) -> frozenset[str]:
    """Mirror the node's action/law and exact producing projection operands."""
    from _lcm.arrival_queries import _ArrivalArguments  # noqa: PLC0415

    plan = (
        node.context.current_iid_plans.get(node.read.period)
        if node.operand_builder is None
        else node.operand_builder.plan
        if isinstance(node.operand_builder, _ArrivalArguments)
        else None
    )
    names = {name for name, _, _ in (() if plan is None else plan.bindings)} | {
        path[0]
        for path in node.read.operand_paths.values()
        if len(path) == _FLAT_OPERAND_PATH_LENGTH and isinstance(path[0], str)
    }
    if not information:
        if node.operand_builder is None:
            names.add(node.context.regime_name)
            names.update(
                name
                for plan in node.context.arrival_queries.get(
                    node.read.period, {}
                ).values()
                for child in plan.queries.definitions.values()
                for name in _query_node_parameter_namespaces(
                    node=child, information=False
                )
            )
        else:
            names.update(query_parameter_namespaces(builder=node.operand_builder))
    return frozenset(names)


def _query_projection_operands(
    *, node: QueryNode, context: CoreBuildContext
) -> Mapping[str, PytreeValue]:
    """Bind each producing context's exact current-call settlement parameters."""
    return _projection_operands(
        node=node,
        source_state_names=frozenset(
            cast("StateActionSpace", context.state_action_space).state_names
        ),
        flat_params=cast("FlatParams", context.flat_params),
    )


def _query_flat_params(
    *, flat_params: FlatParams, fixed_flat_params: FlatParams
) -> FlatParams:
    """Merge canonical namespaces with current-call parameters taking precedence."""
    namespaces: dict[tuple[str, ...], FlatRegimeParams] = {}
    for parameters in (fixed_flat_params, flat_params):
        for path, leaves in flat_namespaces(parameters):
            namespaces[path] = MappingProxyType({**namespaces.get(path, {}), **leaves})
    merged: dict[RegimeName, FlatRegimeParams | FlatEdgeParams] = {
        path[0]: leaves for path, leaves in namespaces.items() if path[0] != EDGES
    }
    if EDGES in fixed_flat_params or EDGES in flat_params:
        merged[EDGES] = MappingProxyType(
            {path[1]: leaves for path, leaves in namespaces.items() if path[0] == EDGES}
        )
    return MappingProxyType(merged)


def _projection_operands(
    *,
    node: QueryNode,
    source_state_names: frozenset[StateName],
    flat_params: FlatParams,
) -> Mapping[str, PytreeValue]:
    """Read the same canonical settlement paths for policies and entry frames."""
    operands = {}
    for name, path in node.read.operand_paths.items():
        if name in source_state_names or name in {"period", "age"}:
            continue
        if len(path) != _FLAT_OPERAND_PATH_LENGTH or not all(
            isinstance(step, str) for step in path
        ):
            raise ValueError("Query operands need canonical flat namespace paths.")
        operands[name] = regime_kernel_params(
            flat_params, regime_name=cast("str", path[0])
        )[cast("str", path[1])]
    return MappingProxyType(operands)


@dataclass(frozen=True, kw_only=True, eq=False)
class _QueryCore:
    """Map canonical target action selection over a fixed source chunk."""

    node: QueryNode
    period: int
    name: str = ""
    definitions: Mapping[str, QueryNode] = MappingProxyType({})

    def __call__(self, **arguments: PytreeValue) -> PolicyAnswer:
        """Select target actions at mapped coordinates, retaining common tuples."""
        source = cast("Mapping[str, FloatND | IntND]", arguments[_SOURCE])
        live = cast("BoolND", arguments[_LIVE])
        dependencies = {
            name: cast("PolicyAnswer", arguments[name]) for name in self.node.inputs
        }
        fixed = {
            name: value
            for name, value in arguments.items()
            if name not in {*self.node.inputs, _SOURCE, _LIVE, _PROJECTION}
        }
        projection_operands = cast("Mapping[str, PytreeValue]", arguments[_PROJECTION])
        from _lcm.query_information import (  # noqa: PLC0415
            InformationQueryCell,
            has_information,
        )

        if has_information(
            node=self.node, definitions=self.definitions, period=self.period
        ):
            cell = InformationQueryCell(
                name=self.name,
                node=self.node,
                definitions=self.definitions,
                period=self.period,
                fixed=fixed,
                projection_operands=projection_operands,
                information_arguments=cast(
                    "Mapping[str, Mapping[str, PytreeValue]]",
                    arguments.get(_INFORMATION, {}),
                ),
            )

            def information_cell(
                item: tuple[
                    BoolND,
                    Mapping[str, FloatND | IntND],
                    Mapping[str, tuple[FloatND, IntND, BoolND]],
                ],
            ) -> PolicyAnswer:
                active, states, rows = item
                return cell(
                    live=active,
                    source=states,
                    dependencies={
                        name: (*answer, dependencies[name][3], dependencies[name][4])
                        for name, answer in rows.items()
                    },
                )

            answer = jax.lax.map(
                information_cell,
                (
                    live,
                    source,
                    {name: answer[:3] for name, answer in dependencies.items()},
                ),
            )
            # Information consumers select their own actions; only values travel.
            return cast(
                "PolicyAnswer",
                (
                    answer[0],
                    jnp.full(live.shape, -1, dtype=jnp.int32),
                    answer[2],
                    jnp.all(answer[3]),
                    _query_landing_totals(counts=answer[4]),
                ),
            )
        cell = functools.partial(
            _query_cell,
            node=self.node,
            period=self.period,
            fixed=fixed,
            projection_operands=projection_operands,
        )
        if self.node.available is None and self.node.decision is None:
            answer = jax.vmap(
                cell, in_axes=(0, 0, dict.fromkeys(dependencies, (0, 0, 0, None, None)))
            )(live, source, dependencies)
        else:
            # A batched cond becomes select. Keep unsafe unavailable benchmark
            # arithmetic behind a scalar conditional, including in transposes.
            def guarded(
                item: tuple[
                    BoolND,
                    Mapping[str, FloatND | IntND],
                    Mapping[str, tuple[FloatND, IntND, BoolND]],
                ],
            ) -> PolicyAnswer:
                active, states, rows = item
                return cell(
                    active,
                    states,
                    {
                        name: (*answer, dependencies[name][3], dependencies[name][4])
                        for name, answer in rows.items()
                    },
                )

            answer = jax.lax.map(
                guarded,
                (
                    live,
                    source,
                    {name: answer[:3] for name, answer in dependencies.items()},
                ),
            )
        return cast(
            "PolicyAnswer",
            (*answer[:3], jnp.all(answer[3]), _query_landing_totals(counts=answer[4])),
        )


# keyword-only-exempt: library-callback=jax.vmap
def _query_cell(
    live: BoolND,
    source: Mapping[str, FloatND | IntND],
    dependencies: Mapping[str, PolicyAnswer],
    *,
    node: QueryNode,
    period: int,
    fixed: Mapping[str, PytreeValue],
    projection_operands: Mapping[str, PytreeValue],
) -> PolicyAnswer:
    """Evaluate ordinary feasibility and the kept set before one canonical max."""
    available = {
        **projection_operands,
        **source,
        "period": fixed["period"],
        "age": fixed["age"],
        _PHYSICAL_CONTEXTS: fixed.get(_PHYSICAL_CONTEXTS, {}),
        **{name: fixed[name] for name in node.read.physical_inputs if name in fixed},
    }
    if node.available is not None or node.decision is not None:
        active = _query_availability(node=node, available=available)

        def admitted() -> PolicyAnswer:
            return _active_query_cell(
                live=live,
                source=source,
                dependencies=dependencies,
                node=node,
                period=period,
                fixed=fixed,
                projection_operands=projection_operands,
            )

        absent = vary_carry_like_body(
            body=lambda _carry: admitted(),
            carry=(
                jnp.full(
                    node.value_shape,
                    -jnp.inf,
                ),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(a=False),
                jnp.asarray(a=True),
                jnp.zeros((len(_query_landing_checks(node=node)),), dtype=jnp.int32),
            ),
        )
        return jax.lax.cond(active & live, admitted, lambda: absent)
    return _active_query_cell(
        live=live,
        source=source,
        dependencies=dependencies,
        node=node,
        period=period,
        fixed=fixed,
        projection_operands=projection_operands,
    )


def _query_availability(
    *, node: QueryNode, available: Mapping[str, PytreeValue]
) -> BoolND:
    """One scalar guard binder for native solve and original-row replay."""
    if node.available is None:
        return jnp.asarray(a=True)
    active = jnp.asarray(
        node.available(
            **{
                key: available[key]
                for key in inspect.signature(node.available).parameters
            }
        )
    )
    if active.shape != () or active.dtype != jnp.bool_:
        raise ValueError("Query availability must be a scalar boolean.")
    return active


def _observation_available(
    *,
    node: QueryNode,
    group: ShockGroupName,
    definitions: Mapping[str, QueryNode],
    arguments: Mapping[str, Mapping[str, PytreeValue]],
    available: Mapping[str, PytreeValue],
) -> BoolND:
    """Short-circuit equivalent aliases without evaluating their settlements."""
    active = jnp.asarray(a=False)
    for producer in node.observation_aliases[group]:
        fixed = arguments[producer]

        def read_available(
            *, producer: str = producer, fixed: Mapping[str, PytreeValue] = fixed
        ) -> BoolND:
            return _query_availability(
                node=definitions[producer],
                available={
                    **available,
                    **cast("Mapping[str, PytreeValue]", fixed[_PROJECTION]),
                    "period": fixed["period"],
                    "age": fixed["age"],
                },
            )

        present = vary_carry_like_body(
            body=lambda _carry: read_available(), carry=jnp.asarray(a=True)
        )
        active = jax.lax.cond(active, lambda present=present: present, read_available)
    return active


def _observation_inputs(
    *,
    node: QueryNode,
    group: ShockGroupName,
    definitions: Mapping[str, QueryNode],
    arguments: Mapping[str, Mapping[str, PytreeValue]],
    period: int,
    available: Mapping[str, PytreeValue],
) -> Mapping[str, FloatND | IntND]:
    """Use proved equal law-input maps, excluding unrelated target settlements."""
    producer = node.observation_sources[group]
    target = definitions[producer]
    plan = target.context.current_iid_plans[period]
    if plan is None:
        raise ValueError("An observation needs an actual producing law.")
    binding = next(binding for name, binding, _ in plan.bindings if name == group)
    fixed = arguments[producer]
    return _mapped_query_states(
        node=target,
        names=binding.state_inputs,
        available={
            **available,
            **cast("Mapping[str, PytreeValue]", fixed[_PROJECTION]),
            "period": fixed["period"],
            "age": fixed["age"],
        },
    )


def _active_query_cell(
    *,
    live: BoolND,
    source: Mapping[str, FloatND | IntND],
    dependencies: Mapping[str, PolicyAnswer],
    node: QueryNode,
    period: int,
    fixed: Mapping[str, PytreeValue],
    projection_operands: Mapping[str, PytreeValue],
) -> PolicyAnswer:
    """Evaluate only a caller-admitted mapped action problem."""
    available = {
        **projection_operands,
        **source,
        "period": fixed["period"],
        "age": fixed["age"],
        _PHYSICAL_CONTEXTS: fixed.get(_PHYSICAL_CONTEXTS, {}),
        **{
            name: value
            for name, value in fixed.items()
            if name == _PHYSICAL_CONTEXTS or name in node.read.physical_inputs
        },
    }
    mapped = _mapped_query_states(node=node, available=available)
    arguments = {**fixed, **projection_operands, **mapped}
    arguments.setdefault(_PHYSICAL_CONTEXTS, {})
    context = node.context
    if node.read.kind in {QueryKind.ARRIVAL, QueryKind.INTEGRATED_FULL}:
        operands = {**arguments, **dependencies}
        decision = node.decision
        if decision is None:
            raise ValueError("An arrival reader needs its declared decision rule.")
        answer = decision(
            **{name: operands[name] for name in inspect.signature(decision).parameters}
        )
        _validate_decision_answer(node=node, answer=answer)
        return (
            answer[0],
            answer[1],
            live & answer[2],
            ~live | (answer[3] & _mapped_query_valid(node=node, mapped=mapped)),
            answer[4],
        )
    action_point = with_signature(
        functools.partial(
            _query_action,
            node=node,
            period=period,
            arguments=arguments,
            dependencies=dependencies,
        ),
        args=list(context.state_action_space.action_names),
        enforce=False,
    )
    evaluate = cast(
        "Callable[..., tuple[FloatND, BoolND, BoolND, IntND]]",
        productmap(
            func=action_point,
            variables=context.state_action_space.action_names,
            batch_sizes=dict.fromkeys(context.state_action_space.action_names, 0),
        ),
    )
    q, feasible, valid, landing_counts = evaluate(
        **{name: arguments[name] for name in context.state_action_space.action_names}
    )
    axes = tuple(range(feasible.ndim))
    support = jnp.any(feasible)
    if context.has_taste_shocks:
        action, value, taste_valid = _taste_query_choice(
            node=node,
            q=q,
            feasible=feasible,
            arguments=arguments,
        )
        valid &= taste_valid
    elif context.stakeholders is not None:
        pareto_weights = cast("ParetoWeights", context.pareto_weights)
        action, values, _ = collective_argmax_and_readout(
            stakeholder_Q={
                name: q[..., index] for index, name in enumerate(context.stakeholders)
            },
            feasibility=feasible,
            weights=_evaluate_pareto_weights(
                pareto_weights=pareto_weights,
                states_actions_params=cast(
                    "QAndFKwargs",
                    {name: arguments[name] for name in pareto_weights.arg_names},
                ),
            ),
            action_axes=axes,
        )
        value = jnp.stack(tuple(values[name] for name in context.stakeholders))
    else:
        action, value = argmax_and_max(a=q, where=feasible, initial=-jnp.inf)
    valid = jnp.all(valid) & _mapped_query_valid(node=node, mapped=mapped)
    answer = (
        value,
        jnp.where(support, action, -1).astype(jnp.int32),
        live & support,
        ~live | valid,
        sum_counts(counts=jnp.where(live, landing_counts, 0), axis=axes),
    )
    if node.decision is not None:
        operands = {
            **arguments,
            **dependencies,
            "q": q,
            "feasible": feasible,
            "answer": answer,
        }
        if context.has_taste_shocks:
            from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
                TASTE_SHOCK_SCALE_PARAM,
            )

            operands[_TASTE] = {
                "scale": arguments[TASTE_SHOCK_SCALE_PARAM],
                **(
                    {"key": arguments["taste_shock_key"]}
                    if "taste_shock_key" in arguments
                    else {}
                ),
            }
        answer = node.decision(
            **{
                name: operands[name]
                for name in inspect.signature(node.decision).parameters
            }
        )
        _validate_decision_answer(node=node, answer=answer)
    return answer


def _mapped_query_valid(
    *, node: QueryNode, mapped: Mapping[str, FloatND | IntND]
) -> BoolND:
    """Keep category and finite-state integrity mandatory for every reader kind."""
    context = node.context
    return jnp.all(
        jnp.stack(
            tuple(
                jnp.any(mapped[name] == context.state_action_space.states[name])
                for name, grid in context.grids.items()
                if name in mapped and isinstance(grid, DiscreteGrid)
            )
            or (jnp.asarray(a=True),)
        )
    ) & jnp.all(
        jnp.stack(
            tuple(jnp.all(jnp.isfinite(value)) for value in mapped.values())
            or (jnp.asarray(a=True),)
        )
    )


def _validate_decision_answer(*, node: QueryNode, answer: PolicyAnswer) -> None:
    """Require the declared answer owner and mandatory scalar status channels."""
    if answer[0].shape != node.value_shape:
        raise ValueError("A bound query decision returned a different payoff shape.")
    if answer[1].shape != () or answer[1].dtype != jnp.int32:
        raise ValueError("A bound query decision must return one stable int32 ID.")
    if answer[2].shape != () or answer[2].dtype != jnp.bool_:
        raise ValueError("A bound query decision must return scalar support.")
    if answer[3].shape != () or answer[3].dtype != jnp.bool_:
        raise ValueError("A bound query decision must return scalar integrity.")
    if (
        answer[4].shape != (len(_query_landing_checks(node=node)),)
        or answer[4].dtype != jnp.int32
    ):
        raise ValueError("A bound query decision lost its declared landing counts.")


def _taste_query_choice(
    *,
    node: QueryNode,
    q: FloatND,
    feasible: BoolND,
    arguments: Mapping[str, PytreeValue],
) -> tuple[IntND, FloatND, BoolND]:
    """Price hidden target taste at its logsum; draw only in the acting pass."""
    from _lcm.logsum import logsum_and_softmax  # noqa: PLC0415
    from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
        TASTE_SHOCK_SCALE_PARAM,
        draw_taste_shock_noise,
    )

    context = node.context
    n_discrete = sum(
        isinstance(context.grids[name], DiscreteGrid)
        for name in context.state_action_space.action_names
    )
    if not n_discrete:
        raise ValueError("EV1 policy queries require a discrete action axis.")
    discrete_values, identifiers = _taste_stay_roster(
        q=q, feasible=feasible, n_discrete=n_discrete
    )
    scale = cast("FloatND", arguments[TASTE_SHOCK_SCALE_PARAM])
    if arguments.get(_ACTING, False):
        noise = draw_taste_shock_noise(
            key=cast("jax.Array", arguments["taste_shock_key"]),
            shape=discrete_values.shape,
            scale=scale,
        )
        discrete = jnp.argmax(discrete_values + noise)
        value = discrete_values[discrete]
    else:
        discrete = jnp.argmax(discrete_values)
        value, _ = logsum_and_softmax(values=discrete_values, scale=scale, axes=(0,))
    action = identifiers[discrete]
    return action, value, jnp.all(jnp.isfinite(scale) & (scale > 0))


def _taste_stay_roster(
    *, q: FloatND, feasible: BoolND, n_discrete: int
) -> tuple[FloatND, IntND]:
    """Optimize continuous actions before assigning one taste per discrete cell."""
    discrete_cells = math.prod(q.shape[:n_discrete])
    continuous_cells = math.prod(q.shape[n_discrete:])
    masked = jnp.where(feasible, q, -jnp.inf).reshape(discrete_cells, continuous_cells)
    continuous = jnp.argmax(masked, axis=1)
    identifiers = (
        jnp.arange(discrete_cells, dtype=jnp.int32) * continuous_cells + continuous
    ).astype(jnp.int32)
    return jnp.max(masked, axis=1), identifiers


def _mapped_query_states(
    *,
    node: QueryNode,
    available: Mapping[str, PytreeValue],
    names: frozenset[str] | None = None,
) -> Mapping[str, FloatND | IntND]:
    """Canonical scalar settlement boundary shared by solve and actual routing."""
    mapped = {
        name: jnp.asarray(
            helper(
                **{key: available[key] for key in inspect.signature(helper).parameters}
            )
        )
        for name, helper in node.read.projection.items()
        if names is None or name in names
    }
    if any(value.ndim != 0 for value in mapped.values()):
        raise ValueError("A mapped query state must be scalar per source cell.")
    return mapped


def _query_action(
    *,
    node: QueryNode,
    period: int,
    arguments: Mapping[str, PytreeValue],
    dependencies: Mapping[str, PolicyAnswer],
    **actions: PytreeValue,
) -> tuple[FloatND, BoolND, BoolND, IntND]:
    """Preserve the canonical Q/F arrays and add only declared participation."""
    function = cast(
        "Callable[..., tuple[FloatND, BoolND]]",
        node.context.Q_and_F_functions[period],
    )
    inputs = {**arguments, **actions}
    call_arguments = {
        name: inputs[name] for name in inspect.signature(function).parameters
    }
    evaluated = _shared_evaluation(
        function=cast("CanonicalActionKernel", function), arguments=call_arguments
    )
    parts = evaluated.parts
    q, ordinary = parts.Q, parts.F
    continuation_valid = (
        jnp.asarray(a=True)
        if parts.continuation_valid is None
        else parts.continuation_valid
    )
    # Participation consumes this Q before it decides keep: its rejection cannot
    # erase a continuation landing already needed by an ordinarily feasible action.
    landings = jnp.where(
        ordinary, _evaluated_query_landings(node=node, evaluated=evaluated), 0
    )
    expected_q_shape = (
        () if node.context.stakeholders is None else (len(node.context.stakeholders),)
    )
    if q.shape != expected_q_shape:
        raise ValueError("Query Q must be scalar or the canonical stakeholder tuple.")
    if ordinary.dtype != jnp.bool_:
        raise TypeError("Canonical query feasibility must be boolean.")
    if ordinary.ndim != 0:
        raise ValueError("Canonical query feasibility must be scalar per action.")
    checked_valid = (
        continuation_valid
        if has_projected_queries(function=function)
        else (~ordinary | continuation_valid)
    )
    valid = checked_valid & ~jnp.any(ordinary & jnp.any(jnp.isnan(q)))
    for status in evaluated.bound_status:
        for flag in status:
            valid &= ~jnp.any(flag)
    if node.participation is None:
        return q, ordinary, valid, landings
    operands = {**inputs, **dependencies, "q": q}
    keep = node.participation(
        **{
            name: operands[name]
            for name in inspect.signature(node.participation).parameters
        }
    )
    if keep.dtype != jnp.bool_:
        raise TypeError("Query participation must be boolean.")
    if keep.ndim != 0:
        raise ValueError("Query participation must be scalar per action.")
    return q, ordinary & keep, valid, landings


_SHARED_EVALUATIONS: weakref.WeakKeyDictionary[
    CanonicalActionKernel, dict[Hashable, Callable[..., EvaluatedQAndF]]
] = weakref.WeakKeyDictionary()


def _shared_evaluation(
    *, function: CanonicalActionKernel, arguments: Mapping[str, PytreeValue]
) -> EvaluatedQAndF:
    """Trace one action problem once per abstract signature, however often read.

    An arrival continuation prices the target's own action problem at the
    landing date, so a query's Q/F contains the next date's queries, whose Q/F
    contains the date after. Inlined, every read re-traces that whole subtree.
    A nested `jax.jit` keys the trace on the function and its abstract inputs:
    later reads, the `eval_shape` that sizes a guarded branch, and every other
    program of the same graph reuse the jaxpr. `inline=True` splices that jaxpr
    into the caller, so the lowered program is the one the inlined trace gives.
    Concrete leaves stay closed over, as they were, and an eager evaluation
    stays eager.
    """
    leaves, treedef = jax.tree.flatten(arguments)
    traced = tuple(isinstance(leaf, Tracer) for leaf in leaves)
    if not any(traced):
        return _evaluate(function=function)(**arguments)
    fixed = tuple(
        None if is_traced else leaf
        for leaf, is_traced in zip(leaves, traced, strict=True)
    )
    key = (treedef, traced, tuple(_closed_over_key(leaf=leaf) for leaf in fixed))
    try:
        hash(key)
    except TypeError:
        return _evaluate(function=function)(**arguments)
    variants = _SHARED_EVALUATIONS.setdefault(function, {})
    if key not in variants:
        # A strong reference here would keep the kernel, the weak key, alive.
        kernel = weakref.ref(function)

        def evaluate(*dynamic: PytreeValue) -> EvaluatedQAndF:
            supplied = iter(dynamic)
            return _evaluate(function=cast("CanonicalActionKernel", kernel()))(
                **jax.tree.unflatten(
                    treedef,
                    [
                        next(supplied) if is_traced else leaf
                        for leaf, is_traced in zip(fixed, traced, strict=True)
                    ],
                )
            )

        variants[key] = jax.jit(evaluate, inline=True)
    return variants[key](
        *(leaf for leaf, is_traced in zip(leaves, traced, strict=True) if is_traced)
    )


def _evaluate(*, function: CanonicalActionKernel) -> Callable[..., EvaluatedQAndF]:
    """The kernel's evaluation, called with the query's flat argument mapping."""
    return cast("Callable[..., EvaluatedQAndF]", function.evaluated)


def _closed_over_key(*, leaf: PytreeValue | None) -> Hashable:
    """Key a concrete leaf by type and bits: `1 == 1.0` and `0.0 == -0.0`."""
    return (type(leaf), leaf.hex() if isinstance(leaf, float) else leaf)


def _check_query_answer(*, name: str, node: QueryNode, answer: PolicyAnswer) -> None:
    """Check mandatory numerical status before this producer's reached landings."""
    if not bool(answer[3]):
        raise ExecutionPlanningError(
            f"Policy query {name!r} has unresolved numerical status."
        )
    fail_on_query_landings(
        totals=answer[4],
        checks=tuple(
            query_landing_check(
                check=check, source=node.read.target, period=node.read.period
            )
            for check in _query_landing_checks(node=node)
        ),
        phase="solve",
    )


@dataclass(frozen=True, kw_only=True)
class _PolicyQueryKernel:
    """One real occupant's transient query closure and durable Full publication."""

    programs: Mapping[str, CoreProgram]
    builders: Mapping[str, _QueryArguments]
    definitions: Mapping[str, QueryNode]
    root: str
    chunk_size: int
    recompute: bool
    root_context: SolverBuildContext
    period: int
    fixed_flat_params: FlatParams = MappingProxyType({})

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Publish the sole native declaration authority."""
        return self.programs

    def with_recomputation(self) -> Self:
        """Change only the closed dispatch schedule, retaining producer functions."""
        schedule = _closed_query_schedule(
            definitions=self.definitions, root=self.root, recompute=True
        )
        return replace(
            self,
            recompute=True,
            programs=MappingProxyType(
                {
                    name: replace(
                        program,
                        requirements=replace(
                            program.requirements, internal_schedule=schedule
                        ),
                    )
                    for name, program in self.programs.items()
                }
            ),
        )

    def narrow_workspace(self) -> PeriodKernel | None:
        """Try exact recomputation before rebuilding a smaller common chunk."""
        if not self.recompute:
            return self.with_recomputation()
        if self.chunk_size == 1:
            return None
        candidate = build_policy_query_kernel(
            root_context=self.root_context,
            period=self.period,
            definitions=self.definitions,
            root=self.root,
            chunk_size=max(1, self.chunk_size // 2),
            recompute=True,
        )
        return candidate.with_fixed_params(fixed_flat_params=self.fixed_flat_params)

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> Self:
        """Expose each target's fixed operands through its canonical argument pool."""
        frozen = MappingProxyType(
            {
                name: MappingProxyType(dict(values))
                for name, values in fixed_flat_params.items()
            }
        )
        builders = MappingProxyType(
            {
                name: replace(builder, fixed_flat_params=frozen)
                for name, builder in self.builders.items()
            }
        )
        return replace(
            self,
            fixed_flat_params=frozen,
            builders=builders,
            programs=MappingProxyType(
                {
                    name: replace(program, argument_builder=builders[name])
                    if name in builders
                    else program
                    for name, program in self.programs.items()
                }
            ),
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        """Dispatch compiled producers, check status, then commit actual readers."""
        del logger
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            same_period_regime_to_V_arr=same_period_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        size = math.prod(value.size for value in state_action_space.states.values())
        owner = PendingSolveWork()
        buffers: dict[tuple[str, int], PolicyAnswer] = {}
        try:
            bank = cast("FloatND", compiled_cores["_initialize"]())
            owner.record(outputs=bank, devices=frozenset(bank.devices()))
            for start in range(0, size, self.chunk_size):
                closed_schedule = self.programs[
                    self.root
                ].requirements.internal_schedule
                schedule = closed_schedule[1:-1]
                ledger = PlannedInputLiveness(
                    dispatch_accesses={
                        step.identity: tuple(
                            (
                                start,
                                producer,
                                generation,
                                self.programs[step.name]
                                .requirements.internal_inputs[operand]
                                .label,
                            )
                            for operand, (producer, generation) in step.inputs.items()
                        )
                        for step in closed_schedule
                    }
                )
                devices = frozenset(bank.devices())
                owner.before(devices=devices)
                frame = compiled_cores["_source"](
                    states=state_action_space.states, start=jnp.int32(start)
                )
                owner.record(outputs=frame, devices=devices)
                ledger.commit_successful_dispatch(dispatch=("_source", 0))
                states, live = cast(
                    "tuple[Mapping[str, FloatND | IntND], BoolND]", frame
                )
                for step in schedule:
                    name = step.name
                    arguments = dict(self.builders[name](context))
                    arguments[_SOURCE], arguments[_LIVE] = states, live
                    arguments.update(
                        {
                            operand: buffers[producer]
                            for operand, producer in step.inputs.items()
                            if operand not in {_SOURCE, _LIVE}
                        }
                    )
                    devices = frozenset(
                        device
                        for value in jax.tree.leaves(arguments)
                        if isinstance(value, jax.Array)
                        for device in value.devices()
                    )
                    owner.before(devices=devices)
                    answer = cast("PolicyAnswer", compiled_cores[name](**arguments))
                    owner.record(outputs=answer, devices=devices)
                    _check_query_answer(
                        name=name, node=self.definitions[name], answer=answer
                    )
                    buffers[step.identity] = answer
                    # Completion witnesses cannot retain an owner past its census.
                    owner.before(devices=devices)
                    ledger.commit_successful_dispatch(dispatch=step.identity)
                    for key in tuple(buffers):
                        if ledger.is_release_eligible(artifact=(start, *key, "answer")):
                            buffers.pop(key)
                    del answer, arguments
                answer = buffers[(self.root, 0)]
                owner.before(devices=frozenset(bank.devices()))
                bank = cast(
                    "FloatND",
                    compiled_cores["_publish"](
                        bank=bank,
                        start=jnp.int32(start),
                        answer=answer,
                    ),
                )
                owner.record(outputs=bank, devices=frozenset(bank.devices()))
                ledger.commit_successful_dispatch(dispatch=("_publish", 0))
                ledger.assert_solve_complete()
                owner.before(devices=frozenset(bank.devices()))
                buffers.clear()
                del answer, states, live, frame
            owner.before(devices=frozenset(bank.devices()))
            value = cast("FloatND", compiled_cores["_finish"](bank=bank))
            owner.record(outputs=value, devices=frozenset(value.devices()))
            return KernelOutput(value=value)
        finally:
            owner.close()
            buffers.clear()


def _closed_query_schedule(
    *, definitions: Mapping[str, QueryNode], root: str, recompute: bool
) -> tuple[InternalDispatch, ...]:
    """One immutable order shared by admission and the actual native driver."""
    steps = _query_schedule(definitions=definitions, root=root, recompute=recompute)
    return (
        InternalDispatch(name="_source", generation=0),
        *(
            replace(
                step,
                inputs={**step.inputs, _SOURCE: ("_source", 0), _LIVE: ("_source", 0)},
            )
            for step in steps
        ),
        InternalDispatch(
            name="_publish", generation=0, inputs={"answer": steps[-1].identity}
        ),
    )


def _query_schedule(
    *,
    definitions: Mapping[str, QueryNode],
    root: str,
    recompute: bool,
) -> tuple[InternalDispatch, ...]:
    """Retain shared answers or replay the same producer for each actual reader."""
    if not recompute:
        return tuple(
            InternalDispatch(
                name=name,
                generation=0,
                inputs=MappingProxyType(
                    {key: (producer, 0) for key, producer in node.inputs.items()}
                ),
            )
            for name, node in definitions.items()
        )
    result: list[InternalDispatch] = []
    generations = {root: 1}
    pending: list[tuple[str, int, Mapping[str, tuple[str, int]] | None]] = [
        (root, 0, None)
    ]
    while pending:
        name, generation, inputs = pending.pop()
        if inputs is not None:
            result.append(
                InternalDispatch(name=name, generation=generation, inputs=inputs)
            )
            continue
        node = definitions[name]
        upstream = {}
        for producer in dict.fromkeys(node.inputs.values()):
            child_generation = generations.get(producer, 0)
            generations[producer] = child_generation + 1
            upstream[producer] = (producer, child_generation)
        pending.append(
            (
                name,
                generation,
                MappingProxyType(
                    {key: upstream[producer] for key, producer in node.inputs.items()}
                ),
            )
        )
        pending.extend(
            (producer, child_generation, None)
            for producer, child_generation in reversed(tuple(upstream.values()))
        )
    return tuple(result)
