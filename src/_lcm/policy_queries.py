"""Private typed current-period policy reads and their native root closures."""

import functools
import inspect
import logging
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalDispatch,
    InternalInputRef,
    InternalOutputSpec,
    core_program_graph,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import collective_argmax_and_readout
from _lcm.regime_building.max_Q_over_a import _evaluate_pareto_weights
from _lcm.regime_building.Q_and_F import runtime_bound_specs
from _lcm.solution.contract import ContinuationPayload, PeriodKernel, SolverBuildContext
from _lcm.solution.grid_search import GridSearch
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue, QAndFKwargs, ShapeDtypePytree
from _lcm.utils.dispatchers import productmap
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, FloatND, IntND, RegimeName

_SOURCE = "_lcm_query_source"
_LIVE = "_lcm_query_live"
_PROJECTION = "_lcm_query_projection"
_FLAT_OPERAND_PATH_LENGTH = 2


class QueryKind(StrEnum):
    """Distinct current-period action problems, never durable value lookups."""

    RAW = "Raw"
    PARTICIPATING = "Participating"
    FULL = "Full"


@dataclass(frozen=True, kw_only=True)
class QueryRead:
    """One mapped policy context, with target and caller information separated."""

    kind: QueryKind
    """Action problem requested at this node."""
    target: RegimeName
    """Actual model regime, never a synthetic scheduler name."""
    phase: Literal["solve", "simulate"]
    """Physical-law and projection phase."""
    projection: Mapping[str, Callable[..., FloatND | IntND]]
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
    """Canonical private kept-set extension; never replaces ordinary feasibility."""

    def __post_init__(self) -> None:
        """Freeze edges and require the actual canonical target authority."""
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        if self.read.target != self.context.regime_name:
            raise ValueError("A query's target differs from its canonical Q/F owner.")
        if self.read.kind is QueryKind.RAW and self.participation is not None:
            raise ValueError("Raw queries cannot impose dissolution participation.")


# Values at one common selected action, stable action id, support, integrity.
type PolicyAnswer = tuple[FloatND, IntND, BoolND, BoolND]


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
            function=_InitializeBank(shape=bank_shape, dtype=dtype),
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
        function=_SourceChunk(width=chunk_size),
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
        baseline = core_program_graph(
            kernel=GridSearch()
            .build_period_kernels(context=node.context)
            .period_kernels[period]
        )["main"]
        builder = _QueryArguments(
            node=node,
            base=baseline.argument_builder,
        )
        builders[name] = builder
        reads = tuple(
            replace(
                read,
                source=replace(
                    read.source, source_regime=root_context.regime_name, core_key=name
                ),
            )
            for read in baseline.requirements.value_reads
        )
        shape = (chunk_size,) + (
            ()
            if node.context.stakeholders is None
            else (len(node.context.stakeholders),)
        )
        programs[name] = CoreProgram(
            name=name,
            function=_QueryCore(node=node, period=period),
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
                StateAxesLeading(state_names=(), shape=(chunk_size,), dtype=jnp.int32),
                StateAxesLeading(state_names=(), shape=(chunk_size,), dtype=jnp.bool_),
                StateAxesLeading(state_names=(), shape=(), dtype=jnp.bool_),
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
        function=_FinishBank(shape=value_shape, size=size),
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


def _admit_node(*, node: QueryNode, period: int) -> None:
    """Refuse an unrepresented action/status or information intersection."""
    context = node.context
    _admit_query_solver(context=context, period=period)
    if node.read.phase != "solve" or node.read.period != period:
        raise UnsupportedOperationError(
            "A policy query needs its exact solve phase/period."
        )
    if period not in context.Q_and_F_functions:
        raise ValueError("The target action problem is inactive at this period.")
    if set(node.read.projection) != set(context.state_action_space.state_names):
        raise ValueError("A query needs a complete canonical target-state projection.")
    if (
        context.has_taste_shocks
        or node.read.physical_inputs
        or context.current_iid_plans.get(period) is not None
    ):
        raise UnsupportedOperationError(
            "Native query node-information preparation is not installed."
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
    if context.same_period_ref_regimes or context.edge_reference_regimes:
        raise UnsupportedOperationError(
            "A query cannot price another root's stored Full value."
        )
    if runtime_bound_specs(function=context.Q_and_F_functions[period]):
        raise UnsupportedOperationError(
            "Query bound integrity must have a native status adapter."
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


@dataclass(frozen=True, kw_only=True)
class _QueryArguments:
    """Canonical target arguments plus one fixed source chunk prototype."""

    node: QueryNode
    base: Callable[..., Mapping[str, PytreeValue | ShapeDtypePytree]]
    fixed_flat_params: FlatParams = MappingProxyType({})

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        """Build the exact prototype shared by lowering and every real chunk."""
        source = cast("StateActionSpace", context.state_action_space)
        context = replace(
            context,
            flat_params=MappingProxyType(
                {
                    name: MappingProxyType(
                        {
                            **self.fixed_flat_params.get(name, {}),
                            **context.flat_params.get(name, {}),
                        }
                    )
                    for name in self.fixed_flat_params.keys()
                    | context.flat_params.keys()
                }
            ),
        )
        arguments = dict(
            self.base(
                replace(
                    context, state_action_space=self.node.context.state_action_space
                )
            )
        )
        # Source prototypes come only from the declared native producer.
        operands = {}
        for name, path in self.node.read.operand_paths.items():
            if name in source.state_names or name in {"period", "age"}:
                continue
            if len(path) != _FLAT_OPERAND_PATH_LENGTH or not all(
                isinstance(step, str) for step in path
            ):
                raise ValueError("Query operands need canonical flat namespace paths.")
            operands[name] = context.flat_params[cast("str", path[0])][
                cast("str", path[1])
            ]
        arguments[_PROJECTION] = operands
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True, eq=False)
class _QueryCore:
    """Map canonical target action selection over a fixed source chunk."""

    node: QueryNode
    period: int

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
        cell = functools.partial(
            _query_cell,
            node=self.node,
            period=self.period,
            fixed=fixed,
            projection_operands=projection_operands,
        )
        answer = jax.vmap(
            cell, in_axes=(0, 0, dict.fromkeys(dependencies, (0, 0, 0, None)))
        )(live, source, dependencies)
        return (*answer[:3], jnp.all(answer[3]))


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
    }
    mapped = {
        name: helper(
            **{key: available[key] for key in inspect.signature(helper).parameters}
        )
        for name, helper in node.read.projection.items()
    }
    if any(value.ndim != 0 for value in mapped.values()):
        raise ValueError("A mapped query state must be scalar per source cell.")
    arguments = {**fixed, **mapped}
    context = node.context
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
        "Callable[..., tuple[FloatND, BoolND, BoolND]]",
        productmap(
            func=action_point,
            variables=context.state_action_space.action_names,
            batch_sizes=dict.fromkeys(context.state_action_space.action_names, 0),
        ),
    )
    q, feasible, valid = evaluate(
        **{name: arguments[name] for name in context.state_action_space.action_names}
    )
    axes = tuple(range(feasible.ndim))
    support = jnp.any(feasible)
    if context.stakeholders is not None:
        action, values, _ = collective_argmax_and_readout(
            stakeholder_Q={
                name: q[..., index] for index, name in enumerate(context.stakeholders)
            },
            feasibility=feasible,
            weights=_evaluate_pareto_weights(
                pareto_weights=context.pareto_weights,
                states_actions_params=cast("QAndFKwargs", arguments),
            ),
            action_axes=axes,
        )
        value = jnp.stack(tuple(values[name] for name in context.stakeholders))
    else:
        action, value = argmax_and_max(a=q, where=feasible, initial=-jnp.inf)
    valid = (
        jnp.all(valid)
        & jnp.all(
            jnp.stack(
                tuple(
                    jnp.any(mapped[name] == context.state_action_space.states[name])
                    for name, grid in context.grids.items()
                    if name in mapped and isinstance(grid, DiscreteGrid)
                )
                or (jnp.asarray(a=True),)
            )
        )
        & jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in mapped.values())
                or (jnp.asarray(a=True),)
            )
        )
    )
    return (
        value,
        jnp.where(support, action, -1).astype(jnp.int32),
        live & support,
        ~live | valid,
    )


def _query_action(
    *,
    node: QueryNode,
    period: int,
    arguments: Mapping[str, PytreeValue],
    dependencies: Mapping[str, PolicyAnswer],
    **actions: PytreeValue,
) -> tuple[FloatND, BoolND, BoolND]:
    """Preserve the canonical Q/F arrays and add only declared participation."""
    function = cast(
        "Callable[..., tuple[FloatND, BoolND]]",
        node.context.Q_and_F_functions[period],
    )
    inputs = {**arguments, **actions}
    q, ordinary = function(
        **{name: inputs[name] for name in inspect.signature(function).parameters}
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
    valid = ~jnp.any(ordinary & jnp.any(jnp.isnan(q)))
    if node.participation is None:
        return q, ordinary, valid
    keep = node.participation(q=q, **dependencies)
    if keep.dtype != jnp.bool_:
        raise TypeError("Query participation must be boolean.")
    if keep.ndim != 0:
        raise ValueError("Query participation must be scalar per action.")
    return q, ordinary & keep, valid


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
                    if not bool(answer[3]):
                        raise ExecutionPlanningError(
                            f"Policy query {name!r} has unresolved numerical status."
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
