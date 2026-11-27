"""Private typed current-period policy reads and their native root closures."""

import functools
import inspect
import logging
import math
import weakref
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature
from jax.core import Tracer

from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    CurrentIIDDrawFrame,
    CurrentIIDPlan,
)
from _lcm.dtypes import Code, as_canonical_float, as_code
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalDispatch,
    InternalInputRef,
    ValueRead,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.grids import Grid
from _lcm.off_grid import ProjectedGridRule, count_projected_off_grid
from _lcm.output_roles import OutputRole, target_role
from _lcm.params.edges import EDGES, flat_namespaces, regime_kernel_params
from _lcm.regime_building.collective import (
    collective_argmax_and_readout,
    decider_choice,
    decider_expected_values,
)
from _lcm.regime_building.decoupled import (
    household_argmax_and_readout,
    smoothed_logsum,
    smoothed_profiles,
)
from _lcm.regime_building.max_Q_over_a import (
    _evaluate_pareto_weights,
    _stacked_pareto_weights,
)
from _lcm.regime_building.Q_and_F import (
    CanonicalActionKernel,
    EvaluatedQAndF,
    LandingCheck,
    QAndFParts,
    runtime_bound_specs,
)
from _lcm.regime_building.V import get_V_interpolator
from _lcm.solution.contract import ContinuationPayload, PeriodKernel, SolverBuildContext
from _lcm.solution.grid_search import (
    GridSearch,
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
from lcm.grids import DiscreteGrid
from lcm.shocks import MonteCarlo, QuasiMonteCarlo, ShockGroupName
from lcm.solver_api import KernelOutput
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    LoopIndex,
    RegimeName,
    StateName,
    UserFunction,
)

_SOURCE = "_lcm_query_source"
_LIVE = "_lcm_query_live"
_PROJECTION = "_lcm_query_projection"
_PHYSICAL_CONTEXTS = "_lcm_query_physical_contexts"
_INFORMATION = "_lcm_query_information"
_DRAWS = "_lcm_query_draws"
_ACTING = "_lcm_query_acting"
_QUERY_ARGUMENTS = "query_arguments"
_EVALUATED = "_lcm_query_evaluated"
_OUTSIDE = "_lcm_query_outside"
# Prefix of the program that tabulates a destination for the reader's stored reads.
OUTSIDE_PROGRAM = "_outside:"
# Support rows evaluated together per step, inside every lane of the cell batch.
# A step evaluates chunk-times-row-batch lanes, and its workspace scales with that
# product. A planned kernel holds the product at `_STEP_LANES`, whatever its
# chunk, so a narrower planned chunk widens its rows and keeps the step count, and
# with it the launch count, from growing. Each narrowing by admission halves the
# product: it halves the chunk down to one source cell and, from there, the rows
# down to `_ROW_BATCH`, where the kernel refuses to narrow further.
# ponytail: a fixed lane target. Admission halves a step that outgrows the budget,
# at one extra lowering each; derive the target from the budget if those retries
# cost more compile time than they save.
_ROW_BATCH = 64
_STEP_LANES = 1048576
_FLAT_OPERAND_PATH_LENGTH = 2


class QueryKind(StrEnum):
    """The action problem a query reads.

    - `RAW` ⇒ the target's problem without its composition action
    - `PARTICIPATING` ⇒ that problem with the target's composition action applied as a
      rule for staying, without leaving
    - `FULL` ⇒ the target's problem with its composition action
    """

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
    """Destination member role in the producing projection context."""
    law_identity: tuple[Hashable, ...] = ()
    """Canonical law, node and information-representation declaration identities."""

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
        object.__setattr__(self, "law_identity", tuple(self.law_identity))
        if not self.target or isinstance(self.period, bool) or self.period < 0:
            raise ValueError("A query needs an actual target and nonnegative period.")

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
    """Expand demanded roots and reject typed cycles and reads deeper than one step.

    A query never reads a Full publication. A Participating query, a rule for staying
    applied within one step, reads Raw queries only: values without the composition
    action, never another rule for staying.
    """
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
                reads[name].kind is QueryKind.PARTICIPATING
                and reads[upstream].kind is QueryKind.PARTICIPATING
            ):
                raise ValueError(
                    f"Participating query {name!r} cannot read Participating query "
                    f"{upstream!r}: a rule for staying reads only the values of "
                    "its targets without the composition action, so two such "
                    "rules cannot chain within one step."
                )
        graph[name] = dependencies[name]
        pending.extend(reversed(dependencies[name]))
    try:
        return tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError(f"Policy query cycle: {error.args[1]!r}") from error


@dataclass(frozen=True, kw_only=True)
class OutsideValueRead:
    """Proof, made when the model is built, that an outside read lands on stored nodes.

    Every state the read projects is a point of the destination's grid, and each
    destination shock group kept as an axis is fixed by an ancestor's frame on the
    destination's own nodes. The destination's other groups are integrated, since no
    ancestor observes them.
    """

    label: str
    """Name of the destination's array, unique per destination and kept groups."""
    groups: tuple[ShockGroupName, ...]
    """The destination's plan groups kept as axes, in plan order."""
    origins: tuple[str, ...]
    """For each kept group, the query whose frame enumerates the group's nodes."""


@dataclass(frozen=True, kw_only=True)
class QueryNode:
    """Canonical action problem and its explicitly named transient readers."""

    read: QueryRead
    """Mapped context and kind of this action problem.

    A destination is read as Raw, the destination's problem without its own
    composition action, because an agent decides on its composition at most once per
    period. A consent target is read as
    Participating: the formed couple applies its own rule for staying, and the
    targets of that rule are read as Raw.
    """
    context: SolverBuildContext
    """Canonical target Q/F, grids, parameters and continuation declarations."""
    inputs: Mapping[str, str] = MappingProxyType({})
    """Predicate operand name to demanded producer key."""
    participation: Callable[..., BoolND] | None = None
    """Canonical kept-set extension; never replaces ordinary feasibility."""
    decision: Callable[..., PolicyAnswer] | None = None
    """Shared scalar Full/Participating rule after canonical stay selection."""
    state_updates: Mapping[StateName, Callable[..., IntND | FloatND]] = (
        MappingProxyType({})
    )
    """Current coordinates selected together with the root's retained policy."""
    gated_channels: Mapping[RegimeName, tuple[int, ...]] = MappingProxyType({})
    """Open-branch value components, stacked inside the admitted query core."""
    available: Callable[..., BoolND] | None = None
    """Caller-context guard evaluated before a demanded benchmark."""
    composition_action_targets: Mapping[int, Mapping[str, str]] = MappingProxyType({})
    """Stable decision ID to source member and its exact destination query key."""
    composition_action_adjustments: Mapping[
        int, Mapping[str, Callable[..., PytreeValue]]
    ] = MappingProxyType({})
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
    own_raw: str | None = None
    """Explicit own-outside producer of a singleton Consent comparison."""
    sample: tuple[MonteCarlo | QuasiMonteCarlo, tuple[int, ...]] | None = None
    """Draw rule of an occupant's rows and the plan positions kept as a product.

    The context's plan indices then span the kept groups alone; every other
    group and every observation axis is drawn once per sample.
    """
    composition_action_availability: Callable[..., PytreeValue] | None = None
    """Availability predicate of the occupant's composition action, if it has one."""
    draw_condition: int | None = None
    """Drawn plan position whose draws fall only where the decision is available.

    The mass where no decision is available then enters once, through one more row per
    kept row, at the value of staying.
    """
    available_nodes: tuple[int, tuple[bool, ...]] | None = None
    """Enumerated plan position whose nodes alone decide if the decision is available.

    It also holds whether it is available at each of them.

    The plan's rows then cross that group's available nodes and its first
    unavailable one with every node of the other groups. A row of that one node
    weights its value by each unavailable node's factors of the group in turn,
    and its landings count once per such node present.
    """
    outside_value: OutsideValueRead | None = None
    """Proof that this outside read lands on the destination's stored nodes.

    None when the build cannot prove it; the read keeps its Raw identity either
    way, since it is the same value without the composition action.
    """
    grid_rules: tuple[ProjectedGridRule, ...] = ()
    """Destination-grid rules of the states that the decision projects into this read.

    The projection already caps each `clip_state` state; each `raise` state adds
    one reached-landing check to the read's own.
    """
    landing: int | None = None
    """The target's occurrence later in the period, whose stored value this read
    takes; `None` for a read of the target's problem at the read's step.

    A decision on the composition enters a regime of a later stage of its period, which
    decides there:
    its value at its occurrence is interpolated at the projected states.
    """

    def __post_init__(self) -> None:
        """Freeze edges and require the actual canonical target authority."""
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(
            self, "state_updates", MappingProxyType(dict(self.state_updates))
        )
        object.__setattr__(
            self, "gated_channels", MappingProxyType(dict(self.gated_channels))
        )
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
            "composition_action_targets",
            MappingProxyType(
                {
                    identifier: MappingProxyType(dict(targets))
                    for identifier, targets in self.composition_action_targets.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "composition_action_adjustments",
            MappingProxyType(
                {
                    identifier: MappingProxyType(dict(adjustments))
                    for identifier, adjustments in (
                        self.composition_action_adjustments.items()
                    )
                }
            ),
        )
        if self.read.target != self.context.regime_name:
            raise ValueError("A query's target differs from its canonical Q/F owner.")
        if self.read.kind is QueryKind.RAW and self.participation is not None:
            raise ValueError("Raw queries cannot impose dissolution participation.")

    def __repr__(self) -> str:
        """Name the read only; `jax.jit` stringifies callables holding a node."""
        return (
            f"QueryNode(kind={self.read.kind.value!r}, target={self.read.target!r}, "
            f"period={self.read.period}, role={self.read.role!r})"
        )

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the answer owner's tuple shape, independent of physical laws."""
        return (len(self.context.members),)


def gated_query_targets(
    *, context: SolverBuildContext, period: int
) -> tuple[RegimeName, ...]:
    """Select this query's future gates; terminal dates have no continuation."""
    if period >= context.solution_reachability.n_steps - 1:
        return ()
    return tuple(
        target
        for target in context.solution_reachability.targets(
            step=period, source=context.regime_name
        )
        if target in context.edge_target_regimes
    )


def _node_inputs(
    *, node: QueryNode, period: int
) -> tuple[CoreArgumentBuilder, tuple[ValueRead, ...]]:
    """Return a query's argument builder and the continuation values it reads.

    A stored outside read takes its value from its destination's table, so it
    reads only the clock and its law's operands, and no continuation.
    """
    if node.outside_value is not None:
        return _OutsideReadArguments(
            period=period, plan=node.context.current_iid_plans.get(period)
        ), ()
    if node.landing is not None:
        target = node.read.target
        return _LandingReadArguments(period=period, target=target), (
            ValueRead(
                target=ValueArtifactAddress(
                    kind=ValueArtifactKind.REGIME_VALUE,
                    period=node.landing,
                    regime=target,
                ),
                source=ValueConsumerAddress(
                    source_period=period,
                    source_regime=target,
                    core_key="main",
                    channel=ValueInputChannel.NEXT_REGIME_VALUE,
                    path=(target,),
                ),
            ),
        )
    builder, reads = grid_search_query_inputs(context=node.context, period=period)
    if not node.gated_channels:
        return builder, reads
    return replace(builder, edge_target_regimes=()), tuple(
        replace(
            read,
            target=ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=read.target.period,
                regime=cast("RegimeName", read.target.target_regime),
            ),
        )
        if read.target.kind is ValueArtifactKind.GATED_CONTINUATION
        else read
        for read in reads
    )


@dataclass(frozen=True, kw_only=True)
class _LandingReadArguments:
    """A landing read's operands: the clock and its target's stored value.

    Also the target's actions and parameters, at which its feasibility at the
    landing state is evaluated.
    """

    period: int
    target: RegimeName

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        if context.ages is None:
            raise ValueError("A landing read needs its actual economic age.")
        arguments: dict[str, PytreeValue | ShapeDtypePytree] = {
            **dict(cast("StateActionSpace", context.state_action_space).actions),
            **dict(
                regime_kernel_params(
                    cast("FlatParams", context.flat_params), regime_name=self.target
                )
            ),
            "period": jnp.int32(self.period),
            "age": context.ages.values[self.period],
        }
        arguments["next_regime_to_V_arr"] = cast(
            "PytreeValue",
            MappingProxyType({self.target: context.next_regime_to_V_arr[self.target]}),
        )
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True)
class _OutsideReadArguments:
    """A stored outside read's operands: the clock and its law's current draws.

    The read gathers its value from the destination's table, so it takes no
    target action bank and no continuation.
    """

    period: int
    plan: CurrentIIDPlan | None

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        if context.ages is None:
            raise ValueError("A stored outside read needs its actual economic age.")
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


# Values at one common action, stable ID, support, integrity, reached-landing counts.
type PolicyAnswer = tuple[FloatND, IntND, BoolND, BoolND, IntND]

# A destination's tabulated values and support, read by its stored outside reads.
type OutsideTable = tuple[FloatND, BoolND]

# Q, feasibility, integrity and landing counts over the action grid, and the
# objective's Q of a regime with a separate objective (`None` without one).
type StayTable = tuple[FloatND, BoolND, BoolND, IntND, FloatND | None]


def _query_landing_checks(*, node: QueryNode) -> tuple[LandingCheck, ...]:
    """The read's reached-landing checks: its problem's, then its projected states'.

    A projected state checks against its destination grid on behalf of the composition
    action that projects it.
    """
    return (
        *_problem_landing_checks(node=node),
        *(
            (rule.source, node.read.period, rule.target, rule.state)
            for rule in node.grid_rules
            if rule.off_grid == "raise"
        ),
    )


def _problem_landing_checks(*, node: QueryNode) -> tuple[LandingCheck, ...]:
    if node.landing is not None:
        return ()
    return getattr(
        node.context.Q_and_F_functions[node.read.period], "landing_checks", ()
    )


def _evaluated_query_landings(*, node: QueryNode, evaluated: EvaluatedQAndF) -> IntND:
    """Require the original canonical count vector in declared check order."""
    checks = _problem_landing_checks(node=node)
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
    """The same bound decision at one original agent or mapped destination.

    Simulation owns native argument placement, inventory and row custody. Every
    query's current parameters and canonical continuation enter explicitly.
    """

    definitions: Mapping[str, QueryNode]
    """Phase-resolved canonical query nodes, including complete mapped contexts."""
    root: str
    """The node that acts.

    - Full for an original occupant's choice.
    - For the once-only action of a member who leaves, the kind its decision was priced
      at: Participating after formation by consent, Raw after any other
      composition action.
    """
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
        """Use the same canonical transfer at the immutable original source."""
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
        """Bind only clock and transfer operands, without policy value inputs."""
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

    @functools.cached_property
    def _node_value_reads(self) -> tuple[tuple[str, ValueRead], ...]:
        """Each node's continuation reads, walked once per graph for every core key.

        Nested graphs share query instances, so the walk is linear in the distinct
        graphs rather than in the paths through them.
        """
        return tuple(
            (name, read)
            for name, node in self.definitions.items()
            for read in _node_inputs(node=node, period=node.read.period)[1]
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
            for name, read in self._node_value_reads
        )

    def __call__(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> PolicyAnswer:
        """Return the same canonical answer without retaining current coordinates."""
        answer, _ = self.evaluate(states=states, arguments=arguments, frame=frame)
        return answer

    def evaluate(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        arguments: Mapping[str, Mapping[str, PytreeValue]],
        frame: CurrentIIDDrawFrame | None = None,
    ) -> tuple[PolicyAnswer, Mapping[StateName, FloatND | IntND]]:
        """Resolve only demanded typed reads and invoke their shared scalar rule."""
        order = demanded_query_order(
            reads={name: node.read for name, node in self.definitions.items()},
            dependencies={
                name: tuple(node.inputs.values())
                for name, node in self.definitions.items()
            },
            roots=(self.root,),
        )
        prepared = {
            name: with_contracted_tables(node=self.definitions[name], arguments=fixed)
            for name, fixed in arguments.items()
        }
        if any(prepared[name] is not fixed for name, fixed in arguments.items()):
            arguments = prepared
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
                and node.read.kind in {QueryKind.RAW, QueryKind.PARTICIPATING},
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
        root = self.definitions[self.root]
        updates = {}
        if root.state_updates:
            fixed = arguments[self.root]
            physical = {} if frame is None else frame.physical
            operands, _ = query_arguments(
                guard=jnp.asarray(a=True),
                source=states,
                node=root,
                fixed={**fixed, **physical, _PHYSICAL_CONTEXTS: {self.root: physical}},
                projection_operands=cast(
                    "Mapping[str, PytreeValue]", fixed[_PROJECTION]
                ),
            )
            operands.update(
                {name: answers[producer] for name, producer in root.inputs.items()}
            )
            updates = {
                name: helper(
                    **{
                        key: operands[key]
                        for key in inspect.signature(helper).parameters
                    }
                )
                for name, helper in root.state_updates.items()
            }
        return (
            (
                answer[0],
                answer[1][0] if answer[1].ndim else answer[1],
                answer[2],
                answer[3],
                jnp.concatenate(tuple(answers[name][4] for name in order)),
            ),
            updates,
        )


def build_policy_query_kernel(
    *,
    root_context: SolverBuildContext,
    period: int,
    definitions: Mapping[str, QueryNode],
    root: str,
    chunk_size: int,
    outside_tables: Mapping[str, tuple[Mapping[str, QueryNode], str]] = (
        MappingProxyType({})
    ),
    halvings: int = 0,
) -> _PolicyQueryKernel:
    """Expand one genuine occupant root into a common-chunk native query loop.

    `outside_tables` maps the label of each stored outside read to its
    destination's bound Raw graph and that graph's root. The kernel tabulates each
    destination its stored reads take, once, before the chunk loop that gathers
    from the tables. A step evaluates `_STEP_LANES` halved `halvings` times.
    """
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("A native query chunk must be positive.")
    _admit_query_solver(context=root_context, period=period)
    root_space = root_context.state_action_space
    size = math.prod(value.size for value in root_space.states.values())
    chunk_size = min(chunk_size, size)
    row_batch = _row_batch(chunk_size=chunk_size, halvings=halvings)
    bank_size = _get_padded_query_bank_size(size=size, chunk_size=chunk_size)
    member_shape = (len(root_context.members),)
    value_shape = (
        *(value.size for value in root_space.states.values()),
        *member_shape,
    )
    bank_shape = (bank_size, *member_shape)
    dtype = jnp.asarray(0.0).dtype
    host_buffers = MappingProxyType(
        {"query_banks": jax.ShapeDtypeStruct((2, *bank_shape), dtype)}
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
    builders: dict[str, _QueryArguments] = {}
    cores: dict[str, _QueryCore] = {}
    reads: list[ValueRead] = []
    from _lcm.query_information import pinned_support  # noqa: PLC0415

    for name in order:
        node = definitions[name]
        _admit_node(node=node, period=period)
        pinned_support(name=name, definitions=definitions)
        base, value_reads = _node_inputs(node=node, period=period)
        builders[name] = _QueryArguments(
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
        cores[name] = _QueryCore(
            node=node,
            period=period,
            name=name,
            definitions=definitions,
            row_batch=row_batch,
        )
        reads.extend(
            query_value_reads(
                key=name,
                reads=value_reads,
                regime=root_context.regime_name,
                core_key="_chunks",
            )
        )
    bank_arguments = _BankArguments(
        regime=root_context.regime_name, shape=bank_shape, dtype=dtype
    )
    schedule = _query_schedule(definitions={name: definitions[name] for name in order})
    stored = {
        name: read
        for name in order
        if (read := definitions[name].outside_value) is not None
    }
    tables = _outside_programs(
        reads=stored,
        outside_tables=outside_tables,
        reader=root_context.regime_name,
        width=chunk_size,
        row_batch=row_batch,
    )
    # The chunk loop takes each table's values and support under these names.
    table_arguments = {
        name: (f"_lcm_outside_value_{index}", f"_lcm_outside_support_{index}")
        for index, name in enumerate(tables)
    }
    programs = {
        **tables,
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
        ),
        "_chunks": CoreProgram(
            name="_chunks",
            function=_QueryChunks(
                source=_source_chunk(width=chunk_size),
                cores=MappingProxyType(cores),
                schedule=schedule,
                raw_sources=_own_raw_evaluation_sources(
                    root=root, definitions=definitions, period=period
                ),
                chunks=bank_shape[0] // chunk_size,
                outside=MappingProxyType(
                    {
                        name: table_arguments[OUTSIDE_PROGRAM + read.label]
                        for name, read in stored.items()
                    }
                ),
            ),
            argument_builder=_ChunkArguments(bank=bank_arguments, builders=builders),
            requirements=CoreExecutionRequirements(
                value_reads=tuple(reads),
                internal_inputs=MappingProxyType(
                    {
                        argument: InternalInputRef(producer=name, label=label)
                        for name, arguments in table_arguments.items()
                        for argument, label in zip(
                            arguments, ("value", "support"), strict=True
                        )
                    }
                ),
                host_buffers=host_buffers,
                host_axis_names=("query_cell",),
            ),
            # The chunk and the rows per step are bound by the function, not its
            # arguments: a narrowed kernel can pad to one bank shape and must not
            # share one executable.
            compiler_options=(("query_chunk", chunk_size), ("query_rows", row_batch)),
            output_roles=(
                StateAxesLeading(state_names=(), shape=bank_shape, dtype=dtype),
                StateAxesLeading(
                    state_names=(), shape=(len(schedule),), dtype=jnp.bool_
                ),
                tuple(
                    StateAxesLeading(
                        state_names=(),
                        shape=(len(_query_landing_checks(node=cores[step.name].node)),),
                        dtype=jnp.int32,
                    )
                    for step in schedule
                ),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
        ),
        "_finish": CoreProgram(
            name="_finish",
            function=_finish_bank(shape=value_shape, size=size),
            argument_builder=bank_arguments,
            requirements=CoreExecutionRequirements(
                host_buffers=host_buffers, host_axis_names=("query_cell",)
            ),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.PLANNED,
        ),
    }
    return _PolicyQueryKernel(
        programs=MappingProxyType(programs),
        builders=MappingProxyType(builders),
        definitions=MappingProxyType({name: definitions[name] for name in order}),
        root=root,
        chunk_size=chunk_size,
        root_context=root_context,
        period=period,
        outside_tables=outside_tables,
        halvings=halvings,
    )


def _outside_programs(
    *,
    reads: Mapping[str, OutsideValueRead],
    outside_tables: Mapping[str, tuple[Mapping[str, QueryNode], str]],
    reader: RegimeName,
    width: int,
    row_batch: int,
) -> Mapping[str, CoreProgram]:
    """One program per destination table the stored reads take, by program name.

    Reads with the same label take one table.
    """
    from _lcm.outside_values import outside_value_program  # noqa: PLC0415

    programs: dict[str, CoreProgram] = {}
    for read in reads.values():
        name = OUTSIDE_PROGRAM + read.label
        if name in programs:
            continue
        if read.label not in outside_tables:
            raise ValueError(f"Stored outside read {read.label!r} has no table.")
        definitions, key = outside_tables[read.label]
        _admit_node(node=definitions[key], period=definitions[key].read.period)
        programs[name] = outside_value_program(
            read=read,
            definitions=definitions,
            key=key,
            reader=reader,
            width=width,
            row_batch=row_batch,
        )
    return MappingProxyType(programs)


def query_value_reads(
    *, key: str, reads: tuple[ValueRead, ...], regime: RegimeName, core_key: str
) -> tuple[ValueRead, ...]:
    """Address a query's continuation reads inside the program that evaluates it.

    The program takes every query's arguments under `query_arguments`, keyed by
    query, so each read moves below its query's key.
    """
    return tuple(
        replace(
            read,
            source=replace(
                read.source,
                source_regime=regime,
                core_key=core_key,
                argument=_QUERY_ARGUMENTS,
                path=(
                    key,
                    read.source.argument or read.source.channel.value,
                    *read.source.path,
                ),
            ),
        )
        for read in reads
    )


def _own_raw_evaluation_sources(
    *, root: str, definitions: Mapping[str, QueryNode], period: int
) -> Mapping[str, str]:
    """Prove a terminal no-action own-RAW and root use identical canonical inputs."""
    from _lcm.query_information import has_information  # noqa: PLC0415

    node = definitions[root]
    if node.own_raw is None:
        return MappingProxyType({})
    raw = definitions[node.own_raw]
    context = node.context
    if (
        node.read.kind is not QueryKind.FULL
        or raw.read.kind is not QueryKind.RAW
        or raw.context is not context
        or context.koopmans_aggregator is not None
        or context.state_action_space.action_names
        or node.ancestors
        or raw.ancestors != (root,)
        or raw.inputs
        or raw.decision is not None
        or node.own_raw not in node.inputs.values()
        or (raw.read.phase, raw.read.period, raw.read.role)
        != (node.read.phase, node.read.period, node.read.role)
        or raw.read.operand_paths != node.read.operand_paths
        or raw.read.projection.keys() != node.read.projection.keys()
        or any(
            raw.read.projection[name] is not helper
            for name, helper in node.read.projection.items()
        )
        or any(
            candidate.available is not None
            or candidate.read.physical_inputs
            or candidate.read.policy_information
            or candidate.read.price_information
            or has_information(node=candidate, definitions=definitions, period=period)
            for candidate in (node, raw)
        )
    ):
        return MappingProxyType({})
    return MappingProxyType({root: node.own_raw})


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

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        sharding = context.next_regime_to_V_arr[self.regime].sharding
        if isinstance(sharding, jax.NamedSharding):
            sharding = jax.NamedSharding(sharding.mesh, jax.P())
        return MappingProxyType(
            {"bank": jax.ShapeDtypeStruct(self.shape, self.dtype, sharding=sharding)}
        )


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


def _admit_node(*, node: QueryNode, period: int) -> None:
    """Refuse an unrepresented action/status or information intersection.

    A landing read evaluates its target's feasibility in the target's problem at
    the occurrence, which is admitted as a read of the target's problem is.
    """
    context = node.context
    if node.read.phase != "solve" or node.read.period != period:
        raise UnsupportedOperationError(
            "A policy query needs its exact solve phase/period."
        )
    problem = period if node.landing is None else node.landing
    _admit_query_solver(context=context, period=problem)
    if problem not in context.Q_and_F_functions:
        raise ValueError("The target action problem is inactive at this period.")
    _fail_if_projection_incomplete(node=node, reader="A query")
    if context.invariant_bindings or context.action_partitions != 1:
        raise UnsupportedOperationError(
            "This query closure needs an unblocked action problem."
        )
    # A landing read takes its target's stored value, not its continuation.
    if node.landing is None and set(node.gated_channels) != set(
        gated_query_targets(context=context, period=period)
    ):
        raise UnsupportedOperationError(
            "A query needs its gated continuation's explicit raw value channels."
        )
    if runtime_bound_specs(function=context.Q_and_F_functions[problem]):
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


def _get_padded_query_bank_size(*, size: int, chunk_size: int) -> int:
    """Admit int32 source positions and return the exact padded bank extent."""
    bank_size = ((size + chunk_size - 1) // chunk_size) * chunk_size
    maximum_index = (1 << 31) - 1
    # The source live mask also compares int32 positions with the logical size.
    if size > maximum_index or bank_size - 1 > maximum_index:
        raise ExecutionPlanningError(
            f"Policy query source count {size} and padded last position "
            f"{bank_size - 1} must each be at most {maximum_index} for int32 "
            "query indices."
        )
    return bank_size


def _admit_query_solver(*, context: SolverBuildContext, period: int) -> None:
    """Never replace an endogenous-grid policy by an action-grid maximization."""
    declaration = context.user_regimes_at(period=period)[context.regime_name]
    if not isinstance(declaration.solver, GridSearch):
        raise UnsupportedOperationError(
            f"Native policy query for {context.regime_name!r} requires its declared "
            "GridSearch solver; endogenous-grid query reconstruction is separate."
        )


@dataclass(frozen=True, kw_only=True)
class _ChunkArguments:
    """Every query's canonical arguments, the source grids and the bank carry."""

    bank: _BankArguments
    builders: Mapping[str, _QueryArguments]

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        return MappingProxyType({**self.operands(context), **self.bank(context)})

    def operands(
        self, context: CoreBuildContext
    ) -> dict[str, PytreeValue | ShapeDtypePytree]:
        """Source grids and per-query arguments, built once per dispatch."""
        return {
            "states": cast("StateActionSpace", context.state_action_space).states,
            _QUERY_ARGUMENTS: cast(
                "PytreeValue",
                MappingProxyType(
                    {name: builder(context) for name, builder in self.builders.items()}
                ),
            ),
        }


type _ChunkOutput = tuple[FloatND, BoolND, tuple[IntND, ...]]


@dataclass(frozen=True, kw_only=True, eq=False)
class _QueryChunks:
    """Run the retained query schedule over every chunk inside one native loop.

    Each iteration gathers one padded source chunk, evaluates every query in
    schedule order and writes the root answer into the bank. Per scheduled
    query, numerical statuses are AND-ed and reached-landing counts summed
    across chunks, so the host checks them once per period.
    """

    source: _SourceChunk
    cores: Mapping[str, _QueryCore]
    schedule: tuple[InternalDispatch, ...]
    chunks: int
    raw_sources: Mapping[str, str] = MappingProxyType({})  # noqa: PAN003 - internal query keys, not regime/state names
    """Consumer to proven own-RAW producer, confined to one source chunk."""
    outside: Mapping[str, tuple[str, str]] = MappingProxyType({})
    """Stored outside read to the arguments holding its table's values and support."""

    def __call__(
        self,
        *,
        states: Mapping[str, FloatND | IntND],
        bank: FloatND,
        query_arguments: Mapping[str, Mapping[str, PytreeValue]],
        **tables: FloatND | BoolND,
    ) -> _ChunkOutput:
        """Run every chunk; `tables` holds the tables the stored reads take."""
        query_arguments = {
            name: (
                with_contracted_tables(node=core.node, arguments=arguments)
                if (core := self.cores.get(name)) is not None
                else arguments
            )
            for name, arguments in query_arguments.items()
        }
        source_size = math.prod(value.size for value in states.values())
        full_chunks = (
            0 < source_size <= jnp.iinfo(jnp.int32).max
            and source_size == self.chunks * self.source.width
        )

        # keyword-only-exempt: library-callback=jax.lax.fori_loop
        def run(index: LoopIndex, carry: _ChunkOutput) -> _ChunkOutput:
            bank, statuses, totals = carry
            # The loop index is a Python int eagerly and int64 under x64 when
            # traced; the source offset is int32.
            start = jnp.asarray(index * self.source.width, dtype=jnp.int32)
            source, live = self.source(states=states, start=start)
            # The owner proves every scheduled lane is live without int32 overflow.
            if full_chunks:
                live = jnp.ones((self.source.width,), dtype=jnp.bool_)
            answers: dict[tuple[str, int], PolicyAnswer] = {}
            evaluations: dict[str, EvaluatedQAndF] = {}
            for step in self.schedule:
                arguments = dict(query_arguments[step.name])
                arguments[_SOURCE], arguments[_LIVE] = source, live
                arguments.update(
                    {
                        operand: answers[producer]
                        for operand, producer in step.inputs.items()
                    }
                )
                if step.name in self.outside:
                    arguments[_OUTSIDE] = tuple(
                        tables[name] for name in self.outside[step.name]
                    )
                if step.name in self.raw_sources.values():
                    evaluations[step.name] = self.cores[step.name].evaluate(**arguments)
                producer = self.raw_sources.get(step.name, step.name)
                if producer in evaluations:
                    arguments[_EVALUATED] = evaluations[producer]
                answers[step.identity] = self.cores[step.name](**arguments)
            return (
                _write_query_chunk(
                    bank=bank, start=start, answer=answers[self.schedule[-1].identity]
                ),
                statuses
                & jnp.stack([answers[step.identity][3] for step in self.schedule]),
                tuple(
                    total + answers[step.identity][4]
                    for total, step in zip(totals, self.schedule, strict=True)
                ),
            )

        return jax.lax.fori_loop(
            0,
            self.chunks,
            run,
            (
                bank,
                jnp.ones((len(self.schedule),), dtype=jnp.bool_),
                tuple(
                    jnp.zeros(
                        (len(_query_landing_checks(node=self.cores[step.name].node)),),
                        dtype=jnp.int32,
                    )
                    for step in self.schedule
                ),
            ),
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
        if self.node.sample is not None:
            arguments[_DRAWS] = _sample_draws(node=self.node)
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
            # Freeze the producer's key order before shape-only tree conversion.
            arguments[_INFORMATION] = cast(
                "ShapeDtypePytree",
                MappingProxyType(
                    {
                        name: MappingProxyType(
                            {
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
                                **(
                                    {}
                                    if node.sample is None
                                    else {_DRAWS: _sample_draws(node=node)}
                                ),
                            }
                        )
                        for name, node in self.information_nodes.items()
                    }
                ),
            )
        return MappingProxyType(arguments)


def _sample_draws(*, node: QueryNode) -> FloatND:
    """A drawn query's unit-cube points, an operand so that no trace holds them."""
    from _lcm.query_information import sample_uniforms  # noqa: PLC0415

    return jnp.asarray(sample_uniforms(node=node))


def query_parameter_namespaces(*, builder: CoreArgumentBuilder) -> frozenset[str]:
    """Name fixed parameter roots read by queries."""
    if isinstance(builder, _OutsideReadArguments):
        return frozenset(
            name
            for name, _, _ in (() if builder.plan is None else builder.plan.bindings)
        )
    # The chunk loop runs every query of the period with its own arguments.
    queries = (
        tuple(builder.builders.values())
        if isinstance(builder, _ChunkArguments)
        else (builder,)
        if isinstance(builder, _QueryArguments)
        else ()
    )
    return frozenset(
        name
        for query in queries
        for name in _query_node_parameter_namespaces(node=query.node, information=False)
    ) | frozenset(
        name
        for query in queries
        for node in query.information_nodes.values()
        for name in _query_node_parameter_namespaces(node=node, information=True)
    )


def _query_node_parameter_namespaces(
    *, node: QueryNode, information: bool
) -> frozenset[str]:
    """Mirror the node's action/law and exact producing projection operands."""
    plan = node.context.current_iid_plans.get(node.read.period)
    names = {name for name, _, _ in (() if plan is None else plan.bindings)} | {
        path[0]
        for path in node.read.operand_paths.values()
        if len(path) == _FLAT_OPERAND_PATH_LENGTH and isinstance(path[0], str)
    }
    if not information:
        names.add(node.context.regime_name)
    return frozenset(names)


def _query_projection_operands(
    *, node: QueryNode, context: CoreBuildContext
) -> Mapping[str, PytreeValue]:
    """Bind each producing context's exact current-call transfer parameters."""
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
    """Read the same canonical transfer paths for policies and entry frames."""
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
    definitions: Mapping[str, QueryNode] = field(
        default=MappingProxyType({}), repr=False
    )
    """Whole query graph; omitted from the repr `jax.jit` takes of this callable."""
    row_batch: int = _ROW_BATCH
    """Information support rows evaluated together per step."""

    def evaluate(self, **arguments: PytreeValue) -> EvaluatedQAndF:
        """Expose an unmasked no-action payload for this chunk's proven consumers."""
        return map_lanes(
            function=_RawQueryEvaluation(
                node=self.node,
                period=self.period,
                fixed={
                    name: value
                    for name, value in arguments.items()
                    if name not in {*self.node.inputs, _SOURCE, _LIVE, _PROJECTION}
                },
                projection_operands=cast(
                    "Mapping[str, PytreeValue]", arguments[_PROJECTION]
                ),
            ),
            operands=(
                cast("BoolND", arguments[_LIVE]),
                cast("Mapping[str, FloatND | IntND]", arguments[_SOURCE]),
            ),
        )

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
            if name
            not in {
                *self.node.inputs,
                _SOURCE,
                _LIVE,
                _PROJECTION,
                _EVALUATED,
                _OUTSIDE,
            }
        }
        projection_operands = cast("Mapping[str, PytreeValue]", arguments[_PROJECTION])
        table = cast("OutsideTable | None", arguments.get(_OUTSIDE))
        from _lcm.query_information import (  # noqa: PLC0415
            InformationQueryCell,
            has_information,
        )

        information = has_information(
            node=self.node, definitions=self.definitions, period=self.period
        )
        if information:
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
                row_batch=self.row_batch,
                outside=table,
            )
        elif table is not None:
            cell = functools.partial(
                _stored_cell,
                node=self.node,
                table=table,
                fixed=fixed,
                projection_operands=projection_operands,
            )
        else:
            cell = functools.partial(
                _query_cell,
                node=self.node,
                period=self.period,
                fixed=fixed,
                projection_operands=projection_operands,
            )

        def lane(
            item: tuple[
                BoolND,
                Mapping[str, FloatND | IntND],
                Mapping[str, tuple[FloatND, IntND, BoolND]],
                EvaluatedQAndF | None,
            ],
        ) -> PolicyAnswer:
            active, states, rows, evaluated = item
            dependencies_at = {
                name: (*answer, dependencies[name][3], dependencies[name][4])
                for name, answer in rows.items()
            }
            if evaluated is not None:
                return _query_cell(
                    live=active,
                    source=states,
                    dependencies=dependencies_at,
                    node=self.node,
                    period=self.period,
                    fixed=fixed,
                    projection_operands=projection_operands,
                    evaluated=evaluated,
                )
            return cell(
                live=active,
                source=states,
                dependencies=dependencies_at,
            )

        # Cells run as lanes: a guard selects each lane's absent answer, and its
        # guarded branch reads stand-in inputs (see `_mapped_query_states`).
        answer = map_lanes(
            function=lane,
            operands=(
                live,
                source,
                {name: answer[:3] for name, answer in dependencies.items()},
                cast("EvaluatedQAndF | None", arguments.get(_EVALUATED)),
            ),
        )
        return (
            answer[0],
            # Information consumers select their own actions; only values travel.
            jnp.full(live.shape, -1, dtype=jnp.int32) if information else answer[1],
            answer[2],
            jnp.all(answer[3]),
            _query_landing_totals(counts=answer[4]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _RawQueryEvaluation:
    """Evaluate identical terminal canonical inputs without caller bookkeeping."""

    node: QueryNode
    period: int
    fixed: Mapping[str, PytreeValue]
    projection_operands: Mapping[str, PytreeValue]

    def __call__(
        self, item: tuple[BoolND, Mapping[str, FloatND | IntND]]
    ) -> EvaluatedQAndF:
        live, source = item
        arguments, _ = query_arguments(
            guard=live,
            source=source,
            node=self.node,
            fixed=self.fixed,
            projection_operands=self.projection_operands,
        )
        return _evaluate_query_action(
            node=self.node, period=self.period, arguments=arguments
        )


def with_contracted_tables(
    *, node: QueryNode, arguments: Mapping[str, PytreeValue]
) -> Mapping[str, PytreeValue]:
    """Return `arguments` with the contracted tables `node`'s problem reads.

    Called before a query's lanes are mapped, so each table is built once per
    call rather than once per lane under the query's guard.
    """
    from _lcm.regime_building.Q_and_F import (  # noqa: PLC0415
        prepare_contracted_values,
    )

    return prepare_contracted_values(
        function=node.context.Q_and_F_functions.get(node.read.period),
        arguments=arguments,
        skip=frozenset(node.gated_channels),
    )


def _row_batch(*, chunk_size: int, halvings: int = 0) -> int:
    """Support rows per step for a chunk of `chunk_size` source cells.

    The step evaluates `_STEP_LANES` halved `halvings` times, and never fewer than
    `_ROW_BATCH` rows.
    """
    return max(_ROW_BATCH, (_STEP_LANES >> halvings) // chunk_size)


def map_lanes[T: PytreeValue, R: PytreeValue](
    *,
    function: Callable[[T], R],
    operands: T,
    batch_size: int | None = None,
) -> R:
    """Evaluate independent lanes together, `batch_size` at a time.

    A guard inside `function` then selects per lane rather than branching, so its
    guarded branch also runs on discarded lanes. That branch reads stand-in
    inputs there, and its loops are bounded, so it stays finite and silent.
    """
    batched = jax.vmap(function)
    size = jax.tree.leaves(operands)[0].shape[0]
    if batch_size is None or batch_size >= size:
        return batched(operands)
    # `lax.map(batch_size=)` cannot reshape an empty output such as zero checks.
    steps, rest = divmod(size, batch_size)
    head = jax.tree.map(
        lambda leaf: leaf.reshape((steps * batch_size, *leaf.shape[2:])),
        jax.lax.map(
            batched,
            jax.tree.map(
                lambda leaf: leaf[: steps * batch_size].reshape(
                    (steps, batch_size, *leaf.shape[1:])
                ),
                operands,
            ),
        ),
    )
    if not rest:
        return head
    tail = batched(jax.tree.map(lambda leaf: leaf[steps * batch_size :], operands))
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves), head, tail)


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
    stay: StayTable | None = None,
    evaluated: EvaluatedQAndF | None = None,
) -> PolicyAnswer:
    """Evaluate ordinary feasibility and the kept set before one canonical max.

    A given `stay` is this cell's action table evaluated without participation
    at a row its action problem cannot tell apart from this one.
    """
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
                guard=active & live,
                source=source,
                dependencies=dependencies,
                node=node,
                period=period,
                fixed=fixed,
                projection_operands=projection_operands,
                stay=stay,
                evaluated=evaluated,
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
        guard=live,
        source=source,
        dependencies=dependencies,
        node=node,
        period=period,
        fixed=fixed,
        projection_operands=projection_operands,
        stay=stay,
        evaluated=evaluated,
    )


def _stored_cell(
    *,
    live: BoolND,
    source: Mapping[str, FloatND | IntND],
    dependencies: Mapping[str, PolicyAnswer],
    node: QueryNode,
    table: OutsideTable,
    fixed: Mapping[str, PytreeValue],
    projection_operands: Mapping[str, PytreeValue],
) -> PolicyAnswer:
    """Gather a stored outside read whose callers realise no shocks.

    A stored read is a Raw read, which depends on no other query.
    """
    del dependencies
    from _lcm.outside_values import read_outside_value  # noqa: PLC0415

    return read_outside_value(
        node=node,
        table=table,
        live=live,
        available={
            **projection_operands,
            **source,
            "period": fixed["period"],
            "age": fixed["age"],
            _PHYSICAL_CONTEXTS: fixed.get(_PHYSICAL_CONTEXTS, {}),
        },
        codes=MappingProxyType({}),
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
    """Short-circuit equivalent aliases without evaluating their transfers."""
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
    guard: BoolND | None = None,
) -> Mapping[str, FloatND | IntND]:
    """Use proved equal law-input maps, excluding unrelated target transfers."""
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
        guard=guard,
    )


def query_arguments(
    *,
    guard: BoolND,
    source: Mapping[str, FloatND | IntND],
    node: QueryNode,
    fixed: Mapping[str, PytreeValue],
    projection_operands: Mapping[str, PytreeValue],
) -> tuple[dict[str, PytreeValue], Mapping[str, FloatND | IntND]]:
    """The action problem's operands at one cell, and its mapped states."""
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
    mapped = _mapped_query_states(node=node, available=available, guard=guard)
    arguments = {**fixed, **projection_operands, **mapped}
    if node.gated_channels:
        values = dict(
            cast("Mapping[RegimeName, FloatND]", arguments["next_regime_to_V_arr"])
        )
        for target, indices in node.gated_channels.items():
            value = values[target]
            values[target] = jnp.stack(
                tuple(value[..., index] for index in indices), axis=-1
            )
        arguments["next_regime_to_V_arr"] = MappingProxyType(values)
    arguments.setdefault(_PHYSICAL_CONTEXTS, {})
    return arguments, mapped


def stay_table(
    *,
    node: QueryNode,
    period: int,
    arguments: Mapping[str, PytreeValue],
    dependencies: Mapping[str, PolicyAnswer],
    participate: bool = True,
    evaluated: EvaluatedQAndF | None = None,
) -> StayTable:
    """Evaluate the action problem over its whole action grid.

    Without `participate`, feasibility is the ordinary one; `kept_table` adds the
    declared participation afterwards.
    """
    action_names = node.context.state_action_space.action_names
    action_point = with_signature(
        functools.partial(
            _query_action,
            node=node,
            period=period,
            arguments=(
                arguments if evaluated is None else {**arguments, _EVALUATED: evaluated}
            ),
            dependencies=dependencies,
            participate=participate,
        ),
        args=list(action_names),
        enforce=False,
    )
    evaluate = cast(
        "Callable[..., StayTable]",
        productmap(
            func=action_point,
            variables=action_names,
            batch_sizes=dict.fromkeys(action_names, 0),
        ),
    )
    return evaluate(**{name: arguments[name] for name in action_names})


def kept_table(
    *,
    node: QueryNode,
    arguments: Mapping[str, PytreeValue],
    dependencies: Mapping[str, PolicyAnswer],
    table: StayTable,
) -> StayTable:
    """Apply the declared participation to a table evaluated without it."""
    if node.participation is None:
        return table
    participation = node.participation
    q, ordinary, valid, landings, objective = table
    names = node.context.state_action_space.action_names
    grids = jnp.meshgrid(
        *(jnp.asarray(arguments[name]) for name in names), indexing="ij"
    )
    size = math.prod(ordinary.shape)

    # keyword-only-exempt: library-callback=jax.vmap
    def keep_at(actions: tuple[PytreeValue, ...], q_point: FloatND) -> BoolND:
        operands = {
            **arguments,
            **dict(zip(names, actions, strict=True)),
            **dependencies,
            "q": q_point,
        }
        keep = participation(
            **{
                name: operands[name]
                for name in inspect.signature(participation).parameters
            }
        )
        if keep.dtype != jnp.bool_:
            raise TypeError("Query participation must be boolean.")
        if keep.ndim != 0:
            raise ValueError("Query participation must be scalar per action.")
        return keep

    keep = jax.vmap(keep_at)(
        tuple(grid.reshape(size) for grid in grids),
        q.reshape((size, *q.shape[ordinary.ndim :])),
    ).reshape(ordinary.shape)
    return q, ordinary & keep, valid, landings, objective


def _active_query_cell(
    *,
    live: BoolND,
    guard: BoolND,
    source: Mapping[str, FloatND | IntND],
    dependencies: Mapping[str, PolicyAnswer],
    node: QueryNode,
    period: int,
    fixed: Mapping[str, PytreeValue],
    projection_operands: Mapping[str, PytreeValue],
    stay: StayTable | None = None,
    evaluated: EvaluatedQAndF | None = None,
) -> PolicyAnswer:
    """Evaluate only a caller-admitted mapped action problem."""
    arguments, mapped = query_arguments(
        guard=guard,
        source=source,
        node=node,
        fixed=fixed,
        projection_operands=projection_operands,
    )
    if node.landing is not None:
        return _landing_answer(
            node=node, live=live, guard=guard, arguments=arguments, mapped=mapped
        )
    context = node.context
    q, feasible, valid, landing_counts, objective = (
        stay_table(
            node=node,
            period=period,
            arguments=arguments,
            dependencies=dependencies,
            evaluated=evaluated,
        )
        if stay is None
        else kept_table(
            node=node, arguments=arguments, dependencies=dependencies, table=stay
        )
    )
    axes = tuple(range(feasible.ndim))
    support = jnp.any(feasible)
    if context.has_taste_shocks:
        action, value, taste_valid = _taste_query_choice(
            node=node,
            q=q,
            objective=objective,
            feasible=feasible,
            arguments=arguments,
        )
        valid &= taste_valid
        # An acting draw without an alternative to draw publishes no action.
        support &= action >= 0
    elif objective is not None:
        # A separate objective chooses as in the regime's own solve.
        action, value = household_argmax_and_readout(
            selection=objective,
            readout=q,
            where=feasible,
            weights=_taste_weights(node=node, arguments=arguments),
        )
    else:
        pareto_weights = context.pareto_weights
        action, values, _ = collective_argmax_and_readout(
            member_Q={
                name: q[..., index] for index, name in enumerate(context.members)
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
        value = jnp.stack(tuple(values[name] for name in context.members))
    valid = (
        jnp.all(valid)
        & ~jnp.any(jnp.isnan(value))
        & _mapped_query_valid(node=node, mapped=mapped)
    )
    answer = (
        value,
        jnp.where(support, action, -1).astype(jnp.int32),
        live & support,
        ~live | valid,
        jnp.concatenate(
            (
                sum_counts(counts=jnp.where(live, landing_counts, 0), axis=axes),
                count_projected_off_grid(
                    rules=node.grid_rules, mapped=mapped, active=guard
                ),
            )
        ),
    )
    if node.decision is not None:
        operands = {
            **arguments,
            **dependencies,
            "q": q,
            "feasible": feasible,
            "answer": answer,
        }
        answer = node.decision(
            **{
                name: operands[name]
                for name in inspect.signature(node.decision).parameters
            }
        )
        _validate_decision_answer(node=node, answer=answer)
    return answer


def _landing_answer(
    *,
    node: QueryNode,
    live: BoolND,
    guard: BoolND,
    arguments: Mapping[str, PytreeValue],
    mapped: Mapping[str, FloatND | IntND],
) -> PolicyAnswer:
    """Read a later stage's target at its occurrence, at the mapped states.

    The value is the one stored at the occurrence, interpolated on the target's
    grid there: the target decides later in the period, undiscounted. The read
    chooses no action. It is supported where its guard holds and the target's
    problem at the occurrence has a feasible action at the mapped states.
    """
    context = node.context
    target = node.read.target
    info = (
        (context.period_to_regime_v_interp or {})
        .get(cast("int", node.landing), {})
        .get(target, context.regime_to_v_interpolation_info[target])
    )
    interpolate = get_V_interpolator(
        v_interpolation_info=info, state_prefix="", V_arr_name="V_arr"
    )
    stored = cast("Mapping[RegimeName, FloatND]", arguments["next_regime_to_V_arr"])[
        target
    ]
    value = jnp.stack(
        tuple(
            interpolate(
                **{state: mapped[state] for state in info.state_names},
                V_arr=stored[..., index],
            )
            for index, _ in enumerate(context.members)
        )
    )
    valid = ~jnp.any(jnp.isnan(value)) & _mapped_query_valid(node=node, mapped=mapped)
    return (
        jnp.where(guard, value, -jnp.inf),
        jnp.asarray(-1, dtype=jnp.int32),
        live & guard & _landing_support(node=node, arguments=arguments),
        ~live | valid,
        count_projected_off_grid(rules=node.grid_rules, mapped=mapped, active=guard),
    )


def _landing_support(
    *, node: QueryNode, arguments: Mapping[str, PytreeValue]
) -> BoolND:
    """Whether the target has a feasible action at the landing, as at its occurrence.

    The target's own action problem there decides, over its whole action grid;
    only its feasibility is read.
    """
    flow = cast(
        "CanonicalActionKernel",
        node.context.Q_and_F_functions[cast("int", node.landing)],
    ).utilities_and_F
    action_names = node.context.state_action_space.action_names
    feasible = cast(
        "Callable[..., BoolND]",
        productmap(
            func=with_signature(
                functools.partial(
                    _feasible_at,
                    flow=flow,
                    operands={**arguments, "period": jnp.int32(node.landing)},
                ),
                args=list(action_names),
                enforce=False,
            ),
            variables=action_names,
            batch_sizes=dict.fromkeys(action_names, 0),
        ),
    )
    return jnp.any(feasible(**{name: arguments[name] for name in action_names}))


def _feasible_at(
    *,
    flow: Callable[..., tuple[PytreeValue, ...]],
    operands: Mapping[str, PytreeValue],
    **actions: PytreeValue,
) -> BoolND:
    """The feasibility a canonical flow returns last, at one action."""
    inputs = {**operands, **actions}
    return jnp.asarray(
        flow(**{name: inputs[name] for name in inspect.signature(flow).parameters})[-1]
    )


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
    objective: FloatND | None,
    feasible: BoolND,
    arguments: Mapping[str, PytreeValue],
) -> tuple[IntND, FloatND, BoolND]:
    """Price hidden target taste at its logsum; draw only in the acting pass.

    With a separate `objective`, the alternatives and their prices are those of
    the regime's own solve (`smoothed_profiles`, `smoothed_logsum`).
    """
    from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
        TASTE_SHOCK_SCALE_PARAM,
    )

    context = node.context
    n_discrete = sum(
        isinstance(context.grids[name], DiscreteGrid)
        for name in context.state_action_space.action_names
    )
    if not n_discrete:
        raise ValueError("EV1 policy queries require a discrete action axis.")
    weights = _taste_weights(node=node, arguments=arguments)
    scale = cast("FloatND", arguments[TASTE_SHOCK_SCALE_PARAM])
    key = (
        cast("jax.Array", arguments["taste_shock_key"])
        if arguments.get(_ACTING, False)
        else None
    )
    if objective is None:
        discrete_values, identifiers, at_choice, decider = _taste_stay_roster(
            q=q, feasible=feasible, n_discrete=n_discrete, weights=weights
        )
        discrete, value = taste_choice(
            values=discrete_values,
            at_choice=at_choice,
            decider=decider,
            scale=scale,
            key=key,
        )
        return (
            jnp.where(discrete < 0, -1, identifiers[discrete]),
            value,
            _scale_is_valid(scale),
        )
    smoothed = smoothed_profiles(
        # The reduction reads only F, Q and Q_objective of the parts.
        parts=QAndFParts(U=q, F=feasible, CE=q, Q=q, Q_objective=objective),
        weights=weights,
        n_discrete_action_axes=n_discrete,
    )
    discrete_values = smoothed.conditional.objective.ravel()
    identifiers = (
        jnp.arange(discrete_values.size, dtype=jnp.int32)
        * math.prod(feasible.shape[n_discrete:])
        + smoothed.conditional.continuous_index.ravel()
    ).astype(jnp.int32)
    if key is None:
        discrete = jnp.argmax(discrete_values)
        value = smoothed_logsum(smoothed=smoothed, scale=scale)[0]
    else:
        discrete, value = taste_choice(
            values=discrete_values,
            at_choice=smoothed.stored.reshape(-1, q.shape[-1]),
            decider=smoothed.decider,
            scale=scale,
            key=key,
        )
    return (
        jnp.where(discrete < 0, -1, identifiers[discrete]),
        value,
        _scale_is_valid(scale),
    )


def _scale_is_valid(scale: FloatND) -> BoolND:
    """Whether the taste-shock scale is finite and positive."""
    return jnp.all(jnp.isfinite(scale) & (scale > 0))


def _taste_weights(*, node: QueryNode, arguments: Mapping[str, PytreeValue]) -> FloatND:
    """The members' Pareto weights in member order."""
    context = node.context
    return _stacked_pareto_weights(
        pareto_weights=context.pareto_weights,
        members=context.members,
        states_actions_params=cast("QAndFKwargs", arguments),
    )


def taste_choice(
    *,
    values: FloatND,
    at_choice: FloatND,
    decider: IntND,
    scale: FloatND,
    key: jax.Array | None,
) -> tuple[jax.Array, FloatND]:
    """Choose among the deciding member's taste alternatives and publish their value.

    With `key`, the deciding member draws: it takes the alternative maximizing
    `values` plus one mean-zero EV1 term each, and publishes `at_choice` there.
    Without it, the choice is the noise-free argmax, and the value is the deciding
    member's log-sum, with every other member at its mean of `at_choice` under the
    deciding member's choice probabilities (`decider_expected_values`).

    Where every alternative is worth `-inf` to the deciding member, the logit has
    no probabilities: the draw takes no alternative (index `-1`) and publishes
    `-inf` for every member, as the regime's own simulation does.

    Args:
        values: The deciding member's value of each alternative.
        at_choice: Each alternative's published value, every member's value in
            a trailing member axis.
        decider: The deciding member's index.
        scale: The taste-shock scale.
        key: The acting draw's key, or `None` to price the alternatives.

    Returns:
        The index of the chosen alternative and the published value.
    """
    from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
        draw_taste_shock_noise,
    )

    if key is not None:
        chosen = jnp.argmax(
            values + draw_taste_shock_noise(key=key, shape=values.shape, scale=scale)
        )
        worthless = jnp.all(jnp.isneginf(values))
        return (
            jnp.where(worthless, -1, chosen),
            jnp.where(worthless, -jnp.inf, at_choice[chosen]),
        )
    chosen = jnp.argmax(values)
    return chosen, decider_expected_values(
        decider=decider, Qc=values, at_choice=at_choice, scale=scale
    )


def _taste_stay_roster(
    *, q: FloatND, feasible: BoolND, n_discrete: int, weights: FloatND
) -> tuple[FloatND, IntND, FloatND, IntND]:
    """Optimize continuous actions before assigning one taste per discrete cell.

    Returns the deciding member's value of each discrete cell, the stable action
    ID attaining it, the published value there (`taste_choice`) and the deciding
    member, the one with the positive Pareto weight (`decider_choice`).
    """
    decider, values, continuous, at_choice = decider_choice(
        Q=q,
        feasibility=feasible,
        weights=weights,
        n_discrete_action_axes=n_discrete,
    )
    continuous_cells = math.prod(feasible.shape[n_discrete:])
    identifiers = (
        jnp.arange(values.size, dtype=jnp.int32) * continuous_cells + continuous
    ).astype(jnp.int32)
    return values, identifiers, at_choice, decider


def _mapped_query_states(
    *,
    node: QueryNode,
    available: Mapping[str, PytreeValue],
    names: frozenset[str] | None = None,
    guard: BoolND | None = None,
) -> Mapping[str, FloatND | IntND]:
    """Canonical scalar transfer boundary shared by solve and actual routing.

    Under a false `guard` a cell runs only as a discarded lane of a batch, and the
    target reads its first grid point instead: a point its own solve evaluates.
    """
    grid = node.context.state_action_space.states
    state_grids = {name: node.context.grids[name] for name in grid}
    # Where the engine receives a projected state: it takes the target state's
    # kind, so a projection may return `1` for a continuous state or a Python
    # integer for a code.
    mapped = {
        name: _as_target_state(
            value=helper(
                **{key: available[key] for key in inspect.signature(helper).parameters}
            ),
            name=name,
            grids=state_grids,
        )
        for name, helper in node.read.projection.items()
        if names is None or name in names
    }
    if any(value.ndim != 0 for value in mapped.values()):
        raise ValueError("A mapped query state must be scalar per source cell.")
    if guard is None:
        return mapped
    return {
        name: jnp.where(guard, value, jnp.asarray(grid[name][0], value.dtype))
        for name, value in mapped.items()
    }


def _as_target_state(
    *, value: PytreeValue, name: str, grids: Mapping[str, Grid]
) -> FloatND | Code:
    """Adapt one projection's output to the role of the target state it lands on."""
    # A projection returns one array-like; the adapters refuse any other kind.
    output = cast("FloatND | IntND | BoolND", value)
    if name not in grids:
        return jnp.asarray(output)
    source = f"The projection of state '{name}'"
    if target_role(grids[name]) is OutputRole.FLOAT:
        return as_canonical_float(value=output, source=source)
    return as_code(value=output, source=source)


def _query_action(
    *,
    node: QueryNode,
    period: int,
    arguments: Mapping[str, PytreeValue],
    dependencies: Mapping[str, PolicyAnswer],
    participate: bool = True,
    **actions: PytreeValue,
) -> StayTable:
    """Preserve the canonical Q/F arrays and add only declared participation."""
    inputs = {**arguments, **actions}
    evaluated = cast("EvaluatedQAndF | None", arguments.get(_EVALUATED))
    if evaluated is None:
        evaluated = _evaluate_query_action(node=node, period=period, arguments=inputs)
    parts = evaluated.parts
    q, ordinary = parts.Q, parts.F
    # Participation consumes this Q before it decides keep: its rejection cannot
    # erase a continuation landing already needed by an ordinarily feasible action.
    landings = jnp.where(
        ordinary, _evaluated_query_landings(node=node, evaluated=evaluated), 0
    )
    if q.shape != (len(node.context.members),):
        raise ValueError("Query Q must carry the canonical member tuple.")
    if ordinary.dtype != jnp.bool_:
        raise TypeError("Canonical query feasibility must be boolean.")
    if ordinary.ndim != 0:
        raise ValueError("Canonical query feasibility must be scalar per action.")
    valid = ~jnp.any(ordinary & jnp.any(jnp.isnan(q)))
    for status in evaluated.bound_status:
        for flag in status:
            valid &= ~jnp.any(flag)
    if node.participation is None or not participate:
        return q, ordinary, valid, landings, parts.Q_objective
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
    return q, ordinary & keep, valid, landings, parts.Q_objective


def _evaluate_query_action(
    *, node: QueryNode, period: int, arguments: Mapping[str, PytreeValue]
) -> EvaluatedQAndF:
    """Evaluate only the canonical signature's current-call operands."""
    function = cast("CanonicalActionKernel", node.context.Q_and_F_functions[period])
    return _shared_evaluation(
        function=function,
        arguments={
            name: arguments[name] for name in inspect.signature(function).parameters
        },
    )


_SHARED_EVALUATIONS: weakref.WeakKeyDictionary[
    CanonicalActionKernel, dict[Hashable, Callable[..., EvaluatedQAndF]]
] = weakref.WeakKeyDictionary()


def _shared_evaluation(
    *, function: CanonicalActionKernel, arguments: Mapping[str, PytreeValue]
) -> EvaluatedQAndF:
    """Trace one action problem once per abstract signature, however often read.

    A nested `jax.jit` keys the trace on the function and its abstract inputs:
    later reads, the `eval_shape` that sizes a guarded branch, and other programs
    of the same graph reuse the jaxpr. Its call boundary also lets lowering share
    one action-problem body across repeated reads instead of expanding that body
    at every call site. The backend compiler remains free to inline those calls.
    Concrete leaves stay closed over, and an eager evaluation stays eager.
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
    key = cast("Hashable", key)
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

        variants[key] = jax.jit(evaluate)
    return variants[key](
        *(leaf for leaf, is_traced in zip(leaves, traced, strict=True) if is_traced)
    )


def _evaluate(*, function: CanonicalActionKernel) -> Callable[..., EvaluatedQAndF]:
    """The kernel's evaluation, called with the query's flat argument mapping."""
    return cast("Callable[..., EvaluatedQAndF]", function.evaluated)


def _closed_over_key(
    *, leaf: PytreeValue | None
) -> tuple[type[object], PytreeValue | str | None]:  # noqa: PAN001 - heterogeneous leaf classes share only object
    """Key a concrete leaf by type and bits: `1 == 1.0` and `0.0 == -0.0`."""
    return (type(leaf), leaf.hex() if isinstance(leaf, float) else leaf)


def _check_query_answer(
    *, name: str, node: QueryNode, status: bool, totals: IntND
) -> None:
    """Check mandatory numerical status before this producer's reached landings."""
    if not status:
        raise ExecutionPlanningError(
            f"Policy query {name!r} has unresolved numerical status."
        )
    fail_on_query_landings(
        totals=totals,
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
    root_context: SolverBuildContext
    period: int
    halvings: int
    """How often admission halved the lanes a step evaluates."""
    fixed_flat_params: FlatParams = MappingProxyType({})
    outside_tables: Mapping[str, tuple[Mapping[str, QueryNode], str]] = (
        MappingProxyType({})
    )
    """Label of each stored outside read to its destination's bound Raw graph and
    that graph's root."""
    shared_functions: dict[tuple[str, int, int], Callable[..., PytreeValue]] | None = (
        None
    )
    """Program functions by name, chunk width and support rows per step, one dict
    per group of periods.

    Periods that trace the same programs hold the same dict and run its
    functions, so one lowering serves the group. The dict is mutable because a
    narrowed period adds its widths for the periods narrowed after it.
    """

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Publish the sole native declaration authority."""
        return self.programs

    def sharing(
        self, *, functions: dict[tuple[str, int, int], Callable[..., PytreeValue]]
    ) -> Self:
        """Run the functions of the first period that entered `functions`."""
        row_batch = _row_batch(chunk_size=self.chunk_size, halvings=self.halvings)
        return replace(
            self,
            shared_functions=functions,
            programs=MappingProxyType(
                {
                    name: replace(
                        program,
                        function=functions.setdefault(
                            (name, self.chunk_size, row_batch), program.function
                        ),
                    )
                    for name, program in self.programs.items()
                }
            ),
        )

    def narrow_workspace(self, *, halvings: int = 1) -> PeriodKernel | None:
        """Halve the lanes a step evaluates `halvings` times.

        The common chunk halves down to one source cell and, from there, the support
        rows per step down to `_ROW_BATCH`. A kernel at both floors refuses.
        """
        if (
            self.chunk_size == 1
            and _row_batch(chunk_size=1, halvings=self.halvings) == _ROW_BATCH
        ):
            return None
        candidate = build_policy_query_kernel(
            root_context=self.root_context,
            period=self.period,
            definitions=self.definitions,
            root=self.root,
            chunk_size=max(1, self.chunk_size >> halvings),
            outside_tables=self.outside_tables,
            halvings=self.halvings + halvings,
        )
        candidate = candidate.with_fixed_params(
            fixed_flat_params=self.fixed_flat_params
        )
        if self.shared_functions is None:
            return candidate
        return candidate.sharing(functions=self.shared_functions)

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> Self:
        """Expose each target's fixed operands through its canonical argument pool.

        A table binds them into its destination's arguments as the chunk loop does.
        """
        from _lcm.outside_values import _TableArguments  # noqa: PLC0415

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
        programs = dict(self.programs)
        programs["_chunks"] = replace(
            programs["_chunks"],
            argument_builder=replace(
                cast("_ChunkArguments", programs["_chunks"].argument_builder),
                builders=builders,
            ),
        )
        for name, program in self.programs.items():
            if isinstance(builder := program.argument_builder, _TableArguments):
                programs[name] = replace(
                    program,
                    argument_builder=replace(
                        builder,
                        builder=replace(builder.builder, fixed_flat_params=frozen),
                    ),
                )
        return replace(
            self,
            fixed_flat_params=frozen,
            builders=builders,
            programs=MappingProxyType(programs),
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
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        """Tabulate the destinations of stored reads, run every chunk in one
        dispatch, check statuses, then publish the bank.

        A table's landing counts are checked when it is built; its failed cells
        fail only the reads that take them. The tables are released once the
        chunks have run.
        """
        del logger
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        program = self.programs["_chunks"]
        owner = PendingSolveWork()
        try:
            bank = cast("FloatND", compiled_cores["_initialize"]())
            owner.record(outputs=bank, devices=frozenset(bank.devices()))
            tables = self._tables(
                compiled_cores=compiled_cores, context=context, owner=owner
            )
            arguments = {
                **cast("_ChunkArguments", program.argument_builder).operands(context),
                "bank": bank,
                **{
                    argument: tables[reference.producer][reference.label]
                    for argument, reference in (
                        program.requirements.internal_inputs.items()
                    )
                },
            }
            del tables
            devices = _devices(tree=arguments)
            owner.before(devices=devices)
            bank, statuses, totals = cast(
                "_ChunkOutput", compiled_cores["_chunks"](**arguments)
            )
            owner.record(outputs=(bank, statuses, totals), devices=devices)
            del arguments
            for step, status, total in zip(
                cast("_QueryChunks", program.function).schedule,
                jax.device_get(statuses).tolist(),
                totals,
                strict=True,
            ):
                _check_query_answer(
                    name=step.name,
                    node=self.definitions[step.name],
                    status=status,
                    totals=total,
                )
            owner.before(devices=frozenset(bank.devices()))
            value = cast("FloatND", compiled_cores["_finish"](bank=bank))
            owner.record(outputs=value, devices=frozenset(value.devices()))
            return KernelOutput(value=value)
        finally:
            owner.close()

    def _tables(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        context: CoreBuildContext,
        owner: PendingSolveWork,
    ) -> Mapping[str, Mapping[str, FloatND | BoolND]]:
        """Build each table and check its landing counts, by program and output."""
        tables = {}
        for label, (definitions, key) in self.outside_tables.items():
            name = OUTSIDE_PROGRAM + label
            if name not in self.programs:
                continue
            program = self.programs[name]
            arguments = program.argument_builder(context)
            devices = _devices(tree=arguments)
            owner.before(devices=devices)
            output = cast(
                "tuple[FloatND, BoolND, IntND]", compiled_cores[name](**arguments)
            )
            owner.record(outputs=output, devices=devices)
            _check_query_answer(
                name=name, node=definitions[key], status=True, totals=output[2]
            )
            tables[name] = {
                spec.label: output[cast("int", spec.path[0])]
                for spec in program.internal_outputs
            }
        return tables


def _devices(
    *, tree: Mapping[str, PytreeValue | ShapeDtypePytree]
) -> frozenset[jax.Device]:
    """The devices holding the arrays of an argument tree."""
    return frozenset(
        device
        for value in jax.tree.leaves(tree)
        if isinstance(value, jax.Array)
        for device in value.devices()
    )


def _query_schedule(
    *, definitions: Mapping[str, QueryNode]
) -> tuple[InternalDispatch, ...]:
    """Evaluate each query once, in demand order, retaining its answer for readers."""
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
