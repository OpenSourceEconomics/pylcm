"""Prove that an outside read lands on stored nodes, and tabulate those nodes.

An exit's outside value is the value of the destination's problem at the member's
projected state. A read is aligned when the build proves all of:

- the destination is solved in the read's period, on a grid that is not
  age-specialized there, and observes no shock through another query;
- every state the read projects is a point of the destination's grid;
- every destination shock group is either fixed by the caller's frame or observed by
  no ancestor, and a fixed group takes its nodes from an ancestor whose frame
  enumerates the destination's own nodes.

The proof is conservative and never evaluates anything inside a compiled program.
What it cannot prove leaves the read unaligned, and the read is then evaluated at its
exact point.

An aligned read's destination is tabulated by a program of its own: the problem's
value at every grid cell of the destination and every node of the groups the read
keeps, with the other groups integrated. The read then gathers its value from that
table at its projected cell and at the nodes its caller's rows name.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np

from _lcm.current_iid import CurrentIIDPlan
from _lcm.engine import Regime
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalOutputSpec,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.grids import Grid
from _lcm.policy_queries import (
    _PHYSICAL_CONTEXTS,
    _PROJECTION,
    _QUERY_ARGUMENTS,
    OUTSIDE_PROGRAM,
    OutsideTable,
    OutsideValueRead,
    PolicyAnswer,
    QueryKind,
    QueryNode,
    _as_target_state,
    _mapped_query_states,
    _mapped_query_valid,
    _node_inputs,
    _query_availability,
    _query_cell,
    _query_landing_checks,
    _query_landing_totals,
    _QueryArguments,
    _row_batch,
    map_lanes,
    query_value_reads,
    with_contracted_tables,
)
from _lcm.query_information import (
    FrameCodes,
    InformationQueryCell,
    _bindings,
    _callable_reads,
    _parent_sources,
    _pinned_columns,
    has_information,
)
from _lcm.shock_projection import (
    _MAX_FINITE_PROJECTION_CELLS,
    finite_outputs,
    returned_input,
)
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.exceptions import RegimeInitializationError
from lcm.grids import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from lcm.shocks import Discrete, Law, NodeSet, ShockGroupName, ShockName, _fingerprint
from lcm.typing import BoolND, FloatND, IntND, RegimeName, StateName

# Grids whose points are fixed when the model is built. An irregular grid also needs
# its points declared rather than supplied at run time.
_STATIC_GRIDS = (
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    IrregSpacedGrid,
)


@dataclass(frozen=True, kw_only=True)
class UnalignedOutsideRead:
    """An outside read the build cannot prove aligned, and why."""

    reason: str
    """What the proof could not establish, naming the state, value or group."""


@dataclass(frozen=True, kw_only=True)
class _FrameShock:
    """A shock a projection reads from one query's frame."""

    owner: str
    """The query whose frame realises the shock."""
    shock: ShockName
    """The shock's name."""
    law: Law
    """The shock's law in that frame."""
    nodes: NodeSet
    """The nodes that realise the shock's group in that frame."""


def align_outside_read(
    *,
    key: str,
    definitions: Mapping[str, QueryNode],
    canonical_regimes: Mapping[RegimeName, Regime],
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> OutsideValueRead | UnalignedOutsideRead:
    """Prove that the outside read `key` lands on its destination's stored nodes.

    Args:
        key: The read's query key; a Raw read of the destination's problem.
        definitions: Mapping of query keys to the bound nodes of its graph.
        canonical_regimes: Mapping of regime names to their canonical regimes,
            which hold each regime's age-specialized state nodes.
        value_periods_by_regime: Mapping of regime names to the periods in which
            each regime is solved.

    Returns:
        The aligned read with the destination groups it keeps as axes, or why the
        read is not provably aligned.

    """
    node = definitions[key]
    reason = _unaligned_destination(
        node=node,
        canonical_regimes=canonical_regimes,
        value_periods_by_regime=value_periods_by_regime,
    )
    for state in sorted(node.read.projection):
        if reason is not None:
            break
        reason = _unaligned_state(
            state=state,
            node=node,
            definitions=definitions,
            canonical_regimes=canonical_regimes,
        )
    kept = _kept_groups(key=key, definitions=definitions) if reason is None else ()
    if isinstance(kept, str):
        reason = kept
    if reason is not None:
        return UnalignedOutsideRead(reason=reason)
    groups = tuple(group for group, _ in kept)
    return OutsideValueRead(
        label=":".join((node.read.target, *groups)),
        groups=groups,
        origins=tuple(origin for _, origin in kept),
    )


def outside_value_program(
    *,
    read: OutsideValueRead,
    definitions: Mapping[str, QueryNode],
    key: str,
    reader: RegimeName,
    width: int,
) -> CoreProgram:
    """Declare the program that tabulates an aligned read's destination problem.

    Args:
        read: The aligned read whose destination's values the program stores.
        definitions: Mapping of query keys to the destination's bound Raw graph,
            whose root `key` reads every destination state as itself.
        key: The query key of that root.
        reader: The regime whose period kernel runs the program.
        width: The grid cells evaluated together per step.

    Returns:
        A planned program returning the values, the support and the reached-landing
        totals of `OutsideValueTable`. The values and the support are published to
        the reader's other programs under the labels "value" and "support".

    """
    node = definitions[key]
    if node.ancestors or node.read.kind is not QueryKind.RAW:
        raise ValueError(
            f"Query {key!r} is not the root Raw read of {node.read.target!r}."
        )
    plan = node.context.current_iid_plans.get(node.read.period)
    groups = () if plan is None else tuple(group for group, _, _ in plan.bindings)
    missing = sorted(set(read.groups) - set(groups))
    if missing:
        raise ValueError(
            f"{node.read.target!r} has no shock groups {missing} to keep in period "
            f"{node.read.period}."
        )
    kept = tuple(groups.index(group) for group in read.groups)
    base, value_reads = _node_inputs(node=node, period=node.read.period)
    name = OUTSIDE_PROGRAM + read.label
    table = OutsideValueTable(key=key, definitions=definitions, kept=kept, width=width)
    values, support = table.shapes()
    dtype = jnp.asarray(0.0).dtype
    return CoreProgram(
        name=name,
        function=table,
        argument_builder=_TableArguments(
            key=key, builder=_QueryArguments(node=node, base=base)
        ),
        requirements=CoreExecutionRequirements(
            value_reads=query_value_reads(
                key=key, reads=value_reads, regime=reader, core_key=name
            )
        ),
        # The width is bound by the function, not its arguments: a narrowed table
        # takes the same arguments and must not share one executable.
        compiler_options=(("query_chunk", width),),
        output_roles=(
            StateAxesLeading(state_names=(), shape=values, dtype=dtype),
            StateAxesLeading(state_names=(), shape=support, dtype=jnp.bool_),
            StateAxesLeading(
                state_names=(),
                shape=(len(_query_landing_checks(node=node)),),
                dtype=jnp.int32,
            ),
        ),
        internal_outputs=(
            InternalOutputSpec(label="value", path=(0,)),
            InternalOutputSpec(label="support", path=(1,)),
        ),
        disposition=CoreExecutionDisposition.PLANNED,
    )


def read_outside_value(
    *,
    node: QueryNode,
    table: OutsideTable,
    live: BoolND,
    available: Mapping[str, PytreeValue],
    codes: FrameCodes,
) -> PolicyAnswer:
    """Gather an aligned read's value and support from its destination's table.

    The read takes the table at its projected grid cell and, for each kept group,
    at the node the group's origin frame names. The build proved the projected
    states to be grid points, so the nearest point is the cell; it absorbs only
    the rounding between the build's eager evaluation and the compiled one.

    Args:
        node: The aligned outside read.
        table: The destination's tabulated values and support.
        live: Whether the source cell is genuine rather than padding.
        available: Mapping of the names the read's projection and guard take to
            their values: projection operands, source states, period, age and the
            physical frames of the read's ancestors.
        codes: Mapping of each ancestor's query key to the node its selected row
            realises, per shock group.

    Returns:
        The read's answer, shaped as a point read's: the value, no action of its
        own, its support, its integrity and no reached landings. A NaN value, a
        cell whose problem failed its integrity check, fails the read's integrity.

    """
    read = cast("OutsideValueRead", node.outside_value)
    guard = live & _query_availability(node=node, available=available)
    mapped = _mapped_query_states(node=node, available=available, guard=guard)
    cell = (
        *(
            jnp.argmin(jnp.abs(jnp.asarray(points) - mapped[state]))
            for state, points in node.context.state_action_space.states.items()
        ),
        *(
            codes[origin][group]
            for group, origin in zip(read.groups, read.origins, strict=True)
        ),
    )
    values, support = table
    value = values[cell]
    integrity = ~jnp.any(jnp.isnan(value)) & _mapped_query_valid(
        node=node, mapped=mapped
    )
    return (
        jnp.where(guard, value, -jnp.inf),
        jnp.asarray(-1, dtype=jnp.int32),
        guard & support[cell],
        ~guard | integrity,
        jnp.zeros((len(_query_landing_checks(node=node)),), dtype=jnp.int32),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class OutsideValueTable:
    """Tabulate a destination's problem value at every grid cell and kept node.

    At each cell of the destination's grid the destination's action problem is
    evaluated as an outside read evaluates it at a point: each member's value at
    one common action, integrating every shock group except the kept ones, whose
    nodes become axes. A cell whose answer fails its integrity check is NaN, so a
    failure surfaces only where a reader takes that cell.
    """

    key: str
    """The query key of the destination's root Raw read."""
    definitions: Mapping[str, QueryNode] = field(repr=False)
    """The destination's bound Raw graph; omitted from the repr `jax.jit` takes."""
    kept: tuple[int, ...]
    """Plan columns of the groups kept as axes, in plan order."""
    width: int
    """Grid cells evaluated together per step."""

    def shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The shapes of the values and of the support.

        Both lead with the destination's state axes, in grid order, followed by one
        axis per kept group; the values end with the members' axis.
        """
        node = self.definitions[self.key]
        plan = node.context.current_iid_plans.get(node.read.period)
        axes = (
            *(
                values.size
                for values in node.context.state_action_space.states.values()
            ),
            *(
                int(cast("CurrentIIDPlan", plan).bindings[column][2].weights.size)
                for column in self.kept
            ),
        )
        return (*axes, *node.value_shape), axes

    def __call__(
        self,
        *,
        states: Mapping[StateName, FloatND | IntND],
        query_arguments: Mapping[str, Mapping[str, PytreeValue]],
    ) -> tuple[FloatND, BoolND, IntND]:
        """Return the values, the support and the reached-landing totals.

        Args:
            states: Mapping of the destination's state names to their grid points.
            query_arguments: Mapping of the root's key to its canonical arguments.

        Returns:
            The values and the support, shaped as `shapes` says, and the landing
            totals over every cell, one per declared landing check.

        """
        node = self.definitions[self.key]
        arguments = query_arguments[self.key]
        fixed = with_contracted_tables(
            node=node,
            arguments={
                name: value for name, value in arguments.items() if name != _PROJECTION
            },
        )
        projection = cast("Mapping[str, PytreeValue]", arguments[_PROJECTION])
        live = jnp.asarray(a=True)
        information = has_information(
            node=node, definitions=self.definitions, period=node.read.period
        )

        def answer(
            point: Mapping[StateName, FloatND | IntND],
        ) -> tuple[FloatND, BoolND, IntND]:
            result = (
                InformationQueryCell(
                    name=self.key,
                    node=node,
                    definitions=self.definitions,
                    period=node.read.period,
                    fixed=fixed,
                    projection_operands=projection,
                    information_arguments=MappingProxyType({}),
                    row_batch=_row_batch(chunk_size=self.width),
                    kept=self.kept,
                )(live=live, source=point, dependencies={})
                if information
                else _query_cell(
                    live,
                    point,
                    {},
                    node=node,
                    period=node.read.period,
                    fixed=fixed,
                    projection_operands=projection,
                )
            )
            return jnp.where(result[3], result[0], jnp.nan), result[2], result[4]

        value_shape, support_shape = self.shapes()
        if not states:
            values, support, landings = answer({})
            return values, support, _query_landing_totals(counts=landings[None])
        mesh = jnp.meshgrid(*states.values(), indexing="ij")
        points = {
            state: coordinate.reshape(-1)
            for state, coordinate in zip(states, mesh, strict=True)
        }
        values, support, landings = map_lanes(
            function=answer, operands=points, batch_size=self.width
        )
        return (
            values.reshape(value_shape),
            support.reshape(support_shape),
            _query_landing_totals(counts=landings),
        )


def _unaligned_destination(
    *,
    node: QueryNode,
    canonical_regimes: Mapping[RegimeName, Regime],
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> str | None:
    """Why the destination cannot serve stored values in the read's period."""
    target, period = node.read.target, node.read.period
    specialized = _age_specialized_states(
        regime=canonical_regimes[target], period=period
    )
    if node.read.phase != "solve" or not node.ancestors:
        return f"the read of {target!r} is not a solved outside read"
    if period not in value_periods_by_regime.get(target, ()):
        return f"{target!r} is not solved in period {period}"
    if specialized:
        return (
            f"{target!r} has age-specialized states {sorted(specialized)} in "
            f"period {period}"
        )
    if node.observation_sources:
        return (
            f"{target!r} observes shock groups {sorted(node.observation_sources)} "
            "through other queries"
        )
    uncarried = set(node.read.projection) - set(
        node.context.state_action_space.state_names
    )
    if uncarried:
        return f"{target!r} carries no states {sorted(uncarried)}"
    return None


def _unaligned_state(
    *,
    state: StateName,
    node: QueryNode,
    definitions: Mapping[str, QueryNode],
    canonical_regimes: Mapping[RegimeName, Regime],
) -> str | None:
    """Why the projected `state` is not provably a point of the destination's grid.

    The projection is composed over the root's inputs: its states, the period and
    its age, and the shocks of the frames above the read.
    """
    target = node.read.target
    if not _has_static_points(node.context.grids[state]):
        return f"the grid of state {state!r} of {target!r} is not fixed until run time"
    helper = node.read.projection[state]
    names = tuple(inspect.signature(helper).parameters)
    parameters = [
        node.read.operand_paths[name]
        for name in names
        if name in node.read.operand_paths
    ]
    if parameters:
        return f"the projection of state {state!r} reads parameter {parameters[0]!r}"
    root = definitions[node.ancestors[0]]
    unknown = sorted(
        set(names)
        - set(root.context.state_action_space.state_names)
        - {_PHYSICAL_CONTEXTS, "period", "age"}
    )
    if unknown:
        return (
            f"the projection of state {state!r} reads {unknown}, neither a state of "
            f"{root.read.target!r}, a shock nor the time"
        )
    try:
        shocks = _read_frame_shocks(
            state=state, helper=helper, names=names, definitions=definitions
        )
        if isinstance(shocks, str):
            return shocks
        return _unproved_points(
            state=state,
            node=node,
            projection=_flat_projection(helper=helper, names=names, shocks=shocks),
            names=(*(name for name in names if name != _PHYSICAL_CONTEXTS), *shocks),
            root=root,
            shocks=shocks,
            specialized=_age_specialized_states(
                regime=canonical_regimes[root.read.target], period=node.read.period
            ),
        )
    except Exception as error:  # noqa: BLE001 - an unanalysable projection is unproved
        return (
            f"the projection of state {state!r} cannot be analysed when the model is "
            f"built ({type(error).__name__}: {error})"
        )


def _read_frame_shocks(
    *,
    state: StateName,
    helper: Callable[..., PytreeValue],
    names: tuple[str, ...],
    definitions: Mapping[str, QueryNode],
) -> Mapping[str, _FrameShock] | str:
    """The frame shocks a projection reads, by input name; else why they are unknown."""
    cells = (
        frozenset()
        if _PHYSICAL_CONTEXTS not in names
        else _callable_reads(function=helper, definitions=definitions)
    )
    if cells is None:
        return (
            f"the projection of state {state!r} reads shocks that no query is known "
            "to realise"
        )
    shocks = {}
    for index, (owner, shock) in enumerate(sorted(cells)):
        found = _frame_shock(owner=owner, shock=shock, definitions=definitions)
        if found is None:
            return (
                f"the projection of state {state!r} reads shock {shock!r} without "
                "its nodes"
            )
        shocks[f"_lcm_frame_shock_{index}"] = found
    return shocks


def _unproved_points(  # noqa: PLR0911 - one refusal per failed proof obligation
    *,
    state: StateName,
    node: QueryNode,
    projection: Callable[..., PytreeValue],
    names: tuple[str, ...],
    root: QueryNode,
    shocks: Mapping[str, _FrameShock],
    specialized: frozenset[StateName],
) -> str | None:
    """Prove one projected state a grid point; None when proved, else the reason.

    - **Direct return.** The projection returns a root state unchanged, whose grid
      and points equal the destination's.
    - **Finite enumeration.** Every input has a finite domain known when the model
      is built: the points of a root state's fixed grid, the period and its age,
      and the codes of a discrete shock. Every combination, evaluated eagerly and
      adapted to the target state's role, is bitwise a point of the destination's
      grid in its dtype.
    """
    root_space = root.context.state_action_space
    period = node.read.period
    grid = node.context.grids[state]
    points = np.asarray(node.context.state_action_space.states[state])
    domains = {
        name: np.asarray([period], dtype=np.int32)
        if name == "period"
        else np.asarray(root.context.ages.values)[[period]]
        if name == "age"
        else _shock_values(frame_shock=shocks[name])
        if name in shocks
        else np.asarray(root_space.states[name])
        for name in names
    }
    returned = returned_input(
        func=projection,
        dtypes={name: domain.dtype for name, domain in domains.items()},
    )
    if (
        returned in root_space.state_names
        and returned not in specialized
        and _has_static_points(root.context.grids[returned])
        and _fingerprint((root.context.grids[returned],)) == _fingerprint((grid,))
        and domains[returned].dtype == points.dtype
        and np.array_equal(domains[returned], points)
    ):
        return None
    for name in names:
        if name in shocks and not isinstance(shocks[name].law, Discrete):
            return (
                f"the projection of state {state!r} reads the continuous shock "
                f"{shocks[name].shock!r}"
            )
        if name in root_space.state_names and (
            name in specialized or not _has_static_points(root.context.grids[name])
        ):
            return (
                f"the projection of state {state!r} reads {name!r}, whose points "
                "are not fixed when the model is built"
            )
    outputs = finite_outputs(func=projection, domains=domains)
    if outputs is None:
        return (
            f"the projection of state {state!r} has more than "
            f"{_MAX_FINITE_PROJECTION_CELLS} input combinations"
        )
    for _, output in outputs:
        value = np.asarray(
            _as_target_state(value=output, name=state, grids={state: grid})
        )
        if value.shape != () or value.dtype != points.dtype:
            return (
                f"the projection of state {state!r} gives a {value.dtype} array of "
                f"shape {value.shape}, not a point of the grid of "
                f"{node.read.target!r}"
            )
        if not np.any(points == value):
            return (
                f"the projection of state {state!r} lands at {value.item()!r}, "
                f"which is not a point of the grid of {node.read.target!r}"
            )
    return None


def _flat_projection(
    *,
    helper: Callable[..., PytreeValue],
    names: tuple[str, ...],
    shocks: Mapping[str, _FrameShock],
) -> Callable[..., PytreeValue]:
    """The projection with one scalar input per state, time and frame shock it reads.

    A frame shock reaches the projection inside the physical frames of the queries
    that realise it, so each becomes an input of its own. Every input enters as a
    JAX array, as it does in the solve.
    """

    def flat(**values: PytreeValue) -> PytreeValue:
        arrays = {name: jnp.asarray(value) for name, value in values.items()}
        frames: dict[str, dict[ShockName, PytreeValue]] = {}
        for operand, frame_shock in shocks.items():
            frames.setdefault(frame_shock.owner, {})[frame_shock.shock] = arrays[
                operand
            ]
        return helper(
            **{
                name: frames if name == _PHYSICAL_CONTEXTS else arrays[name]
                for name in names
            }
        )

    return flat


def _shock_values(*, frame_shock: _FrameShock) -> np.ndarray:
    """The values a shock takes in its frame.

    A discrete shock's codes, or the nodes of a continuous shock's group, which fix
    only the dtype.
    """
    if isinstance(frame_shock.law, Discrete):
        return np.unique(
            np.asarray(frame_shock.nodes.discrete_codes[frame_shock.shock])
        )
    return np.asarray(frame_shock.nodes.base_nodes).ravel()


def _frame_shock(
    *, owner: str, shock: ShockName, definitions: Mapping[str, QueryNode]
) -> _FrameShock | None:
    """The law of a shock in a query's frame and the nodes that realise it there."""
    node = definitions[owner]
    for group, binding in _bindings(
        node=node, definitions=definitions, period=node.read.period
    ).items():
        if shock in binding.group.shocks:
            nodes = _group_nodes(key=owner, group=group, definitions=definitions)
            return (
                None
                if nodes is None
                else _FrameShock(
                    owner=owner,
                    shock=shock,
                    law=binding.group.shocks[shock],
                    nodes=nodes,
                )
            )
    return None


def _kept_groups(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> tuple[tuple[ShockGroupName, str], ...] | str:
    """The destination's groups fixed by its caller, each with its origin.

    A group none of whose shocks the caller observes is integrated by the
    destination's problem. A partly observed group, or a fixed group whose nodes
    no ancestor enumerates, leaves the read unaligned; the reason is returned. So
    does a frame whose copies the build refuses, which the query build then reports.
    """
    target = definitions[key].read.target
    try:
        return _kept_group_origins(key=key, definitions=definitions)
    except (RegimeInitializationError, ValueError) as error:
        return (
            f"the caller's frame does not fix the shock groups of {target!r}: {error}"
        )


def _kept_group_origins(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> tuple[tuple[ShockGroupName, str], ...] | str:
    node = definitions[key]
    target = node.read.target
    plan = node.context.current_iid_plans.get(node.read.period)
    if plan is None:
        return ()
    sources = _parent_sources(
        parent=definitions[node.ancestors[-1]], definitions=definitions
    )
    pinned = _pinned_columns(name=key, definitions=definitions)
    observed = node.read.price_information & sources.keys()
    kept = []
    for column, (group, binding, nodes) in enumerate(plan.bindings):
        shocks = frozenset(binding.group.shocks)
        if column in pinned:
            origin = _group_origin(key=key, shocks=shocks, definitions=definitions)
            if origin is None or not _same_nodes(
                left=nodes,
                right=_group_nodes(
                    key=origin[0], group=origin[1], definitions=definitions
                ),
            ):
                return (
                    f"no ancestor enumerates the nodes of shock group {group!r} of "
                    f"{target!r}"
                )
            kept.append((group, origin[0]))
        elif shocks & observed:
            return f"{target!r} has a partly observed shock group {group!r}"
    return tuple(kept)


def _group_origin(
    *, key: str, shocks: frozenset[ShockName], definitions: Mapping[str, QueryNode]
) -> tuple[str, ShockGroupName] | None:
    """The ancestor whose frame enumerates a group that `key` copies from its caller.

    Walk up the ancestors:
    - a group the ancestor copies whole from its own caller keeps the walk going;
    - a group the ancestor's own plan enumerates, or one it observes through another
      query, makes that ancestor the origin.

    None when the shocks do not form one group of a caller's frame.
    """
    current = key
    while definitions[current].ancestors:
        parent_key = definitions[current].ancestors[-1]
        parent = definitions[parent_key]
        sources = _parent_sources(parent=parent, definitions=definitions)
        groups = {sources[shock][0] for shock in shocks if shock in sources}
        if not shocks <= sources.keys() or len(groups) != 1:
            return None
        (group,) = groups
        if group in parent.observation_sources:
            return parent_key, group
        plan = parent.context.current_iid_plans.get(parent.read.period)
        columns = [
            column
            for column, (name, _, _) in enumerate(() if plan is None else plan.bindings)
            if name == group
        ]
        if not columns:
            return None
        if columns[0] not in _pinned_columns(name=parent_key, definitions=definitions):
            return parent_key, group
        current = parent_key
    return None


def _group_nodes(
    *, key: str, group: ShockGroupName, definitions: Mapping[str, QueryNode]
) -> NodeSet | None:
    """The nodes of `group` in the frame of query `key`, own or observed."""
    node = definitions[key]
    producer = definitions[node.observation_sources.get(group, key)]
    plan = producer.context.current_iid_plans.get(node.read.period)
    bindings = () if plan is None else plan.bindings
    return next((nodes for name, _, nodes in bindings if name == group), None)


def _same_nodes(*, left: NodeSet, right: NodeSet | None) -> bool:
    """Whether two node sets realise the same nodes in the same order."""
    return (
        right is not None
        and np.array_equal(np.asarray(left.base_nodes), np.asarray(right.base_nodes))
        and np.shape(left.weights) == np.shape(right.weights)
        and left.discrete_codes.keys() == right.discrete_codes.keys()
        and all(
            np.array_equal(np.asarray(codes), np.asarray(right.discrete_codes[shock]))
            for shock, codes in left.discrete_codes.items()
        )
    )


def _has_static_points(grid: Grid) -> bool:
    """Whether a grid's points are fixed when the model is built."""
    if type(grid) is IrregSpacedGrid:
        return not grid.pass_points_at_runtime
    return type(grid) in _STATIC_GRIDS


@dataclass(frozen=True, kw_only=True)
class _TableArguments:
    """The destination's grid points and its root read's canonical arguments."""

    key: str
    builder: _QueryArguments

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        return MappingProxyType(
            {
                "states": self.builder.node.context.state_action_space.states,
                _QUERY_ARGUMENTS: cast(
                    "PytreeValue", MappingProxyType({self.key: self.builder(context)})
                ),
            }
        )


def _age_specialized_states(*, regime: Regime, period: int) -> frozenset[StateName]:
    """States with their own nodes at `period`, which the solve overlays on the grid.

    A query's context holds the representative age's grid, so such a state's
    projected values cannot be compared with it.
    """
    axes = regime.solution.period_state_axes
    return frozenset(() if axes is None else axes.get(period, {}))
