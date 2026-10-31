"""Finite physical information support for canonical typed policy producers.

Each producer prices its own selected policy over its hidden rows. Ancestor
information axes remain transient native outputs until their actual caller
selects a policy; only the occupant root integrates into a durable value bank.
"""

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NoReturn, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDDrawFrame,
    CurrentIIDPlan,
    NodeOperands,
)
from _lcm.policy_queries import (
    _ROW_BATCH,
    PolicyAnswer,
    QueryKind,
    QueryNode,
    StayTable,
)
from _lcm.probability import is_represented_zero
from _lcm.shock_registry import ShockBinding
from _lcm.typing import PytreeValue, QAndFKwargs
from _lcm.utils.counts import sum_counts
from _lcm.utils.dispatchers import vary_carry_like_body
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import RegimeInitializationError
from lcm.shocks import (
    Discrete,
    Law,
    LogNormal,
    NodeSet,
    Normal,
    NormalMixture,
    ShockGroupName,
    ShockName,
    Uniform,
)
from lcm.typing import BoolND, FloatND, IntND

type SourceStates = Mapping[str, FloatND | IntND]
type Frames = Mapping[str, CurrentIIDDrawFrame]
type QueryArguments = Mapping[str, PytreeValue]
type Dependencies = Mapping[str, PolicyAnswer]


def support_size(*, node: QueryNode, period: int) -> int:
    """Return the declared finite row count without materializing physical draws."""
    plan = node.context.current_iid_plans.get(period)
    size = 1 if plan is None else plan.indices.shape[0]
    return size * math.prod(node.observation_counts)


def _has_local_information(*, node: QueryNode, period: int) -> bool:
    return (
        bool(node.observation_sources)
        or node.context.current_iid_plans.get(period) is not None
    )


def support_shape(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> tuple[int, ...]:
    """Name only ancestor information axes that have a numerical node operator."""
    if period != node.read.period:
        raise ValueError("Information shape needs the query's actual period.")
    return tuple(
        support_size(node=definitions[name], period=definitions[name].read.period)
        for name in node.ancestors
        if _has_local_information(
            node=definitions[name], period=definitions[name].read.period
        )
    )


def has_information(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> bool:
    """Whether this producer or its producing context has current node support."""
    if period != node.read.period:
        raise ValueError("Information demand needs the query's actual period.")
    return any(
        _has_local_information(node=candidate, period=candidate.read.period)
        for candidate in (node, *(definitions[name] for name in node.ancestors))
    )


def _row(*, frame: CurrentIIDDrawFrame, row: jax.Array) -> CurrentIIDDrawFrame:
    """Select physical/address leaves without indexing scalar law metadata."""
    return frame._replace(
        physical={name: value[row] for name, value in frame.physical.items()},
        addressed_bases={
            name: value[row] for name, value in frame.addressed_bases.items()
        },
        held_bases={name: value[row] for name, value in frame.held_bases.items()},
        recovery_error={
            name: value[row] for name, value in frame.recovery_error.items()
        },
        source_params=jax.tree.map(lambda value: value[row], frame.source_params),
        source_inputs=jax.tree.map(lambda value: value[row], frame.source_inputs),
        valid=frame.valid[row],
        information_available={
            name: value[row] for name, value in frame.information_available.items()
        },
    )


def _empty_frame() -> CurrentIIDDrawFrame:
    return CurrentIIDDrawFrame({}, {}, {}, {}, {}, {}, jnp.asarray(a=True))


def _batch_metadata(*, frame: CurrentIIDDrawFrame, size: int) -> CurrentIIDDrawFrame:
    def broadcast(value: FloatND | IntND | BoolND) -> FloatND | IntND | BoolND:
        value = jnp.asarray(value)
        return jnp.broadcast_to(value, (size, *value.shape))

    return frame._replace(
        source_params=jax.tree.map(broadcast, frame.source_params),
        source_inputs=jax.tree.map(broadcast, frame.source_inputs),
        valid=jnp.broadcast_to(frame.valid, (size,)),
        information_available=jax.tree.map(broadcast, frame.information_available),
    )


def _bindings(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> Mapping[ShockGroupName, ShockBinding]:
    plan = node.context.current_iid_plans.get(period)
    bindings = (
        {} if plan is None else {name: binding for name, binding, _ in plan.bindings}
    )
    for group, producer in node.observation_sources.items():
        target = definitions[producer]
        target_plan = target.context.current_iid_plans[target.read.period]
        if target_plan is None:
            raise ValueError("An observed group requires its producing law plan.")
        bindings[group] = next(
            binding for name, binding, _ in target_plan.bindings if name == group
        )
    return bindings


def _parent_sources(
    *, parent: QueryNode, definitions: Mapping[str, QueryNode]
) -> Mapping[ShockName, tuple[ShockGroupName, ShockBinding, bool]]:
    """Each shock of a parent row frame, its producing law, and whether observed."""
    return {
        shock: (group, binding, group in parent.observation_sources)
        for group, binding in _bindings(
            node=parent, definitions=definitions, period=parent.read.period
        ).items()
        for shock in binding.group.shocks
        if group not in parent.observation_sources
        or shock in parent.read.policy_information
    }


def _refuse_pinning(*, name: str, group: ShockGroupName, reason: str) -> NoReturn:
    raise RegimeInitializationError(
        f"Policy query {name!r} observes shock group {group!r} from its caller, "
        f"but its rows are not provably copies of one point: {reason}"
    )


def _static_informative(*, law: Law) -> bool | None:
    """Whether a law's spread fields are literals giving it a nondegenerate spread."""
    spread = (
        ("sigma",)
        if isinstance(law, Normal | LogNormal)
        else ("start", "stop")
        if isinstance(law, Uniform)
        else ()
    )
    values = [getattr(law, field) for field in spread]
    if not values or not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in values
    ):
        return None
    return values[0] > 0 if len(values) == 1 else values[1] > values[0]


def _pinned_columns(
    *, name: str, definitions: Mapping[str, QueryNode]
) -> tuple[int, ...]:
    """Plan columns whose every shock the caller's row frame fixes.

    A partly observed group keeps its row loop: its rows genuinely differ. A fully
    observed group repeats one physical point in every row:
    - an available informative observation pins each row's base to the held base;
    - an unavailable observation leaves the rows free, but this node is then one
      of the observation's producing aliases and is itself unavailable;
    - an uninformative caller law leaves the rows free. An observed group's law
      here is the one its aliases proved equal to the producer's. A caller's own
      group is pinned when the caller's spread is a positive literal, the law here
      a zero literal, or both the identical law without state inputs; otherwise
      the rows can differ and the model is refused.
    """
    from _lcm.shock_projection import _law_record  # noqa: PLC0415

    node = definitions[name]
    plan = node.context.current_iid_plans.get(node.read.period)
    if plan is None or not node.ancestors:
        return ()
    parent = definitions[node.ancestors[-1]]
    sources = _parent_sources(parent=parent, definitions=definitions)
    observed = node.read.price_information & sources.keys()
    columns = []
    for column, (group, binding, _) in enumerate(plan.bindings):
        shocks = binding.group.shocks
        if not shocks.keys() <= observed:
            continue
        for shock, law in shocks.items():
            source_group, source, observation = sources[shock]
            if observation:
                if name not in parent.observation_aliases[source_group]:
                    _refuse_pinning(
                        name=name,
                        group=group,
                        reason="the query is not a producing alias of the "
                        "observation, so it can run when the observation is "
                        "unavailable. Supporting it needs a representative "
                        "chosen per cell at run time.",
                    )
                continue
            if (
                isinstance(law, Discrete | NormalMixture)
                or _static_informative(law=source.group.shocks[shock]) is True
                or _static_informative(law=law) is False
            ):
                continue
            record = _law_record(
                shock=shock, group=group, binding=binding, parameter_paths={}
            )
            if not isinstance(law, Normal | LogNormal | Uniform) or (
                record[1]
                or record
                != _law_record(
                    shock=shock,
                    group=source_group,
                    binding=source,
                    parameter_paths={},
                )
            ):
                _refuse_pinning(
                    name=name,
                    group=group,
                    reason=f"the law of {shock!r} is not the caller's state-free "
                    "law, so an uninformative caller law need not make it "
                    "degenerate here. Supporting it needs a proof that both laws "
                    "share their informativeness.",
                )
        columns.append(column)
    return tuple(columns)


def pinned_support(
    *, name: str, definitions: Mapping[str, QueryNode]
) -> tuple[IntND, IntND] | None:
    """Representative support rows and the gather spreading them over every row.

    Rows differing only in fully observed groups are copies, so one row with
    those group indices at zero stands for all of them. Observation axes stay.
    """
    pinned = _pinned_rows(name=name, definitions=definitions)
    if pinned is None:
        return None
    return jnp.asarray(pinned[0], jnp.int32), jnp.asarray(pinned[1], jnp.int32)


def _pinned_rows(
    *, name: str, definitions: Mapping[str, QueryNode]
) -> tuple[np.ndarray, np.ndarray] | None:
    """`pinned_support` as host arrays, for indexing at build and trace time."""
    node = definitions[name]
    plan = node.context.current_iid_plans.get(node.read.period)
    columns = _pinned_columns(name=name, definitions=definitions)
    if plan is None or not columns:
        return None
    sizes = tuple(nodes.weights.size for _, _, nodes in plan.bindings)
    codes = np.indices(sizes).reshape(len(sizes), -1)
    codes[list(columns)] = 0
    representatives, gather = np.unique(
        np.ravel_multi_index(tuple(codes), sizes), return_inverse=True
    )
    extra = math.prod(node.observation_counts)
    offsets = np.arange(extra)
    return (
        (representatives[:, None] * extra + offsets).ravel(),
        (gather.ravel()[:, None] * extra + offsets).ravel(),
    )


# A shock of one query's information frame, named by the query's key.
type _FrameCell = tuple[str, ShockName]


def _plan(*, node: QueryNode) -> CurrentIIDPlan | None:
    return node.context.current_iid_plans.get(node.read.period)


def _frame_shocks(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> frozenset[ShockName]:
    """Shocks a query's information frame carries to its descendants."""
    node = definitions[key]
    if not _has_local_information(node=node, period=node.read.period):
        return frozenset()
    return frozenset(_parent_sources(parent=node, definitions=definitions))


def _callable_reads(
    *, function: Callable[..., PytreeValue], definitions: Mapping[str, QueryNode]
) -> frozenset[_FrameCell] | None:
    """Frame shocks a bound projection or guard reads; None when not attributable.

    A composed helper reads its source query's frame through its own signature
    and through the parent projection helpers it applies first. Any other
    callable that takes the physical frames could read anything.
    """
    from _lcm.exit_queries import _Available  # noqa: PLC0415
    from _lcm.policy_queries import _PHYSICAL_CONTEXTS  # noqa: PLC0415
    from _lcm.utils.functools import ComposedProjection  # noqa: PLC0415

    target = inspect.unwrap(function)
    if isinstance(target, _Available):
        helpers: tuple[Callable[..., PytreeValue], ...] = target.predicates
        cells: frozenset[_FrameCell] = frozenset()
    elif isinstance(target, ComposedProjection):
        frame = _frame_shocks(key=target.source, definitions=definitions)
        helpers = tuple(target.projection.values())
        cells = frozenset(
            (target.source, name)
            for name in inspect.signature(target.function).parameters
            if name in frame
        )
    elif _PHYSICAL_CONTEXTS in inspect.signature(function).parameters:
        return None
    else:
        return frozenset()
    for helper in helpers:
        inner = _callable_reads(function=helper, definitions=definitions)
        if inner is None:
            return None
        cells |= inner
    return cells


def _guard_reads(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> frozenset[_FrameCell] | None:
    available = definitions[key].available
    if available is None:
        return frozenset()
    return _callable_reads(function=available, definitions=definitions)


def _projection_reads(
    *,
    key: str,
    definitions: Mapping[str, QueryNode],
    states: frozenset[str] | None = None,
) -> frozenset[_FrameCell] | None:
    cells: frozenset[_FrameCell] = frozenset()
    for state, helper in definitions[key].read.projection.items():
        if states is not None and state not in states:
            continue
        inner = _callable_reads(function=helper, definitions=definitions)
        if inner is None:
            return None
        cells |= inner
    return cells


def _copied_cells(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> frozenset[_FrameCell]:
    """Parent frame cells whose values a query's own rows copy."""
    node = definitions[key]
    plan = _plan(node=node)
    if plan is None or not node.ancestors:
        return frozenset()
    parent = node.ancestors[-1]
    own = frozenset(
        shock for _, binding, _ in plan.bindings for shock in binding.group.shocks
    )
    return frozenset(
        (parent, shock)
        for shock in own
        & node.read.price_information
        & _frame_shocks(key=parent, definitions=definitions)
    )


def _reads(
    *,
    key: str,
    definitions: Mapping[str, QueryNode],
    states: frozenset[str] | None,
    copies: bool = True,
) -> frozenset[_FrameCell] | None:
    node = definitions[key]
    parts = [
        _projection_reads(key=key, definitions=definitions, states=states),
        _guard_reads(key=key, definitions=definitions),
        *(
            reads
            for producers in node.observation_aliases.values()
            for producer in producers
            for reads in (
                _projection_reads(key=producer, definitions=definitions),
                _guard_reads(key=producer, definitions=definitions),
            )
        ),
    ]
    if any(part is None for part in parts):
        return None
    return frozenset().union(
        *cast("list[frozenset[_FrameCell]]", parts),
        _copied_cells(key=key, definitions=definitions) if copies else (),
    )


def _own_reads(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> frozenset[_FrameCell] | None:
    """Frame cells one query's own evaluation reads, its descendants aside.

    - its projection and guard, through their composed helpers;
    - the parent frame's values of the shocks its own rows copy;
    - the projections and guards of the producers of its target-only
      observations, which realise those rows in its cell.

    Its Q, its decision and its laws read only its own mapped states, which
    the projection covers, and its own frame.
    """
    return _reads(key=key, definitions=definitions, states=None)


def _frame_determinants(
    *, key: str, definitions: Mapping[str, QueryNode]
) -> frozenset[_FrameCell] | None:
    """Frame cells that fix every value of a query's own frame below its caller.

    A frame row's physical values follow from its node and the law parameters
    of the states its laws read; a false guard maps those states to a stand-in.
    A copied shock's value is its parent's, which `read_columns` follows shock
    by shock. A frame's validity flag may also follow other copied values, but
    the frame's own query reports that flag at every row it evaluates.
    """
    plan = _plan(node=definitions[key])
    law_states = frozenset(
        state
        for _, binding, _ in (() if plan is None else plan.bindings)
        for state in binding.state_inputs
    )
    return _reads(key=key, definitions=definitions, states=law_states, copies=False)


def read_columns(
    *, name: str, ancestor: str, definitions: Mapping[str, QueryNode]
) -> frozenset[ShockName]:
    """Shocks of an ancestor's frame that a query or any of its descendants reads.

    The read closes over:
    - each query's own reads (`_own_reads`), for the query and its descendants;
    - the guards of every query between the ancestor and them, which their
      cells evaluate;
    - for a frame cell of a query below the ancestor, what fixes that frame
      (`_frame_determinants`).

    An unattributable read anywhere makes every ancestor shock read.
    """
    every = _frame_shocks(key=ancestor, definitions=definitions)
    subtree = (
        name,
        *(key for key, node in definitions.items() if name in node.ancestors),
    )
    pending: list[_FrameCell] = []
    for key in subtree:
        chain = definitions[key].ancestors
        if ancestor not in chain:
            return every
        between = chain[chain.index(ancestor) + 1 :]
        for part in (
            _own_reads(key=key, definitions=definitions),
            *(_guard_reads(key=step, definitions=definitions) for step in between),
        ):
            if part is None:
                return every
            pending.extend(part)
    seen: set[_FrameCell] = set()
    expanded: set[str] = set()
    while pending:
        cell = pending.pop()
        if cell in seen:
            continue
        seen.add(cell)
        owner = cell[0]
        if owner == ancestor or ancestor not in definitions[owner].ancestors:
            continue
        pending.extend(
            copied
            for copied in _copied_cells(key=owner, definitions=definitions)
            if copied[1] == cell[1]
        )
        if owner in expanded:
            continue
        expanded.add(owner)
        determinants = _frame_determinants(key=owner, definitions=definitions)
        if determinants is None:
            return every
        pending.extend(determinants)
    return frozenset(shock for owner, shock in seen if owner == ancestor) & every


def _shock_codes(
    *, binding: ShockBinding, nodes: NodeSet, shock: ShockName
) -> np.ndarray:
    """Per group row, an integer naming the shock's node.

    A continuous shock's value may mix every continuous base of its group, so it
    is named by all of them.
    """
    shocks = binding.group.shocks
    if isinstance(shocks[shock], Discrete):
        return np.asarray(nodes.discrete_codes[shock])
    continuous = [
        axis
        for axis, law in enumerate(shocks.values())
        if not isinstance(law, Discrete)
    ]
    base = np.asarray(nodes.base_nodes)[:, continuous]
    return np.unique(base, axis=0, return_inverse=True)[1].ravel()


def _support_codes(
    *, key: str, definitions: Mapping[str, QueryNode], shocks: tuple[ShockName, ...]
) -> np.ndarray:
    """Per support row of a query, the node code of each named frame shock."""
    node = definitions[key]
    plan = _plan(node=node)
    extents = (1 if plan is None else plan.indices.shape[0], *node.observation_counts)
    bindings = _bindings(node=node, definitions=definitions, period=node.read.period)
    own = () if plan is None else plan.bindings
    indices = np.zeros((1, 0), dtype=np.int32) if plan is None else plan.indices
    columns = []
    for shock in shocks:
        column = None
        for position, (_, binding, nodes) in enumerate(own):
            if shock in binding.group.shocks:
                codes = _shock_codes(binding=binding, nodes=nodes, shock=shock)
                column = (0, codes[np.asarray(indices)[:, position]])
                break
        for axis, (group, producer) in enumerate(
            node.observation_sources.items(), start=1
        ):
            if column is None and shock in bindings[group].group.shocks:
                nodes = next(
                    support
                    for name, _, support in cast(
                        "CurrentIIDPlan", _plan(node=definitions[producer])
                    ).bindings
                    if name == group
                )
                column = (
                    axis,
                    _shock_codes(binding=bindings[group], nodes=nodes, shock=shock),
                )
        if column is None:
            raise ValueError(f"Shock {shock!r} is not in query {key!r}'s frame.")
        axis, codes = column
        shape = [1] * len(extents)
        shape[axis] = codes.size
        columns.append(np.broadcast_to(codes.reshape(shape), extents).ravel())
    if not columns:
        return np.zeros((math.prod(extents), 0), dtype=np.int64)
    return np.stack(columns, axis=1)


def _row_classes(
    *, key: str, shocks: frozenset[ShockName], definitions: Mapping[str, QueryNode]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Classes of a query's row positions that agree on the named frame shocks.

    The row positions are those its descendants see: its pinned
    representatives, else every support row. Returns each class's first row
    position and the class of every row position, or None when every row
    position is its own class.
    """
    node = definitions[key]
    if shocks >= _frame_shocks(key=key, definitions=definitions):
        return None
    # The cell reads its representatives through `pinned_support`; follow it.
    pinned = (
        None
        if pinned_support(name=key, definitions=definitions) is None
        else _pinned_rows(name=key, definitions=definitions)
    )
    rows = (
        np.arange(support_size(node=node, period=node.read.period))
        if pinned is None
        else pinned[0]
    )
    codes = _support_codes(
        key=key, definitions=definitions, shocks=tuple(sorted(shocks))
    )[rows]
    if codes.shape[1] == 0:
        return np.zeros(1, dtype=np.int32), np.zeros(rows.size, dtype=np.int32)
    _, first, classes = np.unique(codes, axis=0, return_index=True, return_inverse=True)
    if first.size == rows.size:
        return None
    return first.astype(np.int32), classes.ravel().astype(np.int32)


def ancestor_classes(
    *, name: str, ancestor: str, definitions: Mapping[str, QueryNode]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Classes of an ancestor's rows that a query cannot tell apart.

    Two rows share a class when every shock the query reads (`read_columns`)
    takes the same node. See `_row_classes` for the result.
    """
    node = definitions[ancestor]
    if not _has_local_information(node=node, period=node.read.period):
        return None
    return _row_classes(
        key=ancestor,
        shocks=read_columns(name=name, ancestor=ancestor, definitions=definitions),
        definitions=definitions,
    )


def stay_read_shocks(
    *, name: str, definitions: Mapping[str, QueryNode]
) -> frozenset[ShockName] | None:
    """Own frame shocks a query's action problem reads; None if not attributable.

    The action problem reads shocks by name: through its Q/F function and through
    projection helpers that take the query's own physical inputs.
    """
    from _lcm.policy_queries import _PHYSICAL_CONTEXTS  # noqa: PLC0415

    node = definitions[name]
    frame = _frame_shocks(key=name, definitions=definitions)
    function = node.context.Q_and_F_functions[node.read.period]
    parameters = set(inspect.signature(function).parameters)
    if _PHYSICAL_CONTEXTS in parameters:
        return None
    for helper in node.read.projection.values():
        parameters |= set(inspect.signature(helper).parameters)
    return frozenset(parameters) & frame


def own_stay_classes(
    *, name: str, definitions: Mapping[str, QueryNode]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Classes of a query's own rows its action problem cannot tell apart.

    Only an unguarded policy query qualifies: its stay table is then the same
    at every row of a class, and its exit decision still runs at every row. A
    guarded query's guard may read any of its rows. See `_row_classes` for the
    result.
    """
    node = definitions[name]
    if (
        node.available is not None
        or node.read.kind
        not in {QueryKind.RAW, QueryKind.PARTICIPATING, QueryKind.FULL}
        or not _has_local_information(node=node, period=node.read.period)
    ):
        return None
    shocks = stay_read_shocks(name=name, definitions=definitions)
    if shocks is None:
        return None
    return _row_classes(key=name, shocks=shocks, definitions=definitions)


@dataclass(frozen=True, kw_only=True)
class InformationQueryCell:
    """One scalar source cell of an existing native policy-query producer."""

    name: str
    node: QueryNode
    definitions: Mapping[str, QueryNode]
    period: int
    fixed: QueryArguments
    projection_operands: QueryArguments
    information_arguments: Mapping[str, QueryArguments]
    realised_root: str | None = None
    root_frame: CurrentIIDDrawFrame | None = None
    row_batch: int = _ROW_BATCH
    """Support rows evaluated together per step."""

    def _size(self, *, name: str, node: QueryNode) -> int:
        return (
            1
            if name == self.realised_root
            else support_size(node=node, period=node.read.period)
        )

    def _pinned(self, *, name: str) -> tuple[IntND, IntND] | None:
        """The representatives of `name`'s rows and the gather spreading them."""
        if name == self.realised_root:
            return None
        return pinned_support(name=name, definitions=self.definitions)

    def _axis(self, *, name: str, node: QueryNode) -> int:
        """Rows `name` gives its descendants' answers: one per representative."""
        pinned = self._pinned(name=name)
        return self._size(name=name, node=node) if pinned is None else pinned[0].size

    def _classes(
        self, *, name: str, ancestor: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Classes of `ancestor`'s row positions that query `name` reads alike."""
        if ancestor == self.realised_root:
            return None
        return ancestor_classes(
            name=name, ancestor=ancestor, definitions=self.definitions
        )

    def _class_axis(self, *, ancestor: str) -> int:
        """Entries this query's answer keeps for one ancestor: one per class."""
        classes = self._classes(name=self.name, ancestor=ancestor)
        if classes is None:
            return self._axis(name=ancestor, node=self.definitions[ancestor])
        return classes[0].size

    def _dependency_index(
        self, *, producer: str, indices: tuple[IntND, ...], at: IntND | None
    ) -> tuple[IntND, ...]:
        """Map this query's positions to a producer's entries.

        Each position names a class of this query (or a row position when it has
        none); the producer's classes are no finer, so the class's first row
        position finds the producer's entry. `at` is this query's own row
        position, absent when it has no rows of its own.
        """
        axes = tuple(
            name
            for name in self.node.ancestors
            if _has_local_information(
                node=self.definitions[name],
                period=self.definitions[name].read.period,
            )
        )
        mapped = []
        for ancestor, index in zip(axes, indices, strict=True):
            mine = self._classes(name=self.name, ancestor=ancestor)
            theirs = self._classes(name=producer, ancestor=ancestor)
            position = index if mine is None else jnp.asarray(mine[0])[index]
            mapped.append(
                position if theirs is None else jnp.asarray(theirs[1])[position]
            )
        if at is not None:
            theirs = self._classes(name=producer, ancestor=self.name)
            mapped.append(at if theirs is None else jnp.asarray(theirs[1])[at])
        return tuple(mapped)

    def _map_support(
        self,
        *,
        point: Callable[[IntND, IntND], PolicyAnswer],
        name: str,
        node: QueryNode,
        spread: bool,
    ) -> PolicyAnswer:
        """Evaluate each distinct support row once, at its row and its axis position.

        With `spread`, the answers are spread over every copy; otherwise they keep
        one entry per representative, the axis a descendant's answer carries. An
        ancestor's rows that this query reads alike share one entry, evaluated
        at the class's first row.
        """
        from _lcm.policy_queries import map_lanes  # noqa: PLC0415

        pinned = self._pinned(name=name)
        classes = None if spread else self._classes(name=self.name, ancestor=name)
        if classes is not None:
            first = classes[0]
            rows = jnp.asarray(
                first
                if pinned is None
                else cast(
                    "tuple[np.ndarray, np.ndarray]",
                    _pinned_rows(name=name, definitions=self.definitions),
                )[0][first],
                dtype=jnp.int32,
            )
            return map_lanes(
                function=lambda position: point(rows[position], position),
                operands=jnp.arange(first.size, dtype=jnp.int32),
                batch_size=self.row_batch,
            )
        positions = jnp.arange(self._axis(name=name, node=node), dtype=jnp.int32)
        if pinned is None:
            return map_lanes(
                function=lambda row: point(row, row),
                operands=positions,
                batch_size=self.row_batch,
            )
        representatives, gather = pinned
        answers = map_lanes(
            function=lambda position: point(representatives[position], position),
            operands=positions,
            batch_size=self.row_batch,
        )
        return jax.tree.map(lambda leaf: leaf[gather], answers) if spread else answers

    def __call__(
        self, *, live: BoolND, source: SourceStates, dependencies: Dependencies
    ) -> PolicyAnswer:
        """Retain caller axes, reduce only this producer's hidden node support."""
        return self._ancestors(
            position=0,
            frames={},
            indices=(),
            live=live,
            source=source,
            dependencies=dependencies,
        )

    def _available(
        self,
        *,
        node: QueryNode,
        fixed: QueryArguments,
        source: SourceStates,
        frames: Frames,
    ) -> BoolND:
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _PROJECTION,
            _query_availability,
        )

        if node.available is None:
            return jnp.asarray(a=True)
        available = {
            **(
                self.projection_operands
                if node is self.node
                else cast("QueryArguments", fixed[_PROJECTION])
            ),
            **source,
            "period": fixed["period"],
            "age": fixed["age"],
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }
        return _query_availability(node=node, available=available)

    def _realise(
        self,
        *,
        node: QueryNode,
        fixed: QueryArguments,
        source: SourceStates,
        frames: Frames,
        guard: BoolND,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _PROJECTION,
            _mapped_query_states,
        )

        key = next(
            name for name, candidate in self.definitions.items() if candidate is node
        )
        if key == self.realised_root:
            if self.root_frame is None:
                raise ValueError(
                    "A realised policy query needs its original physical frame."
                )
            observed = node.read.policy_information
            missing = observed - self.root_frame.physical.keys()
            if missing:
                raise ValueError(
                    f"Unrevealed required policy information: {sorted(missing)!r}."
                )
            frame = self.root_frame._replace(
                physical={
                    name: value[None]
                    for name, value in self.root_frame.physical.items()
                    if name in observed
                },
                addressed_bases={
                    name: value[None]
                    for name, value in self.root_frame.addressed_bases.items()
                    if name in observed
                },
                held_bases={
                    name: value[None]
                    for name, value in self.root_frame.held_bases.items()
                    if name in observed
                },
                recovery_error={
                    name: value[None]
                    for name, value in self.root_frame.recovery_error.items()
                    if name in observed
                },
            )
            return _batch_metadata(frame=frame, size=1), jnp.ones((1, 1))
        available = {
            **(
                self.projection_operands
                if node is self.node
                else cast("QueryArguments", fixed[_PROJECTION])
            ),
            **source,
            "period": fixed["period"],
            "age": fixed["age"],
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }
        mapped = _mapped_query_states(node=node, available=available, guard=guard)
        plan = node.context.current_iid_plans.get(node.read.period)
        parent = node.ancestors[-1] if node.ancestors else None
        previous = None if parent is None else frames[parent]
        if plan is None:
            frame, factors = _empty_frame(), jnp.ones((1, 1))
        else:
            frame, factors = plan.realise_information_nodes(
                arguments=cast("QAndFKwargs", {**fixed, **mapped}),
                group_params=cast("AdmittedGroupParams", fixed[CURRENT_IID_ARGUMENT]),
                node_operands=cast("NodeOperands", fixed[CURRENT_IID_NODES_ARGUMENT]),
                indices=cast("IntND", fixed[CURRENT_IID_ROWS_ARGUMENT]),
                observation=previous,
                observed=frozenset(
                    node.read.price_information
                    & (frozenset() if previous is None else previous.physical.keys())
                ),
                source_bindings={}
                if parent is None
                else _bindings(
                    node=self.definitions[parent],
                    definitions=self.definitions,
                    period=self.definitions[parent].read.period,
                ),
            )
        if not node.observation_sources:
            return (
                frame
                if plan is None
                else _batch_metadata(frame=frame, size=factors.shape[1]),
                factors,
            )
        return self._observations(
            node=node,
            primary=_batch_metadata(frame=frame, size=factors.shape[1]),
            factors=factors,
            source=source,
            frames=frames,
        )

    def _observations(
        self,
        *,
        node: QueryNode,
        primary: CurrentIIDDrawFrame,
        factors: FloatND,
        source: SourceStates,
        frames: Frames,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        key = next(
            name for name, candidate in self.definitions.items() if candidate is node
        )
        extents = (factors.shape[1], *node.observation_counts)

        def point(flat: IntND) -> tuple[CurrentIIDDrawFrame, FloatND]:
            rows = jnp.unravel_index(flat, extents)
            frame = _row(frame=primary, row=rows[0])
            terms = [factors[:, rows[0]]]
            for axis, (group, producer) in enumerate(
                node.observation_sources.items(), start=1
            ):
                target = self.definitions[producer]
                fixed = self.information_arguments[producer]
                chosen, weights = self._observer_row(
                    observer=node,
                    target=target,
                    group=group,
                    fixed=fixed,
                    row=rows[axis],
                    observed=node.read.policy_information,
                    source=source,
                    frames={**frames, key: frame},
                )
                observed = node.read.policy_information & chosen.physical.keys()
                frame = CurrentIIDDrawFrame(
                    physical={
                        **frame.physical,
                        **{name: chosen.physical[name] for name in observed},
                    },
                    addressed_bases={
                        **frame.addressed_bases,
                        **{name: chosen.addressed_bases[name] for name in observed},
                    },
                    held_bases={
                        **frame.held_bases,
                        **{name: chosen.held_bases[name] for name in observed},
                    },
                    source_params={**frame.source_params, **chosen.source_params},
                    source_inputs={**frame.source_inputs, **chosen.source_inputs},
                    recovery_error={
                        **frame.recovery_error,
                        **{name: chosen.recovery_error[name] for name in observed},
                    },
                    valid=frame.valid & chosen.valid,
                    information_available={
                        **frame.information_available,
                        **{
                            name: chosen.information_available[name]
                            for name in observed
                        },
                    },
                )
                terms.append(weights)
            return frame, jnp.concatenate(terms)

        from _lcm.policy_queries import map_lanes  # noqa: PLC0415

        frame, weights = map_lanes(
            function=point,
            operands=jnp.arange(math.prod(extents), dtype=jnp.int32),
            batch_size=self.row_batch,
        )
        return frame, weights.T

    def _observer_row(
        self,
        *,
        observer: QueryNode,
        target: QueryNode,
        group: ShockGroupName,
        fixed: QueryArguments,
        row: IntND,
        observed: frozenset[str],
        source: SourceStates,
        frames: Frames,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        """An unavailable observation yields the neutral representative.

        Its settlement and law run only on a discarded batch lane, reading the
        producer's stand-in state rather than an unusable projection.
        """
        from _lcm.current_iid import (  # noqa: PLC0415
            law_informativeness,
            law_parameters,
        )
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _observation_available,
            _observation_inputs,
        )

        plan = target.context.current_iid_plans[target.read.period]
        if plan is None:
            raise ValueError("An observation needs an actual producing law.")
        binding, nodes = next(
            (binding, nodes) for name, binding, nodes in plan.bindings if name == group
        )
        available = {
            **source,
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }

        active = _observation_available(
            node=observer,
            group=group,
            definitions=self.definitions,
            arguments=self.information_arguments,
            available=available,
        )

        def produce() -> tuple[CurrentIIDDrawFrame, FloatND]:
            mapped = _observation_inputs(
                node=observer,
                group=group,
                definitions=self.definitions,
                arguments=self.information_arguments,
                period=target.read.period,
                available=available,
                guard=active,
            )
            site = CurrentIIDPlan(
                bindings=((group, binding, nodes),),
                indices=jnp.arange(nodes.weights.size, dtype=jnp.int32)[:, None],
            )
            produced, weights = site.realise_information_nodes(
                arguments=cast("QAndFKwargs", {**fixed, **mapped}),
                group_params={
                    group: cast("AdmittedGroupParams", fixed[CURRENT_IID_ARGUMENT])[
                        group
                    ]
                },
                node_operands={
                    group: cast("NodeOperands", fixed[CURRENT_IID_NODES_ARGUMENT])[
                        group
                    ]
                },
                indices=site.indices,
            )
            params = law_parameters(params=produced.source_params[group])
            informative = law_informativeness(
                binding=binding,
                params=params,
                inputs=produced.source_inputs[group],
            )
            mask = jnp.stack(
                tuple(
                    informative[name] & (name in observed)
                    for name in binding.group.shocks
                )
            )
            identity = binding.group._has_identity_correlation(  # noqa: SLF001 - shared exact represented-matrix proof
                params=params
            )
            chosen = _row(
                frame=_batch_metadata(frame=produced, size=nodes.weights.size),
                row=row,
            )
            return chosen._replace(
                valid=chosen.valid & (identity | jnp.all(mask) | ~jnp.any(mask)),
                information_available={
                    name: jnp.asarray(a=True) for name in chosen.physical
                },
            ), weights[:, row]

        def prototype_fields() -> tuple[tuple[PytreeValue, ...], FloatND]:
            frame, weights = produce()
            return tuple(frame), weights

        prototype = jax.eval_shape(prototype_fields)
        neutral_fields, neutral_weights = jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
            prototype,
        )
        neutral_frame = CurrentIIDDrawFrame(*neutral_fields)._replace(
            valid=jnp.asarray(a=True)
        )
        # Inactive support is a single numerical no-information representative;
        # it does not consult an unsafe target law to obtain a probability mass.
        neutral_weights = (
            jnp.ones_like(neutral_weights)
            .at[0]
            .set((row == 0).astype(neutral_weights.dtype))
        )
        neutral = vary_carry_like_body(
            body=lambda _carry: produce(), carry=(neutral_frame, neutral_weights)
        )
        return jax.lax.cond(active, produce, lambda: neutral)

    def _ancestors(
        self,
        *,
        position: int,
        frames: Frames,
        indices: tuple[IntND, ...],
        live: BoolND,
        source: SourceStates,
        dependencies: Dependencies,
    ) -> PolicyAnswer:
        if position == len(self.node.ancestors):
            return self._own(
                frames=frames,
                indices=indices,
                live=live,
                source=source,
                dependencies=dependencies,
            )
        name = self.node.ancestors[position]
        node = self.definitions[name]
        fixed = self.information_arguments[name]

        guard = live & self._available(
            node=node, fixed=fixed, source=source, frames=frames
        )

        def admitted() -> PolicyAnswer:
            frame, factors = self._realise(
                node=node, fixed=fixed, source=source, frames=frames, guard=guard
            )
            if not _has_local_information(node=node, period=node.read.period):
                return self._ancestors(
                    position=position + 1,
                    frames={**frames, name: frame},
                    indices=indices,
                    live=live,
                    source=source,
                    dependencies=dependencies,
                )

            # keyword-only-exempt: library-callback=_map_support
            def point(row: IntND, at: IntND) -> PolicyAnswer:
                return self._ancestors(
                    position=position + 1,
                    frames={**frames, name: _row(frame=frame, row=row)},
                    indices=(*indices, at),
                    live=live,
                    source=source,
                    dependencies=dependencies,
                )

            answers = self._map_support(point=point, name=name, node=node, spread=False)
            present = jnp.all(~is_represented_zero(factors), axis=0)
            pinned = self._pinned(name=name)
            # A representative's landings count once per present row it stands
            # for, and a class's once per row position in it.
            copies = (
                present.astype(jnp.int32)
                if pinned is None
                else jax.ops.segment_sum(
                    present.astype(jnp.int32),
                    pinned[1],
                    num_segments=pinned[0].size,
                )
            )
            classes = self._classes(name=self.name, ancestor=name)
            if classes is not None:
                copies = jax.ops.segment_sum(
                    copies,
                    jnp.asarray(classes[1]),
                    num_segments=classes[0].size,
                )
            return (
                *answers[:4],
                _scaled_counts(counts=answers[4], copies=copies),
            )

        absent = vary_carry_like_body(
            body=lambda _carry: admitted(), carry=self._absent(position=position)
        )
        return jax.lax.cond(guard, admitted, lambda: absent)

    def _absent(self, *, position: int) -> PolicyAnswer:
        dimensions = tuple(
            self._class_axis(ancestor=name)
            for name in self.node.ancestors[position:]
            if _has_local_information(
                node=self.definitions[name], period=self.definitions[name].read.period
            )
        )
        from _lcm.policy_queries import _query_landing_checks  # noqa: PLC0415

        members = self.node.value_shape
        return (
            jnp.full((*dimensions, *members), -jnp.inf),
            jnp.full(
                (*dimensions, self._size(name=self.name, node=self.node)),
                -1,
                jnp.int32,
            ),
            jnp.zeros(dimensions, jnp.bool_),
            jnp.ones(dimensions, jnp.bool_),
            jnp.zeros(
                (*dimensions, len(_query_landing_checks(node=self.node))), jnp.int32
            ),
        )

    def _stay(
        self,
        *,
        frame: CurrentIIDDrawFrame,
        frames: Frames,
        live: BoolND,
        source: SourceStates,
    ) -> tuple[StayTable, IntND] | None:
        """This query's stay table once per class of its own rows, and the class
        of each row position; None when every row position is its own class.

        The table holds no participation, which reads the row's exit payoffs; the
        cell adds it at each row.
        """
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            map_lanes,
            query_arguments,
            stay_table,
        )

        if self.name == self.realised_root:
            return None
        classes = own_stay_classes(name=self.name, definitions=self.definitions)
        if classes is None:
            return None
        first = classes[0]
        rows = jnp.asarray(
            first
            if pinned_support(name=self.name, definitions=self.definitions) is None
            else cast(
                "tuple[np.ndarray, np.ndarray]",
                _pinned_rows(name=self.name, definitions=self.definitions),
            )[0][first],
            dtype=jnp.int32,
        )

        def evaluate(position: IntND) -> StayTable:
            physical = _row(frame=frame, row=rows[position]).physical
            arguments, _ = query_arguments(
                guard=live,
                source=source,
                node=self.node,
                fixed={
                    **self.fixed,
                    **physical,
                    _PHYSICAL_CONTEXTS: {
                        **{name: frame.physical for name, frame in frames.items()},
                        self.name: physical,
                    },
                },
                projection_operands=self.projection_operands,
            )
            return stay_table(
                node=self.node,
                period=self.node.read.period,
                arguments=arguments,
                dependencies={},
                participate=False,
            )

        table = map_lanes(
            function=evaluate,
            operands=jnp.arange(first.size, dtype=jnp.int32),
            batch_size=self.row_batch,
        )
        return table, jnp.asarray(classes[1])

    def _own(
        self,
        *,
        frames: Frames,
        indices: tuple[IntND, ...],
        live: BoolND,
        source: SourceStates,
        dependencies: Dependencies,
    ) -> PolicyAnswer:
        from _lcm.policy_queries import _PHYSICAL_CONTEXTS, _query_cell  # noqa: PLC0415

        guard = live & self._available(
            node=self.node, fixed=self.fixed, source=source, frames=frames
        )

        def admitted() -> PolicyAnswer:
            frame, factors = self._realise(
                node=self.node,
                fixed=self.fixed,
                source=source,
                frames=frames,
                guard=guard,
            )

            stay = self._stay(frame=frame, frames=frames, live=live, source=source)

            # keyword-only-exempt: library-callback=_map_support
            def point(row: IntND, at: IntND) -> PolicyAnswer:
                local = _has_local_information(
                    node=self.node, period=self.node.read.period
                )
                physical = {} if not local else _row(frame=frame, row=row).physical
                projected = {
                    name: (
                        answer[0][index],
                        jnp.asarray(-1, jnp.int32),
                        answer[2][index],
                        answer[3],
                        answer[4],
                    )
                    for name, answer in dependencies.items()
                    for index in (
                        self._dependency_index(
                            producer=self.node.inputs[name],
                            indices=indices,
                            at=at if local else None,
                        ),
                    )
                }
                return _query_cell(
                    live,
                    source,
                    projected,
                    node=self.node,
                    period=self.node.read.period,
                    fixed={
                        **self.fixed,
                        **physical,
                        _PHYSICAL_CONTEXTS: {
                            **{name: frame.physical for name, frame in frames.items()},
                            self.name: physical,
                        },
                    },
                    projection_operands=self.projection_operands,
                    stay=None
                    if stay is None
                    else jax.tree.map(lambda leaf: leaf[stay[1][at]], stay[0]),
                )

            answers = self._map_support(
                point=point, name=self.name, node=self.node, spread=True
            )
            values, actions, support, valid, landings = answers
            expanded = jnp.broadcast_to(
                factors.reshape((*factors.shape, *((1,) * (values.ndim - 1)))),
                (factors.shape[0], *values.shape),
            )
            present = jnp.all(~is_represented_zero(factors), axis=0)
            return (
                jnp.sum(
                    factored_weighted_term(factors=expanded, values=values), axis=0
                ),
                actions,
                jnp.all(~present | support),
                jnp.all(frame.valid) & jnp.all(valid),
                sum_counts(counts=jnp.where(present[:, None], landings, 0), axis=0),
            )

        absent = vary_carry_like_body(
            body=lambda _carry: admitted(),
            carry=self._absent(position=len(self.node.ancestors)),
        )
        return jax.lax.cond(guard, admitted, lambda: absent)


def _scaled_counts(*, counts: IntND, copies: IntND) -> IntND:
    """Each leading-axis entry's counts times its copies, saturating at the int32 cap.

    Zero copies drop an entry, as a zero-mass row's landings are dropped.
    """
    cap = jnp.iinfo(jnp.int32).max
    copies = copies.reshape((-1, *((1,) * (counts.ndim - 1))))
    return jnp.where(counts > cap // jnp.maximum(copies, 1), cap, counts * copies)
