"""Forward simulation of a population grouped by an invariant state.

When the forward phase provably never changes a blocked state, every subject
keeps the code it starts with. Forward simulation then runs each code's
subjects in chunks of their own, reading every stored value that carries the
state through that code's block, while each subject keeps:

- the random keys of its original row, gathered from the full-population
  streams rather than drawn again for its position in a group;
- its original output row, restored when the result is assembled.

A row holding a code outside the state's grid starts where the state is
irrelevant, never enters a regime carrying it, and joins the first code's group.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from _lcm.execution.core_program import InvariantBinding, ValueRead
from _lcm.execution.invariant_blocks import block_layout, selected_block_view
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactKind,
    ValueViewDescriptor,
    classify_value_transfer,
    resolve_value_transfer,
)
from _lcm.typing import RegimeName, StateName
from lcm.typing import FloatND

# Program family named by the binding a grouped read's view selects with.
_FAMILY = "simulation"


@dataclass(frozen=True, kw_only=True)
class SubjectGroupingRoute:
    """The invariant state a model's forward simulation groups its subjects by."""

    state_name: StateName
    """Name of the grouping state."""

    codes: tuple[int, ...]
    """Codes of the state's grid, in grid order; one group per code."""

    value_axis_names: Mapping[RegimeName, tuple[StateName, ...]]
    """Per regime, the axes of its stored value in stored order."""

    def __post_init__(self) -> None:
        """Snapshot the caller's axis mapping."""
        object.__setattr__(
            self, "value_axis_names", MappingProxyType(dict(self.value_axis_names))
        )


@runtime_checkable
class ComponentValueSource(Protocol):
    """Where grouped simulation reads one code's values from, one code at a time.

    The values of each code exist on the device only between its `acquire` and
    its `release` (or `abandon`, when its simulation fails). Each holds the
    code alone along the grouping state's axis.
    """

    @property
    def codes(self) -> tuple[int, ...]:
        """Return every code, in the order they are acquired."""
        ...

    @property
    def subject_codes(self) -> tuple[int, ...] | None:
        """Return the codes whose subjects are simulated, or `None` for every code.

        A source holding a selection of the codes simulates the subjects of
        those codes alone; the result then holds their rows only.
        """
        ...

    def acquire(
        self, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Return the code's values on the device, by period and regime."""
        ...

    def release(self, *, code: int) -> None:
        """Finish with the code's values once its subjects are simulated."""
        ...

    def abandon(self, *, code: int) -> None:
        """Delete the code's device values after its simulation failed."""
        ...

    def values(self) -> Mapping[int, Mapping[RegimeName, FloatND]]:
        """Return the complete logical value store, once every code is released."""
        ...


@dataclass(frozen=True, kw_only=True, eq=False)
class SubjectRows:
    """The original population rows one chunk simulates, in chunk order."""

    rows: np.ndarray
    """Original row of each chunk row; a short tail repeats its last real row."""


@dataclass(frozen=True, kw_only=True, eq=False)
class SubjectChunk:
    """One chunk of a single code's subjects."""

    code: int
    """The code every subject of the chunk holds."""

    rows: np.ndarray
    """Original rows simulated by the chunk."""


@dataclass(frozen=True, kw_only=True, eq=False)
class SubjectGroupPlan:
    """Every grouped chunk of one call and the way back to the public order."""

    chunks: tuple[SubjectChunk, ...]
    """Chunks in dispatch order: by code, then by original row."""

    positions: np.ndarray
    """For each row of `rows`, its row in the chunks' concatenated outputs."""

    rows: np.ndarray
    """The original rows the chunks simulate, ascending; every real row unless
    the plan selects codes."""


def group_codes(
    *, route: SubjectGroupingRoute, codes: np.ndarray | None, n_real: int
) -> np.ndarray:
    """Return the group code of each real row.

    Args:
        route: The grouping route.
        codes: The state's initial codes, or `None` when no subject supplies it.
        n_real: Number of real subjects, which lead the population.

    Returns:
        One code per real row; a code outside the grid becomes the first code.

    """
    if codes is None:
        return np.full(n_real, route.codes[0], dtype=np.int64)
    real = np.asarray(codes)[:n_real].astype(np.int64)
    return np.where(np.isin(real, route.codes), real, route.codes[0])


def group_sizes(
    *, route: SubjectGroupingRoute, codes: np.ndarray | None, n_real: int
) -> tuple[int, ...]:
    """Return the number of real subjects in each code's group, in grid order."""
    keys = group_codes(route=route, codes=codes, n_real=n_real)
    return tuple(int(np.count_nonzero(keys == code)) for code in route.codes)


def grouped_extent(*, sizes: tuple[int, ...], alignment: int) -> int:
    """Return the largest group rounded up to the device alignment."""
    return -(-max(sizes) // alignment) * alignment


def grouped_rows(*, sizes: tuple[int, ...], width: int) -> int:
    """Return the rows every group's chunks of `width` dispatch together."""
    return sum(-(-size // width) * width for size in sizes)


def plan_subject_groups(
    *,
    route: SubjectGroupingRoute,
    codes: np.ndarray | None,
    n_real: int,
    width: int,
    selected: tuple[int, ...] | None = None,
) -> SubjectGroupPlan:
    """Partition the real rows into chunks of one code each.

    Each code's rows keep their original order and are cut into chunks of
    `width`. A short last chunk repeats the group's last row, whose duplicate
    outputs no position reads. An empty group dispatches nothing.

    Args:
        route: The grouping route.
        codes: The state's initial codes, or `None` when no subject supplies it.
        n_real: Number of real subjects, which lead the population.
        width: Rows per chunk.
        selected: Codes whose rows are planned, or `None` for every code. The
            chunks of a selected code are the chunks the whole population's
            plan cuts for it.

    Returns:
        The chunks and the position of each planned original row's output.

    """
    keys = group_codes(route=route, codes=codes, n_real=n_real)
    chunks: list[SubjectChunk] = []
    positions = np.empty(n_real, dtype=np.int32)
    offset = 0
    for code in route.codes:
        if selected is not None and code not in selected:
            continue
        rows = np.flatnonzero(keys == code).astype(np.int32)
        for start in range(0, len(rows), width):
            part = rows[start : start + width]
            positions[part] = offset + np.arange(len(part), dtype=np.int32)
            if len(part) < width:
                part = np.concatenate([part, np.repeat(part[-1:], width - len(part))])
            chunks.append(SubjectChunk(code=int(code), rows=part))
            offset += width
    planned = (
        np.arange(n_real, dtype=np.int32)
        if selected is None
        else np.flatnonzero(np.isin(keys, selected)).astype(np.int32)
    )
    return SubjectGroupPlan(
        chunks=tuple(chunks), positions=positions[planned], rows=planned
    )


def type_local_view(
    *,
    route: SubjectGroupingRoute,
    read: ValueRead,
    stored: object,
    required_sharding: jax.sharding.Sharding,
    code: int,
    stored_codes: tuple[int, ...] | None = None,
) -> ValueViewDescriptor | None:
    """Describe the block of `code` a grouped read takes from a value with the state.

    `stored_codes` are the codes the stored value holds along the state's axis,
    in order; `None` means every code of the route. The selected position is
    the code's position among them, so a stored value holding one code alone
    is read at position zero.

    Returns:
        The selected view, without the state's axis, or `None` for a value read
        whole: one of a regime without the state, or not a regime value.

    """
    if read.target.kind is not ValueArtifactKind.REGIME_VALUE:
        return None
    names = route.value_axis_names.get(read.target.regime, ())
    if route.state_name not in names:
        return None
    return selected_block_view(
        artifact=read.target,
        binding=InvariantBinding(
            state_name=route.state_name,
            start=(route.codes if stored_codes is None else stored_codes).index(code),
            code=code,
            family=_FAMILY,
        ),
        stored_axis_names=names,
        stored_template=stored,
        required_sharding=required_sharding,
    )


def type_local_transfer(
    *,
    route: SubjectGroupingRoute | None,
    code: int | None,
    read: ValueRead,
    stored: object,
    required_sharding: jax.sharding.Sharding,
    stored_codes: tuple[int, ...] | None = None,
) -> ResolvedValueTransfer:
    """Resolve a read onto `required_sharding`, selecting the group's block.

    Without a route the value is read whole. With one, a value carrying the
    state is selected on its stored layout first and only that block moves.
    `stored_codes` are the codes the stored value holds, `None` for all.
    """
    view = (
        None
        if route is None or code is None
        else type_local_view(
            route=route,
            read=read,
            stored=stored,
            required_sharding=required_sharding,
            code=code,
            stored_codes=stored_codes,
        )
    )
    stored_sharding = stored.sharding  # ty: ignore[unresolved-attribute]
    delivered = (
        stored_sharding
        if view is None
        else block_layout(
            layout=stored_sharding,
            axis=view.stored_axis_names.index(route.state_name),  # ty: ignore[unresolved-attribute]
            ndim=len(view.stored_shape),
        )
    )
    return resolve_value_transfer(
        target=read.target,
        source=read.source,
        kind=classify_value_transfer(
            stored_sharding=delivered, required_sharding=required_sharding
        ),
        stored_template=stored,
        source_sharding=required_sharding,
        view=view,
    )


def type_local_template(
    *,
    route: SubjectGroupingRoute | None,
    regime: RegimeName,
    leaf: jax.Array | jax.ShapeDtypeStruct,
) -> jax.Array | jax.ShapeDtypeStruct:
    """Describe what a grouped decision receives for one stored value of `regime`.

    A value carrying the state loses that axis; any other value is unchanged.
    """
    names = () if route is None else route.value_axis_names.get(regime, ())
    if route is None or route.state_name not in names:
        return leaf
    axis = names.index(route.state_name)
    shape = tuple(int(size) for size in leaf.shape)
    return jax.ShapeDtypeStruct(
        (*shape[:axis], *shape[axis + 1 :]),
        leaf.dtype,
        sharding=block_layout(layout=leaf.sharding, axis=axis, ndim=len(shape)),
        weak_type=bool(getattr(leaf, "weak_type", False)),
    )
