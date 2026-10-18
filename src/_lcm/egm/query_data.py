"""Numeric query records with independently declared support and candidate identity."""

import functools
from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from _lcm.grids.discrete import read_at_codes
from lcm.typing import BoolND, FloatND, IntND, ScalarInt

if TYPE_CHECKING:
    from _lcm.typing import PytreeChild
else:
    # Query records are imported while the engine type vocabulary is initialized.
    type PytreeChild = object

type NumericQueryLeaf = FloatND | IntND | BoolND


class QueryNodeKind(IntEnum):
    """Distinguish padding from represented finite or economically impossible nodes."""

    PAD = 0
    FINITE = 1
    ECONOMIC_MINUS_INF = 2


@dataclass(frozen=True, kw_only=True)
class EGMLiquidCornerData:
    """Continuation at the floor on an independently supported liquid-row bank."""

    continuation: FloatND
    """Floor continuation on `(*conditional_cells, liquid_nodes, intervals)`."""
    direct_marginal: FloatND
    """Continuation derivative in the model-declared native row coordinate."""
    node_kind: IntND
    """Independent represented support at each corner node."""
    link_valid: BoolND
    """Permitted adjacent liquid-node links, independently of ordinary rows."""
    owner_id: IntND
    """Canonical corner owner at each liquid node and interval."""
    side_id: IntND
    """Canonical one-sided owner code."""
    interval_id: IntND
    """Canonical interval label associated with each corner node."""


@dataclass(frozen=True, kw_only=True)
class EGMQueryData:
    """Candidate provenance beside the carry's existing conditional numeric rows."""

    savings: FloatND
    """Original candidate saving associated with each row ordinate."""
    node_kind: IntND
    """Represented support of each ordinary row node."""
    link_valid: BoolND
    """Permitted adjacent ordinary-row interpolation links."""
    owner_id: IntND
    """Canonical candidate or segment owner."""
    side_id: IntND
    """One-sided row owner code."""
    floor_used: FloatND
    """Generation floor per fixed state cell, without action or outer axes."""
    corner_continuation: FloatND | None = None
    """Compact interval continuation at the floor, before current feasibility."""
    corner_kind: IntND | None = None
    """Compact corner support, with missing encoded as zero."""
    corner_interval_id: IntND | None = None
    """Canonical compact interval labels."""
    corner_side_id: IntND | None = None
    """Canonical compact one-sided interval owners."""
    liquid_corner_data: EGMLiquidCornerData | None = None
    """Liquid-dependent alternative to the complete compact corner group."""

    def __post_init__(self) -> None:
        compact = (
            self.corner_continuation,
            self.corner_kind,
            self.corner_interval_id,
            self.corner_side_id,
        )
        present = tuple(leaf is not None for leaf in compact)
        if any(present) != all(present) or all(present) == (
            self.liquid_corner_data is not None
        ):
            raise TypeError(
                "Query data requires exactly one complete corner representation."
            )

    def leaves(self) -> MappingProxyType[tuple[str, ...], NumericQueryLeaf]:
        """Expose original array objects under their declared nested field paths."""
        return MappingProxyType(dict(_query_record_leaves(record=self, prefix=())))


type QueryRecord = EGMQueryData | EGMLiquidCornerData


def select_query_data(
    *,
    data: EGMQueryData,
    positions: tuple[ScalarInt | int | slice, ...],
    floor_positions: tuple[ScalarInt | int | slice, ...] | None = None,
) -> EGMQueryData:
    """Select DC row axes while preserving the floor's independently declared axes."""

    def select(leaf: NumericQueryLeaf | None) -> NumericQueryLeaf | None:
        return None if leaf is None else read_at_codes(array=leaf, positions=positions)

    return replace(
        data,
        savings=read_at_codes(array=data.savings, positions=positions),
        node_kind=read_at_codes(array=data.node_kind, positions=positions),
        link_valid=read_at_codes(array=data.link_valid, positions=positions),
        owner_id=read_at_codes(array=data.owner_id, positions=positions),
        side_id=read_at_codes(array=data.side_id, positions=positions),
        floor_used=read_at_codes(
            array=data.floor_used,
            positions=positions if floor_positions is None else floor_positions,
        ),
        corner_continuation=select(data.corner_continuation),
        corner_kind=select(data.corner_kind),
        corner_interval_id=select(data.corner_interval_id),
        corner_side_id=select(data.corner_side_id),
        liquid_corner_data=None
        if data.liquid_corner_data is None
        else jax.tree.map(select, data.liquid_corner_data),
    )


def _query_record_leaves(
    *, record: QueryRecord, prefix: tuple[str, ...]
) -> tuple[tuple[tuple[str, ...], NumericQueryLeaf], ...]:
    leaves = []
    for attribute in fields(record):
        value = getattr(record, attribute.name)
        if value is None:
            continue
        path = (*prefix, attribute.name)
        if isinstance(value, (EGMQueryData, EGMLiquidCornerData)):
            leaves.extend(_query_record_leaves(record=value, prefix=path))
        else:
            leaves.append((path, cast("NumericQueryLeaf", value)))
    return tuple(leaves)


def _flatten_query_record(
    record: QueryRecord,
) -> tuple[tuple[PytreeChild | QueryRecord, ...], None]:
    return tuple(getattr(record, item.name) for item in fields(record)), None


def _flatten_query_record_with_keys(
    record: QueryRecord,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild | QueryRecord], ...], None
]:
    return (
        tuple(
            (jax.tree_util.GetAttrKey(item.name), getattr(record, item.name))
            for item in fields(record)
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_query_record(
    _aux: None,
    children: Iterable[PytreeChild | QueryRecord],
    *,
    record_type: type[QueryRecord],
) -> QueryRecord:
    """Restore tracing and role trees without applying numeric constructor checks."""
    record = object.__new__(record_type)
    for item, child in zip(fields(record_type), children, strict=True):
        object.__setattr__(record, item.name, child)
    return record


for _record_type in (EGMQueryData, EGMLiquidCornerData):
    jax.tree_util.register_pytree_with_keys(
        _record_type,
        _flatten_query_record_with_keys,
        functools.partial(_unflatten_query_record, record_type=_record_type),
        _flatten_query_record,
    )


def build_template_egm_query_data(
    *,
    state_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    row_capacity: int,
    interval_capacity: int,
    outer_capacity: int = 0,
    liquid_corner: bool = False,
) -> EGMQueryData:
    """Build finite inert query records with PAD or missing support throughout."""
    if row_capacity < 1 or interval_capacity < 1 or outer_capacity < 0:
        raise ValueError("Query row and interval capacities must be positive.")
    if any(extent < 1 for extent in (*state_shape, *action_shape)):
        raise ValueError("Query state and action extents must be positive.")
    cells = (*state_shape, *action_shape)
    rows = (*cells, *((outer_capacity,) if outer_capacity else ()))
    row_shape = (*rows, row_capacity)
    corner_shape = (*rows, interval_capacity)
    liquid_shape = (*rows, row_capacity, interval_capacity)
    dtype = canonical_float_dtype()
    liquid = (
        EGMLiquidCornerData(
            continuation=jnp.zeros(liquid_shape, dtype),
            direct_marginal=jnp.zeros(liquid_shape, dtype),
            node_kind=jnp.zeros(liquid_shape, jnp.int32),
            link_valid=jnp.zeros(
                (*rows, row_capacity - 1, interval_capacity), jnp.bool_
            ),
            owner_id=jnp.zeros(liquid_shape, jnp.int32),
            side_id=jnp.zeros(liquid_shape, jnp.int32),
            interval_id=jnp.zeros(liquid_shape, jnp.int32),
        )
        if liquid_corner
        else None
    )
    return EGMQueryData(
        savings=jnp.zeros(row_shape, dtype),
        node_kind=jnp.zeros(row_shape, jnp.int32),
        link_valid=jnp.zeros((*rows, row_capacity - 1), jnp.bool_),
        owner_id=jnp.zeros(row_shape, jnp.int32),
        side_id=jnp.zeros(row_shape, jnp.int32),
        floor_used=jnp.zeros(state_shape, dtype),
        corner_continuation=None if liquid_corner else jnp.zeros(corner_shape, dtype),
        corner_kind=None if liquid_corner else jnp.zeros(corner_shape, jnp.int32),
        corner_interval_id=None
        if liquid_corner
        else jnp.zeros(corner_shape, jnp.int32),
        corner_side_id=None if liquid_corner else jnp.zeros(corner_shape, jnp.int32),
        liquid_corner_data=liquid,
    )
