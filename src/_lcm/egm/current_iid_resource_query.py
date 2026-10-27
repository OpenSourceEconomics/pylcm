"""Status-aware reads of the canonical owned DC resources candidate row."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from _lcm.egm.carry import _get_owned_local_piece, _read_owned_row_with_support
from _lcm.egm.interp import prepare_padded_grid
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.typing import PytreeChild
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import ScalarBool, ScalarFloat, ScalarInt


@dataclass(frozen=True, kw_only=True)
class OwnedResourcesRead:
    """Selected original row or literal borrowing-limit candidate, together."""

    value: ScalarFloat
    """Selected original candidate value."""
    marginal: ScalarFloat
    """Economic marginal of the same selected candidate."""
    policy: ScalarFloat
    """Consumption of the selected candidate."""
    savings: ScalarFloat
    """Saving associated with the selected candidate."""
    native_owner: ScalarInt
    """Original candidate owner, or the borrowing-limit sentinel."""
    side_id: ScalarInt
    """Original selected one-sided record."""
    corner: ScalarBool
    """Whether the literal borrowing-limit candidate was selected."""
    feasible: ScalarBool
    """Whether a feasible row or corner candidate exists."""
    valid: ScalarBool
    """Mandatory numerical and original-source integrity."""

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], OwnedResourcesRead],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_owned_resources_read, (
            None,
            tuple(getattr(self, name) for name in _OWNED_RESOURCES_READ_FIELDS),
        )


def read_resources_candidate(
    *,
    row: OwnedResourcesRow,
    resources: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    owner_capacity: int,
) -> OwnedResourcesRead:
    """Return the original selected record without retaining transient proof data."""
    selected, _, _ = read_resources_candidate_details(
        row=row,
        resources=resources,
        utility_of_action=utility_of_action,
        owner_capacity=owner_capacity,
    )
    return selected


def read_resources_candidate_details(
    *,
    row: OwnedResourcesRow,
    resources: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    owner_capacity: int,
) -> tuple[OwnedResourcesRead, ScalarBool, ScalarInt]:
    """Reuse the associated owned interpolants and analytical constrained branch.

    The corner fields below contain the actual source expectation and literal
    interval, not a query authority or a dummy continuation. The same row,
    resource coordinate and selected saving determine every output channel.
    """
    known_floor = jnp.isfinite(row.expected_floor_value) | jnp.isneginf(
        row.expected_floor_value
    )
    corner_kind = jnp.where(
        known_floor,
        jnp.where(
            jnp.isneginf(row.expected_floor_value),
            QueryNodeKind.ECONOMIC_MINUS_INF,
            QueryNodeKind.FINITE,
        ),
        QueryNodeKind.PAD,
    ).astype(jnp.int32)
    data = EGMQueryData(
        savings=row.envelope.savings,
        node_kind=row.node_kind,
        link_valid=row.envelope.link_valid,
        owner_id=row.envelope.owner_id,
        side_id=row.envelope.side_id,
        floor_used=row.borrowing_limit,
        corner_continuation=row.expected_floor_value[None],
        corner_kind=corner_kind[None],
        corner_interval_id=jnp.zeros((1,), dtype=jnp.int32),
        corner_side_id=jnp.zeros((1,), dtype=jnp.int32),
    )
    selected, supported_node = _read_owned_row_with_support(
        query=resources,
        grid=row.envelope.grid,
        value=row.envelope.value,
        marginal=row.marginal,
        data=data,
        owner_capacity=owner_capacity,
    )
    domain = resources > row.borrowing_limit
    safe_consumption = jnp.where(domain, resources - row.borrowing_limit, 1.0)
    corner_value = utility_of_action(safe_consumption) + zero_safe_weighted_term(
        weight=row.discount_factor,
        value=row.expected_floor_value,
        subnormal_is_accounted_for=True,
    )
    corner_marginal = jax.grad(utility_of_action)(safe_consumption)
    finite_corner = (
        known_floor & jnp.isfinite(corner_value) & jnp.isfinite(corner_marginal)
    )
    row_legal = (
        selected.supported
        & (selected.savings >= row.borrowing_limit)
        & (selected.savings <= row.upper_savings)
        & (resources - selected.savings > 0.0)
    )
    take_row = row_legal & (~finite_corner | (selected.value >= corner_value))
    feasible = domain & (row_legal | finite_corner)
    child_invalid = jnp.zeros((), dtype=jnp.bool_)
    for status in row.child_status or ():
        for leaf in jax.tree.leaves(status):
            child_invalid |= jnp.any(leaf)
    floor_invalid = jnp.zeros((), dtype=jnp.bool_)
    if row.floor_status is not None:
        floor_invalid = (
            row.floor_status.nonfinite_floor | row.floor_status.invalid_shifted_grid
        )
    invalid_corner = (
        domain
        & known_floor
        & (
            (~jnp.isfinite(corner_value) & ~jnp.isneginf(corner_value))
            | ~jnp.isfinite(corner_marginal)
        )
    )
    valid = ~(
        row.construction_invalid
        | child_invalid
        | floor_invalid
        | selected.invalid_numeric
        | invalid_corner
        | ~jnp.isfinite(resources)
        | (domain & ~(row_legal | finite_corner))
    )
    result = OwnedResourcesRead(
        value=jnp.where(
            feasible, jnp.where(take_row, selected.value, corner_value), -jnp.inf
        ),
        marginal=jnp.where(
            feasible, jnp.where(take_row, selected.marginal, corner_marginal), 0.0
        ),
        policy=jnp.where(
            feasible,
            resources - jnp.where(take_row, selected.savings, row.borrowing_limit),
            0.0,
        ),
        savings=jnp.where(take_row, selected.savings, row.borrowing_limit),
        native_owner=jnp.where(take_row, selected.owner_id, -1),
        side_id=jnp.where(take_row, selected.side_id, 0),
        corner=~take_row,
        feasible=feasible,
        valid=valid,
    )

    search, count = prepare_padded_grid(row.envelope.grid)
    last = jnp.maximum(count - 1, 0)
    insertion = jnp.searchsorted(search, resources, side="right")
    exact_index = jnp.clip(insertion - 1, 0, last).astype(jnp.int32)
    endpoint = (count > 0) & (
        (row.envelope.grid[exact_index] == resources)
        | (resources >= row.envelope.grid[last])
    )
    node_index = jnp.where(resources >= row.envelope.grid[last], last, exact_index)
    upper = jnp.clip(insertion, 1, jnp.maximum(count - 1, 1)).astype(jnp.int32)
    lower = upper - 1
    indices, _, _ = _get_owned_local_piece(
        grid=row.envelope.grid,
        data=data,
        supported_node=supported_node,
        lower=lower,
        upper=upper,
        endpoint=endpoint,
        node_index=node_index,
        last=last,
        use_left=(node_index == last) | (data.side_id[node_index] == -1),
    )
    return result, row_legal, indices[0]


_OWNED_RESOURCES_READ_FIELDS = tuple(field.name for field in fields(OwnedResourcesRead))


def _flatten_owned_resources_read(
    record: OwnedResourcesRead,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(getattr(record, name) for name in _OWNED_RESOURCES_READ_FIELDS), None


def _flatten_owned_resources_read_with_keys(
    record: OwnedResourcesRead,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), getattr(record, name))
        for name in _OWNED_RESOURCES_READ_FIELDS
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_owned_resources_read(
    metadata: None,
    children: Iterable[PytreeChild],
) -> OwnedResourcesRead:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(_OWNED_RESOURCES_READ_FIELDS):
        raise TypeError(
            "Native OwnedResourcesRead reconstruction requires its exact fields."
        )
    record = object.__new__(OwnedResourcesRead)
    for name, leaf in zip(_OWNED_RESOURCES_READ_FIELDS, leaves, strict=True):
        object.__setattr__(record, name, leaf)
    return record


jax.tree_util.register_pytree_with_keys(
    OwnedResourcesRead,
    _flatten_owned_resources_read_with_keys,
    _unflatten_owned_resources_read,
    _flatten_owned_resources_read,
)
