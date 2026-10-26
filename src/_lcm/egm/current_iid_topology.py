"""Actual rounded-map pullbacks for a common native candidate row."""

from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple, NoReturn, cast

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_pullback import pullback_resource_boundary
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.typing import PytreeChild
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import ShockName
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


class CommonResourcesPartition(NamedTuple):
    """Declared liquid nodes and represented native-boundary pullbacks."""

    points: Float1D
    length: ScalarInt
    valid: ScalarBool

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], CommonResourcesPartition],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_common_resources_partition, (None, tuple(self))


def pullback_native_partition(
    *,
    row: OwnedResourcesRow,
    resources: Callable[[ScalarFloat, Mapping[ShockName, ScalarFloat]], ScalarFloat],
    physical: Mapping[ShockName, FloatND | IntND],
    live: BoolND,
    liquid_grid: Float1D,
    demanded_queries: Float1D,
    capacity: int,
) -> CommonResourcesPartition:
    """Union the resolved solve grid with native represented-float pullbacks.

    The actual resources map must have the rounded-monotonicity certificate.
    A predecessor coordinate is an actual floating query, not an analytical
    left limit. The row/corner selector can cross between native knots, so
    this union alone never establishes complete selected-candidate topology.
    """
    if liquid_grid.ndim != 1 or liquid_grid.size < _MIN_DOMAIN_POINTS:
        raise ValueError("A resources partition requires its declared liquid grid.")
    if demanded_queries.ndim != 1 or demanded_queries.dtype != liquid_grid.dtype:
        raise TypeError("Exact demands must be a vector in the liquid-grid dtype.")
    if capacity < _MIN_DOMAIN_POINTS:
        raise ValueError("A resources partition requires endpoint capacity.")
    lower, upper = liquid_grid[0], liquid_grid[-1]
    native = jnp.concatenate((row.borrowing_limit[None], row.envelope.grid))
    active = jnp.arange(native.size) <= row.envelope.n_kept
    boundaries = jnp.where(active, native, row.borrowing_limit)

    def one_draw(draw: Mapping[ShockName, ScalarFloat]) -> tuple[FloatND, ScalarBool]:
        def at_liquid(liquid: ScalarFloat) -> ScalarFloat:
            return resources(liquid, draw)

        def one_boundary(
            boundary: ScalarFloat,
        ) -> tuple[ScalarFloat, ScalarBool, ScalarBool]:
            pulled = pullback_resource_boundary(
                resources=at_liquid, boundary=boundary, lower=lower, upper=upper
            )
            return pulled.point, pulled.interior, pulled.valid

        point, interior, valid = jax.vmap(one_boundary)(boundaries)
        keep = active & interior
        left = jnp.nextafter(point, lower)
        records = jnp.stack((left, point), axis=-1)
        records = jnp.where(keep[:, None], records, jnp.inf)
        return records.reshape(-1), jnp.all(~active | valid)

    def supported_draw(
        inputs: tuple[Mapping[ShockName, ScalarFloat], ScalarBool],
    ) -> tuple[FloatND, ScalarBool]:
        draw, active_draw = inputs
        return jax.lax.cond(
            active_draw,
            lambda: one_draw(draw),
            lambda: (
                jnp.full((2 * native.size,), jnp.inf, dtype=native.dtype),
                jnp.ones((), dtype=jnp.bool_),
            ),
        )

    records, valid = jax.lax.map(supported_draw, (physical, live))
    records = jnp.concatenate((liquid_grid, demanded_queries, records.reshape(-1)))
    eligible = jnp.isfinite(records) & (records >= lower) & (records <= upper)
    ordered = jnp.sort(jnp.where(eligible, records, jnp.inf))
    unique = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    ) & jnp.isfinite(ordered)
    length = jnp.sum(unique, dtype=jnp.int32)
    positions = jnp.cumsum(unique, dtype=jnp.int32) - 1
    slots = jnp.where(unique, positions, capacity)
    points = jnp.full((capacity,), jnp.inf, dtype=native.dtype)
    points = points.at[slots].set(ordered, mode="drop")
    required = jnp.concatenate((liquid_grid, demanded_queries))
    emitted = jnp.arange(capacity) < length
    retained = jnp.any(
        (required[:, None] == points[None, :]) & emitted[None, :], axis=-1
    )
    return CommonResourcesPartition(
        points=points,
        length=jnp.minimum(length, capacity),
        valid=(
            jnp.all(valid)
            & (length <= capacity)
            & jnp.all(jnp.isfinite(liquid_grid))
            & jnp.all(liquid_grid[1:] > liquid_grid[:-1])
            & jnp.all(jnp.isfinite(demanded_queries))
            & jnp.all((demanded_queries >= lower) & (demanded_queries <= upper))
            & jnp.all(retained)
            & (lower < upper)
        ),
    )


_MIN_DOMAIN_POINTS = 2


def require_resources_topology() -> NoReturn:
    """Refuse the unproved arbitrary row/corner selector before publication."""
    raise UnsupportedOperationError(
        "Native Case-R requires complete selected-candidate row/corner topology. "
        "Native-knot pullbacks and sampled refinement do not establish its "
        "crossings or source owning sides."
    )


def _flatten_common_resources_partition(
    record: CommonResourcesPartition,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(record), None


def _flatten_common_resources_partition_with_keys(
    record: CommonResourcesPartition,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(CommonResourcesPartition._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_common_resources_partition(
    metadata: None,
    children: Iterable[PytreeChild],
) -> CommonResourcesPartition:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(CommonResourcesPartition._fields):
        raise TypeError(
            "Native CommonResourcesPartition reconstruction requires its exact fields."
        )
    return cast(
        "CommonResourcesPartition", tuple.__new__(CommonResourcesPartition, leaves)
    )


jax.tree_util.register_pytree_with_keys(
    CommonResourcesPartition,
    _flatten_common_resources_partition_with_keys,
    _unflatten_common_resources_partition,
    _flatten_common_resources_partition,
)
