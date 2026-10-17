"""Publish genuine keeper and conditional adjuster rows on one bounded outer axis."""

from dataclasses import replace

import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.nbegm_query_model import NBEGMQueryModel
from _lcm.egm.query_data import (
    EGMAdaptiveOuterData,
    QueryNodeKind,
    build_template_egm_query_data,
)
from lcm.typing import BoolND, Float1D, FloatND, IntND

_MINIMUM_MESH_NODES = 2


def adaptive_nbegm_template(*, inner: EGMCarry, model: NBEGMQueryModel) -> EGMCarry:
    """Declare C,D,O,K from canonical capacities without numerical mesh authority."""
    spec = model.authority
    adaptive = spec.adaptive_outer
    if adaptive is None or inner.query_data is None:
        raise TypeError(
            "An adaptive template requires actual conditional inner records."
        )
    n_states = len(spec.state_names)
    state_shape = inner.endog_grid.shape[:n_states]
    action_shape = inner.endog_grid.shape[n_states:-1]
    shape = (
        *inner.endog_grid.shape[:-1],
        adaptive.adjuster_capacity + 1,
        spec.row_capacity,
    )
    rows = jnp.zeros(shape, dtype=inner.endog_grid.dtype)
    return replace(
        inner,
        endog_grid=rows,
        value=jnp.zeros_like(rows),
        marginal_utility=jnp.zeros_like(rows),
        policy=jnp.zeros_like(rows),
        breakpoints=None
        if inner.breakpoints is None
        else jnp.zeros(
            (*shape[:-1], inner.breakpoints.shape[-1]),
            dtype=rows.dtype,
        ),
        query_data=build_template_egm_query_data(
            state_shape=state_shape,
            action_shape=action_shape,
            outer_capacity=adaptive.adjuster_capacity + 1,
            adaptive_capacity=adaptive.adjuster_capacity,
            row_capacity=spec.row_capacity,
            interval_capacity=len(spec.interval_ids),
        ),
    )


def assemble_adaptive_nbegm_carry(
    *,
    keeper: EGMCarry,
    adjusters: EGMCarry,
    outer_nodes: Float1D,
    capacity: int,
) -> tuple[EGMCarry, BoolND]:
    """Copy each original tuple once into keeper slot0 and active adjuster slots."""
    if keeper.query_data is None or adjusters.query_data is None:
        raise TypeError("Adaptive publication requires genuine conditional NB records.")
    kdata, adata = keeper.query_data, adjusters.query_data
    if kdata.adaptive_outer_data is not None or adata.adaptive_outer_data is not None:
        raise TypeError("An adaptive publisher cannot reinterpret already-nested rows.")
    if kdata.liquid_corner_data is not None or adata.liquid_corner_data is not None:
        raise TypeError("Adaptive NB publication requires its compact held intervals.")
    count = outer_nodes.size
    if not _MINIMUM_MESH_NODES <= count <= capacity:
        raise TypeError("The actual completed mesh exceeds its declared capacity.")

    def stack(
        *, kept: FloatND | IntND | BoolND, moved: FloatND | IntND | BoolND
    ) -> FloatND | IntND | BoolND:
        combined = jnp.concatenate(
            (kept[..., None, :], jnp.moveaxis(moved, 0, -2)), axis=-2
        )
        return jnp.pad(
            combined, [(0, 0)] * (combined.ndim - 2) + [(0, capacity - count), (0, 0)]
        )

    def optional(
        *, kept: FloatND | IntND | None, moved: FloatND | IntND | None
    ) -> FloatND | IntND | None:
        if kept is None or moved is None:
            if kept is not None or moved is not None:
                raise TypeError("Adaptive keeper and adjuster tuple groups differ.")
            return None
        return stack(kept=kept, moved=moved)

    cells = keeper.endog_grid.shape[:-1]
    active = jnp.arange(capacity) < count
    mesh = EGMAdaptiveOuterData(
        outer_nodes=jnp.pad(
            outer_nodes, (0, capacity - count), constant_values=outer_nodes[0]
        ),
        outer_count=jnp.asarray(count, dtype=jnp.int32),
        outer_node_kind=jnp.where(
            active, QueryNodeKind.FINITE, QueryNodeKind.PAD
        ).astype(jnp.int32),
        outer_link_valid=jnp.broadcast_to(
            (jnp.arange(capacity - 1) < count - 1), (*cells, capacity - 1)
        ),
        outer_owner_id=jnp.broadcast_to(
            jnp.where(active, jnp.arange(capacity), -1), (*cells, capacity)
        ).astype(jnp.int32),
        outer_side_id=jnp.zeros((*cells, capacity), dtype=jnp.int32),
    )
    data = replace(
        kdata,
        floor_used=kdata.floor_used,
        savings=stack(kept=kdata.savings, moved=adata.savings),
        node_kind=stack(kept=kdata.node_kind, moved=adata.node_kind),
        link_valid=stack(kept=kdata.link_valid, moved=adata.link_valid),
        owner_id=stack(kept=kdata.owner_id, moved=adata.owner_id),
        side_id=stack(kept=kdata.side_id, moved=adata.side_id),
        corner_continuation=optional(
            kept=kdata.corner_continuation, moved=adata.corner_continuation
        ),
        corner_kind=optional(kept=kdata.corner_kind, moved=adata.corner_kind),
        corner_interval_id=optional(
            kept=kdata.corner_interval_id, moved=adata.corner_interval_id
        ),
        corner_side_id=optional(kept=kdata.corner_side_id, moved=adata.corner_side_id),
        adaptive_outer_data=mesh,
    )
    return replace(
        keeper,
        endog_grid=stack(kept=keeper.endog_grid, moved=adjusters.endog_grid),
        value=optional(kept=keeper.value, moved=adjusters.value),
        marginal_utility=stack(
            kept=keeper.marginal_utility, moved=adjusters.marginal_utility
        ),
        policy=optional(kept=keeper.policy, moved=adjusters.policy),
        breakpoints=optional(kept=keeper.breakpoints, moved=adjusters.breakpoints),
        query_data=data,
    ), jnp.any(adata.floor_used != kdata.floor_used[None])
