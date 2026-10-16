"""Signed passive reads distinguish represented economic support from absence."""

from dataclasses import replace
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import read_owned_row
from _lcm.egm.dcegm_query import (
    DCEGMQuery,
    FiniteNEGMQueryResult,
    _evaluate,
    _passive_weights,
)
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind, select_query_data
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    refine_envelope_exact,
)
from _lcm.typing import EconFunctionKwargs
from lcm.typing import FloatND, ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.finite_germ_query_fixture import make_canonical_query
from tests.egm.test_finite_nnbegm_query import _bank as nb_bank
from tests.egm.test_finite_nnbegm_query import _query as nb_query

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _Bank = tuple[FloatND, FloatND, FloatND, EGMQueryData]


class _Component(NamedTuple):
    query: DCEGMQuery | NBEGMQuery
    pool: EconFunctionKwargs
    bank: _Bank
    coordinate: float


def _dc_component() -> _Component:
    """Retain source-emitted geometry on the canonical finite DC native grid."""
    query, pool = make_canonical_query()
    capacity = query.model.authority.row_capacity
    emitted = refine_envelope_exact(
        endog_grid=jnp.asarray((1.0, 1.5)),
        policy=jnp.asarray((0.5, 0.75)),
        value=jnp.asarray((1.0, 1.0)),
        savings=jnp.asarray((0.5, 0.75)),
        n_refined=capacity,
    )
    assert isinstance(emitted, OwnedEnvelope)
    assert int(emitted.n_kept) == 2
    np.testing.assert_array_equal(emitted.grid[:2], (1.0, 1.5))
    np.testing.assert_array_equal(emitted.savings[:2], (0.5, 0.75))
    assert bool(emitted.link_valid[0])
    known = jnp.arange(capacity) < emitted.n_kept
    record = EGMQueryData(
        savings=emitted.savings,
        node_kind=jnp.where(known, QueryNodeKind.FINITE, QueryNodeKind.PAD).astype(
            jnp.int32
        ),
        link_valid=emitted.link_valid,
        owner_id=emitted.owner_id,
        side_id=emitted.side_id,
        floor_used=jnp.asarray(0.0),
        corner_continuation=jnp.asarray((-100.0,)),
        corner_kind=jnp.asarray((QueryNodeKind.FINITE,), dtype=jnp.int32),
        corner_interval_id=jnp.asarray((0,), dtype=jnp.int32),
        corner_side_id=jnp.asarray((0,), dtype=jnp.int32),
    )
    outer_capacity = query.model.authority.outer_capacity
    assert outer_capacity == 3
    data = jax.tree.map(
        lambda leaf: jnp.stack((jnp.stack((leaf,) * outer_capacity),) * 2), record
    )
    data = replace(data, floor_used=jnp.asarray((0.0, 0.25)))
    shape = (2, outer_capacity, capacity)
    bank = (
        jnp.broadcast_to(emitted.grid, shape),
        jnp.broadcast_to(emitted.value, shape),
        jnp.broadcast_to(jnp.where(known, 7.0, jnp.nan), shape),
        data,
    )
    # Common resources=11/8. Zero secant limits both value-piece slopes to zero,
    # retaining constant V1 while the separate ordinary native marginal stays7.
    return _Component(
        query=query,
        pool={**pool, "liquid": jnp.asarray(1.0 / 8.0)},
        bank=bank,
        coordinate=11.0 / 8.0,
    )


def _nb_component() -> _Component:
    """Reuse genuine NB producer geometry without changing its original grids."""
    query = nb_query(actions=False)
    values = jnp.ones((2, 2))
    bank = nb_bank(
        values=values,
        marginals=jnp.full_like(values, 7.0),
        savings=jnp.full_like(values, 0.5),
    )
    assert query.model.authority.outer_capacity == 2
    return _Component(
        query=query,
        pool={
            "liquid": jnp.asarray(1.0),
            "stock": jnp.asarray(0.0),
            "floor_scale": jnp.asarray(0.0),
            "bonus": jnp.asarray(-10.0),
            "discount_factor": jnp.asarray(1.0),
        },
        bank=bank,
        coordinate=1.0,
    )


def _with_economic_node(*, bank: _Bank) -> _Bank:
    """Change supported economic channels together while retaining source geometry."""
    grid, value, marginal, data = bank
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    known = data.node_kind[0] == QueryNodeKind.FINITE
    assert bool(jnp.all(jnp.any(known, axis=-1)))
    changed = replace(
        data,
        node_kind=data.node_kind.at[0].set(
            jnp.where(known, QueryNodeKind.ECONOMIC_MINUS_INF, data.node_kind[0])
        ),
        corner_continuation=data.corner_continuation.at[0]
        .set(-jnp.inf)
        .at[1]
        .set(-100.0),
        corner_kind=data.corner_kind.at[0].set(QueryNodeKind.ECONOMIC_MINUS_INF),
    )
    for original, preserved in (
        (data.savings, changed.savings),
        (data.link_valid, changed.link_valid),
        (data.owner_id, changed.owner_id),
        (data.side_id, changed.side_id),
        (data.floor_used, changed.floor_used),
        (data.corner_interval_id, changed.corner_interval_id),
        (data.corner_side_id, changed.corner_side_id),
    ):
        np.testing.assert_array_equal(original, preserved)
    np.testing.assert_array_equal(changed.node_kind[1], data.node_kind[1])
    assert changed.corner_kind is not None
    np.testing.assert_array_equal(changed.corner_kind[1], data.corner_kind[1])
    return (
        grid,
        value.at[0].set(jnp.where(known, -jnp.inf, value[0])),
        marginal.at[0].set(jnp.where(known, 0.0, marginal[0])),
        changed,
    )


def _assert_original_support(*, component: _Component, passive: float) -> None:
    """Qualify original supported tuples and all source/query saving domains."""
    query, pool, bank, coordinate = component
    grid, value, marginal, data = bank
    authority = query.model.authority
    topology = authority.row_topology
    outer = authority.finite_outer
    assert topology is not None
    assert outer is not None
    assert grid.shape[:2] == (2, authority.outer_capacity)
    assert authority.action_names == ()
    assert authority.state_names == ("stock",)
    if isinstance(query, NBEGMQuery):
        assert query.finite_queries is not None
    floor, nodes, status = query.evaluate_floor(pool=pool)
    assert not bool(status.nonfinite_floor)
    assert not bool(status.invalid_shifted_grid)
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    assert bool(jnp.all(jnp.isneginf(data.corner_continuation[0])))
    np.testing.assert_array_equal(data.corner_kind[0], QueryNodeKind.ECONOMIC_MINUS_INF)
    for cell in range(2):
        source_floor, source_nodes, source_status = query.evaluate_floor(
            pool={**pool, "stock": jnp.asarray(float(cell))}
        )
        assert not bool(source_status.nonfinite_floor)
        assert not bool(source_status.invalid_shifted_grid)
        np.testing.assert_array_equal(data.floor_used[cell], source_floor)
        for outer_index in range(authority.outer_capacity):
            if isinstance(query, DCEGMQuery):
                reader = query
            else:
                assert query.finite_queries is not None
                reader = query.finite_queries[0 if outer_index == 0 else 1]
            identity = (
                passive if outer_index == 0 else outer.outer_nodes[outer_index - 1]
            )
            chosen = {**pool, outer.outer_post_decision: jnp.asarray(identity)}
            resources = _evaluate(function=reader.resources, pool=chosen)
            for predicate in reader.predicates:
                assert bool(_evaluate(function=predicate, pool=chosen))
            record = select_query_data(
                data=data, positions=(cell, outer_index), floor_positions=(cell,)
            )
            live = record.node_kind != QueryNodeKind.PAD
            assert bool(jnp.any(live))
            assert bool(jnp.all(~live | (record.savings >= source_floor)))
            assert bool(jnp.all(~live | (record.savings <= source_nodes[-1])))
            assert bool(jnp.all(~live | (record.savings >= floor)))
            assert bool(jnp.all(~live | (record.savings <= nodes[-1])))
            assert bool(jnp.all(~live | (record.savings < resources)))
            first = jnp.min(jnp.where(live, grid[cell, outer_index], jnp.inf))
            last = jnp.max(jnp.where(live, grid[cell, outer_index], -jnp.inf))
            assert bool(first < coordinate < last)
            row = read_owned_row(
                query=jnp.asarray(coordinate),
                grid=grid[cell, outer_index],
                value=value[cell, outer_index],
                marginal=marginal[cell, outer_index],
                data=record,
                owner_capacity=topology.owner_capacity,
                side_is_row_side=True,
            )
            assert bool(row.supported)
            assert not bool(row.invalid_numeric)
            np.testing.assert_array_equal(row.value, -jnp.inf if cell == 0 else 1.0)
            np.testing.assert_array_equal(row.marginal, 0.0 if cell == 0 else 7.0)
            _assert_corner_dominance(
                reader=reader,
                pool=chosen,
                resources=resources,
                floor=floor,
                continuation=data.corner_continuation[cell, outer_index, 0],
                ordinary_value=row.value,
                cell=cell,
            )


def _assert_corner_dominance(
    *,
    reader: DCEGMQuery | NBEGMQuery,
    pool: EconFunctionKwargs,
    resources: ScalarFloat,
    floor: ScalarFloat,
    continuation: ScalarFloat,
    ordinary_value: ScalarFloat,
    cell: int,
) -> None:
    """Economic corners stay economic; finite corners cannot rescue or displace rows."""
    preferences = reader.build_preferences(pool)
    beta = reader.read_discount_factor(pool)
    assert bool(beta > 0.0)
    corner = preferences.utility(resources - floor) + beta * continuation
    if cell == 0:
        assert bool(jnp.isneginf(corner))
    else:
        assert bool(jnp.isfinite(corner))
        assert bool(corner < ordinary_value)


@pytest.mark.parametrize(
    ("family", "passive", "weights"),
    [
        ("finite-dc", 0.5, (0.5, 0.5)),
        ("finite-dc", 1.5, (-0.5, 1.5)),
        ("finite-dc", 1.0, (0.0, 1.0)),
        ("finite-nb", 0.5, (0.5, 0.5)),
        ("finite-nb", 1.5, (-0.5, 1.5)),
        ("finite-nb", 1.0, (0.0, 1.0)),
    ],
    ids=[
        "finite-dc-positive",
        "finite-dc-negative",
        "finite-dc-zero",
        "finite-nb-positive",
        "finite-nb-negative",
        "finite-nb-zero",
    ],
)
def test_finite_outer_signed_economic_support(
    *,
    family: Literal["finite-dc", "finite-nb"],
    passive: float,
    weights: tuple[float, float],
) -> None:
    """Economic support blends only through nonnegative active passive weights."""
    component = _dc_component() if family == "finite-dc" else _nb_component()
    component = component._replace(
        pool={**component.pool, "stock": jnp.asarray(passive)},
        bank=_with_economic_node(bank=component.bank),
    )
    factors, active, negative = _passive_weights(
        values=(jnp.asarray(passive),), grids=(jnp.asarray((0.0, 1.0)),)
    )
    np.testing.assert_array_equal(factors[0], weights)
    np.testing.assert_array_equal(active, np.asarray(weights) != 0.0)
    np.testing.assert_array_equal(negative, np.asarray(weights) < 0.0)
    assert bool(jnp.any(active))
    _assert_original_support(component=component, passive=passive)
    grid, value, marginal, data = component.bank
    result = component.query.read_choices(
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        row_queries=jnp.full((2,), component.coordinate),
        row_gradients=jnp.ones((2,)),
        passive_values=(jnp.asarray(passive),),
        passive_grids=(jnp.asarray((0.0, 1.0)),),
        action_names=(),
        action_values=(),
        pool=component.pool,
        euler_gradient=jnp.asarray(1.0),
        collect_continuation_read_counts=False,
    )
    assert isinstance(result, FiniteNEGMQueryResult)
    assert not bool(jnp.any(result.bound_status.nonfinite_floor))
    assert not bool(jnp.any(result.bound_status.invalid_shifted_grid))
    if passive == 1.5:
        np.testing.assert_array_equal(result.coverage_missing, (True,))
        assert not bool(jnp.any(jnp.isposinf(result.value) & ~result.coverage_missing))
    else:
        np.testing.assert_array_equal(result.coverage_missing, (False,))
        np.testing.assert_array_equal(
            result.value, (-jnp.inf,) if passive == 0.5 else (1.0,)
        )
        np.testing.assert_array_equal(
            result.marginal, (0.0,) if passive == 0.5 else (7.0,)
        )
