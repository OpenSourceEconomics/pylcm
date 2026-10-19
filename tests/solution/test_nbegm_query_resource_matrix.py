"""The ordinary NB query's actual-resource status survives every production seam.

A lane's mandatory slot is set exactly when some alternative's actual budget
is nonfinite, independent of the alternative's eligibility, economic domain,
row or corner support, stencil sign class or winning status, and of read
counting. Finite budgets, including exactly empty ones and those whose value
is economic minus infinity, stay economic. Inactive lanes, zero-mass nodes and
edges and padding slots are neutral. Every expectation comes from scalar
Python arithmetic in `_nb_query_resource_seam.literal_resources`; the observed
outcome is whether the mandatory host validation rejects the reduced slot.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.execution.reductions import WEIGHTED_EXPECTATION_REDUCTION
from _lcm.probability import is_live, is_represented_zero
from _lcm.solution import feasibility
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from lcm.typing import ContinuousState, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_nbegm_counted_original_rows import _canonical_query, _produce
from tests.solution._nb_query_resource_oracle import ResourceKind
from tests.solution._nb_query_resource_seam import (
    JOB_CODES,
    LANES,
    Lane,
    LaneConditions,
    Seam,
    build_seam,
    expected_lane_flag,
    host_rejects,
    lane_conditions,
    permute_jobs,
    read_lane,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

INVALID_KINDS = (
    ResourceKind.NAN,
    ResourceKind.POSITIVE_INFINITY,
    ResourceKind.NEGATIVE_INFINITY,
)
FINITE_KINDS = (
    ResourceKind.POSITIVE,
    ResourceKind.ZERO_AT_FLOOR,
    ResourceKind.NEGATIVE_FINITE,
)
ALL_KINDS = INVALID_KINDS + FINITE_KINDS
LANE_ORDER = (
    "finite_node",
    "exact_zero_factor",
    "positive_factors",
    "predicate_false",
    "negative_factor",
)
_TILE_WIDTH = 2

type LaneReader = Callable[..., BoundedResult[tuple[FloatND, FloatND]]]


def _seam(*, kind: ResourceKind) -> Seam:
    return build_seam(kind=kind, x64=bool(jax.config.read("jax_enable_x64")))


@functools.cache
def _jitted_reader(
    *,
    kind: ResourceKind,
    x64: bool,  # noqa: ARG001
    collect_counts: bool,
    order: tuple[int, ...] = JOB_CODES,
) -> LaneReader:
    seam = _seam(kind=kind)
    carry = (
        seam.carry
        if order == JOB_CODES
        else permute_jobs(carry=seam.carry, order=order)
    )
    return jax.jit(
        functools.partial(
            read_lane,
            seam=seam,
            carry=carry,
            jobs=order,
            collect_counts=collect_counts,
        )
    )


def _reader(
    *, kind: ResourceKind, collect_counts: bool, order: tuple[int, ...] = JOB_CODES
) -> LaneReader:
    return _jitted_reader(
        kind=kind,
        x64=bool(jax.config.read("jax_enable_x64")),
        collect_counts=collect_counts,
        order=order,
    )


def slot_of(result: BoundedResult[tuple[FloatND, FloatND]]) -> QueryInputStatus:
    if result.query_status is None or len(result.query_status) != 1:
        raise TypeError("An installed NB read must publish exactly one slot.")
    return result.query_status[0]


def lane_arrays(names: tuple[str, ...]) -> tuple[FloatND, FloatND, FloatND]:
    lanes = [LANES[name] for name in names]
    return (
        jnp.asarray([lane.liquid for lane in lanes]),
        jnp.asarray([lane.stock for lane in lanes]),
        jnp.asarray([lane.z for lane in lanes]),
    )


@pytest.mark.parametrize("collect_counts", [False, True])
@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda kind: kind.value)
def test_lane_status_matches_literal_budget_class(
    *, kind: ResourceKind, lane: str, collect_counts: bool
) -> None:
    """A lane is rejected exactly when one alternative's actual budget is nonfinite."""
    point = LANES[lane]
    result = _reader(kind=kind, collect_counts=collect_counts)(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    assert host_rejects(
        seam=_seam(kind=kind), status=slot_of(result)
    ) == expected_lane_flag(kind=kind, lane=point)


@pytest.mark.parametrize("lane", ["positive_factors", "finite_node"])
@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda kind: kind.value)
def test_eager_lane_status_matches_literal_budget_class(
    *, kind: ResourceKind, lane: str
) -> None:
    """The unjitted seam reaches the same host decision as the literal class."""
    point = LANES[lane]
    seam = _seam(kind=kind)
    result = read_lane(seam=seam, liquid=point.liquid, stock=point.stock, z=point.z)
    assert host_rejects(seam=seam, status=slot_of(result)) == expected_lane_flag(
        kind=kind, lane=point
    )


@pytest.mark.parametrize("order", [(1, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1)])
@pytest.mark.parametrize(
    "kind", [ResourceKind.NAN, ResourceKind.POSITIVE], ids=lambda kind: kind.value
)
def test_reordered_and_repeated_jobs_keep_lane_status(
    *, kind: ResourceKind, order: tuple[int, ...]
) -> None:
    """Reordering or repeating the job identities leaves the lane's status alone."""
    point = LANES["positive_factors"]
    result = _reader(kind=kind, collect_counts=False, order=order)(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    assert host_rejects(
        seam=_seam(kind=kind), status=slot_of(result)
    ) == expected_lane_flag(kind=kind, lane=point)


ACTIVE_MASKS = {
    "all_lanes": (True, True, True, True, True),
    "invalid_lanes_padded": (True, False, False, False, False),
    "last_invalid_lane_only": (True, False, False, False, True),
}


@pytest.mark.parametrize("mask", sorted(ACTIVE_MASKS))
@pytest.mark.parametrize(
    "kind", [ResourceKind.NAN, ResourceKind.POSITIVE], ids=lambda kind: kind.value
)
def test_vmapped_lanes_reduce_through_active_mask(
    *, kind: ResourceKind, mask: str
) -> None:
    """Mapped lanes flag through every active lane and never through a pad."""
    active = ACTIVE_MASKS[mask]
    liquid, stock, z = lane_arrays(LANE_ORDER)
    results = jax.vmap(_reader(kind=kind, collect_counts=False))(
        liquid=liquid, stock=stock, z=z
    )
    reduced = feasibility.reduce_query_input_status(
        statuses=results.query_status, active=jnp.asarray(active)
    )
    assert reduced is not None
    expected = any(
        is_active and expected_lane_flag(kind=kind, lane=LANES[name])
        for name, is_active in zip(LANE_ORDER, active, strict=True)
    )
    assert host_rejects(seam=_seam(kind=kind), status=reduced[0]) == expected


NODE_WEIGHTS = {
    "invalid_only_in_partial_final_tile": (0.0, 0.5, 0.0, 0.0, 0.5),
    "zero_mass_invalid_nodes": (0.0, 1.0, 0.0, 0.0, 0.0),
    "all_nodes_live": (0.25, 0.25, 0.25, 0.125, 0.125),
}
NODE_LANES = (
    "positive_factors",
    "finite_node",
    "predicate_false",
    "negative_factor",
    "exact_zero_factor",
)


@pytest.mark.parametrize("collect_counts", [False, True])
@pytest.mark.parametrize("weights", sorted(NODE_WEIGHTS))
@pytest.mark.parametrize(
    "kind", [ResourceKind.NAN, ResourceKind.POSITIVE], ids=lambda kind: kind.value
)
def test_scanned_node_tiles_reduce_by_live_mass(
    *, kind: ResourceKind, weights: str, collect_counts: bool
) -> None:
    """The production node fold flags live invalid nodes, even in a partial tile.

    Five nodes in tiles of two leave one padding slot in the final tile, which
    rereads node 0 at zero weight; node 0 is itself invalid for a nonfinite
    fixture and carries zero mass wherever an invalid node is meant to be dead.
    """
    node_weights = NODE_WEIGHTS[weights]
    status = fold_nodes(
        read=_reader(kind=kind, collect_counts=collect_counts),
        node_weights=node_weights,
        collect_counts=collect_counts,
    )
    expected = any(
        weight > 0.0 and expected_lane_flag(kind=kind, lane=LANES[name])
        for name, weight in zip(NODE_LANES, node_weights, strict=True)
    )
    assert host_rejects(seam=_seam(kind=kind), status=status) == expected


def fold_nodes(
    *,
    read: LaneReader,
    node_weights: tuple[float, ...],
    collect_counts: bool,
) -> QueryInputStatus:
    """Run the production weighted node fold over padded node tiles."""
    liquid, stock, z = lane_arrays(NODE_LANES)

    def read_at_nodes(
        indices: tuple[jax.Array, ...],
    ) -> BoundedResult[tuple[FloatND, FloatND]]:
        (index,) = indices
        return read(liquid=liquid[index], stock=stock[index], z=z[index])

    n_nodes = len(node_weights)
    n_blocks = -(-n_nodes // _TILE_WIDTH)
    pad = n_blocks * _TILE_WIDTH - n_nodes
    assert pad > 0
    indices = jnp.concatenate(
        [jnp.arange(n_nodes, dtype=jnp.int32), jnp.zeros(pad, dtype=jnp.int32)]
    ).reshape(n_blocks, _TILE_WIDTH)
    weights = jnp.asarray(node_weights, dtype=liquid.dtype)
    blocked_weights = jnp.concatenate(
        [weights, jnp.zeros(pad, dtype=weights.dtype)]
    ).reshape(n_blocks, _TILE_WIDTH)
    blocks: tuple[tuple[jax.Array, ...] | jax.Array, ...] = (
        (indices,),
        blocked_weights,
    )
    if collect_counts:
        activity = jnp.pad(
            ~is_represented_zero(weights), (0, pad), constant_values=False
        ).reshape(n_blocks, _TILE_WIDTH)
        blocks = (*blocks, activity)
    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=True)
    empty = reduction.initialize(value_template=jnp.zeros((), dtype=weights.dtype))
    seed = continuation_module._seed_child_query(
        output=(empty, empty),
        installed=True,
        collect_continuation_read_counts=collect_counts,
    )
    states, _ = jax.lax.scan(
        continuation_module._WeightedNodeBlockFold(
            read_at_nodes=read_at_nodes, reduction=reduction
        ),
        seed,
        blocks,
    )
    assert isinstance(states, BoundedResult)
    assert states.query_status is not None
    return states.query_status[0]


@pytest.mark.parametrize("probability", [0.0, -0.0, 0.5, "smallest_subnormal"])
def test_edge_mass_gates_only_dead_targets(*, probability: float | str) -> None:
    """A zero-probability edge is neutral; any live probability keeps the flag."""
    point = LANES["positive_factors"]
    seam = _seam(kind=ResourceKind.NAN)
    result = _reader(kind=ResourceKind.NAN, collect_counts=False)(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    dtype = seam.carry.endog_grid.dtype
    mass = jnp.asarray(
        np.finfo(dtype).smallest_subnormal
        if probability == "smallest_subnormal"
        else probability,
        dtype=dtype,
    )
    reduced = feasibility.reduce_query_input_status(
        statuses=result.query_status, active=is_live(mass)
    )
    assert reduced is not None
    expected = probability == "smallest_subnormal" or float(probability) > 0.0
    assert host_rejects(seam=seam, status=reduced[0]) == expected


def _active_rows(lane: Lane) -> int:
    """Count passive rows whose literal stencil factor is not exactly zero."""
    rows = 1
    for coordinate in (lane.stock, lane.z):
        rows *= sum(factor != 0.0 for factor in (1.0 - coordinate, coordinate))
    return rows


@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda kind: kind.value)
def test_counted_reads_ignore_budget_status(*, kind: ResourceKind, lane: str) -> None:
    """Each active passive row of each job is one read, none below the first node."""
    point = LANES[lane]
    result = _reader(kind=kind, collect_counts=True)(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    assert result.continuation_read_counts is not None
    np.testing.assert_array_equal(
        np.asarray(result.continuation_read_counts),
        [0, len(JOB_CODES) * _active_rows(point)],
    )


type _Seen = tuple[bool | float, ...]


@dataclass(frozen=True)
class _Condition:
    """One named reader condition at one lane and job."""

    lane: str
    """The lane label."""
    job: int
    """The job code read."""
    observe: Callable[[LaneConditions], _Seen]
    """Select what the label claims from the readers' view."""
    expected: _Seen
    """What the label claims."""


_CONDITIONS = {
    "negative_factor_row_without_corner": _Condition(
        lane="negative_factor",
        job=0,
        observe=lambda seen: (seen.row_supported, seen.corner_supported),
        expected=(True, False),
    ),
    "negative_factor_is_negative": _Condition(
        lane="negative_factor",
        job=0,
        observe=lambda seen: (seen.min_factor < 0.0,),
        expected=(True,),
    ),
    "exact_zero_factor_is_zero": _Condition(
        lane="exact_zero_factor",
        job=0,
        observe=lambda seen: (seen.min_abs_factor,),
        expected=(0.0,),
    ),
    "positive_factors_are_positive": _Condition(
        lane="positive_factors",
        job=0,
        observe=lambda seen: (seen.min_factor > 0.0,),
        expected=(True,),
    ),
    "supported_corner_is_economic_minus_infinity": _Condition(
        lane="positive_factors",
        job=0,
        observe=lambda seen: (seen.corner_supported, seen.corner_value),
        expected=(True, -math.inf),
    ),
    "predicate_rejects_stay": _Condition(
        lane="predicate_false",
        job=0,
        observe=lambda seen: (
            seen.eligible,
            seen.row_supported,
            seen.corner_supported,
        ),
        expected=(False, True, True),
    ),
    "predicate_admits_work": _Condition(
        lane="predicate_false",
        job=1,
        observe=lambda seen: (seen.eligible,),
        expected=(True,),
    ),
}


@pytest.mark.parametrize("condition", sorted(_CONDITIONS))
def test_lane_reaches_its_named_reader_condition(*, condition: str) -> None:
    """Each lane label names what the installed readers actually see there."""
    claim = _CONDITIONS[condition]
    seen = lane_conditions(
        seam=_seam(kind=ResourceKind.POSITIVE), lane=LANES[claim.lane], job=claim.job
    )
    assert claim.observe(seen) == claim.expected


@pytest.mark.parametrize(
    "kind", [ResourceKind.NAN, ResourceKind.POSITIVE], ids=lambda kind: kind.value
)
def test_work_alternative_wins_at_reached_lane(*, kind: ResourceKind) -> None:
    """Reading `work` alone reproduces the lane value, so `stay` loses there."""
    point = LANES["positive_factors"]
    result = _reader(kind=kind, collect_counts=False)(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    finite_work = _reader(kind=kind, collect_counts=False, order=(1,))(
        liquid=point.liquid, stock=point.stock, z=point.z
    )
    np.testing.assert_array_equal(result.output[0], finite_work.output[0])


def _absent_row_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 4.0


def _absent_row_nan_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 4.0 + jnp.sqrt(5.0 - liquid)


def _absent_row_positive_infinity_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 4.0 + 1.0 / (liquid - 6.0) ** 2


def _absent_row_negative_infinity_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 4.0 - 1.0 / (liquid - 6.0) ** 2


_ABSENT_ROW_RESOURCES = {
    "finite": (_absent_row_resources, False),
    "nan": (_absent_row_nan_resources, True),
    "positive_infinity": (_absent_row_positive_infinity_resources, True),
    "negative_infinity": (_absent_row_negative_infinity_resources, True),
}


@pytest.mark.parametrize("budget", sorted(_ABSENT_ROW_RESOURCES))
def test_supported_corner_with_absent_rows_flags_only_nonfinite_budget(
    *, budget: str
) -> None:
    """With every source row padded, the legal corner masks only finite budgets.

    The producer's specimen has no live row and a finite compact corner; the
    query reads it at liquid 6, where each nonfinite fixture is singular.
    """
    resource_function, expected = _ABSENT_ROW_RESOURCES[budget]
    grid, produced = _produce(unaffordable=True)
    query, target_pool = _canonical_query(resource_function=resource_function)
    point = jnp.asarray(6.0, dtype=grid.dtype)
    pool = {**target_pool, "liquid": point}
    result = query.read_prepared(
        grid=grid,
        value=produced.value,
        marginal=produced.marginal,
        data=produced.query_data,
        row_queries=point,
        row_gradients=jnp.asarray(1.0, dtype=grid.dtype),
        passive_values=(),
        passive_grids=(),
        pool=pool,
        euler_gradient=jnp.asarray(1.0, dtype=grid.dtype),
        floor_state=query.evaluate_floor(pool=pool),
    )
    assert bool(result.coverage_missing) == expected
