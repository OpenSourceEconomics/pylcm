"""Finite query input integrity and associated pointwise corner tie ownership."""

import math
from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery, FiniteNEGMQueryResult
from _lcm.egm.finite_negm_query import _evaluate, _read_finite_native_row
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind, select_query_data
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    refine_envelope_exact,
)
from _lcm.typing import EconFunctionKwargs
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_germ_oracle import _emitted_row
from tests.egm.finite_germ_query_fixture import make_canonical_query

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _three_jets(
    *, function: Callable[[ScalarFloat], ScalarFloat], coordinate: ScalarFloat
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
    """Evaluate the first three local derivatives of an analytic utility."""
    first = jax.grad(function)
    second = jax.grad(first)
    return first(coordinate), second(coordinate), jax.grad(second)(coordinate)


def _bank(
    *,
    query: DCEGMQuery,
    keeper_row: bool,
    economic_rows: bool = False,
    keeper_value: FloatND | None = None,
) -> tuple[FloatND, FloatND, FloatND, EGMQueryData]:
    """Use native source geometry and separately supplied component economics."""
    capacity = query.model.authority.row_capacity
    empty = _emitted_row(capacity=capacity, empty=True)
    value = jnp.asarray(0.0) if keeper_value is None else keeper_value
    live = refine_envelope_exact(
        endog_grid=jnp.asarray([1.0, 1.5]),
        policy=jnp.asarray([0.875, 1.25]),
        value=jnp.stack((value, value)),
        savings=jnp.asarray([0.125, 0.25]),
        n_refined=capacity,
    )
    assert isinstance(live, OwnedEnvelope)
    assert int(live.n_kept) == 2
    np.testing.assert_array_equal(live.grid[:2], [1.0, 1.5])
    np.testing.assert_array_equal(live.value[:2], jnp.stack((value, value)))
    np.testing.assert_array_equal(live.savings[:2], [0.125, 0.25])
    assert bool(live.link_valid[0])

    def data_for(*, row: OwnedEnvelope) -> EGMQueryData:
        return EGMQueryData(
            savings=row.savings,
            node_kind=jnp.where(
                jnp.arange(capacity) < row.n_kept,
                QueryNodeKind.FINITE,
                QueryNodeKind.PAD,
            ).astype(jnp.int32),
            link_valid=row.link_valid,
            owner_id=row.owner_id,
            side_id=row.side_id,
            floor_used=jnp.asarray(0.0),
            corner_continuation=jnp.asarray([0.0]),
            corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
            corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
            corner_side_id=jnp.asarray([0], dtype=jnp.int32),
        )

    rows = (
        (live, live, live)
        if economic_rows
        else ((live, empty, empty) if keeper_row else (empty, empty, empty))
    )
    per_outer = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), *(data_for(row=row) for row in rows)
    )
    data = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), per_outer)
    data = replace(data, floor_used=jnp.asarray([0.0, 0.25]))
    grid = jnp.stack(tuple(row.grid for row in rows))
    values = jnp.stack(tuple(row.value for row in rows))
    marginal = jnp.where(jnp.isfinite(grid), 17.0, jnp.nan)
    if economic_rows:
        known = data.node_kind == QueryNodeKind.FINITE
        data = replace(
            data,
            node_kind=jnp.where(
                known, QueryNodeKind.ECONOMIC_MINUS_INF, data.node_kind
            ),
        )
        values = jnp.where(jnp.isfinite(grid), -jnp.inf, jnp.nan)
        marginal = jnp.where(jnp.isfinite(grid), 0.0, jnp.nan)
    return (
        jnp.stack((grid, grid)),
        jnp.stack((values, values)),
        jnp.stack((marginal, marginal)),
        data,
    )


def _read(
    *,
    query: DCEGMQuery,
    pool: EconFunctionKwargs,
    bank: tuple[FloatND, FloatND, FloatND, EGMQueryData],
    passive_coordinate: float = 0.0,
) -> FiniteNEGMQueryResult:
    grid, value, marginal, data = bank
    result = query.read_choices(
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        row_queries=jnp.asarray([1.25, 1.25]),
        row_gradients=jnp.asarray([1.0, 1.0]),
        passive_values=(jnp.asarray(passive_coordinate),),
        passive_grids=(jnp.asarray([0.0, 1.0]),),
        action_names=(),
        action_values=(),
        pool=pool,
        euler_gradient=jnp.asarray(1.0),
    )
    assert isinstance(result, FiniteNEGMQueryResult)
    assert not bool(jnp.any(result.bound_status.nonfinite_floor))
    assert not bool(jnp.any(result.bound_status.invalid_shifted_grid))
    return result


@pytest.mark.parametrize(
    "resource",
    [-math.inf, math.inf, math.nan],
    ids=["negative-inf", "positive-inf", "nan"],
)
@pytest.mark.parametrize(
    "false_predicate", [False, True], ids=["eligible", "false-predicate"]
)
def test_nonfinite_actual_resources_are_not_economic_emptiness(
    *, resource: float, false_predicate: bool
) -> None:
    """Invalid actual budgets retain a mandatory query error across predicate masks."""
    query, pool = make_canonical_query(false_predicate=false_predicate)
    assert len(query.predicates) == int(false_predicate)
    pool = {**pool, "liquid": jnp.asarray(resource)}
    if false_predicate:
        assert not bool(_evaluate(function=query.predicates[0], pool=pool))
    actual = _evaluate(
        function=query.resources, pool={**pool, "new_stock": jnp.asarray(0.0)}
    )
    if math.isnan(resource):
        assert bool(jnp.isnan(actual))
    else:
        np.testing.assert_array_equal(actual, resource)
    result = _read(query=query, pool=pool, bank=_bank(query=query, keeper_row=False))
    np.testing.assert_array_equal(result.coverage_missing, [True])


def test_false_predicate_with_finite_resources_is_economically_empty() -> None:
    """A valid finite budget with an excluded current state keeps (-inf,0)."""
    query, pool = make_canonical_query(false_predicate=True)
    assert len(query.predicates) == 1
    assert not bool(_evaluate(function=query.predicates[0], pool=pool))
    actual = _evaluate(
        function=query.resources, pool={**pool, "new_stock": jnp.asarray(0.0)}
    )
    np.testing.assert_array_equal(actual, 1.25)
    result = _read(query=query, pool=pool, bank=_bank(query=query, keeper_row=False))
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [-jnp.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])


@pytest.mark.parametrize(
    "economic_rows", [False, True], ids=["missing-interior", "known-economic-interior"]
)
def test_economic_corner_does_not_classify_missing_interior_support(
    *, economic_rows: bool
) -> None:
    """Known economic rows keep -inf while a lone -inf corner leaves coverage open."""
    query, pool = make_canonical_query()
    grid, value, marginal, data = _bank(
        query=query, keeper_row=False, economic_rows=economic_rows
    )
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    data = replace(
        data,
        corner_continuation=jnp.full_like(data.corner_continuation, -jnp.inf),
        corner_kind=jnp.full_like(data.corner_kind, QueryNodeKind.ECONOMIC_MINUS_INF),
    )
    result = _read(query=query, pool=pool, bank=(grid, value, marginal, data))
    np.testing.assert_array_equal(result.coverage_missing, [not economic_rows])
    if economic_rows:
        np.testing.assert_array_equal(result.value, [-jnp.inf])
        np.testing.assert_array_equal(result.marginal, [0.0])


def _positive_quartic(
    *, consumption: ContinuousAction, new_stock: ContinuousState
) -> FloatND:
    t = consumption - 1.0
    return (
        jnp.log(consumption)
        + (t**3 / 6.0 + t**4 / 2.0) * jnp.exp(-100.0 * t**2)
        - (new_stock - 1.0) ** 2
    )


def _negative_quartic(
    *, consumption: ContinuousAction, new_stock: ContinuousState
) -> FloatND:
    t = consumption - 1.0
    return (
        jnp.log(consumption)
        + (t**3 / 6.0 - t**4 / 2.0) * jnp.exp(-100.0 * t**2)
        - (new_stock - 1.0) ** 2
    )


@pytest.mark.parametrize(
    "flow_utility",
    [_positive_quartic, _negative_quartic],
    ids=["positive-fourth", "negative-fourth"],
)
def test_mixed_corner_exact_tie_keeps_canonical_keeper_row(
    flow_utility: Callable[..., FloatND],
) -> None:
    """A maximum tie with an actual corner preserves keeper row identity and M."""
    query, pool = make_canonical_query(flow_utility=flow_utility)
    adjusted = {**pool, "new_stock": jnp.asarray(1.0)}
    resources = _evaluate(function=query.resources, pool=adjusted)
    np.testing.assert_array_equal(resources, 1.0)
    corner = query.build_preferences(adjusted).utility(resources)
    np.testing.assert_array_equal(corner, 0.0)
    np.testing.assert_array_equal(
        query.build_preferences(adjusted).marginal_utility(resources), 1.0
    )
    # The component row has constant V but independent ordinary linear M17.
    # Its genuine source ownership is emitted above; it is not an Euler solve.
    result = _read(query=query, pool=pool, bank=_bank(query=query, keeper_row=True))
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, [17.0])


def test_same_outer_row_precedes_its_exact_corner_tie() -> None:
    """An ordinary tuple keeps its associated M when its own corner has equal V."""
    query, pool = make_canonical_query()
    chosen = {**pool, "new_stock": jnp.asarray(0.0)}
    resources = _evaluate(function=query.resources, pool=chosen)
    keeper_value = query.build_preferences(chosen).utility(resources)
    bank = _bank(query=query, keeper_row=True, keeper_value=keeper_value)
    grid, value, marginal, data = bank
    w = jnp.asarray([0.0, -100.0, -100.0])
    data = replace(
        data, corner_continuation=jnp.broadcast_to(w[None, :, None], (2, 3, 1))
    )
    result = _read(query=query, pool=pool, bank=(grid, value, marginal, data))
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, jnp.asarray([keeper_value]))
    np.testing.assert_array_equal(result.marginal, [17.0])


@pytest.mark.parametrize(
    "keeper_tied", [False, True], ids=["declared-outer-order", "keeper-first"]
)
def test_multiple_exact_corner_ties_follow_canonical_outer_order(
    *, keeper_tied: bool
) -> None:
    """Canonical outer identity owns tied corners even when another has larger M."""
    query, pool = make_canonical_query()
    grid, value, marginal, data = _bank(query=query, keeper_row=False)
    values = []
    marginals = []
    for outer in (0.0, 1.0, 2.0):
        chosen = {**pool, "new_stock": jnp.asarray(outer)}
        resources = _evaluate(function=query.resources, pool=chosen)
        preferences = query.build_preferences(chosen)
        values.append(preferences.utility(resources))
        marginals.append(preferences.marginal_utility(resources))
    w = -jnp.stack(values)
    np.testing.assert_array_equal(jnp.stack(values) + w, [0.0, 0.0, 0.0])
    if not keeper_tied:
        w = w.at[0].add(-10.0)
    data = replace(
        data, corner_continuation=jnp.broadcast_to(w[None, :, None], (2, 3, 1))
    )
    result = _read(query=query, pool=pool, bank=(grid, value, marginal, data))
    expected = marginals[0 if keeper_tied else 1]
    assert bool(marginals[2] > expected)
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, jnp.asarray([expected]))


def _pointwise_reference(*, values: tuple[float, ...]) -> int:
    """The canonical enumeration owns an exact scalar maximum tie."""
    return max(range(len(values)), key=lambda index: values[index])


def test_coincident_and_flat_oscillatory_values_have_a_pointwise_tie_owner() -> None:
    """A pointwise owner exists without an eventual punctured-neighborhood winner."""

    def perturb(t: float) -> float:
        return 0.0 if t == 0.0 else math.exp(-1.0 / t**2) * math.sin(1.0 / t)

    np.testing.assert_array_equal([perturb(0.0), -perturb(0.0)], [0.0, 0.0])
    assert _pointwise_reference(values=(0.0, perturb(0.0))) == 0
    assert _pointwise_reference(values=(0.0, -perturb(0.0))) == 0
    assert _pointwise_reference(values=(0.0, 0.0)) == 0
    # exp(-1/t²) underflows near0; the positive factor's magnitude cannot be
    # a floating ownership certificate. Its exact sign is the sine sign.
    for winding in range(1, 9):
        positive_phase = 2.0 * math.pi * winding + math.pi / 2.0
        negative_phase = 2.0 * math.pi * winding + 3.0 * math.pi / 2.0
        assert math.sin(positive_phase) > 0.0
        assert math.sin(negative_phase) < 0.0
        assert 0.0 < 1.0 / negative_phase < 1.0 / positive_phase


@pytest.mark.parametrize(
    "flow_utility",
    [_positive_quartic, _negative_quartic],
    ids=["positive-fourth", "negative-fourth"],
)
def test_full_three_jet_mixed_tie_uses_canonical_pointwise_owner(
    flow_utility: Callable[..., FloatND],
) -> None:
    """Opposed higher-order corners keep the same canonical row at an exact tie."""
    query, pool = make_canonical_query(flow_utility=flow_utility)
    grid, value, marginal, data = _bank(query=query, keeper_row=True)
    emitted = _emitted_row(capacity=query.model.authority.row_capacity, empty=False)
    np.testing.assert_array_equal(grid[0, 0], emitted.grid)
    np.testing.assert_array_equal(data.savings[0, 0], emitted.savings)
    np.testing.assert_array_equal(data.owner_id[0, 0], emitted.owner_id)
    np.testing.assert_array_equal(data.link_valid[0, 0], emitted.link_valid)
    row_m = (
        jnp.full_like(emitted.grid, jnp.nan)
        .at[:2]
        .set(jnp.asarray([43.0 / 32.0, 27.0 / 32.0]))
    )
    topology = query.model.authority.row_topology
    assert topology is not None
    raw = _read_finite_native_row(
        coordinate=jnp.asarray(1.25),
        grid=emitted.grid,
        value=emitted.value,
        marginal=row_m,
        data=select_query_data(data=data, positions=(0, 0), floor_positions=(0,)),
        owner_capacity=topology.owner_capacity,
        side_is_row_side=True,
    )
    assert bool(raw.ordinary.supported)
    np.testing.assert_array_equal(raw.ordinary.value, 0.0)
    np.testing.assert_array_equal(raw.ordinary.marginal, 35.0 / 32.0)
    np.testing.assert_array_equal(jnp.stack(raw.right_germ[1:]), [1.0, -1.0, 3.0])
    adjusted = {**pool, "new_stock": jnp.asarray(1.0)}
    preferences = query.build_preferences(adjusted)
    np.testing.assert_array_equal(preferences.utility(jnp.asarray(1.0)), 0.0)
    np.testing.assert_array_equal(
        jnp.stack(
            _three_jets(function=preferences.utility, coordinate=jnp.asarray(1.0))
        ),
        [1.0, -1.0, 3.0],
    )
    value = value.at[:, 0].set(emitted.value)
    marginal = marginal.at[:, 0].set(row_m)
    result = _read(query=query, pool=pool, bank=(grid, value, marginal, data))
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, [35.0 / 32.0])


@pytest.mark.parametrize(
    "corner_tied", [False, True], ids=["ordinary-only-germ", "corner-presence"]
)
def test_tied_corner_presence_survives_its_local_row_priority(
    *, corner_tied: bool
) -> None:
    """A locally unselected tied corner still triggers canonical global ownership."""
    query, pool = make_canonical_query()
    grid, value, marginal, data = _bank(query=query, keeper_row=True)
    capacity = query.model.authority.row_capacity
    second = refine_envelope_exact(
        endog_grid=jnp.asarray([1.0, 1.5]),
        policy=jnp.asarray([0.875, 1.25]),
        value=jnp.asarray([-0.25, 0.25]),
        savings=jnp.asarray([0.125, 0.25]),
        n_refined=capacity,
    )
    assert isinstance(second, OwnedEnvelope)
    assert int(second.n_kept) == 2
    np.testing.assert_array_equal(second.grid[:2], [1.0, 1.5])
    np.testing.assert_array_equal(second.value[:2], [-0.25, 0.25])
    assert bool(second.link_valid[0])
    known = jnp.arange(capacity) < second.n_kept
    grid = grid.at[:, 1].set(second.grid)
    value = value.at[:, 1].set(second.value)
    marginal = marginal.at[:, 1].set(jnp.where(known, 1.0, jnp.nan))
    data = replace(
        data,
        savings=data.savings.at[:, 1].set(second.savings),
        node_kind=data.node_kind.at[:, 1].set(
            jnp.where(known, QueryNodeKind.FINITE, QueryNodeKind.PAD).astype(jnp.int32)
        ),
        link_valid=data.link_valid.at[:, 1].set(second.link_valid),
        owner_id=data.owner_id.at[:, 1].set(second.owner_id),
        side_id=data.side_id.at[:, 1].set(second.side_id),
    )
    adjusted = {**pool, "new_stock": jnp.asarray(1.0)}
    np.testing.assert_array_equal(
        query.build_preferences(adjusted).utility(jnp.asarray(1.0)), 0.0
    )
    if not corner_tied:
        assert data.corner_continuation is not None
        data = replace(
            data, corner_continuation=jnp.full_like(data.corner_continuation, -100.0)
        )
    # O0 row V0,M17 and O1 row V0,M1 tie; the O1 corner also has V0.
    # Without that corner, the ordinary-only right germ chooses O1.
    result = _read(query=query, pool=pool, bank=(grid, value, marginal, data))
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, [17.0 if corner_tied else 1.0])


def test_pointwise_corner_tie_keeps_signed_passive_marginal_algebra() -> None:
    """Each node's canonical tuple is chosen before signed passive extrapolation."""
    query, pool = make_canonical_query()
    pool = {**pool, "stock": jnp.asarray(-0.5)}
    grid, value, marginal, data = _bank(query=query, keeper_row=True)
    marginal = marginal.at[1, 0].set(jnp.where(jnp.isfinite(grid[1, 0]), 19.0, jnp.nan))
    adjusted = {**pool, "new_stock": jnp.asarray(1.0)}
    resources = _evaluate(function=query.resources, pool=adjusted)
    floor, _, bound_status = query.evaluate_floor(pool=pool)
    np.testing.assert_array_equal(resources, 0.6875)
    np.testing.assert_array_equal(floor, -0.125)
    assert not bool(bound_status.nonfinite_floor)
    preferences = query.build_preferences(adjusted)
    corner = preferences.utility(resources - floor)
    w = jnp.asarray([-100.0, 0.0, -100.0]).at[1].set(-corner)
    np.testing.assert_array_equal(corner + w[1], 0.0)
    data = replace(
        data, corner_continuation=jnp.broadcast_to(w[None, :, None], (2, 3, 1))
    )
    result = _read(
        query=query,
        pool=pool,
        bank=(grid, value, marginal, data),
        passive_coordinate=-0.5,
    )
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, [1.5 * 17.0 - 0.5 * 19.0])
