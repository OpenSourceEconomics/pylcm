"""NB linear scalar values retain economic statuses and other reader conventions."""

import math
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan, bind_continuation
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.solution.continuation_target import union_free_params
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.typing import EconFunctionKwargs
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_nbegm_counted_original_rows import _canonical_query
from tests.egm.test_nbegm_literal_predicate_consumers import _create_model

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _data(*, economic: bool) -> EGMQueryData:
    return EGMQueryData(
        savings=jnp.asarray([0.6, 0.7, 0.8]),
        node_kind=jnp.asarray(
            [
                QueryNodeKind.FINITE,
                *(
                    [QueryNodeKind.ECONOMIC_MINUS_INF] * 2
                    if economic
                    else [QueryNodeKind.FINITE] * 2
                ),
            ],
            dtype=jnp.int32,
        ),
        link_valid=jnp.asarray([True, True]),
        owner_id=jnp.zeros(3, dtype=jnp.int32),
        side_id=jnp.zeros(3, dtype=jnp.int32),
        floor_used=jnp.asarray(0.5),
        corner_continuation=jnp.asarray([-100.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    ("point", "expected", "supported", "below"),
    [
        (0.75, (10.0, 2.0, 0.6), True, 0),
        (1.125, (-math.inf, 0.0, 0.65), True, 0),
        (0.6, (0.0, 0.0, 0.0), False, 1),
    ],
)
def test_scalar_linear_values_keep_economic_mass_rules(
    *,
    point: float,
    expected: tuple[float, float, float],
    supported: bool,
    below: int,
) -> None:
    """Zero economic mass is neutral; positive mass is -inf; negative mass is
    missing.
    """
    query, _pool = _canonical_query()
    arguments = {
        "grid": jnp.asarray([0.75, 1.5, 3.0]),
        "value": jnp.asarray([10.0, -jnp.inf, -jnp.inf]),
        "marginal": jnp.asarray([2.0, 0.0, 0.0]),
        "data": _data(economic=True),
        "row_queries": jnp.asarray(point),
        "row_gradients": jnp.asarray(1.0),
        "weights": (),
        "active": jnp.asarray([True]),
        "negative": jnp.asarray([False]),
        "paired_marginal_read": False,
    }
    ordinary = query._read_rows(**arguments)
    for call in (
        query._read_rows_with_counts,
        jax.jit(
            query._read_rows_with_counts, static_argnames=("paired_marginal_read",)
        ),
    ):
        actual, counts = call(**arguments)
        for observed, wanted in zip(actual, ordinary, strict=True):
            np.testing.assert_array_equal(observed, wanted)
        np.testing.assert_array_equal(counts, [below, 1])
    np.testing.assert_array_almost_equal(
        ordinary[:3], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(ordinary[3:], [supported, False])


@pytest.mark.parametrize("route", ["paired", "ride"])
def test_scalar_linear_override_preserves_paired_and_ride_hermite_values(
    *, route: str
) -> None:
    """Paired reads and actual ride authorities retain their Hermite value contract."""
    if route == "paired":
        query, _pool = _canonical_query()
        assert query.model.authority.state_names == ()
    else:
        model, _params = _create_model(predicate="island")
        kernel = model._regimes["parent"].solution.period_kernels[0]
        assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
        query = kernel.continuation_plan.child_reads["child"].query
        assert isinstance(query, NBEGMQuery)
        assert query.model.authority.state_names == ("z", "w")
    grid = jnp.asarray([0.75, 1.5, 3.0])
    arguments = {
        "grid": grid,
        "value": 10.0 + grid**2,
        "marginal": 2.0 * grid,
        "data": _data(economic=False),
        "row_queries": jnp.asarray(1.125),
        "row_gradients": jnp.asarray(1.0),
        "weights": (),
        "active": jnp.asarray([True]),
        "negative": jnp.asarray([False]),
        "paired_marginal_read": route == "paired",
    }
    ordinary = query._read_rows(**arguments)
    for call in (
        query._read_rows_with_counts,
        jax.jit(
            query._read_rows_with_counts, static_argnames=("paired_marginal_read",)
        ),
    ):
        result, counts = call(**arguments)
        for actual, expected in zip(result, ordinary, strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(counts, [0, 1])
    np.testing.assert_array_almost_equal(
        ordinary[:3], [11.265625, 2.25, 0.65], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(ordinary[3:], [True, False])


def _cubic_data() -> EGMQueryData:
    """Associate both finite cubic knots with one linked owned interval."""
    return EGMQueryData(
        savings=jnp.asarray([0.5, 0.5]),
        node_kind=jnp.full(2, QueryNodeKind.FINITE, dtype=jnp.int32),
        link_valid=jnp.asarray([True]),
        owner_id=jnp.zeros(2, dtype=jnp.int32),
        side_id=jnp.zeros(2, dtype=jnp.int32),
        floor_used=jnp.asarray(0.5),
        corner_continuation=jnp.asarray([-100.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )


def _read_cubic_query(*, route: Literal["paired", "ride"]) -> tuple[FloatND, FloatND]:
    """Read one represented cubic through the actual canonical NB row adapter."""
    if route == "paired":
        query, _pool = _canonical_query()
        assert query.model.authority.state_names == ()
    else:
        model, _params = _create_model(predicate="island")
        kernel = model._regimes["parent"].solution.period_kernels[0]
        assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
        query = kernel.continuation_plan.child_reads["child"].query
        assert isinstance(query, NBEGMQuery)
        assert query.model.authority.state_names == ("z", "w")
        assert query.model.authority.target_periods == (1,)
    arguments = {
        "grid": jnp.asarray([1.0, 2.0]),
        "value": jnp.asarray([1.0, 8.0]),
        "marginal": jnp.asarray([3.0, 12.0]),
        "data": _cubic_data(),
        "row_queries": jnp.asarray(1.5),
        "row_gradients": jnp.asarray(1.0),
        "weights": (),
        "active": jnp.asarray([True]),
        "negative": jnp.asarray([False]),
        "paired_marginal_read": route == "paired",
    }
    ordinary = query._read_rows(**arguments)
    compiled = jax.jit(query._read_rows, static_argnames=("paired_marginal_read",))(
        **arguments
    )
    for observed, expected in zip(compiled, ordinary, strict=True):
        np.testing.assert_array_equal(observed, expected)
    for call in (
        query._read_rows_with_counts,
        jax.jit(
            query._read_rows_with_counts, static_argnames=("paired_marginal_read",)
        ),
    ):
        result, counts = call(**arguments)
        for observed, expected in zip(result, ordinary, strict=True):
            np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(counts, [0, 1])
    np.testing.assert_array_equal(ordinary.savings, 0.5)
    np.testing.assert_array_equal(ordinary.supported, desired=True)
    np.testing.assert_array_equal(ordinary.invalid_numeric, desired=False)
    return ordinary.value, ordinary.marginal


def _terminal_cubic_inputs() -> tuple[ContinuationPlan, EconFunctionKwargs, EGMCarry]:
    """Use the canonical period-one child's period-two terminal continuation plan."""
    model, params = _create_model(predicate="island")
    child = model._regimes["child"]
    terminal = model._regimes["done"]
    assert not child.terminal
    assert child.active_periods == (1,)
    assert terminal.terminal
    assert terminal.active_periods == (2,)
    terminal_kernel = terminal.solution.period_kernels[2]
    assert isinstance(terminal_kernel, _TerminalCarryPeriodKernel)
    assert terminal_kernel.regime_name == "done"
    kernel = child.solution.period_kernels[1]
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    assert kernel.source_period == 1
    plan = kernel.continuation_plan
    assert plan.stateful_targets == ("done",)
    assert plan.scalar_targets == plan.query_targets == ()
    read = plan.child_reads["done"]
    assert read.query is None
    assert read.query_target is None
    assert read.passive_state_names == ("z", "w")
    assert read.row_block_shape == (2, 3)
    assert read.query_action_names == read.discrete_state_names == ()
    assert read.n_outer_candidates == 0
    assert model.ages is not None
    flat = model._process_params(params, runtime_checks=False)
    pool: EconFunctionKwargs = {
        **union_free_params(
            flat_params=flat,
            regime_name="child",
            transition_target_names=kernel.transition_target_names,
        ),
        "period": jnp.int32(1),
        "age": model.ages.values[1],
        "liquid": jnp.asarray(1.0),
        "z": jnp.asarray(0.5),
        "w": jnp.asarray(0.5),
        "job": jnp.int32(0),
    }
    np.testing.assert_array_equal(
        read.euler_state_func(**pool, savings=jnp.asarray(1.5))["next_liquid"], 1.5
    )
    np.testing.assert_array_equal(read.resources_func(liquid=jnp.asarray(1.5)), 1.5)
    shape = (*read.row_block_shape, 2)
    carry = EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.asarray([1.0, 2.0]), shape),
        value=jnp.broadcast_to(jnp.asarray([1.0, 8.0]), shape),
        marginal_utility=jnp.broadcast_to(jnp.asarray([3.0, 12.0]), shape),
        taste_shock_scale=jnp.asarray(0.0),
    )
    assert carry.query_data_for_transport() is None
    return plan, pool, carry


def _read_cubic_terminal() -> tuple[FloatND, FloatND]:
    """Exercise the actual terminal plan and its ordinary raw-carry reader."""
    plan, pool, carry = _terminal_cubic_inputs()
    results = []
    for counted in (False, True):
        continuation = bind_continuation(
            plan=plan,
            combo_pool=pool,
            next_regime_to_continuation=MappingProxyType({"done": carry}),
            dtype=carry.value.dtype,
            collect_continuation_read_counts=counted,
        )
        for call in (continuation, jax.jit(continuation)):
            result = call(jnp.asarray(1.5))
            if counted:
                assert isinstance(result, BoundedResult)
                assert result.bound_status == ()
                assert result.query_status is None
                assert result.continuation_read_counts is not None
                output = result.output
            else:
                assert isinstance(result, tuple)
                output = result
            np.testing.assert_array_equal(output, [27.0 / 8.0, 15.0 / 2.0])
            results.append(output)
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    return results[0]


@pytest.mark.parametrize("route", ["ride", "paired", "terminal"])
def test_ride_cubic_preserves_distinct_ordinary_and_paired_marginals(
    *, route: Literal["ride", "paired", "terminal"]
) -> None:
    """Hermite values, ordinary linear marginals and paired slopes stay distinct."""
    result = (
        _read_cubic_terminal()
        if route == "terminal"
        else _read_cubic_query(route=route)
    )
    expected_marginal = 27.0 / 4.0 if route == "paired" else 15.0 / 2.0
    np.testing.assert_array_equal(result, [27.0 / 8.0, expected_marginal])
