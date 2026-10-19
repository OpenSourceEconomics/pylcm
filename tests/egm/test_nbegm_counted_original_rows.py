"""Checked NB reads count genuine original rows before current-domain masking."""

import functools
import math
from collections.abc import Callable
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_query import NBEGMQuery, NBEGMQueryResult
from _lcm.egm.nbegm_step import nbegm_multi_interval_step_savings
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.continuation_target import union_free_params
from _lcm.solution.egm import _EGMCore, _EGMPeriodKernel, _RegimeMassGuardedCore
from _lcm.solution.feasibility import BoundInputStatus, ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.solvers import EGM, NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    ScalarFloat,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import RegimeId
from tests.egm.test_nbegm_conditional_query_producer import _QueryStep
from tests.solution._crra_preferences import crra_preferences

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _Operands(TypedDict):
    grid: FloatND
    value: FloatND
    marginal: FloatND
    data: EGMQueryData
    row_queries: FloatND
    row_gradients: FloatND
    passive_values: tuple[ScalarFloat, ...]
    passive_grids: tuple[Float1D, ...]
    pool: EconFunctionKwargs
    euler_gradient: ScalarFloat
    floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus]


def _resources(*, liquid: ContinuousState) -> FloatND:
    return liquid


def _unaffordable_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 4.0


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _floor() -> FloatND:
    return jnp.asarray(0.5)


def _identity(*, savings: FloatND) -> ContinuousState:
    return savings


def _terminal_landing(*, savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def _terminal(*, liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _canonical_query(
    *, resource_function: UserFunction = _resources
) -> tuple[NBEGMQuery, EconFunctionKwargs]:
    """Obtain the real NB adapter from a canonically built parent program."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions = {
        "resources": resource_function,
        "savings": _savings,
        "utility": _utility,
        "inverse_marginal_utility": _inverse,
    }
    model = Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=np.array([0.5, 1.0]))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.025, 0.25, 0.5, 1.0])
                    )
                },
                functions={**functions, "resources": _resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.0)
                },
                state_transitions={"liquid": _identity},
                solver=EGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 2.0]))
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=np.array([0.75, 1.5, 3.0]))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.025, 0.05, 0.25, 0.5, 1.0])
                    )
                },
                functions=functions,
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=_floor)
                },
                state_transitions={"liquid": _terminal_landing},
                solver=NBEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.5, 1.0, 1.5]))
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([4.0, 5.5, 10.0]))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    kernel = model._regimes["parent"].solution.period_kernels[0]
    assert isinstance(kernel, _EGMPeriodKernel)
    function = kernel.core_programs()["main"].function
    assert isinstance(function, _RegimeMassGuardedCore)
    assert isinstance(function.core, _EGMCore)
    query = function.core.next_dc_query
    assert isinstance(query, NBEGMQuery)
    assert query.model.authority.coordinate_kind == "liquid"
    assert query.model.authority.corner_coordinate_kind == "compact_interval"
    assert query.model.authority.target_periods == (1,)
    assert query.model.authority.bound.kind == "dag"
    assert query.model.authority.state_names == query.model.authority.action_names == ()
    assert query.model.intervals.source_identity == ()
    assert query.model.authority.row_capacity == 3
    flat = model._process_params({"discount_factor": 1.0}, runtime_checks=False)
    # The target's own processed namespace and clock are the exact scalar-child
    # pool projection used by _EGMArgumentBuilder, without a fabricated carry.
    projected = union_free_params(
        flat_params=flat, regime_name="parent", transition_target_names=("child",)
    )
    assert model.ages is not None
    pool = {
        **{
            name.removeprefix("child__"): value
            for name, value in projected.items()
            if name.startswith("child__")
        },
        "period": jnp.int32(1),
        "age": model.ages.values[1],
    }
    np.testing.assert_array_equal(query.read_discount_factor(pool), 1.0)
    return query, pool


def _produce(*, unaffordable: bool = False) -> tuple[Float1D, _QueryStep]:
    """Retain the accepted empty-route producer's original three-node specimen."""
    liquid = jnp.asarray([0.75, 1.5, 3.0])
    savings = jnp.asarray([0.5, 1.0, 1.5])
    continuation = jnp.log(3.0 * savings + 4.0)
    arguments = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "discount_factor": jnp.asarray(1.0),
        "preferences": crra_preferences(crra=1.0),
        "cont_value": continuation,
        "cont_marginal": 3.0 / (3.0 * savings + 4.0),
        "coh_slopes": jnp.asarray([1.0]),
        "coh_intercepts": jnp.asarray([-4.0 if unaffordable else 0.0]),
        "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        "arithmetic": "certified",
    }
    legacy = cast(
        "Callable[..., tuple[FloatND, FloatND, FloatND]]",
        nbegm_multi_interval_step_savings,
    )(**arguments)
    produced = cast("Callable[..., _QueryStep]", nbegm_multi_interval_step_savings)(
        **arguments, retain_query=True
    )
    for actual, expected in zip(
        (produced.value, produced.marginal, produced.policy), legacy, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    data = produced.query_data
    if unaffordable:
        np.testing.assert_array_equal(produced.value, -np.inf)
        np.testing.assert_array_equal(produced.marginal, 0.0)
        np.testing.assert_array_equal(
            np.isnan(np.asarray(produced.policy)), desired=True
        )
        np.testing.assert_array_equal(data.node_kind, QueryNodeKind.PAD)
    else:
        for actual, expected in zip(
            (produced.value[0], produced.marginal[0], produced.policy[0]),
            (math.log(11.0 / 8.0), 4.0, 0.25),
            strict=True,
        ):
            np.testing.assert_array_almost_equal(
                actual, expected, decimal=DECIMAL_PRECISION
            )
        np.testing.assert_array_equal(data.savings[0], 0.5)
        np.testing.assert_array_equal(data.node_kind[0], QueryNodeKind.FINITE)
        np.testing.assert_array_equal(data.owner_id[0], 0)
        np.testing.assert_array_equal(data.side_id[0], 0)
    np.testing.assert_array_equal(data.floor_used, 0.5)
    assert data.corner_continuation is not None
    np.testing.assert_array_almost_equal(
        data.corner_continuation, [math.log(5.5)], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(data.corner_kind, [QueryNodeKind.FINITE])
    np.testing.assert_array_equal(data.corner_interval_id, [0])
    np.testing.assert_array_equal(data.corner_side_id, [0])
    return liquid, produced


def test_checked_nb_reads_count_original_support_before_current_domain_masking(
    *,
    request: pytest.FixtureRequest,
) -> None:
    """Actual NB rows yield [1,1], [0,1], [0,1], [1,1] at the four raw queries."""
    grid, produced = _produce()
    query, target_pool = _canonical_query()
    first = grid[0]
    queries = jnp.asarray(
        [
            jnp.nextafter(first, -jnp.inf),
            first,
            jnp.nextafter(first, jnp.inf),
            jnp.asarray(0.5, dtype=grid.dtype),
        ]
    )
    dtype = np.asarray(grid).dtype
    bits = np.uint32 if dtype.itemsize == 4 else np.uint64
    raw_bits = np.asarray(queries).view(bits)
    assert raw_bits[0] < raw_bits[1] < raw_bits[2]
    assert bool(queries[0] > 0.5)
    operands: list[_Operands] = []
    ordinary: list[NBEGMQueryResult] = []
    for point in queries:
        pool = {**target_pool, "liquid": point}
        floor_state = query.evaluate_floor(pool=pool)
        np.testing.assert_array_equal(floor_state[0], 0.5)
        np.testing.assert_array_equal(floor_state[1], [0.5, 1.0, 1.5])
        np.testing.assert_array_equal(floor_state[2], [False, False])
        item: _Operands = {
            "grid": grid,
            "value": produced.value,
            "marginal": produced.marginal,
            "data": produced.query_data,
            "row_queries": point,
            "row_gradients": jnp.asarray(1.0, dtype=grid.dtype),
            "passive_values": (),
            "passive_grids": (),
            "pool": pool,
            "euler_gradient": jnp.asarray(1.0, dtype=grid.dtype),
            "floor_state": floor_state,
        }
        result = query.read_prepared(**item)
        assert type(result) is NBEGMQueryResult
        assert len(result) == 5
        assert not bool(result.coverage_missing)
        np.testing.assert_array_equal(result.bound_status, [False, False])
        operands.append(item)
        ordinary.append(result)
    np.testing.assert_array_almost_equal(
        [ordinary[1].value, ordinary[1].marginal, ordinary[1].savings],
        [math.log(11.0 / 8.0), 4.0, 0.5],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(
        [ordinary[3].value, ordinary[3].marginal, ordinary[3].savings],
        [-np.inf, 0.0, 0.5],
    )
    request.node.user_properties.extend(
        [
            ("actual_query_bits", ",".join(hex(int(bit)) for bit in raw_bits)),
            ("first_original_coordinate", format(float(first), ".17g")),
            ("canonical_target_period", "1"),
            ("original_floor_saving", ".5"),
            ("held_compact_W", format(math.log(5.5), ".17g")),
        ]
    )
    method = vars(type(query)).get("read_prepared_with_counts")
    assert callable(method), (
        "The actual independent NB adapter must provide counted original-row reads."
    )
    checked = cast(
        "Callable[..., tuple[NBEGMQueryResult, ContinuationReadCounts]]", method
    )
    selected = jax.jit(functools.partial(checked, query))
    counts = []
    for expected, item in zip(ordinary, operands, strict=True):
        actual, summary = selected(**item)
        assert isinstance(actual, NBEGMQueryResult)
        assert isinstance(summary, ContinuationReadCounts)
        for observed, wanted in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(observed, wanted)
        assert summary.below_first_live.shape == summary.eligible.shape == ()
        assert summary.below_first_live.dtype == summary.eligible.dtype == jnp.int32
        counts.append(np.asarray(summary))
    np.testing.assert_array_equal(counts, [[1, 1], [0, 1], [0, 1], [1, 1]])
    np.testing.assert_array_equal(np.sum(counts, axis=0), [2, 4])


def test_checked_nb_pad_rows_do_not_count_a_legal_compact_corner() -> None:
    """Genuine no-action source nodes have zero raw events despite finite recovery."""
    grid, produced = _produce(unaffordable=True)
    query, target_pool = _canonical_query(resource_function=_unaffordable_resources)
    point = jnp.asarray(6.0, dtype=grid.dtype)
    pool = {**target_pool, "liquid": point}
    item: _Operands = {
        "grid": grid,
        "value": produced.value,
        "marginal": produced.marginal,
        "data": produced.query_data,
        "row_queries": point,
        "row_gradients": jnp.asarray(1.0, dtype=grid.dtype),
        "passive_values": (),
        "passive_grids": (),
        "pool": pool,
        "euler_gradient": jnp.asarray(1.0, dtype=grid.dtype),
        "floor_state": query.evaluate_floor(pool=pool),
    }
    ordinary = query.read_prepared(**item)
    assert type(ordinary) is NBEGMQueryResult
    assert len(ordinary) == 5
    assert not bool(ordinary.coverage_missing)
    np.testing.assert_array_equal(ordinary.bound_status, [False, False])
    np.testing.assert_array_almost_equal(
        [ordinary.value, ordinary.marginal, ordinary.savings],
        [math.log(1.5) + math.log(5.5), 2.0 / 3.0, 0.5],
        decimal=DECIMAL_PRECISION,
    )
    method = vars(type(query)).get("read_prepared_with_counts")
    assert callable(method), (
        "The actual independent NB adapter must provide counted original-row reads."
    )
    checked = cast(
        "Callable[..., tuple[NBEGMQueryResult, ContinuationReadCounts]]", method
    )
    actual, counts = jax.jit(functools.partial(checked, query))(**item)
    assert isinstance(actual, NBEGMQueryResult)
    assert isinstance(counts, ContinuationReadCounts)
    for observed, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(ordinary), strict=True
    ):
        np.testing.assert_array_equal(observed, wanted)
    assert counts.below_first_live.shape == counts.eligible.shape == ()
    assert counts.below_first_live.dtype == counts.eligible.dtype == jnp.int32
    np.testing.assert_array_equal(counts, [0, 0])
