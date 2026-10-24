"""Paired DC reads compare actual supported publication with its first coordinate."""

import functools
import logging
from collections.abc import Callable, Mapping
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import CoreBuildContext, materialize_core_program
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.egm import (
    _EGMCore,
    _EGMPeriodKernel,
    _read_one_row_dc_child,
    _RegimeMassGuardedCore,
)
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
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY
from lcm.solvers import DCEGM, EGM
from lcm.typing import ContinuousState, Float1D, FloatND, ScalarFloat, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    ChildRegimeId,
    RegimeId,
    _inverse,
    _lower,
    _ordinary_law,
    _savings,
    _terminal,
    _utility,
)
from tests.solution._nbegm_direct_oracle import ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _ReadOperands(TypedDict):
    grid: FloatND
    value: FloatND
    marginal: FloatND
    data: EGMQueryData
    row_queries: FloatND
    row_gradients: FloatND
    passive_values: tuple[ScalarFloat, ...]
    passive_grids: tuple[Float1D, ...]
    pool: EconFunctionKwargs
    resource_gradient: ScalarFloat
    floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus]


def _resources(*, liquid: ContinuousState) -> FloatND:
    return liquid


def _identity(*, savings: FloatND) -> ContinuousState:
    return savings


def _model(*, literal: bool, parent_nodes: tuple[float, ...] | None = None) -> Model:
    """Publish an ordinary positive DC row with identity parent resource units."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    regimes = {
        "child": ConsumptionSavingsRegime(
            liquid=margin,
            states={"liquid": IrregSpacedGrid(points=np.array([0.25, 0.5, 1.0, 2.0]))},
            actions={
                "consumption": IrregSpacedGrid(
                    points=np.array([0.025, 0.05, 0.25, 0.5, 1.0])
                )
            },
            functions={
                "utility": _utility,
                "resources": _resources,
                "savings": _savings,
                "inverse_marginal_utility": _inverse,
            },
            constraints={
                "borrowing": post_decision_lower_bound(
                    margin=margin, lower=0.0 if literal else _lower
                )
            },
            state_transitions={"liquid": _ordinary_law},
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 2.0])),
                n_constrained_points=4,
            ),
        ),
        "done": Regime(
            states={"liquid": IrregSpacedGrid(points=np.array([4.0, 5.5, 10.0]))},
            functions={"utility": _terminal},
        ),
    }
    if parent_nodes is not None:
        regimes["parent"] = ConsumptionSavingsRegime(
            liquid=margin,
            states={"liquid": IrregSpacedGrid(points=np.array([0.5, 1.0]))},
            actions={
                "consumption": IrregSpacedGrid(points=np.array([0.025, 0.25, 0.5, 1.0]))
            },
            functions={
                "utility": _utility,
                "resources": _resources,
                "savings": _savings,
                "inverse_marginal_utility": _inverse,
            },
            constraints={
                "borrowing": post_decision_lower_bound(
                    margin=margin, lower=parent_nodes[0]
                )
            },
            state_transitions={"liquid": _identity},
            solver=EGM(savings_grid=IrregSpacedGrid(points=np.array(parent_nodes))),
        )
    return Model(
        regimes=regimes,
        ages=AgeGrid(
            start=0, inclusive_stop=1 if parent_nodes is None else 2, step="Y"
        ),
        edges={"child": {"done": 0}}
        if parent_nodes is None
        else {"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "child" if parent_nodes is None else "parent"},
        regime_id_class=ChildRegimeId if parent_nodes is None else RegimeId,
        enable_jit=False,
    )


def _params(*, literal: bool, parent: bool = False) -> UserParams:
    return {
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **({} if literal else {"borrowing_lower": {"floor": 0.0}}),
        },
        "done": {},
        **(
            {"parent": {"koopmans_aggregator": {"discount_factor": 1.0}}}
            if parent
            else {}
        ),
    }


def _assert_first_tuple(*, carry: EGMCarry, policy: EGMSimPolicy) -> float:
    """A real constrained first tuple precedes the independently inverted Euler node."""
    leaves = carry.leaves()
    grid = np.asarray(leaves[("endog_grid",)])
    value = np.asarray(leaves[("value",)])
    marginal = np.asarray(leaves[("marginal_utility",)])
    consumption = np.asarray(policy.policy)
    assert grid.ndim == value.ndim == marginal.ndim == consumption.ndim == 1
    euler_first = np.asarray(4.0 / 3.0, dtype=grid.dtype)
    expected_first = euler_first * np.asarray(1e-4, dtype=grid.dtype)
    np.testing.assert_array_equal(grid[0], expected_first)
    assert 0.0 < grid[0] < euler_first
    np.testing.assert_array_equal(consumption[0], grid[0])
    np.testing.assert_allclose(
        [value[0], marginal[0]],
        [np.log(consumption[0]) + np.log(4.0), 1.0 / consumption[0]],
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0.0,
    )
    data = carry.query_data_for_transport()
    if data is not None:
        np.testing.assert_array_equal(data.savings[0], 0.0)
        np.testing.assert_array_equal(data.node_kind[0], QueryNodeKind.FINITE)
        assert data.corner_continuation is not None
        np.testing.assert_array_almost_equal(
            data.corner_continuation[0], np.log(4.0), decimal=DECIMAL_PRECISION
        )
    return float(grid[0])


def _produced_row(*, literal: bool) -> tuple[EGMCarry, float]:
    model = _model(literal=literal)
    kernel, context = ride_along_kernel(
        model=model, params=_params(literal=literal), regime_name="child", period=0
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    program = kernel.core_programs()["replay"]
    assert isinstance(program.function, _EGMStep)
    assert program.function.pieces.feasibility_func is None
    output = kernel(
        compiled_cores={"replay": program.function},
        **context,
        logger=logging.getLogger(__name__),
    )
    carry = output.continuations[EGM_CONTINUATION]
    policy = output.replay[SIMULATION_POLICY]
    assert isinstance(carry, EGMCarry)
    assert isinstance(policy, EGMSimPolicy)
    return carry, _assert_first_tuple(carry=carry, policy=policy)


def test_literal_dc_first_support_is_qualified_by_its_real_producer(
    request: pytest.FixtureRequest,
) -> None:
    """A predicate-free positive literal DC publication owns its constrained prefix."""
    carry, first = _produced_row(literal=True)
    assert carry.query_data_for_transport() is None
    assert len(carry.leaves()) == 4
    request.node.user_properties.append(
        ("first_carried_coordinate", format(first, ".17g"))
    )
    request.node.user_properties.append(
        ("independent_first_euler", format(4.0 / 3.0, ".17g"))
    )


def _installed_reads(
    *, produced: EGMCarry, first: float
) -> tuple[DCEGMQuery, list[_ReadOperands], Float1D, Float1D, str]:
    """Read the actual parent candidates through its installed target adapter."""
    dtype = np.asarray(produced.leaves()[("endog_grid",)]).dtype
    first_scalar = np.asarray(first, dtype=dtype)
    queries = np.asarray(
        [
            np.nextafter(first_scalar, np.asarray(-np.inf, dtype=dtype)),
            first_scalar,
            np.nextafter(first_scalar, np.asarray(np.inf, dtype=dtype)),
        ],
        dtype=dtype,
    )
    parent_nodes = (*tuple(float(point) for point in queries), 1.0)
    model = _model(literal=False, parent_nodes=parent_nodes)
    kernel, context = ride_along_kernel(
        model=model,
        params=_params(literal=False, parent=True),
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    program = kernel.core_programs()["main"]
    assert isinstance(program.function, _RegimeMassGuardedCore)
    assert isinstance(program.function.core, _EGMCore)
    query = program.function.core.next_dc_query
    assert isinstance(query, DCEGMQuery)
    assert query.model.authority.target_periods == (1,)
    materialized = materialize_core_program(
        program=program, context=CoreBuildContext(**context)
    )
    arguments = materialized.arguments
    landed = arguments["next_liquid"]
    assert isinstance(landed, jax.Array)
    bits = np.uint32 if dtype.itemsize == 4 else np.uint64
    np.testing.assert_array_equal(
        np.asarray(landed).view(bits), np.asarray(parent_nodes, dtype=dtype).view(bits)
    )
    np.testing.assert_array_equal(arguments["marginal_return"], np.ones(4))
    landed = landed[:3]
    np.testing.assert_array_equal(np.asarray(landed).view(bits), queries.view(bits))
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    leaves = child.leaves()
    grid, value, marginal = (
        leaves[("endog_grid",)],
        leaves[("value",)],
        leaves[("marginal_utility",)],
    )
    assert isinstance(grid, jax.Array)
    assert isinstance(value, jax.Array)
    assert isinstance(marginal, jax.Array)
    np.testing.assert_array_equal(grid[0], first_scalar)
    data = child.query_data_for_transport()
    assert data is not None
    raw_pool = arguments["next_query_pool"]
    assert isinstance(raw_pool, Mapping)
    pool = cast("EconFunctionKwargs", raw_pool)
    np.testing.assert_array_equal(pool["period"], 1)
    np.testing.assert_array_equal(pool["age"], context["ages"].values[1])
    floor_state = query.evaluate_floor(pool=pool)
    np.testing.assert_array_equal(floor_state[0], 0.0)
    np.testing.assert_array_equal(floor_state[1], [0.0, 0.5, 2.0])
    np.testing.assert_array_equal(floor_state[2], [False, False])
    boundary_value, boundary_marginal, status, returned_floor = _read_one_row_dc_child(
        query=query,
        liquid=landed,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        pool=pool,
        floor_state=floor_state,
    )
    assert returned_floor is floor_state
    np.testing.assert_array_equal(status, [False, False, False])
    np.testing.assert_allclose(
        boundary_value, np.log(queries) + np.log(4.0), rtol=10.0**-DECIMAL_PRECISION
    )
    np.testing.assert_allclose(
        boundary_marginal, 1.0 / queries, rtol=10.0**-DECIMAL_PRECISION
    )
    operands: list[_ReadOperands] = [
        {
            "grid": grid,
            "value": value,
            "marginal": marginal,
            "data": data,
            "row_queries": point,
            "row_gradients": jnp.asarray(1.0, dtype=grid.dtype),
            "passive_values": (),
            "passive_grids": (),
            "pool": {**pool, "liquid": point},
            "resource_gradient": jnp.asarray(1.0, dtype=grid.dtype),
            "floor_state": floor_state,
        }
        for point in landed
    ]
    query_bits = ",".join(
        hex(int(bit)) for bit in np.asarray(arguments["next_liquid"]).view(bits)
    )
    return query, operands, boundary_value, boundary_marginal, query_bits


def test_checked_dc_pair_counts_three_actual_adjacent_queries_once(
    request: pytest.FixtureRequest,
) -> None:
    """Three eligible V/M pairs straddling first support contribute one below read."""
    produced, first = _produced_row(literal=False)
    assert produced.query_data_for_transport() is not None
    query, operands, boundary_value, boundary_marginal, query_bits = _installed_reads(
        produced=produced, first=first
    )
    ordinary_read = jax.jit(query.read_prepared)
    ordinary = [ordinary_read(**item) for item in operands]
    np.testing.assert_array_equal([result.savings for result in ordinary], np.zeros(3))
    np.testing.assert_array_equal(
        [result.coverage_missing for result in ordinary], [False, False, False]
    )
    np.testing.assert_array_equal(
        [result.bound_status for result in ordinary], [[False, False]] * 3
    )
    np.testing.assert_allclose(
        [result.value for result in ordinary],
        boundary_value,
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0.0,
    )
    np.testing.assert_allclose(
        [result.marginal for result in ordinary],
        boundary_marginal,
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0.0,
    )
    request.node.user_properties.extend(
        [
            ("first_carried_coordinate", format(first, ".17g")),
            ("independent_first_euler", format(4.0 / 3.0, ".17g")),
            (
                "actual_query_bits",
                query_bits,
            ),
            ("query_target_period", "1"),
        ]
    )

    method = vars(type(query)).get("read_prepared_with_counts")
    assert callable(method), (
        "Checked paired DC reads must return first-support counts "
        "at the actual row read."
    )
    checked = cast(
        "Callable[..., tuple[DCEGMQueryResult, ContinuationReadCounts]]", method
    )
    selected = jax.jit(functools.partial(checked, query))
    counts = []
    for expected, item in zip(ordinary, operands, strict=True):
        actual, summary = selected(**item)
        assert isinstance(actual, DCEGMQueryResult)
        assert isinstance(summary, ContinuationReadCounts)
        for found, wanted in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(found, wanted)
        assert summary.below_first_live.dtype == summary.eligible.dtype == jnp.int32
        assert summary.below_first_live.shape == summary.eligible.shape == ()
        counts.append(np.asarray(summary))
    np.testing.assert_array_equal(counts, [[1, 1], [0, 1], [0, 1]])
    np.testing.assert_array_equal(np.sum(counts, axis=0), [1, 3])
