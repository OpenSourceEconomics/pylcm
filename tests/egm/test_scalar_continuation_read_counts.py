"""Scalar owned rows count one attempt for each actual paired continuation query."""

from collections.abc import Mapping
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import EGMQueryData
from _lcm.egm.scalar_query import PreparedScalarEGMQuery, ScalarEGMQuery
from _lcm.execution.core_program import CoreBuildContext, materialize_core_program
from _lcm.solution.egm import _EGMCore, _EGMPeriodKernel, _RegimeMassGuardedCore
from _lcm.solution.feasibility import ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from lcm import IrregSpacedGrid, Model
from lcm.solvers import EGM
from lcm.typing import Float1D
from tests.conftest import DECIMAL_PRECISION
from tests.egm.test_below_first_live_counts import _model, _params
from tests.egm.test_dynamic_dcegm_activation import RegimeId
from tests.solution._nbegm_direct_oracle import ride_along_kernel


class _ScalarReadInputs(TypedDict):
    query: Float1D
    grid: Float1D
    value: Float1D
    marginal: Float1D
    data: EGMQueryData
    prepared: PreparedScalarEGMQuery


def test_scalar_owned_read_counts_actual_first_node_and_rescore_activity() -> None:
    """Adjacent scalar queries count once and an inactive rescore contributes none."""
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    first = np.asarray(0.25, dtype=dtype)
    points = (
        float(np.nextafter(first, np.asarray(-np.inf, dtype=dtype))),
        float(first),
        float(np.nextafter(first, np.asarray(np.inf, dtype=dtype))),
        1.0,
    )
    seed = _model(literal=False, parent_nodes=points)
    child = seed.user_regimes["child"].replace(
        solver=EGM(savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 2.0])))
    )
    model = Model(
        regimes={**seed.user_regimes, "child": child},
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    kernel, context = ride_along_kernel(
        model=model,
        params=_params(literal=False, parent=True),
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    program = kernel.core_programs()["main"]
    assert isinstance(program.function, _RegimeMassGuardedCore)
    core = program.function.core
    assert isinstance(core, _EGMCore)
    query = core.next_query
    assert isinstance(query, ScalarEGMQuery)
    args = materialize_core_program(
        program=program, context=CoreBuildContext(**context)
    ).arguments
    carries = context["next_regime_to_continuation"]
    carry = carries["child"]
    assert isinstance(carry, EGMCarry)
    leaves = carry.leaves()
    grid, value, marginal = (
        leaves[("endog_grid",)],
        leaves[("value",)],
        leaves[("marginal_utility",)],
    )
    np.testing.assert_array_equal(grid[0], first)
    np.testing.assert_allclose(
        [value[0], marginal[0]],
        [np.log(0.25) + np.log(4.0), 4.0],
        rtol=10.0**-DECIMAL_PRECISION,
    )
    data = carry.query_data_for_transport()
    assert data is not None
    raw_pool = args["next_query_pool"]
    assert isinstance(raw_pool, Mapping)
    pool: EconFunctionKwargs = {}
    for name, item in raw_pool.items():
        assert isinstance(item, jax.Array | float | int)
        pool[name] = item
    prepared = query.prepare(pool=pool)
    operands: _ScalarReadInputs = {
        "query": jnp.asarray(points[:3]),
        "grid": grid,
        "value": value,
        "marginal": marginal,
        "data": data,
        "prepared": prepared,
    }
    ordinary = query.read_prepared(**operands)
    counted, counts = query.read_prepared_with_counts(**operands)
    for actual, expected in zip(
        jax.tree.leaves(counted), jax.tree.leaves(ordinary), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert isinstance(counts, ContinuationReadCounts)
    np.testing.assert_array_equal(counts, [1, 3])
    _, rescore = query.read_prepared_with_counts(
        **operands, active=jnp.asarray([False, True, True])
    )
    np.testing.assert_array_equal(rescore, [0, 2])
    for count in (*counts, *rescore):
        assert count.shape == ()
        assert count.dtype == jnp.int32
