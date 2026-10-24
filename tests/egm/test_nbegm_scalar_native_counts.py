"""Native scalar NB consumers count their genuine paired continuation reads."""

import logging
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.execution.core_program import CoreBuildContext, materialize_core_program
from _lcm.solution.egm import _EGMPeriodKernel, _RegimeMassGuardedCore
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    FeasibilityKernelOutput,
)
from _lcm.solution.nbegm import _NBEGMOwnedSingleAxisCore
from lcm import IrregSpacedGrid, Model
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import NBEGM
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_below_first_live_counts import _model, _params
from tests.egm.test_dynamic_dcegm_activation import RegimeId
from tests.solution._nbegm_direct_oracle import ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _nb_parent_and_child() -> Model:
    """Use the existing scalar count fixture with actual NB solvers on both nodes."""
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    first = np.asarray(0.25, dtype=dtype)
    points = (
        float(np.nextafter(first, np.asarray(-np.inf, dtype=dtype))),
        float(first),
        float(np.nextafter(first, np.asarray(np.inf, dtype=dtype))),
        1.0,
    )
    seed = _model(literal=False, parent_nodes=points)
    return Model(
        regimes={
            **seed.user_regimes,
            "parent": seed.user_regimes["parent"].replace(
                solver=NBEGM(savings_grid=IrregSpacedGrid(points=np.array(points)))
            ),
            "child": seed.user_regimes["child"].replace(
                solver=NBEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 2.0]))
                )
            ),
        },
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=False,
    )


def test_scalar_nbegm_native_checked_counts_preserve_ordinary_results() -> None:
    """Only the lower adjacent raw query is below the child's actual first node."""
    model = _nb_parent_and_child()
    kernel, context = ride_along_kernel(
        model=model,
        params=_params(literal=False, parent=True),
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    program = kernel.core_programs()["main"]
    assert isinstance(program.function, _RegimeMassGuardedCore)
    assert isinstance(program.function.core, _NBEGMOwnedSingleAxisCore)
    assert program.function.core.continuation_plan.query_targets == ("child",)
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    np.testing.assert_array_equal(child.endog_grid[0], 0.25)
    assert child.value is not None
    np.testing.assert_allclose(
        (child.value[0], child.marginal_utility[0]),
        (0.0, 4.0),
        rtol=10.0**-DECIMAL_PRECISION,
        atol=10.0**-DECIMAL_PRECISION,
    )
    materialized = materialize_core_program(
        program=program, context=CoreBuildContext(**context)
    )
    arguments = materialized.arguments
    assert isinstance(arguments["boundary_savings_targets"], jax.Array)
    assert arguments["boundary_savings_targets"].size == 0
    ordinary = jax.jit(materialized.function)(**arguments)
    assert isinstance(ordinary, BoundedResult)
    assert ordinary.continuation_read_counts is None
    assert len(ordinary.query_status or ()) == 1
    np.testing.assert_array_equal(ordinary.query_status, desired=False)

    checked = program.checked_variant
    assert checked is not None
    assert isinstance(checked.output_roles, BoundedResult)
    assert checked.output_roles.continuation_read_counts is not None
    # The invocation-local summary must be returned from its owning trace.
    with jax.checking_leaks():
        counted = jax.jit(checked.function)(**arguments)
    assert isinstance(counted, BoundedResult)
    assert isinstance(counted.continuation_read_counts, ContinuationReadCounts)
    np.testing.assert_array_equal(counted.continuation_read_counts, (1, 4))
    for actual, expected in zip(
        jax.tree.leaves(counted.output), jax.tree.leaves(ordinary.output), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(counted.bound_status, ordinary.bound_status)
    np.testing.assert_array_equal(counted.query_status, ordinary.query_status)
    for count in counted.continuation_read_counts:
        assert count.shape == ()
        assert count.dtype == jnp.int32

    published = kernel(
        compiled_cores={"main": checked.function},
        **context,
        logger=logging.getLogger(__name__),
    )
    assert type(published) is FeasibilityKernelOutput
    np.testing.assert_array_equal(published.continuation_read_counts, (1, 4))
    published_carry = cast("EGMCarry", published.continuations[EGM_CONTINUATION])
    np.testing.assert_array_equal(published.value, counted.output[0])
    for actual, expected in zip(
        jax.tree.leaves(published_carry),
        jax.tree.leaves(counted.output[1]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
