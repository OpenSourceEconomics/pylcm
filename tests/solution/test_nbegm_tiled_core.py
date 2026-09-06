"""A ride-along NB-EGM period solves in one tile-local core.

The continuation read and the envelope solve run per cell block inside one compiled
body, so the expected-continuation stacks over every ride cell never exist as a
complete array: the core's inputs are the states, the filtered carries, and the
params, and its outputs are the value array, the carry, and (for the replay
program) the policy. Agreement with the direct scalar oracle is
`test_nbegm_direct_oracle`'s job; this module pins the core's interface.
"""

import inspect
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
)
from _lcm.solution.period_replay import _compiler_memory_bytes
from tests.conftest import invariance_tolerances
from tests.solution._nbegm_direct_oracle import ride_along_kernel as _ride_along_kernel
from tests.test_models import nbegm_ride_along_toy


def _assert_same_result(*, actual: object, expected: object) -> None:
    """Check exact structure and working-dtype numerical invariance separately."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        actual_arr = np.asarray(actual_leaf)
        expected_arr = np.asarray(expected_leaf)
        assert actual_arr.shape == expected_arr.shape
        assert actual_arr.dtype == expected_arr.dtype
        if np.issubdtype(expected_arr.dtype, np.floating):
            rtol, atol = invariance_tolerances(expected_arr)
            np.testing.assert_allclose(actual_arr, expected_arr, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(actual_arr, expected_arr)


def _ride_model(**overrides: Any) -> Any:
    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=12,
        n_savings=16,
        **overrides,
    )


def _materialize(*, kernel: Any, context: Mapping[str, Any], name: str) -> Any:
    return materialize_core_program(
        program=core_program_graph(kernel=kernel)[name],
        context=CoreBuildContext(**context),
    )


def _run(*, kernel: Any, context: Mapping[str, Any], name: str) -> tuple:
    materialized = _materialize(kernel=kernel, context=context, name=name)
    return tuple(jax.jit(materialized.function)(**materialized.arguments))


@pytest.mark.parametrize("name", ["main", "replay"])
def test_tile_local_core_signature_has_no_materialized_continuation_stacks(
    name: str,
) -> None:
    """The stacks are compiler-local values, never core inputs; nor is any target V."""
    kernel, _ = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    names = inspect.signature(
        core_program_graph(kernel=kernel)[name].function
    ).parameters
    assert "next_regime_to_continuation" in names
    assert "next_regime_to_V_arr" not in names
    assert "cont_value_stack" not in names
    assert "cont_marginal_stack" not in names
    assert "cliff_savings_stack" not in names


@pytest.mark.parametrize("cell_block_size", [1, 3])
def test_tile_local_core_is_invariant_to_the_cell_block_size(
    *, cell_block_size: int
) -> None:
    """The cell block is a memory window: which cells share a pass changes nothing."""
    params = nbegm_ride_along_toy.build_params()
    whole_kernel, whole_context = _ride_along_kernel(model=_ride_model(), params=params)
    blocked_kernel, blocked_context = _ride_along_kernel(
        model=_ride_model(nbegm_overrides={"cell_block_size": cell_block_size}),
        params=params,
    )
    whole = _run(kernel=whole_kernel, context=whole_context, name="replay")
    blocked = _run(kernel=blocked_kernel, context=blocked_context, name="replay")
    _assert_same_result(actual=blocked, expected=whole)


def test_tile_local_core_arguments_are_exactly_the_declared_inputs() -> None:
    """The compiled core's argument bytes carry nothing beyond the builder's leaves.

    The builder hands the core the state grids, the filtered carries, the params,
    and the period and age; a continuation stack over every ride cell would be
    larger than all of them together, so its absence shows in the argument bytes.
    """
    kernel, context = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    materialized = _materialize(kernel=kernel, context=context, name="main")
    declared_bytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree.leaves(dict(materialized.arguments))
    )
    compiled = jax.jit(materialized.function).lower(**materialized.arguments).compile()
    report = _compiler_memory_bytes(compiled=compiled)
    assert report is not None
    assert report.argument_size_in_bytes is not None
    assert declared_bytes > 0
    assert report.argument_size_in_bytes <= declared_bytes
