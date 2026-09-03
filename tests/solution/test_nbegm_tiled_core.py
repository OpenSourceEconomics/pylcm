"""A ride-along NB-EGM period solves in one tile-local core.

The continuation read and the envelope solve run per cell block inside one compiled
body, so the expected-continuation stacks over every ride cell never exist as a
complete array: the core's inputs are the states, the filtered carries, the target
values, and the params, and its outputs are the value array, the carry, and the
policy. The two split cores stay on the kernel as an independent oracle of the same
calculation.
"""

import inspect
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.solution.period_replay import _compiler_memory_bytes
from tests.conftest import invariance_tolerances
from tests.solution._nbegm_direct_oracle import ride_along_kernel as _ride_along_kernel
from tests.test_models import nbegm_jump_ride_along_toy, nbegm_ride_along_toy


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


def _split_oracle(*, kernel: Any, context: dict[str, Any]) -> object:
    """Solve one period through the continuation core, then the envelope core."""
    split = kernel.split_cores()
    continuation_args = kernel.build_lower_args(core_key="continuation", **context)
    stacks = jax.jit(split["continuation"])(**continuation_args)
    envelope_args = dict(kernel.build_lower_args(core_key="envelope", **context))
    envelope_args["cont_value_stack"] = stacks[0]
    envelope_args["cont_marginal_stack"] = stacks[1]
    if kernel.cliff_candidates:
        envelope_args["cliff_savings_stack"] = stacks[2]
    return jax.jit(split["envelope"])(**envelope_args)


def _ride_model(**overrides: Any) -> Any:
    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=12,
        n_savings=16,
        **overrides,
    )


def test_ride_along_kernel_publishes_one_tile_local_core() -> None:
    """Production runs one core; the split pair is available only as an oracle."""
    kernel, _ = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    assert set(kernel.cores()) == {"main"}
    assert set(kernel.split_cores()) == {"continuation", "envelope"}


def test_tile_local_core_signature_has_no_materialized_continuation_stacks() -> None:
    """The stacks are compiler-local values, never core inputs."""
    kernel, _ = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    names = inspect.signature(kernel.cores()["main"]).parameters
    assert "next_regime_to_continuation" in names
    assert "next_regime_to_V_arr" in names
    assert "cont_value_stack" not in names
    assert "cont_marginal_stack" not in names
    assert "cliff_savings_stack" not in names


def test_tile_local_core_matches_the_split_oracle() -> None:
    """Fusing the two halves per cell block changes lifetimes, not the result."""
    kernel, context = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    args = kernel.build_lower_args(core_key="main", **context)
    actual = jax.jit(kernel.cores()["main"])(**args)
    _assert_same_result(
        actual=actual, expected=_split_oracle(kernel=kernel, context=context)
    )


def test_tile_local_core_matches_the_split_oracle_with_cliff_candidates() -> None:
    """A published jump schedule keeps all three continuation channels internal."""
    model = nbegm_jump_ride_along_toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=12, n_savings=16
    )
    kernel, context = _ride_along_kernel(
        model=model, params=nbegm_jump_ride_along_toy.build_params()
    )
    assert kernel.cliff_candidates
    args = kernel.build_lower_args(core_key="main", **context)
    actual = jax.jit(kernel.cores()["main"])(**args)
    _assert_same_result(
        actual=actual, expected=_split_oracle(kernel=kernel, context=context)
    )


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
    whole = jax.jit(whole_kernel.cores()["main"])(
        **whole_kernel.build_lower_args(core_key="main", **whole_context)
    )
    blocked = jax.jit(blocked_kernel.cores()["main"])(
        **blocked_kernel.build_lower_args(core_key="main", **blocked_context)
    )
    _assert_same_result(actual=blocked, expected=whole)


def test_tile_local_core_matches_the_split_oracle_under_co_map() -> None:
    """A co-mapped distributed ride state keeps the per-slice carry read."""
    params = nbegm_ride_along_toy.build_params()
    kernel, context = _ride_along_kernel(
        model=_ride_model(distributed_kind=True), params=params
    )
    args = kernel.build_lower_args(core_key="main", **context)
    actual = jax.jit(kernel.cores()["main"])(**args)
    _assert_same_result(
        actual=actual, expected=_split_oracle(kernel=kernel, context=context)
    )


def test_tile_local_core_arguments_exclude_the_continuation_stacks() -> None:
    """The compiled core's argument bytes carry no cross-core stack.

    The tile-local core takes the continuation read's inputs plus the envelope
    step's state and param inputs, never the stacks the oracle envelope core takes
    as arguments, so its argument bytes stay below the envelope core's and within
    the two oracle cores' argument bytes net of the stacks.
    """
    kernel, context = _ride_along_kernel(
        model=_ride_model(), params=nbegm_ride_along_toy.build_params()
    )
    split = kernel.split_cores()
    envelope_args = kernel.build_lower_args(core_key="envelope", **context)
    stack_bytes = sum(
        envelope_args[name].nbytes
        for name in ("cont_value_stack", "cont_marginal_stack")
    )
    tiled_bytes = _compiler_memory_bytes(
        compiled=jax.jit(kernel.cores()["main"])
        .lower(**kernel.build_lower_args(core_key="main", **context))
        .compile()
    )
    continuation_bytes = _compiler_memory_bytes(
        compiled=jax.jit(split["continuation"])
        .lower(**kernel.build_lower_args(core_key="continuation", **context))
        .compile()
    )
    envelope_bytes = _compiler_memory_bytes(
        compiled=jax.jit(split["envelope"]).lower(**envelope_args).compile()
    )
    assert tiled_bytes is not None
    assert continuation_bytes is not None
    assert envelope_bytes is not None
    assert tiled_bytes.argument_size_in_bytes is not None
    assert continuation_bytes.argument_size_in_bytes is not None
    assert envelope_bytes.argument_size_in_bytes is not None
    assert stack_bytes > 0
    assert tiled_bytes.argument_size_in_bytes < envelope_bytes.argument_size_in_bytes
    assert (
        tiled_bytes.argument_size_in_bytes
        <= continuation_bytes.argument_size_in_bytes
        + envelope_bytes.argument_size_in_bytes
        - stack_bytes
    )
