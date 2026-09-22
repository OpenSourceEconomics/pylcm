"""Chunk entry and publication use admitted copies of their exact expressions."""

import importlib

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_population_allocation_budget import (
    _forbid_concrete,
    _memory,
    _UnadmittedAllocationError,
)


@pytest.mark.parametrize(
    "operation",
    [
        "slice_population",
        "period_age",
        "regime_mask",
        "broadcast_value",
        "empty_fallback",
        "broadcast_collective",
    ],
)
def test_chunk_operation_admits_before_allocation_and_preserves_outputs(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    values = jnp.asarray([0.0, -0.0, jnp.nan, 3.0, 4.0, 5.0, 6.0])
    scalar = jnp.asarray(-0.0)
    ids = jnp.asarray([1, 0, 1], dtype=jnp.int32)
    mask = jnp.asarray([True, False, True])
    vector = jnp.asarray([-0.0, jnp.nan])
    index = jnp.asarray(2, dtype=jnp.int32)
    originals = (values, scalar, ids, mask, vector, index)
    expected_by_name = {
        "slice_population": np.asarray(values)[1:4],
        "period_age": np.asarray(values)[3],
        "regime_mask": np.asarray([True, False, True]),
        "broadcast_value": np.asarray([-0.0, -0.0, -0.0]),
        "empty_fallback": np.asarray([False, False, False]),
        "broadcast_collective": (
            np.asarray([2, 2, 2], dtype=np.int32),
            np.tile(np.asarray(vector), (3, 1)),
        ),
    }
    arguments_by_name = {
        "slice_population": {"array": values, "start": 1, "width": 3},
        "period_age": {"values": values, "period": 3},
        "regime_mask": {"regime_ids": ids, "regime_id": index - 1},
        "broadcast_value": {"value": scalar, "n_subjects": 3},
        "empty_fallback": {"mask": mask},
        "broadcast_collective": {"indices": index, "value": vector, "n_subjects": 3},
    }
    originals = (originals, arguments_by_name[operation])
    low = _memory(inputs=originals, budget=1)
    generous = _memory(inputs=originals, budget=1_000_000)
    module = importlib.import_module("_lcm.simulation.chunk_operations")
    function = getattr(module, operation)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        with pytest.raises(ExecutionPlanningError):
            function(**arguments_by_name[operation], memory=low)
    actual = function(**arguments_by_name[operation], memory=generous)
    for observed, expected in zip(
        jax.tree.leaves(actual),
        jax.tree.leaves(expected_by_name[operation]),
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)
        if np.issubdtype(observed.dtype, np.floating):
            np.testing.assert_array_equal(np.signbit(observed), np.signbit(expected))
    np.testing.assert_array_equal(values, [0.0, -0.0, np.nan, 3.0, 4.0, 5.0, 6.0])


def test_dynamic_population_slices_share_one_compilation() -> None:
    values = jnp.arange(9, dtype=jnp.int32)
    memory = _memory(inputs=values, budget=1_000_000)
    module = importlib.import_module("_lcm.simulation.chunk_operations")
    for start in (0, 3, 6):
        actual = module.slice_population(
            array=values, start=start, width=3, memory=memory
        )
        np.testing.assert_array_equal(actual, np.arange(start, start + 3))
    assert len(memory.operations.cache) == 1
