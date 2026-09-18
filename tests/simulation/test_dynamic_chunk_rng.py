"""Chunk starts share code while retaining the exact full-population RNG stream."""

import jax
import numpy as np
import pytest

from _lcm.simulation.random import generate_simulation_keys
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_population_allocation_budget import _memory


@pytest.mark.parametrize("partitionable", [False, True])
@pytest.mark.parametrize("original", [7, 9])
@pytest.mark.parametrize("width", [1, 3, 9])
def test_budgeted_chunk_windows_share_one_profile_and_preserve_every_key(
    *, partitionable: bool, original: int, width: int
) -> None:
    key = jax.random.key(41)
    memory = _memory(inputs=key, budget=1_000_000)
    with jax.threefry_partitionable(partitionable):
        for start in range(0, 9, width):
            expected_carry = key
            expected = {}
            for name in ("health", "shock"):
                split = jax.random.split(expected_carry, original + 1)
                expected_carry = split[0]
                raw = np.asarray(jax.random.key_data(split[1:]))
                padded = np.concatenate(
                    [raw, np.repeat(raw[-1:], 9 - original, axis=0)]
                )
                expected[f"key_{name}"] = padded[start : start + width]
            actual_carry, actual = generate_simulation_keys(
                key=key,
                names=["health", "shock"],
                n_initial_states=9,
                original_n_subjects=original,
                subject_slice=slice(start, start + width),
                memory=memory,
            )
            np.testing.assert_array_equal(
                jax.random.key_data(actual_carry), jax.random.key_data(expected_carry)
            )
            assert set(actual) == set(expected)
            for name, value in actual.items():
                np.testing.assert_array_equal(
                    jax.random.key_data(value), expected[name]
                )
            memory.close_unit()
    assert len(memory.operations.cache) == 1
    np.testing.assert_array_equal(jax.random.key_data(key), [0, 41])


@pytest.mark.parametrize(
    "window", [slice(-1, 2), slice(0, 10), slice(2, 2), slice(0, 3, 2)]
)
def test_dynamic_rng_windows_refuse_clamping_or_striding(window: slice) -> None:
    key = jax.random.key(41)
    memory = _memory(inputs=key, budget=1_000_000)
    with pytest.raises(ExecutionPlanningError, match="window"):
        generate_simulation_keys(
            key=key,
            names=["health"],
            n_initial_states=9,
            original_n_subjects=7,
            subject_slice=window,
            memory=memory,
        )
    assert not memory.operations.cache
