import inspect
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.solution.action_reduction import (
    HARD_MAX_REDUCTION,
    HardMaxAccumulator,
    HardMaxReduction,
    HardMaxResult,
)


def _scalar_oracle(
    *,
    values: np.ndarray,
    feasible: np.ndarray,
    action_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct scalar spelling of the existing full-array GridSearch reduction."""
    state_shape = values.shape[:-1]
    best_values = np.full(state_shape, -np.inf, dtype=values.dtype)
    best_ids = np.full(state_shape, -1, dtype=np.int32)
    any_feasible = np.zeros(state_shape, dtype=bool)

    for state_index in np.ndindex(state_shape):
        has_feasible_nan = False
        for local_index in range(values.shape[-1]):
            if not feasible[(*state_index, local_index)]:
                continue

            value = values[(*state_index, local_index)]
            action_id = int(action_ids[local_index])
            has_feasible_nan |= bool(np.isnan(value))
            if not any_feasible[state_index]:
                choose = True
            else:
                choose = value > best_values[state_index] or (
                    value == best_values[state_index]
                    and action_id < best_ids[state_index]
                )

            if choose:
                best_values[state_index] = value
                best_ids[state_index] = action_id
            any_feasible[state_index] = True

        # GridSearch computes the value with a masked max, then the identity from
        # ``argmax(feasible & (value == max_value))``. A feasible NaN therefore
        # poisons the max; every equality is false and argmax publishes position 0,
        # even when action 0 is infeasible. This preserves exact equivalence; it is not
        # a preferred mathematical NaN policy.
        if has_feasible_nan:
            best_values[state_index] = np.nan
            best_ids[state_index] = 0

    return best_ids, best_values, any_feasible


def _reduce_blocks(
    *,
    values: jax.Array,
    feasible: jax.Array,
    action_ids: jax.Array,
    blocks: Iterable[tuple[int, ...]],
) -> HardMaxResult:
    accumulator = HARD_MAX_REDUCTION.initialize(
        value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype)
    )
    for block in blocks:
        indices = jnp.asarray(block, dtype=jnp.int32)
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=jnp.take(values, indices, axis=-1),
            feasible=jnp.take(feasible, indices, axis=-1),
            action_ids=jnp.take(action_ids, indices),
        )
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def test_hard_max_matches_scalar_oracle_for_arbitrary_blocks_and_state_axes():
    values = np.array(
        [
            [[4.0, 9.0, 9.0, 2.0, 7.0], [8.0, -2.0, 6.0, 8.0, 1.0]],
            [[-3.0, 0.0, 5.0, 5.0, 4.0], [1.0, 3.0, 2.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    feasible = np.array(
        [
            [[True, True, True, True, False], [True, False, True, True, True]],
            [[False, True, True, True, True], [True, True, False, True, True]],
        ]
    )
    action_ids = np.array([40, 7, 23, 11, 99], dtype=np.int32)

    expected = _scalar_oracle(values=values, feasible=feasible, action_ids=action_ids)
    result = _reduce_blocks(
        values=jnp.asarray(values),
        feasible=jnp.asarray(feasible),
        action_ids=jnp.asarray(action_ids),
        blocks=((3, 1), (4,), (0, 2)),
    )

    assert_array_equal(result.best_global_action_id, expected[0])
    assert_array_equal(result.best_value, expected[1])
    assert_array_equal(result.any_feasible, expected[2])


def test_hard_max_distinguishes_feasible_negative_infinity_from_no_feasible_action():
    values = jnp.array([[-jnp.inf, 4.0], [-jnp.inf, 8.0]])
    feasible = jnp.array([[True, False], [False, False]])

    result = _reduce_blocks(
        values=values,
        feasible=feasible,
        action_ids=jnp.array([17, 3], dtype=jnp.int32),
        blocks=((0,), (1,)),
    )

    assert_array_equal(result.best_value, jnp.array([-jnp.inf, -jnp.inf]))
    assert_array_equal(result.best_global_action_id, jnp.array([17, -1]))
    assert_array_equal(result.any_feasible, jnp.array([True, False]))


def test_hard_max_merge_is_order_independent_and_uses_global_identity_for_ties():
    values = jnp.array([[3.0, 6.0, 6.0, 1.0]])
    feasible = jnp.ones_like(values, dtype=bool)
    action_ids = jnp.array([8, 41, 5, 2], dtype=jnp.int32)

    def partial(indices: tuple[int, ...]) -> HardMaxAccumulator:
        empty = HARD_MAX_REDUCTION.initialize(
            value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype)
        )
        index = jnp.asarray(indices, dtype=jnp.int32)
        return HARD_MAX_REDUCTION.add(
            accumulator=empty,
            values=jnp.take(values, index, axis=-1),
            feasible=jnp.take(feasible, index, axis=-1),
            action_ids=jnp.take(action_ids, index),
        )

    left = partial((0, 1))
    right = partial((2, 3))
    left_right = HARD_MAX_REDUCTION.finalize(
        accumulator=HARD_MAX_REDUCTION.merge(left=left, right=right)
    )
    right_left = HARD_MAX_REDUCTION.finalize(
        accumulator=HARD_MAX_REDUCTION.merge(left=right, right=left)
    )

    assert_array_equal(left_right.best_value, jnp.array([6.0]))
    assert_array_equal(left_right.best_global_action_id, jnp.array([5]))
    assert_array_equal(right_left.best_value, left_right.best_value)
    assert_array_equal(
        right_left.best_global_action_id, left_right.best_global_action_id
    )


def test_hard_max_preserves_grid_search_feasible_nan_identity_quirk():
    values = np.array([[10.0, 12.0, np.nan, 7.0]], dtype=np.float32)
    feasible = np.array([[False, True, True, True]])
    action_ids = np.arange(4, dtype=np.int32)
    expected = _scalar_oracle(values=values, feasible=feasible, action_ids=action_ids)
    left_to_right = _reduce_blocks(
        values=jnp.asarray(values),
        feasible=jnp.asarray(feasible),
        action_ids=jnp.asarray(action_ids),
        blocks=((0, 2), (1, 3)),
    )
    right_to_left = _reduce_blocks(
        values=jnp.asarray(values),
        feasible=jnp.asarray(feasible),
        action_ids=jnp.asarray(action_ids),
        blocks=((1, 3), (0, 2)),
    )

    for result in (left_to_right, right_to_left):
        assert_array_equal(jnp.isnan(result.best_value), jnp.isnan(expected[1]))
        assert_array_equal(result.best_global_action_id, expected[0])
        assert_array_equal(result.any_feasible, expected[2])


def test_hard_max_supports_jit_and_vmap():
    action_ids = jnp.array([30, 2, 17], dtype=jnp.int32)

    # keyword-only-exempt: library-callback=jax.vmap
    @jax.jit
    def reduce_one(values: jax.Array, feasible: jax.Array) -> HardMaxResult:
        accumulator = HARD_MAX_REDUCTION.initialize(
            value_template=jnp.zeros((), dtype=values.dtype)
        )
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=values,
            feasible=feasible,
            action_ids=action_ids,
        )
        return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)

    values = jnp.array([[1.0, 5.0, 5.0], [7.0, 9.0, 8.0]])
    feasible = jnp.array([[True, True, True], [True, False, True]])
    result = jax.vmap(reduce_one)(values, feasible)

    assert_array_equal(result.best_value, jnp.array([5.0, 8.0]))
    assert_array_equal(result.best_global_action_id, jnp.array([2, 17]))
    assert_array_equal(result.any_feasible, jnp.array([True, True]))


@pytest.mark.parametrize(
    ("dtype", "raw_ids"),
    [
        (jnp.float32, [1.9, 2.1]),
        (jnp.int64, [2**31, 7]),
        (jnp.uint32, [2**32 - 1, 7]),
    ],
    ids=["float-truncation", "int64-wrap", "uint32-wrap"],
)
def test_hard_max_rejects_action_id_dtypes_that_would_cast_lossily(
    *,
    dtype: np.dtype,
    raw_ids: list[float | int],
) -> None:
    """Global identities are int32 by contract, never silently narrowed."""
    if dtype == jnp.int64 and not jax.config.x64_enabled:
        pytest.skip("JAX cannot materialize int64 when x64 is disabled")
    action_ids = jnp.asarray(raw_ids, dtype=dtype)
    accumulator = HARD_MAX_REDUCTION.initialize(value_template=jnp.asarray(0.0))
    unwrapped_add = inspect.unwrap(HardMaxReduction.add)

    with pytest.raises(TypeError, match=r"action_ids.*int32"):
        unwrapped_add(
            HARD_MAX_REDUCTION,
            accumulator=accumulator,
            values=jnp.asarray([1.0, 2.0]),
            feasible=jnp.asarray([True, True]),
            action_ids=action_ids,
        )
