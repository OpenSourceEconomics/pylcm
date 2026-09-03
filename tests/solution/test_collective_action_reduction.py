import inspect
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxAccumulator,
    CollectiveHardMaxReduction,
    CollectiveHardMaxResult,
)


def _scalar_oracle(
    *,
    objectives: np.ndarray,
    stakeholder_values: np.ndarray,
    feasible: np.ndarray,
    action_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each state cell without sharing the reducer's control flow."""
    state_shape = objectives.shape[:-1]
    n_stakeholders = stakeholder_values.shape[-1]
    best_objectives = np.full(state_shape, -np.inf, dtype=objectives.dtype)
    best_values = np.full(
        (*state_shape, n_stakeholders),
        -np.inf,
        dtype=stakeholder_values.dtype,
    )
    best_ids = np.full(state_shape, -1, dtype=np.int32)
    any_feasible = np.zeros(state_shape, dtype=bool)

    for state_index in np.ndindex(state_shape):
        feasible_nan = False
        for local_index in range(objectives.shape[-1]):
            if not feasible[(*state_index, local_index)]:
                continue

            objective = objectives[(*state_index, local_index)]
            action_id = int(action_ids[local_index])
            feasible_nan |= bool(np.isnan(objective))
            choose = not any_feasible[state_index] or (
                objective > best_objectives[state_index]
                or (
                    objective == best_objectives[state_index]
                    and action_id < best_ids[state_index]
                )
            )
            if choose:
                best_objectives[state_index] = objective
                best_values[state_index] = stakeholder_values[
                    (*state_index, local_index)
                ]
                best_ids[state_index] = action_id
            any_feasible[state_index] = True

        # Dense GridSearch first takes the masked maximum, then argmaxes the
        # equality mask. A feasible NaN poisons that maximum, so the equality
        # mask is all false and the dense positional argmax publishes action zero.
        if feasible_nan:
            zero_position = int(np.flatnonzero(action_ids == 0)[0])
            best_objectives[state_index] = np.nan
            best_values[state_index] = stakeholder_values[(*state_index, zero_position)]
            best_ids[state_index] = 0

    return best_objectives, best_values, best_ids, any_feasible


def _empty_accumulator(
    *, stakeholder_values: jax.Array
) -> CollectiveHardMaxAccumulator:
    return COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        stakeholder_template=jnp.zeros(
            (*stakeholder_values.shape[:-2], stakeholder_values.shape[-1]),
            dtype=stakeholder_values.dtype,
        )
    )


def _partial(
    *,
    objectives: jax.Array,
    stakeholder_values: jax.Array,
    feasible: jax.Array,
    action_ids: jax.Array,
    block: tuple[int, ...],
) -> CollectiveHardMaxAccumulator:
    indices = jnp.asarray(block, dtype=jnp.int32)
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=_empty_accumulator(stakeholder_values=stakeholder_values),
        objectives=jnp.take(objectives, indices, axis=-1),
        stakeholder_values=jnp.take(stakeholder_values, indices, axis=-2),
        feasible=jnp.take(feasible, indices, axis=-1),
        action_ids=jnp.take(action_ids, indices),
    )


def _reduce_blocks(
    *,
    objectives: jax.Array,
    stakeholder_values: jax.Array,
    feasible: jax.Array,
    action_ids: jax.Array,
    blocks: Iterable[tuple[int, ...]],
) -> CollectiveHardMaxResult:
    accumulator = _empty_accumulator(stakeholder_values=stakeholder_values)
    for block in blocks:
        partial = _partial(
            objectives=objectives,
            stakeholder_values=stakeholder_values,
            feasible=feasible,
            action_ids=action_ids,
            block=block,
        )
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.merge(
            left=accumulator,
            right=partial,
        )
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def test_collective_hard_max_matches_scalar_oracle_for_arbitrary_block_order():
    objectives = np.array(
        [
            [[4.0, 9.0, 9.0, 2.0, np.nan], [8.0, -2.0, 6.0, 8.0, 1.0]],
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
    action_ids = np.array([40, 7, 23, 11, 0], dtype=np.int32)
    stakeholder_values = np.stack(
        (
            np.arange(objectives.size, dtype=np.float32).reshape(objectives.shape),
            -np.arange(objectives.size, dtype=np.float32).reshape(objectives.shape),
        ),
        axis=-1,
    )
    expected = _scalar_oracle(
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=action_ids,
    )

    for blocks in (
        ((3, 1), (4,), (0, 2)),
        ((0, 2), (4,), (3, 1)),
    ):
        result = _reduce_blocks(
            objectives=jnp.asarray(objectives),
            stakeholder_values=jnp.asarray(stakeholder_values),
            feasible=jnp.asarray(feasible),
            action_ids=jnp.asarray(action_ids),
            blocks=blocks,
        )

        assert_allclose(result.best_objective, expected[0], equal_nan=True)
        assert_array_equal(result.best_stakeholder_values, expected[1])
        assert_array_equal(result.best_global_action_id, expected[2])
        assert_array_equal(result.any_feasible, expected[3])


def test_collective_hard_max_ties_use_one_global_winner_for_every_stakeholder():
    objectives = jnp.array([[3.0, 6.0, 6.0, 1.0]])
    stakeholder_values = jnp.array(
        [[[100.0, 0.0], [2.0, 90.0], [7.0, 8.0], [0.0, 120.0]]]
    )
    feasible = jnp.ones_like(objectives, dtype=bool)
    action_ids = jnp.array([8, 41, 5, 0], dtype=jnp.int32)
    left = _partial(
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=action_ids,
        block=(0, 1),
    )
    right = _partial(
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=action_ids,
        block=(2, 3),
    )

    left_right = COLLECTIVE_HARD_MAX_REDUCTION.finalize(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.merge(left=left, right=right)
    )
    right_left = COLLECTIVE_HARD_MAX_REDUCTION.finalize(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.merge(left=right, right=left)
    )

    for result in (left_right, right_left):
        assert_array_equal(result.best_objective, jnp.array([6.0]))
        assert_array_equal(result.best_stakeholder_values, jnp.array([[7.0, 8.0]]))
        assert_array_equal(result.best_global_action_id, jnp.array([5]))
        assert_array_equal(result.any_feasible, jnp.array([True]))


@pytest.mark.parametrize(
    "objectives",
    [
        pytest.param([-0.0, 0.0], id="negative-then-positive"),
        pytest.param([0.0, -0.0], id="positive-then-negative"),
        pytest.param([-0.0, -0.0], id="both-negative"),
    ],
)
@pytest.mark.parametrize("blocks", [((0,), (1,)), ((1,), (0,))])
def test_collective_hard_max_matches_dense_signed_zero_and_keeps_one_winner(
    *,
    objectives: list[float],
    blocks: tuple[tuple[int, ...], ...],
) -> None:
    """Objective max semantics do not change the smallest-ID stakeholder readout."""
    objective_array = jnp.asarray(objectives, dtype=jnp.float32)[jnp.newaxis, :]
    stakeholder_values = jnp.array(
        [[[-5.0, 10.0], [7.0, 8.0]]],
        dtype=jnp.float32,
    )
    expected_objective = jnp.max(objective_array, axis=-1)

    result = _reduce_blocks(
        objectives=objective_array,
        stakeholder_values=stakeholder_values,
        feasible=jnp.ones_like(objective_array, dtype=bool),
        action_ids=jnp.array([0, 1], dtype=jnp.int32),
        blocks=blocks,
    )

    assert_array_equal(result.best_objective, expected_objective)
    assert_array_equal(
        jnp.signbit(result.best_objective), jnp.signbit(expected_objective)
    )
    assert_array_equal(result.best_stakeholder_values, stakeholder_values[:, 0])
    assert_array_equal(result.best_global_action_id, jnp.array([0]))
    assert_array_equal(result.any_feasible, jnp.array([True]))


def test_collective_hard_max_distinguishes_all_infeasible_from_feasible_minus_inf():
    result = _reduce_blocks(
        objectives=jnp.array([[-jnp.inf, 4.0], [-jnp.inf, 8.0]]),
        stakeholder_values=jnp.array(
            [[[-3.0, 2.0], [4.0, 5.0]], [[7.0, 9.0], [8.0, 1.0]]]
        ),
        feasible=jnp.array([[True, False], [False, False]]),
        action_ids=jnp.array([17, 0], dtype=jnp.int32),
        blocks=((1,), (0,)),
    )

    assert_array_equal(result.best_objective, jnp.array([-jnp.inf, -jnp.inf]))
    assert_array_equal(
        result.best_stakeholder_values,
        jnp.array([[-3.0, 2.0], [-jnp.inf, -jnp.inf]]),
    )
    assert_array_equal(result.best_global_action_id, jnp.array([17, -1]))
    assert_array_equal(result.any_feasible, jnp.array([True, False]))


def test_collective_hard_max_preserves_dense_action_zero_nan_readout():
    objectives = np.array([[10.0, 12.0, np.nan, 7.0]], dtype=np.float32)
    feasible = np.array([[False, True, True, True]])
    action_ids = np.array([0, 7, 12, 3], dtype=np.int32)
    stakeholder_values = np.array(
        [[[101.0, -5.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=np.float32,
    )
    expected = _scalar_oracle(
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=action_ids,
    )

    for blocks in (((1, 2), (3,), (0,)), ((0,), (3,), (1, 2))):
        result = _reduce_blocks(
            objectives=jnp.asarray(objectives),
            stakeholder_values=jnp.asarray(stakeholder_values),
            feasible=jnp.asarray(feasible),
            action_ids=jnp.asarray(action_ids),
            blocks=blocks,
        )

        assert_array_equal(jnp.isnan(result.best_objective), jnp.array([True]))
        assert_array_equal(result.best_stakeholder_values, expected[1])
        assert_array_equal(result.best_global_action_id, expected[2])
        assert_array_equal(result.any_feasible, expected[3])


def test_collective_hard_max_supports_jit_vmap_and_int32_global_ids():
    action_ids = jnp.array([30, 2, 17], dtype=jnp.int32)

    # keyword-only-exempt: library-callback=jax.vmap
    @jax.jit
    def reduce_one(
        objectives: jax.Array,
        stakeholder_values: jax.Array,
        feasible: jax.Array,
    ) -> CollectiveHardMaxResult:
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
            stakeholder_template=jnp.zeros(
                (stakeholder_values.shape[-1],), dtype=stakeholder_values.dtype
            )
        )
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            objectives=objectives,
            stakeholder_values=stakeholder_values,
            feasible=feasible,
            action_ids=action_ids,
        )
        return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)

    objectives = jnp.array([[1.0, 5.0, 5.0], [7.0, 9.0, 8.0]])
    stakeholder_values = jnp.array(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]],
        ]
    )
    feasible = jnp.array([[True, True, True], [True, False, True]])
    result = jax.vmap(reduce_one)(objectives, stakeholder_values, feasible)

    assert_array_equal(result.best_objective, jnp.array([5.0, 8.0]))
    assert_array_equal(
        result.best_stakeholder_values,
        jnp.array([[20.0, 21.0], [60.0, 61.0]]),
    )
    assert_array_equal(result.best_global_action_id, jnp.array([2, 17]))
    assert result.best_global_action_id.dtype == jnp.dtype(jnp.int32)
    assert_array_equal(result.any_feasible, jnp.array([True, True]))


def test_collective_hard_max_rejects_non_int32_global_action_ids():
    accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        stakeholder_template=jnp.zeros((2,))
    )
    unwrapped_add = inspect.unwrap(CollectiveHardMaxReduction.add)

    with pytest.raises(TypeError, match=r"action_ids.*int32"):
        unwrapped_add(
            COLLECTIVE_HARD_MAX_REDUCTION,
            accumulator=accumulator,
            objectives=jnp.array([1.0, 2.0]),
            stakeholder_values=jnp.array([[3.0, 4.0], [5.0, 6.0]]),
            feasible=jnp.array([True, True]),
            action_ids=jnp.array([0.0, 1.0]),
        )
