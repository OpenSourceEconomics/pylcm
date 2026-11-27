from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxAccumulator,
    CollectiveHardMaxResult,
)
from tests.solution._reduced_maximum_probe import (
    count_equalities_with_a_reduced_maximum,
)


def _scalar_oracle(
    *,
    objectives: np.ndarray,
    member_values: np.ndarray,
    feasible: np.ndarray,
    action_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each state cell without sharing the reducer's control flow.

    The winner maximizes the objective, then each member's value in member
    order, a NaN member value ranking below every number, then the smallest id.
    A feasible NaN objective makes the objective and every member value NaN at
    action zero.
    """
    state_shape = objectives.shape[:-1]
    n_members = member_values.shape[-1]
    best_objectives = np.full(state_shape, -np.inf, dtype=objectives.dtype)
    best_values = np.full(
        (*state_shape, n_members),
        -np.inf,
        dtype=member_values.dtype,
    )
    best_ids = np.full(state_shape, -1, dtype=np.int32)
    any_feasible = np.zeros(state_shape, dtype=bool)

    def rank(*, state_index: tuple[int, ...], position: int) -> tuple[float, ...]:
        values = member_values[(*state_index, position)]
        members = tuple(
            item
            for value in values
            for item in ((0.0, 0.0) if np.isnan(value) else (1.0, float(value)))
        )
        objective = float(objectives[(*state_index, position)])
        return (objective, *members, -float(action_ids[position]))

    for state_index in np.ndindex(state_shape):
        candidates = [
            position
            for position in range(objectives.shape[-1])
            if feasible[(*state_index, position)]
        ]
        if not candidates:
            continue
        any_feasible[state_index] = True
        if any(np.isnan(objectives[(*state_index, p)]) for p in candidates):
            best_objectives[state_index] = np.nan
            best_values[state_index] = np.nan
            best_ids[state_index] = 0
            continue
        winner = max(
            candidates,
            key=lambda position, index=state_index: rank(
                state_index=index, position=position
            ),
        )
        best_objectives[state_index] = objectives[(*state_index, winner)]
        best_values[state_index] = member_values[(*state_index, winner)]
        best_ids[state_index] = action_ids[winner]

    return best_objectives, best_values, best_ids, any_feasible


def _empty_accumulator(*, member_values: jax.Array) -> CollectiveHardMaxAccumulator:
    return COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        member_template=jnp.zeros(
            (*member_values.shape[:-2], member_values.shape[-1]),
            dtype=member_values.dtype,
        )
    )


def _partial(
    *,
    objectives: jax.Array,
    member_values: jax.Array,
    feasible: jax.Array,
    action_ids: jax.Array,
    block: tuple[int, ...],
) -> CollectiveHardMaxAccumulator:
    indices = jnp.asarray(block, dtype=jnp.int32)
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=_empty_accumulator(member_values=member_values),
        objectives=jnp.take(objectives, indices, axis=-1),
        member_values=jnp.take(member_values, indices, axis=-2),
        feasible=jnp.take(feasible, indices, axis=-1),
        action_ids=jnp.take(action_ids, indices),
    )


def _reduce_blocks(
    *,
    objectives: jax.Array,
    member_values: jax.Array,
    feasible: jax.Array,
    action_ids: jax.Array,
    blocks: Iterable[tuple[int, ...]],
) -> CollectiveHardMaxResult:
    accumulator = _empty_accumulator(member_values=member_values)
    for block in blocks:
        partial = _partial(
            objectives=objectives,
            member_values=member_values,
            feasible=feasible,
            action_ids=action_ids,
            block=block,
        )
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.merge(
            left=accumulator,
            right=partial,
        )
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def test_collective_hard_max_matches_scalar_oracle_for_arbitrary_block_order() -> None:
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
    member_values = np.stack(
        (
            np.arange(objectives.size, dtype=np.float32).reshape(objectives.shape),
            -np.arange(objectives.size, dtype=np.float32).reshape(objectives.shape),
        ),
        axis=-1,
    )
    expected = _scalar_oracle(
        objectives=objectives,
        member_values=member_values,
        feasible=feasible,
        action_ids=action_ids,
    )

    for blocks in (
        ((3, 1), (4,), (0, 2)),
        ((0, 2), (4,), (3, 1)),
    ):
        result = _reduce_blocks(
            objectives=jnp.asarray(objectives),
            member_values=jnp.asarray(member_values),
            feasible=jnp.asarray(feasible),
            action_ids=jnp.asarray(action_ids),
            blocks=blocks,
        )

        assert_allclose(result.best_objective, expected[0], equal_nan=True)
        assert_array_equal(result.best_member_values, expected[1])
        assert_array_equal(result.best_global_action_id, expected[2])
        assert_array_equal(result.any_feasible, expected[3])


def test_collective_hard_max_ties_use_one_global_winner_for_every_member() -> None:
    objectives = jnp.array([[3.0, 6.0, 6.0, 1.0]])
    member_values = jnp.array([[[100.0, 0.0], [2.0, 90.0], [7.0, 8.0], [0.0, 120.0]]])
    feasible = jnp.ones_like(objectives, dtype=bool)
    action_ids = jnp.array([8, 41, 5, 0], dtype=jnp.int32)
    left = _partial(
        objectives=objectives,
        member_values=member_values,
        feasible=feasible,
        action_ids=action_ids,
        block=(0, 1),
    )
    right = _partial(
        objectives=objectives,
        member_values=member_values,
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
        assert_array_equal(result.best_member_values, jnp.array([[7.0, 8.0]]))
        assert_array_equal(result.best_global_action_id, jnp.array([5]))
        assert_array_equal(result.any_feasible, jnp.array([True]))


@pytest.mark.parametrize(
    ("objectives", "member_values"),
    [
        pytest.param(
            [-0.0, 0.0], [[-5.0, 10.0], [7.0, 8.0]], id="negative-then-positive"
        ),
        pytest.param(
            [0.0, -0.0], [[-5.0, 10.0], [7.0, 8.0]], id="positive-then-negative"
        ),
        pytest.param([-0.0, -0.0], [[-5.0, 10.0], [7.0, 8.0]], id="both-negative"),
        pytest.param(
            [3.0, 3.0, 3.0],
            [[2.0, jnp.nan], [2.0, -jnp.inf], [jnp.nan, 9.0]],
            id="nan-member-value-loses",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_collective_hard_max_tied_objectives_go_to_the_next_member(
    *,
    objectives: list[float],
    member_values: list[list[float]],
    reverse: bool,
) -> None:
    """Tied objectives, `-0` and `+0` included, go to the members in order.

    A NaN member value ranks below every number. Action 1 wins in every case, in
    either block order, and publishes its own objective, signed zero included.
    """
    objective_array = jnp.asarray(objectives, dtype=jnp.float32)[jnp.newaxis, :]
    member_array = jnp.asarray(member_values, dtype=jnp.float32)[jnp.newaxis]
    blocks = tuple((position,) for position in range(len(objectives)))

    result = _reduce_blocks(
        objectives=objective_array,
        member_values=member_array,
        feasible=jnp.ones_like(objective_array, dtype=bool),
        action_ids=jnp.arange(len(objectives), dtype=jnp.int32),
        blocks=blocks[::-1] if reverse else blocks,
    )

    assert_array_equal(result.best_objective, objective_array[:, 1])
    assert_array_equal(
        jnp.signbit(result.best_objective), jnp.signbit(objective_array[:, 1])
    )
    assert_array_equal(result.best_member_values, member_array[:, 1])
    assert_array_equal(result.best_global_action_id, jnp.array([1]))
    assert_array_equal(result.any_feasible, jnp.array([True]))


def test_collective_hard_max_distinguishes_all_infeasible_from_feasible_minus_inf() -> (
    None
):
    result = _reduce_blocks(
        objectives=jnp.array([[-jnp.inf, 4.0], [-jnp.inf, 8.0]]),
        member_values=jnp.array([[[-3.0, 2.0], [4.0, 5.0]], [[7.0, 9.0], [8.0, 1.0]]]),
        feasible=jnp.array([[True, False], [False, False]]),
        action_ids=jnp.array([17, 0], dtype=jnp.int32),
        blocks=((1,), (0,)),
    )

    assert_array_equal(result.best_objective, jnp.array([-jnp.inf, -jnp.inf]))
    assert_array_equal(
        result.best_member_values,
        jnp.array([[-3.0, 2.0], [-jnp.inf, -jnp.inf]]),
    )
    assert_array_equal(result.best_global_action_id, jnp.array([17, -1]))
    assert_array_equal(result.any_feasible, jnp.array([True, False]))


def test_collective_hard_max_feasible_nan_objective_makes_every_member_nan() -> None:
    """A feasible NaN objective at a non-zero action publishes NaN for every member.

    The identity reported is action zero, as on the dense route.
    """
    objectives = np.array([[10.0, 12.0, np.nan, 7.0]], dtype=np.float32)
    feasible = np.array([[False, True, True, True]])
    action_ids = np.array([0, 7, 12, 3], dtype=np.int32)
    member_values = np.array(
        [[[101.0, -5.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=np.float32,
    )
    expected = _scalar_oracle(
        objectives=objectives,
        member_values=member_values,
        feasible=feasible,
        action_ids=action_ids,
    )

    for blocks in (((1, 2), (3,), (0,)), ((0,), (3,), (1, 2))):
        result = _reduce_blocks(
            objectives=jnp.asarray(objectives),
            member_values=jnp.asarray(member_values),
            feasible=jnp.asarray(feasible),
            action_ids=jnp.asarray(action_ids),
            blocks=blocks,
        )

        assert_array_equal(jnp.isnan(result.best_objective), jnp.array([True]))
        assert_array_equal(result.best_member_values, expected[1])
        assert_array_equal(jnp.isnan(result.best_member_values), [[True, True]])
        assert_array_equal(result.best_global_action_id, expected[2])
        assert_array_equal(result.any_feasible, expected[3])


def test_collective_hard_max_supports_jit_vmap_and_int32_global_ids() -> None:
    action_ids = jnp.array([30, 2, 17], dtype=jnp.int32)

    # keyword-only-exempt: library-callback=jax.vmap
    @jax.jit
    def reduce_one(
        objectives: jax.Array,
        member_values: jax.Array,
        feasible: jax.Array,
    ) -> CollectiveHardMaxResult:
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
            member_template=jnp.zeros(
                (member_values.shape[-1],), dtype=member_values.dtype
            )
        )
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            objectives=objectives,
            member_values=member_values,
            feasible=feasible,
            action_ids=action_ids,
        )
        return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)

    objectives = jnp.array([[1.0, 5.0, 5.0], [7.0, 9.0, 8.0]])
    member_values = jnp.array(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]],
        ]
    )
    feasible = jnp.array([[True, True, True], [True, False, True]])
    result = jax.vmap(reduce_one)(objectives, member_values, feasible)

    # The first row's objective ties at ids 2 and 17; the first member values
    # id 17 more.
    assert_array_equal(result.best_objective, jnp.array([5.0, 8.0]))
    assert_array_equal(
        result.best_member_values,
        jnp.array([[30.0, 31.0], [60.0, 61.0]]),
    )
    assert_array_equal(result.best_global_action_id, jnp.array([17, 17]))
    assert result.best_global_action_id.dtype == jnp.dtype(jnp.int32)
    assert_array_equal(result.any_feasible, jnp.array([True, True]))


def test_collective_hard_max_identity_is_not_matched_against_a_reduced_maximum() -> (
    None
):
    """The household winner and its objective come out of one reduction."""
    objectives = jnp.array([[-1650.6389, -14.865698, -14.989168, -7401.933]])
    member_values = jnp.stack([objectives, objectives + 1.0], axis=-1)

    jaxpr = jax.make_jaxpr(_partial)(
        objectives=objectives,
        member_values=member_values,
        feasible=jnp.ones_like(objectives, dtype=bool),
        action_ids=jnp.arange(4, dtype=jnp.int32),
        block=(0, 1, 2, 3),
    ).jaxpr

    assert count_equalities_with_a_reduced_maximum(jaxpr) == 0
