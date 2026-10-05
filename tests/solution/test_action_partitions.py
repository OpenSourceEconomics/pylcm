"""Exact action partitions: per-participant hard maxima merged in a fixed order.

The canonical action product is cut into contiguous intervals of whole action
blocks, one interval per participant. Each participant reduces its own interval
with the exact hard-max accumulator, the participants exchange only those
compact accumulators, and every participant merges them in partition order.
The published value, winning global action identity and feasibility flag equal
both a literal scalar loop over every action and the unpartitioned streamed
reduction at the same block width — bit for bit, including signed zeros,
infinities, NaNs, ties and all-infeasible cells.

The participants here are the lanes of a named `jax.vmap` axis: the reduction
reads its partition from the axis index and exchanges accumulators with
`all_gather`, which is exactly what a device mesh axis provides.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from numpy.testing import assert_array_equal

from _lcm.solution.action_reduction import HardMaxAccumulator, HardMaxResult
from _lcm.solution.action_streaming import (
    ActionPartitionLayout,
    build_partitioned_streaming_max_Q_over_a,
    build_streaming_max_Q_over_a,
    merge_partition_accumulators,
)
from lcm.typing import Float2D, ScalarFloat

_AXIS = "participants"

# Values chosen to collide: repeated finite values make ties, both zeros make
# signed-zero ties, and the non-finite values cover every IEEE special case.
_VALUE_POOL = (
    -np.inf,
    np.inf,
    np.nan,
    -0.0,
    0.0,
    -1.0,
    1.0,
    2.5,
    2.5,
    7.0,
)


def _table_Q_and_F(
    *, choice: jax.Array, cell: jax.Array, values: jax.Array, feasible: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Read one action cell's value and feasibility from an explicit table."""
    return values[cell, choice], feasible[cell, choice]


def _scalar_oracle(
    *, values: np.ndarray, feasible: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every action of every cell in canonical order, one at a time.

    GridSearch's published convention: the maximum over feasible actions with
    ties to the smallest identity; a feasible NaN publishes NaN with identity
    zero; a `{-0, +0}` maximum publishes `+0`; an empty feasible set publishes
    `-inf` with identity `-1`.
    """
    n_cells, n_actions = values.shape
    best_values = np.empty(n_cells, dtype=values.dtype)
    best_ids = np.empty(n_cells, dtype=np.int32)
    any_feasible = np.zeros(n_cells, dtype=bool)
    for cell in range(n_cells):
        best_value = -np.inf
        best_id = -1
        saw_nan = False
        saw_positive_zero = False
        for action in range(n_actions):
            if not feasible[cell, action]:
                continue
            value = values[cell, action]
            if np.isnan(value):
                saw_nan = True
                continue
            if value == 0 and not np.signbit(value):
                saw_positive_zero = True
            if best_id == -1 or value > best_value:
                best_value = value
                best_id = action
            any_feasible[cell] = True
        any_feasible[cell] |= saw_nan
        if saw_nan:
            best_value, best_id = np.nan, 0
        elif best_id != -1 and best_value == 0 and saw_positive_zero:
            best_value = 0.0
        best_values[cell] = best_value
        best_ids[cell] = best_id
    return best_values, best_ids, any_feasible


def _partitioned(
    *, n_partitions: int, block_width: int, values: np.ndarray, feasible: np.ndarray
) -> HardMaxResult:
    """Run the partitioned reduction with one named-axis lane per participant."""
    lanes = _partitioned_lanes(
        n_partitions=n_partitions,
        block_width=block_width,
        values=values,
        feasible=feasible,
    )
    _assert_every_participant_publishes_the_same_bits(lanes=lanes)
    return jax.tree.map(lambda leaf: leaf[0], lanes)


def _partitioned_lanes(
    *,
    n_partitions: int,
    block_width: int,
    values: np.ndarray | Float2D,
    feasible: np.ndarray,
) -> HardMaxResult:
    """Return jitted participant lanes for numerical transformations."""
    n_cells, n_actions = values.shape
    reduce_cell = build_partitioned_streaming_max_Q_over_a(
        Q_and_F=_table_Q_and_F,
        action_names=("choice",),
        block_width=block_width,
        n_partitions=n_partitions,
        axis_name=_AXIS,
    )

    def participant(_lane: jax.Array) -> HardMaxResult:
        return jax.vmap(
            lambda cell: reduce_cell(
                choice=jnp.arange(n_actions, dtype=jnp.int32),
                cell=cell,
                values=jnp.asarray(values),
                feasible=jnp.asarray(feasible),
            )
        )(jnp.arange(n_cells))

    return jax.jit(jax.vmap(participant, axis_name=_AXIS))(jnp.arange(n_partitions))


def _unpartitioned(
    *, block_width: int, values: np.ndarray, feasible: np.ndarray
) -> HardMaxResult:
    """Run the existing single-participant streamed reduction."""
    n_cells, n_actions = values.shape
    reduce_cell = build_streaming_max_Q_over_a(
        Q_and_F=_table_Q_and_F,
        action_names=("choice",),
        block_width=block_width,
    )
    return jax.jit(
        jax.vmap(
            lambda cell: reduce_cell(
                choice=jnp.arange(n_actions, dtype=jnp.int32),
                cell=cell,
                values=jnp.asarray(values),
                feasible=jnp.asarray(feasible),
            )
        )
    )(jnp.arange(n_cells))


def _bits(array: object) -> np.ndarray:
    """Return the exact storage bits of a floating array, NaN payloads included."""
    host = np.asarray(array)
    return host.view(np.dtype(f"u{host.dtype.itemsize}"))


def _assert_every_participant_publishes_the_same_bits(*, lanes: HardMaxResult) -> None:
    for leaf in (lanes.best_value, lanes.best_global_action_id, lanes.any_feasible):
        host = np.asarray(leaf)
        reference = host[0]
        for lane in host[1:]:
            if np.issubdtype(host.dtype, np.floating):
                assert_array_equal(_bits(lane), _bits(reference))
            else:
                assert_array_equal(lane, reference)


def _assert_results_identical(
    *, got: HardMaxResult, expected: tuple[object, object, object]
) -> None:
    assert_array_equal(_bits(got.best_value), _bits(np.asarray(expected[0])))
    assert_array_equal(np.asarray(got.best_global_action_id), np.asarray(expected[1]))
    assert_array_equal(np.asarray(got.any_feasible), np.asarray(expected[2]))


def _draw_case(
    *, seed: int, n_cells: int, n_actions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a table of colliding values and an arbitrary feasibility mask."""
    rng = np.random.default_rng(seed=seed)
    dtype = jnp.zeros(()).dtype
    pool = np.asarray(_VALUE_POOL, dtype=dtype)
    values = pool[rng.integers(0, len(pool), size=(n_cells, n_actions))]
    feasible = rng.random(size=(n_cells, n_actions)) < 0.6
    # One cell with no feasible action at all, so the empty set is always seen.
    feasible[0] = False
    assert values.dtype == dtype
    return values, feasible


_LAYOUTS = tuple(
    (n_partitions, block_width, n_actions)
    for n_partitions in (1, 2, 3, 4, 7)
    for block_width in (1, 2, 3, 5)
    for n_actions in (1, 6, 7, 13)
)


@pytest.mark.parametrize(("n_partitions", "block_width", "n_actions"), _LAYOUTS)
def test_partitioned_reduction_equals_the_scalar_oracle_bit_for_bit(
    *, n_partitions: int, block_width: int, n_actions: int
) -> None:
    values, feasible = _draw_case(
        seed=1000 * n_partitions + 10 * block_width + n_actions,
        n_cells=24,
        n_actions=n_actions,
    )

    got = _partitioned(
        n_partitions=n_partitions,
        block_width=block_width,
        values=values,
        feasible=feasible,
    )

    _assert_results_identical(
        got=got, expected=_scalar_oracle(values=values, feasible=feasible)
    )


@pytest.mark.parametrize(("n_partitions", "block_width", "n_actions"), _LAYOUTS)
def test_partitioned_reduction_equals_the_unpartitioned_stream_bit_for_bit(
    *, n_partitions: int, block_width: int, n_actions: int
) -> None:
    values, feasible = _draw_case(
        seed=7 + 1000 * n_partitions + 10 * block_width + n_actions,
        n_cells=24,
        n_actions=n_actions,
    )
    expected = _unpartitioned(block_width=block_width, values=values, feasible=feasible)

    got = _partitioned(
        n_partitions=n_partitions,
        block_width=block_width,
        values=values,
        feasible=feasible,
    )

    _assert_results_identical(
        got=got,
        expected=(
            expected.best_value,
            expected.best_global_action_id,
            expected.any_feasible,
        ),
    )


def _one_cell(*, row: list[float], mask: list[bool]) -> tuple[np.ndarray, np.ndarray]:
    dtype = jnp.zeros(()).dtype
    return np.asarray([row], dtype=dtype), np.asarray([mask], dtype=bool)


def test_partitioned_unique_max_preserves_its_analytic_parameter_derivative() -> None:
    """The winning action in a nonzero partition carries its slope through exchange."""
    coefficients = jnp.asarray([[1.0, 3.0, 2.0, 4.0, 0.0]])

    def maximum(theta: ScalarFloat) -> ScalarFloat:
        return _partitioned_lanes(
            n_partitions=3,
            block_width=2,
            values=theta * coefficients,
            feasible=np.ones((1, 5), dtype=bool),
        ).best_value[0, 0]

    derivative = jax.grad(maximum)(jnp.asarray(2.0))

    assert_array_equal(
        np.asarray(derivative), np.asarray(4.0, dtype=coefficients.dtype)
    )


# Each case names one convention; every value of interest sits in a later
# partition than a competitor, so the merge, not one participant, decides it.
_EDGE_CASES = {
    "tie_goes_to_lowest_global_id": (
        [1.0, 5.0, 2.0, 5.0, 5.0, 0.0],
        [True] * 6,
        (5.0, 1, True),
    ),
    "opposite_signed_zeros_publish_positive_zero": (
        [-0.0, -1.0, -2.0, 0.0, -3.0, -4.0],
        [True] * 6,
        (0.0, 0, True),
    ),
    "positive_zero_first_still_publishes_positive_zero": (
        [0.0, -1.0, -2.0, -0.0, -3.0, -4.0],
        [True] * 6,
        (0.0, 0, True),
    ),
    "only_negative_zeros_publish_negative_zero": (
        [-0.0, -1.0, -2.0, -0.0, -3.0, -4.0],
        [True] * 6,
        (-0.0, 0, True),
    ),
    "positive_infinity_in_a_later_partition_wins": (
        [3.0, 1.0, 2.0, 1.0, np.inf, 0.0],
        [True] * 6,
        (np.inf, 4, True),
    ),
    "feasible_negative_infinity_is_not_infeasible": (
        [-np.inf, 9.0, 9.0, 9.0, -np.inf, 9.0],
        [False, False, False, False, True, False],
        (-np.inf, 4, True),
    ),
    "all_infeasible_publishes_the_empty_set": (
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [False] * 6,
        (-np.inf, -1, False),
    ),
    "feasible_nan_in_a_later_partition_publishes_nan_and_id_zero": (
        [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
        [False, True, True, True, True, True],
        (np.nan, 0, True),
    ),
    "infeasible_nan_is_ignored": (
        [1.0, np.nan, 3.0, 4.0, np.nan, 2.0],
        [True, False, True, True, False, True],
        (4.0, 3, True),
    ),
}


@pytest.mark.parametrize("n_partitions", [2, 3, 4, 6])
@pytest.mark.parametrize("case", list(_EDGE_CASES))
def test_partitioned_reduction_keeps_each_hard_max_convention(
    *, case: str, n_partitions: int
) -> None:
    row, mask, expected = _EDGE_CASES[case]
    values, feasible = _one_cell(row=row, mask=mask)

    got = _partitioned(
        n_partitions=n_partitions, block_width=1, values=values, feasible=feasible
    )

    dtype = values.dtype
    _assert_results_identical(
        got=got,
        expected=(
            np.asarray([expected[0]], dtype=dtype),
            np.asarray([expected[1]], dtype=np.int32),
            np.asarray([expected[2]]),
        ),
    )


def test_more_participants_than_blocks_leaves_empty_partitions_that_change_nothing():
    values, feasible = _draw_case(seed=3, n_cells=8, n_actions=5)
    layout = ActionPartitionLayout(n_actions=5, block_width=2, n_partitions=7)
    empty = [
        partition
        for partition in range(7)
        if layout.action_interval(partition=partition)[0]
        == layout.action_interval(partition=partition)[1]
    ]

    got = _partitioned(n_partitions=7, block_width=2, values=values, feasible=feasible)

    assert empty == [0, 1, 3, 5]
    _assert_results_identical(
        got=got, expected=_scalar_oracle(values=values, feasible=feasible)
    )


@pytest.mark.parametrize("n_actions", range(1, 15))
@pytest.mark.parametrize("block_width", [1, 2, 3, 4])
@pytest.mark.parametrize("n_partitions", [1, 2, 3, 5])
def test_partition_intervals_are_ordered_whole_blocks_covering_every_action_once(
    *, n_actions: int, block_width: int, n_partitions: int
) -> None:
    layout = ActionPartitionLayout(
        n_actions=n_actions, block_width=block_width, n_partitions=n_partitions
    )
    intervals = [
        layout.action_interval(partition=partition) for partition in range(n_partitions)
    ]

    covered = [action for start, stop in intervals for action in range(start, stop)]
    starts_on_block_boundaries = all(
        start % block_width == 0 or start == stop for start, stop in intervals
    )
    block_counts = [
        stop - start
        for start, stop in (
            layout.block_range(partition=partition) for partition in range(n_partitions)
        )
    ]
    no_idle_participant = layout.n_blocks < n_partitions or min(block_counts) >= 1

    assert covered == list(range(n_actions))
    assert starts_on_block_boundaries
    assert max(block_counts) - min(block_counts) <= 1
    assert max(block_counts) == layout.blocks_per_partition
    assert no_idle_participant
    assert layout.n_blocks == -(-n_actions // block_width)


def _local_accumulators(
    *,
    n_partitions: int,
    block_width: int,
    values: np.ndarray,
    feasible: np.ndarray,
) -> list[HardMaxAccumulator]:
    n_cells, n_actions = values.shape
    reduce_cell = build_partitioned_streaming_max_Q_over_a(
        Q_and_F=_table_Q_and_F,
        action_names=("choice",),
        block_width=block_width,
        n_partitions=n_partitions,
        axis_name=_AXIS,
    )
    return [
        jax.vmap(
            lambda cell, partition=partition: reduce_cell.local(
                partition=jnp.int32(partition),
                choice=jnp.arange(n_actions, dtype=jnp.int32),
                cell=cell,
                values=jnp.asarray(values),
                feasible=jnp.asarray(feasible),
            )
        )(jnp.arange(n_cells))
        for partition in range(n_partitions)
    ]


def test_merging_partition_accumulators_in_any_order_publishes_the_same_bits():
    values, feasible = _draw_case(seed=11, n_cells=32, n_actions=9)
    local = _local_accumulators(
        n_partitions=4, block_width=2, values=values, feasible=feasible
    )
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *local)
    expected = _scalar_oracle(values=values, feasible=feasible)

    for order in itertools.permutations(range(4)):
        got = merge_partition_accumulators(accumulators=stacked, order=order)
        _assert_results_identical(got=got, expected=expected)


def test_padded_action_slots_never_publish_an_identity_outside_the_product():
    # Seven actions in blocks of three over three participants: the last block
    # carries two padded slots, and every value there would win if it counted.
    dtype = jnp.zeros(()).dtype
    values = np.full((1, 7), -5.0, dtype=dtype)
    feasible = np.ones((1, 7), dtype=bool)

    got = _partitioned(n_partitions=3, block_width=3, values=values, feasible=feasible)

    assert int(got.best_global_action_id[0]) == 0
    assert 0 <= int(got.best_global_action_id[0]) < 7


@pytest.mark.parametrize("n_partitions", [0, -1, True])
def test_partition_count_must_be_a_positive_exact_int(n_partitions: object) -> None:
    with pytest.raises((TypeError, ValueError), match="n_partitions"):
        build_partitioned_streaming_max_Q_over_a(
            Q_and_F=_table_Q_and_F,
            action_names=("choice",),
            block_width=2,
            n_partitions=n_partitions,  # ty: ignore[invalid-argument-type]
            axis_name=_AXIS,
        )


def test_a_float_partition_count_is_refused_by_the_type_contract() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_partitions"):
        build_partitioned_streaming_max_Q_over_a(
            Q_and_F=_table_Q_and_F,
            action_names=("choice",),
            block_width=2,
            n_partitions=1.5,  # ty: ignore[invalid-argument-type]
            axis_name=_AXIS,
        )


def test_partitioned_reduction_requires_an_action_product() -> None:
    with pytest.raises(ValueError, match="action"):
        build_partitioned_streaming_max_Q_over_a(
            Q_and_F=_table_Q_and_F,
            action_names=(),
            block_width=2,
            n_partitions=2,
            axis_name=_AXIS,
        )
