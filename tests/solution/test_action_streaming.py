import itertools
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.solution.action_streaming import build_streaming_max_Q_over_a


def _direct_scalar_oracle(
    *,
    Q_and_F: Callable[..., tuple[object, object]],
    action_names: tuple[str, ...],
    action_grids: Mapping[str, np.ndarray],
    fixed_kwargs: Mapping[str, object],
) -> tuple[float, int, bool]:
    """Enumerate the canonical row-major product without using JAX mapping."""
    grids = [np.asarray(action_grids[name]) for name in action_names]
    coordinates = itertools.product(*grids) if grids else [()]

    best_value = -np.inf
    best_id = -1
    any_feasible = False
    has_feasible_nan = False
    for global_id, coordinate in enumerate(coordinates):
        action_kwargs = dict(zip(action_names, coordinate, strict=True))
        value, feasible = Q_and_F(**fixed_kwargs, **action_kwargs)
        value = float(np.asarray(value))
        feasible = bool(np.asarray(feasible))
        if not feasible:
            continue
        has_feasible_nan |= bool(np.isnan(value))
        if not any_feasible or value > best_value:
            best_value = value
            best_id = global_id
        any_feasible = True

    # Match GridSearch's masked-max-then-equality-argmax behavior. A feasible NaN
    # poisons the value, and its all-false equality mask publishes action id 0.
    if has_feasible_nan:
        best_value = np.nan
        best_id = 0

    return best_value, best_id, any_feasible


@pytest.mark.parametrize("block_width", [1, 2, 3, 4, 8])
def test_streaming_one_action_matches_direct_oracle_for_arbitrary_block_widths(
    block_width: int,
):
    def Q_and_F(*, choice, state, offset):
        return -((choice - state) ** 2) + offset, choice != 2

    choice = np.array([-1.0, 0.5, 2.0, 3.5, 5.0], dtype=np.float32)
    expected = _direct_scalar_oracle(
        Q_and_F=Q_and_F,
        action_names=("choice",),
        action_grids={"choice": choice},
        fixed_kwargs={"state": 3.0, "offset": 0.25},
    )
    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("choice",),
        block_width=block_width,
    )

    result = streamed(
        choice=jnp.asarray(choice), state=jnp.float32(3), offset=jnp.float32(0.25)
    )

    assert_array_equal(result.best_value, expected[0])
    assert_array_equal(result.best_global_action_id, expected[1])
    assert_array_equal(result.any_feasible, expected[2])


@pytest.mark.parametrize("block_width", [2, 5, 7, 20])
def test_streaming_multiple_actions_uses_canonical_row_major_global_ids(
    block_width: int,
):
    def Q_and_F(*, hours, saving, target):
        value = -jnp.abs(hours + saving - target)
        feasible = (hours != 1) | (saving >= 20)
        return value, feasible

    hours = np.array([0, 1, 2], dtype=np.float32)
    saving = np.array([10, 20, 40, 80], dtype=np.float32)
    expected = _direct_scalar_oracle(
        Q_and_F=Q_and_F,
        action_names=("hours", "saving"),
        action_grids={"hours": hours, "saving": saving},
        fixed_kwargs={"target": 41.0},
    )
    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("hours", "saving"),
        block_width=block_width,
    )

    result = streamed(
        hours=jnp.asarray(hours), saving=jnp.asarray(saving), target=jnp.float32(41)
    )

    assert expected[1] == 6
    assert_array_equal(result.best_value, expected[0])
    assert_array_equal(result.best_global_action_id, expected[1])
    assert_array_equal(result.any_feasible, expected[2])


def test_streaming_ties_use_smallest_global_id_across_blocks_and_partial_final_block():
    def Q_and_F(*, first, second):
        value = jnp.where(
            ((first == 0) & (second == 30)) | ((first == 1) & (second == 10)),
            9.0,
            0.0,
        )
        return value, jnp.ones((), dtype=bool)

    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("first", "second"),
        block_width=4,
    )

    result = streamed(first=jnp.array([0, 1]), second=jnp.array([10, 20, 30]))

    # C-order identities: (0, 30) -> 2 and (1, 10) -> 3.
    assert_array_equal(result.best_value, 9.0)
    assert_array_equal(result.best_global_action_id, 2)
    assert_array_equal(result.any_feasible, np.ones((), dtype=bool))


def test_streaming_distinguishes_feasible_negative_infinity_from_all_infeasible():
    def Q_and_F(*, choice, row):
        value = jnp.where(choice == 0, -jnp.inf, 5.0)
        feasible = jnp.where(row == 0, choice == 0, jnp.zeros((), dtype=bool))
        return value, feasible

    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("choice",),
        block_width=3,
    )
    choices = jnp.array([0, 1])

    result = jax.vmap(lambda row: streamed(choice=choices, row=row))(jnp.array([0, 1]))

    assert_array_equal(result.best_value, jnp.array([-jnp.inf, -jnp.inf]))
    assert_array_equal(result.best_global_action_id, jnp.array([0, -1]))
    assert_array_equal(result.any_feasible, jnp.array([True, False]))


@pytest.mark.parametrize("block_width", [1, 2, 3, 8])
def test_streaming_preserves_grid_search_feasible_nan_identity_quirk(
    block_width: int,
) -> None:
    def Q_and_F(*, choice):
        value = jnp.where(choice == 2, jnp.nan, choice + 10.0)
        feasible = choice != 0
        return value, feasible

    choices = np.arange(4, dtype=np.float32)
    expected = _direct_scalar_oracle(
        Q_and_F=Q_and_F,
        action_names=("choice",),
        action_grids={"choice": choices},
        fixed_kwargs={},
    )
    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("choice",),
        block_width=block_width,
    )

    result = jax.jit(streamed)(choice=jnp.asarray(choices))

    assert_array_equal(jnp.isnan(result.best_value), np.isnan(expected[0]))
    assert_array_equal(result.best_global_action_id, expected[1])
    assert_array_equal(result.any_feasible, expected[2])


def test_streaming_no_action_product_is_the_identity_and_is_jittable():
    def Q_and_F(*, state, feasible):
        return state * 2, feasible

    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=(),
        block_width=7,
    )
    jitted = jax.jit(streamed)

    feasible = jitted(state=jnp.float32(1.5), feasible=jnp.ones((), dtype=bool))
    infeasible = jitted(state=jnp.float32(1.5), feasible=jnp.zeros((), dtype=bool))

    assert_array_equal(feasible.best_value, 3.0)
    assert_array_equal(feasible.best_global_action_id, 0)
    assert_array_equal(feasible.any_feasible, np.ones((), dtype=bool))
    assert_array_equal(infeasible.best_value, -jnp.inf)
    assert_array_equal(infeasible.best_global_action_id, -1)
    assert_array_equal(infeasible.any_feasible, np.zeros((), dtype=bool))


def test_q_and_f_sees_scalar_action_cells_and_never_the_full_action_product():
    def Q_and_F(*, first, second):
        assert first.ndim == 0
        assert second.ndim == 0
        return first + second, jnp.ones((), dtype=bool)

    streamed = build_streaming_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("first", "second"),
        block_width=4,
    )

    closed_jaxpr = jax.make_jaxpr(streamed)(
        first=jnp.arange(3.0), second=jnp.arange(5.0)
    )
    result = jax.jit(streamed)(first=jnp.arange(3.0), second=jnp.arange(5.0))

    assert "scan" in str(closed_jaxpr)
    assert_array_equal(result.best_value, 6.0)
    assert_array_equal(result.best_global_action_id, 14)


@pytest.mark.parametrize("block_width", [0, -2])
def test_streaming_rejects_non_positive_block_width(block_width: int):
    with pytest.raises(ValueError, match="block_width must be positive"):
        build_streaming_max_Q_over_a(
            Q_and_F=lambda: (jnp.float32(0), jnp.ones((), dtype=bool)),
            action_names=(),
            block_width=block_width,
        )
