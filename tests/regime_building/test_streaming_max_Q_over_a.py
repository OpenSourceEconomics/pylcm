"""Full-V action streaming preserves singleton and collective GridSearch results."""

import functools
import inspect
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.logsum import logsum_and_softmax
from _lcm.regime_building.collective import ParetoWeights
from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.solution.action_streaming import (
    GridSearchEV1ActionReduction,
)


def _Q_and_F(
    *,
    first,
    second,
    row,
    shift,
    next_regime_to_V_arr,
    offset,
):
    del next_regime_to_V_arr
    tied_winner = ((first == 0) & (second == 2)) | ((first == 1) & (second == 0))
    value = jnp.where(
        tied_winner,
        9.0 + shift + offset,
        -(((first + 2 * second) - (row + shift)) ** 2) + offset,
    )
    feasible = (row != 0) & ((row != 1) | tied_winner)
    feasible = feasible & ((row != 2) | (second != 2))
    return value, feasible


@pytest.mark.parametrize("block_width", [1, 4, 5, 6])
def test_full_V_streaming_matches_dense_over_states_and_partial_action_tail(
    block_width: int,
) -> None:
    action_names = ("first", "second")
    state_names = ("row", "shift")
    batch_sizes = {"row": 2, "shift": 0}
    arguments = {
        "first": jnp.array([0, 1]),
        "second": jnp.array([0, 1, 2]),
        "row": jnp.array([0, 1, 2]),
        "shift": jnp.array([-0.5, 0.75]),
        "next_regime_to_V_arr": MappingProxyType({}),
        "offset": jnp.float32(0.25),
    }
    dense = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
    )
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
    )

    width = inspect.signature(raw_streamed).parameters["_lcm_action_block_width"]
    assert width.default is inspect.Parameter.empty

    streamed = functools.partial(
        raw_streamed,
        _lcm_action_block_width=block_width,
    )
    expected = jax.jit(dense)(**arguments)
    actual = jax.jit(streamed)(**arguments)

    assert_array_equal(actual, expected)
    assert_array_equal(actual[0], jnp.full((2,), -jnp.inf))
    assert_array_equal(actual[1], jnp.array([8.75, 10.0]))


def _signed_zero_Q_and_F(*, action, next_regime_to_V_arr):
    del next_regime_to_V_arr
    value = jnp.where(action == 0, jnp.float32(-0.0), jnp.float32(0.0))
    return value, jnp.ones((), dtype=bool)


def test_full_V_streaming_matches_dense_signed_zero_across_width_one_blocks() -> None:
    """A value tie keeps identity zero but publishes dense max's positive zero."""
    arguments = {
        "action": jnp.array([0, 1], dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType({}),
    }
    dense = get_max_Q_over_a(
        Q_and_F=_signed_zero_Q_and_F,
        batch_sizes={},
        action_names=("action",),
        state_names=(),
    )
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_signed_zero_Q_and_F,
        batch_sizes={},
        action_names=("action",),
        state_names=(),
    )
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=1)

    expected = jax.jit(dense)(**arguments)
    actual = jax.jit(streamed)(**arguments)

    assert_array_equal(actual, expected)
    assert_array_equal(jnp.signbit(actual), jnp.signbit(expected))
    assert not bool(jnp.signbit(expected))


def _collective_Q_and_F(
    *,
    first,
    second,
    row,
    next_regime_to_V_arr,
):
    del next_regime_to_V_arr
    global_id = first * 3 + second
    stakeholder_f = jnp.array([10.0, 2.0, 40.0, 5.0, 12.0, 20.0])[global_id]
    stakeholder_m = jnp.array([0.0, 50.0, 0.0, 45.0, 30.0, 20.0])[global_id]
    return jnp.stack((stakeholder_f, stakeholder_m)), row != 0


def _collective_weights(*, row, base_weight):
    weight_f = base_weight + 0.5 * (row == 2)
    return {"f": weight_f, "m": 1.0 - weight_f}


_PARETO_WEIGHTS = ParetoWeights(
    compute=_collective_weights,
    declared=_collective_weights,
    arg_names=("row", "base_weight"),
    param_names=("base_weight",),
    normalization="none",
)


@pytest.mark.parametrize("block_width", [1, 4, 5, 6])
def test_collective_full_V_streaming_matches_dense_with_one_household_winner(
    block_width: int,
) -> None:
    action_names = ("first", "second")
    state_names = ("row",)
    batch_sizes = {"row": 0}
    arguments = {
        "first": jnp.array([0, 1], dtype=jnp.int32),
        "second": jnp.array([0, 1, 2], dtype=jnp.int32),
        "row": jnp.array([0, 1, 2], dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType({}),
        "base_weight": jnp.float32(0.25),
    }
    dense = get_max_Q_over_a(
        Q_and_F=_collective_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
        stakeholders=("f", "m"),
        pareto_weights=_PARETO_WEIGHTS,
    )
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_collective_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
        stakeholders=("f", "m"),
        pareto_weights=_PARETO_WEIGHTS,
    )
    streamed = functools.partial(
        raw_streamed,
        _lcm_action_block_width=block_width,
    )

    expected_values, expected_dissolution = jax.jit(dense)(**arguments)
    actual_values, actual_dissolution = jax.jit(streamed)(**arguments)

    assert_array_equal(actual_values, expected_values)
    assert_array_equal(actual_dissolution, expected_dissolution)
    assert_array_equal(
        actual_values,
        jnp.array([[-jnp.inf, -jnp.inf], [2.0, 50.0], [40.0, 0.0]]),
    )
    assert_array_equal(actual_dissolution, jnp.array([True, False, False]))


def _ev1_Q_and_F(
    *,
    first,
    second,
    continuous,
    taste_shocks__scale,
    next_regime_to_V_arr,
):
    del next_regime_to_V_arr, taste_shocks__scale
    branch = first * 2 + second
    values = jnp.array(
        [
            [1.0, 4.0, 3.0],
            [5.0, 2.0, 6.0],
            [100.0, 100.0, 100.0],
            [0.0, 8.0, 7.0],
        ]
    )
    feasible = (branch != 2) & ~((branch == 1) & (continuous == 2))
    return values[branch, continuous], feasible


@pytest.mark.parametrize("block_width", [1, 4, 5, 7, 12])
def test_ev1_full_V_streaming_hard_maxes_branches_before_logsum(
    block_width: int,
) -> None:
    """Arbitrary block boundaries preserve discrete-prefix branch semantics."""
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_ev1_Q_and_F,
        batch_sizes={},
        action_names=("first", "second", "continuous"),
        state_names=(),
        n_discrete_action_axes=2,
        has_taste_shocks=True,
    )
    streamed = jax.jit(
        functools.partial(
            raw_streamed,
            _lcm_action_block_width=block_width,
        )
    )
    arguments = {
        "first": jnp.array([0, 1], dtype=jnp.int32),
        "second": jnp.array([0, 1], dtype=jnp.int32),
        "continuous": jnp.array([0, 1, 2], dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType({}),
    }
    branch_values = np.array([4.0, 5.0, -np.inf, 8.0])

    for scale in (0.2, 1.1):
        anchor = np.max(branch_values)
        expected = anchor + scale * np.log(
            np.sum(np.exp((branch_values - anchor) / scale))
        )
        actual = streamed(
            **arguments,
            taste_shocks__scale=jnp.asarray(scale),
        )
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def _ev1_nonfinite_Q_and_F(
    *,
    discrete,
    continuous,
    special,
    next_regime_to_V_arr,
):
    del next_regime_to_V_arr
    global_id = discrete * 2 + continuous
    baseline = jnp.array([0.0, 1.0, 2.0, 3.0])
    value = jnp.where(global_id == 0, special, baseline[global_id])
    return value, jnp.ones((), dtype=bool)


@pytest.mark.parametrize(
    "special",
    [pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="posinf")],
)
def test_ev1_streaming_preserves_composite_nonfinite_semantics(
    special: float,
) -> None:
    """Branch finalization preserves dense NaN and positive-infinity poisoning."""
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_ev1_nonfinite_Q_and_F,
        batch_sizes={},
        action_names=("discrete", "continuous"),
        state_names=(),
        n_discrete_action_axes=1,
        has_taste_shocks=True,
    )
    streamed = jax.jit(functools.partial(raw_streamed, _lcm_action_block_width=3))
    scale = jnp.asarray(0.4)
    actual = streamed(
        discrete=jnp.array([0, 1], dtype=jnp.int32),
        continuous=jnp.array([0, 1], dtype=jnp.int32),
        special=jnp.asarray(special),
        next_regime_to_V_arr=MappingProxyType({}),
        taste_shocks__scale=scale,
    )
    expected, _ = logsum_and_softmax(
        values=jnp.asarray([special, 3.0]),
        scale=scale,
        axes=(0,),
    )

    assert bool(jnp.isnan(expected))
    assert bool(jnp.isnan(actual))


def test_ev1_reduction_semantic_key_pins_the_composite_contract() -> None:
    reduction = GridSearchEV1ActionReduction(n_discrete_action_axes=2)

    assert reduction.semantic_key == (
        "grid-search-ev1-action-reduction",
        1,
        2,
        ("hard-max", 1),
        ("logsumexp", 1),
    )


def test_full_V_streaming_fails_closed_on_unmigrated_routes() -> None:
    with pytest.raises(NotImplementedError, match="fold"):
        get_streaming_max_Q_over_a(
            Q_and_F=_Q_and_F,
            batch_sizes={"row": 0, "shift": 0},
            action_names=("first", "second"),
            state_names=("row", "shift"),
            fold_state_names=("row",),
        )
    with pytest.raises(NotImplementedError, match="co-map"):
        get_streaming_max_Q_over_a(
            Q_and_F=_Q_and_F,
            batch_sizes={"row": 0, "shift": 0},
            action_names=("first", "second"),
            state_names=("row", "shift"),
            co_map_state_names=("row",),
            co_map_v_arr_in_axes=(MappingProxyType({"target": 0}),),
        )
