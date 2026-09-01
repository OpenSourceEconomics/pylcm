"""Full-V action streaming preserves singleton and collective GridSearch results."""

import functools
import inspect
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

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
from _lcm.solution.grid_search import _supports_action_streaming


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


def _co_mapped_Q_and_F(
    *,
    action,
    kind,
    region,
    row,
    next_regime_to_V_arr,
):
    carrying_both = next_regime_to_V_arr["carrying_both"][row]
    carrying_kind = next_regime_to_V_arr["carrying_kind"][row]
    carrying_region = next_regime_to_V_arr["carrying_region"][row]
    passed_through = next_regime_to_V_arr["passed_through"][row]
    base = (
        carrying_both
        + carrying_kind
        + carrying_region
        + passed_through
        + 100_000 * kind
        + 1_000_000 * region
    )
    tied_winner = (action == 1) | (action == 4)
    value = jnp.where(tied_winner, base + 5.0, base - action)
    feasible = (row != 0) & ((row != 2) | (action != 4))
    return value, feasible


def _co_map_arguments():
    return {
        "action": jnp.arange(5, dtype=jnp.int32),
        "kind": jnp.arange(2, dtype=jnp.int32),
        "region": jnp.arange(2, dtype=jnp.int32),
        "row": jnp.arange(3, dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType(
            {
                "carrying_both": jnp.asarray(
                    [
                        [[10, 20, 30], [40, 50, 60]],
                        [[100, 200, 300], [400, 500, 600]],
                    ]
                ),
                "carrying_kind": jnp.asarray([[1, 2, 3], [10, 20, 30]]),
                "carrying_region": jnp.asarray(
                    [[100, 200, 300], [1_000, 2_000, 3_000]]
                ),
                "passed_through": jnp.asarray([10_000, 20_000, 30_000]),
            }
        ),
    }


def _co_map_kernels():
    kwargs = {
        "Q_and_F": _co_mapped_Q_and_F,
        "batch_sizes": {"kind": 0, "region": 0, "row": 0},
        "action_names": ("action",),
        "state_names": ("kind", "region", "row"),
        "co_map_state_names": ("kind", "region"),
        "co_map_v_arr_in_axes": (
            MappingProxyType(
                {
                    "carrying_both": 0,
                    "carrying_kind": 0,
                    "carrying_region": None,
                    "passed_through": None,
                }
            ),
            MappingProxyType(
                {
                    "carrying_both": 0,
                    "carrying_kind": None,
                    "carrying_region": 0,
                    "passed_through": None,
                }
            ),
        ),
    }
    return get_max_Q_over_a(**cast("Any", kwargs)), get_streaming_max_Q_over_a(
        **cast("Any", kwargs)
    )


@pytest.mark.parametrize("block_width", [1, 3])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
def test_co_mapped_full_V_streaming_matches_dense_and_preserves_state_axes(
    *, block_width: int, execution: str
) -> None:
    dense, raw_streamed = _co_map_kernels()
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=block_width)
    arguments = _co_map_arguments()

    def evaluate(func):
        if execution == "eager":
            return func(**arguments)
        jitted = jax.jit(func)
        if execution == "jit":
            return jitted(**arguments)
        return jitted.lower(**arguments).compile()(**arguments)

    expected = evaluate(dense)
    actual = evaluate(streamed)

    assert_array_equal(actual, expected)
    assert_array_equal(
        actual,
        jnp.asarray(
            [
                [
                    [-jnp.inf, 20_227.0, 30_338.0],
                    [-jnp.inf, 1_022_057.0, 1_033_068.0],
                ],
                [
                    [-jnp.inf, 120_425.0, 130_635.0],
                    [-jnp.inf, 1_122_525.0, 1_133_635.0],
                ],
            ]
        ),
    )


def _co_mapped_ev1_Q_and_F(
    *,
    discrete,
    continuous,
    kind,
    region,
    row,
    next_regime_to_V_arr,
    taste_shocks__scale,
):
    del taste_shocks__scale
    base, _ = _co_mapped_Q_and_F(
        action=jnp.int32(0),
        kind=kind,
        region=region,
        row=row,
        next_regime_to_V_arr=next_regime_to_V_arr,
    )
    action_value = jnp.asarray([[1.0, 4.0, 3.0], [5.0, 2.0, 6.0]])[discrete, continuous]
    feasible = ~((region == 1) & (discrete == 1) & (continuous == 2))
    return 1e-5 * base + action_value, feasible


@pytest.mark.parametrize("block_width", [1, 4])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
def test_ev1_co_mapped_streaming_hard_maxes_branches_before_logsum(
    *, block_width: int, execution: str
) -> None:
    co_map_v_arr_in_axes = (
        MappingProxyType(
            {
                "carrying_both": 0,
                "carrying_kind": 0,
                "carrying_region": None,
                "passed_through": None,
            }
        ),
        MappingProxyType(
            {
                "carrying_both": 0,
                "carrying_kind": None,
                "carrying_region": 0,
                "passed_through": None,
            }
        ),
    )
    kernel_kwargs = {
        "Q_and_F": _co_mapped_ev1_Q_and_F,
        "batch_sizes": {"kind": 0, "region": 0, "row": 0},
        "action_names": ("discrete", "continuous"),
        "state_names": ("kind", "region", "row"),
        "n_discrete_action_axes": 1,
        "has_taste_shocks": True,
        "co_map_state_names": ("kind", "region"),
        "co_map_v_arr_in_axes": co_map_v_arr_in_axes,
    }
    dense = get_max_Q_over_a(**cast("Any", kernel_kwargs))
    streamed = functools.partial(
        get_streaming_max_Q_over_a(**cast("Any", kernel_kwargs)),
        _lcm_action_block_width=block_width,
    )
    arguments = _co_map_arguments()
    arguments.pop("action")
    scale = 0.75
    arguments.update(
        discrete=jnp.arange(2, dtype=jnp.int32),
        continuous=jnp.arange(3, dtype=jnp.int32),
        taste_shocks__scale=jnp.asarray(scale),
    )

    def evaluate(func):
        if execution == "eager":
            return func(**arguments)
        jitted = jax.jit(func)
        if execution == "jit":
            return jitted(**arguments)
        return jitted.lower(**arguments).compile()(**arguments)

    expected = evaluate(dense)
    actual = evaluate(streamed)

    base = 1e-5 * np.asarray(
        [
            [[10_111, 20_222, 30_333], [1_011_041, 1_022_052, 1_033_063]],
            [[110_210, 120_420, 130_630], [1_111_410, 1_122_520, 1_133_630]],
        ]
    )
    # Continuous hard maxima are (4, 6) in region 0 and (4, 5) in region 1.
    branch_maxima = np.asarray([[4.0, 6.0], [4.0, 5.0]])
    increments = scale * np.logaddexp(
        branch_maxima[:, 0] / scale, branch_maxima[:, 1] / scale
    )
    oracle = base + increments[None, :, None]

    eps = np.finfo(np.asarray(actual).dtype).eps
    assert_allclose(actual, expected, rtol=8 * eps, atol=8 * eps)
    assert_allclose(actual, oracle, rtol=8 * eps, atol=8 * eps)


def test_streaming_co_map_state_and_continuation_axis_sizes_must_align() -> None:
    _, streamed = _co_map_kernels()
    arguments = dict(_co_map_arguments())
    continuation = dict(arguments["next_regime_to_V_arr"])
    continuation["carrying_both"] = jnp.zeros((3, 2, 3))
    arguments["next_regime_to_V_arr"] = MappingProxyType(continuation)

    with pytest.raises(ValueError, match="inconsistent sizes"):
        streamed(**arguments, _lcm_action_block_width=3)


@pytest.mark.parametrize(
    ("state_names", "co_map_state_names", "co_map_v_arr_in_axes", "message"),
    [
        (
            ("row", "kind", "region"),
            ("kind", "region"),
            (
                MappingProxyType({"carrying": 0, "passed": None}),
                MappingProxyType({"carrying": 0, "passed": None}),
            ),
            "leading axes",
        ),
        (
            ("kind", "region", "row"),
            ("kind", "region"),
            (MappingProxyType({"carrying": 0}),),
            "same length",
        ),
        (
            ("kind", "region", "row"),
            ("kind", "region"),
            (MappingProxyType({}), MappingProxyType({})),
            "at least one target",
        ),
        (
            ("kind", "region", "row"),
            ("kind", "region"),
            (
                MappingProxyType({"carrying": 0, "passed": None}),
                MappingProxyType({"carrying": 0}),
            ),
            "same target keys",
        ),
        (
            ("kind", "region", "row"),
            ("kind", "region"),
            (
                MappingProxyType({"carrying": 1}),
                MappingProxyType({"carrying": 0}),
            ),
            "only 0 or None",
        ),
    ],
    ids=[
        "not-leading",
        "missing-in-axes",
        "empty-target-map",
        "inconsistent-target-keys",
        "non-leading-continuation-axis",
    ],
)
def test_streaming_co_map_layout_fails_closed(
    *,
    state_names,
    co_map_state_names,
    co_map_v_arr_in_axes,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        get_streaming_max_Q_over_a(
            Q_and_F=_co_mapped_Q_and_F,
            batch_sizes={"kind": 0, "region": 0, "row": 0},
            action_names=("action",),
            state_names=state_names,
            co_map_state_names=co_map_state_names,
            co_map_v_arr_in_axes=co_map_v_arr_in_axes,
        )


def _co_map_route_context(
    *,
    action_names=("action",),
    actions_grid_shapes=(5,),
    discrete_actions=(),
    Q_and_F=_co_mapped_Q_and_F,
    has_taste_shocks=False,
    stakeholders=None,
    fold_state_names=(),
    co_map_state_names=("kind", "region"),
    same_period_ref_regimes=(),
    edge_reference_regimes=(),
    edge_target_regimes=(),
):
    return SimpleNamespace(
        state_action_space=SimpleNamespace(
            action_names=action_names,
            actions_grid_shapes=actions_grid_shapes,
            discrete_actions=discrete_actions,
        ),
        Q_and_F_functions=MappingProxyType({0: Q_and_F}),
        has_taste_shocks=has_taste_shocks,
        enable_jit=True,
        stakeholders=stakeholders,
        fold_state_names=fold_state_names,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=tuple(
            MappingProxyType({"target": 0}) for _ in co_map_state_names
        ),
        same_period_ref_regimes=same_period_ref_regimes,
        edge_reference_regimes=edge_reference_regimes,
        edge_target_regimes=edge_target_regimes,
    )


def _width_collision_Q_and_F(*, action, _lcm_action_block_width, next_regime_to_V_arr):
    del _lcm_action_block_width, next_regime_to_V_arr
    return action, jnp.ones((), dtype=bool)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, True, id="ordinary-co-map"),
        pytest.param(
            {"edge_target_regimes": ("target",)},
            True,
            id="gated-target-substitution",
        ),
        pytest.param(
            {
                "fold_state_names": ("shock",),
                "co_map_state_names": (),
            },
            True,
            id="singleton-fold",
        ),
        pytest.param(
            {"fold_state_names": ("shock",)},
            True,
            id="singleton-fold-plus-ordinary-co-map",
        ),
        pytest.param(
            {"same_period_ref_regimes": ("reference",)},
            False,
            id="co-map-plus-same-period-reference",
        ),
        pytest.param(
            {"edge_reference_regimes": ("reference",)},
            False,
            id="co-map-plus-edge-reference",
        ),
        pytest.param(
            {"actions_grid_shapes": (1,)},
            False,
            id="trivial-action-product",
        ),
        pytest.param(
            {"Q_and_F": _width_collision_Q_and_F},
            False,
            id="reserved-width-collision",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "stakeholders": ("f", "m"),
                "discrete_actions": ("action",),
            },
            False,
            id="collective-ev1",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "fold_state_names": ("shock",),
                "co_map_state_names": (),
            },
            False,
            id="singleton-ev1-fold",
        ),
    ],
)
def test_action_streaming_route_eligibility_matrix(
    *, overrides: dict[str, Any], expected: bool
) -> None:
    context = _co_map_route_context(**overrides)

    assert inspect.unwrap(_supports_action_streaming)(context=context) is expected


def _fold_Q_and_F(
    *,
    action,
    risk_type,
    shock,
    poison_zero_weight_node,
    next_regime_to_V_arr,
):
    del next_regime_to_V_arr
    values = jnp.asarray(
        [
            [
                [8.0, 1.0, 0.0, -2.0, 3.0],
                [0.0, 10.0, 2.0, 1.0, 3.0],
                [1.0, 2.0, 12.0, 0.0, 4.0],
            ],
            [
                [5.0, 0.0, 1.0, 2.0, 3.0],
                [0.0, 6.0, 2.0, 1.0, 3.0],
                [1.0, 2.0, 7.0, 0.0, 4.0],
            ],
        ]
    )
    value = values[risk_type, shock, action]
    value = jnp.where(
        poison_zero_weight_node & (shock == 1),
        -jnp.inf,
        value,
    )
    return value, jnp.ones((), dtype=bool)


def _evaluate_kernel(*, func: Any, arguments: dict[str, Any], execution: str):
    if execution == "eager":
        return func(**arguments)
    jitted = jax.jit(func)
    if execution == "jit":
        return jitted(**arguments)
    return jitted.lower(**arguments).compile()(**arguments)


@pytest.mark.parametrize("block_width", [1, 3])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
def test_folded_co_map_streaming_reduces_inside_the_device_local_state_maps(
    *, block_width: int, execution: str
) -> None:
    fold_weights = jnp.asarray([0.0, 0.4, 0.6], dtype=jnp.float32)
    kernel_kwargs = {
        "Q_and_F": _co_mapped_Q_and_F,
        "batch_sizes": {"kind": 0, "region": 0, "row": 0},
        "action_names": ("action",),
        "state_names": ("kind", "region", "row"),
        "co_map_state_names": ("kind", "region"),
        "co_map_v_arr_in_axes": (
            MappingProxyType(
                {
                    "carrying_both": 0,
                    "carrying_kind": 0,
                    "carrying_region": None,
                    "passed_through": None,
                }
            ),
            MappingProxyType(
                {
                    "carrying_both": 0,
                    "carrying_kind": None,
                    "carrying_region": 0,
                    "passed_through": None,
                }
            ),
        ),
        "fold_state_names": ("row",),
        "fold_weights": MappingProxyType({"row": fold_weights}),
    }
    dense = get_max_Q_over_a(**cast("Any", kernel_kwargs))
    streamed = functools.partial(
        get_streaming_max_Q_over_a(**cast("Any", kernel_kwargs)),
        _lcm_action_block_width=block_width,
    )
    arguments = _co_map_arguments()

    expected = _evaluate_kernel(func=dense, arguments=arguments, execution=execution)
    actual = _evaluate_kernel(func=streamed, arguments=arguments, execution=execution)

    node_values = np.asarray(
        [
            [
                [-np.inf, 20_227.0, 30_338.0],
                [-np.inf, 1_022_057.0, 1_033_068.0],
            ],
            [
                [-np.inf, 120_425.0, 130_635.0],
                [-np.inf, 1_122_525.0, 1_133_635.0],
            ],
        ],
        dtype=np.asarray(actual).dtype,
    )
    weights = np.asarray(fold_weights)
    weighted = np.zeros_like(node_values)
    np.multiply(node_values, weights, out=weighted, where=weights != 0)
    oracle = weighted.sum(axis=-1)

    eps = np.finfo(np.asarray(actual).dtype).eps
    assert actual.shape == (2, 2)
    assert_allclose(actual, expected, rtol=8 * eps, atol=8 * eps)
    assert_allclose(actual, oracle, rtol=8 * eps, atol=8 * eps)
    assert not bool(jnp.any(jnp.isnan(actual)))


@pytest.mark.parametrize("block_width", [1, 3])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
@pytest.mark.parametrize(
    ("weights", "conditioned", "poison_zero_weight_node", "oracle"),
    [
        pytest.param(
            (0.25, 0.5, 0.25),
            False,
            False,
            (10.0, 6.0),
            id="positive-unconditioned-weights",
        ),
        pytest.param(
            (0.5, 0.0, 0.5),
            False,
            True,
            (10.0, 6.0),
            id="zero-weight-minus-infinity-node",
        ),
        pytest.param(
            ((0.25, 0.5, 0.25), (0.5, 0.25, 0.25)),
            True,
            False,
            (10.0, 5.75),
            id="conditioned-weight-rows",
        ),
    ],
)
def test_folded_singleton_streams_action_max_before_exact_quadrature(
    *,
    block_width: int,
    execution: str,
    weights,
    conditioned: bool,
    poison_zero_weight_node: bool,
    oracle: tuple[float, float],
) -> None:
    kernel_kwargs = {
        "Q_and_F": _fold_Q_and_F,
        "batch_sizes": {"risk_type": 0, "shock": 0},
        "action_names": ("action",),
        "state_names": ("risk_type", "shock"),
        "fold_state_names": ("shock",),
        "fold_weights": MappingProxyType(
            {"shock": jnp.asarray(weights, dtype=jnp.float32)}
        ),
        "fold_conditioning": MappingProxyType(
            {"shock": "risk_type"} if conditioned else {}
        ),
    }
    dense = get_max_Q_over_a(**cast("Any", kernel_kwargs))
    streamed = functools.partial(
        get_streaming_max_Q_over_a(**cast("Any", kernel_kwargs)),
        _lcm_action_block_width=block_width,
    )
    arguments = {
        "action": jnp.arange(5, dtype=jnp.int32),
        "risk_type": jnp.arange(2, dtype=jnp.int32),
        "shock": jnp.arange(3, dtype=jnp.int32),
        "poison_zero_weight_node": jnp.asarray(poison_zero_weight_node),
        "next_regime_to_V_arr": MappingProxyType({}),
    }

    expected = _evaluate_kernel(func=dense, arguments=arguments, execution=execution)
    actual = _evaluate_kernel(func=streamed, arguments=arguments, execution=execution)

    eps = np.finfo(np.asarray(actual).dtype).eps
    assert_allclose(actual, expected, rtol=8 * eps, atol=8 * eps)
    assert_allclose(actual, np.asarray(oracle), rtol=8 * eps, atol=8 * eps)
    assert not bool(jnp.any(jnp.isnan(actual)))


def _ev1_fold_Q_and_F(*, discrete, shock, next_regime_to_V_arr):
    del next_regime_to_V_arr
    return discrete + shock, jnp.ones((), dtype=bool)


def test_full_V_streaming_fails_closed_on_ev1_fold_route() -> None:
    with pytest.raises(NotImplementedError, match=r"EV1.*fold|fold.*EV1"):
        get_streaming_max_Q_over_a(
            Q_and_F=_ev1_fold_Q_and_F,
            batch_sizes={"shock": 0},
            action_names=("discrete",),
            state_names=("shock",),
            n_discrete_action_axes=1,
            has_taste_shocks=True,
            fold_state_names=("shock",),
            fold_weights=MappingProxyType({"shock": jnp.asarray([0.25, 0.5, 0.25])}),
        )
