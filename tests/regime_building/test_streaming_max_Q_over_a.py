"""Full-V action streaming preserves singleton and collective GridSearch results."""

import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.core import ShapedArray
from jax.extend.core import Jaxpr
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.engine import StateActionSpace
from _lcm.regime_building.collective import ParetoWeights, build_pareto_weights
from _lcm.regime_building.max_Q_over_a import (
    BoundedMaxOutput,
    RawActionFunction,
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.solution.action_streaming import build_streaming_collective_max_Q_over_a
from _lcm.solution.collective_action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxResult,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import (
    _ActionStreamingDisposition,
    _classify_action_streaming,
    _GridSearchPeriodKernel,
)
from _lcm.typing import MaxQOverAFunction, QAndFFunction, QAndFKwargs
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.execution import ExecutionConfig
from lcm.solver_api import SolutionResult
from lcm.solvers import ACTION_PRODUCT_AXIS
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
    StateName,
)
from tests.collective_fixtures import Work
from tests.conftest import assert_agrees_to_ulp
from tests.single_member import SELF, SELF_PARETO_WEIGHTS, with_member_axis


class _KernelKwargs(TypedDict):
    Q_and_F: Callable[..., tuple[FloatND, BoolND]]
    batch_sizes: dict[StateName, int]
    action_names: tuple[ActionName, ...]
    state_names: tuple[StateName, ...]
    co_map_state_names: NotRequired[tuple[StateName, ...]]
    co_map_v_arr_in_axes: NotRequired[
        tuple[MappingProxyType[RegimeName, int | None], ...]
    ]
    members: tuple[str, ...]
    pareto_weights: ParetoWeights
    cell_width_keyword: NotRequired[str | None]
    untiled_state_names: NotRequired[tuple[StateName, ...]]
    broadcast_state_names: NotRequired[tuple[StateName, ...]]


class _RouteOverrides(TypedDict, total=False):
    action_names: tuple[ActionName, ...]
    actions_grid_shapes: tuple[int, ...]
    discrete_actions: tuple[ActionName, ...]
    Q_and_F: QAndFFunction
    has_taste_shocks: bool
    co_map_state_names: tuple[StateName, ...]
    edge_reference_regimes: tuple[RegimeName, ...]
    edge_target_regimes: tuple[RegimeName, ...]
    enable_jit: bool


def _single(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
) -> _KernelKwargs:
    """Kernel arguments of a single: the household of the one member `"self"`."""
    return {
        "Q_and_F": with_member_axis(Q_and_F),
        "batch_sizes": batch_sizes,
        "action_names": action_names,
        "state_names": state_names,
        "co_map_state_names": co_map_state_names,
        "co_map_v_arr_in_axes": co_map_v_arr_in_axes,
        "members": SELF,
        "pareto_weights": SELF_PARETO_WEIGHTS,
    }


def _self_value(output: BoundedMaxOutput) -> FloatND:
    """The value of the member `"self"` in a reducer's `(value, dissolution)` output."""
    value, _dissolution = cast("tuple[FloatND, BoolND]", output)
    return value[..., 0]


def _Q_and_F(
    *,
    first: IntND,
    second: IntND,
    row: IntND,
    shift: FloatND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    offset: FloatND,
) -> tuple[FloatND, BoolND]:
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
    kernel_kwargs = _single(
        Q_and_F=_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
    )
    dense = get_max_Q_over_a(**kernel_kwargs)
    raw_streamed = get_streaming_max_Q_over_a(**kernel_kwargs)

    width = inspect.signature(raw_streamed).parameters["_lcm_action_block_width"]
    assert width.default is inspect.Parameter.empty

    streamed = functools.partial(
        raw_streamed,
        _lcm_action_block_width=block_width,
    )
    expected = _self_value(jax.jit(dense)(**arguments))
    actual = _self_value(jax.jit(streamed)(**arguments))

    assert_array_equal(actual, expected)
    assert_array_equal(actual[0], jnp.full((2,), -jnp.inf))
    assert_array_equal(actual[1], jnp.array([8.75, 10.0]))


def _feasible_nan_Q_and_F(
    *,
    first: IntND,
    second: IntND,
    row: IntND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    del next_regime_to_V_arr
    value = jnp.where((row == 0) & (first == 1) & (second == 7), jnp.nan, 0.0)
    return value, jnp.ones((), dtype=bool)


@pytest.mark.parametrize("block_width", [7, 60])
def test_feasible_nan_survives_a_large_whole_product_block(*, block_width: int) -> None:
    """A feasible NaN publishes NaN at every width, however large the reduction.

    XLA:CPU's vectorized maximum drops NaN once a reduction spans more than a few
    thousand elements, so the whole-product block states the rule explicitly.
    """
    raw_streamed = get_streaming_max_Q_over_a(
        **_single(
            Q_and_F=_feasible_nan_Q_and_F,
            batch_sizes={"row": 0},
            action_names=("first", "second"),
            state_names=("row",),
        )
    )
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=block_width)

    actual = _self_value(
        jax.jit(streamed)(
            first=jnp.arange(3),
            second=jnp.arange(20),
            row=jnp.arange(100),
            next_regime_to_V_arr=MappingProxyType({}),
        )
    )

    assert bool(jnp.isnan(actual[0]))
    assert_array_equal(actual[1:], jnp.zeros(99))


def _signed_zero_Q_and_F(
    *, action: IntND, next_regime_to_V_arr: Mapping[RegimeName, FloatND]
) -> tuple[FloatND, BoolND]:
    del next_regime_to_V_arr
    value = jnp.where(action == 0, -0.0, 0.0)
    return value, jnp.ones((), dtype=bool)


def test_full_V_streaming_matches_dense_signed_zero_across_width_one_blocks() -> None:
    """A `{-0, +0}` tie is one value; the solve of a single publishes it as `+0`.

    The dense route reads the value of identity zero, a negative zero. The
    streamed solve of one member keeps no identity, and of `-0` and `+0` it
    keeps `+0`.
    """
    arguments = {
        "action": jnp.array([0, 1], dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType({}),
    }
    kernel_kwargs = _single(
        Q_and_F=_signed_zero_Q_and_F,
        batch_sizes={},
        action_names=("action",),
        state_names=(),
    )
    dense = get_max_Q_over_a(**kernel_kwargs)
    raw_streamed = get_streaming_max_Q_over_a(**kernel_kwargs)
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=1)

    expected = _self_value(jax.jit(dense)(**arguments))
    actual = _self_value(jax.jit(streamed)(**arguments))

    assert_array_equal(actual, expected)
    assert bool(jnp.signbit(expected))
    assert not bool(jnp.signbit(actual))


def _collective_Q_and_F(
    *,
    first: IntND,
    second: IntND,
    row: IntND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    del next_regime_to_V_arr
    global_id = first * 3 + second
    member_f = jnp.array([10.0, 2.0, 40.0, 5.0, 12.0, 20.0])[global_id]
    member_m = jnp.array([0.0, 50.0, 0.0, 45.0, 30.0, 20.0])[global_id]
    return jnp.stack((member_f, member_m)), row != 0


def _collective_weights(*, row: IntND, base_weight: FloatND) -> dict[str, FloatND]:
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
        members=("f", "m"),
        pareto_weights=_PARETO_WEIGHTS,
    )
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_collective_Q_and_F,
        batch_sizes=batch_sizes,
        action_names=action_names,
        state_names=state_names,
        members=("f", "m"),
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


def _tied_shares_Q_and_F(
    *,
    a: FloatND,
    b: FloatND,
    row: IntND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    del row, next_regime_to_V_arr
    return jnp.stack((a, b)), a + b == 1.0


def _equal_weights() -> dict[str, FloatND]:
    return {"f": jnp.asarray(0.5), "m": jnp.asarray(0.5)}


_EQUAL_PARETO_WEIGHTS = ParetoWeights(
    compute=_equal_weights,
    declared=_equal_weights,
    arg_names=(),
    param_names=(),
    normalization="none",
)


@pytest.mark.parametrize("action_order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("block_width", [1, 2, 7, 25])
def test_tied_household_objective_goes_to_the_first_member_at_every_width(
    *, action_order: tuple[ActionName, ActionName], block_width: int
) -> None:
    """Every feasible split ties; `f`, declared first, takes the whole unit.

    Two members split one unit, `u_f = a` and `u_m = b`, under equal weights, so
    the objective is `1/2` at all five feasible actions. The streamed value is
    `(1, 0)` at every block width, including one that covers all 25 actions, in
    either declaration order of the actions.
    """
    shares = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    raw_streamed = get_streaming_max_Q_over_a(
        Q_and_F=_tied_shares_Q_and_F,
        batch_sizes={"row": 0},
        action_names=action_order,
        state_names=("row",),
        members=("f", "m"),
        pareto_weights=_EQUAL_PARETO_WEIGHTS,
    )
    kernel = functools.partial(raw_streamed, _lcm_action_block_width=block_width)

    values, dissolution = jax.jit(kernel)(
        a=shares,
        b=shares,
        row=jnp.array([0], dtype=jnp.int32),
        next_regime_to_V_arr=MappingProxyType({}),
    )

    assert_array_equal(values, jnp.array([[1.0, 0.0]]))
    assert_array_equal(dissolution, jnp.array([False]))


_SHARES = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

# Each route's kernel and arguments; the single has 6 actions, the household 25.
_FULL_WIDTH_ROUTES: dict[str, tuple[_KernelKwargs, QAndFKwargs]] = {
    "single": (
        _single(
            Q_and_F=_Q_and_F,
            batch_sizes={"row": 0, "shift": 0},
            action_names=("first", "second"),
            state_names=("row", "shift"),
        ),
        {
            "first": jnp.array([0, 1]),
            "second": jnp.array([0, 1, 2]),
            "row": jnp.array([0, 1, 2]),
            "shift": jnp.array([-0.5, 0.75]),
            "next_regime_to_V_arr": MappingProxyType({}),
            "offset": jnp.float32(0.25),
        },
    ),
    "household": (
        {
            "Q_and_F": _tied_shares_Q_and_F,
            "batch_sizes": {"row": 0},
            "action_names": ("a", "b"),
            "state_names": ("row",),
            "members": ("f", "m"),
            "pareto_weights": _EQUAL_PARETO_WEIGHTS,
        },
        {
            "a": _SHARES,
            "b": _SHARES,
            "row": jnp.array([0, 1], dtype=jnp.int32),
            "next_regime_to_V_arr": MappingProxyType({}),
        },
    ),
}


@pytest.mark.parametrize(
    ("route", "block_width", "decodes"),
    [
        ("single", 4, True),
        ("single", 6, False),
        ("single", 8, False),
        ("household", 7, True),
        ("household", 25, False),
        ("household", 30, False),
    ],
)
def test_full_width_action_block_maps_the_named_action_grids(
    *, route: str, block_width: int, decodes: bool
) -> None:
    """A block covering the action product reads each grid instead of a flat index,
    and carries no accumulator across blocks."""
    kernel_kwargs, arguments = _FULL_WIDTH_ROUTES[route]
    raw_streamed = get_streaming_max_Q_over_a(**kernel_kwargs)
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=block_width)

    jaxpr = str(jax.make_jaxpr(streamed)(**arguments))

    assert ("gather" in jaxpr) is decodes
    assert ("scan" in jaxpr) is decodes


def _co_mapped_Q_and_F(
    *,
    action: IntND,
    kind: IntND,
    region: IntND,
    row: IntND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
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


class _CoMapArguments(TypedDict):
    action: NotRequired[IntND]
    kind: IntND
    region: IntND
    row: IntND
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND]


def _co_map_arguments() -> _CoMapArguments:
    return {
        "action": jnp.arange(5, dtype=jnp.int32),
        "kind": jnp.arange(2, dtype=jnp.int32),
        "region": jnp.arange(2, dtype=jnp.int32),
        "row": jnp.arange(3, dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType(
            {
                "carrying_both": jnp.asarray(
                    [
                        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                        [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                    ]
                ),
                "carrying_kind": jnp.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
                "carrying_region": jnp.asarray(
                    [[100.0, 200.0, 300.0], [1_000.0, 2_000.0, 3_000.0]]
                ),
                "passed_through": jnp.asarray([10_000.0, 20_000.0, 30_000.0]),
            }
        ),
    }


def _co_map_kernels() -> tuple[
    MaxQOverAFunction[BoundedMaxOutput], MaxQOverAFunction[BoundedMaxOutput]
]:
    kwargs = _single(
        Q_and_F=_co_mapped_Q_and_F,
        batch_sizes={"kind": 0, "region": 0, "row": 0},
        action_names=("action",),
        state_names=("kind", "region", "row"),
        co_map_state_names=("kind", "region"),
        co_map_v_arr_in_axes=(
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
    )
    return get_max_Q_over_a(**kwargs), get_streaming_max_Q_over_a(**kwargs)


@pytest.mark.parametrize("block_width", [1, 3])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
def test_co_mapped_full_V_streaming_matches_dense_and_preserves_state_axes(
    *, block_width: int, execution: str
) -> None:
    dense, raw_streamed = _co_map_kernels()
    streamed = functools.partial(raw_streamed, _lcm_action_block_width=block_width)
    arguments = _co_map_arguments()

    def evaluate(func: Callable[..., BoundedMaxOutput]) -> BoundedMaxOutput:
        if execution == "eager":
            return func(**arguments)
        jitted = jax.jit(func)
        if execution == "jit":
            return jitted(**arguments)
        return jitted.lower(**arguments).compile()(**arguments)

    expected = _self_value(evaluate(dense))
    actual = _self_value(evaluate(streamed))

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


def test_streaming_co_map_state_and_continuation_axis_sizes_must_align() -> None:
    _, streamed = _co_map_kernels()
    arguments = dict(_co_map_arguments())
    continuation = dict(
        cast("MappingProxyType[RegimeName, FloatND]", arguments["next_regime_to_V_arr"])
    )
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
    state_names: tuple[StateName, ...],
    co_map_state_names: tuple[StateName, ...],
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        get_streaming_max_Q_over_a(
            **_single(
                Q_and_F=_co_mapped_Q_and_F,
                batch_sizes={"kind": 0, "region": 0, "row": 0},
                action_names=("action",),
                state_names=state_names,
                co_map_state_names=co_map_state_names,
                co_map_v_arr_in_axes=co_map_v_arr_in_axes,
            )
        )


def _action_streaming_route_context(
    *,
    action_names: tuple[ActionName, ...] = ("action",),
    actions_grid_shapes: tuple[int, ...] = (5,),
    discrete_actions: tuple[ActionName, ...] = (),
    Q_and_F: RawActionFunction = _co_mapped_Q_and_F,
    has_taste_shocks: bool = False,
    co_map_state_names: tuple[StateName, ...] = (),
    edge_reference_regimes: tuple[RegimeName, ...] = (),
    edge_target_regimes: tuple[RegimeName, ...] = (),
    enable_jit: bool = True,
) -> SolverBuildContext:
    action_nodes = {
        name: jnp.arange(extent, dtype=jnp.int32)
        for name, extent in zip(action_names, actions_grid_shapes, strict=True)
    }
    discrete_action_names = frozenset(discrete_actions)
    state_action_space = StateActionSpace(
        states=MappingProxyType({}),
        discrete_actions=MappingProxyType(
            {
                name: nodes
                for name, nodes in action_nodes.items()
                if name in discrete_action_names
            }
        ),
        continuous_actions=MappingProxyType(
            {
                name: nodes
                for name, nodes in action_nodes.items()
                if name not in discrete_action_names
            }
        ),
        state_and_discrete_action_names=tuple(
            name for name in action_names if name in discrete_action_names
        ),
    )

    # The classifier owns only these route facts. Use a real context instance so
    # its runtime type boundary remains exercised without fabricating unrelated
    # model-processing inputs.
    context = object.__new__(SolverBuildContext)
    attributes = {
        "state_action_space": state_action_space,
        "Q_and_F_functions": MappingProxyType({0: Q_and_F}),
        "has_taste_shocks": has_taste_shocks,
        "enable_jit": enable_jit,
        "co_map_state_names": co_map_state_names,
        "co_map_v_arr_in_axes": tuple(
            MappingProxyType({"target": 0}) for _ in co_map_state_names
        ),
        "edge_reference_regimes": edge_reference_regimes,
        "edge_target_regimes": edge_target_regimes,
    }
    for name, value in attributes.items():
        object.__setattr__(context, name, value)
    return context


def _width_collision_Q_and_F(
    *,
    action: IntND,
    _lcm_action_block_width: FloatND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[IntND, BoolND]:
    del _lcm_action_block_width, next_regime_to_V_arr
    return action, jnp.ones((), dtype=bool)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {},
            _ActionStreamingDisposition.STREAMED,
            id="singleton-hard-max",
        ),
        pytest.param(
            {"co_map_state_names": ("kind", "region")},
            _ActionStreamingDisposition.STREAMED,
            id="singleton-hard-max-plus-ordinary-co-map",
        ),
        pytest.param(
            {
                "edge_target_regimes": ("target",),
                "edge_reference_regimes": ("fallback",),
            },
            _ActionStreamingDisposition.STREAMED,
            id="gated-target-and-reference-without-co-map",
        ),
        pytest.param(
            {"Q_and_F": _width_collision_Q_and_F},
            _ActionStreamingDisposition.STREAMED,
            id="width-collision-is-remapped",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
            },
            _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL,
            id="singleton-ev1",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "co_map_state_names": ("kind", "region"),
            },
            _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL,
            id="singleton-ev1-plus-ordinary-co-map",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "edge_reference_regimes": ("reference",),
            },
            _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL,
            id="singleton-ev1-edge-reference-without-co-map",
        ),
        pytest.param(
            {"enable_jit": False},
            _ActionStreamingDisposition.STREAMED,
            id="singleton-hard-max-jit-disabled",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "enable_jit": False,
            },
            _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL,
            id="singleton-ev1-jit-disabled",
        ),
        pytest.param(
            {"action_names": (), "actions_grid_shapes": ()},
            _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT,
            id="no-actions",
        ),
        pytest.param(
            {"actions_grid_shapes": (1,)},
            _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT,
            id="unit-action-product",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "actions_grid_shapes": (1,),
            },
            _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT,
            id="unit-action-product-precedes-ev1-gate",
        ),
        pytest.param(
            {
                "co_map_state_names": ("kind", "region"),
                "edge_target_regimes": ("target",),
                "edge_reference_regimes": ("fallback",),
            },
            _ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL,
            id="co-map-plus-gated-edge-reference",
        ),
        pytest.param(
            {
                "has_taste_shocks": True,
                "discrete_actions": ("action",),
                "co_map_state_names": ("kind", "region"),
                "edge_reference_regimes": ("reference",),
            },
            _ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL,
            id="singleton-ev1-co-map-plus-edge-reference",
        ),
        pytest.param(
            {"has_taste_shocks": True, "discrete_actions": ()},
            _ActionStreamingDisposition.UNSUPPORTED_EV1_WITHOUT_DISCRETE_ACTION,
            id="singleton-ev1-without-discrete-action",
        ),
    ],
)
def test_action_streaming_route_classification_matrix(
    *, overrides: _RouteOverrides, expected: _ActionStreamingDisposition
) -> None:
    context = _action_streaming_route_context(**overrides)

    actual = inspect.unwrap(_classify_action_streaming)(context=context)

    assert actual is expected


def _evaluate_kernel(
    *,
    func: Callable[..., BoundedMaxOutput],
    arguments: QAndFKwargs,
    execution: str,
) -> FloatND:
    """Run a single's kernel in its execution mode and read the member's value."""
    if execution == "eager":
        return _self_value(func(**arguments))
    jitted = jax.jit(func)
    if execution == "jit":
        return _self_value(jitted(**arguments))
    return _self_value(jitted.lower(**arguments).compile()(**arguments))


def _parameter_width_collision_Q_and_F(
    *,
    action: IntND,
    _lcm_action_block_width: FloatND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    del next_regime_to_V_arr
    return _lcm_action_block_width + action, jnp.ones((), dtype=bool)


def test_streaming_width_keyword_collision_fails_closed() -> None:
    """Direct builders must keep planner width outside Q arguments."""
    with pytest.raises(ValueError, match="collides with a streamed core argument"):
        get_streaming_max_Q_over_a(
            **_single(
                Q_and_F=_parameter_width_collision_Q_and_F,
                batch_sizes={},
                action_names=("action",),
                state_names=(),
            )
        )


@pytest.mark.parametrize("block_width", [1, 3])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
def test_streaming_width_keyword_stays_separate_from_Q_parameter(
    *, block_width: int, execution: str
) -> None:
    selected_width_keyword = "_lcm_action_block_width_1"
    kernel_kwargs = _single(
        Q_and_F=_parameter_width_collision_Q_and_F,
        batch_sizes={},
        action_names=("action",),
        state_names=(),
    )
    dense = get_max_Q_over_a(**kernel_kwargs)
    raw_streamed = get_streaming_max_Q_over_a(
        **kernel_kwargs,
        action_width_keyword=selected_width_keyword,
    )
    signature = inspect.signature(raw_streamed)

    assert "_lcm_action_block_width" in signature.parameters
    assert selected_width_keyword in signature.parameters
    assert selected_width_keyword != "_lcm_action_block_width"

    streamed = functools.partial(
        raw_streamed,
        **{selected_width_keyword: block_width},
    )
    arguments = {
        "action": jnp.arange(4, dtype=jnp.int32),
        "_lcm_action_block_width": jnp.asarray(7.5),
        "next_regime_to_V_arr": MappingProxyType({}),
    }

    expected = _evaluate_kernel(func=dense, arguments=arguments, execution=execution)
    actual = _evaluate_kernel(
        func=streamed,
        arguments=arguments,
        execution=execution,
    )

    eps = np.finfo(np.asarray(actual).dtype).eps
    assert selected_width_keyword not in arguments
    assert_allclose(actual, expected, rtol=8 * eps, atol=8 * eps)
    assert_allclose(actual, 10.5, rtol=8 * eps, atol=8 * eps)


@categorical(ordered=False)
class _WidthRegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt
    single: ScalarInt
    single_terminal: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=4)
_N_ACTIONS = 18


def _consumption(*, wealth: ContinuousState, share: ContinuousAction) -> FloatND:
    return wealth * share


def _utility_f(*, consumption: FloatND, work: DiscreteAction) -> FloatND:
    return consumption - 0.25 * consumption**2 - 0.5 * work


def _utility_m(*, consumption: FloatND, work: DiscreteAction) -> FloatND:
    return 0.5 * consumption + work * (1.0 - consumption)


def _next_wealth(
    *, wealth: ContinuousState, consumption: FloatND, work: DiscreteAction
) -> ContinuousState:
    return 1.0 + 0.5 * (wealth - consumption) + work


def _width_regime(*, household: bool, terminal: bool) -> Regime:
    utility = (
        CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        if household
        else _utility_f
    )
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={} if terminal else {"wealth": _next_wealth},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "share": LinSpacedGrid(start=0.1, stop=0.9, n_points=_N_ACTIONS // 2),
        },
        functions={"consumption": _consumption, "utility": utility},
    )


@functools.cache
def _solve_and_simulate_at_action_width(
    width: int,
) -> tuple[Model, SolutionResult, pd.DataFrame]:
    """Solve and simulate a couple and a single, both on the household kernels."""
    model = Model(
        regimes={
            "couple": _width_regime(household=True, terminal=False),
            "couple_terminal": _width_regime(household=True, terminal=True),
            "single": _width_regime(household=False, terminal=False),
            "single_terminal": _width_regime(household=False, terminal=True),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_WidthRegimeId,
        initial_nodes={0: ("couple", "single")},
        edges={"couple": {"couple_terminal": 0}, "single": {"single_terminal": 0}},
        execution_config=ExecutionConfig(axis_widths={"action_product": width}),
    )
    params = {
        regime: {"koopmans_aggregator": {"discount_factor": 0.9}}
        for regime in ("couple", "single")
    }
    solution = model.solve(params=params, log_level="off")
    n_agents = 8
    frame = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.linspace(1.0, 4.0, n_agents),
            "age": jnp.zeros(n_agents),
            "regime_id": jnp.where(
                jnp.arange(n_agents) % 2 == 0,
                _WidthRegimeId.couple,
                _WidthRegimeId.single,
            ).astype(jnp.int32),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe()
    return model, solution, frame


@pytest.mark.parametrize("width", [1, 2, 7])
def test_households_stream_to_the_values_and_choices_of_the_whole_product(
    width: int,
) -> None:
    """Blocks of any width give the whole product's values and simulated choices.

    A two-member couple and a single, the household of the one member `"self"`,
    solve and decide over their actions in blocks of `width`, folded by the
    household reduction. The household argmax is exact, so every simulated
    action and state is bit for bit the one a block covering all actions gives,
    and so is every solved value at a width of two or more. A block of one
    action is a different compiled program, which may round the action values
    differently in the last bits, so its solved values agree within eight units
    in the last place. A simulated value is the action value at the winner,
    evaluated at the agent's own state by a program compiled for the block's
    width, so it agrees to a few units in the last place.
    """
    model, solution, frame = _solve_and_simulate_at_action_width(width)
    _, whole_solution, whole_frame = _solve_and_simulate_at_action_width(_N_ACTIONS)

    for regime in ("couple", "single"):
        engine_regime = model._regimes[regime]
        kernel = cast(
            "_GridSearchPeriodKernel", engine_regime.solution.period_kernels[0]
        )
        for program in (
            kernel.core_programs()["main"],
            engine_regime.simulation.programs.decision[0],
        ):
            (axis,) = program.requirements.reduced_axes
            assert axis.name == ACTION_PRODUCT_AXIS
            assert axis.reduction is COLLECTIVE_HARD_MAX_REDUCTION
    for period, values in whole_solution.values.items():
        for regime, expected in values.items():
            got = np.asarray(solution.values[period][regime])
            if width == 1:
                assert_agrees_to_ulp(
                    got=got, expected=expected, n_ulp=8, err_msg=f"{period} {regime}"
                )
            else:
                assert got.tobytes() == np.asarray(expected).tobytes(), (period, regime)
    values = [column for column in frame.columns if column.startswith("value")]
    pd.testing.assert_frame_equal(
        frame.drop(columns=values), whole_frame.drop(columns=values), check_exact=True
    )
    for column in values:
        assert_agrees_to_ulp(
            got=frame[column].to_numpy(),
            expected=whole_frame[column].to_numpy(),
            n_ulp=4,
            err_msg=column,
        )


_N_CELLS = 2000
_COUPLE = ("f", "m")


def _interpolated_Q_and_F(
    *,
    consumption: FloatND,
    wealth: FloatND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    """Log consumption plus a continuation interpolated linearly on a wealth grid."""
    v_arr = next_regime_to_V_arr["next"]
    position = jnp.clip((wealth - consumption) * (v_arr.shape[0] - 1) / 10.0, 0, None)
    lower = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, v_arr.shape[0] - 2)
    share = position - lower
    continuation = (1 - share) * v_arr[lower] + share * v_arr[lower + 1]
    return jnp.log(consumption) + 0.95 * continuation, consumption <= wealth


def _interpolated_couple_Q_and_F(
    *,
    consumption: FloatND,
    wealth: FloatND,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
) -> tuple[FloatND, BoolND]:
    """Two members' values of the interpolated kernel, on a trailing member axis."""
    value, feasible = _interpolated_Q_and_F(
        consumption=consumption,
        wealth=wealth,
        next_regime_to_V_arr=next_regime_to_V_arr,
    )
    return jnp.stack([value, 0.5 * value - consumption], axis=-1), feasible


def _interpolated_arguments(
    n_actions: int,
) -> dict[str, jax.Array | Mapping[RegimeName, FloatND]]:
    return {
        "consumption": jnp.linspace(0.01, 10.0, n_actions),
        "wealth": jnp.linspace(0.1, 10.0, _N_CELLS),
        "next_regime_to_V_arr": MappingProxyType({"next": jnp.linspace(0.0, 1.0, 50)}),
    }


def _interpolated_kernel_kwargs(*, household: bool) -> _KernelKwargs:
    if not household:
        return _single(
            Q_and_F=_interpolated_Q_and_F,
            batch_sizes={"wealth": 0},
            action_names=("consumption",),
            state_names=("wealth",),
        )
    return {
        "Q_and_F": _interpolated_couple_Q_and_F,
        "batch_sizes": {"wealth": 0},
        "action_names": ("consumption",),
        "state_names": ("wealth",),
        "members": _COUPLE,
        "pareto_weights": build_pareto_weights(
            objective=None, members=_COUPLE, state_names=frozenset()
        ),
    }


def _temp_bytes(compiled: jax.stages.Compiled) -> int:
    analysis = compiled.memory_analysis()
    assert analysis is not None
    return analysis.temp_size_in_bytes


def _solve_kernel(
    *, household: bool, width: int | None
) -> Callable[..., BoundedMaxOutput]:
    """The solve kernel; no width is the dense route."""
    kwargs = _interpolated_kernel_kwargs(household=household)
    if width is None:
        return get_max_Q_over_a(**kwargs)
    return functools.partial(
        get_streaming_max_Q_over_a(**kwargs), _lcm_action_block_width=width
    )


def _solve_temp_bytes(*, household: bool, n_actions: int, width: int | None) -> int:
    """Temporary bytes of a compiled solve kernel; no width is the dense route."""
    kernel = _solve_kernel(household=household, width=width)
    arguments = _interpolated_arguments(n_actions)
    return _temp_bytes(jax.jit(kernel).lower(**arguments).compile())


def _decision_kernel(
    *, household: bool, n_actions: int
) -> Callable[..., tuple[IntND, FloatND]]:
    """The simulation decision over the whole product, mapped over wealth."""
    kwargs = _interpolated_kernel_kwargs(household=household)
    cell = build_streaming_collective_max_Q_over_a(
        publishes_identity=True,
        Q_and_F=kwargs["Q_and_F"],
        action_names=("consumption",),
        block_width=n_actions,
        members=kwargs["members"],
        weights=kwargs["pareto_weights"].compute(),
    )

    # keyword-only-exempt: library-callback=jax.vmap
    def decide(
        wealth: FloatND, consumption: FloatND, V_arr: Mapping[RegimeName, FloatND]
    ) -> tuple[IntND, FloatND]:
        result = cast(
            "CollectiveHardMaxResult",
            cell(consumption=consumption, wealth=wealth, next_regime_to_V_arr=V_arr),
        )
        return result.best_global_action_id, result.best_member_values

    return jax.vmap(decide, in_axes=(0, None, None))


def _decision_temp_bytes(*, household: bool, n_actions: int, width: int | None) -> int:
    """Temporary bytes of a compiled decision kernel over the whole product."""
    del width
    arguments = _interpolated_arguments(n_actions)
    return _temp_bytes(
        jax.jit(_decision_kernel(household=household, n_actions=n_actions))
        .lower(
            arguments["wealth"],
            arguments["consumption"],
            arguments["next_regime_to_V_arr"],
        )
        .compile()
    )


@pytest.mark.parametrize("household", [False, True], ids=["single", "couple"])
@pytest.mark.parametrize(
    ("temp_bytes", "streamed"),
    [
        (_solve_temp_bytes, None),
        (_solve_temp_bytes, False),
        (_solve_temp_bytes, True),
        (_decision_temp_bytes, False),
    ],
    ids=["solve-dense", "solve-whole-product", "solve-streamed", "decision"],
)
def test_household_reduction_keeps_no_value_per_cell_and_action(
    *, temp_bytes: Callable[..., int], streamed: bool | None, household: bool
) -> None:
    """The action values fuse into the household's reduction: none is stored.

    The kernel's temporary grows by less than one float per cell and added
    action when the reduced action block doubles: on the dense route, over the
    whole product (`n_actions` follows the width) and in streamed blocks of a
    fixed product.
    """
    widths = (512, 1024)
    temps = [
        temp_bytes(
            household=household,
            n_actions=4096 if streamed else block,
            width=None if streamed is None else block,
        )
        for block in widths
    ]
    itemsize = jnp.asarray(0.0).dtype.itemsize

    assert temps[1] - temps[0] < _N_CELLS * (widths[1] - widths[0]) * itemsize, temps


def _action_reduce_operand_kinds(
    *, jaxpr: Jaxpr, n_actions: int
) -> list[tuple[str, ...]]:
    """Dtype kinds of the operands of each reduce over a trailing action axis."""
    kinds = []
    for eqn in jaxpr.eqns:
        operands = [cast("ShapedArray", var.aval) for var in eqn.invars]
        if eqn.primitive.name == "reduce" and operands[0].shape[-1:] == (n_actions,):
            kinds.append(
                tuple(operand.dtype.kind for operand in operands[: len(operands) // 2])
            )
        for value in eqn.params.values():
            nested = getattr(value, "jaxpr", value)
            if isinstance(nested, Jaxpr):
                kinds.extend(
                    _action_reduce_operand_kinds(jaxpr=nested, n_actions=n_actions)
                )
    return kinds


def test_single_solve_reduces_value_and_nan_flag_decision_adds_identity() -> None:
    """The solve of one member reduces its actions on the value and the NaN flag.

    Nothing in the solve reads the winner's identity, so its one reduction over
    the actions carries the value's order key and the NaN flag only. The
    simulation decision publishes the identity and carries it as well.
    """
    n_actions = 512
    arguments = _interpolated_arguments(n_actions)
    solve = jax.make_jaxpr(_solve_kernel(household=False, width=n_actions))(**arguments)
    decision = jax.make_jaxpr(_decision_kernel(household=False, n_actions=n_actions))(
        arguments["wealth"],
        arguments["consumption"],
        arguments["next_regime_to_V_arr"],
    )

    assert _action_reduce_operand_kinds(jaxpr=solve.jaxpr, n_actions=n_actions) == [
        ("b", "i")
    ]
    assert _action_reduce_operand_kinds(jaxpr=decision.jaxpr, n_actions=n_actions) == [
        ("b", "i", "i")
    ]
