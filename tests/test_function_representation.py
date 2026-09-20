from dataclasses import make_dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.regime_building.V import (
    VInterpolationInfo,
    _fail_if_interpolation_axes_are_not_last,
    _get_coordinate_finder,
    _get_interpolator,
    _get_lookup_function,
    get_V_interpolator,
)
from _lcm.utils.dispatchers import productmap
from lcm import LinSpacedGrid, NormalIIDProcess
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION


@pytest.fixture
def binary_discrete_grid():
    cls = make_dataclass("BinaryCategory", [("a", ScalarInt), ("b", ScalarInt)])
    type.__setattr__(cls, "a", jnp.int32(0))
    type.__setattr__(cls, "b", jnp.int32(1))
    return DiscreteGrid(category_class=cls)


@pytest.fixture
def dummy_continuous_grid():
    return LinSpacedGrid(start=0, stop=1, n_points=2)


def test_function_evaluator_with_one_continuous_variable():
    wealth_grid = LinSpacedGrid(start=-3, stop=3, n_points=7)

    v_interpolation_info = VInterpolationInfo(
        state_names=("wealth",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType(
            {
                "wealth": wealth_grid,
            }
        ),
    )

    next_V_arr = jnp.pi * wealth_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
    )

    # partial the function values into the evaluator
    func = partial(evaluator, next_V_arr=next_V_arr)

    # test the evaluator
    got = func(next_wealth=jnp.asarray(0.5))
    expected = 0.5 * jnp.pi + 2
    assert jnp.allclose(got, expected)


def test_function_evaluator_with_one_discrete_variable(binary_discrete_grid):
    next_V_arr = jnp.array([1, 2])

    v_interpolation_info = VInterpolationInfo(
        state_names=("working",),
        discrete_states=MappingProxyType({"working": binary_discrete_grid}),
        continuous_states=MappingProxyType({}),
    )

    # create the evaluator
    evaluator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
    )

    # partial the function values into the evaluator
    func = partial(evaluator, next_V_arr=next_V_arr)

    # test the evaluator
    assert func(next_working=0) == 1
    assert func(next_working=1) == 2


def test_function_evaluator(binary_discrete_grid):
    """Test get_precalculated_function_evaluator in simple example.

    - One discrete state variable: retired (True, False)
    - One discrete action variable: insured ("yes", "no")
    - Two continuous state variables:
        - wealth (linspace(100, 1100, 6))
        - human_capital (linspace(-3, 3, 7))

    The utility function is wealth + human_capital + c. c takes a different
    value for each discrete state action combination.

    The setup of v_interpolation_info here is quite long. Usually these inputs will be
    generated from a model specification.

    """
    # create a value function array
    discrete_part = jnp.arange(4).repeat(6 * 7).reshape((2, 2, 6, 7)) * 100

    cont_func = productmap(
        func=lambda x, y: x + y,
        variables=("x", "y"),
        batch_sizes=dict.fromkeys(("x", "y"), 0),
    )
    cont_part = cont_func(x=jnp.linspace(100, 1100, 6), y=jnp.linspace(-3, 3, 7))

    next_V_arr = discrete_part + cont_part

    # create info on discrete variables
    discrete_vars = {
        "retired": binary_discrete_grid,
        "insured": binary_discrete_grid,
    }

    # create info on continuous grids
    continuous_vars: dict[str, ContinuousGrid] = {
        "wealth": LinSpacedGrid(start=100, stop=1100, n_points=6),
        "human_capital": LinSpacedGrid(start=-3, stop=3, n_points=7),
    }

    # create info on axis of value function array
    var_names = ("retired", "insured", "wealth", "human_capital")

    v_interpolation_info = VInterpolationInfo(
        state_names=var_names,
        discrete_states=MappingProxyType(discrete_vars),
        continuous_states=MappingProxyType(continuous_vars),
    )

    # create the evaluator
    evaluator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
    )

    out = evaluator(
        next_retired=1,
        next_insured=0,
        next_wealth=jnp.asarray(600.0),
        next_human_capital=jnp.asarray(1.5),
        next_V_arr=next_V_arr,
    )

    assert jnp.allclose(out, 801.5)


def test_get_lookup_function():
    array = jnp.arange(6).reshape(3, 2)
    func = _get_lookup_function(array_name="my_array", axis_names=["a", "b"])

    pure_lookup_func = partial(func, my_array=array)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4


def test_get_coordinate_finder():
    find_coordinate = _get_coordinate_finder(
        in_name="wealth",
        grid=LinSpacedGrid(start=0, stop=10, n_points=21),
    )
    find_coordinate = partial(find_coordinate)
    calculated = find_coordinate(wealth=jnp.asarray(5.75))
    assert calculated == 11.5


def test_get_interpolator():
    interpolate = _get_interpolator(
        name_of_values_on_grid="vf",
        axis_names=["wealth", "working"],
    )

    def _utility(*, wealth, working):
        return 2 * wealth - working

    prod_utility = productmap(
        func=_utility,
        variables=("wealth", "working"),
        batch_sizes=dict.fromkeys(("wealth", "working"), 0),
    )

    values = prod_utility(
        wealth=jnp.arange(4, dtype=float),
        working=jnp.arange(3, dtype=float),
    )

    calculated = interpolate(vf=values, wealth=2.5, working=2)

    assert calculated == 3


@pytest.mark.illustrative
def test_get_function_evaluator_illustrative():
    a_grid = LinSpacedGrid(start=0, stop=1, n_points=3)

    v_interpolation_info = VInterpolationInfo(
        state_names=("a",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType(
            {
                "a": a_grid,
            }
        ),
    )

    values = jnp.pi * a_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        V_arr_name="values_name",
        state_prefix="prefix_",
    )

    # partial the function values into the evaluator
    f = partial(evaluator, values_name=values)

    got = f(prefix_a=jnp.asarray(0.25))
    expected = jnp.pi * 0.25 + 2

    assert jnp.allclose(got, expected)


@pytest.mark.illustrative
def test_get_lookup_function_illustrative():
    values = jnp.array([0, 1, 4])
    func = _get_lookup_function(array_name="xyz", axis_names=["a"])
    pure_lookup_func = partial(func, xyz=values)

    assert pure_lookup_func(a=2) == 4


@pytest.mark.illustrative
def test_get_coordinate_finder_illustrative():
    find_coordinate = _get_coordinate_finder(
        in_name="a",
        grid=LinSpacedGrid(start=0, stop=1, n_points=3),
    )
    assert find_coordinate(a=jnp.asarray(0.0)) == 0
    assert find_coordinate(a=jnp.asarray(0.5)) == 1
    assert find_coordinate(a=jnp.asarray(1.0)) == 2
    assert find_coordinate(a=jnp.asarray(0.25)) == 0.5


@pytest.mark.illustrative
def test_get_interpolator_illustrative():
    interpolate = _get_interpolator(
        name_of_values_on_grid="test_name",
        axis_names=["a", "b"],
    )

    def f(*, a, b):
        return a - b

    prod_f = productmap(
        func=f, variables=("a", "b"), batch_sizes=dict.fromkeys(("a", "b"), 0)
    )

    values = prod_f(a=jnp.arange(2, dtype=float), b=jnp.arange(3, dtype=float))

    assert interpolate(test_name=values, a=0.5, b=0) == 0.5
    assert interpolate(test_name=values, a=0.5, b=1) == -0.5
    assert interpolate(test_name=values, a=0, b=0.5) == -0.5
    assert interpolate(test_name=values, a=0.5, b=1.5) == -1


@pytest.mark.illustrative
def test_fail_if_interpolation_axes_are_not_last_illustrative(dummy_continuous_grid):
    # Empty intersection of var_names and continuous_vars

    v_interpolation_info = VInterpolationInfo(
        state_names=("a", "b"),
        continuous_states=MappingProxyType(
            {
                "c": dummy_continuous_grid,
            }
        ),
        discrete_states=MappingProxyType({}),
    )

    _fail_if_interpolation_axes_are_not_last(v_interpolation_info)  # does not fail

    # Non-empty intersection but correct order

    v_interpolation_info = VInterpolationInfo(
        state_names=("a", "b", "c"),
        continuous_states=MappingProxyType(
            {
                "b": dummy_continuous_grid,
                "c": dummy_continuous_grid,
                "d": dummy_continuous_grid,
            }
        ),
        discrete_states=MappingProxyType({}),
    )

    _fail_if_interpolation_axes_are_not_last(v_interpolation_info)  # does not fail

    # Non-empty intersection and in-correct order

    v_interpolation_info = VInterpolationInfo(
        state_names=("b", "c", "a"),  # "b", "c" are not last anymore
        continuous_states=MappingProxyType(
            {
                "b": dummy_continuous_grid,
                "c": dummy_continuous_grid,
                "d": dummy_continuous_grid,
            }
        ),
        discrete_states=MappingProxyType({}),
    )

    with pytest.raises(ValueError, match="Continuous variables need to be the last"):
        _fail_if_interpolation_axes_are_not_last(v_interpolation_info)


def test_function_evaluator_performs_linear_extrapolation():
    wealth_grid = LinSpacedGrid(start=0, stop=3, n_points=7)

    v_interpolation_info = VInterpolationInfo(
        state_names=("wealth",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType(
            {
                "wealth": wealth_grid,
            }
        ),
    )

    next_V_arr = jnp.pi * wealth_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
    )

    # partial the function values into the evaluator
    func = partial(evaluator, next_V_arr=next_V_arr)

    # test the evaluator on values outside the grid
    wealth_outside_of_grid = [-0.5, 3.5, 10.0]
    # We expect linear extrapolation
    expected = jnp.pi * jnp.array(wealth_outside_of_grid) + 2

    got = jnp.array([func(next_wealth=jnp.asarray(w)) for w in wealth_outside_of_grid])
    assert jnp.allclose(got, expected)


@pytest.fixture
def mixed_corner_interpolator(
    binary_discrete_grid: DiscreteGrid,
) -> tuple[VInterpolationInfo, jax.Array]:
    """A bilinear surface with distinct discrete labels and unequal grid extents."""
    cls = make_dataclass(
        "ThreeCategories", [(name, ScalarInt) for name in ("a", "b", "c")]
    )
    for index, name in enumerate(("a", "b", "c")):
        type.__setattr__(cls, name, jnp.int32(index))
    info = VInterpolationInfo(
        state_names=("group", "kind", "x", "y"),
        discrete_states=MappingProxyType(
            {
                "group": binary_discrete_grid,
                "kind": DiscreteGrid(category_class=cls),
            }
        ),
        continuous_states=MappingProxyType(
            {
                "x": LinSpacedGrid(start=0, stop=3, n_points=4),
                "y": LinSpacedGrid(start=-2, stop=2, n_points=5),
            }
        ),
    )
    group = jnp.arange(2)[:, None, None, None]
    kind = jnp.arange(3)[None, :, None, None]
    x = jnp.arange(4)[None, None, :, None]
    y = jnp.arange(-2, 3)[None, None, None, :]
    values = 100 * group + 10 * kind + 2 * x + 3 * y + 0.5 * x * y
    return info, values


def _gather_slice_sizes(jaxpr: Any) -> list[tuple[int, ...]]:
    """Read gather windows recursively from traced calls and control-flow bodies."""
    if hasattr(jaxpr, "eqns"):
        found = []
        for equation in jaxpr.eqns:
            if equation.primitive.name == "gather":
                found.append(tuple(equation.params["slice_sizes"]))
            found.extend(_gather_slice_sizes(equation.params))
        return found
    if hasattr(jaxpr, "jaxpr"):
        return _gather_slice_sizes(jaxpr.jaxpr)
    if isinstance(jaxpr, dict):
        return [
            sizes for value in jaxpr.values() for sizes in _gather_slice_sizes(value)
        ]
    if isinstance(jaxpr, (tuple, list)):
        return [sizes for value in jaxpr for sizes in _gather_slice_sizes(value)]
    return []


@pytest.mark.parametrize("co_mapped", [False, True])
def test_mixed_interpolator_gathers_only_corner_values(
    *, mixed_corner_interpolator: tuple[VInterpolationInfo, jax.Array], co_mapped: bool
) -> None:
    """A mixed lookup reads scalar corners without retaining full continuous grids."""
    info, values = mixed_corner_interpolator
    evaluator = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
        co_map_state_names=("group",) if co_mapped else (),
    )

    # keyword-only-exempt: library-callback=jax.vmap
    def point(
        array: jax.Array,
        group: jax.Array,
        kind: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        kwargs = {"next_V_arr": array, "next_kind": kind, "next_x": x, "next_y": y}
        if not co_mapped:
            kwargs["next_group"] = group
        return evaluator(**kwargs)

    array = values[1] if co_mapped else values
    traced = jax.make_jaxpr(jax.jit(jax.vmap(point, in_axes=(None, 0, 0, 0, 0))))(
        array,
        jnp.asarray([0, 1]),
        jnp.asarray([0, 2]),
        jnp.asarray([1.25, -0.5]),
        jnp.asarray([0.5, 2.5]),
    )
    sizes = _gather_slice_sizes(traced)
    assert sizes, "The traced interpolator must contain actual corner gathers"
    assert all(all(extent == 1 for extent in window) for window in sizes), sizes


@pytest.mark.parametrize("co_mapped", [False, True])
def test_mixed_interpolator_preserves_bilinear_values_and_derivatives(
    *, mixed_corner_interpolator: tuple[VInterpolationInfo, jax.Array], co_mapped: bool
) -> None:
    """Interior reads and extrapolation retain the labeled bilinear value and slope."""
    info, values = mixed_corner_interpolator
    evaluator = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
        co_map_state_names=("group",) if co_mapped else (),
    )

    # keyword-only-exempt: library-callback=jax.vmap
    def point(x: jax.Array, y: jax.Array) -> jax.Array:
        kwargs = {
            "next_V_arr": values[1] if co_mapped else values,
            "next_kind": jnp.int32(2),
            "next_x": x,
            "next_y": y,
        }
        if not co_mapped:
            kwargs["next_group"] = jnp.int32(1)
        return evaluator(**kwargs)

    x = jnp.asarray([1.25, -0.5, 3.5])
    y = jnp.asarray([0.5, -2.5, 2.5])
    expected = 120 + 2 * x + 3 * y + 0.5 * x * y
    np.testing.assert_allclose(
        jax.jit(jax.vmap(point))(x, y),
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    dx, dy = jax.vmap(jax.grad(point, argnums=(0, 1)))(x, y)
    np.testing.assert_allclose(
        dx,
        2 + 0.5 * y,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        dy,
        3 + 0.5 * x,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


def test_mixed_interpolator_ignores_zero_weight_infinite_corners(
    mixed_corner_interpolator: tuple[VInterpolationInfo, jax.Array],
) -> None:
    """An exact grid-node read ignores neighboring infeasible values."""
    info, values = mixed_corner_interpolator
    values = jnp.full_like(values, -jnp.inf).at[1, 2, 1, 2].set(7.0)
    evaluator = get_V_interpolator(
        v_interpolation_info=info, state_prefix="next_", V_arr_name="next_V_arr"
    )
    actual = jax.jit(evaluator)(
        next_V_arr=values,
        next_group=jnp.int32(1),
        next_kind=jnp.int32(2),
        next_x=jnp.asarray(1.0),
        next_y=jnp.asarray(0.0),
    )
    np.testing.assert_array_equal(actual, 7.0)


@pytest.mark.parametrize("co_mapped", [False, True])
def test_entered_process_between_indexed_axes_preserves_physical_coordinates(
    *,
    mixed_corner_interpolator: tuple[VInterpolationInfo, jax.Array],
    co_mapped: bool,
) -> None:
    """An entered process interpolates physical values between indexed axes."""
    base, _ = mixed_corner_interpolator
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=False, mu=0.0, sigma=0.5, n_std=2.0
    )
    info = VInterpolationInfo(
        state_names=("group", "shock", "kind", "x"),
        discrete_states=MappingProxyType(
            {
                "group": base.discrete_states["group"],
                "shock": process,
                "kind": base.discrete_states["kind"],
            }
        ),
        continuous_states=MappingProxyType({"x": base.continuous_states["x"]}),
    )
    group = jnp.arange(2.0)[:, None, None, None]
    shock = process.to_jax()[None, :, None, None]
    kind = jnp.arange(3.0)[None, None, :, None]
    x = jnp.arange(4.0)[None, None, None, :]
    values = 100 * group + 5 * shock + 10 * kind + 2 * x + shock * x
    reader = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="next_V_arr",
        entered_process_names=("shock",),
        co_map_state_names=("group",) if co_mapped else (),
    )
    kwargs = {
        "next_V_arr": values[1] if co_mapped else values,
        "next_kind": jnp.int32(2),
        "next_shock": jnp.asarray(0.25),
        "next_x": jnp.asarray(1.5),
    }
    if not co_mapped:
        kwargs["next_group"] = jnp.int32(1)
    actual = jax.jit(reader)(**kwargs)
    np.testing.assert_allclose(
        actual,
        124.625,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
