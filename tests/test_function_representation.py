from __future__ import annotations

import re
from dataclasses import make_dataclass
from functools import partial

import jax.numpy as jnp
import pytest
from dags.exceptions import InvalidFunctionArgumentsError

from lcm import LinspaceGrid
from lcm.dispatchers import productmap
from lcm.function_representation import (
    _fail_if_interpolation_axes_are_not_last,
    _get_coordinate_finder,
    _get_interpolator,
    _get_label_translator,
    _get_lookup_function,
    get_value_function_representation,
)
from lcm.grids import DiscreteGrid
from lcm.interfaces import (
    StateSpaceInfo,
)


@pytest.fixture
def binary_discrete_grid():
    return DiscreteGrid(
        make_dataclass("BinaryCategory", [("a", bool, False), ("b", bool, True)])
    )


@pytest.fixture
def dummy_continuous_grid():
    return LinspaceGrid(start=0, stop=1, n_points=2)


def test_function_evaluator_with_one_continuous_variable():
    wealth_grid = LinspaceGrid(start=-3, stop=3, n_points=7)

    state_space_info = StateSpaceInfo(
        states_names=("wealth",),
        discrete_states={},
        continuous_states={
            "wealth": wealth_grid,
        },
    )

    next_V_arr = jnp.pi * wealth_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_value_function_representation(state_space_info)

    # partial the function values into the evaluator
    func = partial(evaluator, next_V_arr=next_V_arr)

    # test the evaluator
    got = func(next_wealth=0.5)
    expected = 0.5 * jnp.pi + 2
    assert jnp.allclose(got, expected)


def test_function_evaluator_with_one_discrete_variable(binary_discrete_grid):
    next_V_arr = jnp.array([1, 2])

    state_space_info = StateSpaceInfo(
        states_names=("working",),
        discrete_states={"working": binary_discrete_grid},
        continuous_states={},
    )

    # create the evaluator
    evaluator = get_value_function_representation(state_space_info)

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

    The setup of state_space_info here is quite long. Usually these inputs will be
    generated from a model specification.

    """
    # create a value function array
    discrete_part = jnp.arange(4).repeat(6 * 7).reshape((2, 2, 6, 7)) * 100

    cont_func = productmap(lambda x, y: x + y, ("x", "y"))
    cont_part = cont_func(x=jnp.linspace(100, 1100, 6), y=jnp.linspace(-3, 3, 7))

    next_V_arr = discrete_part + cont_part

    # create info on discrete variables
    discrete_vars = {
        "retired": binary_discrete_grid,
        "insured": binary_discrete_grid,
    }

    # create info on continuous grids
    continuous_vars = {
        "wealth": LinspaceGrid(start=100, stop=1100, n_points=6),
        "human_capital": LinspaceGrid(start=-3, stop=3, n_points=7),
    }

    # create info on axis of value function array
    var_names = ("retired", "insured", "wealth", "human_capital")

    state_space_info = StateSpaceInfo(
        states_names=var_names,
        discrete_states=discrete_vars,
        continuous_states=continuous_vars,
    )

    # create the evaluator
    evaluator = get_value_function_representation(
        state_space_info=state_space_info,
    )

    # test the evaluator; note that the prefix 'next_' is added to the variable names
    # by default, and that the argument name of the value function array is 'next_V_arr'
    # by default; these can be changed when calling get_value_function_representation
    out = evaluator(
        next_retired=1,
        next_insured=0,
        next_wealth=600,
        next_human_capital=1.5,
        next_V_arr=next_V_arr,
    )

    assert jnp.allclose(out, 801.5)


def test_get_label_translator_with_args():
    func = _get_label_translator(
        in_name="schooling",
    )
    assert func(1) == 1


def test_get_label_translator_with_kwargs():
    func = _get_label_translator(
        in_name="schooling",
    )
    assert func(schooling=1) == 1


def test_get_label_translator_wrong_kwarg():
    func = _get_label_translator(
        in_name="schooling",
    )
    with pytest.raises(
        InvalidFunctionArgumentsError,
        match=re.escape("translate_label() got unexpected keyword argument health"),
    ):
        func(health=1)


def test_get_lookup_function():
    array = jnp.arange(6).reshape(3, 2)
    func = _get_lookup_function(array_name="my_array", axis_names=["a", "b"])

    pure_lookup_func = partial(func, my_array=array)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4


def test_get_coordinate_finder():
    find_coordinate = _get_coordinate_finder(
        in_name="wealth",
        grid=LinspaceGrid(start=0, stop=10, n_points=21),
    )

    calculated = find_coordinate(wealth=5.75)
    assert calculated == 11.5


def test_get_interpolator():
    interpolate = _get_interpolator(
        name_of_values_on_grid="vf",
        axis_names=["wealth", "working"],
    )

    def _utility(wealth, working):
        return 2 * wealth - working

    prod_utility = productmap(_utility, variables=("wealth", "working"))

    values = prod_utility(
        wealth=jnp.arange(4, dtype=float),
        working=jnp.arange(3, dtype=float),
    )

    calculated = interpolate(vf=values, wealth=2.5, working=2)

    assert calculated == 3


# ======================================================================================
# Illustrative
# ======================================================================================


@pytest.mark.illustrative
def test_get_function_evaluator_illustrative():
    a_grid = LinspaceGrid(start=0, stop=1, n_points=3)

    state_space_info = StateSpaceInfo(
        states_names=("a",),
        discrete_states={},
        continuous_states={
            "a": a_grid,
        },
    )

    values = jnp.pi * a_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_value_function_representation(
        state_space_info=state_space_info,
        name_of_values_on_grid="values_name",
        input_prefix="prefix_",
    )

    # partial the function values into the evaluator
    f = partial(evaluator, values_name=values)

    got = f(prefix_a=0.25)
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
        grid=LinspaceGrid(start=0, stop=1, n_points=3),
    )

    assert find_coordinate(a=0) == 0
    assert find_coordinate(a=0.5) == 1
    assert find_coordinate(a=1) == 2
    assert find_coordinate(a=0.25) == 0.5


@pytest.mark.illustrative
def test_get_interpolator_illustrative():
    interpolate = _get_interpolator(
        name_of_values_on_grid="test_name",
        axis_names=["a", "b"],
    )

    def f(a, b):
        return a - b

    prod_f = productmap(f, variables=("a", "b"))

    values = prod_f(a=jnp.arange(2, dtype=float), b=jnp.arange(3, dtype=float))

    assert interpolate(test_name=values, a=0.5, b=0) == 0.5
    assert interpolate(test_name=values, a=0.5, b=1) == -0.5
    assert interpolate(test_name=values, a=0, b=0.5) == -0.5
    assert interpolate(test_name=values, a=0.5, b=1.5) == -1


@pytest.mark.illustrative
def test_fail_if_interpolation_axes_are_not_last_illustrative(dummy_continuous_grid):
    # Empty intersection of var_names and continuous_vars
    # ==================================================================================

    state_space_info = StateSpaceInfo(
        states_names=("a", "b"),
        continuous_states={
            "c": dummy_continuous_grid,
        },
        discrete_states={},
    )

    _fail_if_interpolation_axes_are_not_last(state_space_info)  # does not fail

    # Non-empty intersection but correct order
    # ==================================================================================

    state_space_info = StateSpaceInfo(
        states_names=("a", "b", "c"),
        continuous_states={
            "b": dummy_continuous_grid,
            "c": dummy_continuous_grid,
            "d": dummy_continuous_grid,
        },
        discrete_states={},
    )

    _fail_if_interpolation_axes_are_not_last(state_space_info)  # does not fail

    # Non-empty intersection and in-correct order
    # ==================================================================================

    state_space_info = StateSpaceInfo(
        states_names=("b", "c", "a"),  # "b", "c" are not last anymore
        continuous_states={
            "b": dummy_continuous_grid,
            "c": dummy_continuous_grid,
            "d": dummy_continuous_grid,
        },
        discrete_states={},
    )

    with pytest.raises(ValueError, match="Continuous variables need to be the last"):
        _fail_if_interpolation_axes_are_not_last(state_space_info)


def test_function_evaluator_performs_linear_extrapolation():
    wealth_grid = LinspaceGrid(start=0, stop=3, n_points=7)

    state_space_info = StateSpaceInfo(
        states_names=("wealth",),
        discrete_states={},
        continuous_states={
            "wealth": wealth_grid,
        },
    )

    next_V_arr = jnp.pi * wealth_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_value_function_representation(state_space_info)

    # partial the function values into the evaluator
    func = partial(evaluator, next_V_arr=next_V_arr)

    # test the evaluator on values outside the grid
    wealth_outside_of_grid = [-0.5, 3.5, 10]
    # We expect linear extrapolation
    expected = jnp.pi * jnp.array(wealth_outside_of_grid) + 2

    got = jnp.array([func(next_wealth=w) for w in wealth_outside_of_grid])
    assert jnp.allclose(got, expected)
