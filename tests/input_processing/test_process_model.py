from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid, LinspaceGrid, grid_helpers
from lcm.input_processing.process_model import (
    _get_stochastic_weight_function,
    get_function_info,
    get_grids,
    get_gridspecs,
    get_variable_info,
    process_model,
)
from lcm.mark import StochasticInfo
from tests.test_models import get_model_config


@dataclass
class ModelMock:
    """A model mock for testing the process_model function.

    This dataclass has the same attributes as the Model dataclass, but does not perform
    any checks, which helps us to test the process_model function in isolation.

    """

    n_periods: int
    functions: dict[str, Any]
    actions: dict[str, Any]
    states: dict[str, Any]


@pytest.fixture
def model(binary_category_class):
    def next_c(a, b):
        return a + b

    return ModelMock(
        n_periods=2,
        functions={
            "next_c": next_c,
        },
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
    )


def test_get_function_info(model):
    got = get_function_info(model)
    exp = pd.DataFrame(
        {
            "is_constraint": [False],
            "is_next": [True],
            "is_stochastic_next": [False],
        },
        index=["next_c"],
    )
    assert_frame_equal(got, exp)


def test_get_variable_info(model):
    got = get_variable_info(model)
    exp = pd.DataFrame(
        {
            "is_state": [False, True],
            "is_action": [True, False],
            "is_continuous": [False, False],
            "is_discrete": [True, True],
            "is_stochastic": [False, False],
            "is_auxiliary": [False, True],
        },
        index=["a", "c"],
    )
    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_gridspecs(model):
    got = get_gridspecs(model)
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids(model):
    got = get_grids(model)
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([0, 1]))


def test_process_model_iskhakov_et_al_2017():
    model_config = get_model_config("iskhakov_et_al_2017", n_periods=3)
    model = process_model(model_config)

    # Variable Info
    assert (
        model.variable_info["is_state"].to_numpy()
        == np.array([True, False, True, False])
    ).all()

    assert (
        model.variable_info["is_continuous"].to_numpy()
        == np.array([False, False, True, True])
    ).all()

    # Gridspecs
    wealth_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=model_config.states["wealth"].n_points,  # type: ignore[attr-defined]
    )

    assert model.gridspecs["wealth"] == wealth_grid

    consumption_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=model_config.actions["consumption"].n_points,  # type: ignore[attr-defined]
    )
    assert model.gridspecs["consumption"] == consumption_grid

    assert isinstance(model.gridspecs["retirement"], DiscreteGrid)
    assert model.gridspecs["retirement"].categories == ("working", "retired")
    assert model.gridspecs["retirement"].codes == (0, 1)

    assert isinstance(model.gridspecs["lagged_retirement"], DiscreteGrid)
    assert model.gridspecs["lagged_retirement"].categories == ("working", "retired")
    assert model.gridspecs["lagged_retirement"].codes == (0, 1)

    # Grids
    expected = grid_helpers.linspace(**model_config.actions["consumption"].__dict__)
    assert_array_equal(model.grids["consumption"], expected)

    expected = grid_helpers.linspace(**model_config.states["wealth"].__dict__)
    assert_array_equal(model.grids["wealth"], expected)

    assert (model.grids["retirement"] == jnp.array([0, 1])).all()
    assert (model.grids["lagged_retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, True, False, False, False, False])
    ).all()

    assert (
        model.function_info["is_constraint"].to_numpy()
        == np.array([False, False, False, True, True, False, False])
    ).all()

    assert ~model.function_info.loc["utility"].to_numpy().any()


def test_process_model():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    model = process_model(model_config)

    # Variable Info
    assert (
        model.variable_info["is_state"].to_numpy() == np.array([False, True, False])
    ).all()

    assert (
        model.variable_info["is_continuous"].to_numpy() == np.array([False, True, True])
    ).all()

    # Gridspecs
    wealth_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=model_config.states["wealth"].n_points,  # type: ignore[attr-defined]
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=model_config.actions["consumption"].n_points,  # type: ignore[attr-defined]
    )
    assert model.gridspecs["consumption"] == consumption_specs

    assert isinstance(model.gridspecs["retirement"], DiscreteGrid)
    assert model.gridspecs["retirement"].categories == ("working", "retired")
    assert model.gridspecs["retirement"].codes == (0, 1)

    # Grids
    expected = grid_helpers.linspace(**model_config.actions["consumption"].__dict__)
    assert_array_equal(model.grids["consumption"], expected)

    expected = grid_helpers.linspace(**model_config.states["wealth"].__dict__)
    assert_array_equal(model.grids["wealth"], expected)

    assert (model.grids["retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, False, False, False, False, False])
    ).all()

    assert (
        model.function_info["is_constraint"].to_numpy()
        == np.array([False, False, True, False, False, False, False])
    ).all()

    assert ~model.function_info.loc["utility"].to_numpy().any()


def test_get_stochastic_weight_function():
    def raw_func(health, wealth):
        pass

    raw_func._stochastic_info = StochasticInfo()  # type: ignore[attr-defined]

    variable_info = pd.DataFrame(
        {"is_discrete": [True, True]},
        index=["health", "wealth"],
    )

    got_function = _get_stochastic_weight_function(
        raw_func,
        name="health",
        variable_info=variable_info,
    )

    params = {"shocks": {"health": np.arange(12).reshape(2, 3, 2)}}

    got = got_function(health=1, wealth=0, params=params)
    expected = np.array([6, 7])
    assert_array_equal(got, expected)


def test_get_stochastic_weight_function_non_state_dependency():
    def raw_func(health, wealth):
        pass

    raw_func._stochastic_info = StochasticInfo()  # type: ignore[attr-defined]

    variable_info = pd.DataFrame(
        {"is_discrete": [False, True]},
        index=["health", "wealth"],
    )

    with pytest.raises(ValueError, match="Stochastic variables"):
        _get_stochastic_weight_function(
            raw_func,
            name="health",
            variable_info=variable_info,
        )


def test_variable_info_with_continuous_constraint_has_unique_index():
    model = get_model_config("iskhakov_et_al_2017", n_periods=3)

    def wealth_constraint(wealth):
        return wealth > 200

    model.functions["wealth_constraint"] = wealth_constraint

    got = get_variable_info(model)
    assert got.index.is_unique
