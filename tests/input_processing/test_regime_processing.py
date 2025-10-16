from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid, LinspaceGrid, grid_helpers
from lcm.input_processing.regime_processing import (
    _get_stochastic_weight_function,
    get_grids,
    get_gridspecs,
    get_variable_info,
    process_regime,
)
from lcm.mark import StochasticInfo
from tests.regime_mock import RegimeMock
from tests.test_models.utils import get_regime


@pytest.fixture
def regime(binary_category_class):
    def utility(c):
        pass

    def next_c(a, b):
        pass

    return RegimeMock(
        n_periods=2,
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
        utility=utility,
        transitions={"next_c": next_c},
    )


def test_get_variable_info(regime):
    got = get_variable_info(regime)
    exp = pd.DataFrame(
        {
            "is_state": [False, True],
            "is_action": [True, False],
            "is_continuous": [False, False],
            "is_discrete": [True, True],
            "is_stochastic": [False, False],
            "enters_concurrent_valuation": [False, True],
            "enters_transition": [True, False],
        },
        index=["a", "c"],
    )
    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_gridspecs(regime):
    got = get_gridspecs(regime)
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids(regime):
    got = get_grids(regime)
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([0, 1]))


def test_process_regime_iskhakov_et_al_2017():
    regime = get_regime("iskhakov_et_al_2017", n_periods=3)
    internal_regime = process_regime(regime)

    # Variable Info
    assert (
        internal_regime.variable_info["is_state"].to_numpy()
        == np.array([True, False, True, False])
    ).all()

    assert (
        internal_regime.variable_info["is_continuous"].to_numpy()
        == np.array([False, False, True, True])
    ).all()

    # Gridspecs
    wealth_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=regime.states["wealth"].n_points,  # type: ignore[attr-defined]
    )

    assert internal_regime.gridspecs["wealth"] == wealth_grid

    consumption_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=regime.actions["consumption"].n_points,  # type: ignore[attr-defined]
    )
    assert internal_regime.gridspecs["consumption"] == consumption_grid

    assert isinstance(internal_regime.gridspecs["retirement"], DiscreteGrid)
    assert internal_regime.gridspecs["retirement"].categories == ("working", "retired")
    assert internal_regime.gridspecs["retirement"].codes == (0, 1)

    assert isinstance(internal_regime.gridspecs["lagged_retirement"], DiscreteGrid)
    assert internal_regime.gridspecs["lagged_retirement"].categories == (
        "working",
        "retired",
    )
    assert internal_regime.gridspecs["lagged_retirement"].codes == (0, 1)

    # Grids
    expected = grid_helpers.linspace(**regime.actions["consumption"].__dict__)
    assert_array_equal(internal_regime.grids["consumption"], expected)

    expected = grid_helpers.linspace(**regime.states["wealth"].__dict__)
    assert_array_equal(internal_regime.grids["wealth"], expected)

    assert (internal_regime.grids["retirement"] == jnp.array([0, 1])).all()
    assert (internal_regime.grids["lagged_retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert internal_regime.transitions is not None
    assert internal_regime.constraints is not None
    assert internal_regime.utility is not None


def test_process_regime():
    regime = get_regime("iskhakov_et_al_2017_stripped_down", n_periods=3)
    internal_regime = process_regime(regime)

    # Variable Info
    assert (
        internal_regime.variable_info["is_state"].to_numpy()
        == np.array([False, True, False])
    ).all()

    assert (
        internal_regime.variable_info["is_continuous"].to_numpy()
        == np.array([False, True, True])
    ).all()

    # Gridspecs
    wealth_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=regime.states["wealth"].n_points,  # type: ignore[attr-defined]
    )

    assert internal_regime.gridspecs["wealth"] == wealth_specs

    consumption_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=regime.actions["consumption"].n_points,  # type: ignore[attr-defined]
    )
    assert internal_regime.gridspecs["consumption"] == consumption_specs

    assert isinstance(internal_regime.gridspecs["retirement"], DiscreteGrid)
    assert internal_regime.gridspecs["retirement"].categories == ("working", "retired")
    assert internal_regime.gridspecs["retirement"].codes == (0, 1)

    # Grids
    expected = grid_helpers.linspace(**regime.actions["consumption"].__dict__)
    assert_array_equal(internal_regime.grids["consumption"], expected)

    expected = grid_helpers.linspace(**regime.states["wealth"].__dict__)
    assert_array_equal(internal_regime.grids["wealth"], expected)

    assert (internal_regime.grids["retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert internal_regime.transitions is not None
    assert internal_regime.constraints is not None
    assert internal_regime.utility is not None


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
    regime = get_regime("iskhakov_et_al_2017", n_periods=3)

    def wealth_constraint(wealth):
        return wealth > 200

    regime.constraints["wealth_constraint"] = wealth_constraint

    got = get_variable_info(regime)
    assert got.index.is_unique
