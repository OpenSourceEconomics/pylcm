from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid, LinspaceGrid, grid_helpers
from lcm.input_processing.regime_processing import (
    convert_flat_to_nested_transitions,
    create_default_regime_id_cls,
    get_grids,
    get_gridspecs,
    get_variable_info,
    process_regimes,
)
from tests.regime_mock import RegimeMock
from tests.test_models.utils import get_regime

# ======================================================================================
# Tests for convert_flat_to_nested_transitions
# ======================================================================================


def test_convert_flat_to_nested_transitions():
    def next_wealth():
        pass

    def next_regime():
        pass

    flat_transitions = {
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    }

    result = convert_flat_to_nested_transitions(
        flat_transitions=flat_transitions,
        states_per_regime={"alive": {"wealth"}},
    )

    expected = {
        "alive": {
            "next_wealth": next_wealth,
        },
        "next_regime": next_regime,
    }

    assert result == expected


def test_convert_flat_to_nested_only_next_regime():
    """Regime with only next_regime (like dead regime with no state transitions)."""

    def next_regime():
        pass

    flat_transitions = {
        "next_regime": next_regime,
    }

    result = convert_flat_to_nested_transitions(
        flat_transitions=flat_transitions,
        states_per_regime={"dead": set()},
    )

    # Empty regimes (no states) get complete transitions by definition
    expected = {
        "dead": {},
        "next_regime": next_regime,
    }

    assert result == expected


def test_convert_flat_to_nested_multi_regime():
    def next_wealth():
        pass

    def next_education():
        pass

    def next_pension():
        pass

    def next_regime():
        pass

    flat_transitions = {
        "next_wealth": next_wealth,
        "next_education": next_education,
        "next_pension": next_pension,
        "next_regime": next_regime,
    }

    result = convert_flat_to_nested_transitions(
        flat_transitions=flat_transitions,
        states_per_regime={
            "young": {"wealth", "education"},
            "old": {"wealth", "pension"},
        },
    )

    # Both regimes have complete transitions
    expected = {
        "young": {
            "next_wealth": next_wealth,
            "next_education": next_education,
        },
        "old": {
            "next_wealth": next_wealth,
            "next_pension": next_pension,
        },
        "next_regime": next_regime,
    }

    assert result == expected


def test_convert_flat_to_nested_absorbing_multi_regime():
    def next_wealth():
        pass

    def next_regime():
        pass

    flat_transitions_dead = {
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    }

    result = convert_flat_to_nested_transitions(
        flat_transitions=flat_transitions_dead,
        states_per_regime={
            "alive": {"wealth", "retired"},
            "dead": {"wealth"},
        },
    )

    # Since "next_retired" is missing, "alive" is not included in the nested structure
    expected = {
        "dead": {
            "next_wealth": next_wealth,
        },
        "next_regime": next_regime,
    }

    assert result == expected


@pytest.fixture
def regime_mock(binary_category_class):
    def utility(c):
        pass

    def next_c(a, b):
        pass

    return RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
        utility=utility,
        transitions={"next_c": next_c},
    )


def test_get_variable_info(regime_mock):
    got = get_variable_info(regime_mock)
    exp = pd.DataFrame(
        {
            "is_state": [False, True],
            "is_action": [True, False],
            "is_continuous": [False, False],
            "is_discrete": [True, True],
            "enters_concurrent_valuation": [False, True],
            "enters_transition": [True, False],
        },
        index=["a", "c"],
    )

    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_gridspecs(regime_mock):
    got = get_gridspecs(regime_mock)
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids(regime_mock):
    got = get_grids(regime_mock)
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([0, 1]))


def test_process_regime_iskhakov_et_al_2017():
    regime = get_regime("iskhakov_et_al_2017")
    internal_regime = process_regimes(
        [regime],
        n_periods=3,
        regime_id_cls=create_default_regime_id_cls(regime.name),
        enable_jit=True,
    )[regime.name]

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
    regime = get_regime("iskhakov_et_al_2017_stripped_down")
    internal_regime = process_regimes(
        [regime],
        n_periods=3,
        regime_id_cls=create_default_regime_id_cls(regime.name),
        enable_jit=True,
    )[regime.name]

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


def test_variable_info_with_continuous_constraint_has_unique_index():
    regime = get_regime("iskhakov_et_al_2017")

    def wealth_constraint(wealth):
        return wealth > 200

    regime.constraints["wealth_constraint"] = wealth_constraint

    got = get_variable_info(regime)
    assert got.index.is_unique
