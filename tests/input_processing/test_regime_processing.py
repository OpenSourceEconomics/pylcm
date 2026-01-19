from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid, LinspaceGrid, grid_helpers
from lcm.ages import AgeGrid
from lcm.input_processing.regime_processing import (
    _convert_flat_to_nested_transitions,
    get_grids,
    get_gridspecs,
    get_variable_info,
    process_regimes,
)
from tests.regime_mock import RegimeMock
from tests.test_models.deterministic.base import dead, working


def test_convert_flat_to_nested_transitions():
    def next_wealth():
        pass

    def next_regime():
        pass

    flat_transitions = {
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    }

    result = _convert_flat_to_nested_transitions(
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

    result = _convert_flat_to_nested_transitions(
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

    result = _convert_flat_to_nested_transitions(
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

    result = _convert_flat_to_nested_transitions(
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
        index=pd.Index(["a", "c"]),
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


def test_process_regimes():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working": working, "dead": dead}
    regime_id = MappingProxyType({name: idx for idx, name in enumerate(regimes.keys())})
    internal_regimes = process_regimes(
        regimes,
        ages=ages,
        regime_id=regime_id,
        enable_jit=True,
    )
    internal_working_regime = internal_regimes["working"]

    # Variable Info
    assert (
        internal_working_regime.variable_info["is_state"].to_numpy()
        == np.array([False, True, False])
    ).all()

    assert (
        internal_working_regime.variable_info["is_continuous"].to_numpy()
        == np.array([False, True, True])
    ).all()

    # Gridspecs
    wealth_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=working.states["wealth"].n_points,  # ty: ignore[unresolved-attribute]
    )

    assert internal_working_regime.gridspecs["wealth"] == wealth_grid

    consumption_grid = LinspaceGrid(
        start=1,
        stop=400,
        n_points=working.actions["consumption"].n_points,  # ty: ignore[unresolved-attribute]
    )
    assert internal_working_regime.gridspecs["consumption"] == consumption_grid

    assert isinstance(internal_working_regime.gridspecs["labor_supply"], DiscreteGrid)
    assert internal_working_regime.gridspecs["labor_supply"].categories == (
        "work",
        "retire",
    )
    assert internal_working_regime.gridspecs["labor_supply"].codes == (0, 1)

    # Grids
    expected = grid_helpers.linspace(**working.actions["consumption"].__dict__)
    assert_array_equal(internal_working_regime.grids["consumption"], expected)

    expected = grid_helpers.linspace(**working.states["wealth"].__dict__)
    assert_array_equal(internal_working_regime.grids["wealth"], expected)

    assert (internal_working_regime.grids["labor_supply"] == jnp.array([0, 1])).all()

    # Functions
    assert internal_working_regime.transitions is not None
    assert internal_working_regime.constraints is not None
    assert internal_working_regime.utility is not None


def test_variable_info_with_continuous_constraint_has_unique_index():
    def wealth_constraint(wealth):
        return wealth > 200

    working_copy = working.replace(
        constraints=dict(working.constraints) | {"wealth_constraint": wealth_constraint}
    )

    got = get_variable_info(working_copy)
    assert got.index.is_unique
