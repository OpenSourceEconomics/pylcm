from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid
from lcm.ages import AgeGrid
from lcm.input_processing.regime_processing import (
    process_regimes,
)
from lcm.input_processing.util import get_grids, get_gridspecs, get_variable_info
from tests.regime_mock import RegimeMock
from tests.test_models.deterministic.base import dead, working


def test_get_variable_info(binary_category_class):
    def utility(c):
        pass

    def next_c(a, b):
        pass

    regime_mock = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class, transition=next_c),
        },
        functions={"utility": utility},
    )

    got = get_variable_info(regime_mock)  # ty: ignore[invalid-argument-type]
    exp = pd.DataFrame(
        {
            "is_state": [False, True],
            "is_shock": [False, False],
            "is_action": [True, False],
            "is_continuous": [False, False],
            "is_discrete": [True, True],
            "enters_concurrent_valuation": [False, True],
            "enters_transition": [True, False],
        },
        index=pd.Index(["a", "c"]),
    )

    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_gridspecs(binary_category_class):
    def next_c(a, b):
        pass

    regime_mock = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class, transition=next_c),
        },
        functions={"utility": lambda c: None},
    )

    got = get_gridspecs(regime_mock)  # ty: ignore[invalid-argument-type]
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids(binary_category_class):
    def next_c(a, b):
        pass

    regime_mock = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class, transition=next_c),
        },
        functions={"utility": lambda c: None},
    )

    got = get_grids(regime_mock)  # ty: ignore[invalid-argument-type]
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([0, 1]))


def test_process_regimes():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working": working, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: idx for idx, name in enumerate(regimes.keys())}
    )
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
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

    # Gridspecs — compare the grid objects (which now include transition attributes)
    assert internal_working_regime.gridspecs["wealth"] == working.states["wealth"]
    assert (
        internal_working_regime.gridspecs["consumption"]
        == working.actions["consumption"]
    )

    assert isinstance(internal_working_regime.gridspecs["labor_supply"], DiscreteGrid)
    assert internal_working_regime.gridspecs["labor_supply"].categories == (
        "work",
        "retire",
    )
    assert internal_working_regime.gridspecs["labor_supply"].codes == (0, 1)

    # Grids
    assert_array_equal(
        internal_working_regime.grids["consumption"],
        working.actions["consumption"].to_jax(),
    )
    assert_array_equal(
        internal_working_regime.grids["wealth"],
        working.states["wealth"].to_jax(),
    )

    assert (internal_working_regime.grids["labor_supply"] == jnp.array([0, 1])).all()

    # Functions
    assert internal_working_regime.transitions is not None
    assert internal_working_regime.constraints is not None
    assert "utility" in internal_working_regime.functions


def test_variable_info_with_continuous_constraint_has_unique_index():
    def wealth_constraint(wealth):
        return wealth > 200

    working_copy = working.replace(
        constraints=dict(working.constraints) | {"wealth_constraint": wealth_constraint}
    )

    got = get_variable_info(working_copy)
    assert got.index.is_unique
