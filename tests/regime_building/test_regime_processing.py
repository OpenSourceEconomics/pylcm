import functools
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import DiscreteGrid
from lcm.ages import AgeGrid
from lcm.regime_building.processing import (
    _rename_params_to_qnames,
    process_regimes,
)
from lcm.regime_building.variable_info import get_grids, get_variable_info
from tests.regime_mock import RegimeMock
from tests.test_models.deterministic.base import dead, working_life


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
            "c": DiscreteGrid(binary_category_class),
        },
        state_transitions={"c": next_c},
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
        },
        index=pd.Index(["a", "c"]),
    )

    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_grids(binary_category_class):
    def next_c(a, b):
        pass

    regime_mock = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
        state_transitions={"c": next_c},
        functions={"utility": lambda _c: None},
    )

    got = get_grids(regime_mock)  # ty: ignore[invalid-argument-type]
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_process_regimes():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: idx for idx, name in enumerate(regimes.keys())}
    )
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
    )
    internal_working_regime = internal_regimes["working_life"]

    # Variable Info
    assert (
        internal_working_regime.variable_info["is_state"].to_numpy()
        == np.array([True, False, False])
    ).all()

    assert (
        internal_working_regime.variable_info["is_continuous"].to_numpy()
        == np.array([True, False, True])
    ).all()

    # Grids — compare the grid objects (which now include transition attributes)
    assert internal_working_regime.grids["wealth"] == working_life.states["wealth"]
    assert (
        internal_working_regime.grids["consumption"]
        == working_life.actions["consumption"]
    )

    assert isinstance(internal_working_regime.grids["labor_supply"], DiscreteGrid)
    assert internal_working_regime.grids["labor_supply"].categories == (
        "work",
        "retire",
    )
    assert internal_working_regime.grids["labor_supply"].codes == (0, 1)

    # Materialized grids
    assert_array_equal(
        internal_working_regime.grids["consumption"].to_jax(),
        working_life.actions["consumption"].to_jax(),
    )
    assert_array_equal(
        internal_working_regime.grids["wealth"].to_jax(),
        working_life.states["wealth"].to_jax(),
    )

    assert (
        internal_working_regime.grids["labor_supply"].to_jax() == jnp.array([0, 1])
    ).all()

    # Functions
    assert internal_working_regime.solve_functions.transitions is not None
    assert internal_working_regime.solve_functions.constraints is not None
    assert "utility" in internal_working_regime.solve_functions.functions


def test_variable_info_with_continuous_constraint_has_unique_index():
    def wealth_constraint(wealth):
        return wealth > 200

    working_copy = working_life.replace(
        constraints=dict(working_life.constraints)
        | {"wealth_constraint": wealth_constraint}
    )

    got = get_variable_info(working_copy)
    assert got.index.is_unique


def test_rename_params_to_qnames_with_partial():
    """Regression: dags >=0.5.1 renames bound partial keywords to qualified names."""

    def utility(consumption, risk_aversion):
        return consumption ** (1 - risk_aversion)

    func = functools.partial(utility, risk_aversion=2.0)
    regime_params_template = MappingProxyType(
        {"utility": MappingProxyType({"risk_aversion": "float"})}
    )

    result = _rename_params_to_qnames(
        func=func,
        regime_params_template=regime_params_template,
        param_key="utility",
    )
    # 1. The bound default must work under the qualified name. Before dags >=0.5.1,
    #    the manual unwrap/re-wrap rebound keywords under the old name, so this call
    #    would raise TypeError.
    assert result(consumption=5.0) == 5.0 ** (1 - 2.0)

    # 2. The qualified name must be usable to override the default. This fails if
    #    _rename_params_to_qnames is a no-op (no renaming happened).
    assert result(consumption=5.0, utility__risk_aversion=3.0) == 5.0 ** (1 - 3.0)
