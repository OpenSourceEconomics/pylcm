import functools
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm import categorical
from lcm.ages import AgeGrid
from lcm.engine import Regime, VariableInfo, Variables
from lcm.grids import DiscreteGrid, LinSpacedGrid
from lcm.regime_building.processing import (
    _rename_params_to_qnames,
    process_regimes,
)
from lcm.typing import ScalarInt
from lcm.user_regime import Regime as UserRegime
from lcm.variables import from_regime, get_grids
from tests.mock_regime import MockRegime
from tests.test_models.deterministic.base import dead, working_life


def test_variables_from_regime_tags_kind_and_topology(binary_category_class):
    def utility(c):
        pass

    def next_c(a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
        state_transitions={"c": next_c},
        functions={"utility": utility},
    )

    got = from_regime(mock_regime)

    assert isinstance(got, Variables)
    assert set(got) == {"a", "c"}
    assert got["a"] == VariableInfo(kind="action", topology="discrete", is_shock=False)
    assert got["c"] == VariableInfo(kind="state", topology="discrete", is_shock=False)


def test_get_grids(binary_category_class):
    def next_c(a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "c": DiscreteGrid(binary_category_class),
        },
        state_transitions={"c": next_c},
        functions={"utility": lambda _c: None},
    )

    got = get_grids(mock_regime)
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids_reorder(binary_category_class):
    def next_state(a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "b": DiscreteGrid(binary_category_class),
            "c": DiscreteGrid(binary_category_class, batch_size=1),
            "d": LinSpacedGrid(start=0, stop=1, n_points=5, batch_size=3),
            "e": LinSpacedGrid(start=0, stop=1, n_points=5, batch_size=1),
            "f": LinSpacedGrid(start=0, stop=1, n_points=5),
        },
        state_transitions={
            "b": next_state,
            "c": next_state,
            "d": next_state,
            "e": next_state,
            "f": next_state,
        },
        functions={"utility": lambda _c: None},
    )

    got = get_grids(mock_regime)
    assert list(got.keys()) == ["c", "b", "e", "d", "f", "a"]


def test_process_regimes():
    ages = AgeGrid(start=0, stop=4, step="Y")
    user_regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(idx) for idx, name in enumerate(user_regimes.keys())}
    )
    regimes = process_regimes(
        user_regimes=user_regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
    )
    working_regime = regimes["working_life"]

    # Variable Info
    variables = working_regime.variables
    assert variables["wealth"] == VariableInfo(
        kind="state", topology="continuous", is_shock=False
    )
    assert variables["labor_supply"] == VariableInfo(
        kind="action", topology="discrete", is_shock=False
    )
    assert variables["consumption"] == VariableInfo(
        kind="action", topology="continuous", is_shock=False
    )

    # Grids — compare the grid objects (which now include transition attributes)
    assert working_regime.grids["wealth"] == working_life.states["wealth"]
    assert working_regime.grids["consumption"] == working_life.actions["consumption"]

    assert isinstance(working_regime.grids["labor_supply"], DiscreteGrid)
    assert working_regime.grids["labor_supply"].categories == (
        "work",
        "retire",
    )
    assert working_regime.grids["labor_supply"].codes == (0, 1)

    # Materialized grids
    assert_array_equal(
        working_regime.grids["consumption"].to_jax(),
        working_life.actions["consumption"].to_jax(),
    )
    assert_array_equal(
        working_regime.grids["wealth"].to_jax(),
        working_life.states["wealth"].to_jax(),
    )

    assert (working_regime.grids["labor_supply"].to_jax() == jnp.array([0, 1])).all()

    # Functions
    assert working_regime.solve_functions.transitions is not None
    assert working_regime.solve_functions.constraints is not None
    assert "utility" in working_regime.solve_functions.functions


def test_variables_excludes_constraint_names():
    """Constraint functions do not appear as variables in the Variables container."""

    def wealth_constraint(wealth):
        return wealth > 200

    working_copy = working_life.replace(
        constraints=dict(working_life.constraints)
        | {"wealth_constraint": wealth_constraint}
    )

    got = from_regime(working_copy)
    assert "wealth_constraint" not in got


@pytest.fixture(name="two_non_terminal_regimes")
def _two_non_terminal_regimes() -> MappingProxyType[str, Regime]:
    """Two non-terminal regimes that share underlying user functions."""

    def next_x(x):
        return x

    def regime_transition(age, final_age):
        return jnp.where(age >= final_age, 1, 0)

    @categorical(ordered=False)
    class TwoRegimeId:
        early: ScalarInt
        late: ScalarInt

    early = UserRegime(
        transition=regime_transition,
        states={"x": LinSpacedGrid(start=0, stop=10, n_points=4)},
        state_transitions={"x": next_x},
        functions={"utility": lambda x: x},
        active=lambda age: age < 1,
    )
    late = UserRegime(
        transition=regime_transition,
        states={"x": LinSpacedGrid(start=0, stop=10, n_points=6)},
        state_transitions={"x": next_x},
        functions={"utility": lambda x: x},
        active=lambda age: age >= 1,
    )
    return process_regimes(
        user_regimes={"early": early, "late": late},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_names_to_ids=MappingProxyType(
            {"early": jnp.int32(0), "late": jnp.int32(1)}
        ),
        enable_jit=True,
    )


@pytest.mark.parametrize(
    "attr",
    ["next_state", "compute_regime_transition_probs"],
)
def test_simulate_functions_use_per_regime_callables(
    two_non_terminal_regimes: MappingProxyType[str, Regime],
    attr: str,
) -> None:
    """Two regimes built from shared user functions get distinct simulate callables."""
    early_func = getattr(two_non_terminal_regimes["early"].simulate_functions, attr)
    late_func = getattr(two_non_terminal_regimes["late"].simulate_functions, attr)
    assert id(early_func) != id(late_func)


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
