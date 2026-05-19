"""Tests for `Variables` — per-regime states + actions with named accessors."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lcm._grids import DiscreteGrid, LinSpacedGrid
from lcm.engine import VariableInfo, Variables
from lcm.variables import from_regime
from tests.mock_regime import MockRegime


@pytest.fixture(name="variables")
def _variables() -> Variables:
    """Variables covering every kind/topology combination, including one shock."""
    return Variables(
        info=MappingProxyType(
            {
                "edu": VariableInfo(kind="state", topology="discrete", is_shock=False),
                "wealth": VariableInfo(
                    kind="state", topology="continuous", is_shock=False
                ),
                "wage_shock": VariableInfo(
                    kind="state", topology="discrete", is_shock=True
                ),
                "labor_supply": VariableInfo(
                    kind="action", topology="discrete", is_shock=False
                ),
                "consumption": VariableInfo(
                    kind="action", topology="continuous", is_shock=False
                ),
            }
        )
    )


def test_state_names_filters_kind(variables: Variables) -> None:
    """`state_names` returns every variable with kind='state', in iteration order."""
    assert variables.state_names == ("edu", "wealth", "wage_shock")


def test_action_names_filters_kind(variables: Variables) -> None:
    """`action_names` returns every variable with kind='action', in iteration order."""
    assert variables.action_names == ("labor_supply", "consumption")


def test_discrete_state_names_filters_kind_and_topology(
    variables: Variables,
) -> None:
    """`discrete_state_names` is states with topology='discrete' (includes shocks)."""
    assert variables.discrete_state_names == ("edu", "wage_shock")


def test_continuous_state_names_filters_kind_and_topology(
    variables: Variables,
) -> None:
    """`continuous_state_names` is states with topology='continuous'."""
    assert variables.continuous_state_names == ("wealth",)


def test_discrete_action_names_filters_kind_and_topology(
    variables: Variables,
) -> None:
    """`discrete_action_names` is actions with topology='discrete'."""
    assert variables.discrete_action_names == ("labor_supply",)


def test_continuous_action_names_filters_kind_and_topology(
    variables: Variables,
) -> None:
    """`continuous_action_names` is actions with topology='continuous'."""
    assert variables.continuous_action_names == ("consumption",)


def test_state_and_discrete_action_names_is_gridded_set(
    variables: Variables,
) -> None:
    """`state_and_discrete_action_names` is every state plus every discrete action."""
    assert variables.state_and_discrete_action_names == (
        "edu",
        "wealth",
        "wage_shock",
        "labor_supply",
    )


def test_shock_names_filters_is_shock(variables: Variables) -> None:
    """`shock_names` is every variable with `is_shock=True`."""
    assert variables.shock_names == ("wage_shock",)


def test_getitem_returns_variable_info(variables: Variables) -> None:
    """`vars[name]` returns the underlying `VariableInfo` record."""
    assert variables["wealth"] == VariableInfo(
        kind="state", topology="continuous", is_shock=False
    )


def test_iter_yields_names_in_definition_order(variables: Variables) -> None:
    """Iteration yields variable names in the order the info dict supplies."""
    assert list(variables) == [
        "edu",
        "wealth",
        "wage_shock",
        "labor_supply",
        "consumption",
    ]


def test_len_equals_variable_count(variables: Variables) -> None:
    """`len(vars)` equals the number of variables."""
    assert len(variables) == 5


def test_contains_returns_true_for_known_variable(variables: Variables) -> None:
    """`name in vars` works through the Mapping interface."""
    assert "edu" in variables


def test_is_mapping_instance(variables: Variables) -> None:
    """`Variables` registers as a `collections.abc.Mapping`."""
    assert isinstance(variables, Mapping)


def test_items_yields_name_info_pairs(variables: Variables) -> None:
    """`vars.items()` yields (name, VariableInfo) pairs."""
    items = list(variables.items())
    assert items[0] == (
        "edu",
        VariableInfo(kind="state", topology="discrete", is_shock=False),
    )


def test_frozen_dataclass_rejects_field_assignment(
    variables: Variables,
) -> None:
    """`Variables` is immutable — assigning to a field raises `FrozenInstanceError`."""
    with pytest.raises(FrozenInstanceError):
        variables.state_names = ()  # ty: ignore[invalid-assignment]


def test_from_regime_orders_discrete_states_continuous_states_actions(
    binary_category_class,
) -> None:
    """`Variables.from_regime` orders discrete states → continuous states → actions."""

    def next_state(x):
        return x

    regime = MockRegime(
        states={
            "a_discrete": DiscreteGrid(binary_category_class),
            "b_continuous": LinSpacedGrid(start=0, stop=1, n_points=5),
        },
        state_transitions={
            "a_discrete": next_state,
            "b_continuous": next_state,
        },
        actions={
            "c_action": DiscreteGrid(binary_category_class),
        },
        functions={"utility": lambda c_action: 0},  # noqa: ARG005
    )
    variables = from_regime(regime)
    assert list(variables) == ["a_discrete", "b_continuous", "c_action"]


def test_from_regime_within_states_orders_by_batch_size(
    binary_category_class,
) -> None:
    """Within a topology group, states are ordered by batch_size (0 sorts last)."""

    def next_state(x):
        return x

    regime = MockRegime(
        states={
            "third": DiscreteGrid(binary_category_class),
            "first": DiscreteGrid(binary_category_class, batch_size=1),
            "second": DiscreteGrid(binary_category_class, batch_size=2),
        },
        state_transitions={
            "first": next_state,
            "second": next_state,
            "third": next_state,
        },
        actions={"a": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda a: 0},  # noqa: ARG005
    )
    variables = from_regime(regime)
    assert variables.discrete_state_names == ("first", "second", "third")
