"""Test Regime class validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lcm import LinspaceGrid, Regime
from lcm.exceptions import RegimeInitializationError
from lcm.utils import REGIME_SEPARATOR


@dataclass
class BinaryState:
    off: int = 0
    on: int = 1


def simple_utility(consumption):
    return consumption


def simple_next_wealth(wealth, consumption):
    return wealth - consumption


class TestRegimeSeparatorValidation:
    """Test that regime names and function names cannot contain the separator."""

    def test_regime_name_with_separator_raises_error(self):
        """Regime name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name=f"work{REGIME_SEPARATOR}test",  # Invalid name
                utility=simple_utility,
                states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
                actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
                transitions={"next_wealth": simple_next_wealth},
            )

    def test_transition_name_with_separator_raises_error(self):
        """Transition name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name="work",
                utility=simple_utility,
                states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
                actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
                transitions={
                    f"next{REGIME_SEPARATOR}wealth": simple_next_wealth,  # Invalid
                },
            )

    def test_constraint_name_with_separator_raises_error(self):
        """Constraint name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name="work",
                utility=simple_utility,
                states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
                actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
                transitions={"next_wealth": simple_next_wealth},
                constraints={
                    f"budget{REGIME_SEPARATOR}constraint": lambda c, w: c <= w,
                },
            )

    def test_function_name_with_separator_raises_error(self):
        """Function name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name="work",
                utility=simple_utility,
                states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
                actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
                transitions={"next_wealth": simple_next_wealth},
                functions={
                    f"helper{REGIME_SEPARATOR}func": lambda: 1,  # Invalid
                },
            )

    def test_state_name_with_separator_raises_error(self):
        """State name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name="work",
                utility=simple_utility,
                states={
                    f"my{REGIME_SEPARATOR}wealth": LinspaceGrid(
                        start=1, stop=10, n_points=5
                    ),
                },
                actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
                transitions={f"next_my{REGIME_SEPARATOR}wealth": simple_next_wealth},
            )

    def test_action_name_with_separator_raises_error(self):
        """Action name containing the separator should raise an error."""
        with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
            Regime(
                name="work",
                utility=simple_utility,
                states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
                actions={
                    f"my{REGIME_SEPARATOR}consumption": LinspaceGrid(
                        start=1, stop=5, n_points=5
                    ),
                },
                transitions={"next_wealth": simple_next_wealth},
            )

    def test_valid_names_work(self):
        """Names without the separator should work fine."""
        # This should not raise
        regime = Regime(
            name="work_regime",  # Single underscore is fine
            utility=simple_utility,
            states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
            actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
            transitions={"next_wealth": simple_next_wealth},
            constraints={"budget_constraint": lambda c, w: c <= w},
            functions={"helper_func": lambda: 1},
        )
        assert regime.name == "work_regime"
