"""Test Regime class validation."""

import pytest

from lcm import LinSpacedGrid, Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.utils import REGIME_SEPARATOR


def utility(consumption):
    return consumption


def next_wealth(wealth, consumption):
    return wealth - consumption


WEALTH_GRID = LinSpacedGrid(start=1, stop=10, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=5, n_points=5)


def test_regime_name_does_not_contain_separator():
    """Regime name validation happens at Model level, not Regime level."""

    @categorical
    class RegimeId:
        work__test: int  # Contains separator - but RegimeId class has matching field
        dead: int

    working = Regime(
        utility=utility,
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transitions={"next_wealth": next_wealth, "next_regime": lambda: 0},
        active=lambda age: age < 5,
    )
    dead = Regime(
        utility=lambda: 0,
        terminal=True,
        active=lambda age: age >= 5,
    )
    ages = AgeGrid(start=0, stop=5, step="Y")

    # Regime name containing separator should raise at Model creation
    with pytest.raises(ModelInitializationError, match=REGIME_SEPARATOR):
        Model(
            regimes={f"work{REGIME_SEPARATOR}test": working, "dead": dead},
            ages=ages,
            regime_id_class=RegimeId,
        )


def test_function_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            utility=utility,
            states={"wealth": WEALTH_GRID},
            actions={f"consumption{REGIME_SEPARATOR}action": CONSUMPTION_GRID},
            transitions={"next_wealth": next_wealth},
            functions={f"helper{REGIME_SEPARATOR}func": lambda: 1},
            active=lambda age: age < 5,
        )


def test_state_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            utility=utility,
            states={f"my{REGIME_SEPARATOR}wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={f"next_my{REGIME_SEPARATOR}wealth": next_wealth},
            active=lambda age: age < 5,
        )


# ======================================================================================
# Terminal Regime Tests
# ======================================================================================


def test_terminal_regime_creation():
    """Terminal regime can be created with states and utility."""
    regime = Regime(
        utility=lambda wealth: wealth * 0.5,
        states={"wealth": WEALTH_GRID},
        terminal=True,
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True
    assert regime.transitions == {}


def test_terminal_regime_with_actions():
    """Terminal regime can have actions for final decisions."""
    regime = Regime(
        utility=lambda wealth, bequest_share: wealth * bequest_share,
        states={"wealth": WEALTH_GRID},
        actions={"bequest_share": LinSpacedGrid(start=0, stop=1, n_points=11)},
        terminal=True,
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True
    assert "bequest_share" in regime.actions


def test_terminal_regime_cannot_have_transitions():
    """Terminal regime cannot have transitions."""
    with pytest.raises(RegimeInitializationError, match="cannot have transitions"):
        Regime(
            utility=lambda wealth: wealth,
            states={"wealth": WEALTH_GRID},
            transitions={"next_wealth": lambda wealth: wealth},
            terminal=True,
            active=lambda age: age >= 5,
        )


def test_terminal_regime_can_be_created_without_states():
    """Terminal regime can be created without states (e.g., death state)."""
    regime = Regime(
        utility=lambda: 0,
        states={},
        terminal=True,
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True
    assert regime.states == {}


# ======================================================================================
# Active Attribute Tests
# ======================================================================================


def test_regime_with_active_callable():
    """Regime can specify active periods with a callable."""
    regime = Regime(
        utility=utility,
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transitions={"next_wealth": next_wealth},
        active=lambda age: age < 5,
    )
    assert callable(regime.active)
    assert regime.active(3) is True
    assert regime.active(5) is False


def test_active_validation_rejects_non_callable():
    """Active attribute must be a callable."""
    with pytest.raises(RegimeInitializationError, match="must be a callable"):
        Regime(
            utility=utility,
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={"next_wealth": next_wealth},
            active=[0, 1, 2],  # ty: ignore[invalid-argument-type]  # Not a callable
        )
