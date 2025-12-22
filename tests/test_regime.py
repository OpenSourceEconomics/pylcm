"""Test Regime class validation."""

import pytest

from lcm import LinspaceGrid, Regime
from lcm.exceptions import RegimeInitializationError
from lcm.utils import REGIME_SEPARATOR


def utility(consumption):
    return consumption


def next_wealth(wealth, consumption):
    return wealth - consumption


WEALTH_GRID = LinspaceGrid(start=1, stop=10, n_points=5)
CONSUMPTION_GRID = LinspaceGrid(start=1, stop=5, n_points=5)


def test_regime_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            name=f"work{REGIME_SEPARATOR}test",
            utility=utility,
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={"next_wealth": next_wealth},
            active=range(5),
        )


def test_function_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            name="work",
            utility=utility,
            states={"wealth": WEALTH_GRID},
            actions={f"consumption{REGIME_SEPARATOR}action": CONSUMPTION_GRID},
            transitions={"next_wealth": next_wealth},
            functions={f"helper{REGIME_SEPARATOR}func": lambda: 1},
            active=range(5),
        )


def test_state_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            name="work",
            utility=utility,
            states={f"my{REGIME_SEPARATOR}wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={f"next_my{REGIME_SEPARATOR}wealth": next_wealth},
            active=range(5),
        )


# ======================================================================================
# Terminal Regime Tests
# ======================================================================================


def test_terminal_regime_creation():
    """Terminal regime can be created with states and utility."""
    regime = Regime(
        name="dead",
        utility=lambda wealth: wealth * 0.5,
        states={"wealth": WEALTH_GRID},
        terminal=True,
        active=[5],
    )
    assert regime.terminal is True
    assert regime.transitions == {}


def test_terminal_regime_with_actions():
    """Terminal regime can have actions for final decisions."""
    regime = Regime(
        name="dead",
        utility=lambda wealth, bequest_share: wealth * bequest_share,
        states={"wealth": WEALTH_GRID},
        actions={"bequest_share": LinspaceGrid(start=0, stop=1, n_points=11)},
        terminal=True,
        active=[5],
    )
    assert regime.terminal is True
    assert "bequest_share" in regime.actions


def test_terminal_regime_cannot_have_transitions():
    """Terminal regime cannot have transitions."""
    with pytest.raises(RegimeInitializationError, match="cannot have transitions"):
        Regime(
            name="dead",
            utility=lambda wealth: wealth,
            states={"wealth": WEALTH_GRID},
            transitions={"next_wealth": lambda wealth: wealth},
            terminal=True,
            active=[5],
        )


@pytest.mark.xfail(reason="Stateless regimes are not yet supported.")
def test_terminal_regime_can_be_created_without_states():
    """Terminal regime can be created without states (e.g., absorbing death state)."""
    regime = Regime(
        name="dead",
        utility=lambda: 0,
        states={},
        terminal=True,
        active=[5],
    )
    assert regime.terminal is True
    assert regime.states == {}


# ======================================================================================
# Active Attribute Tests
# ======================================================================================


def test_regime_with_active_periods():
    """Regime can specify active periods with various iterable types."""
    # Tuple
    regime = Regime(
        name="work",
        utility=utility,
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transitions={"next_wealth": next_wealth},
        active=(0, 1, 2),
    )
    assert regime.active is not None
    assert list(regime.active) == [0, 1, 2]

    # Range
    regime2 = Regime(
        name="work",
        utility=utility,
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transitions={"next_wealth": next_wealth},
        active=range(5),
    )
    assert regime2.active is not None
    assert list(regime2.active) == [0, 1, 2, 3, 4]


def test_active_validation_rejects_invalid_values():
    """Active attribute must be iterable of non-negative unique integers."""
    # Empty list
    with pytest.raises(RegimeInitializationError, match="cannot be empty"):
        Regime(
            name="work",
            utility=utility,
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={"next_wealth": next_wealth},
            active=[],
        )

    # Negative periods
    with pytest.raises(RegimeInitializationError, match="cannot be negative"):
        Regime(
            name="dead",
            utility=lambda wealth: wealth,
            states={"wealth": WEALTH_GRID},
            terminal=True,
            active=[-1, 0, 1],
        )
