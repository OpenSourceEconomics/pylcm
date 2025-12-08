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
        )


def test_state_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=REGIME_SEPARATOR):
        Regime(
            name="work",
            utility=utility,
            states={f"my{REGIME_SEPARATOR}wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transitions={f"next_my{REGIME_SEPARATOR}wealth": next_wealth},
        )
