"""Test Regime class validation."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from lcm import LinspaceGrid, Model, Regime
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


# ======================================================================================
# State-less Terminal Regime Integration Tests
# ======================================================================================


def test_stateless_terminal_regime_solve_and_simulate():
    """Test that state-less terminal regimes can be solved and simulated."""

    @dataclass
    class RegimeId:
        working: int = 0
        dead: int = 1

    def utility_working(consumption):
        return jnp.log(consumption)

    def next_wealth(wealth, consumption, interest_rate):
        return (1 + interest_rate) * (wealth - consumption)

    def next_regime_from_working(period, n_periods):
        return jnp.where(period == n_periods - 2, RegimeId.dead, RegimeId.working)

    def borrowing_constraint(consumption, wealth):
        return consumption <= wealth

    working = Regime(
        name="working",
        actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
        states={"wealth": LinspaceGrid(start=1, stop=100, n_points=20)},
        utility=utility_working,
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_regime": next_regime_from_working,
        },
        active=[0, 1],
    )

    # State-less terminal regime with constant utility
    dead = Regime(
        name="dead",
        terminal=True,
        utility=lambda: 0.0,
        active=[2],
    )

    model = Model(
        [working, dead],
        n_periods=3,
        regime_id_cls=RegimeId,
    )

    params = {
        "working": {
            "beta": 0.95,
            "utility": {},
            "next_wealth": {"interest_rate": 0.05},
            "next_regime": {"n_periods": 3},
            "borrowing_constraint": {},
        },
        "dead": {},
    }

    # Test solve
    V_arr_dict = model.solve(params, debug_mode=False)
    assert 2 in V_arr_dict
    assert "dead" in V_arr_dict[2]
    # V_arr for state-less regime should be a scalar
    assert V_arr_dict[2]["dead"].ndim == 0
    assert float(V_arr_dict[2]["dead"]) == 0.0

    # Test simulate
    initial_states = {"wealth": jnp.array([50.0, 80.0])}
    initial_regimes = ["working", "working"]

    result = model.simulate(
        params=params,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        V_arr_dict=V_arr_dict,
        debug_mode=False,
    )

    # Check that dead regime results are correct
    assert "dead" in result
    dead_df = result["dead"]
    assert len(dead_df) == 2  # Both subjects end up in dead regime
    assert all(dead_df["value"] == 0.0)
    # State-less regime should have no state columns except period, subject_id, value
    assert set(dead_df.columns) == {"period", "subject_id", "value"}


def test_terminal_regime_with_states_and_scalar_utility():
    """Test that terminal regimes with states can return scalar utility."""

    @dataclass
    class RegimeId:
        working: int = 0
        dead: int = 1

    def utility_working(consumption):
        return jnp.log(consumption)

    def next_wealth(wealth, consumption, interest_rate):
        return (1 + interest_rate) * (wealth - consumption)

    def next_regime_from_working(period, n_periods):
        return jnp.where(period == n_periods - 2, RegimeId.dead, RegimeId.working)

    def borrowing_constraint(consumption, wealth):
        return consumption <= wealth

    working = Regime(
        name="working",
        actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
        states={"wealth": LinspaceGrid(start=1, stop=100, n_points=20)},
        utility=utility_working,
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_regime": next_regime_from_working,
        },
        active=[0, 1],
    )

    # Terminal regime WITH states but utility returns scalar (broadcast)
    dead = Regime(
        name="dead",
        terminal=True,
        utility=lambda wealth: 0.0,  # noqa: ARG005
        states={"wealth": LinspaceGrid(start=1, stop=100, n_points=20)},
        active=[2],
    )

    model = Model(
        [working, dead],
        n_periods=3,
        regime_id_cls=RegimeId,
    )

    params = {
        "working": {
            "beta": 0.95,
            "utility": {},
            "next_wealth": {"interest_rate": 0.05},
            "next_regime": {"n_periods": 3},
            "borrowing_constraint": {},
        },
        "dead": {},
    }

    # Test solve - V_arr should have shape matching state grid
    V_arr_dict = model.solve(params, debug_mode=False)
    assert V_arr_dict[2]["dead"].shape == (20,)
    assert all(V_arr_dict[2]["dead"] == 0.0)
