"""Tests for initial states conversion and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid, Model, Regime
from lcm.exceptions import InvalidInitialStatesError
from lcm.simulation.util import (
    convert_flat_to_nested_initial_states,
    validate_flat_initial_states,
)

if TYPE_CHECKING:
    from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt


@pytest.fixture
def model() -> Model:
    """Minimal model with two states (wealth, health) for initial states tests."""

    @dataclass
    class HealthStatus:
        healthy: int = 0
        sick: int = 1

    @dataclass
    class RegimeId:
        active: int = 0
        terminal: int = 1

    def utility(wealth: ContinuousState, health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_regime(period: int) -> ScalarInt:
        return jnp.where(
            period + 1 >= 2,
            RegimeId.terminal,
            RegimeId.active,
        )

    n_periods = 2

    alive = Regime(
        name="active",
        utility=utility,
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(HealthStatus),
        },
        transitions={
            "next_wealth": lambda wealth: wealth,
            "next_health": lambda health: health,
            "next_regime": next_regime,
        },
        active=range(n_periods - 1),
    )

    dead = Regime(
        name="terminal",
        terminal=True,
        utility=lambda: 0.0,
        active=[n_periods - 1],
    )

    return Model(
        [alive, dead],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
    )


# ==============================================================================
# Tests
# ==============================================================================
def test_convert_flat_to_nested_single_regime(model: Model) -> None:
    """Single regime gets its states from flat dict."""
    flat = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    nested = convert_flat_to_nested_initial_states(flat, model.internal_regimes)

    assert set(nested) == {"active", "terminal"}
    assert "wealth" in nested["active"]
    assert "health" in nested["active"]


def test_validate_flat_initial_states_valid_input(model: Model) -> None:
    """Valid input should not raise."""
    flat = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_missing_state(model: Model) -> None:
    """Missing state should raise InvalidInitialStatesError."""
    flat = {"wealth": jnp.array([10.0, 50.0])}

    with pytest.raises(
        InvalidInitialStatesError, match=r"Missing initial states: \['health'\].*"
    ):
        validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_extra_state(model: Model) -> None:
    """Extra state should raise InvalidInitialStatesError."""
    flat = {
        "wealth": jnp.array([10.0]),
        "health": jnp.array([0]),
        "unknown": jnp.array([1.0]),
    }

    with pytest.raises(InvalidInitialStatesError, match="Unknown initial states"):
        validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_inconsistent_lengths(model: Model) -> None:
    """Arrays with different lengths should raise InvalidInitialStatesError."""
    flat = {
        "wealth": jnp.array([10.0, 20.0]),
        "health": jnp.array([0]),
    }

    with pytest.raises(InvalidInitialStatesError, match="same length"):
        validate_flat_initial_states(flat, model.internal_regimes)
