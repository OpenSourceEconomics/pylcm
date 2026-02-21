"""Tests for initial states conversion and validation utilities."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidInitialStatesError
from lcm.simulation.util import (
    convert_initial_states_to_nested,
    validate_initial_states,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


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
    ages = AgeGrid(start=0, stop=n_periods, step="Y")

    alive = Regime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=lambda wealth: wealth
            ),
            "health": DiscreteGrid(
                category_class=HealthStatus, transition=lambda health: health
            ),
        },
        transition=next_regime,
        active=lambda age: age < n_periods - 1,
    )

    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= n_periods - 1,
    )

    return Model(
        regimes={"active": alive, "terminal": dead},
        ages=ages,
        regime_id_class=RegimeId,
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
    nested = convert_initial_states_to_nested(
        initial_states=flat, internal_regimes=model.internal_regimes
    )

    assert set(nested) == {"active", "terminal"}
    assert "wealth" in nested["active"]
    assert "health" in nested["active"]


def test_validate_initial_states_valid_input(model: Model) -> None:
    """Valid input should not raise."""
    flat = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    validate_initial_states(
        initial_states=flat, internal_regimes=model.internal_regimes
    )


def test_validate_initial_states_missing_state(model: Model) -> None:
    """Missing state should raise InvalidInitialStatesError."""
    flat = {"wealth": jnp.array([10.0, 50.0])}

    with pytest.raises(
        InvalidInitialStatesError, match=r"Missing initial states: \['health'\].*"
    ):
        validate_initial_states(
            initial_states=flat, internal_regimes=model.internal_regimes
        )


def test_validate_initial_states_extra_state(model: Model) -> None:
    """Extra state should raise InvalidInitialStatesError."""
    flat = {
        "wealth": jnp.array([10.0]),
        "health": jnp.array([0]),
        "unknown": jnp.array([1.0]),
    }

    with pytest.raises(InvalidInitialStatesError, match="Unknown initial states"):
        validate_initial_states(
            initial_states=flat, internal_regimes=model.internal_regimes
        )


def test_validate_initial_states_inconsistent_lengths(model: Model) -> None:
    """Arrays with different lengths should raise InvalidInitialStatesError."""
    flat = {
        "wealth": jnp.array([10.0, 20.0]),
        "health": jnp.array([0]),
    }

    with pytest.raises(InvalidInitialStatesError, match="same length"):
        validate_initial_states(
            initial_states=flat, internal_regimes=model.internal_regimes
        )


# ==============================================================================
# Reproducer for GitHub issue #64
# ==============================================================================

# The key distinction is between constraint feasibility and grid bounds:
#
#   consumption grid start = 0.5  (= effective constraint threshold on wealth)
#   wealth grid start = 2.0       (= grid minimum, extrapolation below is fine)
#
#   v_0 = 0.25 < 0.5 = constraint < v_1 = 1.0 < 2.0 = grid_min < v_2 = 5.0

_FINAL_AGE_64 = 1


@categorical
class _RegimeId64:
    working: int
    dead: int


def _utility_64(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _next_wealth_64(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def _borrowing_constraint_64(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def _next_regime_64(age: float, final_age_alive: float) -> ScalarInt:
    dead = _RegimeId64.dead
    working = _RegimeId64.working
    return jnp.where(age >= final_age_alive, dead, working)


@pytest.fixture
def constraint_model():
    """Model where constraint threshold (0.5) < grid minimum (2.0)."""
    working_regime = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.5, stop=10, n_points=20),
        },
        states={
            "wealth": LinSpacedGrid(
                start=2.0, stop=10, n_points=15, transition=_next_wealth_64
            ),
        },
        constraints={"borrowing_constraint": _borrowing_constraint_64},
        transition=_next_regime_64,
        functions={"utility": _utility_64},
        active=lambda age: age <= _FINAL_AGE_64,
    )
    dead_regime = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > _FINAL_AGE_64,
    )
    model = Model(
        regimes={"working": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=_FINAL_AGE_64 + 1, step="Y"),
        regime_id_class=_RegimeId64,
    )
    params = {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": _FINAL_AGE_64}},
    }
    return model, params


@pytest.mark.xfail(reason="Issue #64: infeasible initial states not checked")
def test_infeasible_initial_states_detected(constraint_model):
    """Issue #64: wealth below constraint threshold makes all actions infeasible.

    wealth=0.25 < min consumption (0.5), so consumption <= wealth is always False.
    """
    model, params = constraint_model
    with pytest.raises(InvalidInitialStatesError):
        model.solve_and_simulate(
            params=params,
            initial_states={"wealth": jnp.array([0.25])},
            initial_regimes=["working"],
        )


def test_extrapolated_initial_states_accepted(constraint_model):
    """wealth=1.0 is above constraint threshold but below grid min — feasible."""
    model, params = constraint_model
    model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([1.0])},
        initial_regimes=["working"],
    )


def test_on_grid_initial_states_accepted(constraint_model):
    """wealth=5.0 is above grid min — fully on grid, feasible."""
    model, params = constraint_model
    model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([5.0])},
        initial_regimes=["working"],
    )
