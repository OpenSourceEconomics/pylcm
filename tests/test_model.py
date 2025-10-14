"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid, Model
from lcm.regime import Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        FloatND,
        IntND,
    )


# ======================================================================================
# Categorical Variables (shared across regimes)
# ======================================================================================


@dataclass
class WorkingStatus:
    not_working: int = 0
    working: int = 1


# ======================================================================================
# Model Functions (shared where possible)
# ======================================================================================


def utility(
    consumption: ContinuousAction, working: IntND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - disutility_of_work * working


def labor_income(working: IntND, wage: float) -> FloatND:
    return working * wage


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# API Demonstration Tests
# ======================================================================================


def next_wealth_regime_transition(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def regime_transition_probs_working_to_retirement(*args, **kwargs) -> dict[str, float]:
    return {"work": 0.6, "retirement": 0.4}


def regime_transition_probs_retirement_absorbing(*args, **kwargs) -> dict[str, float]:
    return {"work": 0.0, "retirement": 1.0}


def working_during_retirement() -> IntND:
    return 0


def test_work_retirement_model_solution():
    """Test that a complete work-retirement model can be solved using new Regime API."""
    # Create work regime
    work_regime = Regime(
        name="work",
        active=range(7),  # Periods 0-6
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "labor_income": labor_income,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_state_transitions={
            "retirement": {"next_wealth": next_wealth_regime_transition},
        },
        regime_transition_probs=regime_transition_probs_working_to_retirement,
    )

    # Create retirement regime
    retirement_regime = Regime(
        name="retirement",
        active=range(7, 10),  # Periods 7-9
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "working": working_during_retirement,  # Always not working in retirement
            "labor_income": labor_income,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_state_transitions={
            "work": {"next_wealth": next_wealth_regime_transition},
        },  # Retirement is absorbing
        regime_transition_probs=regime_transition_probs_retirement_absorbing,
    )

    # Create complete model using new regime-based API
    model = Model(regimes=[work_regime, retirement_regime], n_periods=10)

    # Verify model properties
    assert model.n_periods == 10
    assert len(model.internal_regimes) == 2

    # Define parameters (similar to deterministic model)
    params = {
        "disutility_of_work": 2.0,
        "wage": 10.0,
        "interest_rate": 0.05,
    }

    # The core test: solve should work and return value functions
    solution = model.solve(params)

    # Basic checks: solution should be a dict with one entry per period
    assert isinstance(solution, dict)
    assert len(solution) == 10
    assert all(period in solution for period in range(10))


@pytest.mark.skip(
    reason="Not clear what behavior we want for this yet, and not urgent at all!!!"
)
def test_regime_to_model_uses_regime_description():
    regime = Regime(
        name="described_regime",
        active=range(1),
        description="This is a test regime description",
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
        states={},
        functions={"utility": lambda consumption: jnp.log(consumption)},
    )

    # Should use regime's description
    model = Model(regime, n_periods=1)
    assert model.description == "described_regime: This is a test regime description"

    # Explicit description should override regime's description
    model_override = Model(regime, n_periods=1, description="Override description")
    assert model_override.description == "Override description"
