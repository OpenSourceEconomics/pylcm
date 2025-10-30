"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid
from lcm.model import Model
from lcm.regime import Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteState,
        FloatND,
        IntND,
        ParamsDict,
    )

type RegimeName = str


# ======================================================================================
# Categorical Variables (shared across regimes)
# ======================================================================================


@dataclass
class WorkingStatus:
    not_working: int = 0
    working: int = 1


@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


# ======================================================================================
# Model Functions (shared where possible)
# ======================================================================================


def utility_work(
    consumption: ContinuousAction,
    working: IntND,
    disutility_of_work: float,
    health: DiscreteState,
) -> FloatND:
    return jnp.log(consumption) - (1 - health / 2) * disutility_of_work * working


def utility_retirement(
    consumption: ContinuousAction, working: IntND, disutility_of_work: float, health
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


def next_health(health: DiscreteState) -> DiscreteState:  # type: ignore[empty-body]
    return health


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
    health: DiscreteState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def regime_transition_probs_working_to_retirement() -> dict[str, float]:
    return {"work": 0.6, "retirement": 0.4}


def regime_transition_probs_retirement_absorbing() -> dict[str, float]:
    return {"work": 0.0, "retirement": 1.0}


def working_during_retirement() -> IntND:
    return 0


def test_work_retirement_model_solution():
    """Test that a complete work-retirement model can be solved using new Regime API."""
    # Create work regime
    work_regime = Regime(
        name="work",
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
            "health": DiscreteGrid(HealthStatus),
        },
        utility=utility_work,
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={
            "labor_income": labor_income,
        },
        transitions={
            "work": {
                "next_wealth": next_wealth,
                "next_health": next_health,
            },
            "retirement": {
                "next_wealth": next_wealth_regime_transition,
                "next_health": next_health,
            },
        },
        regime_transition_probabilities=regime_transition_probs_working_to_retirement,
    )

    # Create retirement regime
    retirement_regime = Regime(
        name="retirement",
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
            "health": DiscreteGrid(HealthStatus),
        },
        utility=utility_retirement,
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={
            "working": working_during_retirement,  # Always not working in retirement
            "labor_income": labor_income,
        },
        transitions={
            "work": {
                "next_wealth": next_wealth_regime_transition,
                "next_health": next_health,
            },
            "retirement": {"next_wealth": next_wealth, "next_health": next_health},
        },  # Retirement is absorbing
        regime_transition_probabilities=regime_transition_probs_retirement_absorbing,
    )

    # Create complete model using new regime-based API
    model = Model(regimes=[work_regime, retirement_regime], n_periods=10)

    # Verify model properties
    assert model.n_periods == 10
    assert len(model.internal_regimes) == 2

    health_transition = jnp.array(
        [
            # From bad health today to (bad, good) tomorrow
            [0.9, 0.1],
            # From good health today to (bad, good) tomorrow
            [0.5, 0.5],
        ],
    )

    # Define parameters
    params_working = {
        "beta": 0.9,
        "utility": {"disutility_of_work": 2.0},
        "labor_income": {"wage": 25},
        "work__next_wealth": {"interest_rate": 0.1},
        "retirement__next_wealth": {"interest_rate": 0.1},
        "work__next_health": {},
        "retirement__next_health": {},
        "borrowing_constraint": {},
        "shocks": {},
    }

    params_retired = {
        "beta": 0.8,
        "utility": {"disutility_of_work": 2.0},
        "labor_income": {"wage": 20},
        "work__next_wealth": {"interest_rate": 0.1},
        "retirement__next_wealth": {"interest_rate": 0.1},
        "work__next_health": {},
        "retirement__next_health": {},
        "working": {},
        "borrowing_constraint": {},
        "shocks": {},
    }

    params: dict[RegimeName, ParamsDict] = {
        "work": params_working,
        "retirement": params_retired,
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
