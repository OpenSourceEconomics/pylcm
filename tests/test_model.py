"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid, Model
from lcm.exceptions import ModelInitializationError
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
# Regime Transition Functions
# ======================================================================================


def work_to_retirement_transition(
    wealth: ContinuousState,
) -> dict[str, ContinuousState]:
    return {
        "wealth": wealth,
    }


# ======================================================================================
# API Demonstration Tests
# ======================================================================================


@pytest.mark.skip(reason="Regime model implementation not yet complete")
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
        regime_transitions={
            "retirement": work_to_retirement_transition,
        },
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
            "working": lambda: 0,  # Always not working in retirement
            "labor_income": labor_income,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_transitions={},  # Retirement is absorbing
    )

    # Create complete model using new regime-based API
    model = Model(regimes=[work_regime, retirement_regime])

    # Verify model properties
    assert model.is_regime_model is True
    assert model.n_periods == 10
    assert len(model.regimes) == 2

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

    # Additional API test: simulate should also work
    initial_states = {
        "wealth": jnp.array([10.0]),
    }

    simulation = model.simulate(
        params=params, initial_states=initial_states, V_arr_dict=solution, seed=42
    )

    # Basic simulation checks
    assert simulation is not None
    assert len(simulation) > 0


def test_legacy_api_deprecation_warning():
    """Test that legacy API shows deprecation warning."""
    warn_msg = re.escape(
        "Legacy Model API lcm.Model(n_periods, actions, states, functions) is "
        "deprecated and will be removed in version 0.1.0."
    )

    # The deprecation warning should trigger before the initialization error
    with (
        pytest.warns(DeprecationWarning, match=warn_msg),
        pytest.raises(ModelInitializationError),
    ):
        # Model creation will fail due to function signature issues,
        # but the deprecation warning should be triggered first
        Model(
            n_periods=5,
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=10)},
            states={"wealth": LinspaceGrid(start=1, stop=100, n_points=11)},
            functions={
                "utility": lambda consumption: jnp.log(consumption),
                "next_wealth": lambda wealth, consumption: wealth - consumption,
            },
        )


def test_regime_to_model_uses_regime_description():
    regime = Regime(
        name="described_regime",
        description="This is a test regime description",
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
        states={},
        functions={"utility": lambda consumption: jnp.log(consumption)},
    )

    # Should use regime's description
    model = Model(regime, n_periods=1)
    assert model.description == "This is a test regime description"

    # Explicit description should override regime's description
    model_override = Model(regime, n_periods=1, description="Override description")
    assert model_override.description == "Override description"
