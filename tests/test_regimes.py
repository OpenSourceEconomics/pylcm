"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
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


def working_from_action(working: DiscreteAction) -> IntND:
    return working


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


def test_regime_creation():
    """Test that individual Regimes can be created with range-based active periods."""
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
            "working": working_from_action,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_transitions={
            "retirement": work_to_retirement_transition,
        },
    )

    # Basic validation that the regime was created correctly
    assert work_regime.name == "work"
    assert work_regime.active == range(7)
    assert "consumption" in work_regime.actions
    assert "working" in work_regime.actions
    assert "wealth" in work_regime.states
    assert len(work_regime.functions) == 5


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
            "working": working_from_action,
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
    assert model.computed_n_periods == 10
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


@pytest.mark.skip(reason="Regime model implementation not yet complete")
def test_single_regime_model():
    """Test that single-regime models work with new Regime API."""
    single_regime = Regime(
        name="default",
        active=range(10),  # All periods 0-9
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
            "working": working_from_action,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
    )

    model = Model(regimes=[single_regime])

    # Basic validation
    assert model.is_regime_model is True
    assert model.computed_n_periods == 10
    assert len(model.regimes) == 1


@pytest.mark.skip(reason="Legacy model test function signatures need to be fixed")
def test_legacy_api_deprecation_warning():
    """Test that legacy API shows deprecation warning but still works."""
    # Capture deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Use legacy API
        model = Model(
            n_periods=5,
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=10)},
            states={"wealth": LinspaceGrid(start=1, stop=100, n_points=11)},
            functions={
                "utility": lambda consumption, _wealth: jnp.log(consumption),
                "next_wealth": lambda wealth, consumption: wealth - consumption,
            },
        )

        # Check that deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Model should still work
    assert model.is_regime_model is False
    assert hasattr(model, "computed_n_periods")
