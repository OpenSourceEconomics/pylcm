"""Test Model Blocks functionality using work-retirement example.

This test file serves two purposes:
1. Defines the target API and behavior for Model Blocks
2. Provides a clear test for when the feature is complete
"""

from __future__ import annotations

# import pytest  # Comment out for basic testing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, ModelBlock, Model

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
# Model Functions - Work Block
# ======================================================================================

def work_utility(
    consumption: ContinuousAction,
    work_hours: ContinuousAction,
    disutility_of_work: float
) -> FloatND:
    """Utility function during working periods."""
    return jnp.log(consumption) - disutility_of_work * work_hours


def labor_income(work_hours: ContinuousAction, wage: float) -> FloatND:
    """Labor income from working."""
    return work_hours * wage


def next_wealth_work(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    """Wealth evolution during work."""
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_experience(
    experience: ContinuousState,
    work_hours: ContinuousAction,
) -> ContinuousState:
    """Experience accumulation."""
    return experience + work_hours


def work_feasibility_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Cannot consume more than wealth."""
    return consumption <= wealth


# ======================================================================================
# Model Functions - Retirement Block
# ======================================================================================

def retirement_utility(
    consumption: ContinuousAction,
    leisure_bonus: float
) -> FloatND:
    """Utility function during retirement."""
    return jnp.log(consumption) + leisure_bonus


def pension_income(pension: ContinuousState) -> FloatND:
    """Pension income during retirement."""
    return pension


def next_wealth_retirement(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    pension_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    """Wealth evolution during retirement."""
    return (1 + interest_rate) * (wealth - consumption) + pension_income


def next_pension(pension: ContinuousState) -> ContinuousState:
    """Pension remains constant."""
    return pension


def retirement_feasibility_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Cannot consume more than wealth."""
    return consumption <= wealth


# ======================================================================================
# Block Transition Functions
# ======================================================================================

def work_to_retirement_transition(
    wealth: ContinuousState,
    experience: ContinuousState,
    pension_replacement_rate: float
) -> dict[str, ContinuousState]:
    """Transform work states to retirement states."""
    return {
        "wealth": wealth,
        "pension": experience * pension_replacement_rate
    }


# ======================================================================================
# Test Model Blocks
# ======================================================================================

# @pytest.fixture
def work_block():
    """Work phase block definition."""
    return ModelBlock(
        name="work",
        actions={
            "consumption": LinspaceGrid(start=0.1, stop=50.0, n_points=100),
            "work_hours": LinspaceGrid(start=0.0, stop=1.0, n_points=21),
        },
        states={
            "wealth": LinspaceGrid(start=0.0, stop=100.0, n_points=51),
            "experience": LinspaceGrid(start=0.0, stop=40.0, n_points=41),
        },
        functions={
            "utility": work_utility,
            "labor_income": labor_income,
            "next_wealth": next_wealth_work,
            "next_experience": next_experience,
            "work_feasibility_constraint": work_feasibility_constraint,
        },
        block_transitions={
            "retirement": work_to_retirement_transition,
        }
    )


# @pytest.fixture
def retirement_block():
    """Retirement phase block definition."""
    return ModelBlock(
        name="retirement",
        actions={
            "consumption": LinspaceGrid(start=0.1, stop=50.0, n_points=100),
        },
        states={
            "wealth": LinspaceGrid(start=0.0, stop=100.0, n_points=51),
            "pension": LinspaceGrid(start=0.0, stop=20.0, n_points=21),
        },
        functions={
            "utility": retirement_utility,
            "pension_income": pension_income,
            "next_wealth": next_wealth_retirement,
            "next_pension": next_pension,
            "retirement_feasibility_constraint": retirement_feasibility_constraint,
        },
        block_transitions={}  # Retirement is absorbing
    )


# @pytest.fixture
def work_retirement_model(work_block, retirement_block):
    """Complete work-retirement model."""
    return Model(
        description="Work-retirement life cycle model with deterministic transition",
        n_periods=10,
        blocks={
            "work": work_block,
            "retirement": retirement_block,
        },
        block_schedule={
            # Work periods 0-6, retirement periods 7-9
            0: "work", 1: "work", 2: "work", 3: "work",
            4: "work", 5: "work", 6: "work",
            7: "retirement", 8: "retirement", 9: "retirement"
        }
    )


# ======================================================================================
# Tests
# ======================================================================================

# @pytest.mark.skip(reason="Model Blocks not yet implemented")
def disabled_test_block_model_solve(work_retirement_model):
    """Test that block models can be solved."""
    model = work_retirement_model

    # Define parameters
    params = {
        "disutility_of_work": 0.5,
        "wage": 10.0,
        "interest_rate": 0.03,
        "leisure_bonus": 2.0,
        "pension_replacement_rate": 0.6,
    }

    # The core test: solve should work and return value functions
    solution = model.solve(params)

    # Basic checks: solution should be a dict with one entry per period
    assert isinstance(solution, dict)
    assert len(solution) == 10
    assert all(period in solution for period in range(10))


def test_block_model_creation():
        work_block = ModelBlock(
            name="work",
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=10)},
            states={"wealth": LinspaceGrid(start=0, stop=100, n_points=11)},
            functions={"utility": lambda c: jnp.log(c)},
        )

            model = Model(
                n_periods=3,
                blocks={"work": work_block},
                block_schedule={0: "work", 1: "work", 2: "work"}
            )
