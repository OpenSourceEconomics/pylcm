from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model, ModelBlock

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
# Categorical Variables (shared across blocks)
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
# Block Transition Functions
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


def test_model_blocks_solution():
    """Test that a complete work-retirement model can be solved."""
    work_block = ModelBlock(
        name="work",
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
        block_transitions={
            "retirement": work_to_retirement_transition,
        },
    )

    retirement_block = ModelBlock(
        name="retirement",
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
        block_transitions={},  # Retirement is absorbing
    )

    model = Model(
        description="Work-retirement life cycle model with deterministic transition",
        n_periods=10,
        blocks={
            "work": work_block,
            "retirement": retirement_block,
        },
        block_schedule={
            # Work periods 0-6, retirement periods 7-9
            0: "work",
            1: "work",
            2: "work",
            3: "work",
            4: "work",
            5: "work",
            6: "work",
            7: "retirement",
            8: "retirement",
            9: "retirement",
        },
    )

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
