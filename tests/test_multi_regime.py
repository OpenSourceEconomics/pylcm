"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteState,
        FloatND,
        IntND,
        ParamsDict,
        RegimeName,
    )


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


@dataclass
class RegimeID:
    work: int = 0
    retirement: int = 1


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
    consumption: ContinuousAction,
    working: IntND,
    disutility_of_work: float,
    health: DiscreteState,  # noqa: ARG001
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


@lcm.mark.stochastic
def next_health(health: DiscreteState, health_transition: FloatND) -> FloatND:
    return health_transition[health]


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


def next_regime_from_working(period: int) -> dict[str, float | Array]:
    return {
        "work": jnp.where(period < 6, 1, 0.5),
        "retirement": jnp.where(period < 6, 0, 0.5),
    }


def next_regime_from_retirement() -> dict[str, float | Array]:
    return {"work": 0.0, "retirement": 1.0}


def working_during_retirement() -> IntND:
    return jnp.array(0)


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
            "next_regime": next_regime_from_working,
        },
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
            "next_regime": next_regime_from_retirement,
        },
    )

    # Create complete model using new regime-based API
    model = Model(
        regimes=[work_regime, retirement_regime],
        n_periods=10,
        regime_id_cls=RegimeID,
    )

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
        "work__next_health": {"health_transition": health_transition},
        "retirement__next_health": {"health_transition": health_transition},
        "borrowing_constraint": {},
    }

    params_retired = {
        "beta": 0.8,
        "utility": {"disutility_of_work": 2.0},
        "labor_income": {"wage": 20},
        "work__next_wealth": {"interest_rate": 0.1},
        "retirement__next_wealth": {"interest_rate": 0.1},
        "work__next_health": {"health_transition": health_transition},
        "retirement__next_health": {"health_transition": health_transition},
        "working": {},
        "borrowing_constraint": {},
    }

    params: dict[RegimeName, ParamsDict] = {
        "work": params_working,
        "retirement": params_retired,
    }

    # The core test: solve should work and return value functions
    solution = model.solve(params)
    simulation = model.simulate(
        params=params,
        initial_states={
            "work": {
                "wealth": jnp.array([5.0, 20, 40, 70]),
                "health": jnp.array([1, 1, 1, 1]),
            },
            "retirement": {
                "wealth": jnp.array([5.0, 20, 40, 70]),
                "health": jnp.array([1, 1, 1, 1]),
            },
        },
        initial_regimes=["work"] * 4,
        V_arr_dict=solution,
    )

    # Basic checks: solution should be a dict with one entry per period
    assert isinstance(solution, dict)
    assert len(solution) == 10
    assert all(period in solution for period in range(10))

    assert isinstance(simulation, dict)
    assert len(simulation) == 2
