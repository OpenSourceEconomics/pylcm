"""Example specification for a consumption-savings model with health and exercise.

People work for n-1 periods and are retired in the last period.

Note that the parameterization of the model does not make a whole lot of sense, so don't
look too closely inside the functions as opposed to their interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
        ScalarInt,
    )

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables and regime ID
# --------------------------------------------------------------------------------------
@dataclass
class WorkingStatus:
    retired: int = 0
    working: int = 1


@dataclass
class RegimeId:
    working: int = 0
    retirement: int = 1


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    consumption: ContinuousAction,
    working: DiscreteAction,
    health: ContinuousState,
    exercise: ContinuousAction,
    disutility_of_work: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) - (disutility_of_work - health) * working - exercise


def utility_retired(
    wealth: ContinuousState,
    health: ContinuousState,
) -> FloatND:
    return jnp.log(wealth) * health


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(wage: float | FloatND, working: DiscreteAction) -> FloatND:
    return wage * working


def wage(age: float) -> float | FloatND:
    return 1 + 0.1 * age


# --------------------------------------------------------------------------------------
# State and regime transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(
    health: ContinuousState,
    exercise: ContinuousAction,
    working: DiscreteAction,
) -> ContinuousState:
    return health * (1 + exercise - working / 2)


def next_regime(period: int, n_periods: int) -> ScalarInt:
    certain_retirement = period >= n_periods - 2
    return jnp.where(certain_retirement, RegimeId.retirement, RegimeId.working)


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    labor_income: FloatND,
) -> BoolND:
    return consumption <= wealth + labor_income


# ======================================================================================
# Model specification and parameters
# ======================================================================================
RETIREMENT_AGE = 24


def working_is_active(age: float) -> bool:
    return age < RETIREMENT_AGE


def retired_is_active(age: float) -> bool:
    return age >= RETIREMENT_AGE


working = Regime(
    name="working",
    utility=utility,
    functions={
        "labor_income": labor_income,
        "wage": wage,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "consumption": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
        "exercise": LinspaceGrid(
            start=0,
            stop=1,
            n_points=200,
        ),
    },
    states={
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
        "health": LinspaceGrid(
            start=0,
            stop=1,
            n_points=100,
        ),
    },
    transitions={
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_regime": next_regime,
    },
    active=working_is_active,
)


retired = Regime(
    name="retirement",
    terminal=True,
    utility=utility_retired,
    states={
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
        "health": LinspaceGrid(
            start=0,
            stop=1,
            n_points=100,
        ),
    },
    active=retired_is_active,
)


model = Model(
    regimes=[working, retired],
    ages=AgeGrid(start=18, stop=RETIREMENT_AGE + 1, step="Y"),
    regime_id_cls=RegimeId,
)

params = {
    "working": {
        "discount_factor": 0.95,
        "utility": {"disutility_of_work": 0.05},
        "next_wealth": {"interest_rate": 0.05},
        "next_regime": {"n_periods": model.n_periods},
    },
    "retirement": {},
}

# ======================================================================================
# Solve and simulate the model
# ======================================================================================

if __name__ == "__main__":
    n_simulation_subjects = 1_000

    simulation_result = model.solve_and_simulate(
        params=params,
        initial_regimes=["working"] * n_simulation_subjects,
        initial_states={
            "wealth": jnp.full(n_simulation_subjects, 1),
            "health": jnp.full(n_simulation_subjects, 1),
        },
    )
