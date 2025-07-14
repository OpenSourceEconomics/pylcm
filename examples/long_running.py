"""Example specification for a consumption-savings model with health and exercise."""

from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class WorkingStatus:
    retired: int = 0
    working: int = 1


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(consumption, working, health, exercise, disutility_of_work):
    return jnp.log(consumption) - (disutility_of_work - health) * working - exercise


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(wage, working):
    return wage * working


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(health, exercise, working):
    return health * (1 + exercise - working / 2)


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(consumption, wealth, labor_income):
    return consumption <= wealth + labor_income


# ======================================================================================
# Model specification and parameters
# ======================================================================================
RETIREMENT_AGE = 65


MODEL_CONFIG = Model(
    n_periods=RETIREMENT_AGE - 18,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "borrowing_constraint": borrowing_constraint,
        "labor_income": labor_income,
        "wage": wage,
        "age": age,
    },
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
)

PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
