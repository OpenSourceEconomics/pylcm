"""Example specification for a consumption-savings model with health and exercise.

People work for n-1 periods and are retired in the last period.

Note that the parameterization of the model does not make a whole lot of sense, so don't
look too closely inside the functions as opposed to their interfaces.
"""

import jax

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
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
@categorical
class LaborSupply:
    not_working: int
    working: int


@categorical
class RegimeId:
    working_life: int
    retirement: int


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    consumption: ContinuousAction,
    work: DiscreteAction,
    health: ContinuousState,
    exercise: ContinuousAction,
    disutility_of_work: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) - (disutility_of_work - health) * work - exercise


def utility_retirement(
    wealth: ContinuousState,
    health: ContinuousState,
) -> FloatND:
    return jnp.log(wealth) * health


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(wage: float | FloatND, work: DiscreteAction) -> FloatND:
    return wage * work


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
    work: DiscreteAction,
) -> ContinuousState:
    return health * (1 + exercise - work / 2)


def next_regime(period: int, n_periods: int) -> ScalarInt:
    certain_retirement = period >= n_periods - 2
    return jnp.where(certain_retirement, RegimeId.retirement, RegimeId.working_life)


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


def retirement_is_active(age: float) -> bool:
    return age >= RETIREMENT_AGE


working_life = Regime(
    transition=next_regime,
    active=working_is_active,
    states={
        "wealth": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
            transition=next_wealth,
        ),
        "health": LinSpacedGrid(
            start=0,
            stop=1,
            n_points=100,
            transition=next_health,
        ),
    },
    actions={
        "work": DiscreteGrid(LaborSupply),
        "consumption": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
        "exercise": LinSpacedGrid(
            start=0,
            stop=1,
            n_points=200,
        ),
    },
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "wage": wage,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
)


retirement = Regime(
    transition=None,
    active=retirement_is_active,
    states={
        "wealth": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
            transition=None,
        ),
        "health": LinSpacedGrid(
            start=0,
            stop=1,
            n_points=100,
            transition=None,
        ),
    },
    functions={"utility": utility_retirement},
)


model = Model(
    regimes={
        "working_life": working_life,
        "retirement": retirement,
    },
    ages=AgeGrid(start=18, stop=RETIREMENT_AGE, step="Y"),
    regime_id_class=RegimeId,
)

params = {
    "discount_factor": 0.95,
    "working_life": {
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
        initial_regimes=["working_life"] * n_simulation_subjects,
        initial_states={
            "age": jnp.full(n_simulation_subjects, model.ages.values[0]),
            "wealth": jnp.full(n_simulation_subjects, 1),
            "health": jnp.full(n_simulation_subjects, 1),
        },
    )
