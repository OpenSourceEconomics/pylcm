from __future__ import annotations

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


# --------------------------------------------------------------------------------------
# Categorical variables and constants
# --------------------------------------------------------------------------------------
@categorical
class LaborSupply:
    work: int
    retire: int


@categorical
class RegimeId:
    working: int
    retired: int
    dead: int


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility_working(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


def utility_retired(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


# --------------------------------------------------------------------------------------
# State and regime transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_regime_from_working(
    labor_supply: DiscreteAction,
    age: float,
    final_age_alive: float,
) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        jnp.where(
            labor_supply == LaborSupply.retire,
            RegimeId.retired,
            RegimeId.working,
        ),
    )


def next_regime_from_retired(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retired,
    )


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# Regime specifications
# ======================================================================================

working = Regime(
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": LinSpacedGrid(
            start=1,
            stop=400,
            n_points=500,
        ),
    },
    states={
        "wealth": LinSpacedGrid(
            start=1,
            stop=400,
            n_points=100,
        ),
    },
    utility=utility_working,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_regime": next_regime_from_working,
    },
    functions={
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=lambda _age: True,  # Placeholder, overridden at model creation
)

retired = Regime(
    actions={"consumption": LinSpacedGrid(start=1, stop=400, n_points=500)},
    states={
        "wealth": LinSpacedGrid(
            start=1,
            stop=400,
            n_points=100,
        ),
    },
    utility=utility_retired,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_regime": next_regime_from_retired,
    },
    active=lambda _age: True,  # Placeholder, overridden at model creation
)


dead = Regime(
    terminal=True,
    utility=lambda: 0.0,
    active=lambda _age: True,  # Placeholder, overridden at model creation
)


def get_model(n_periods: int) -> Model:
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    return Model(
        regimes={
            "working": working.replace(active=lambda age, n=n_periods: age < n - 1),
            "retired": retired.replace(active=lambda age, n=n_periods: age < n - 1),
            "dead": dead.replace(active=lambda age, n=n_periods: age >= n - 1),
        },
        ages=ages,
    )


def get_params(
    n_periods,
    discount_factor=0.95,
    disutility_of_work=0.5,
    interest_rate=0.05,
    wage=10.0,
):
    final_age_alive = n_periods - 2  # Last age before death transition
    return {
        "working": {
            "discount_factor": discount_factor,
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_regime": {"final_age_alive": final_age_alive},
            "borrowing_constraint": {},
            "labor_income": {"wage": wage},
        },
        "retired": {
            "discount_factor": discount_factor,
            "utility": {},
            "next_wealth": {"interest_rate": interest_rate, "labor_income": 0.0},
            "next_regime": {"final_age_alive": final_age_alive},
            "borrowing_constraint": {},
        },
        "dead": {},
    }
