from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical

if TYPE_CHECKING:
    from lcm.grids import ContinuousGrid
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
    dead: int


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


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
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.working,
    )


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption < wealth


# ======================================================================================
# Regime specifications
# ======================================================================================

working = Regime(
    name="working",
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
    utility=utility,
    constraints={"borrowing_constraint": borrowing_constraint},
    transitions={
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    },
    functions={
        "labor_income": labor_income,
        "is_working": is_working,
        "wage": wage,
    },
    active=lambda _age: True,  # placeholder, will be replaced by get_model()
)


dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda: 0.0,
    active=lambda _age: True,  # placeholder, will be replaced by get_model()
)


START_AGE = 18


DEFAULT_WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
DEFAULT_CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)


def get_model(
    n_periods: int,
    wealth_grid: ContinuousGrid = DEFAULT_WEALTH_GRID,
    consumption_grid: ContinuousGrid = DEFAULT_CONSUMPTION_GRID,
) -> Model:
    final_age_alive = START_AGE + n_periods - 2
    return Model(
        [
            working.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": wealth_grid},
                actions={
                    "labor_supply": DiscreteGrid(LaborSupply),
                    "consumption": consumption_grid,
                },
            ),
            dead.replace(active=lambda age: age > final_age_alive),
        ],
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_cls=RegimeId,
    )


def get_params(
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
) -> dict[str, Any]:
    final_age_alive = START_AGE + n_periods - 2
    return {
        "working": {
            "discount_factor": discount_factor,
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_regime": {"final_age_alive": final_age_alive},
            "borrowing_constraint": {},
            "labor_income": {},
        },
        "dead": {},
    }
