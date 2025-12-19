from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
        IntND,
        Period,
        ScalarInt,
    )


# --------------------------------------------------------------------------------------
# Categorical variables and constants
# --------------------------------------------------------------------------------------
@dataclass
class LaborSupply:
    work: int = 0
    retire: int = 1


@dataclass
class RegimeID:
    working: int = 0
    dead: int = 1


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


def wage(age: int | IntND) -> float | FloatND:
    return 1 + 0.1 * age


def age(period: Period) -> int | IntND:
    return period + 18


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


def next_regime(period: Period, n_periods: int) -> ScalarInt:
    certain_death_transition = period == n_periods - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeID.dead,
        RegimeID.working,
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
    name="working",
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": LinspaceGrid(
            start=1,
            stop=400,
            n_points=500,
        ),
    },
    states={
        "wealth": LinspaceGrid(
            start=1,
            stop=400,
            n_points=100,
        ),
    },
    utility=utility,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_regime": next_regime,
    },
    functions={
        "labor_income": labor_income,
        "is_working": is_working,
        "age": age,
        "wage": wage,
    },
)


dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda wealth: jnp.array([0.0]),  # noqa: ARG005
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=2)},
)


def get_model(n_periods: int) -> Model:
    return Model(
        [working, dead],
        n_periods=n_periods,
        regime_id_cls=RegimeID,
    )


def get_params(
    n_periods: int,
    beta: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
) -> dict[str, Any]:
    return {
        "working": {
            "beta": beta,
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_regime": {"n_periods": n_periods},
            "borrowing_constraint": {},
            "labor_income": {},
        },
        "dead": {},
    }
