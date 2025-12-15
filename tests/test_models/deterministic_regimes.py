from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
        Period,
        ScalarInt,
    )


# --------------------------------------------------------------------------------------
# Categorical variables and constants
# --------------------------------------------------------------------------------------
@dataclass
class LaborStatus:
    work: int = 0
    retire: int = 1


@dataclass
class RegimeID:
    working: int = 0
    retired: int = 1
    dead: int = 2


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


def is_working(labor_choice: DiscreteAction) -> BoolND:
    return labor_choice == LaborStatus.work


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
    labor_choice: DiscreteAction,
    period: Period,
    n_periods: int,
) -> ScalarInt:
    certain_death_transition = period == n_periods - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeID.dead,
        jnp.where(
            labor_choice == LaborStatus.retire,
            RegimeID.retired,
            RegimeID.working,
        ),
    )


def next_regime_from_retired(period: Period, n_periods: int) -> ScalarInt:
    certain_death_transition = period == n_periods - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeID.dead,
        RegimeID.retired,
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
        "labor_choice": DiscreteGrid(LaborStatus),
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
)

retired = Regime(
    name="retired",
    actions={"consumption": LinspaceGrid(start=1, stop=400, n_points=500)},
    states={
        "wealth": LinspaceGrid(
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
)


dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda wealth: jnp.array([0.0]),  # noqa: ARG005
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=2)},
)


def get_model(n_periods: int) -> Model:
    return Model(
        [working, retired, dead],
        n_periods=n_periods,
        regime_id_cls=RegimeID,
    )


def get_params(
    n_periods,
    beta=0.95,
    disutility_of_work=0.5,
    interest_rate=0.05,
    wage=10.0,
):
    return {
        "working": {
            "beta": beta,
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_regime": {"n_periods": n_periods},
            "borrowing_constraint": {},
            "labor_income": {"wage": wage},
        },
        "retired": {
            "beta": beta,
            "utility": {},
            "next_wealth": {"interest_rate": interest_rate, "labor_income": 0.0},
            "next_regime": {"n_periods": n_periods},
            "borrowing_constraint": {},
        },
        "dead": {},
    }
