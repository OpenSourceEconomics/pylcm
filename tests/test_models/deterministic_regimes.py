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
        IntND,
        Period,
    )


# --------------------------------------------------------------------------------------
# Categorical variables and constants
# --------------------------------------------------------------------------------------
@dataclass
class RetirementStatus:
    work: int = 0
    retire: int = 1


@dataclass
class RegimeID:
    working: int = 0
    retired: int = 1
    dead: int = 2


N_PERIODS = 8


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility_working(
    consumption: ContinuousAction, is_working: IntND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - disutility_of_work * is_working


def utility_retired(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(is_working: IntND, wage: float | FloatND) -> FloatND:
    return is_working * wage


def is_working(retirement: DiscreteAction) -> IntND:
    return jnp.where(
        retirement == RetirementStatus.retire,
        0,
        1,
    )


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


def next_regime_from_working(
    retirement: DiscreteAction,
    period: Period,
) -> int:
    certain_death_transition = period == N_PERIODS - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeID.dead,
        jnp.where(
            retirement == RetirementStatus.retire,
            RegimeID.retired,
            RegimeID.working,
        ),
    )


def next_regime_from_retired(period: Period) -> int:
    certain_death_transition = period == N_PERIODS - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeID.dead,
        RegimeID.retired,
    )


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction | DiscreteAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# Regime specifications
# ======================================================================================

working = Regime(
    name="working",
    active=range(6),  # only active in periods 0 to 5
    actions={
        "retirement": DiscreteGrid(RetirementStatus),
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
    active=range(4, N_PERIODS - 1),  # only active in periods 4 to 6
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
    functions={},
)


dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda wealth: 0.0,
    states={"wealth": LinspaceGrid(start=1, stop=2, n_points=2)},
)


model = Model(
    [working, retired, dead],
    n_periods=N_PERIODS,
    regime_id_cls=RegimeID,
)
