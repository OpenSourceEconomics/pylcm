"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic action models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Regime

if TYPE_CHECKING:
    from jax import Array

    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        FloatND,
        Period,
    )

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class PartnerStatus:
    single: int = 0
    partnered: int = 1


@dataclass
class WorkingStatus:
    retired: int = 0
    working: int = 1


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    consumption: ContinuousAction,
    working: DiscreteAction,
    health: DiscreteState,
    disutility_of_work: float,
    partner: DiscreteState,  # noqa: ARG001
) -> FloatND:
    return jnp.log(consumption) - (1 - health / 2) * disutility_of_work * working


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working: DiscreteAction, wage: FloatND) -> FloatND:
    return working * wage


# --------------------------------------------------------------------------------------
# Deterministic state transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    partner: DiscreteState,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income + partner


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
@lcm.mark.stochastic
def next_health(health: DiscreteState, partner: DiscreteState) -> DiscreteState:  # type: ignore[empty-body]
    pass


@lcm.mark.stochastic
def next_partner(  # type: ignore[empty-body]
    period: Period, working: DiscreteAction, partner: DiscreteState
) -> DiscreteState:
    pass


def next_regime() -> dict[str, float | Array]:
    return {"iskhakov_et_al_2017_stochastic": 1.0}


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# Model specification
# ======================================================================================

ISKHAKOV_ET_AL_2017_STOCHASTIC = Regime(
    name="iskhakov_et_al_2017_stochastic",
    description=(
        "Starts from Iskhakov et al. (2017), removes absorbing retirement constraint "
        "and the lagged_retirement state, and adds discrete stochastic state variables "
        "health and partner."
    ),
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "consumption": LinspaceGrid(
            start=1,
            stop=100,
            n_points=200,
        ),
    },
    states={
        "health": DiscreteGrid(HealthStatus),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
    },
    utility=utility,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "iskhakov_et_al_2017_stochastic": {
            "next_wealth": next_wealth,
            "next_health": next_health,
            "next_partner": next_partner,
        },
        "next_regime": next_regime,
    },
    functions={
        "labor_income": labor_income,
    },
)
