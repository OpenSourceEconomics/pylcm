"""Example specifications of a deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic action models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid
from lcm.regime import Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        FloatND,
        Int1D,
        IntND,
    )

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class RetirementStatus:
    working: int = 0
    retired: int = 1


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility(
    consumption: ContinuousAction, working: IntND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - disutility_of_work * working


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working: IntND, wage: float | FloatND) -> FloatND:
    return working * wage


def working(retirement: DiscreteAction) -> IntND:
    return 1 - retirement


def wage(age: int | IntND) -> float | FloatND:
    return 1 + 0.1 * age


def age(_period: int | Int1D) -> int | IntND:
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_lagged_retirement(retirement: DiscreteAction) -> DiscreteState:
    return retirement


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction | DiscreteAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def absorbing_retirement_constraint(
    retirement: DiscreteAction, lagged_retirement: DiscreteState
) -> BoolND:
    return jnp.logical_or(
        retirement == RetirementStatus.retired,
        lagged_retirement == RetirementStatus.working,
    )


# ======================================================================================
# Model specifications
# ======================================================================================


def regime_transition_probs_ishkakov_et_al_2017(
    retirement: DiscreteAction,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    lagged_retirement: DiscreteState | IntND,
    _period: int | IntND,
) -> dict[str, float]:
    return {"iskhakov_et_al_2017": 1.0}


ISKHAKOV_ET_AL_2017 = Regime(
    name="iskhakov_et_al_2017",
    description=(
        "Corresponds to the example model in Iskhakov et al. (2017). In comparison to "
        "the extensions below, wage is treated as a constant parameter and therefore "
        "there is no need for the wage and age functions."
    ),
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_lagged_retirement": next_lagged_retirement,
        "borrowing_constraint": borrowing_constraint,
        "absorbing_retirement_constraint": absorbing_retirement_constraint,
        "labor_income": labor_income,
        "working": working,
    },
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
        "lagged_retirement": DiscreteGrid(RetirementStatus),
    },
    regime_transition_probs=regime_transition_probs_ishkakov_et_al_2017,
    regime_state_transitions={"iskhakov_et_al_2017": {}},
)


def regime_transition_probs_ishkakov_et_al_2017_stripped_down(
    retirement: DiscreteAction,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    lagged_retirement: DiscreteState | IntND,
    _period: int | IntND,
) -> dict[str, float]:
    return {"iskhakov_et_al_2017_stripped_down": 1.0}


ISKHAKOV_ET_AL_2017_STRIPPED_DOWN = Regime(
    name="iskhakov_et_al_2017_stripped_down",
    description=(
        "Starts from Iskhakov et al. (2017), removes absorbing retirement constraint "
        "and the lagged_retirement state, and adds wage function that depends on age."
    ),
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "borrowing_constraint": borrowing_constraint,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
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
    regime_transition_probs=regime_transition_probs_ishkakov_et_al_2017_stripped_down,
    regime_state_transitions={"iskhakov_et_al_2017_stripped_down": {}},
)
