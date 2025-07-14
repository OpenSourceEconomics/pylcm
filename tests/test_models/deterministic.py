"""Example specifications of a deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic action models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643).

"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model
from lcm.typing import (
    AuxiliaryVariable,
    ConstraintMask,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    Utility,
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
    consumption: ContinuousAction, working: AuxiliaryVariable, disutility_of_work: float
) -> Utility:
    return jnp.log(consumption) - disutility_of_work * working


def utility_with_constraint(
    consumption: ContinuousAction,
    working: AuxiliaryVariable,
    disutility_of_work: float,
    # Temporary workaround for bug described in issue #30, which requires us to pass
    # all state variables to the utility function.
    # TODO(@timmens): Remove function once #30 is fixed (re-use "utility").
    # https://github.com/opensourceeconomics/pylcm/issues/30
    lagged_retirement: DiscreteState,  # noqa: ARG001
) -> Utility:
    return utility(consumption, working, disutility_of_work)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working: AuxiliaryVariable, wage: float) -> AuxiliaryVariable:
    return working * wage


def working(retirement: DiscreteAction) -> AuxiliaryVariable:
    return 1 - retirement


def wage(age: int) -> float:
    return 1 + 0.1 * age


def age(_period: int) -> int:
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: AuxiliaryVariable,
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
) -> ConstraintMask:
    return consumption <= wealth


def absorbing_retirement_constraint(
    retirement: DiscreteAction, lagged_retirement: DiscreteState
) -> ConstraintMask:
    return jnp.logical_or(
        retirement == RetirementStatus.retired,
        lagged_retirement == RetirementStatus.working,
    )


# ======================================================================================
# Model specifications
# ======================================================================================

ISKHAKOV_ET_AL_2017 = Model(
    description=(
        "Corresponds to the example model in Iskhakov et al. (2017). In comparison to "
        "the extensions below, wage is treated as a constant parameter and therefore "
        "there is no need for the wage and age functions."
    ),
    n_periods=3,
    functions={
        "utility": utility_with_constraint,
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
)


ISKHAKOV_ET_AL_2017_STRIPPED_DOWN = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes absorbing retirement constraint "
        "and the lagged_retirement state, and adds wage function that depends on age."
    ),
    n_periods=3,
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
)
