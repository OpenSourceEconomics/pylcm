"""Example specifications of fully discrete deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic action models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643). See module `tests.test_models.deterministic` for the
continuous version.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm import DiscreteGrid, Model
from tests.test_models.deterministic import (
    RetirementStatus,
    borrowing_constraint,
    labor_income,
    next_wealth,
    utility,
    working,
)

if TYPE_CHECKING:
    from lcm.typing import (
        DiscreteAction,
        DiscreteState,
        FloatND,
        IntND,
    )

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class ConsumptionChoice:
    low: int = 0
    high: int = 1


@dataclass
class WealthStatus:
    low: int = 0
    medium: int = 1
    high: int = 2


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility_discrete(
    consumption: DiscreteAction,
    working: IntND,
    disutility_of_work: float,
) -> FloatND:
    # In the discrete model, consumption is defined as "low" or "high". This can be
    # translated to the levels 1 and 2.
    consumption_level = 1 + (consumption == ConsumptionChoice.high)
    return utility(consumption_level, working, disutility_of_work)


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth_discrete(
    wealth: DiscreteState,
    consumption: DiscreteAction,
    labor_income: FloatND,
    interest_rate: float,
) -> DiscreteState:
    # For discrete state variables, we need to assure that the next state is also a
    # valid state, i.e., it is a member of the discrete grid.
    continuous = next_wealth(wealth, consumption, labor_income, interest_rate)
    return jnp.clip(jnp.rint(continuous), WealthStatus.low, WealthStatus.high).astype(
        jnp.int32
    )


# ======================================================================================
# Model specifications
# ======================================================================================
ISKHAKOV_ET_AL_2017_DISCRETE = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes absorbing retirement constraint "
        "and the lagged_retirement state, and makes the consumption decision discrete."
    ),
    n_periods=3,
    actions={
        "retirement": DiscreteGrid(RetirementStatus),
        "consumption": DiscreteGrid(ConsumptionChoice),
    },
    states={
        "wealth": DiscreteGrid(WealthStatus),
    },
    utility=utility_discrete,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth_discrete,
    },
    functions={
        "labor_income": labor_income,
        "working": working,
    },
)
