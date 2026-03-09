"""Regression test model — 2-regime subset of the mortality model.

Extends the mortality model with an age-dependent wage function and supports
configurable grid types for testing various grid classes.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    Regime,
    categorical,
)
from lcm.grids import UniformContinuousGrid
from lcm.typing import (
    FloatND,
    ScalarInt,
    UserParams,
)
from lcm_examples.mortality import (
    LaborSupply,
    borrowing_constraint,
    is_working,
    labor_income,
    next_wealth,
)
from lcm_examples.mortality import (
    utility_working as utility,
)


# --------------------------------------------------------------------------------------
# Regression-specific: RegimeId (2 regimes) and wage function
# --------------------------------------------------------------------------------------
@categorical(ordered=False)
class RegimeId:
    working_life: int
    dead: int


def wage(age: float) -> float | FloatND:
    return 1 + 0.1 * age


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.working_life,
    )


# ======================================================================================
# Regime specifications
# ======================================================================================

START_AGE = 18
DEFAULT_WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
DEFAULT_CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)

working_life = Regime(
    actions={
        "work": DiscreteGrid(LaborSupply),
        "consumption": DEFAULT_CONSUMPTION_GRID,
    },
    states={
        "wealth": DEFAULT_WEALTH_GRID,
    },
    state_transitions={
        "wealth": next_wealth,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
    transition=next_regime,
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "is_working": is_working,
        "wage": wage,
    },
    active=lambda _age: True,
)


dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda _age: True,
)


def get_model(
    n_periods: int,
    wealth_grid: UniformContinuousGrid
    | IrregSpacedGrid
    | PiecewiseLinSpacedGrid
    | PiecewiseLogSpacedGrid = DEFAULT_WEALTH_GRID,
    consumption_grid: UniformContinuousGrid
    | IrregSpacedGrid
    | PiecewiseLinSpacedGrid
    | PiecewiseLogSpacedGrid = DEFAULT_CONSUMPTION_GRID,
) -> Model:
    final_age_alive = START_AGE + n_periods - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": wealth_grid},
                actions={
                    "work": DiscreteGrid(LaborSupply),
                    "consumption": consumption_grid,
                },
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def get_params(
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
) -> UserParams:
    final_age_alive = START_AGE + n_periods - 2
    return {
        "discount_factor": discount_factor,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_regime": {"final_age_alive": final_age_alive},
        },
    }
