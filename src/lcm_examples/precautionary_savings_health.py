"""Consumption-savings model with health and exercise.

People work for n-1 periods and are retired in the last period. The agent chooses
whether to work, how much to consume, and how much to exercise. Two continuous states
(wealth, health) evolve over time.

Note that the parameterization is chosen to showcase pylcm's features, not to match any
empirical calibration.
"""

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

# ======================================================================================
# Categorical variables and regime ID
# ======================================================================================


@categorical(ordered=False)
class LaborSupply:
    not_working: int
    working: int


@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int


# ======================================================================================
# Model functions
# ======================================================================================


def utility(
    consumption: ContinuousAction,
    work: DiscreteAction,
    health: ContinuousState,
    exercise: ContinuousAction,
    disutility_of_work: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) - (disutility_of_work - health) * work - exercise


def utility_retirement(
    wealth: ContinuousState,
    health: ContinuousState,
) -> FloatND:
    return jnp.log(wealth) * health


def labor_income(wage: float | FloatND, work: DiscreteAction) -> FloatND:
    return wage * work


def wage(age: float) -> float | FloatND:
    return 1 + 0.1 * age


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(
    health: ContinuousState,
    exercise: ContinuousAction,
    work: DiscreteAction,
) -> ContinuousState:
    return health * (1 + exercise - work / 2)


def next_regime(period: int, n_periods: int) -> ScalarInt:
    certain_retirement = period >= n_periods - 2
    return jnp.where(certain_retirement, RegimeId.retirement, RegimeId.working_life)


def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    labor_income: FloatND,
) -> BoolND:
    return consumption <= wealth + labor_income


# ======================================================================================
# Default regime objects
# ======================================================================================

_DEFAULT_RETIREMENT_AGE = 24

working_life = Regime(
    transition=next_regime,
    active=lambda age: age < _DEFAULT_RETIREMENT_AGE,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=100),
        "health": LinSpacedGrid(start=0, stop=1, n_points=100),
    },
    state_transitions={
        "wealth": next_wealth,
        "health": next_health,
    },
    actions={
        "work": DiscreteGrid(LaborSupply),
        "consumption": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
        "exercise": LinSpacedGrid(
            start=0,
            stop=1,
            n_points=200,
        ),
    },
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "wage": wage,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
)


retirement = Regime(
    transition=None,
    active=lambda age: age >= _DEFAULT_RETIREMENT_AGE,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=100),
        "health": LinSpacedGrid(start=0, stop=1, n_points=100),
    },
    functions={"utility": utility_retirement},
)


# ======================================================================================
# Factories
# ======================================================================================


def get_model(retirement_age: int = 24) -> Model:
    """Create the consumption-savings model with health and exercise.

    Args:
        retirement_age: Age at which the agent retires (default 24).

    Returns:
        A configured Model instance.

    """
    wl = working_life.replace(
        active=lambda age, _ra=retirement_age: age < _ra,
    )
    ret = retirement.replace(
        active=lambda age, _ra=retirement_age: age >= _ra,
    )

    return Model(
        regimes={
            "working_life": wl,
            "retirement": ret,
        },
        ages=AgeGrid(start=18, stop=retirement_age, step="Y"),
        regime_id_class=RegimeId,
    )


def get_params(retirement_age: int = 24) -> dict:
    """Get default parameters for the health model.

    Args:
        retirement_age: Age at which the agent retires (must match get_model).

    Returns:
        Parameter dict ready for model.solve().

    """
    model = get_model(retirement_age=retirement_age)
    return {
        "discount_factor": 0.95,
        "working_life": {
            "utility": {"disutility_of_work": 0.05},
            "next_wealth": {"interest_rate": 0.05},
            "next_regime": {"n_periods": model.n_periods},
        },
        "retirement": {},
    }
