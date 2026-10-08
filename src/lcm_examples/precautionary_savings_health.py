"""Consumption-savings model with health and exercise.

People work for n-1 periods and are retired in the last period. The agent chooses
whether to work, how much to consume, and how much to exercise. Two continuous states
(wealth, health) evolve over time.

Note that the parameterization is chosen to showcase pylcm's features, not to match any
empirical calibration.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt


def utility(
    *,
    consumption: ContinuousAction,
    labor_supply: DiscreteAction,
    health: ContinuousState,
    exercise: ContinuousAction,
    disutility_of_work: ContinuousAction,
) -> FloatND:
    return (
        jnp.log(consumption) - (disutility_of_work - health) * labor_supply - exercise
    )


def utility_retirement(
    *,
    wealth: ContinuousState,
    health: ContinuousState,
) -> FloatND:
    return jnp.log(wealth) * health


def labor_income(*, wage: float | FloatND, labor_supply: DiscreteAction) -> FloatND:
    return wage * labor_supply


def wage(age: int) -> float | FloatND:
    return 1 + 0.1 * age


def next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(
    *,
    health: ContinuousState,
    exercise: ContinuousAction,
    labor_supply: DiscreteAction,
) -> ContinuousState:
    return health * (1 + exercise - labor_supply / 2)


def next_regime(*, period: int, n_periods: int) -> ScalarInt:
    certain_retirement = period >= n_periods - 2
    return jnp.where(certain_retirement, RegimeId.retirement, RegimeId.working_life)


def borrowing_constraint(
    *,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    labor_income: FloatND,
) -> BoolND:
    return consumption <= wealth + labor_income


working_life = Regime(
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=100),
        "health": LinSpacedGrid(start=0, stop=1, n_points=100),
    },
    state_transitions={
        "wealth": next_wealth,
        "health": next_health,
    },
    actions={
        "labor_supply": DiscreteGrid(category_class=LaborSupply),
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
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=100),
        "health": LinSpacedGrid(start=0, stop=1, n_points=100),
    },
    functions={"utility": utility_retirement},
)


def get_model(retirement_age: int = 24) -> Model:
    """Create the consumption-savings model with health and exercise.

    Args:
        retirement_age: Age at which the agent retires (default 24).

    Returns:
        A configured Model instance.

    """
    working_targets = {
        "retirement": tuple(range(18, retirement_age)),
        **(
            {"working_life": tuple(range(18, retirement_age - 1))}
            if tuple(range(18, retirement_age - 1))
            else {}
        ),
    }
    return Model(
        edges={
            "working_life": Transition(
                targets=working_targets, law=DeterministicTransition(func=next_regime)
            )
        },
        regimes={
            "working_life": working_life,
            "retirement": retirement,
        },
        ages=_ages(retirement_age),
        regime_id_class=RegimeId,
        initial_nodes={18: "working_life"},
    )


def get_params(retirement_age: int = 24) -> dict:
    """Get default parameters for the health model.

    Args:
        retirement_age: Age at which the agent retires (must match get_model).

    Returns:
        Parameter dict ready for model.solve().

    """
    return {
        "discount_factor": 0.95,
        "working_life": {
            "utility": {"disutility_of_work": 0.05},
            "next_wealth": {"interest_rate": 0.05},
        },
        "retirement": {},
        "edges": {"working_life": {"n_periods": _ages(retirement_age).n_periods}},
    }


def _ages(retirement_age: int) -> AgeGrid:
    """Return the yearly age grid from 18 to the retirement age."""
    return AgeGrid(start=18, inclusive_stop=retirement_age, step="Y")
