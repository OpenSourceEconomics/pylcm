"""A three-regime consumption-savings model with mortality.

Working life, retirement, and death. The agent chooses labor supply and consumption.
Log utility with work disutility. Death is certain at the final age; from the second
period onwards it can also occur randomly with age-dependent survival probabilities.

Based on tests/test_models/deterministic/base.py.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
)

# ---------------------------------------------------------------------------
# Categorical variables
# ---------------------------------------------------------------------------


@categorical
class LaborSupply:
    work: int
    retire: int


@categorical
class RegimeId:
    working_life: int
    retirement: int
    dead: int


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def utility_working(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


def utility_retirement(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def labor_income(is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(work: DiscreteAction) -> BoolND:
    return work == LaborSupply.work


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_regime_from_working(
    work: DiscreteAction,
    age: float,
    final_age_alive: float,
    survival_prob: float,
) -> FloatND:
    """Return regime transition probabilities [P(working), P(retired), P(dead)].

    At the final age alive, death is certain. Otherwise the agent survives with
    probability `survival_prob` and, conditional on survival, transitions to
    retirement if they chose to retire.

    """
    retire_choice = work == LaborSupply.retire
    return jnp.where(
        age >= final_age_alive,
        jnp.array([0.0, 0.0, 1.0]),
        jnp.where(
            retire_choice,
            jnp.array([0.0, survival_prob, 1 - survival_prob]),
            jnp.array([survival_prob, 0.0, 1 - survival_prob]),
        ),
    )


def next_regime_from_retirement(
    age: float,
    final_age_alive: float,
    survival_prob: float,
) -> FloatND:
    """Return regime transition probabilities [P(working), P(retired), P(dead)].

    At the final age alive, death is certain. Otherwise the agent survives with
    probability `survival_prob`.

    """
    return jnp.where(
        age >= final_age_alive,
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([0.0, survival_prob, 1 - survival_prob]),
    )


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# ---------------------------------------------------------------------------
# Default grids
# ---------------------------------------------------------------------------

WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)

# ---------------------------------------------------------------------------
# Default regime objects
# ---------------------------------------------------------------------------

working_life = Regime(
    actions={
        "work": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    transition=MarkovTransition(next_regime_from_working),
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=lambda _age: True,
)

retirement = Regime(
    transition=MarkovTransition(next_regime_from_retirement),
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
    active=lambda _age: True,
)

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda _age: True,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def get_model(n_periods: int) -> Model:
    """Create the three-regime mortality model.

    Args:
        n_periods: Number of periods.

    Returns:
        A configured Model instance.

    """
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead.replace(active=lambda age, la=last_age: age >= la),
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


def get_params(
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
    wage: float = 10.0,
    survival_prob: float = 0.97,
) -> dict:
    """Get default parameters for the mortality model.

    Args:
        n_periods: Number of periods (must match get_model).
        discount_factor: Discount factor.
        disutility_of_work: Disutility of work.
        interest_rate: Interest rate.
        wage: Wage.
        survival_prob: Per-period survival probability (default 0.97).

    Returns:
        Parameter dict ready for model.solve().

    """
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": final_age_alive,
        "survival_prob": survival_prob,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }
