"""A three-regime consumption-savings model with mortality.

Working life, retirement, and death. The agent chooses labor supply and consumption.
Log utility with work disutility. Mortality is age-dependent: a vector of survival
probabilities (one per period, last entry = 0.0) governs the transition to death.

Based on tests/test_models/deterministic/base.py.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    Period,
    ScalarInt,
    UserAge,
)


@categorical(ordered=True)
class LaborSupply:
    work: ScalarInt
    retire: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def utility_working(
    *, consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


def utility_retirement(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def labor_income(*, is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


def next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_regime_from_working(
    *,
    labor_supply: DiscreteAction,
    period: Period,
    survival_probs: FloatND,
) -> FloatND:
    """Return regime transition probabilities [P(working), P(retired), P(dead)].

    The survival probability is looked up from `survival_probs` by period. The last
    entry must be 0.0 (certain death). Conditional on survival, the agent transitions
    to retirement if they chose to retire.

    """
    sp = survival_probs[period]
    retire_choice = labor_supply == LaborSupply.retire
    return jnp.where(
        retire_choice,
        jnp.array([0.0, sp, 1 - sp]),
        jnp.array([sp, 0.0, 1 - sp]),
    )


def next_regime_from_retirement(
    *,
    period: Period,
    survival_probs: FloatND,
) -> FloatND:
    """Return regime transition probabilities [P(working), P(retired), P(dead)].

    The survival probability is looked up from `survival_probs` by period. The last
    entry must be 0.0 (certain death).

    """
    sp = survival_probs[period]
    return jnp.array([0.0, sp, 1 - sp])


def borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)

_DEFAULT_AGE_GRID = AgeGrid(start=40, inclusive_stop=70, step="10Y")  # 4 periods
_DEFAULT_LAST_AGE = _DEFAULT_AGE_GRID.exact_values[-1]


def working_life_transitions(*, last_age: UserAge | float) -> ByAge:
    """Work, retire or die until the age before `last_age`, then die."""
    return ByAge.until(
        stop_age_exclusive=last_age,
        law=StochasticTransition(
            func=next_regime_from_working,
        ),
        then=StochasticTransition(func=next_regime_from_working),
    )


def retirement_transitions(*, last_age: UserAge | float) -> ByAge:
    """Stay retired or die until the age before `last_age`, then die."""
    return ByAge.until(
        stop_age_exclusive=last_age,
        law=StochasticTransition(func=next_regime_from_retirement),
        then=StochasticTransition(func=next_regime_from_retirement),
    )


working_life = Regime(
    actions={
        "labor_supply": DiscreteGrid(category_class=LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    regime_transitions=working_life_transitions(last_age=_DEFAULT_LAST_AGE),
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
)

retirement = Regime(
    regime_transitions=retirement_transitions(last_age=_DEFAULT_LAST_AGE),
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
)

dead = Regime(
    regime_transitions=None,
    functions={"utility": lambda: 0.0},
)


def get_model(n_periods: int) -> Model:
    """Create the three-regime mortality model.

    Args:
        n_periods: Number of periods.

    Returns:
        A configured Model instance.

    """
    ages = AgeGrid(start=40, inclusive_stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        edges={
            "working_life": {
                "dead": tuple(ages.exact_values[:-1]),
                **(
                    {
                        "working_life": tuple(ages.exact_values[:-2]),
                        "retirement": tuple(ages.exact_values[:-2]),
                    }
                    if ages.exact_values[:-2]
                    else {}
                ),
            },
            "retirement": {
                "dead": tuple(ages.exact_values[:-1]),
                **(
                    {"retirement": tuple(ages.exact_values[:-2])}
                    if ages.exact_values[:-2]
                    else {}
                ),
            },
        },
        regimes={
            "working_life": working_life.replace(
                regime_transitions=working_life_transitions(last_age=last_age)
            ),
            "retirement": retirement.replace(
                regime_transitions=retirement_transitions(last_age=last_age)
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
    )


def _default_survival_probs(n_periods: int) -> FloatND:
    """Build a default survival probability vector with `n_periods - 1` entries.

    Broadly plausible 10-year survival probabilities for ages 40+. The first entry
    uses 0.98, the second-to-last uses 0.82, with linear interpolation in between.
    The last entry is always 0.0 (certain death at the penultimate period).

    The vector is indexed by period; entry `i` gives the probability of surviving from
    period `i` to period `i + 1`.

    """
    if n_periods <= 2:  # noqa: PLR2004
        return jnp.array([0.0])
    if n_periods == 3:  # noqa: PLR2004
        return jnp.array([0.98, 0.0])
    probs = jnp.linspace(0.98, 0.82, n_periods - 2)
    return jnp.concatenate([probs, jnp.array([0.0])])


def get_params(
    *,
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
    wage: float = 10.0,
    survival_probs: FloatND | None = None,
) -> dict:
    """Get default parameters for the mortality model.

    Args:
        n_periods: Number of periods (must match get_model).
        discount_factor: Discount factor.
        disutility_of_work: Disutility of work.
        interest_rate: Interest rate.
        wage: Wage.
        survival_probs: Per-period survival probabilities (last entry must be 0.0).
            If None, a default schedule is generated.

    Returns:
        Parameter dict ready for model.solve().

    """
    if survival_probs is None:
        survival_probs = _default_survival_probs(n_periods)
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "survival_probs": survival_probs,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }
