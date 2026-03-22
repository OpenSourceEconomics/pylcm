"""A tiny two-regime consumption-savings model.

Three-period model with working life and retirement. The agent chooses whether to
work and how much to consume. A simple tax-and-transfer system guarantees a
consumption floor. Savings earn interest.

Based on docs/getting_started/tiny_example.ipynb.
"""

import jax.numpy as jnp

from lcm import (
    DiscreteGrid,
    IntAgeGrid,
    LinSpacedGrid,
    LogSpacedGrid,
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
    ScalarInt,
)


@categorical(ordered=True)
class LaborSupply:
    do_not_work: int
    work: int


@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int


def utility(
    consumption: ContinuousAction,
    labor_supply: DiscreteAction,
    disutility_of_work: float,
    risk_aversion: float,
) -> FloatND:
    return consumption ** (1 - risk_aversion) / (
        1 - risk_aversion
    ) - disutility_of_work * (labor_supply == LaborSupply.work)


def utility_retirement(wealth: ContinuousState, risk_aversion: float) -> FloatND:
    return wealth ** (1 - risk_aversion) / (1 - risk_aversion)


def earnings(labor_supply: DiscreteAction, wage: float) -> FloatND:
    return jnp.where(labor_supply == LaborSupply.work, wage, 0.0)


def taxes_transfers(
    earnings: FloatND,
    wealth: ContinuousState,
    consumption_floor: float,
    tax_rate: float,
) -> FloatND:
    return jnp.where(
        earnings >= consumption_floor,
        tax_rate * (earnings - consumption_floor),
        jnp.minimum(0.0, wealth + earnings - consumption_floor),
    )


def end_of_period_wealth(
    wealth: ContinuousState,
    earnings: FloatND,
    taxes_transfers: FloatND,
    consumption: ContinuousAction,
) -> FloatND:
    return wealth + earnings - taxes_transfers - consumption


def next_wealth(end_of_period_wealth: FloatND, interest_rate: float) -> ContinuousState:
    return (1 + interest_rate) * end_of_period_wealth


def borrowing_constraint(end_of_period_wealth: FloatND) -> BoolND:
    return end_of_period_wealth >= 0


def next_regime(age: int, last_working_age: float) -> ScalarInt:
    return jnp.where(
        age >= last_working_age, RegimeId.retirement, RegimeId.working_life
    )


WEALTH_GRID = LinSpacedGrid(start=0, stop=50, n_points=25)
CONSUMPTION_GRID = LogSpacedGrid(start=4, stop=50, n_points=100)


_DEFAULT_AGE_GRID = IntAgeGrid(start=25, stop=65, step="20Y")
_RETIREMENT_AGE = _DEFAULT_AGE_GRID.exact_values[-1]

working_life = Regime(
    transition=next_regime,
    active=lambda age: age < _RETIREMENT_AGE,
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    functions={
        "utility": utility,
        "earnings": earnings,
        "taxes_transfers": taxes_transfers,
        "end_of_period_wealth": end_of_period_wealth,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
)

retirement = Regime(
    transition=None,
    active=lambda age: age >= _RETIREMENT_AGE,
    states={"wealth": WEALTH_GRID},
    functions={"utility": utility_retirement},
)


def get_model(
    *,
    n_periods: int = 3,
    step: str = "20Y",
) -> Model:
    """Create the tiny consumption-savings model.

    Args:
        n_periods: Number of periods (default 3).
        step: Age step string for AgeGrid (default "20Y").

    Returns:
        A configured Model instance.

    """
    age_grid = IntAgeGrid(
        start=25, stop=25 + (n_periods - 1) * int(step[:-1]), step=step
    )
    retirement_age = age_grid.exact_values[-1]

    wl = working_life.replace(
        active=lambda age, _ra=retirement_age: age < _ra,
    )
    ret = retirement.replace(
        active=lambda age, _ra=retirement_age: age >= _ra,
    )

    return Model(
        regimes={"working_life": wl, "retirement": ret},
        ages=age_grid,
        regime_id_class=RegimeId,
        description="A tiny consumption-savings model.",
    )


def get_params(
    *,
    n_periods: int = 3,
    step: str = "20Y",
) -> dict:
    """Get default parameters for the tiny model.

    Args:
        n_periods: Number of periods (must match get_model).
        step: Age step string (must match get_model).

    Returns:
        Parameter dict ready for model.solve().

    """
    age_grid = IntAgeGrid(
        start=25, stop=25 + (n_periods - 1) * int(step[:-1]), step=step
    )
    return {
        "discount_factor": 0.95,
        "risk_aversion": 1.5,
        "interest_rate": 0.03,
        "working_life": {
            "utility": {"disutility_of_work": 1.0},
            "earnings": {"wage": 20.0},
            "taxes_transfers": {"consumption_floor": 2.0, "tax_rate": 0.2},
            "next_regime": {"last_working_age": age_grid.exact_values[-2]},
        },
    }
