"""The deterministic retirement model of Iskhakov, Jørgensen, Rust & Schjerning (2017).

A worker chooses consumption and whether to keep working or retire; retirement is
absorbing and death arrives deterministically at a known age. Log utility with a
work disutility. The model replicates "The endogenous grid method for
discrete-continuous dynamic choice models with (or without) taste shocks",
Quantitative Economics 8(2), 317-365, https://doi.org/10.3982/QE643, which provides
a closed-form solution (shipped as test data in `tests/data/analytical_solution/`).

The discrete retirement choice makes the value function non-concave and produces
the paper's signature saw-tooth consumption function: each tooth corresponds to a
different optimal retirement age. As people retire later, their lifetime wealth
increases, which changes the optimal consumption path; the optimal retirement age
moves period by period as current wealth rises, so consumption jumps at the wealth
thresholds in between.
"""

import functools

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime
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
    work: ScalarInt
    retire: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def utility_working(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


def utility_retirement(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def labor_income(is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def next_regime_from_working(
    labor_supply: DiscreteAction,
    age: int,
    final_age_alive: float,
) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        jnp.where(
            labor_supply == LaborSupply.retire,
            RegimeId.retirement,
            RegimeId.working_life,
        ),
    )


def next_regime_from_retirement(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retirement,
    )


WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)

_DEFAULT_AGE_GRID = AgeGrid(start=40, stop=70, step="10Y")  # 4 periods
_DEFAULT_LAST_AGE = _DEFAULT_AGE_GRID.exact_values[-1]


working_life = Regime(
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    transition=next_regime_from_working,
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=lambda age: age < _DEFAULT_LAST_AGE,
)

retirement = Regime(
    transition=next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
    active=lambda age: age < _DEFAULT_LAST_AGE,
)

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda _age: True,
)


@functools.cache
def get_model(n_periods: int) -> Model:
    """Create the three-regime retirement model.

    Args:
        n_periods: Number of periods. The last period is spent in the terminal
            `dead` regime; the paper's five-decision-period parametrization
            corresponds to `n_periods=6`.

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
            "dead": dead,
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
) -> dict:
    """Get parameters for the retirement model.

    The paper's analytical-solution parametrization is `discount_factor=0.98`,
    `disutility_of_work=1.0`, `interest_rate=0.0`, `wage=20.0`.

    Args:
        n_periods: Number of periods (must match `get_model`).
        discount_factor: Discount factor.
        disutility_of_work: Utility cost of working.
        interest_rate: Interest rate on savings.
        wage: Per-period labor income when working.

    Returns:
        Parameter dict ready for `model.solve()`.

    """
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": final_age_alive,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }


__all__ = [
    "CONSUMPTION_GRID",
    "WEALTH_GRID",
    "LaborSupply",
    "RegimeId",
    "borrowing_constraint",
    "dead",
    "get_model",
    "get_params",
    "is_working",
    "labor_income",
    "next_regime_from_retirement",
    "next_regime_from_working",
    "next_wealth",
    "retirement",
    "utility_retirement",
    "utility_working",
    "working_life",
]
