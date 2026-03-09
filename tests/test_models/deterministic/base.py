"""Deterministic base model — extends lcm_examples.mortality with deterministic regime
transitions (no stochastic mortality).
"""

import jax.numpy as jnp

from lcm import DiscreteGrid, Regime, categorical
from lcm.typing import (
    DiscreteAction,
    ScalarInt,
)
from lcm_examples.mortality import (
    CONSUMPTION_GRID,
    WEALTH_GRID,
    LaborSupply,
    borrowing_constraint,
    dead,
    is_working,
    labor_income,
    next_wealth,
    utility_retirement,
    utility_working,
)

# ---------------------------------------------------------------------------
# Deterministic regime transitions (override the stochastic ones)
# ---------------------------------------------------------------------------


@categorical
class RegimeId:
    working_life: int
    retirement: int
    dead: int


def next_regime_from_working(
    work: DiscreteAction,
    age: float,
    final_age_alive: float,
) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        jnp.where(
            work == LaborSupply.retire,
            RegimeId.retirement,
            RegimeId.working_life,
        ),
    )


def next_regime_from_retirement(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retirement,
    )


# ---------------------------------------------------------------------------
# Deterministic regime objects
# ---------------------------------------------------------------------------

working_life = Regime(
    actions={
        "work": DiscreteGrid(LaborSupply),
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
    active=lambda _age: True,
)

retirement = Regime(
    transition=next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
    active=lambda _age: True,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

from lcm import AgeGrid, Model  # noqa: E402


def get_model(n_periods: int) -> Model:
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
