"""Retirement-only variant of the deterministic base model.

The `retirement` regime in `tests.test_models.deterministic.base` is absorbing: its
value function never depends on the working regime. A two-regime model (retirement +
dead) therefore reproduces the retired part of the Iskhakov et al. (2017) analytical
solution exactly. This makes it the concave (no-discrete-choice) oracle for solver
comparisons: one continuous state, one continuous action, no discrete actions.
"""

import functools

import jax.numpy as jnp

from lcm import AgeGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt
from lcm_examples.iskhakov_et_al_2017 import (
    CONSUMPTION_GRID,
    WEALTH_GRID,
    borrowing_constraint,
    dead,
    next_wealth,
    utility_retirement,
)


@categorical(ordered=False)
class RetirementOnlyRegimeId:
    retirement: ScalarInt
    dead: ScalarInt


def next_regime_from_retirement(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RetirementOnlyRegimeId.dead,
        RetirementOnlyRegimeId.retirement,
    )


retirement = UserRegime(
    transition=next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
)


@functools.cache
def get_model(n_periods: int) -> Model:
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RetirementOnlyRegimeId,
    )


def get_params(
    n_periods: int,
    *,
    discount_factor: float = 0.98,
    interest_rate: float = 0.0,
) -> dict:
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": final_age_alive,
        "retirement": {"next_wealth": {"labor_income": 0.0}},
    }


__all__ = [
    "RetirementOnlyRegimeId",
    "get_model",
    "get_params",
    "next_regime_from_retirement",
    "retirement",
]
