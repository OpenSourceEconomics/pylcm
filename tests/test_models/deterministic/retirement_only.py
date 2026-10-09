"""Retirement-only variant of the deterministic base model.

The `retirement` regime in `tests.test_models.deterministic.base` is absorbing: its
value function never depends on the working regime. A two-regime model (retirement +
dead) therefore reproduces the retired part of the Iskhakov et al. (2017) analytical
solution exactly. This makes it the concave (no-discrete-choice) oracle for solver
comparisons: one continuous state, one continuous action, no discrete actions.
"""

import functools

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DeterministicTransition,
    Model,
    Transition,
    categorical,
)
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


def next_regime_from_retirement(*, age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RetirementOnlyRegimeId.dead,
        RetirementOnlyRegimeId.retirement,
    )


# Law of the `retirement` edges: stay retired until `final_age_alive`, then die.
RETIREMENT_LAW = DeterministicTransition(func=next_regime_from_retirement)


def retirement_edges(
    ages: AgeGrid,
) -> dict[str, Transition]:
    """Stay retired before the second-to-last age; die from every non-final age."""
    stays = tuple(ages.exact_values[:-2])
    return {
        "retirement": Transition(
            targets={
                "dead": tuple(ages.exact_values[:-1]),
                **({"retirement": stays} if stays else {}),
            },
            law=RETIREMENT_LAW,
        )
    }


retirement = UserRegime(
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
)


@functools.cache
def get_model(n_periods: int) -> Model:
    ages = AgeGrid(start=40, inclusive_stop=40 + (n_periods - 1) * 10, step="10Y")
    return Model(
        regimes={
            "retirement": retirement,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RetirementOnlyRegimeId,
        initial_nodes={ages.exact_values[0]: "retirement"},
        edges=retirement_edges(ages),
    )


def get_params(
    *,
    n_periods: int,
    discount_factor: float = 0.98,
    interest_rate: float = 0.0,
) -> dict:
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": 40 + (n_periods - 2) * 10,
        "retirement": {"next_wealth": {"labor_income": 0.0}},
    }


__all__ = [
    "RETIREMENT_LAW",
    "RetirementOnlyRegimeId",
    "get_model",
    "get_params",
    "next_regime_from_retirement",
    "retirement",
    "retirement_edges",
]
