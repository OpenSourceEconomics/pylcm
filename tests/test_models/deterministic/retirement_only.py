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
    ByAge,
    DeterministicTransition,
    Model,
    Transition,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt, UserAge
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


def retirement_transitions(*, last_age: UserAge | float) -> ByAge:
    """Stay retired or die until the age before `last_age`, then die."""
    return ByAge.until(
        stop_age_exclusive=last_age,
        law=DeterministicTransition(func=next_regime_from_retirement),
        then=DeterministicTransition(func=next_regime_from_retirement),
    )


def retirement_edges(
    ages: AgeGrid,
) -> dict[str, dict[str, tuple[UserAge, ...]] | Transition]:
    """Stay retired before the second-to-last age; die from every non-final age.

    Where both edges leave an age, `retirement_transitions` chooses between them.
    """
    stays = tuple(ages.exact_values[:-2])
    dies = tuple(ages.exact_values[:-1])
    if not stays:
        return {"retirement": {"dead": dies}}
    return {
        "retirement": Transition(
            targets={"retirement": stays, "dead": dies},
            law=retirement_transitions(last_age=ages.exact_values[-1]),
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
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        # The law reading `final_age_alive` exists only where some age has two
        # outgoing edges, which takes at least three periods.
        **({"final_age_alive": final_age_alive} if n_periods > 2 else {}),
        "retirement": {"next_wealth": {"labor_income": 0.0}},
    }


__all__ = [
    "RetirementOnlyRegimeId",
    "get_model",
    "get_params",
    "next_regime_from_retirement",
    "retirement",
    "retirement_edges",
    "retirement_transitions",
]
