"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic action models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

Extends the deterministic mortality model with Health and PartnerStatus states.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Period,
)
from lcm_examples.mortality import (
    LaborSupply,
    RegimeId,
    dead,
    is_working,
)
from lcm_examples.mortality import retirement as _base_retirement
from lcm_examples.mortality import working_life as _base_working_life

# ======================================================================================
# Additional categorical variables
# ======================================================================================


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=False)
class PartnerStatus:
    single: int
    partnered: int


# ======================================================================================
# Stochastic model functions (different signatures due to health/partner)
# ======================================================================================


def utility_working(
    consumption: ContinuousAction,
    is_working: BoolND,
    health: DiscreteState,
    disutility_of_work: float,
    partner: DiscreteState,  # noqa: ARG001
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - (1 - health / 2) * work_disutility


def utility_retirement(
    consumption: ContinuousAction,
    health: DiscreteState,  # noqa: ARG001
    partner: DiscreteState,  # noqa: ARG001
) -> FloatND:
    return jnp.log(consumption)


def labor_income(is_working: BoolND, wage: FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    partner: DiscreteState,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income + partner


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
def next_health(health: DiscreteState, partner: DiscreteState) -> FloatND:
    """Stochastic transition with JIT-calculated markov transition probabilities."""
    return jnp.where(
        health == Health.bad,
        jnp.where(
            partner == PartnerStatus.single,
            jnp.array([0.9, 0.1]),
            jnp.array([0.5, 0.5]),
        ),
        jnp.where(
            partner == PartnerStatus.partnered,
            jnp.array([0.5, 0.5]),
            jnp.array([0.1, 0.9]),
        ),
    )


def next_partner(
    period: Period,
    labor_supply: DiscreteAction,
    partner: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    """Stochastic transition using pre-calculated markov transition probabilities."""
    return probs_array[period, labor_supply, partner]


# ======================================================================================
# Model specification (extend base regimes via .replace())
# ======================================================================================

# Smaller grids than the base mortality model to keep solve time manageable with the
# additional stochastic states (health, partner).
WEALTH_GRID = LinSpacedGrid(start=1, stop=100, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=100, n_points=200)

working_life = _base_working_life.replace(
    states={
        "health": DiscreteGrid(Health),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": WEALTH_GRID,
    },
    state_transitions={
        "health": MarkovTransition(next_health),
        "partner": MarkovTransition(next_partner),
        "wealth": next_wealth,
    },
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
)

retirement = _base_retirement.replace(
    states={
        "health": DiscreteGrid(Health),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": WEALTH_GRID,
    },
    state_transitions={
        "health": MarkovTransition(next_health),
        "partner": MarkovTransition(next_partner),
        "wealth": next_wealth,
    },
    actions={"consumption": CONSUMPTION_GRID},
    functions={"utility": utility_retirement},
)


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
    n_periods,
    discount_factor=0.95,
    disutility_of_work=0.5,
    interest_rate=0.05,
    wage=10.0,
    probs_array=None,
):
    default_probs_array = jnp.array(
        [
            # Transition from period 0 to period 1
            [
                # Current labor decision 0
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [1.0, 0],
                ],
                # Current labor decision 1
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [0.0, 1.0],
                ],
            ],
            # Transition from period 1 to period 2
            [
                # Current labor decision 0
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [1.0, 0],
                ],
                # Current labor decision 1
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [0.0, 1.0],
                ],
            ],
        ],
    )
    if probs_array is None:
        probs_array = default_probs_array

    return {
        "discount_factor": discount_factor,
        "survival_probs": jnp.concatenate(
            [
                jnp.full(n_periods - 2, 0.97),
                jnp.array([0.0]),
            ]
        ),
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_partner": {"probs_array": probs_array},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"interest_rate": interest_rate, "labor_income": 0.0},
            "next_partner": {
                "labor_supply": LaborSupply.retire,
                "probs_array": probs_array,
            },
        },
    }
