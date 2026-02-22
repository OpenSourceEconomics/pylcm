"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic action models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    DiscreteMarkovGrid,
    LinSpacedGrid,
    Model,
    Regime,
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
    ScalarInt,
)

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@categorical
class HealthStatus:
    bad: int
    good: int


@categorical
class PartnerStatus:
    single: int
    partnered: int


@categorical
class LaborSupply:
    work: int
    retire: int


@categorical
class RegimeId:
    working: int
    retired: int
    dead: int


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility_working(
    consumption: ContinuousAction,
    is_working: BoolND,
    health: DiscreteState,
    disutility_of_work: float,
    partner: DiscreteState,  # noqa: ARG001
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - (1 - health / 2) * work_disutility


def utility_retired(
    consumption: ContinuousAction,
    health: DiscreteState,  # noqa: ARG001
    partner: DiscreteState,  # noqa: ARG001
) -> FloatND:
    return jnp.log(consumption)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(is_working: BoolND, wage: FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


# --------------------------------------------------------------------------------------
# Deterministic state and regime transitions
# --------------------------------------------------------------------------------------
def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    partner: DiscreteState,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income + partner


def next_regime_from_working(
    labor_supply: DiscreteAction,
    age: float,
    final_age_alive: float,
) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        jnp.where(
            labor_supply == LaborSupply.retire,
            RegimeId.retired,
            RegimeId.working,
        ),
    )


def next_regime_from_retired(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retired,
    )


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
def next_health(health: DiscreteState, partner: DiscreteState) -> FloatND:
    """Stochastic transition with JIT-calculated markov transition probabilities."""
    return jnp.where(
        health == HealthStatus.bad,
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
    partner_transition: FloatND,
) -> FloatND:
    """Stochastic transition using pre-calculated markov transition probabilities."""
    return partner_transition[period, labor_supply, partner]


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# Model specification
# ======================================================================================

working = Regime(
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=200,
        ),
    },
    states={
        "health": DiscreteMarkovGrid(HealthStatus, transition=next_health),
        "partner": DiscreteMarkovGrid(PartnerStatus, transition=next_partner),
        "wealth": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
            transition=next_wealth,
        ),
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transition=next_regime_from_working,
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=lambda _age: True,  # Placeholder, overridden at model creation
)


retired = Regime(
    actions={"consumption": LinSpacedGrid(start=1, stop=100, n_points=200)},
    states={
        "health": DiscreteMarkovGrid(HealthStatus, transition=next_health),
        "partner": DiscreteMarkovGrid(PartnerStatus, transition=next_partner),
        "wealth": LinSpacedGrid(
            start=1,
            stop=100,
            n_points=100,
            transition=next_wealth,
        ),
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transition=next_regime_from_retired,
    functions={
        "utility": utility_retired,
    },
    active=lambda _age: True,  # Placeholder, overridden at model creation
)


dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda _age: True,  # Placeholder, overridden at model creation
)


def get_model(n_periods: int) -> Model:
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    return Model(
        regimes={
            "working": working.replace(active=lambda age, n=n_periods: age < n - 1),
            "retired": retired.replace(active=lambda age, n=n_periods: age < n - 1),
            "dead": dead.replace(active=lambda age, n=n_periods: age >= n - 1),
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
    partner_transition=None,
):
    default_partner_transition = jnp.array(
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
    if partner_transition is None:
        partner_transition = default_partner_transition

    final_age_alive = n_periods - 2
    return {
        "discount_factor": discount_factor,
        "working": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_partner": {"partner_transition": partner_transition},
            "next_regime": {"final_age_alive": final_age_alive},
            "labor_income": {"wage": wage},
        },
        "retired": {
            "next_wealth": {"interest_rate": interest_rate, "labor_income": 0.0},
            "next_partner": {
                "labor_supply": LaborSupply.retire,
                "partner_transition": partner_transition,
            },
            "next_regime": {"final_age_alive": final_age_alive},
        },
    }
