"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic action models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
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
@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class PartnerStatus:
    single: int = 0
    partnered: int = 1


@dataclass
class LaborSupply:
    work: int = 0
    retire: int = 1


@dataclass
class RegimeId:
    working: int = 0
    retired: int = 1
    dead: int = 2


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
    period: Period,
    n_periods: int,
) -> ScalarInt:
    certain_death_transition = period == n_periods - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeId.dead,
        jnp.where(
            labor_supply == LaborSupply.retire,
            RegimeId.retired,
            RegimeId.working,
        ),
    )


def next_regime_from_retired(period: Period, n_periods: int) -> ScalarInt:
    certain_death_transition = period == n_periods - 2  # dead in last period
    return jnp.where(
        certain_death_transition,
        RegimeId.dead,
        RegimeId.retired,
    )


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
@lcm.mark.stochastic
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


@lcm.mark.stochastic
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
    name="working",
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": LinspaceGrid(
            start=1,
            stop=100,
            n_points=200,
        ),
    },
    states={
        "health": DiscreteGrid(HealthStatus),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
    },
    utility=utility_working,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_partner": next_partner,
        "next_regime": next_regime_from_working,
    },
    functions={
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=[0],  # Needs to be specified to avoid initialization errors
)


retired = Regime(
    name="retired",
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=200)},
    states={
        "health": DiscreteGrid(HealthStatus),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
    },
    utility=utility_retired,
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    transitions={
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_partner": next_partner,
        "next_regime": next_regime_from_retired,
    },
    active=[0],  # Needs to be specified to avoid initialization errors
)


dead = Regime(
    name="dead",
    terminal=True,
    utility=lambda: 0.0,
    active=[0],  # Needs to be specified to avoid initialization errors
)


def get_model(n_periods: int) -> Model:
    return Model(
        [
            working.replace(active=range(n_periods - 1)),
            retired.replace(active=range(n_periods - 1)),
            dead.replace(active=[n_periods - 1]),
        ],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
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

    return {
        "working": {
            "discount_factor": discount_factor,
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "next_health": {},
            "next_partner": {"partner_transition": partner_transition},
            "next_regime": {"n_periods": n_periods},
            "borrowing_constraint": {},
            "labor_income": {"wage": wage},
        },
        "retired": {
            "discount_factor": discount_factor,
            "utility": {},
            "next_wealth": {"interest_rate": interest_rate, "labor_income": 0.0},
            "next_health": {},
            "next_partner": {
                "labor_supply": LaborSupply.retire,
                "partner_transition": partner_transition,
            },
            "next_regime": {"n_periods": n_periods},
            "borrowing_constraint": {},
        },
        "dead": {},
    }
