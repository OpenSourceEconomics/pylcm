"""Consumption-savings model with state-conditioned income risk (stochastic volatility).

A minimal example of the *state-conditioned shock* primitive: the standard deviation of
the IID income shock depends on a discrete ``uncertainty`` regime (low vs high) — a
current-regime conditioning of ``sigma``. The regime follows its own
``MarkovTransition``; the income shock is discretized once on a FIXED common grid (from
the scalar ``sigma``), and each regime's transition row is evaluated directly at the
from-value with that regime's ``sigma``.

Supported for CDF-binned ``NormalIIDProcess`` (``gauss_hermite=False``) and
``TauchenAR1Process``; Gauss-Hermite IID and Rouwenhorst are rejected (their fixed-node
kernels cannot carry a state-conditioned ``sigma``).
"""

import functools

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.processes import StateConditioned
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> FloatND:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def next_uncertainty(uncertainty: DiscreteState, persistence: float) -> FloatND:
    """Symmetric two-state regime: stay with probability ``persistence``."""
    stay = persistence
    return jnp.where(
        uncertainty == Uncertainty.low,
        jnp.array([stay, 1.0 - stay]),
        jnp.array([1.0 - stay, stay]),
    )


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def wealth_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


@functools.cache
def get_model(
    n_periods: int = 6,
    *,
    sigma_low: float = 0.05,
    sigma_high: float = 0.30,
    wealth_n_points: int = 20,
    consumption_n_points: int = 30,
    income_n_points: int = 7,
) -> Model:
    """Create the stochastic-volatility consumption-savings model."""
    final_age_alive = 20 + (n_periods - 2) * 10
    income = NormalIIDProcess(
        n_points=income_n_points,
        gauss_hermite=False,
        mu=0.0,
        sigma=max(sigma_low, sigma_high),  # the FIXED common node grid (widest regime)
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": sigma_low, "high": sigma_high}
        ),
    )
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=40.0, n_points=wealth_n_points),
            "income": income,
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty),
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=10.0, n_points=consumption_n_points
            )
        },
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 10, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def get_params(
    *,
    interest_rate: float = 0.03,
    discount_factor: float = 0.95,
    persistence: float = 0.9,
) -> dict:
    """Parameters for the stochastic-volatility model.

    The per-regime income sigmas are baked into the process at construction; only the
    regime-persistence, interest rate, and discount factor are runtime params here.
    """
    return {
        "discount_factor": discount_factor,
        "alive": {
            "next_wealth": {"interest_rate": interest_rate},
            "next_uncertainty": {"persistence": persistence},
        },
    }
