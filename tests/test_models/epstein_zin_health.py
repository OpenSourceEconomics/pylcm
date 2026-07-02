"""Toy Epstein-Zin savings model with health-dependent mortality.

A pared-down version of the consumer block of Atal, Fang, Karlsson &
Ziebarth (2025, JPE 133(6), doi:10.1086/734781) — savings, a two-state
health Markov chain, and health-dependent survival into a terminal `dead`
regime — with the recursion swapped to Epstein-Zin:

- `V = ((1 - b) * c^r + b * CE^r)^(1/r)` via a user-supplied `H`,
- `CE = (E[V'^(1-g)])^(1/(1-g))` via `PowerCertaintyEquivalent`.

Utility is consumption itself, so values stay in (positive) consumption
units and the power transform is well-defined. The `dead` bequest value
`sqrt(wealth)` is strictly positive at every reachable wealth. Grids are
sized so an in-test numpy backward induction on the same grids reproduces
the solve exactly: next wealth `w - c + income` stays inside the wealth
grids (`income` equals the consumption-grid lower bound), so linear
interpolation never extrapolates.
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
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    Period,
    ScalarInt,
)

WEALTH_GRID = LinSpacedGrid(start=0.5, stop=12.0, n_points=6)
DEAD_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=5.0, n_points=7)

INCOME = 0.5
SURVIVAL_PROBS = (0.95, 0.85, 0.0)
BAD_HEALTH_SURVIVAL_FACTOR = 0.9
HEALTH_TRANSITION = ((0.8, 0.2), (0.1, 0.9))  # rows: bad, good


@categorical(ordered=False)
class EzRegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class HealthStatus:
    bad: ScalarInt
    good: ScalarInt


def utility_alive(consumption: ContinuousAction) -> FloatND:
    return consumption


def utility_dead(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def H_epstein_zin(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND, rho: FloatND
) -> FloatND:
    return (
        (1.0 - discount_factor) * utility**rho + discount_factor * E_next_V**rho
    ) ** (1.0 / rho)


def next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction, income: FloatND
) -> ContinuousState:
    return wealth - consumption + income


def health_probs(health: DiscreteState) -> FloatND:
    return jnp.where(
        health == HealthStatus.good,
        jnp.array(HEALTH_TRANSITION[1]),
        jnp.array(HEALTH_TRANSITION[0]),
    )


def next_regime(
    health: DiscreteState, period: Period, survival_probs: Float1D
) -> FloatND:
    sp = survival_probs[period] * jnp.where(
        health == HealthStatus.good, 1.0, BAD_HEALTH_SURVIVAL_FACTOR
    )
    return jnp.array([sp, 1.0 - sp])


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def get_model(*, certainty_equivalent: CertaintyEquivalent | None = None) -> Model:
    alive = UserRegime(
        transition=MarkovTransition(next_regime),
        states={
            "wealth": WEALTH_GRID,
            "health": DiscreteGrid(HealthStatus),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": {"alive": MarkovTransition(health_probs)},
        },
        actions={"consumption": CONSUMPTION_GRID},
        constraints={"budget_constraint": budget_constraint},
        functions={"utility": utility_alive, "H": H_epstein_zin},
        certainty_equivalent=certainty_equivalent,
        active=lambda age: age < 63,
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": DEAD_WEALTH_GRID},
        functions={"utility": utility_dead},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=60, stop=63, step="Y"),
        regime_id_class=EzRegimeId,
    )


def get_params(
    *,
    risk_aversion: float | None,
    discount_factor: float = 0.9,
    rho: float = 0.5,
) -> dict:
    params: dict = {
        "alive": {
            "H": {"discount_factor": discount_factor, "rho": rho},
            "next_wealth": {"income": INCOME},
            "next_regime": {"survival_probs": jnp.array(SURVIVAL_PROBS)},
        },
    }
    if risk_aversion is not None:
        params["alive"]["certainty_equivalent"] = {"risk_aversion": risk_aversion}
    return params
