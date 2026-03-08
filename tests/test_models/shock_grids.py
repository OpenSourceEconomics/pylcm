from typing import Literal

from jax import numpy as jnp

import lcm
from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid, LinSpacedGrid, categorical
from lcm.model import Model
from lcm.regime import MarkovTransition, Regime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

_SHOCK_GRID_CLASSES = {
    "uniform": lcm.shocks.iid.Uniform,
    "normal": lcm.shocks.iid.Normal,
    "lognormal": lcm.shocks.iid.LogNormal,
    "tauchen": lcm.shocks.ar1.Tauchen,
    "rouwenhorst": lcm.shocks.ar1.Rouwenhorst,
}

_SHOCK_GRID_KWARGS: dict[str, dict[str, bool]] = {
    "uniform": {},
    "normal": {"gauss_hermite": True},
    "lognormal": {"gauss_hermite": True},
    "tauchen": {"gauss_hermite": True},
    "rouwenhorst": {},
}


def next_health(health: DiscreteState, probs_array: FloatND) -> FloatND:
    return probs_array[health]


def next_wealth(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return wealth - consumption


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.alive,
    )


def wealth_constraint(
    wealth: ContinuousState, income: ContinuousState, consumption: ContinuousAction
):
    return wealth - consumption + jnp.exp(income) >= 0


def utility(
    wealth: ContinuousState,  # noqa: ARG001
    income: ContinuousState,  # noqa: ARG001
    health: DiscreteState,
    consumption: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) * (1.0 - (1.0 - health) * 0.3)


@categorical
class Health:
    bad: int = 0
    good: int = 1


@categorical
class RegimeId:
    alive: int
    dead: int


def get_model(
    n_periods: int,
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ],
):
    final_age_alive = n_periods - 2

    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=5),
            "income": _SHOCK_GRID_CLASSES[distribution_type](
                n_points=5, **_SHOCK_GRID_KWARGS[distribution_type]
            ),
            "health": DiscreteGrid(Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(next_health),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4),
        },
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


_SHOCK_PARAMS: dict[str, dict[str, float]] = {
    "uniform": {"start": 0.0, "stop": 1.0},
    "normal": {"mu": 0.0, "sigma": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "tauchen": {"rho": 0.975, "sigma": 1.0, "mu": 0.0},
    "rouwenhorst": {"rho": 0.975, "sigma": 1.0, "mu": 0.0},
}


def get_params(
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ] = "tauchen",
):
    return {
        "alive": {
            "discount_factor": 1,
            "next_health": {"probs_array": jnp.full((2, 2), fill_value=0.5)},
            "income": _SHOCK_PARAMS[distribution_type],
        },
        "dead": {},
    }
