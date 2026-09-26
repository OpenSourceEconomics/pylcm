"""Simulation rejects an unknown initial state whatever the log level.

A mapping that names a state the model does not have would otherwise simulate from
states that were never supplied. The DataFrame form rejects it at every log level;
the mapping form must too.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import InvalidInitialConditionsError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + 0.1 * health


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2


def _next_health(health: DiscreteState) -> DiscreteState:
    return health


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 1, _RegimeId.alive, _RegimeId.dead)


_MODEL = Model(
    regimes={
        "alive": Regime(
            active=lambda age: age < 2,
            transition=_next_regime,
            states={
                "wealth": LinSpacedGrid(start=1, stop=10, n_points=4),
                "health": DiscreteGrid(_Health),
            },
            actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
            functions={"utility": _utility},
            constraints={"feasible": _feasible},
            state_transitions={"wealth": _next_wealth, "health": _next_health},
        ),
        "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
    },
    ages=AgeGrid(start=0, stop=3, step="Y"),
    regime_id_class=_RegimeId,
)
_PARAMS = {"discount_factor": 0.95}


def _simulate(initial_conditions: dict) -> None:
    _MODEL.simulate(
        params=_PARAMS,
        solution=_MODEL.solve(params=_PARAMS, log_level="off"),
        initial_conditions=initial_conditions,
        seed=1,
        log_level="off",
    )


def test_simulate_rejects_an_unknown_state_with_logging_off():
    """A state no regime has raises and names the key, as a DataFrame column would."""
    with pytest.raises(
        InvalidInitialConditionsError, match=r"Unknown initial states: \['hlth'\]"
    ):
        _simulate(
            {
                "wealth": jnp.ones(2),
                "health": jnp.zeros(2, int),
                "hlth": jnp.ones(2),
                "age": jnp.zeros(2),
                "regime_id": jnp.full(2, _RegimeId.alive),
            }
        )
