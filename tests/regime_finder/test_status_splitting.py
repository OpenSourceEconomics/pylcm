"""Status models where some statuses need fewer states than others.

In the one-status model, the unemployed neither read experience nor carry it
forward. In the two-status model, only the employed and healthy need experience,
and health matters only to the employed. The layout-search and facts-backend
tests build on these models.
"""

import jax.numpy as jnp

from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime, categorical
from lcm.typing import DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class Employment:
    unemployed: ScalarInt
    employed: ScalarInt


@categorical(ordered=False)
class Health:
    unhealthy: ScalarInt
    healthy: ScalarInt


@categorical(ordered=False)
class Experience:
    none: ScalarInt
    some: ScalarInt
    much: ScalarInt


@categorical(ordered=False)
class Regimes:
    working_age: ScalarInt
    final: ScalarInt


def wage(experience: DiscreteState, employment: DiscreteState) -> FloatND:
    """Only the employed earn from experience."""
    return jnp.where(employment == Employment.employed, 1.0 + experience, 0.0)


def next_experience(
    experience: DiscreteState, employment: DiscreteState
) -> DiscreteState:
    """Unemployment resets experience."""
    return jnp.where(employment == Employment.employed, experience, Experience.none)


def next_employment(employment: DiscreteState) -> DiscreteState:
    return employment


def joint_wage(
    experience: DiscreteState, employment: DiscreteState, health: DiscreteState
) -> FloatND:
    """Experience pays only the employed and healthy; health matters only at work."""
    at_work = jnp.where(health == Health.healthy, 1.0 + experience, 0.0)
    return jnp.where(employment == Employment.employed, at_work, 0.0)


def joint_next_experience(
    experience: DiscreteState, employment: DiscreteState, health: DiscreteState
) -> DiscreteState:
    """Experience is kept only by the employed and healthy."""
    at_work = jnp.where(health == Health.healthy, experience, Experience.none)
    return jnp.where(employment == Employment.employed, at_work, Experience.none)


def next_health(health: DiscreteState) -> DiscreteState:
    return health


def _model(states, utility, transitions) -> Model:
    return Model(
        n_periods=2,
        regime_id_class=Regimes,
        regimes={
            "working_age": Regime(
                states=states,
                functions={"utility": utility},
                state_transitions=transitions,
            ),
            "final": Regime(states=states, functions={"utility": utility}),
        },
        initial_nodes=InitialNodes(by_period={0: "working_age"}),
        edges={"working_age": {"final": Periods(values=(0,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )


def model(*, status_matters: bool = True) -> Model:
    return _model(
        {
            "employment": DiscreteGrid(Employment),
            "experience": DiscreteGrid(Experience),
        },
        wage if status_matters else _always_experience,
        {
            "experience": next_experience if status_matters else _carry,
            "employment": next_employment,
        },
    )


def joint_model() -> Model:
    return _model(
        {
            "employment": DiscreteGrid(Employment),
            "health": DiscreteGrid(Health),
            "experience": DiscreteGrid(Experience),
        },
        joint_wage,
        {
            "experience": joint_next_experience,
            "employment": next_employment,
            "health": next_health,
        },
    )


def _always_experience(experience: DiscreteState, employment: DiscreteState) -> FloatND:
    return 1.0 + experience + 0 * employment


def _carry(experience: DiscreteState, employment: DiscreteState) -> DiscreteState:
    return experience + 0 * employment
