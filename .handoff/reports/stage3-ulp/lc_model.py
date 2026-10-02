"""The Stage 5A life-cycle model (copied from tests/simulation/test_type_grouped_simulation.py @ ad7a443)."""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import BoolND, ContinuousAction, ContinuousState, DiscreteState, FloatND, ScalarInt
from tests.test_models.schedules import until_exit

_LAST_AGE = 4


@categorical(ordered=False)
class _PrefType:
    patient: ScalarInt
    average: ScalarInt
    impatient: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    dead: ScalarInt


def _work_utility(*, consumption: ContinuousAction, pref_type: DiscreteState, health: DiscreteState, weight: FloatND) -> FloatND:
    return weight[pref_type] * jnp.log(consumption) + 0.2 * health


def _typed_bequest(*, wealth: ContinuousState, pref_type: DiscreteState, bequest: FloatND) -> FloatND:
    return bequest[pref_type] * jnp.log(wealth)


def _type_free_bequest(*, wealth: ContinuousState) -> FloatND:
    return 0.7 * jnp.log(wealth)


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> ContinuousState:
    return wealth - consumption + 2.0


def _next_health(*, health: DiscreteState, pref_type: DiscreteState) -> FloatND:
    stay = 0.6 + 0.1 * pref_type
    return jnp.where(jnp.arange(2) == health, stay, 1.0 - stay)


def _alive(*, pref_type: DiscreteState, health: DiscreteState, age: float) -> FloatND:
    alive = 0.9 - 0.15 * pref_type - 0.1 * (1 - health)
    return jnp.where(age < _LAST_AGE - 1, alive, 0.0)


def _survival(*, pref_type: DiscreteState, health: DiscreteState, age: float) -> FloatND:
    alive = _alive(pref_type=pref_type, health=health, age=age)
    return jnp.array([alive, 1.0 - alive])


def model(*, typed_dead: bool, blocked: bool, extra: dict | None = None) -> Model:
    wealth = LinSpacedGrid(start=1, stop=10, n_points=6)
    pref_type = DiscreteGrid(_PrefType)
    consumption = {"consumption": LinSpacedGrid(start=1, stop=3, n_points=5)}
    regimes = {
        "work": Regime(
            regime_transitions=until_exit(
                _LAST_AGE,
                law=MarkovTransition(func=_survival, targets=("work", "dead")),
                exits=("dead",),
            ),
            states={"wealth": wealth, "pref_type": pref_type, "health": DiscreteGrid(_Health)},
            state_transitions={
                "wealth": _next_wealth,
                "pref_type": fixed_transition("pref_type"),
                "health": MarkovTransition(func=_next_health),
            },
            actions=consumption,
            functions={"utility": _work_utility},
            constraints={"feasible": _feasible},
        ),
        "dead": Regime(
            regime_transitions=None,
            states={"wealth": wealth, "pref_type": pref_type} if typed_dead else {"wealth": wealth},
            functions={"utility": _typed_bequest if typed_dead else _type_free_bequest},
        ),
    }
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, stop=_LAST_AGE, step="Y"),
        regime_id_class=_RegimeId,
        initial_regimes={0: "work"},
        execution_config=ExecutionConfig(
            invariant_block_widths={"pref_type": 1} if blocked else {},
            **{"axis_widths": {"subject": 3}, **(extra or {})},
        ),
    )


def params(*, typed_dead: bool) -> dict:
    return {
        "discount_factor": 0.9,
        "work": {"utility": {"weight": jnp.asarray([1.0, 1.4, 0.7])}},
        "dead": {"utility": {"bequest": jnp.asarray([0.4, 1.1, 2.3])}} if typed_dead else {},
    }
