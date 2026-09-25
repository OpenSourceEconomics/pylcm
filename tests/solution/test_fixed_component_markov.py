"""A Markov state declaring a fixed component solves as the hand-split model does.

The toy state `kind_health` codes (kind, health) as `2 * kind + health`; the law keeps
`kind` and moves `health`. Declaring `fixed_component` must give the value function
of the model in which `kind` is its own identity-law state, and carry the state as a
group axis and a within-group axis instead of one axis over every code.
"""

import re

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_distributed import _compiled_solve_kernel_hlo


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _KindHealth:
    k0_h0: ScalarInt
    k0_h1: ScalarInt
    k1_h0: ScalarInt
    k1_h1: ScalarInt


@categorical(ordered=False)
class _Kind:
    k0: ScalarInt
    k1: ScalarInt


@categorical(ordered=False)
class _Health:
    h0: ScalarInt
    h1: ScalarInt


_HEALTH_LAW = jnp.array([[[0.9, 0.1], [0.3, 0.7]], [[0.6, 0.4], [0.2, 0.8]]])


def _utility(
    *,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    kind_health: DiscreteState,
) -> FloatND:
    return jnp.log(consumption) + 0.1 * kind_health + 0.01 * wealth


def _next_kind_health(kind_health: DiscreteState) -> FloatND:
    kind, health = kind_health // 2, kind_health % 2
    within = _HEALTH_LAW[kind, health]
    return jnp.where(jnp.arange(4) // 2 == kind, within[jnp.arange(4) % 2], 0.0)


def _next_health(*, kind: DiscreteState, health: DiscreteState) -> FloatND:
    return _HEALTH_LAW[kind, health]


def _kind_health(*, kind: DiscreteState, health: DiscreteState) -> DiscreteState:
    return 2 * kind + health


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 2, _RegimeId.alive, _RegimeId.dead)


def _model(*, factored: bool, fixed_component: tuple[int, ...] = (0, 0, 1, 1)) -> Model:
    if factored:
        states = {"kind_health": DiscreteGrid(_KindHealth)}
        laws = {
            "kind_health": MarkovTransition(
                _next_kind_health, fixed_component=fixed_component
            )
        }
        functions = {}
    else:
        states = {"health": DiscreteGrid(_Health), "kind": DiscreteGrid(_Kind)}
        laws = {
            "health": MarkovTransition(_next_health),
            "kind": fixed_transition("kind"),
        }
        functions = {"kind_health": _kind_health}
    return Model(
        regimes={
            "alive": Regime(
                active=lambda age: age < 3,
                transition=_next_regime,
                states={
                    "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
                    **states,
                },
                actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
                functions={"utility": _utility, **functions},
                constraints={"feasible": _feasible},
                state_transitions={
                    "wealth": _next_wealth,
                    **laws,
                },
            ),
            "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
        },
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_RegimeId,
    )


def _values(model: Model) -> dict:
    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    return {
        (period, regime): np.asarray(v)
        for period, by_regime in solution.values.items()
        for regime, v in by_regime.items()
    }


def test_fixed_component_carries_group_and_position_as_separate_axes():
    """The alive value function has a position axis and a group axis, not 4 codes."""
    values = _values(_model(factored=True))
    assert values[(0, "alive")].shape == (2, 2, 5)


def test_fixed_component_solve_equals_the_hand_split_model():
    """Declaring the fixed component reproduces the hand-split value functions."""
    factored, split = _values(_model(factored=True)), _values(_model(factored=False))
    assert all(np.array_equal(factored[key], split[key]) for key in split)


def test_fixed_component_rejects_unequal_groups():
    """Groups of different sizes cannot share one within-group axis."""
    with pytest.raises(RegimeInitializationError, match="equal size"):
        _model(factored=True, fixed_component=(0, 0, 0, 1))


def _gather_shapes(model: Model) -> set[str]:
    hlo = _compiled_solve_kernel_hlo(model=model, regime_name="alive", period=0)
    return set(re.findall(r"= (\w+\[[\d,]*\])[^\n]*? gather\(", hlo))


def test_fixed_component_lowers_the_hand_split_gathers():
    """The optimized kernel reads next-period values one group at a time.

    Every gather of the hand-split kernel, including the continuation read whose
    group axis is a size-1 slice, appears in the kernel of the annotated model.
    """
    split = _gather_shapes(_model(factored=False))
    assert split <= _gather_shapes(_model(factored=True))
