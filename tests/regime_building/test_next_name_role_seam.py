"""When `next_<state>` is a law's output and when it is an ordinary parameter.

`next_<state>` names next-period vocabulary only where a next-period value
exists. A transition, and anything feeding one, always has one. For everything
evaluated within the period — utility, its helpers, constraints — the regime's
own law decides, and only a law that is both deterministic and
target-independent counts:

- a bare law is what this period's states and actions determine on their own,
  so the decision evaluation can read it. This is the NEGM/DC-EGM durable
  pattern: the service flow accrues from the newly chosen stock, and the budget
  constraint bounds it;
- a per-target law is a handover value, not well defined until the destination
  is known — the same ground on which the decision evaluation refuses a
  `next_<state>` whose law differs across targets;
- a `MarkovTransition` names a draw that has not been realised yet;
- a state this regime does not move at all has no law here, so an argument
  spelled that way is a parameter whatever else the model does with the name.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarFloat,
    ScalarInt,
)

_WEALTH = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
_CONSUMPTION = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _to_dead(period: ScalarInt) -> ScalarInt:
    return jnp.where(period >= 0, RegimeId.dead, RegimeId.alive)


def _utility_reading_the_law(
    consumption: ContinuousAction, next_wealth: ContinuousState
) -> FloatND:
    """`next_wealth` here is the bare law's output, at the chosen action."""
    return consumption + next_wealth


def _utility_reading_a_parameter(
    consumption: ContinuousAction, next_wealth: ScalarFloat
) -> FloatND:
    """`next_wealth` here is a parameter: the law is per-target."""
    return consumption + next_wealth


def _constraint_reading_the_law(next_wealth: ContinuousState) -> BoolND:
    return next_wealth >= 0.0


def _dead_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _build(*, functions: dict, constraints: dict, state_transitions: dict) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                transition=_to_dead,
                active=lambda age: age < 21,
                states={"wealth": _WEALTH},
                actions={"consumption": _CONSUMPTION},
                state_transitions=state_transitions,
                functions=functions,
                constraints=constraints,
            ),
            "dead": Regime(
                transition=None,
                active=lambda age: age >= 21,
                states={"wealth": _WEALTH},
                functions={"utility": _dead_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.fixture
def bare_law_model() -> Model:
    return _build(
        functions={"utility": _utility_reading_the_law},
        constraints={"affordable": _constraint_reading_the_law},
        state_transitions={"wealth": _next_wealth},
    )


@pytest.fixture
def per_target_law_model() -> Model:
    return _build(
        functions={"utility": _utility_reading_a_parameter},
        constraints={},
        state_transitions={"wealth": {"dead": _next_wealth}},
    )


def test_a_bare_law_serves_utility(bare_law_model: Model) -> None:
    """Utility reading the durable-style law asks for no parameter."""
    assert bare_law_model.get_params_template()["alive"]["utility"] == {}


def test_a_bare_law_serves_a_constraint(bare_law_model: Model) -> None:
    """A budget constraint cutting on the next stock asks for no parameter."""
    assert bare_law_model.get_params_template()["alive"]["affordable"] == {}


def test_a_per_target_law_leaves_utility_asking_for_the_parameter(
    per_target_law_model: Model,
) -> None:
    """A handover law is not a within-period value, so the argument stays a param."""
    assert per_target_law_model.get_params_template()["alive"]["utility"] == {
        "next_wealth": "ScalarFloat"
    }
