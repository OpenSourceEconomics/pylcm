"""Which consumers of `next_<state>` are served by a law, and which by a parameter.

`next_<state>` names next-period vocabulary only where a next-period value
exists. A transition, and anything feeding one, always has one. Beyond that the
two consumers evaluated within the period differ:

- a **constraint** may cut on a next state the chosen action determines — the
  NEGM/DC-EGM durable pattern, where the budget constraint bounds the next
  durable stock — so a deterministic own-regime law for that state serves it;
- **utility** and ordinary functions are evaluated at this period's states, so
  an argument spelled that way there is a parameter the user supplies, whether
  or not the regime happens to move a state of that name.

Nothing decides this for the engine: the parameter template decides, and the
decision DAG follows, since a name bound as a parameter never becomes a free
argument the DAG could satisfy from a law.
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


def _utility_reading_a_parameter(
    consumption: ContinuousAction, next_wealth: ScalarFloat
) -> FloatND:
    """`next_wealth` here is a parameter, not the law's output."""
    return consumption + next_wealth


def _utility(consumption: ContinuousAction) -> FloatND:
    return consumption


def _constraint_reading_the_law(next_wealth: ContinuousState) -> BoolND:
    """`next_wealth` here is the law's output, evaluated at the chosen action."""
    return next_wealth >= 0.0


def _dead_utility() -> FloatND:
    return jnp.array(0.0)


def _build(*, functions: dict, constraints: dict) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                transition=_to_dead,
                active=lambda age: age < 21,
                states={"wealth": _WEALTH},
                actions={"consumption": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
                state_transitions={"wealth": _next_wealth},
                functions=functions,
                constraints=constraints,
            ),
            "dead": Regime(
                transition=None,
                active=lambda age: age >= 21,
                functions={"utility": _dead_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.fixture
def model_with_constraint() -> Model:
    return _build(
        functions={"utility": _utility},
        constraints={"affordable": _constraint_reading_the_law},
    )


@pytest.fixture
def model_with_utility_parameter() -> Model:
    return _build(functions={"utility": _utility_reading_a_parameter}, constraints={})


def test_a_constraint_reading_a_next_state_asks_for_no_parameter(
    model_with_constraint: Model,
) -> None:
    """The declared law serves the constraint, so nothing is left to supply."""
    assert model_with_constraint.get_params_template()["alive"]["affordable"] == {}


def test_utility_reading_a_next_name_still_asks_for_the_parameter(
    model_with_utility_parameter: Model,
) -> None:
    """Declaring a wealth law does not turn utility's `next_wealth` into it."""
    assert model_with_utility_parameter.get_params_template()["alive"]["utility"] == {
        "next_wealth": "ScalarFloat"
    }
