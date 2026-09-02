"""Simulating a regime with nowhere to go says so, in pylcm's own vocabulary.

Declaring a transition target that is never active in an adjacent period is
legal — a declared-but-unreachable edge retains no period and needs no state
handoff. What is not possible is simulating a subject that stands in such a
regime at a period it cannot leave: there is no target to draw from. That draw
is refused by name, rather than reaching jax with an empty set of candidates.

The refusal is not one of the runtime validations `log_level` governs, so it
holds at `"off"` too: an empty candidate set makes the draw itself impossible,
not merely suspect.
"""

import re

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import ContinuousState, FloatND, ScalarInt

_AGES = AgeGrid(start=0, stop=3, step="Y")
_WEALTH = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

# `dead` becomes active only at the last age, so `retired` — active from age 1 —
# has no target to enter at age 2.
_DEAD_ACTIVE_FROM = 3.0


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the worker / retired / dead model in this module."""

    worker: ScalarInt  # code 0
    retired: ScalarInt  # code 1
    dead: ScalarInt  # code 2


def test_simulating_a_regime_with_no_active_target_names_the_regime_and_period():
    """A subject retired at age 1 has nowhere to go at age 2, and is told so."""
    model = _build_model()
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=re.escape(
            "Regime 'retired' has no regime to move into at period 2: none of "
            "its declared transition targets ('dead') is active there"
        ),
    ):
        model.simulate(
            params={"discount_factor": 0.9},
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="off",
        )


def _build_model() -> Model:
    """Build a worker who retires into a regime that cannot be left."""
    worker = Regime(
        transition={"retired": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": {"retired": _keep_wealth}},
        functions={"utility": _utility},
    )
    retired = Regime(
        transition={"dead": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 3),
        states={"wealth": _WEALTH},
        state_transitions={"wealth": {"dead": _keep_wealth}},
        functions={"utility": _utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= _DEAD_ACTIVE_FROM,
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )
    return Model(
        regimes={"worker": worker, "retired": retired, "dead": dead},
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def _initial_conditions() -> dict[str, FloatND]:
    """Seed every subject as a worker at age 0."""
    return {
        "age": jnp.zeros(2),
        "wealth": jnp.full(2, 1.0),
        "regime_id": jnp.full(2, RegimeId.worker, dtype=jnp.int32),
    }


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with certainty."""
    return jnp.ones_like(age, dtype=float)


def _keep_wealth(wealth: ContinuousState) -> ContinuousState:
    """Wealth carries over unchanged into the target regime."""
    return wealth


def _utility(wealth: ContinuousState) -> FloatND:
    """Every regime pays out its wealth."""
    return wealth
