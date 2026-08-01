"""A reachable target may not carry a stochastic process its source lacks.

A stochastic process supplies its own transition, and that transition is a law
from *this* period's value of the process. So a source regime can only integrate
a target's process out if the source itself carries it and can supply the
from-value. When the target declares a process the source does not, there is no
from-value to condition on and no entry law is defined.

Such a model is rejected when it builds, rather than solved with the target's
continuation silently omitted. The message names the target, the process, and
the two ways out: declare the process on the source as well, or give the target
a non-process state.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    TauchenAR1Process,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ScalarInt

_LAST_AGE = 22


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def _utility(consumption):
    return jnp.log(consumption)


def _utility_with_shock(consumption, shock):
    """Read `shock` so a source-declared process counts as used."""
    return jnp.log(consumption) + shock


def _next_wealth(wealth, consumption):
    return wealth - consumption


def _leaves(wealth, age):
    return (wealth >= 3.0) | (age >= _LAST_AGE - 1)


def _next_regime(wealth, age):
    return jnp.where(_leaves(wealth, age), RegimeId.gone, RegimeId.alive)


def _build(target_states, source_states=None, *, coarse=False):
    # A per-target transition declares its support exactly, which is what makes
    # the check decidable; `coarse=True` exercises the undecidable case.
    transition = (
        _next_regime
        if coarse
        else {
            "alive": MarkovTransition(lambda wealth, age: 1.0 - _leaves(wealth, age)),
            "gone": MarkovTransition(lambda wealth, age: 1.0 * _leaves(wealth, age)),
        }
    )
    alive = Regime(
        transition=transition,
        active=lambda age: age < _LAST_AGE,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=4),
            **(source_states or {}),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility_with_shock if source_states else _utility},
    )
    gone = Regime(
        transition=None,
        states=target_states,
        functions={"utility": lambda shock: shock},
    )
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize(
    "process",
    [
        TauchenAR1Process(n_points=3, gauss_hermite=False),
        NormalIIDProcess(n_points=3, gauss_hermite=False),
    ],
    ids=["ar1", "iid"],
)
def test_a_target_only_process_is_rejected_at_build(process):
    """A target carrying a process the source lacks is refused, not silently dropped."""
    with pytest.raises(ModelInitializationError, match=r"gone.*shock|shock.*gone"):
        _build(target_states={"shock": process})


def test_a_process_carried_by_both_source_and_target_is_accepted():
    """Declaring the process on the source too is the documented way out."""
    process = TauchenAR1Process(n_points=3, gauss_hermite=False)
    model = _build(target_states={"shock": process}, source_states={"shock": process})
    assert "gone" in model.user_regimes


def test_a_coarse_transition_does_not_reject_a_target_only_process():
    """A coarse transition's support is unknown, so candidacy alone cannot reject.

    `transition=func` picks its target at runtime, so every regime is only a
    *candidate* and most are never returned. Refusing a model on candidacy would
    reject working ones — an absorbing regime names every other regime as a
    candidate and goes to none of them. This half of the class stays open by
    design rather than be closed with false positives.
    """
    model = _build(
        target_states={"shock": TauchenAR1Process(n_points=3, gauss_hermite=False)},
        coarse=True,
    )
    assert "gone" in model.user_regimes
