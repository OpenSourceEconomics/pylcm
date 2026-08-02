"""Temporal reachability governs target-only stochastic-process validation."""

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
    return jnp.log(consumption) + shock


def _next_wealth(wealth, consumption):
    return wealth - consumption


def _leaves(wealth, age):
    return (wealth >= 3.0) | (age >= _LAST_AGE - 1)


def _next_regime(wealth, age):
    return jnp.where(_leaves(wealth, age), RegimeId.gone, RegimeId.alive)


def _build(
    target_states,
    source_states=None,
    *,
    coarse=False,
    source_active=lambda age: age < _LAST_AGE,
    target_active=lambda _age: True,
):
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
        active=source_active,
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
        active=target_active,
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
def test_activity_compatible_granular_target_only_process_is_rejected(process):
    with pytest.raises(ModelInitializationError, match=r"gone.*shock|shock.*gone"):
        _build(target_states={"shock": process})


def test_activity_compatible_coarse_target_only_process_is_rejected():
    """Coarse candidacy is CONDITIONAL support and is validated, not skipped."""
    with pytest.raises(ModelInitializationError, match=r"gone.*shock|shock.*gone"):
        _build(
            target_states={"shock": TauchenAR1Process(n_points=3, gauss_hermite=False)},
            coarse=True,
        )


@pytest.mark.parametrize("coarse", [False, True])
def test_temporally_disjoint_target_only_process_is_accepted(coarse):
    """A source active only after its target is not connected to that target."""
    model = _build(
        target_states={"shock": TauchenAR1Process(n_points=3, gauss_hermite=False)},
        coarse=coarse,
        source_active=lambda age: age >= 21,
        target_active=lambda age: age < 21,
    )
    assert "gone" in model.user_regimes


def test_process_carried_by_source_and_target_is_accepted():
    process = TauchenAR1Process(n_points=3, gauss_hermite=False)
    model = _build(target_states={"shock": process}, source_states={"shock": process})
    assert "gone" in model.user_regimes
