"""A within-period decision may not read an unrealised stochastic next state.

`_get_deterministic_transitions` deliberately excludes a stochastic `next_<state>`
from the flow DAG (its value is not known at choice time). Without a guard, `dags`
leaves it an unresolved external Q argument that fails much later with a confusing
missing-argument error -- and only in the phase where the law is stochastic. The
build must reject it early and clearly.

Mixed stochasticity makes the phase matter: the same state can be readable in the
phase where its law is deterministic and rejected in the phase where it is
stochastic.
"""

from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.typing import DiscreteAction, FloatND, Period, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Good:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def _utility_reads_next(next_good: FloatND, good: DiscreteAction) -> FloatND:
    """Within-period utility reads the CHOSEN next state (service-flow pattern)."""
    return 1.0 * next_good + 0.0 * good


def _flat_utility(good: DiscreteAction, move: DiscreteAction) -> FloatND:
    return 0.0 * good + 0.0 * move


def _point_mass(to_good: FloatND) -> FloatND:
    return jnp.stack([1.0 - to_good, to_good], axis=-1)


def _markov(move: DiscreteAction) -> FloatND:
    return _point_mass(jnp.where(move == Move.stay, 1.0, 0.0))


def _deterministic(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Good.good, Good.bad)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "good": ["bad"] * 8})


def _build_and_solve(*, live_functions, law: Any) -> None:
    common_states = {"good": DiscreteGrid(Good)}
    common_actions = {"move": DiscreteGrid(Move)}
    live = Regime(
        transition=_next_regime,
        state_transitions={"good": law},
        states=common_states,
        actions=common_actions,
        functions=live_functions,
    ).replace(active=lambda age: age < 2)
    last = Regime(
        transition=None,
        state_transitions={},
        states=common_states,
        actions=common_actions,
        functions={"utility": _flat_utility},
    ).replace(active=lambda age: age >= 2)
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="stochastic flow-read rejection",
    )
    model.solve(params=PARAMS, log_level="off")


def test_utility_reading_a_stochastic_next_state_is_rejected():
    """A bare `MarkovTransition` state read by utility is stochastic in BOTH phases."""
    with pytest.raises(ValueError, match="stochastic state transition"):
        _build_and_solve(
            live_functions={"utility": _utility_reads_next},
            law=MarkovTransition(_markov),
        )


def test_utility_reading_next_state_stochastic_only_in_simulate_is_rejected():
    """Deterministic in solve, stochastic in simulate: the simulate-flow read fails."""
    with pytest.raises(ValueError, match="stochastic state transition"):
        _build_and_solve(
            live_functions={"utility": _utility_reads_next},
            law=Phased(solve=_deterministic, simulate=MarkovTransition(_markov)),
        )
