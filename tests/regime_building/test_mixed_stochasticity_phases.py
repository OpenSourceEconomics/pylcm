"""A state law may be stochastic in one phase and deterministic in the other.

The two phases need not agree on whether a state's law is stochastic: a deterministic
law is a degenerate kernel, not a different kind of state. An agent may perceive risk
where there is none, or believe a transition certain that in fact is not. Both
directions — `Phased(solve=MarkovTransition(...), simulate=<deterministic>)` and its
reverse — build, solve, and simulate with the belief/truth split intact: Q is priced
under the belief, the draw follows the truth.
"""

from typing import Any

import jax.numpy as jnp
import pandas as pd

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


def utility(*, good: DiscreteAction, move: DiscreteAction) -> FloatND:
    return 1.0 * good + 0.0 * move


def _point_mass(to_good: FloatND) -> FloatND:
    return jnp.stack([1.0 - to_good, to_good], axis=-1)


def markov_belief(move: DiscreteAction) -> FloatND:
    """BELIEF as a degenerate kernel: `stay` -> good."""
    return _point_mass(jnp.where(move == Move.stay, 1.0, 0.0))


def markov_actual(move: DiscreteAction) -> FloatND:
    """TRUTH as a degenerate kernel: `stay` -> bad."""
    return _point_mass(jnp.where(move == Move.stay, 0.0, 1.0))


def deterministic_belief(move: DiscreteAction) -> FloatND:
    """The SAME belief, written as a point value: `stay` -> good."""
    return jnp.where(move == Move.stay, Good.good, Good.bad)


def deterministic_actual(move: DiscreteAction) -> FloatND:
    """The SAME truth, written as a point value: `stay` -> bad."""
    return jnp.where(move == Move.stay, Good.bad, Good.good)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "good": ["bad"] * 8})


def _simulate(law: Any) -> pd.DataFrame:
    common: dict[str, Any] = {
        "states": {"good": DiscreteGrid(category_class=Good)},
        "actions": {"move": DiscreteGrid(category_class=Move)},
        "functions": {"utility": utility},
    }
    live = Regime(
        transition=_next_regime, state_transitions={"good": law}, **common
    ).replace(active=lambda age: age < 2)
    last = Regime(transition=None, state_transitions={}, **common).replace(
        active=lambda age: age >= 2
    )
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="mixed stochasticity probe",
    )
    V = model.solve(params=PARAMS, log_level="debug")
    return (
        model.simulate(
            params=PARAMS,
            initial_conditions=IC,
            period_to_regime_to_V_arr=V,
            log_level="debug",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )


def test_stochastic_solve_deterministic_simulate():
    """Perceived law is a kernel; the world realizes a point value."""
    df = _simulate(
        Phased(solve=MarkovTransition(markov_belief), simulate=deterministic_actual)
    )
    assert (df[df["period"] == 0]["move"] == "stay").all(), "Q must price under BELIEF"
    assert (df[df["period"] == 1]["good"] == "bad").all(), "draw must follow TRUTH"


def test_deterministic_solve_stochastic_simulate():
    """Perceived law is a point value; the world realizes from a kernel."""
    df = _simulate(
        Phased(solve=deterministic_belief, simulate=MarkovTransition(markov_actual))
    )
    assert (df[df["period"] == 0]["move"] == "stay").all(), "Q must price under BELIEF"
    assert (df[df["period"] == 1]["good"] == "bad").all(), "draw must follow TRUTH"
