"""A simulate-phase law may read a sibling's realized draw.

During simulation a stochastic transition has drawn, and its realization is published
under its `next_<state>` name. A deterministic law of the same target reads it like any
other producer. During backward induction the draw has no value, so the same read is
rejected — which is why the dependence is declared per phase.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _health_probs(health: DiscreteState) -> FloatND:
    """Always transition to `good`, so the realized draw is known."""
    del health
    return jnp.array([0.0, 1.0])


def _wealth_utility(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _wealth_and_health_utility(
    wealth: ScalarFloat, health: DiscreteState
) -> ScalarFloat:
    return wealth + health.astype(jnp.float32)


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _wealth_from_realized_health(next_health: DiscreteState) -> ScalarFloat:
    """In simulation, next wealth is three times the realized health code."""
    return 3.0 * next_health.astype(jnp.float32)


@pytest.fixture
def model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(lambda: jnp.float32(1))},
                active=lambda age: age < 22,
                states={"wealth": _WEALTH, "health": DiscreteGrid(Health)},
                state_transitions={
                    "health": MarkovTransition(_health_probs),
                    "wealth": Phased(
                        solve=_keep_wealth, simulate=_wealth_from_realized_health
                    ),
                },
                functions={"utility": _wealth_utility},
            ),
            "target": Regime(
                transition=None,
                states={"wealth": _WEALTH, "health": DiscreteGrid(Health)},
                functions={"utility": _wealth_and_health_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_the_model_builds_and_solves(model: Model) -> None:
    """The solve phase never reads the draw, so backward induction is unaffected."""
    V = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    )

    assert not np.isnan(np.asarray(V[0]["source"])).any()


def test_simulated_wealth_follows_the_realized_draw(model: Model) -> None:
    """Health always draws `good` (code one), so next wealth is three."""
    result = model.simulate(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        initial_conditions={
            "age": jnp.array([20.0, 20.0]),
            "wealth": jnp.array([0.0, 4.0]),
            "health": jnp.array([Health.bad, Health.bad]),
            "regime_id": jnp.array([RegimeId.source, RegimeId.source]),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False).reset_index()

    np.testing.assert_allclose(
        df.query("period == 1")["wealth"].to_numpy(),
        np.array([3.0, 3.0]),
        atol=1e-5,
    )
