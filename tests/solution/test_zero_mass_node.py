"""A stochastic node reached with zero probability contributes nothing.

A `MarkovTransition` row may place zero mass on one node of a target's lottery
while the other nodes carry the whole distribution. That node is never reached,
so its value cannot matter — and `-inf`, the value of a state at which every
action is infeasible, is exactly the value such an unreachable node tends to
carry.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, MarkovTransition, Model, Regime, categorical
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _Health:
    frail: ScalarInt
    hale: ScalarInt


@categorical(ordered=False)
class _Spend:
    little: ScalarInt
    lots: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    last: ScalarInt


def _health_probs(health: DiscreteState) -> FloatND:
    """Health persists with certainty, so the other node carries exactly zero."""
    return jnp.identity(2)[health]


def _next_regime(period: ScalarInt) -> ScalarInt:
    """The only non-terminal period hands over to the terminal regime."""
    return jnp.where(period >= 0, _RegimeId.last, _RegimeId.alive)


def _alive_utility(health: DiscreteState) -> FloatND:
    return health + 0.0


def _last_utility(health: DiscreteState, spend: DiscreteAction) -> FloatND:
    return health + 0.0 * spend


def _survives_to_spend(health: DiscreteState) -> BoolND:
    """Nothing is feasible in frail health, so its value is `-inf`."""
    return health == _Health.hale


@pytest.fixture
def model() -> Model:
    return Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=lambda age: age < 26,
                states={"health": DiscreteGrid(_Health)},
                state_transitions={"health": MarkovTransition(_health_probs)},
                functions={"utility": _alive_utility},
            ),
            "last": Regime(
                transition=None,
                active=lambda age: age >= 26,
                states={"health": DiscreteGrid(_Health)},
                actions={"spend": DiscreteGrid(_Spend)},
                constraints={"survives_to_spend": _survives_to_spend},
                functions={"utility": _last_utility},
            ),
        },
        ages=AgeGrid(start=25, stop=26, step="1Y"),
        regime_id_class=_RegimeId,
    )


def test_unreachable_node_does_not_destroy_the_reachable_ones(model: Model) -> None:
    """`V(hale) = 1.9`, though the node beside it is unreachable and `-inf`.

    Utility is `health` (frail 0, hale 1) in both regimes and the discount
    factor is `0.9`. Health persists with certainty, so from hale the frail node
    carries probability exactly zero; the terminal regime admits no action in
    frail health, so that node's value is `-inf`. The hale continuation is
    therefore `1.0`, and `V(hale) = 1 + 0.9 · 1`.
    """
    V = model.solve(params={"discount_factor": 0.9}, log_level="off")

    got = np.asarray(V[0]["alive"]).ravel()

    assert np.isneginf(got[_Health.frail])
    np.testing.assert_allclose(got[_Health.hale], 1.9, atol=1e-5)
