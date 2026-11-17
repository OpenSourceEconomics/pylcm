"""An event that can occur may not be priced as one that cannot.

A joint node's probability is the product of one factor per stochastic axis.
Each factor can sit comfortably inside the dtype's normal range while the
product falls below it: in float32, `sqrt(tiny)/2` squared is subnormal. The
hardware then delivers exactly zero, and a zero weight is how this engine
spells "this cannot happen" — so an event with strictly positive probability
would be dropped, silently, and a `-inf` standing at that node (the ordinary
value of a state where no action is feasible) would never reach the answer.

The representation cannot carry such a weight, so the engine refuses it rather
than rounding it to impossible: the continuation is `NaN`. That is arithmetic
rather than validation, so it holds at every log level.

A factor of exactly zero is a different thing entirely — a genuine null event —
and still contributes nothing, whatever value stands at it.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=4)
_WEALTH_VALUES = np.array([1.0, 2.0, 3.0, 4.0])


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Binary:
    low: ScalarInt
    high: ScalarInt


def _no_utility(*, health: DiscreteState, mood: DiscreteState) -> FloatND:
    return jnp.asarray(0.0) + 0.0 * health + 0.0 * mood


def _wealth_utility(
    *, wealth: ScalarFloat, health: DiscreteState, mood: DiscreteState
) -> FloatND:
    return wealth + 0.0 * health + 0.0 * mood


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def test_a_model_whose_joint_node_underflows_still_solves() -> None:
    """Two ordinary Markov rows can still make a joint node the dtype cannot hold.

    Each row is a valid distribution of normal probabilities, so every
    input-level check passes. Their joint node is the product of the two small
    entries, which underflows. Its value is finite, so its contribution is below
    the last bit of the continuation and the solve answers as though the node
    were absent — which it is, to every tolerance the model declares.
    """
    active = jnp.zeros(()).dtype
    small = float(jnp.finfo(active).tiny) ** 0.5 / 2

    def _health_probs() -> FloatND:
        return jnp.asarray([small, 1.0 - small], dtype=active)

    def _mood_probs() -> FloatND:
        return jnp.asarray([small, 1.0 - small], dtype=active)

    model = Model(
        edges={"alive": {"dead": 20}},
        regimes={
            "alive": Regime(
                states={
                    "wealth": _WEALTH,
                    "health": DiscreteGrid(category_class=_Binary),
                    "mood": DiscreteGrid(category_class=_Binary),
                },
                state_transitions={
                    "wealth": _keep_wealth,
                    "health": StochasticTransition(func=_health_probs),
                    "mood": StochasticTransition(func=_mood_probs),
                },
                functions={"utility": _no_utility},
            ),
            "dead": Regime(
                states={
                    "wealth": _WEALTH,
                    "health": DiscreteGrid(category_class=_Binary),
                    "mood": DiscreteGrid(category_class=_Binary),
                },
                functions={"utility": _wealth_utility},
            ),
        },
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={20: "alive"},
    )

    V = model.solve(
        params={"alive": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    ).values

    np.testing.assert_allclose(
        np.asarray(V[0]["alive"]), np.broadcast_to(_WEALTH_VALUES, (2, 2, 4))
    )
