"""A support with a single node behaves as the constant it is.

A one-node grid holds one value, so every query resolves to it. Its coordinate map has
no spacing to divide by, and the value function it indexes has one entry.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    Regime,
    UniformIIDProcess,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt

_ONE_NODE = UniformIIDProcess(n_points=1, start=0.0, stop=2.0)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _enter_at_the_node() -> ScalarFloat:
    return jnp.float32(0)


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_plus_ten(shock: ScalarFloat) -> ScalarFloat:
    return shock + 10.0


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0, 5.0])
def test_one_node_grid_maps_every_value_to_its_only_node(value: float) -> None:
    """Every value resolves to coordinate zero, the grid's only index."""
    coordinate = _ONE_NODE.get_coordinate(jnp.float32(value))

    np.testing.assert_allclose(np.asarray(coordinate), 0.0, atol=1e-6)


def test_entering_a_one_node_support_yields_the_targets_value_there() -> None:
    """Entering a one-node process gives the target's value at that single node.

    The target's terminal utility is `shock + 10` and its only node is at zero, so the
    continuation is 10 and the source's zero utility leaves `V = 10`.
    """
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"shock": {"target": _enter_at_the_node}},
                functions={"utility": _no_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": _ONE_NODE},
                functions={"utility": _shock_plus_ten},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )

    V = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    )

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([10.0]), atol=1e-5
    )
