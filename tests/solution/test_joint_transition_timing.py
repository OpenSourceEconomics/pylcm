"""A joint transition is realized after the source action is chosen."""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    JointTransition,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import DiscreteAction, FloatND, ScalarInt, UserParams
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class Choice:
    risky: ScalarInt
    safe: ScalarInt


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def _uniform() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _next_payoff(*, choice: DiscreteAction, shock: Mapping[str, FloatND]) -> FloatND:
    risky = jnp.where(shock["high"], 10.0, 0.0)
    return jnp.where(choice == Choice.risky, risky, 6.0)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_action_maximizes_expected_continuation_not_each_realized_node(
    *, enable_jit: bool
) -> None:
    """Correct `max E` chooses the safe value 6; the folded `E max` mutant is 8."""
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_certain)},
                active=lambda age: age < 21,
                actions={"choice": DiscreteGrid(category_class=Choice)},
                functions={"utility": lambda: jnp.asarray(0.0)},
                joint_transitions={
                    "target": {
                        "shock": JointTransition(
                            support_size=2,
                            support={"high": jnp.asarray([True, False])},
                            probabilities=_uniform,
                            outputs={"payoff": _next_payoff},
                        )
                    }
                },
            ),
            "target": Regime(
                transition=None,
                states={"payoff": IrregSpacedGrid(points=(0.0, 6.0, 10.0))},
                functions={"utility": lambda payoff: payoff},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    params: UserParams = {
        "source": {
            "target": {
                "next_regime": {},
                "shock": {"support": {}, "probabilities": {}},
                "next_payoff": {},
            },
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "target": {"utility": {}},
    }

    source_value = model.solve(params=params, log_level="debug").values[0]["source"]

    np.testing.assert_array_almost_equal(source_value, 6.0, decimal=DECIMAL_PRECISION)
    assert not np.isclose(float(source_value), 8.0)
