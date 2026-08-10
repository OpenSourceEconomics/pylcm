"""A `MarkovTransition` law is rejected when its state's grid is continuous.

`MarkovTransition` declares a probability vector over a discrete outcome space. A
continuous stochastic process owns its own transition mechanism, so wrapping an entry
law into one in `MarkovTransition` names no meaningful object; it is rejected at model
build with the state, the regime and the target named.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _shock_probs() -> FloatND:
    return jnp.array([0.25, 0.5, 0.25], dtype=jnp.float32)


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _build_model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"shock": {"target": MarkovTransition(_shock_probs)}},
                functions={"utility": _no_utility},
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
                    )
                },
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_markov_law_on_continuous_process_names_state_regime_and_target() -> None:
    """The rejection names the state, the source regime and the target regime."""
    with pytest.raises(ModelInitializationError) as excinfo:
        _build_model()

    message = str(excinfo.value)
    assert "shock" in message
    assert "source" in message
    assert "target" in message


def test_markov_law_on_continuous_process_names_the_wrapper() -> None:
    """The rejection names `MarkovTransition`, so the offending law is findable."""
    with pytest.raises(ModelInitializationError, match="MarkovTransition"):
        _build_model()
