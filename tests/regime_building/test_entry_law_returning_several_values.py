"""An entry law names one value, and is rejected when it names several.

Entering a process means handing over the point on the target's support the source
arrives at. A law returning an array names a distribution instead, which the process
already carries — so it is rejected at model build, naming the process and both ways
out.
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


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _enter_at_several_values() -> FloatND:
    return jnp.array([1.0, 2.0], dtype=jnp.float32)


def _build() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"shock": {"target": _enter_at_several_values}},
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


def test_the_rejection_names_the_process_and_its_regime() -> None:
    """The message identifies the process entered and the regime holding it."""
    with pytest.raises(ModelInitializationError) as excinfo:
        _build()

    message = str(excinfo.value)
    assert "shock" in message
    assert "target" in message


def test_the_rejection_reports_the_shape_returned() -> None:
    """The message states what the law actually returned."""
    with pytest.raises(ModelInitializationError, match=r"shape \(2,\)"):
        _build()
