"""Handing a value to a process whose law is supplied at runtime is rejected.

Placing a value on a discretized process's nodes happens inside the *source's* Bellman
equation, which reads only the source's own parameters. A process the target
parameterizes at runtime has no nodes the source can read, so the handoff is rejected
at model build rather than interpolated against a support that does not exist yet.

A source carrying the very same process is untouched: nothing is placed there, because
the process transitions under its own law.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ScalarFloat, ScalarInt

_RUNTIME_PROCESS = NormalIIDProcess(n_points=3, gauss_hermite=False)
_SHOCK_PARAMS = {"mu": 1.0, "sigma": 0.5, "n_std": 2.0}
_PARAMS = {
    "source": {
        "koopmans_aggregator": {"discount_factor": 1.0},
        "shock": _SHOCK_PARAMS,
    },
    "target": {"shock": _SHOCK_PARAMS},
}


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _reset_shock(shock: ScalarFloat) -> ScalarFloat:
    del shock
    return jnp.float32(1)


def _build(source_states, source_state_transitions) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states=source_states,
                state_transitions=source_state_transitions,
                functions={"utility": _shock_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": _RUNTIME_PROCESS},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_handing_an_ordinary_grid_to_a_runtime_process_is_rejected() -> None:
    """A source grid shadowing a runtime-parameterized target process is rejected."""
    with pytest.raises(ModelInitializationError):
        _build(
            {"shock": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
            {"shock": _reset_shock},
        )


def test_the_rejection_names_the_process_and_the_target() -> None:
    """The message names the state and the regime whose law is supplied at runtime."""
    with pytest.raises(ModelInitializationError) as excinfo:
        _build(
            {"shock": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
            {"shock": _reset_shock},
        )

    message = str(excinfo.value)
    assert "shock" in message
    assert "target" in message


def test_carrying_the_same_runtime_process_still_solves() -> None:
    """A source carrying the target's own process is unaffected and solves.

    Utility is the source's own shock and the continuation is the target's process
    mean of one, so `V = shock + 1` across the process's nodes at zero, one and two.
    """
    model = _build({"shock": _RUNTIME_PROCESS}, {})

    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([1.0, 2.0, 3.0]), atol=1e-5
    )
