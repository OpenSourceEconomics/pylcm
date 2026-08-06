"""A source law may not read a target's stochastic draw during backward induction.

`next_<state>` of a target's stochastic process has no realized value while the
expectation over that process is still being built, so a source law or helper reading
one is rejected at model build. It is never silently rebound to a runtime parameter:
that would leave the user supplying a scalar where the model names a random draw.
"""

import jax.numpy as jnp
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

_SHOCK = NormalIIDProcess(n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _wealth_and_shock(wealth: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return wealth + shock


def _wealth_utility(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _build_model(source_functions, source_state_transitions) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
                state_transitions=source_state_transitions,
                functions=source_functions,
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": _SHOCK,
                    "wealth": LinSpacedGrid(start=0.0, stop=4.0, n_points=5),
                },
                functions={"utility": _wealth_and_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def _build_with_direct_read() -> Model:
    def next_wealth(next_shock: ScalarFloat) -> ScalarFloat:
        return 2.0 * next_shock

    return _build_model(
        {"utility": _wealth_utility},
        {"wealth": {"target": next_wealth}},
    )


def _build_with_helper_read() -> Model:
    def scaled(next_shock: ScalarFloat) -> ScalarFloat:
        return 2.0 * next_shock

    def next_wealth(scaled: ScalarFloat) -> ScalarFloat:
        return scaled

    return _build_model(
        {"utility": _wealth_utility, "scaled": scaled},
        {"wealth": {"target": next_wealth}},
    )


@pytest.mark.parametrize("build", [_build_with_direct_read, _build_with_helper_read])
def test_reading_an_entered_draw_is_rejected_at_build(build) -> None:
    """Both a direct read and a read through a helper are rejected at model build."""
    with pytest.raises(ModelInitializationError):
        build()


@pytest.mark.parametrize("build", [_build_with_direct_read, _build_with_helper_read])
def test_rejection_names_the_draw_and_the_regimes(build) -> None:
    """The message names the draw, the source regime and the target regime."""
    with pytest.raises(ModelInitializationError) as excinfo:
        build()

    message = str(excinfo.value)
    assert "next_shock" in message
    assert "source" in message
    assert "target" in message


@pytest.mark.parametrize("build", [_build_with_direct_read, _build_with_helper_read])
def test_the_draw_never_becomes_a_runtime_parameter(build) -> None:
    """The draw is not rebound to a parameter the user is asked to supply."""
    with pytest.raises(ModelInitializationError) as excinfo:
        build()

    assert "ScalarFloat" not in str(excinfo.value)
