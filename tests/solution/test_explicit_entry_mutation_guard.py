"""Calling a declared entry a draw is rejected, not silently repriced.

A test battery is only evidence if some reachable change to the code breaks it.
The mutation here is the exact defect the interpolation-basis distinction exists
to prevent: re-label a declared entry law as a draw, so its node-basis
coefficients would flow into the joint lottery and reach the certainty
equivalent as probabilities.

A declared entry publishes a physical value that its own weights read, and a
draw publishes nothing until the expectation over it is complete. Relabelling
therefore leaves the weights without a producer, which the model rejects at
build rather than resolving into some other number.

The mutation is applied to the descriptor the engine reads, not by editing
source, so it stays valid as the surrounding code changes.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building import processing
from _lcm.transition_laws import TransitionLawInfo
from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import FloatND, ScalarFloat, ScalarInt

_RISK_AVERSION = 2.0
# Nodes `(0, 1, 2)` under payoff `shock**2` give `V = (0, 1, 4)`; entering at
# `1.5` interpolates to `2.5`, while reading the coefficients as probabilities
# would give the weighted harmonic mean `1.6`.
_INTERPOLATED = 2.5


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _squared_shock_utility(shock: ScalarFloat) -> FloatND:
    return shock**2


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _enter_between_nodes() -> ScalarFloat:
    return jnp.asarray(1.5)


def _build_model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                state_transitions={"shock": {"target": _enter_between_nodes}},
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
                    )
                },
                functions={"utility": _squared_shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )


_PARAMS = {
    "source": {
        "utility": {},
        "koopmans_aggregator": {"discount_factor": 1.0},
        "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
        "target": {"next_regime": {}, "next_shock": {}},
    },
    "target": {"utility": {}},
}


def _source_value(model: Model) -> float:
    solution = model.solve(params=_PARAMS, log_level="debug")
    last_living = max(period for period in solution if "source" in solution[period])
    return float(np.asarray(solution[last_living]["source"]).ravel()[0])


def test_the_declared_entry_prices_at_its_interpolated_value() -> None:
    """A deterministic entry at `1.5` between nodes `(0, 1, 2)` prices at `2.5`.

    Under `PowerMean` the coefficients read as probabilities would give the
    weighted harmonic mean `1.6` instead, so this pins the value against exactly
    the reading the mutation below makes unbuildable.
    """
    got = _source_value(_build_model())

    np.testing.assert_allclose(got, _INTERPOLATED, rtol=1e-6)


def test_relabelling_the_declared_entry_as_a_draw_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling a declared entry stochastic fails the model build, naming the law.

    An entry's coefficients express one value in the target's node basis; a
    draw's weights are probabilities over those nodes. Nothing is both, and the
    model says so instead of pricing the coefficients as probabilities.
    """
    original = processing._build_transition_laws

    def _as_all_stochastic(**kwargs: Any) -> Any:
        laws = original(**kwargs)
        return type(laws)(
            {
                target: type(bundle)(
                    {
                        name: dataclasses.replace(
                            info,
                            stochastic=info.stochastic or info.interpolation_basis,
                            interpolation_basis=False,
                        )
                        if isinstance(info, TransitionLawInfo)
                        else info
                        for name, info in bundle.items()
                    }
                )
                for target, bundle in laws.items()
            }
        )

    monkeypatch.setattr(processing, "_build_transition_laws", _as_all_stochastic)

    with pytest.raises(ModelInitializationError, match="carries the node basis"):
        _build_model()
