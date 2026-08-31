"""A target-only LogNormal entry is priced at its own mean, not at its node mean.

A transition descriptor is qualified by its target regime, so an entry that
reaches only one target is priced from that target's own distribution rather
than from the pooled distribution over every node the state can occupy. The
model below separates the two: the target-only entry's mean differs from the
node mean, so a pooled reading shows up as a wrong price rather than as a
tolerance miss.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LogNormalIIDProcess,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero() -> ScalarFloat:
    return jnp.float32(0)


def _shock(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _one() -> ScalarFloat:
    return jnp.float32(1)


def _target_id() -> ScalarInt:
    return RegimeId.target


def _source_early(age: float) -> bool:
    return age < 22


def _oracle() -> float:
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(3)
    values = np.exp(math.sqrt(2.0) * raw_nodes)
    return float(np.dot(values, raw_weights / math.sqrt(math.pi)))


@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_target_only_lognormal_iid_uses_quadrature_weights(*, coarse, enable_jit):
    transition = _target_id if coarse else {"target": MarkovTransition(_one)}
    process = LogNormalIIDProcess(
        n_points=3,
        gauss_hermite=True,
        mu=0.0,
        sigma=1.0,
    )
    model = Model(
        regimes={
            "source": Regime(
                transition=transition,
                active=_source_early,
                functions={"utility": _zero},
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": _shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    got = float(np.asarray(solution[0]["source"]))
    np.testing.assert_allclose(got, _oracle(), rtol=2e-6)
