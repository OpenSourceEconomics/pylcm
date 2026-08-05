"""A declared process entry is an ordinary deterministic value in the next-state DAG.

Where a target holds its value function on discrete nodes, solve additionally
needs a node axis to interpolate over. That axis is a private, engine-internal
object: the public `next_<state>` name keeps producing the physical value the
user's law names, so any other law may consume it exactly as it consumes any
other transition's output.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _source_is_early(age: float) -> bool:
    return age < 22


def _process() -> NormalIIDProcess:
    # Binned nodes are exactly (0, 1, 2), so entry at 1.5 is off-node but on-support.
    return NormalIIDProcess(
        n_points=3,
        gauss_hermite=False,
        mu=1.0,
        sigma=0.5,
        n_std=2.0,
    )


def _target_utility(shock: ScalarFloat, other: ScalarFloat) -> FloatND:
    return shock + 10.0 * other


def _enter_shock() -> ScalarFloat:
    return jnp.asarray(1.5)


def _enter_other(next_shock: ScalarFloat) -> ScalarFloat:
    return next_shock


PARAMS = {
    "source": {
        "utility": {},
        "koopmans_aggregator": {"discount_factor": 1.0},
        "certainty_equivalent": {"risk_aversion": 2.0},
        "target": {
            "next_regime": {},
            "next_shock": {},
            "next_other": {},
        },
    },
    "target": {"utility": {}},
}


def _build_model(*, enable_jit: bool) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={
                    "shock": {"target": _enter_shock},
                    "other": {"target": _enter_other},
                },
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={"shock": _process(), "other": _process()},
                functions={"utility": _target_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_explicit_process_entry_can_feed_another_explicit_entry(
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """Solve prices the single declared point `(1.5, 1.5)` at 16.5.

    `next_other` reads `next_shock`, so the first entry has to publish its
    physical value rather than the node vector its own interpolation runs over.
    `V(shock, other)` is affine, so bilinear interpolation at `(1.5, 1.5)` is
    exactly `1.5 + 10 * 1.5`. The nonlinear certainty equivalent is present
    deliberately: a deterministic entry stays one value before the CE.
    """
    model = _build_model(enable_jit=enable_jit)

    solution = model.solve(params=PARAMS, log_level="debug")
    last_source_period = max(
        period for period in solution if "source" in solution[period]
    )
    got = float(np.asarray(solution[last_source_period]["source"]).ravel()[0])

    np.testing.assert_almost_equal(got, 16.5, decimal=DECIMAL_PRECISION)
