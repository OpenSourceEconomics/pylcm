"""Replayed outer actions must reproduce the target the solve chose.

N-NB-EGM searches the outer margin over post-decision targets and stores those
targets, not the action that reaches them. Simulation recovers the action by
inverting the declared affine map, and every downstream reader — the durable law
of motion, the credited cost, the liquid budget — then evaluates that map
forward again. The recovered action therefore has to be the representable one
whose forward image is the stored target, or the reassembled post-decision state
drifts off it.

The drift is a full ULP of the operands at a divestment corner, which is enough
to carry a state one step outside the grid that declares its domain, where the
value function is not defined.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models import n_nbegm_toy as toy
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95}

# `1.37` is not representable in binary, so full divestment from it cancels two
# operands of its own magnitude — exact at float64, one operand-ULP off at
# float32. The other three subjects hold stocks that cancel exactly at both.
_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 17.5]),
    "age": jnp.full(4, 20.0),
    "regime_id": jnp.full(4, RegimeId.alive, dtype=jnp.int32),
}


@pytest.fixture(scope="module")
def simulated():
    return (
        toy.build_model(variant="n_nbegm", n_periods=3)
        .simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            period_to_regime_to_V_arr=None,
            log_level="off",
            seed=42,
        )
        .to_dataframe()
    )


def test_simulated_durable_stock_stays_inside_its_declared_grid(simulated) -> None:
    """No simulated durable stock leaves the domain its own grid declares."""
    points = np.asarray(toy.ILLIQUID_GRID.to_jax())
    stock = simulated["illiquid"].to_numpy()
    observed = stock[np.isfinite(stock)]
    assert observed.min() >= points[0]
    assert observed.max() <= points[-1]


def test_simulated_value_is_finite_for_every_alive_subject_period(simulated) -> None:
    """A feasible corner yields a value, not the infeasible sentinel."""
    alive = simulated.loc[simulated["regime_name"] == "alive", "value"].to_numpy()
    assert np.isfinite(alive).all()
