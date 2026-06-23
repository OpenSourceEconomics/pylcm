"""NEGM forward simulation runs on CPU and yields a finite consumption policy.

NEGM regimes carry a within-period durable law (`next_<durable>`) that the
budget constraint and — for a service-flow durable — `utility` read. The
forward-simulation decision computes that next state from the chosen action
rather than demanding it as an external input, so the simulate argmax solves
and the realised consumption policy is finite.
"""

import jax.numpy as jnp
import pytest

from tests.test_models import negm_kinked_toy, negm_serviceflow_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _simulate_alive_consumption(model, regime_id):
    solution = model.solve(params=_PARAMS, log_level="off")
    initial_conditions = {
        "wealth": jnp.array([5.0, 10.0, 15.0]),
        "illiquid": jnp.array([4.0, 6.0, 8.0]),
        "age": jnp.full(3, 20.0),
        "regime_id": jnp.full(3, regime_id, dtype=jnp.int32),
    }
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
    )
    df = result.to_dataframe()
    return df.query("regime_name == 'alive'")["consumption"]


@pytest.mark.parametrize(
    ("build_model", "regime_id"),
    [
        (negm_kinked_toy.build_model, negm_kinked_toy.RegimeId.alive),
        (negm_serviceflow_toy.build_negm_model, negm_serviceflow_toy.RegimeId.alive),
    ],
)
def test_negm_simulate_yields_finite_consumption(build_model, regime_id):
    """The simulated alive-regime consumption policy is finite and positive."""
    consumption = _simulate_alive_consumption(build_model(), regime_id)
    assert len(consumption) > 0
    assert bool(consumption.notna().all())
    assert bool((consumption > 0.0).all())
