"""Simulation under a nonlinear certainty equivalent."""

import jax.numpy as jnp
import numpy as np

from lcm import PowerMean
from lcm_examples.epstein_zin import (
    EZRegimeId,
    HealthStatus,
    get_model,
    get_params,
)
from tests.test_certainty_equivalent import _reference_backward_induction


def test_simulated_period0_consumption_matches_reference_policy():
    """Period-0 consumption equals the reference argmax at the initial states."""
    risk_aversion, discount_factor, rho = 0.5, 0.9, 0.5
    model = get_model(certainty_equivalent=PowerMean())
    params = get_params(
        risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
    )
    initial_wealth = jnp.array([2.8, 7.4, 12.0])  # on-grid nodes
    initial_health = jnp.array([HealthStatus.bad, HealthStatus.good, HealthStatus.good])
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(3, 60.0),
            "wealth": initial_wealth,
            "health": initial_health,
            "regime_id": jnp.full(3, EZRegimeId.alive),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe(use_labels=False)
    period0 = df.query("period == 0").sort_index()

    _, policy_c = _reference_backward_induction(
        risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
    )
    wealth_grid = np.linspace(0.5, 12.0, 6)
    expected = np.array(
        [
            # Nearest-node lookup, not `searchsorted`: linspace values and
            # float literals can differ in the last bit.
            policy_c[int(np.argmin(np.abs(wealth_grid - w))), h]
            for w, h in [(2.8, 0), (7.4, 1), (12.0, 1)]
        ]
    )
    np.testing.assert_allclose(period0["consumption"].to_numpy(), expected, rtol=1e-5)
