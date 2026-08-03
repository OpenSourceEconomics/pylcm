"""Simulation under a nonlinear certainty equivalent."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import PowerMean
from lcm_examples.epstein_zin import (
    EZRegimeId,
    HealthStatus,
    get_model,
    get_params,
)
from tests.test_certainty_equivalent import (
    _Health,
    _make_scale_equivariant_model,
    _reference_backward_induction,
    _RegimeId,
    _scaled_model_params,
)


def test_simulated_period0_consumption_matches_reference_policy():
    """Period-0 consumption equals the reference argmax at the initial states."""
    risk_aversion, discount_factor, ies = 0.5, 0.9, 2.0
    model = get_model(certainty_equivalent=PowerMean())
    params = get_params(
        risk_aversion=risk_aversion,
        discount_factor=discount_factor,
        intertemporal_elasticity_of_substitution=ies,
    )
    initial_wealth = jnp.array([2.8, 7.4, 12.0])  # on-grid nodes
    initial_health = jnp.array([HealthStatus.bad, HealthStatus.good, HealthStatus.good])
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(3, 25.0),
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
        risk_aversion=risk_aversion,
        discount_factor=discount_factor,
        intertemporal_elasticity_of_substitution=ies,
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


def test_bad_health_cost_drains_wealth_in_simulation():
    """`next_wealth = clip(w - c + income - cost·1[bad], grid bounds)` in simulation."""
    model = get_model(certainty_equivalent=PowerMean())
    params = get_params(
        risk_aversion=2.0,
        income=1.0,
        health_cost=1.5,
        bequest_scale=0.3,
    )
    n_subjects = 40
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 25.0),
            "wealth": jnp.linspace(0.5, 12.0, n_subjects),
            "health": jnp.tile(
                jnp.array([HealthStatus.bad, HealthStatus.good]), n_subjects // 2
            ),
            "regime_id": jnp.full(n_subjects, EZRegimeId.alive),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=7,
    )
    df = result.to_dataframe(use_labels=False)
    alive = df[df["regime_name"] == "alive"].sort_values(["subject_id", "period"])
    curr = alive.groupby("subject_id").nth(0)
    following = alive.groupby("subject_id").nth(1)
    both = curr.merge(
        following, on="subject_id", suffixes=("", "_next"), how="inner"
    ).query("period_next == period + 1")
    assert len(both) > 0
    expected_next_wealth = np.clip(
        both["wealth"]
        - both["consumption"]
        + 1.0
        - 1.5 * (both["health"] == int(HealthStatus.bad)),
        0.5,
        12.0,
    )
    np.testing.assert_allclose(
        both["wealth_next"].to_numpy(), expected_next_wealth.to_numpy(), atol=1e-6
    )


def test_bequest_scale_scales_the_dead_regime_utility():
    """Dead-regime utility equals `bequest_scale * sqrt(wealth)`."""
    model = get_model(certainty_equivalent=PowerMean())
    params = get_params(risk_aversion=2.0, bequest_scale=0.3)
    n_subjects = 30
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 25.0),
            "wealth": jnp.linspace(0.5, 12.0, n_subjects),
            "health": jnp.full(n_subjects, HealthStatus.bad),
            "regime_id": jnp.full(n_subjects, EZRegimeId.alive),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=7,
    )
    df = result.to_dataframe(use_labels=False, additional_targets=["utility"])
    dead = df[df["regime_name"] == "dead"]
    assert len(dead) > 0
    np.testing.assert_allclose(
        dead["utility"].to_numpy(), 0.3 * np.sqrt(dead["wealth"].to_numpy()), atol=1e-6
    )


@pytest.mark.parametrize("risk_aversion", [2.0, 50.0])
def test_simulated_consumption_is_equivariant_to_rescaling_the_model(
    x64_enabled: None,  # noqa: ARG001
    risk_aversion: float,
):
    """Scaling a homogeneous model by `k > 0` scales its simulated choices by `k`."""
    scale = 1e-7
    params = _scaled_model_params(risk_aversion)

    def simulate(model_scale: float) -> np.ndarray:
        model = _make_scale_equivariant_model(model_scale)
        result = model.simulate(
            params=params,
            initial_conditions={
                "age": jnp.full(2, 25.0),
                "wealth": jnp.array([2.8, 12.0]) * model_scale,
                "health": jnp.array([_Health.bad, _Health.good]),
                "regime_id": jnp.full(2, _RegimeId.alive),
            },
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )
        df = result.to_dataframe(use_labels=False)
        return df.query("period == 0")["consumption"].to_numpy()

    np.testing.assert_allclose(simulate(scale) / scale, simulate(1.0), rtol=1e-6)
