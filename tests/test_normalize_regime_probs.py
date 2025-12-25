import jax.numpy as jnp

from lcm.utils import normalize_regime_transition_probs


def test_normalize_with_float_values():
    """Test normalization with float values (solve phase)."""
    probs = {"working": 0.7, "retired": 0.1, "unemployed": 0.2}
    active_regimes = ["working", "retired"]
    got = normalize_regime_transition_probs(probs, active_regimes)
    assert jnp.allclose(got["working"], jnp.array(0.7 / 0.8))
    assert jnp.allclose(got["retired"], jnp.array(0.1 / 0.8))


def test_normalize_with_array_values():
    """Test normalization with array values (simulation phase)."""
    probs = {
        "working": jnp.array([0.7, 0.6]),
        "retired": jnp.array([0.1, 0.3]),
        "unemployed": jnp.array([0.2, 0.1]),
    }
    active_regimes = ["working", "retired"]
    got = normalize_regime_transition_probs(probs, active_regimes)
    assert jnp.allclose(got["working"], jnp.array([0.7 / 0.8, 0.6 / 0.9]))
    assert jnp.allclose(got["retired"], jnp.array([0.1 / 0.8, 0.3 / 0.9]))
