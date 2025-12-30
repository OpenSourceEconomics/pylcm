import jax.numpy as jnp
import pytest

from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.simulation.util import _validate_normalized_regime_transition_probs
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


# ======================================================================================
# Tests for _validate_normalized_regime_transition_probs
# ======================================================================================


def test_validate_normalized_probs_passes_for_valid_probs():
    """Test that validation passes for valid normalized probabilities."""
    normalized_probs = {
        "working": jnp.array([0.7, 0.6]),
        "retired": jnp.array([0.3, 0.4]),
    }
    # Should not raise
    _validate_normalized_regime_transition_probs(
        normalized_probs, regime_name="working", period=0
    )


def test_validate_normalized_probs_raises_for_nan_values():
    """Test that validation raises error when probabilities contain NaN.

    This happens when all active regimes have zero probability and division by zero
    produces NaN values. Since NaN values can't sum to 1.0, the "do not sum to 1"
    error is triggered.
    """
    # Simulate what happens when normalization divides by zero
    # (all active regimes had 0 probability before normalization)
    normalized_probs = {
        "working": jnp.array([jnp.nan, 0.5]),
        "retired": jnp.array([jnp.nan, 0.5]),
    }
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="do not sum to 1 after normalization",
    ):
        _validate_normalized_regime_transition_probs(
            normalized_probs, regime_name="working", period=0
        )


def test_validate_normalized_probs_raises_for_inf_values():
    """Test that validation raises error when probabilities contain Inf values.

    Since Inf values can't sum to 1.0, the "do not sum to 1" error is triggered.
    """
    normalized_probs = {
        "working": jnp.array([jnp.inf, 0.5]),
        "retired": jnp.array([0.0, 0.5]),
    }
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="do not sum to 1 after normalization",
    ):
        _validate_normalized_regime_transition_probs(
            normalized_probs, regime_name="working", period=0
        )


def test_validate_normalized_probs_raises_for_probs_not_summing_to_one():
    """Test that validation raises error when probabilities don't sum to 1."""
    normalized_probs = {
        "working": jnp.array([0.5, 0.6]),
        "retired": jnp.array([0.3, 0.4]),  # Sums to 0.8 and 1.0
    }
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="do not sum to 1 after normalization",
    ):
        _validate_normalized_regime_transition_probs(
            normalized_probs, regime_name="working", period=0
        )


def test_normalize_produces_nan_when_all_active_probs_zero():
    """Test that normalization produces NaN when all active regime probs are 0.

    This demonstrates the scenario that triggers
    InvalidRegimeTransitionProbabilitiesError during simulation - when the next_regime
    function assigns 0 probability to all regimes that are active in the next period.
    """
    probs = {
        "working": jnp.array([0.0, 0.5]),
        "retired": jnp.array([0.0, 0.3]),
        "unemployed": jnp.array([1.0, 0.2]),  # Only this regime has probability
    }
    # But only working and retired are active
    active_regimes = ["working", "retired"]
    got = normalize_regime_transition_probs(probs, active_regimes)

    # First subject has all zeros for active regimes -> NaN after normalization
    assert jnp.isnan(got["working"][0])
    assert jnp.isnan(got["retired"][0])

    # Second subject has valid probabilities
    assert jnp.allclose(got["working"][1], jnp.array(0.5 / 0.8))
    assert jnp.allclose(got["retired"][1], jnp.array(0.3 / 0.8))
