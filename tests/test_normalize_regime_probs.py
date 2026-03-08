from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.simulation.utils import validate_regime_transition_probs

# ======================================================================================
# Tests for validate_regime_transition_probs
# ======================================================================================


def test_valid_probs_all_active():
    """Valid probabilities with all regimes active pass validation."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
        }
    )
    validate_regime_transition_probs(
        regime_transition_probs=probs,
        active_regimes_next_period=("working_life", "retirement"),
        regime_name="working_life",
        period=0,
    )


def test_valid_probs_with_inactive_regime_at_zero():
    """Inactive regime with zero probability passes validation."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
            "dead": jnp.array([0.0, 0.0]),
        }
    )
    validate_regime_transition_probs(
        regime_transition_probs=probs,
        active_regimes_next_period=("working_life", "retirement"),
        regime_name="working_life",
        period=0,
    )


def test_raises_for_probs_not_summing_to_one():
    """Probabilities that don't sum to 1 raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.5, 0.6]),
            "retirement": jnp.array([0.3, 0.4]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"sum to .* instead of 1\.0",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
        )


def test_raises_for_positive_probability_on_inactive_regime():
    """Positive probability on an inactive regime raises an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([0.7, 0.6]),
            "retirement": jnp.array([0.1, 0.2]),
            "dead": jnp.array([0.2, 0.2]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="'dead' is inactive",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
        )


def test_raises_for_nan_values():
    """NaN values in probabilities raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([jnp.nan, 0.5]),
            "retirement": jnp.array([jnp.nan, 0.5]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="Non-finite values",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
        )


def test_raises_for_inf_values():
    """Inf values in probabilities raise an error."""
    probs = MappingProxyType(
        {
            "working_life": jnp.array([jnp.inf, 0.5]),
            "retirement": jnp.array([0.0, 0.5]),
        }
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match="Non-finite values",
    ):
        validate_regime_transition_probs(
            regime_transition_probs=probs,
            active_regimes_next_period=("working_life", "retirement"),
            regime_name="working_life",
            period=0,
        )
