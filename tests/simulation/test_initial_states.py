"""Tests for initial states conversion and validation utilities."""

import jax.numpy as jnp
import pytest

from lcm.simulation.util import (
    convert_flat_to_nested_initial_states,
    validate_flat_initial_states,
)
from tests.test_models.utils import get_model


def test_convert_flat_to_nested_single_regime():
    """Single regime gets its states from flat dict."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=2)

    flat = {"wealth": jnp.array([10.0, 50.0])}
    nested = convert_flat_to_nested_initial_states(flat, model.internal_regimes)

    assert "iskhakov_et_al_2017_stripped_down" in nested
    assert "wealth" in nested["iskhakov_et_al_2017_stripped_down"]


def test_validate_flat_initial_states_valid_input():
    """Valid input should not raise."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=2)

    flat = {"wealth": jnp.array([10.0, 50.0])}
    validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_missing_state():
    """Missing state should raise ValueError."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=2)

    flat: dict[str, jnp.ndarray] = {}  # Missing "wealth"

    with pytest.raises(ValueError, match="Missing initial states"):
        validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_extra_state():
    """Extra state should raise ValueError."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=2)

    flat = {
        "wealth": jnp.array([10.0]),
        "unknown": jnp.array([1.0]),
    }

    with pytest.raises(ValueError, match="Unknown initial states"):
        validate_flat_initial_states(flat, model.internal_regimes)


def test_validate_flat_initial_states_inconsistent_lengths():
    """Arrays with different lengths should raise ValueError."""
    model = get_model("iskhakov_et_al_2017_stochastic", n_periods=2)

    flat = {
        "wealth": jnp.array([10.0, 50.0, 100.0]),  # Length 3
        "health": jnp.array([0, 1]),  # Length 2
        "partner": jnp.array([0, 1]),  # Length 2
    }

    with pytest.raises(ValueError, match="same length"):
        validate_flat_initial_states(flat, model.internal_regimes)
