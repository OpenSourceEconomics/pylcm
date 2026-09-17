"""Tests for arranging initial states per regime."""

import jax.numpy as jnp

from _lcm.simulation.initial_conditions import build_initial_states
from lcm import Model


def test_build_initial_states_single_regime(model: Model) -> None:
    """Each regime's state arrays land under its name in the nested carrier."""
    initial = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    result = build_initial_states(initial_states=initial, regimes=model._regimes)

    assert "wealth" in result["active"]
    assert "health" in result["active"]
    # Terminal regime has no states, so its inner mapping is empty.
    assert dict(result["terminal"]) == {}
