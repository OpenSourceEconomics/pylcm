"""Tests for initial states conversion utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pandas as pd
from pybaum import tree_equal

from lcm.simulation.util import convert_flat_to_nested_initial_states


def _create_mock_internal_regime(name: str, state_names: list[str]) -> MagicMock:
    """Create a mock InternalRegime with specified state names.

    Args:
        name: Name of the regime.
        state_names: List of state variable names for this regime.

    Returns:
        A mock InternalRegime with a variable_info DataFrame.

    """
    mock = MagicMock()
    mock.name = name

    # Create variable_info DataFrame with is_state column
    # Include both states and some actions to make it realistic
    all_vars = [*state_names, "action1"]
    is_state = [var in state_names for var in all_vars]

    mock.variable_info = pd.DataFrame(
        {"is_state": is_state},
        index=all_vars,
    )
    return mock


class TestConvertFlatToNestedInitialStates:
    """Tests for convert_flat_to_nested_initial_states function."""

    def test_single_regime(self):
        """Single regime with two states."""
        internal_regimes = {
            "regime1": _create_mock_internal_regime("regime1", ["wealth", "health"]),
        }

        flat_initial_states = {
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 1]),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        expected = {
            "regime1": {
                "wealth": jnp.array([10.0, 50.0]),
                "health": jnp.array([0, 1]),
            },
        }

        assert set(result.keys()) == {"regime1"}
        assert set(result["regime1"].keys()) == {"wealth", "health"}
        assert tree_equal(result, expected)

    def test_multi_regime_shared_states(self):
        """Multiple regimes with identical state spaces (shared states)."""
        internal_regimes = {
            "work": _create_mock_internal_regime("work", ["wealth", "health"]),
            "retirement": _create_mock_internal_regime(
                "retirement", ["wealth", "health"]
            ),
        }

        flat_initial_states = {
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 1]),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        # Both regimes should get the same values
        assert set(result.keys()) == {"work", "retirement"}
        assert set(result["work"].keys()) == {"wealth", "health"}
        assert set(result["retirement"].keys()) == {"wealth", "health"}

        # Values should be identical (same array references)
        assert result["work"]["wealth"] is result["retirement"]["wealth"]
        assert result["work"]["health"] is result["retirement"]["health"]

    def test_multi_regime_disjoint_states(self):
        """Multiple regimes with completely different state spaces."""
        internal_regimes = {
            "alive": _create_mock_internal_regime(
                "alive", ["wealth", "health", "education"]
            ),
            "dead": _create_mock_internal_regime("dead", ["funerary_wealth"]),
        }

        flat_initial_states = {
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 1]),
            "education": jnp.array([1, 2]),
            "funerary_wealth": jnp.array([5.0, 25.0]),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        assert set(result.keys()) == {"alive", "dead"}
        assert set(result["alive"].keys()) == {"wealth", "health", "education"}
        assert set(result["dead"].keys()) == {"funerary_wealth"}

        # Verify correct values assigned
        assert tree_equal(result["alive"]["wealth"], jnp.array([10.0, 50.0]))
        assert tree_equal(result["dead"]["funerary_wealth"], jnp.array([5.0, 25.0]))

    def test_multi_regime_partially_overlapping_states(self):
        """Multiple regimes with partially overlapping state spaces."""
        internal_regimes = {
            "work": _create_mock_internal_regime("work", ["wealth", "health"]),
            "retirement": _create_mock_internal_regime(
                "retirement", ["wealth", "pension"]
            ),
        }

        flat_initial_states = {
            "wealth": jnp.array([10.0, 50.0]),
            "health": jnp.array([0, 1]),
            "pension": jnp.array([1, 0]),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        assert set(result.keys()) == {"work", "retirement"}
        assert set(result["work"].keys()) == {"wealth", "health"}
        assert set(result["retirement"].keys()) == {"wealth", "pension"}

        # Shared state (wealth) should be same reference
        assert result["work"]["wealth"] is result["retirement"]["wealth"]

        # Unique states should have correct values
        assert tree_equal(result["work"]["health"], jnp.array([0, 1]))
        assert tree_equal(result["retirement"]["pension"], jnp.array([1, 0]))

    def test_empty_regime_states(self):
        """Regime with no states (e.g., terminal absorbing regime)."""
        internal_regimes = {
            "active": _create_mock_internal_regime("active", ["wealth"]),
            "terminal": _create_mock_internal_regime("terminal", []),
        }

        flat_initial_states = {
            "wealth": jnp.array([10.0, 50.0]),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        assert set(result.keys()) == {"active", "terminal"}
        assert set(result["active"].keys()) == {"wealth"}
        assert set(result["terminal"].keys()) == set()  # Empty dict

    def test_preserves_array_dtypes(self):
        """Verify that array dtypes are preserved through conversion."""
        internal_regimes = {
            "regime1": _create_mock_internal_regime(
                "regime1", ["continuous", "discrete"]
            ),
        }

        flat_initial_states = {
            "continuous": jnp.array([1.5, 2.5], dtype=jnp.float32),
            "discrete": jnp.array([0, 1], dtype=jnp.int32),
        }

        result = convert_flat_to_nested_initial_states(
            flat_initial_states,
            internal_regimes,  # type: ignore[arg-type]
        )

        assert result["regime1"]["continuous"].dtype == jnp.float32
        assert result["regime1"]["discrete"].dtype == jnp.int32
