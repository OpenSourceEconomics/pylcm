from __future__ import annotations

import pytest

from lcm import Model, Regime
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.grids import DiscreteGrid


def test_regime_invalid_states():
    with pytest.raises(RegimeInitializationError, match="states must be a dictionary"):
        Regime(
            name="test",
            states="health",  # type: ignore[arg-type]
            actions={},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_invalid_actions():
    with pytest.raises(RegimeInitializationError, match="actions must be a dictionary"):
        Regime(
            name="test",
            states={},
            actions="exercise",  # type: ignore[arg-type]
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_invalid_functions():
    with pytest.raises(
        RegimeInitializationError, match="functions must be a dictionary"
    ):
        Regime(
            name="test",
            states={},
            actions={},
            transitions={"test": {"next_health": lambda: 0}},
            utility=lambda: 0,
            functions="utility",  # type: ignore[arg-type]
        )


def test_regime_invalid_functions_values():
    with pytest.raises(
        RegimeInitializationError,
        match=r"function values must be a callable, but is 0.",
    ):
        Regime(
            name="test",
            states={},
            actions={},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
            functions={"function": 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_functions_keys():
    with pytest.raises(
        RegimeInitializationError, match=r"function keys must be a strings, but is 0."
    ):
        Regime(
            name="test",
            states={},
            actions={},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
            functions={0: lambda: 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_actions_values():
    with pytest.raises(
        RegimeInitializationError, match=r"actions value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            states={},
            actions={"exercise": 0},  # type: ignore[dict-item]
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_invalid_states_values():
    with pytest.raises(
        RegimeInitializationError, match=r"states value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            states={"health": 0},  # type: ignore[dict-item]
            actions={},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_missing_next_func(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match=r"Each state must have a corresponding transition function.",
    ):
        Regime(
            name="test",
            states={
                "health": DiscreteGrid(binary_category_class),
                "wealth": DiscreteGrid(binary_category_class),
            },
            actions={"exercise": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_invalid_utility():
    with pytest.raises(
        RegimeInitializationError,
        match=(r"utility must be a callable."),
    ):
        Regime(
            name="test",
            states={},
            actions={},
            functions={},
            utility=0,  # type: ignore[arg-type]
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_regime_invalid_transition_names():
    with pytest.raises(
        RegimeInitializationError,
        match=(r"Each transitions name must start with 'next_'."),
    ):
        Regime(
            name="test",
            states={},
            actions={},
            functions={},
            utility=lambda: 0,
            transitions={"test": {"invalid_name": lambda: 0}},
        )


def test_regime_overlapping_states_actions(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        Regime(
            name="test",
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
            transitions={"test": {"next_health": lambda: 0}},
        )


def test_single_regime_without_next_regime_works(binary_category_class):
    """Single-regime models should not require explicit next_regime."""
    regime = Regime(
        name="test",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={"test": {"next_health": lambda health: health}},
        # Note: no next_regime defined
    )
    model = Model(regimes=regime, n_periods=2)
    # Should not raise, and internal regime should have next_regime
    assert "next_regime" not in regime.transitions  # Original unchanged
    # Model processes successfully
    assert model.internal_regimes is not None


def test_single_regime_with_next_regime_warns(binary_category_class):
    """Single-regime models with user-defined next_regime should warn and override."""
    regime = Regime(
        name="test",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "test": {"next_health": lambda health: health},
            "next_regime": lambda: {
                "test": 0.5
            },  # Invalid probability, should be ignored
        },
    )
    with pytest.warns(UserWarning, match="will be ignored"):
        model = Model(regimes=regime, n_periods=2)
    # Model should still work
    assert model.internal_regimes is not None


def test_multi_regime_without_next_regime_raises(binary_category_class):
    """Multi-regime models must have next_regime in each regime."""
    regime1 = Regime(
        name="regime1",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "regime1": {"next_health": lambda health: health},
            "regime2": {"next_health": lambda health: health},
            # Missing next_regime
        },
    )
    regime2 = Regime(
        name="regime2",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "regime1": {"next_health": lambda health: health},
            "regime2": {"next_health": lambda health: health},
            "next_regime": lambda: {"regime1": 0.5, "regime2": 0.5},
        },
    )
    with pytest.raises(ModelInitializationError, match="next_regime"):
        Model(regimes=[regime1, regime2], n_periods=2)
