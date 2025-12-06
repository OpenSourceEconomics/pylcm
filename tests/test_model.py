from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import lcm
from lcm import Model, Regime
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.grids import DiscreteGrid
from lcm.input_processing.regime_processing import create_default_regime_id_cls
from lcm.model import validate_regime_id_cls


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
            # Invalid probability (0.5 instead of 1.0), should be ignored
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5])),
        },
    )
    with pytest.warns(UserWarning, match="will be ignored"):
        model = Model(regimes=regime, n_periods=2)
    # Model should still work
    assert model.internal_regimes is not None


def test_multi_regime_without_next_regime_raises(binary_category_class):
    """Multi-regime models must have next_regime in each regime."""

    @dataclass
    class RegimeID:
        regime1: int = 0
        regime2: int = 1

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
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
    )
    with pytest.raises(ModelInitializationError, match="next_regime"):
        Model(regimes=[regime1, regime2], n_periods=2, regime_id_cls=RegimeID)


def test_single_regime_with_regime_id_cls_warns(binary_category_class):
    """Single-regime models with user-defined regime_id_cls should warn and ignore."""

    @dataclass
    class RegimeID:
        test: int = 0

    regime = Regime(
        name="test",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={"test": {"next_health": lambda health: health}},
    )
    with pytest.warns(UserWarning, match="will be ignored"):
        model = Model(regimes=regime, n_periods=2, regime_id_cls=RegimeID)
    # Model should still work
    assert model.internal_regimes is not None


def test_multi_regime_without_regime_id_cls_raises(binary_category_class):
    """Multi-regime models must have regime_id_cls provided."""
    regime1 = Regime(
        name="regime1",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "regime1": {"next_health": lambda health: health},
            "regime2": {"next_health": lambda health: health},
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
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
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
    )
    with pytest.raises(ModelInitializationError, match="must be provided"):
        Model(regimes=[regime1, regime2], n_periods=2)


def test_multi_regime_with_invalid_regime_id_cls_raises(binary_category_class):
    """Multi-regime models must have valid regime_id_cls."""

    @dataclass
    class RegimeID:
        regime1: int = 0
        wrong_name: int = 1  # Should be "regime2"

    regime1 = Regime(
        name="regime1",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "regime1": {"next_health": lambda health: health},
            "regime2": {"next_health": lambda health: health},
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
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
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
    )
    with pytest.raises(ModelInitializationError, match="regime_id_cls"):
        Model(regimes=[regime1, regime2], n_periods=2, regime_id_cls=RegimeID)


# ======================================================================================
# Tests for validate_regime_id_cls
# ======================================================================================


def test_validate_regime_id_cls_valid():
    """Valid RegimeID class should return empty error list."""

    @dataclass
    class RegimeID:
        work: int = 0
        retirement: int = 1

    errors = validate_regime_id_cls(RegimeID, ["work", "retirement"])
    assert errors == []


def test_validate_regime_id_cls_missing_regime():
    """RegimeID missing a regime attribute should return error."""

    @dataclass
    class RegimeID:
        work: int = 0

    errors = validate_regime_id_cls(RegimeID, ["work", "retirement"])
    assert len(errors) == 1
    assert "missing attributes" in errors[0]
    assert "retirement" in errors[0]


def test_validate_regime_id_cls_extra_regime():
    """RegimeID with extra attribute should return error."""

    @dataclass
    class RegimeID:
        work: int = 0
        retirement: int = 1
        unknown: int = 2

    errors = validate_regime_id_cls(RegimeID, ["work", "retirement"])
    assert len(errors) == 1
    assert "extra attributes" in errors[0]
    assert "unknown" in errors[0]


def test_validate_regime_id_cls_non_consecutive():
    """RegimeID with non-consecutive values should return error."""

    @dataclass
    class RegimeID:
        work: int = 0
        retirement: int = 2  # Should be 1

    errors = validate_regime_id_cls(RegimeID, ["work", "retirement"])
    assert len(errors) == 1
    assert "consecutive integers" in errors[0]


def test_validate_regime_id_cls_not_dataclass():
    """Non-dataclass should return error."""

    class RegimeID:
        work = 0
        retirement = 1

    errors = validate_regime_id_cls(RegimeID, ["work", "retirement"])
    assert len(errors) == 1
    assert "must be a dataclass" in errors[0]


# ======================================================================================
# Tests for create_default_regime_id_cls
# ======================================================================================


def test_create_default_regime_id_cls():
    """Auto-generated RegimeID should be a valid dataclass with correct attribute."""
    regime_id_cls = create_default_regime_id_cls("my_regime")

    # Should be a valid category class
    errors = validate_regime_id_cls(regime_id_cls, ["my_regime"])
    assert errors == []

    # Should have the correct attribute
    assert hasattr(regime_id_cls, "my_regime")
    assert regime_id_cls.my_regime == 0
