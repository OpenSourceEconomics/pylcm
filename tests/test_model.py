from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import lcm
from lcm import Model, Regime
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.grids import DiscreteGrid
from lcm.model import _validate_regime_id_cls


def test_regime_invalid_states():
    """Regime rejects non-dict states argument."""
    with pytest.raises(RegimeInitializationError, match="states must be a dictionary"):
        Regime(
            name="test",
            states="health",  # ty: ignore[invalid-argument-type]
            actions={},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_invalid_actions():
    """Regime rejects non-dict actions argument."""
    with pytest.raises(RegimeInitializationError, match="actions must be a dictionary"):
        Regime(
            name="test",
            states={},
            actions="exercise",  # ty: ignore[invalid-argument-type]
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_invalid_functions():
    """Regime rejects non-dict functions argument."""
    with pytest.raises(
        RegimeInitializationError, match="functions must be a dictionary"
    ):
        Regime(
            name="test",
            states={},
            actions={},
            transitions={"next_health": lambda: 0},
            utility=lambda: 0,
            functions="utility",  # ty: ignore[invalid-argument-type]
            active=range(5),
        )


def test_regime_invalid_functions_values():
    """Regime rejects non-callable function values."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"function values must be a callable, but is 0.",
    ):
        Regime(
            name="test",
            states={},
            actions={},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            functions={"function": 0},  # ty: ignore[invalid-argument-type]
            active=range(5),
        )


def test_regime_invalid_functions_keys():
    """Regime rejects non-string function keys."""
    with pytest.raises(
        RegimeInitializationError, match=r"function keys must be a strings, but is 0."
    ):
        Regime(
            name="test",
            states={},
            actions={},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            functions={0: lambda: 0},  # ty: ignore[invalid-argument-type]
            active=range(5),
        )


def test_regime_invalid_actions_values():
    """Regime rejects non-grid action values."""
    with pytest.raises(
        RegimeInitializationError, match=r"actions value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            states={},
            actions={"exercise": 0},  # ty: ignore[invalid-argument-type]
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_invalid_states_values():
    """Regime rejects non-grid state values."""
    with pytest.raises(
        RegimeInitializationError, match=r"states value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            states={"health": 0},  # ty: ignore[invalid-argument-type]
            actions={},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_missing_next_func(binary_category_class):
    """Regime rejects states without corresponding transition functions."""
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
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_invalid_utility():
    """Regime rejects non-callable utility argument."""
    with pytest.raises(
        RegimeInitializationError,
        match=(r"utility must be a callable."),
    ):
        Regime(
            name="test",
            states={},
            actions={},
            functions={},
            utility=0,  # ty: ignore[invalid-argument-type]
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_regime_invalid_transition_names():
    """Regime rejects transition names not starting with 'next_'."""
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
            transitions={"invalid_name": lambda: 0},
            active=range(5),
        )


def test_regime_overlapping_states_actions(binary_category_class):
    """Regime rejects overlapping state and action names."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        Regime(
            name="test",
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=range(5),
        )


def test_model_requires_terminal_regime(binary_category_class):
    """Model must have at least one terminal regime."""

    @dataclass
    class RegimeId:
        test: int = 0

    regime = Regime(
        name="test",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([1.0])),
        },
        active=range(1),
    )
    with pytest.raises(ModelInitializationError, match="at least one terminal regime"):
        Model(regimes=[regime], n_periods=2, regime_id_cls=RegimeId)


def test_model_requires_non_terminal_regime(binary_category_class):
    """Model must have at least one non-terminal regime."""

    @dataclass
    class RegimeId:
        dead: int = 0

    dead = Regime(
        name="dead",
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
        active=[1],
    )
    with pytest.raises(ModelInitializationError, match="at least one non-terminal"):
        Model(regimes=[dead], n_periods=2, regime_id_cls=RegimeId)


def test_multi_regime_without_next_regime_raises(binary_category_class):
    """Multi-regime models must have next_regime in each regime."""

    @dataclass
    class RegimeId:
        regime1: int = 0
        regime2: int = 1

    regime1 = Regime(
        name="regime1",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            # Missing next_regime
        },
        active=range(1),
    )
    regime2 = Regime(
        name="regime2",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=range(1),
    )
    with pytest.raises(ModelInitializationError, match="next_regime"):
        Model(regimes=[regime1, regime2], n_periods=2, regime_id_cls=RegimeId)


def test_model_requires_regime_id_cls():
    """Model requires regime_id_cls as a keyword argument."""
    # regime_id_cls is a required keyword argument, so omitting it raises TypeError
    with pytest.raises(TypeError, match="regime_id_cls"):
        Model(regimes=[], n_periods=2)  # ty: ignore[missing-argument]


def test_multi_regime_with_invalid_regime_id_cls_raises(binary_category_class):
    """Multi-regime models must have valid regime_id_cls."""

    @dataclass
    class RegimeId:
        regime1: int = 0
        wrong_name: int = 1  # Should be "regime2"

    regime1 = Regime(
        name="regime1",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=range(1),
    )
    regime2 = Regime(
        name="regime2",
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=range(1),
    )
    with pytest.raises(ModelInitializationError, match="regime_id_cls"):
        Model(regimes=[regime1, regime2], n_periods=2, regime_id_cls=RegimeId)


def test_validate_regime_id_cls_valid():
    """Valid RegimeId class should return empty error list."""

    @dataclass
    class RegimeId:
        work: int = 0
        retirement: int = 1

    errors = _validate_regime_id_cls(RegimeId, ["work", "retirement"])
    assert errors == []


def test_validate_regime_id_cls_missing_regime():
    """RegimeId missing a regime attribute should return error."""

    @dataclass
    class RegimeId:
        work: int = 0

    errors = _validate_regime_id_cls(RegimeId, ["work", "retirement"])
    assert len(errors) == 1
    assert "missing attributes" in errors[0]
    assert "retirement" in errors[0]


def test_validate_regime_id_cls_extra_regime():
    """RegimeId with extra attribute should return error."""

    @dataclass
    class RegimeId:
        work: int = 0
        retirement: int = 1
        unknown: int = 2

    errors = _validate_regime_id_cls(RegimeId, ["work", "retirement"])
    assert len(errors) == 1
    assert "extra attributes" in errors[0]
    assert "unknown" in errors[0]


def test_validate_regime_id_cls_non_consecutive():
    """RegimeId with non-consecutive values should return error."""

    @dataclass
    class RegimeId:
        work: int = 0
        retirement: int = 2  # Should be 1

    errors = _validate_regime_id_cls(RegimeId, ["work", "retirement"])
    assert len(errors) == 1
    assert "consecutive integers" in errors[0]


def test_validate_regime_id_cls_not_dataclass():
    """Non-dataclass should return error."""

    class RegimeId:
        work = 0
        retirement = 1

    errors = _validate_regime_id_cls(RegimeId, ["work", "retirement"])
    assert len(errors) == 1
    assert "must be a dataclass" in errors[0]


def test_model_accepts_multiple_terminal_regimes(binary_category_class):
    """Model can have multiple terminal regimes."""

    @dataclass
    class RegimeId:
        alive: int = 0
        dead1: int = 1
        dead2: int = 2

    alive = Regime(
        name="alive",
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.8, 0.1, 0.1])),
        },
        active=range(1),
    )
    dead1 = Regime(
        name="dead1",
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
        active=[1],
    )
    dead2 = Regime(
        name="dead2",
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
        active=[1],
    )
    # Should not raise - multiple terminal regimes are allowed
    model = Model(regimes=[alive, dead1, dead2], n_periods=2, regime_id_cls=RegimeId)
    assert model.internal_regimes is not None
