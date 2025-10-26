from __future__ import annotations

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.grids import DiscreteGrid
from lcm.regime import Regime


def test_regime_invalid_states():
    with pytest.raises(RegimeInitializationError, match="states must be a dictionary"):
        Regime(
            name="test",
            active=list(range(2)),
            states="health",  # type: ignore[arg-type]
            actions={},
            utility=lambda: 0,
        )


def test_regime_invalid_actions():
    with pytest.raises(RegimeInitializationError, match="actions must be a dictionary"):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions="exercise",  # type: ignore[arg-type]
            utility=lambda: 0,
        )


def test_regime_invalid_functions():
    with pytest.raises(
        RegimeInitializationError, match="functions must be a dictionary"
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions={},
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
            active=list(range(2)),
            states={},
            actions={},
            utility=lambda: 0,
            functions={"function": 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_functions_keys():
    with pytest.raises(
        RegimeInitializationError, match=r"function keys must be a strings, but is 0."
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions={},
            utility=lambda: 0,
            functions={0: lambda: 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_actions_values():
    with pytest.raises(
        RegimeInitializationError, match=r"actions value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions={"exercise": 0},  # type: ignore[dict-item]
            utility=lambda: 0,
        )


def test_regime_invalid_states_values():
    with pytest.raises(
        RegimeInitializationError, match=r"states value 0 must be an LCM grid."
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={"health": 0},  # type: ignore[dict-item]
            actions={},
            utility=lambda: 0,
        )


def test_regime_invalid_n_periods():
    with pytest.raises(
        RegimeInitializationError,
        match=r"Number of periods must be a positive integer.",
    ):
        Regime(
            name="test",
            active=list(range(0)),
            states={},
            actions={},
            utility=lambda: 0,
        )


def test_regime_missing_next_func(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match=r"Each state must have a corresponding transition function.",
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"exercise": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
        )


def test_regime_invalid_utility():
    with pytest.raises(
        RegimeInitializationError,
        match=(r"utility must be a callable."),
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions={},
            functions={},
            utility=0,  # type: ignore[arg-type]
        )


def test_regime_invalid_transition_names():
    with pytest.raises(
        RegimeInitializationError,
        match=(r"Each transitions name must start with 'next_'."),
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={},
            actions={},
            functions={},
            utility=lambda: 0,
            transitions={"invalid_name": lambda: 0},
        )


def test_regime_overlapping_states_actions(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        Regime(
            name="test",
            active=list(range(2)),
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
        )
