from __future__ import annotations

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.grids import DiscreteGrid
from lcm.regime import Regime


def test_regime_invalid_states():
    with pytest.raises(RegimeInitializationError, match="states must be a dictionary"):
        Regime(
            name="foo",
            active=[0],
            states="health",  # type: ignore[arg-type]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_actions():
    with pytest.raises(RegimeInitializationError, match="actions must be a dictionary"):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions="exercise",  # type: ignore[arg-type]
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_functions():
    with pytest.raises(
        RegimeInitializationError, match="functions must be a dictionary"
    ):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions={},
            functions="utility",  # type: ignore[arg-type]
        )


def test_regime_invalid_functions_values():
    msg = (
        "function values must be callables, but the following values are not: 0 "
        r"\(type: int\)\."
    )
    with pytest.raises(RegimeInitializationError, match=msg):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions={},
            functions={"utility": 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_functions_keys():
    msg = (
        "function keys must be strings, but the following keys are not: 0 "
        r"\(type: int\)\."
    )
    with pytest.raises(RegimeInitializationError, match=msg):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions={},
            functions={0: lambda: 0},  # type: ignore[dict-item]
        )


def test_regime_invalid_actions_values():
    with pytest.raises(
        RegimeInitializationError,
        match="actions value 0 must be a PyLCM grid, such as lcm.DiscreteGrid or",
    ):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions={"exercise": 0},  # type: ignore[dict-item]
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_states_values():
    with pytest.raises(
        RegimeInitializationError,
        match="states value 0 must be a PyLCM grid, such as lcm.DiscreteGrid or",
    ):
        Regime(
            name="foo",
            active=[0],
            states={"health": 0},  # type: ignore[dict-item]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_regime_missing_next_func(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match="Each state must have a corresponding next state function.",
    ):
        Regime(
            name="foo",
            active=[0],
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"exercise": DiscreteGrid(binary_category_class)},
            functions={"utility": lambda: 0},
        )


def test_regime_missing_utility():
    msg = (
        "Utility function is not defined. PyLCM expects a function with dictionary key "
        "'utility' in the functions dictionary."
    )
    with pytest.raises(RegimeInitializationError, match=msg):
        Regime(
            name="foo",
            active=[0],
            states={},
            actions={},
            functions={},
        )


def test_regime_overlapping_states_actions(binary_category_class):
    with pytest.raises(
        RegimeInitializationError,
        match="States and actions cannot have overlapping names.",
    ):
        Regime(
            name="foo",
            active=[0],
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            functions={"utility": lambda: 0},
        )
