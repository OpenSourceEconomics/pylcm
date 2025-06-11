from __future__ import annotations

import pytest

from lcm.exceptions import ModelInitilizationError
from lcm.grids import DiscreteGrid
from lcm.user_model import Model


def test_model_invalid_states():
    with pytest.raises(ModelInitilizationError, match="states must be a dictionary"):
        Model(
            n_periods=2,
            states="health",  # type: ignore[arg-type]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_model_invalid_actions():
    with pytest.raises(ModelInitilizationError, match="actions must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            actions="exercise",  # type: ignore[arg-type]
            functions={"utility": lambda: 0},
        )


def test_model_invalid_functions():
    with pytest.raises(ModelInitilizationError, match="functions must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions="utility",  # type: ignore[arg-type]
        )


def test_model_invalid_functions_values():
    with pytest.raises(
        ModelInitilizationError, match="function values must be a callable, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions={"utility": 0},  # type: ignore[dict-item]
        )


def test_model_invalid_functions_keys():
    with pytest.raises(
        ModelInitilizationError, match="function keys must be a strings, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions={0: lambda: 0},  # type: ignore[dict-item]
        )


def test_model_invalid_actions_values():
    with pytest.raises(
        ModelInitilizationError, match="actions value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={},
            actions={"exercise": 0},  # type: ignore[dict-item]
            functions={"utility": lambda: 0},
        )


def test_model_invalid_states_values():
    with pytest.raises(
        ModelInitilizationError, match="states value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={"health": 0},  # type: ignore[dict-item]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_model_invalid_n_periods():
    with pytest.raises(
        ModelInitilizationError, match="Number of periods must be a positive integer."
    ):
        Model(
            n_periods=0,
            states={},
            actions={},
            functions={"utility": lambda: 0},
        )


def test_model_missing_next_func(binary_category_class):
    with pytest.raises(
        ModelInitilizationError,
        match="Each state must have a corresponding next state function.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"exercise": DiscreteGrid(binary_category_class)},
            functions={"utility": lambda: 0},
        )


def test_model_missing_utility():
    with pytest.raises(
        ModelInitilizationError,
        match="Utility function is not defined. LCM expects a function called 'utility",
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions={},
        )


def test_model_overlapping_states_actions(binary_category_class):
    with pytest.raises(
        ModelInitilizationError,
        match="States and actions cannot have overlapping names.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            functions={"utility": lambda: 0},
        )
