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
            utility=lambda: 0,
        )


def test_model_invalid_actions():
    with pytest.raises(ModelInitilizationError, match="actions must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            actions="exercise",  # type: ignore[arg-type]
            utility=lambda: 0,
        )


def test_model_invalid_functions():
    with pytest.raises(ModelInitilizationError, match="functions must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            actions={},
            utility=lambda: 0,
            functions="utility",  # type: ignore[arg-type]
        )


def test_model_invalid_functions_values():
    with pytest.raises(
        ModelInitilizationError, match=r"function values must be a callable, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            utility=lambda: 0,
            functions={"function": 0},  # type: ignore[dict-item]
        )


def test_model_invalid_functions_keys():
    with pytest.raises(
        ModelInitilizationError, match=r"function keys must be a strings, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            utility=lambda: 0,
            functions={0: lambda: 0},  # type: ignore[dict-item]
        )


def test_model_invalid_actions_values():
    with pytest.raises(
        ModelInitilizationError, match=r"actions value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={},
            actions={"exercise": 0},  # type: ignore[dict-item]
            utility=lambda: 0,
        )


def test_model_invalid_states_values():
    with pytest.raises(
        ModelInitilizationError, match=r"states value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={"health": 0},  # type: ignore[dict-item]
            actions={},
            utility=lambda: 0,
        )


def test_model_invalid_n_periods():
    with pytest.raises(
        ModelInitilizationError, match=r"Number of periods must be a positive integer."
    ):
        Model(
            n_periods=0,
            states={},
            actions={},
            utility=lambda: 0,
        )


def test_model_missing_next_func(binary_category_class):
    with pytest.raises(
        ModelInitilizationError,
        match=r"Each state must have a corresponding transition function.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"exercise": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
        )


def test_model_invalid_utility():
    with pytest.raises(
        ModelInitilizationError,
        match=(r"utility must be a callable."),
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions={},
            utility=0,  # type: ignore[arg-type]
        )


def test_model_invalid_transition_names():
    with pytest.raises(
        ModelInitilizationError,
        match=(r"Each transitions name must start with 'next_'."),
    ):
        Model(
            n_periods=2,
            states={},
            actions={},
            functions={},
            utility=lambda: 0,
            transitions={"invalid_name": lambda: 0},
        )


def test_model_overlapping_states_actions(binary_category_class):
    with pytest.raises(
        ModelInitilizationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
        )
