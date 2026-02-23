import pytest

from lcm.exceptions import InvalidNameError
from lcm.grids import DiscreteGrid
from lcm.input_processing.create_regime_params_template import (
    create_regime_params_template,
)
from tests.regime_mock import RegimeMock


def test_create_params_without_shocks(binary_category_class):
    regime = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "b": DiscreteGrid(binary_category_class, transition=lambda b: b),
        },
        transition=lambda: 0,
        functions={"utility": lambda a, b, c: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {
        "H": {"discount_factor": float},
        "utility": {"c": "no_annotation_found"},
        "next_b": {},
        "next_regime": {},
    }


def test_create_params_with_custom_H_no_extra_params():
    """A custom H with no extra params beyond utility and E_next_V."""

    def custom_H(utility: float, E_next_V: float) -> float:
        return utility + E_next_V

    regime = RegimeMock(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        functions={"utility": lambda a, b, c: None, "H": custom_H},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {"H": {}, "utility": {"c": "no_annotation_found"}}


def test_default_H_with_state_named_discount_factor_raises():
    """Default H has a discount_factor param; a state with the same name must error."""
    regime = RegimeMock(
        actions={"a": None},
        states={"discount_factor": None},
        functions={"utility": lambda a, discount_factor: None},  # noqa: ARG005
        transition=lambda discount_factor: discount_factor,
    )
    with pytest.raises(InvalidNameError, match="shadow state/action"):
        create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]


def test_custom_function_shadowing_state_raises():
    """A custom function whose param name matches a state must error."""

    def custom_H(utility: float, E_next_V: float, wealth: float) -> float:
        return utility + wealth * E_next_V

    regime = RegimeMock(
        actions={"a": None},
        states={"wealth": None},
        functions={"utility": lambda a, wealth: None, "H": custom_H},  # noqa: ARG005
    )
    with pytest.raises(InvalidNameError, match="shadow state/action"):
        create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]


def test_regular_function_taking_state_as_argument_no_error(binary_category_class):
    """Regular functions that use states as arguments should not trigger the error."""
    regime = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "wealth": DiscreteGrid(
                binary_category_class, transition=lambda wealth: wealth
            ),
        },
        transition=lambda: 0,
        functions={"utility": lambda a, wealth, risk_aversion: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {
        "H": {"discount_factor": float},
        "utility": {"risk_aversion": "no_annotation_found"},
        "next_wealth": {},
        "next_regime": {},
    }
