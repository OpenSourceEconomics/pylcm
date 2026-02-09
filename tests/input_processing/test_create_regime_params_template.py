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
            "b": DiscreteGrid(binary_category_class),
        },
        n_periods=None,
        utility=lambda a, b, c: None,  # noqa: ARG005
        transitions={
            "next_b": lambda b: b,
        },
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {
        "H": {"discount_factor": float},
        "utility": {"c": "no_annotation_found"},
        "next_b": {},
    }


def test_create_params_with_custom_H_no_extra_params():
    """A custom H with no extra params beyond utility and continuation_value."""

    def custom_H(utility: float, continuation_value: float) -> float:
        return utility + continuation_value

    regime = RegimeMock(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        utility=lambda a, b, c: None,  # noqa: ARG005
        functions={"H": custom_H},
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {"H": {}, "utility": {"c": "no_annotation_found"}}


def test_default_H_with_state_named_discount_factor_raises():
    """Default H has a discount_factor param; a state with the same name must error."""
    regime = RegimeMock(
        actions={"a": None},
        states={"discount_factor": None},
        utility=lambda a, discount_factor: None,  # noqa: ARG005
        transitions={"next_discount_factor": lambda discount_factor: discount_factor},
    )
    with pytest.raises(InvalidNameError, match="shadow state/action"):
        create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]


def test_custom_function_shadowing_state_raises():
    """A custom function whose param name matches a state must error."""

    def custom_H(utility: float, continuation_value: float, wealth: float) -> float:
        return utility + wealth * continuation_value

    regime = RegimeMock(
        actions={"a": None},
        states={"wealth": None},
        utility=lambda a, wealth: None,  # noqa: ARG005
        functions={"H": custom_H},
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
            "wealth": DiscreteGrid(binary_category_class),
        },
        utility=lambda a, wealth, risk_aversion: None,  # noqa: ARG005
        transitions={"next_wealth": lambda wealth: wealth},
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {
        "H": {"discount_factor": float},
        "utility": {"risk_aversion": "no_annotation_found"},
        "next_wealth": {},
    }
