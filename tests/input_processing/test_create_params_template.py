from lcm.grids import DiscreteGrid
from lcm.input_processing.create_params_template import (
    _create_function_params,
    create_params_template,
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
    got = create_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == {
        "discount_factor": float,
        "utility": {"c": "no_annotation_found"},
        "next_b": {},
    }


def test_create_function_params():
    regime = RegimeMock(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        utility=lambda a, b, c: None,  # noqa: ARG005
    )
    got = _create_function_params(regime)  # ty: ignore[invalid-argument-type]
    assert got == {"utility": {"c": "no_annotation_found"}}
