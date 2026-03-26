import pytest

from lcm._utils.containers import ensure_containers_are_immutable
from lcm.exceptions import InvalidNameError
from lcm.grids import DiscreteGrid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.params.regime_template import (
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
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={"utility": lambda a, b, c: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == ensure_containers_are_immutable(
        {
            "H": {"discount_factor": "float"},
            "utility": {"c": "no_annotation_found"},
            "next_b": {},
            "next_regime": {},
        }
    )


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
    assert got == ensure_containers_are_immutable(
        {"H": {}, "utility": {"c": "no_annotation_found"}}
    )


def test_default_H_with_state_named_discount_factor_raises():
    """Default H has a discount_factor param; a state with the same name must error."""
    regime = RegimeMock(
        actions={"a": None},
        states={"discount_factor": None},
        state_transitions={"discount_factor": None},
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


def test_solve_simulate_pair_template_contains_union_of_params() -> None:
    """Template for a SolveSimulateFunctionPair H contains params from both variants.

    The solve variant (`exponential_H`) takes `discount_factor`; the simulate
    variant (`beta_delta_H`) takes `beta` and `delta`. The template must contain
    all three so the user can provide a single flat params dict that satisfies
    both phases.
    """

    def exponential_h(utility: float, E_next_V: float, discount_factor: float) -> float:
        return utility + discount_factor * E_next_V

    def beta_delta_h(
        utility: float, E_next_V: float, beta: float, delta: float
    ) -> float:
        return utility + beta * delta * E_next_V

    regime = RegimeMock(
        actions={"a": None},
        states={"b": None},
        functions={  # ty: ignore[invalid-argument-type]
            "utility": lambda a, b: None,  # noqa: ARG005
            "H": SolveSimulateFunctionPair(solve=exponential_h, simulate=beta_delta_h),
        },
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert set(got["H"]) == {"discount_factor", "beta", "delta"}


def test_regular_function_taking_state_as_argument_no_error(binary_category_class):
    """Regular functions that use states as arguments should not trigger the error."""
    regime = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "wealth": DiscreteGrid(binary_category_class),
        },
        state_transitions={"wealth": lambda wealth: wealth},
        transition=lambda: 0,
        functions={"utility": lambda a, wealth, risk_aversion: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)  # ty: ignore[invalid-argument-type]
    assert got == ensure_containers_are_immutable(
        {
            "H": {"discount_factor": "float"},
            "utility": {"risk_aversion": "no_annotation_found"},
            "next_wealth": {},
            "next_regime": {},
        }
    )
