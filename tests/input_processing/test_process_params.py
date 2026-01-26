"""Tests for process_params function."""

from types import MappingProxyType

import pytest

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.input_processing.process_params import create_params_template, process_params


@pytest.fixture
def params_template():
    """Fixture providing a params_template with two regimes and two functions each."""
    return {
        "regime_0": {
            "fun_0": {"arg_0": float, "arg_1": float},
            "fun_1": {"arg_0": float, "arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": float, "arg_1": float},
            "fun_1": {"arg_0": float, "arg_1": 1.0},
        },
    }


# ======================================================================================
# Tests for valid parameter passing (should eventually pass)
# ======================================================================================


def test_params_at_function_level(params_template):
    """Test 1: Passing params at the function level (same structure as template)."""
    params = {
        "regime_0": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    internal_params = process_params(params, params_template)

    # Check skeleton matches template
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        assert set(internal_params[regime].keys()) == set(
            params_template[regime].keys()
        )
        for func in params_template[regime]:
            assert set(internal_params[regime][func].keys()) == set(  # ty: ignore[possibly-missing-attribute]
                params_template[regime][func].keys()
            )


def test_params_at_regime_level(params_template):
    """Test 2: Passing parameters for a regime at the regime level."""
    params = {
        "regime_0": {"arg_0": 0.0, "arg_1": 1.0},
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    internal_params = process_params(params, params_template)

    # Check skeleton matches template
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        assert set(internal_params[regime].keys()) == set(
            params_template[regime].keys()
        )


def test_params_mixed_regime_function_level(params_template):
    """Test 3: Passing parameters as a mix of regime/function level."""
    params = {
        "regime_0": {
            "arg_0": 0.0,
            "fun_0": {"arg_1": 1.0},
            "fun_1": {"arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    internal_params = process_params(params, params_template)

    # Check skeleton matches template
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        assert set(internal_params[regime].keys()) == set(
            params_template[regime].keys()
        )


def test_params_at_model_level(params_template):
    """Test 4: Passing all parameters at the model level."""
    params = {"arg_0": 0.0, "arg_1": 1.0}
    internal_params = process_params(params, params_template)

    # Check skeleton matches template
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        assert set(internal_params[regime].keys()) == set(
            params_template[regime].keys()
        )


# ======================================================================================
# Tests for ambiguous parameter passing (should raise InvalidNameError)
# ======================================================================================


def test_ambiguous_regime_function_level(params_template):
    """Test 5: Passing ambiguously at regime/function level should raise error."""
    params = {
        "regime_0": {
            "arg_0": 0.0,  # arg_0 at regime level
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},  # arg_0 also in fun_0
            "fun_1": {"arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


def test_ambiguous_model_function_level(params_template):
    """Test 6: Passing ambiguously at model/function level should raise error."""
    params = {
        "arg_0": 0.0,  # arg_0 at model level
        "regime_0": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},  # arg_0 also in fun_0
            "fun_1": {"arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


def test_ambiguous_model_regime_level(params_template):
    """Test 7: Passing ambiguously at model/regime level should raise error."""
    params = {
        "arg_0": 0.0,  # arg_0 at model level
        "regime_0": {
            "arg_0": 0.0,  # arg_0 also at regime level
            "fun_0": {"arg_1": 1.0},
            "fun_1": {"arg_1": 1.0},
        },
        "regime_1": {
            "fun_0": {"arg_0": 0.0, "arg_1": 1.0},
            "fun_1": {"arg_0": 0.0, "arg_1": 1.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


# ======================================================================================
# Tests for name validation
# ======================================================================================


# ======================================================================================
# Tests for name validation in create_params_template
# ======================================================================================


class MockRegime:
    """Mock regime with params_template for testing create_params_template."""

    def __init__(self, params_template: dict) -> None:
        self._params_template = params_template

    @property
    def params_template(self) -> MappingProxyType:
        """Return params_template as MappingProxyType."""
        return MappingProxyType(self._params_template)


def test_function_params_no_qname_separator():
    """Function parameters should not contain the qname separator."""
    internal_regimes = {
        "regime_0": MockRegime(
            {"fun_0": {"arg__0": float}}  # Invalid: contains '__'
        ),
    }
    with pytest.raises(InvalidNameError):
        create_params_template(internal_regimes)  # ty: ignore[invalid-argument-type]


def test_regime_name_no_qname_separator():
    """Regime names should not contain the qname separator."""
    internal_regimes = {
        "regime__0": MockRegime(  # Invalid: contains '__'
            {"fun_0": {"arg_0": float}}
        ),
    }
    with pytest.raises(InvalidNameError):
        create_params_template(internal_regimes)  # ty: ignore[invalid-argument-type]


def test_function_name_no_qname_separator():
    """Function names should not contain the qname separator."""
    internal_regimes = {
        "regime_0": MockRegime(
            {"fun__0": {"arg_0": float}}  # Invalid: contains '__'
        ),
    }
    with pytest.raises(InvalidNameError):
        create_params_template(internal_regimes)  # ty: ignore[invalid-argument-type]


def test_regime_function_names_disjoint():
    """Regime names and function names must be disjoint."""
    # Case: function name same as regime name
    internal_regimes = {
        "regime_0": MockRegime(
            {"regime_0": {"arg_0": float}}  # Invalid: function name = regime name
        ),
    }
    with pytest.raises(InvalidNameError):
        create_params_template(internal_regimes)  # ty: ignore[invalid-argument-type]


def test_regime_argument_names_disjoint():
    """Regime names and argument names must be disjoint."""
    # Case: argument name same as regime name
    internal_regimes = {
        "regime_0": MockRegime(
            {"fun_0": {"regime_0": float}}  # Invalid: arg name = regime name
        ),
    }
    with pytest.raises(InvalidNameError):
        create_params_template(internal_regimes)  # ty: ignore[invalid-argument-type]


# ======================================================================================
# Tests for missing parameters
# ======================================================================================


def test_missing_parameter_raises_error(params_template):
    """Test that missing required parameters raise InvalidParamsError."""
    # Only provide arg_0, but arg_1 is also required in the template
    params = {"arg_0": 0.0}
    with pytest.raises(InvalidParamsError, match="Missing required parameter"):
        process_params(params, params_template)
