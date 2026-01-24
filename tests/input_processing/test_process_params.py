"""Tests for process_params function."""

import pytest

from lcm.exceptions import InvalidNameError
from lcm.input_processing.process_params import process_params


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
            assert set(internal_params[regime][func].keys()) == set(
                params_template[regime][func].keys()
            )


@pytest.mark.xfail(reason="Not yet implemented: params at regime level")
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


@pytest.mark.xfail(reason="Not yet implemented: mixed regime/function level params")
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


@pytest.mark.xfail(reason="Not yet implemented: params at model level")
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


@pytest.mark.xfail(reason="Not yet implemented: ambiguous params detection")
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


@pytest.mark.xfail(reason="Not yet implemented: ambiguous params detection")
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


@pytest.mark.xfail(reason="Not yet implemented: ambiguous params detection")
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


@pytest.mark.xfail(reason="Not yet implemented: qname_separator validation")
def test_function_params_no_qname_separator():
    """Function parameters should not contain the qname separator."""
    params_template = {
        "regime_0": {
            "fun_0": {"arg__0": float},  # Invalid: contains '__'
        },
    }
    params = {
        "regime_0": {
            "fun_0": {"arg__0": 0.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


@pytest.mark.xfail(reason="Not yet implemented: disjoint names validation")
def test_regime_function_argument_names_disjoint():
    """Regime names, function names, and argument names must be disjoint sets."""
    # Case: argument name same as function name
    params_template = {
        "regime_0": {
            "fun_0": {"fun_0": float},  # Invalid: arg name = function name
        },
    }
    params = {
        "regime_0": {
            "fun_0": {"fun_0": 0.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


@pytest.mark.xfail(reason="Not yet implemented: disjoint names validation")
def test_regime_function_names_disjoint():
    """Regime names and function names must be disjoint."""
    # Case: function name same as regime name
    params_template = {
        "regime_0": {
            "regime_0": {"arg_0": float},  # Invalid: function name = regime name
        },
    }
    params = {
        "regime_0": {
            "regime_0": {"arg_0": 0.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)


@pytest.mark.xfail(reason="Not yet implemented: disjoint names validation")
def test_regime_argument_names_disjoint():
    """Regime names and argument names must be disjoint."""
    # Case: argument name same as regime name
    params_template = {
        "regime_0": {
            "fun_0": {"regime_0": float},  # Invalid: arg name = regime name
        },
    }
    params = {
        "regime_0": {
            "fun_0": {"regime_0": 0.0},
        },
    }
    with pytest.raises(InvalidNameError):
        process_params(params, params_template)
