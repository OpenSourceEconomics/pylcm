"""Tests for process_params function."""

from types import MappingProxyType

import pytest

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.input_processing.params_processing import (
    create_params_template,
    process_params,
)


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

    # Check that output has regime-level keys
    assert set(internal_params.keys()) == set(params_template.keys())
    # Check that output is flat per regime (function__param format)
    for regime in params_template:
        expected_flat_keys = set()
        for func, func_params in params_template[regime].items():
            for arg in func_params:
                expected_flat_keys.add(f"{func}__{arg}")
        assert set(internal_params[regime].keys()) == expected_flat_keys


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

    # Check that output has regime-level keys with flat format
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        expected_flat_keys = set()
        for func, func_params in params_template[regime].items():
            for arg in func_params:
                expected_flat_keys.add(f"{func}__{arg}")
        assert set(internal_params[regime].keys()) == expected_flat_keys


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

    # Check that output has regime-level keys with flat format
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        expected_flat_keys = set()
        for func, func_params in params_template[regime].items():
            for arg in func_params:
                expected_flat_keys.add(f"{func}__{arg}")
        assert set(internal_params[regime].keys()) == expected_flat_keys


def test_params_at_model_level(params_template):
    """Test 4: Passing all parameters at the model level."""
    params = {"arg_0": 0.0, "arg_1": 1.0}
    internal_params = process_params(params, params_template)

    # Check that output has regime-level keys with flat format
    assert set(internal_params.keys()) == set(params_template.keys())
    for regime in params_template:
        expected_flat_keys = set()
        for func, func_params in params_template[regime].items():
            for arg in func_params:
                expected_flat_keys.add(f"{func}__{arg}")
        assert set(internal_params[regime].keys()) == expected_flat_keys


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
    """Mock regime with regime_params_template for testing create_params_template."""

    def __init__(self, regime_params_template: dict) -> None:
        self._regime_params_template = regime_params_template

    @property
    def regime_params_template(self) -> MappingProxyType:
        """Return regime_params_template as MappingProxyType."""
        return MappingProxyType(self._regime_params_template)


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


# ======================================================================================
# Tests for unknown parameters
# ======================================================================================


def test_unknown_keys_raises_error(params_template):
    """Test that unknown parameter keys raise InvalidParamsError."""
    params = {
        "arg_0": 0.0,
        "arg_1": 1.0,
        "unknown_arg": 2.0,  # Not in template
    }
    with pytest.raises(InvalidParamsError, match="Unknown keys"):
        process_params(params, params_template)


def test_passing_same_params_to_regimes_with_different_templates():
    """Test that passing same params dict to regimes with different templates fails.

    This documents expected behavior when:
    1. A non-terminal regime (e.g., 'alive') has many functions with parameters
    2. A terminal regime (e.g., 'dead') has only a simple utility function
    3. The user attempts to pass the same params dict to both regimes

    The terminal regime's regime_params_template only contains
    {"discount_factor": float} because terminal regimes have no transitions,
    constraints, or auxiliary functions.

    When the user does: params={"alive": shared_params, "dead": shared_params}

    The process_params function correctly rejects the extra keys in the 'dead' part
    as "Unknown keys" because they don't exist in the dead regime's template.

    WORKAROUND: Users should pass only the parameters needed by each regime:
        params = {
            "alive": shared_params,
            "dead": {"discount_factor": 1.0},
        }
    """
    # Template for a non-terminal regime with functions that have parameters
    alive_template = {
        "discount_factor": float,
        "utility": {"beta_mean": float, "beta_std": float},
        "cons_util": {"sigma": float, "bb": float, "kappa": float},
        "next_health": {"health_transition": float},
    }

    # Template for a terminal regime - only has discount_factor
    dead_template = {
        "discount_factor": float,
    }

    params_template = {
        "alive": alive_template,
        "dead": dead_template,
    }

    # User's shared params dict - has all parameters for the alive regime
    shared_params = {
        "discount_factor": 1.0,
        "utility": {"beta_mean": 0.95, "beta_std": 0.02},
        "cons_util": {"sigma": 2.0, "bb": 13.0, "kappa": 0.87},
        "next_health": {"health_transition": 0.5},
    }

    # This is what the user might do - pass same params to both regimes
    # This fails because dead regime doesn't expect cons_util, utility, etc.
    params = {
        "alive": shared_params,
        "dead": shared_params,  # This has keys not in dead's template
    }

    # InvalidParamsError is raised because dead__cons_util__*, dead__utility__*,
    # etc. are not in the template
    with pytest.raises(InvalidParamsError, match="Unknown keys"):
        process_params(params, params_template)  # ty: ignore[invalid-argument-type]


@pytest.mark.xfail(
    reason=(
        "ShockGrid params should be passable via regular params, not just via "
        "fixed_params or shock_params. The params_template should include "
        "ShockGrid discretization parameters for the corresponding transition."
    ),
    strict=True,
)
def test_shock_params_via_regular_params():
    """Test that ShockGrid params can be passed via regular params.

    Currently, when a regime has a ShockGrid state (e.g., adjustment_cost with
    uniform distribution), the transition function signature is:
        def next_adjustment_cost(adjustment_cost): ...
    which has no start/stop parameters.

    The params_template is built from function signatures, so next_adjustment_cost
    gets an empty dict {} in the template. Passing {"start": 0, "stop": 1} via
    regular params fails with "Unknown keys".

    DESIRED BEHAVIOR: The params_template should include ShockGrid discretization
    parameters, allowing users to pass them via regular params just like any other
    function parameter.

    CURRENT WORKAROUND: Pass ShockGrid params via:
    1. fixed_params on Model: Model(..., fixed_params={"adjustment_cost": {"start": 0}})
    2. shock_params on ShockGrid: ShockGrid(..., shock_params={"start": 0, "stop": 1})
    """
    # Template where next_adjustment_cost has no parameters (empty dict)
    # because the transition function signature is: def next_adjustment_cost(x): ...
    params_template = {
        "working": {
            "discount_factor": float,
            "utility": {"param": float},
            "next_adjustment_cost": {},  # Empty - function has no param arguments
        },
    }

    # User tries to pass ShockGrid params via regular params
    params = {
        "working": {
            "discount_factor": 1.0,
            "utility": {"param": 0.5},
            "next_adjustment_cost": {
                "start": 0,
                "stop": 1,
            },  # These are ShockGrid params
        },
    }

    # This should succeed - ShockGrid params should be accepted via regular params
    result = process_params(params, params_template)  # ty: ignore[invalid-argument-type]
    assert result["working"]["next_adjustment_cost"]["start"] == 0  # ty: ignore[not-subscriptable]
    assert result["working"]["next_adjustment_cost"]["stop"] == 1  # ty: ignore[not-subscriptable]
