"""Tests for AST-based array parameter indexing detection and validation."""

import warnings
from typing import Any

import pytest

from lcm.error_handling import (
    _get_func_indexing_params,
    _validate_array_param_indexing,
)


def _good_multi(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period, health]


def _good_single(health: int, probs_array: Any) -> Any:
    return probs_array[health]


def _good_three(period: int, work: int, partner: int, probs_array: Any) -> Any:
    return probs_array[period, work, partner]


@pytest.mark.parametrize("func", [_good_multi, _good_single, _good_three])
def test_get_func_indexing_params_correct(func: Any) -> None:
    """AST-based detection returns correct indexing param names."""
    indexing = _get_func_indexing_params(func, "probs_array")
    _validate_array_param_indexing(
        func=func, array_param_name="probs_array", indexing_params=indexing
    )


def test_get_func_indexing_params_multi() -> None:
    assert _get_func_indexing_params(_good_multi, "probs_array") == [
        "period",
        "health",
    ]


def test_get_func_indexing_params_single() -> None:
    assert _get_func_indexing_params(_good_single, "probs_array") == ["health"]


def test_get_func_indexing_params_three() -> None:
    assert _get_func_indexing_params(_good_three, "probs_array") == [
        "period",
        "work",
        "partner",
    ]


def _swapped(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[health, period]


def test_get_func_indexing_params_detects_actual_order() -> None:
    """AST returns the ACTUAL subscript order, not signature order."""
    assert _get_func_indexing_params(_swapped, "probs_array") == ["health", "period"]


def test_validate_catches_mismatch_with_expected_order() -> None:
    """Validation catches mismatch when caller provides expected order."""
    with pytest.raises(ValueError, match="probs_array"):
        _validate_array_param_indexing(
            func=_swapped,
            array_param_name="probs_array",
            indexing_params=["period", "health"],
        )


def _computed_index(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period - 1, health]


def _mixed_bare_and_computed(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period, health - 1]


@pytest.mark.parametrize("func", [_computed_index, _mixed_bare_and_computed])
def test_computed_index_raises(func: Any) -> None:
    """Computed indices raise ValueError with recipe for fix."""
    with pytest.raises(ValueError, match="computed indices"):
        _get_func_indexing_params(func, "probs_array")


def _aliased_variable(period: int, health: int, probs_array: Any) -> Any:
    idx = period
    return probs_array[idx, health]


def test_aliased_variable_detected_by_ast() -> None:
    """Aliased variable `idx` is not a function param — raises ValueError."""
    with pytest.raises(ValueError, match="not function parameters"):
        _get_func_indexing_params(_aliased_variable, "probs_array")


def _no_subscript(period: int, health: int, probs_array: Any) -> Any:  # noqa: ARG001
    return probs_array


def test_no_subscript_returns_empty() -> None:
    """Array param without subscript — no indexing params."""
    assert _get_func_indexing_params(_no_subscript, "probs_array") == []


def test_no_subscript_warns_during_validation() -> None:
    """Validation warns when array param has no subscript."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_array_param_indexing(
            func=_no_subscript,
            array_param_name="probs_array",
            indexing_params=[],
        )
    assert any("probs_array" in str(w.message) for w in caught)


def _no_probs_param(period: int, health: int) -> int:
    return period + health


def test_scalar_function_returns_empty() -> None:
    assert _get_func_indexing_params(_no_probs_param, "nonexistent") == []


def test_lambda_raises() -> None:
    """Lambda functions raise TypeError."""
    func = lambda health, probs_array: probs_array[health]  # noqa: E731
    with pytest.raises(TypeError, match="lambda"):
        _get_func_indexing_params(func, "probs_array")


def _custom_array(period: int, health: int, wage_grid: Any) -> Any:
    return wage_grid[period, health]


def test_non_probs_array_param() -> None:
    """Work with array param names other than probs_array."""
    assert _get_func_indexing_params(_custom_array, "wage_grid") == [
        "period",
        "health",
    ]


def _custom_array_wrong_order(period: int, health: int, wage_grid: Any) -> Any:
    return wage_grid[health, period]


def test_non_probs_array_param_actual_order() -> None:
    """AST returns actual subscript order for non-probs_array params."""
    assert _get_func_indexing_params(_custom_array_wrong_order, "wage_grid") == [
        "health",
        "period",
    ]


def _dict_subscript(config: dict, period: int, probs_array: Any) -> Any:
    threshold = config["threshold"]  # noqa: F841
    return probs_array[period]


def test_dict_subscript_not_confused_with_array() -> None:
    """Dict subscripts (string keys) are not mistaken for array indexing."""
    assert _get_func_indexing_params(_dict_subscript, "probs_array") == ["period"]


def _param_subscripted_before_array(
    lookup: Any, period: int, health: int, arr: Any
) -> Any:
    threshold = lookup[period]  # noqa: F841
    return arr[period, health]


def test_array_param_name_skips_false_positive() -> None:
    """Named array_param_name avoids false positive from earlier subscripts."""
    assert _get_func_indexing_params(_param_subscripted_before_array, "arr") == [
        "period",
        "health",
    ]
