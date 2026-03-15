"""Tests for AST-based probs_array indexing validation."""

import warnings
from typing import Any

import pytest

from lcm.error_handling import _validate_probs_array_indexing

# ---------------------------------------------------------------------------
# Detectable: correct order — should pass silently
# ---------------------------------------------------------------------------


def _good_multi(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period, health]


def _good_single(health: int, probs_array: Any) -> Any:
    return probs_array[health]


def _good_three(period: int, work: int, partner: int, probs_array: Any) -> Any:
    return probs_array[period, work, partner]


@pytest.mark.parametrize("func", [_good_multi, _good_single, _good_three])
def test_correct_order_passes(func: Any) -> None:
    _validate_probs_array_indexing(func)


# ---------------------------------------------------------------------------
# Detectable: wrong order — should raise ValueError
# ---------------------------------------------------------------------------


def _bad_swapped(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[health, period]


def _bad_swapped_three(period: int, work: int, partner: int, probs_array: Any) -> Any:
    return probs_array[partner, work, period]


def _bad_subset(period: int, health: int, probs_array: Any) -> Any:  # noqa: ARG001
    return probs_array[period]


def _bad_extra(health: int, probs_array: Any) -> Any:
    return probs_array[health, health]


def _bad_wrong_single(period: int, health: int, probs_array: Any) -> Any:  # noqa: ARG001
    return probs_array[period]


@pytest.mark.parametrize(
    "func",
    [_bad_swapped, _bad_swapped_three, _bad_subset, _bad_extra, _bad_wrong_single],
)
def test_wrong_order_raises(func: Any) -> None:
    with pytest.raises(ValueError, match="probs_array"):
        _validate_probs_array_indexing(func)


# ---------------------------------------------------------------------------
# Non-detectable: should warn (not raise)
# ---------------------------------------------------------------------------


def _computed_index(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period - 1, health]


def _mixed_bare_and_computed(period: int, health: int, probs_array: Any) -> Any:
    return probs_array[period, health - 1]


def _multiple_subscripts(period: int, health: int, probs_array: Any) -> Any:
    if period == 0:
        return probs_array[health]
    return probs_array[period, health]


@pytest.mark.parametrize(
    "func",
    [_computed_index, _mixed_bare_and_computed, _multiple_subscripts],
)
def test_non_detectable_warns(func: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_probs_array_indexing(func)
    assert any("probs_array" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Aliased variable — bare name mismatch, caught as ValueError
# ---------------------------------------------------------------------------


def _aliased_variable(period: int, health: int, probs_array: Any) -> Any:
    idx = period
    return probs_array[idx, health]


def test_aliased_variable_raises() -> None:
    with pytest.raises(ValueError, match="probs_array"):
        _validate_probs_array_indexing(_aliased_variable)


# ---------------------------------------------------------------------------
# Edge: no probs_array subscript at all — should warn
# ---------------------------------------------------------------------------


def _no_subscript(period: int, health: int, probs_array: Any) -> Any:  # noqa: ARG001
    return probs_array


def test_no_subscript_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_probs_array_indexing(_no_subscript)
    assert any("probs_array" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Edge: no probs_array param — should skip silently
# ---------------------------------------------------------------------------


def _no_probs_param(period: int, health: int) -> int:
    return period + health


def test_no_probs_param_skips() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_probs_array_indexing(_no_probs_param)
    assert not caught


# ---------------------------------------------------------------------------
# Edge: lambda — getsource fails, should skip silently
# ---------------------------------------------------------------------------


def test_lambda_skips() -> None:
    func = lambda health, probs_array: probs_array[health]  # noqa: E731
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_probs_array_indexing(func)
    assert not caught
