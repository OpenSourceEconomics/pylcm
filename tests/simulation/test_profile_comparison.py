"""Negative controls for the parity comparisons the execution-equivalence tests use.

Each control feeds a deliberately wrong output into the assertion entry point a
parity test actually calls and requires it to be rejected, so a comparator that
admits the defect is caught here rather than by a silently green parity test.
"""

import numpy as np
import pandas as pd
import pytest

from tests.execution.test_axis_widths_per_regime import (
    assert_agrees_to_ulp as value_checker,
)
from tests.simulation._profile_comparison import assert_same_bytes
from tests.simulation.test_subject_batching import (
    _assert_columns_invariant as frame_checker,
)


def _steps_below_one(*, dtype: type[np.floating], steps: int) -> np.ndarray:
    value = np.asarray(1.0, dtype=dtype)
    for _ in range(steps):
        value = np.nextafter(value, np.asarray(0.0, dtype=dtype))
    return value


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("steps", [9, 16])
@pytest.mark.parametrize("sign", [-1, 1])
def test_value_gate_rejects_more_than_eight_steps_across_a_binade_edge(
    *, dtype: type[np.floating], steps: int, sign: int
) -> None:
    """Nine or more representable steps below +-1 fail an eight-step gate."""
    reference = _steps_below_one(dtype=dtype, steps=steps)
    with pytest.raises(AssertionError):
        value_checker(
            got=sign * np.asarray(1.0, dtype=dtype), expected=sign * reference, n_ulp=8
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_value_gate_accepts_exactly_eight_steps(*, dtype: type[np.floating]) -> None:
    """Exactly eight representable steps below 1 pass an eight-step gate."""
    reference = _steps_below_one(dtype=dtype, steps=8)
    value_checker(got=np.asarray(1.0, dtype=dtype), expected=reference, n_ulp=8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_value_gate_rejects_maximum_finite_against_one(
    *, dtype: type[np.floating]
) -> None:
    """The largest finite value never yields a tolerance wide enough to match 1."""
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(AssertionError):
        value_checker(
            got=np.asarray(np.finfo(dtype).max, dtype=dtype),
            expected=np.asarray(1.0, dtype=dtype),
            n_ulp=8,
        )


def test_value_gate_rejects_changed_dtype() -> None:
    """Equal numbers in float64 and float32 are not the same published value."""
    with pytest.raises(AssertionError):
        value_checker(
            got=np.asarray([1.0], dtype=np.float64),
            expected=np.asarray([1.0], dtype=np.float32),
            n_ulp=8,
        )


@pytest.mark.parametrize("column", ["wealth", "consumption", "utility"])
def test_panel_rejects_one_step_in_a_non_value_float_column(*, column: str) -> None:
    """A float64 state, action or target column moved by one step fails the panel."""
    reference = pd.DataFrame(
        {"subject_id": [0], "period": [0], column: np.asarray([1.0], dtype=np.float64)}
    )
    actual = reference.copy()
    actual.loc[0, column] = np.nextafter(np.float64(1.0), np.float64(2.0))
    with pytest.raises(AssertionError):
        frame_checker(baseline=reference, batched=actual)


def test_panel_rejects_a_non_value_dtype_change() -> None:
    """A float32 column delivered as float64 fails the panel."""
    reference = pd.DataFrame(
        {
            "subject_id": [0],
            "period": [0],
            "consumption": np.asarray([1.0], dtype=np.float32),
        }
    )
    with pytest.raises(AssertionError):
        frame_checker(
            baseline=reference, batched=reference.astype({"consumption": np.float64})
        )


def test_panel_accepts_eight_steps_in_the_value_column() -> None:
    """The published `value` column alone receives the eight-step allowance."""
    reference = pd.DataFrame(
        {"subject_id": [0], "value": _steps_below_one(dtype=np.float64, steps=8)[None]}
    )
    actual = reference.copy()
    actual.loc[0, "value"] = 1.0
    frame_checker(baseline=reference, batched=actual)


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (np.asarray([-0.0]), np.asarray([0.0])),
        (
            np.asarray([0x7FF8000000000001], dtype=np.uint64).view(np.float64),
            np.asarray([np.nan]),
        ),
    ],
    ids=["signed_zero", "nan_payload"],
)
def test_same_bytes_rejects_signed_zero_and_nan_payload(
    *, got: np.ndarray, expected: np.ndarray
) -> None:
    """Same-program comparison distinguishes signed zeros and NaN payloads."""
    with pytest.raises(AssertionError):
        assert_same_bytes(got=got, expected=expected)
