"""Parity comparisons between two executions of the same model.

Two families of comparison, kept apart so a value allowance never leaks onto
structure:

- `assert_same_bytes` — same program on the same inputs: every leaf is byte-identical,
  including signed zeros and NaN payloads.
- `assert_value_steps` — independently compiled programs (a different block width,
  batch size or layout): published value leaves may differ by a bounded number of
  ordered representable steps of their own format.

`assert_public_frames` applies both to a published `to_dataframe()` panel: only the
`value` column of an independently compiled run receives the step allowance; states,
actions, regimes, subjects and derived targets stay exact.
"""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def assert_value_steps(
    *,
    got: ArrayLike,
    expected: ArrayLike,
    n_ulp: int = 8,
    err_msg: str = "",
) -> None:
    """Assert two value arrays lie within `n_ulp` ordered representable steps.

    The distance between two finite floats is the number of representable values of
    their format one must step through to get from one to the other. It is computed
    on monotone integer encodings of the IEEE bit patterns, so no operand magnitude,
    spacing, subtraction or promotion enters: a value just below a power of two never
    borrows the coarser spacing above it, and a large value elsewhere in the array
    never widens another element's tolerance. Negative and positive zero are one step
    apart.

    Shapes and dtypes must match exactly. Non-finite entries must agree in position
    and sign; NaN payloads are left to `assert_same_bytes`.

    Args:
        got: Value leaf under the compared execution.
        expected: Value leaf under the reference execution.
        n_ulp: Largest tolerated number of ordered representable steps.
        err_msg: Context appended to the failure message.

    """
    if type(n_ulp) is not int or n_ulp < 0:
        msg = f"n_ulp must be a nonnegative Python int, got {n_ulp!r}"
        raise ValueError(msg)
    actual, reference = _matching_arrays(got=got, expected=expected)
    # Validate both formats even when there are no finite entries to compare.
    _unsigned_words(array=actual)
    _unsigned_words(array=reference)
    np.testing.assert_array_equal(
        np.isnan(actual), np.isnan(reference), err_msg=err_msg
    )
    np.testing.assert_array_equal(
        np.isposinf(actual), np.isposinf(reference), err_msg=err_msg
    )
    np.testing.assert_array_equal(
        np.isneginf(actual), np.isneginf(reference), err_msg=err_msg
    )
    finite = np.isfinite(reference)
    actual_keys = _ordered_keys(array=actual[finite])
    reference_keys = _ordered_keys(array=reference[finite])
    distance = np.maximum(actual_keys, reference_keys) - np.minimum(
        actual_keys, reference_keys
    )
    worst = int(distance.max()) if distance.size else 0
    if worst > n_ulp:
        msg = f"{worst} ordered representable steps exceed {n_ulp} ULP. {err_msg}"
        raise AssertionError(msg)


def assert_same_bytes(
    *, got: ArrayLike, expected: ArrayLike, err_msg: str = ""
) -> None:
    """Assert two numeric leaves are byte-identical, with no broadcasting.

    Signed zeros and NaN payloads count. Shapes and dtypes must match exactly.

    Args:
        got: Leaf under the compared execution.
        expected: Leaf under the reference execution.
        err_msg: Context appended to the failure message.

    """
    actual, reference = _matching_arrays(got=got, expected=expected)
    if actual.dtype.hasobject:
        msg = "Object-pointer bytes are not a semantic equality test"
        raise TypeError(msg)
    if actual.tobytes(order="C") != reference.tobytes(order="C"):
        msg = f"Exact leaf bytes changed. {err_msg}"
        raise AssertionError(msg)


def assert_public_frames(
    *,
    got: pd.DataFrame,
    expected: pd.DataFrame,
    mode: Literal["same_program", "independently_compiled"],
    n_ulp: int = 8,
) -> None:
    """Assert two published panels agree in addresses, schema and values.

    Index and columns must match exactly, in delivered order; nothing is sorted or
    reset. Every column's dtype must match. Float columns are byte-exact except the
    `value` column under `mode="independently_compiled"`, which uses
    `assert_value_steps`. Non-float columns are compared exactly.

    Sharding and other raw-solution metadata that a DataFrame cannot carry remain the
    caller's responsibility.

    Args:
        got: Panel under the compared execution.
        expected: Panel under the reference execution.
        mode: Provenance of the comparison; only independently compiled runs
            receive the value allowance.
        n_ulp: Step allowance for the `value` column.

    """
    if mode not in ("same_program", "independently_compiled"):
        msg = f"Name the comparison provenance explicitly, got {mode!r}"
        raise ValueError(msg)
    pd.testing.assert_index_equal(got.index, expected.index, exact=True)
    pd.testing.assert_index_equal(got.columns, expected.columns, exact=True)
    if not got.columns.is_unique:
        msg = f"Duplicate publication columns: {list(got.columns)}"
        raise AssertionError(msg)
    for name in expected:
        actual, reference = got[name], expected[name]
        if actual.dtype != reference.dtype:
            msg = (
                f"Dtype changed in column {name!r}: {actual.dtype} != {reference.dtype}"
            )
            raise AssertionError(msg)
        if pd.api.types.is_float_dtype(reference.dtype):
            if name == "value" and mode == "independently_compiled":
                assert_value_steps(
                    got=actual.to_numpy(),
                    expected=reference.to_numpy(),
                    n_ulp=n_ulp,
                    err_msg=str(name),
                )
            else:
                assert_same_bytes(
                    got=actual.to_numpy(),
                    expected=reference.to_numpy(),
                    err_msg=str(name),
                )
        else:
            pd.testing.assert_series_equal(actual, reference, check_exact=True)


def _matching_arrays(
    *, got: ArrayLike, expected: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    actual, reference = np.asarray(got), np.asarray(expected)
    if actual.shape != reference.shape:
        msg = f"Shape changed: {actual.shape} != {reference.shape}"
        raise AssertionError(msg)
    if actual.dtype != reference.dtype:
        msg = f"Dtype changed: {actual.dtype} != {reference.dtype}"
        raise AssertionError(msg)
    return actual, reference


def _unsigned_words(*, array: np.ndarray) -> np.ndarray:
    if array.dtype == np.dtype(np.float32):
        return array.view(np.uint32)
    if array.dtype == np.dtype(np.float64):
        return array.view(np.uint64)
    msg = f"Value comparison supports native float32/float64 only, got {array.dtype}"
    raise TypeError(msg)


def _ordered_keys(*, array: np.ndarray) -> np.ndarray:
    """Map IEEE bit patterns to unsigned integers in the order of the reals."""
    words = _unsigned_words(array=array)
    sign = np.asarray(1 << (8 * array.dtype.itemsize - 1), dtype=words.dtype)
    return np.where((words & sign) != 0, ~words, words | sign)
