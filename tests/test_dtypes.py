"""Tests for `lcm.dtypes` boundary-cast helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.dtypes import safe_to_int32


@pytest.mark.parametrize(
    "value",
    [7, np.asarray([0, 1, -3], dtype=np.int64)],
    ids=["python-int", "int64-array"],
)
def test_safe_to_int32_returns_int32(value: object) -> None:
    """`safe_to_int32` returns a `jnp.int32` array for any in-range int input."""
    out = safe_to_int32(value, name="x")
    assert out.dtype == jnp.int32


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (7, 7),
        (np.asarray([0, 1, -3], dtype=np.int64), [0, 1, -3]),
    ],
    ids=["python-int", "int64-array"],
)
def test_safe_to_int32_preserves_in_range_values(
    value: object, expected: object
) -> None:
    """`safe_to_int32` preserves element values for in-range inputs."""
    out = safe_to_int32(value, name="x")
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_safe_to_int32_raises_on_python_int_overflow() -> None:
    """A Python int above int32 max raises `ValueError` naming the leaf."""
    with pytest.raises(ValueError, match="my_param"):
        safe_to_int32(2**32, name="my_param")


def test_safe_to_int32_raises_on_array_overflow() -> None:
    """An int64 array containing values above int32 max raises with the leaf name."""
    # Use numpy here: `jnp.asarray(..., dtype=jnp.int64)` truncates to int32
    # under `jax_enable_x64=False` and trips JAX's own overflow guard before
    # `safe_to_int32` ever sees the value.
    arr = np.asarray([1, 2, 2**32], dtype=np.int64)
    with pytest.raises(ValueError, match="regime"):
        safe_to_int32(arr, name="regime")


def test_safe_to_int32_raises_on_underflow() -> None:
    """A Python int below int32 min raises `ValueError` naming the leaf."""
    with pytest.raises(ValueError, match="offset"):
        safe_to_int32(-(2**40), name="offset")
