"""Tests for `lcm.dtypes` boundary-cast helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.dtypes import safe_to_int32


def test_safe_to_int32_casts_python_int_in_range() -> None:
    """A Python int within int32 range becomes a `jnp.int32` 0-d array."""
    out = safe_to_int32(7, name="x")
    assert out.dtype == jnp.int32
    assert int(out) == 7


def test_safe_to_int32_casts_int64_array_in_range() -> None:
    """An int64 array within int32 range becomes int32 with the same values."""
    arr = jnp.asarray([0, 1, -3], dtype=jnp.int64)
    out = safe_to_int32(arr, name="x")
    assert out.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out), [0, 1, -3])


def test_safe_to_int32_raises_on_python_int_overflow() -> None:
    """A Python int above int32 max raises `ValueError` naming the leaf."""
    with pytest.raises(ValueError, match="my_param"):
        safe_to_int32(2**32, name="my_param")


def test_safe_to_int32_raises_on_array_overflow() -> None:
    """An int64 array containing values above int32 max raises with the leaf name."""
    arr = jnp.asarray([1, 2, 2**32], dtype=jnp.int64)
    with pytest.raises(ValueError, match="regime"):
        safe_to_int32(arr, name="regime")


def test_safe_to_int32_raises_on_underflow() -> None:
    """A Python int below int32 min raises `ValueError` naming the leaf."""
    with pytest.raises(ValueError, match="offset"):
        safe_to_int32(-(2**40), name="offset")
