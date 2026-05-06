"""Tests for `lcm.dtypes` boundary-cast helpers."""

from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config

from lcm.dtypes import canonical_float_dtype, safe_to_float_dtype, safe_to_int32


@pytest.fixture(name="x64_disabled")
def _fixture_x64_disabled() -> Iterator[None]:
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=False)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)


@pytest.fixture(name="x64_enabled")
def _fixture_x64_enabled() -> Iterator[None]:
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=True)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)


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


def test_canonical_float_dtype_is_float32_under_no_x64(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """`canonical_float_dtype()` is `float32` when `jax_enable_x64=False`."""
    assert canonical_float_dtype() == jnp.float32


def test_canonical_float_dtype_is_float64_under_x64(
    x64_enabled: None,  # noqa: ARG001
) -> None:
    """`canonical_float_dtype()` is `float64` when `jax_enable_x64=True`."""
    assert canonical_float_dtype() == jnp.float64


def test_safe_to_float_dtype_casts_python_float_to_canonical(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """A Python float lands at `float32` under no-x64."""
    out = safe_to_float_dtype(0.5, name="x")
    assert out.dtype == jnp.float32
    assert float(out) == 0.5


def test_safe_to_float_dtype_casts_float64_array_to_float32(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """A `float64` array within float32 range is downcast to `float32`."""
    arr = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    out = safe_to_float_dtype(arr, name="x")
    assert out.dtype == jnp.float32


def test_safe_to_float_dtype_passes_array_through_under_x64(
    x64_enabled: None,  # noqa: ARG001
) -> None:
    """Under x64, a `float64` array is preserved (no down-cast required)."""
    arr = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    out = safe_to_float_dtype(arr, name="x")
    assert out.dtype == jnp.float64


def test_safe_to_float_dtype_raises_on_overflow_when_downcasting(
    x64_disabled: None,  # noqa: ARG001
) -> None:
    """A `float64` value above float32 max raises `OverflowError`, naming the leaf."""
    big = 1e40
    with pytest.raises(OverflowError, match="big_param"):
        safe_to_float_dtype(big, name="big_param")


def test_safe_to_float_dtype_no_overflow_check_when_upcasting(
    x64_enabled: None,  # noqa: ARG001
) -> None:
    """Casting `float32` -> `float64` (up) skips the overflow check."""
    arr = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    out = safe_to_float_dtype(arr, name="x")
    assert out.dtype == jnp.float64
