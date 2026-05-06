"""Boundary-cast helpers that pin user-supplied data to canonical pylcm dtypes.

These helpers run **outside JIT** at every API boundary (params, initial
conditions, regime-id arrays). They check the value fits the target dtype
and raise a clearly-named error if not. Inside-JIT casts (e.g. on
transition outputs landing in the simulate state pool) keep the silent
saturation/wrap semantics — overflow there means a broken user transition,
which is out of scope for the boundary helpers.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def canonical_float_dtype() -> jnp.dtype:
    """Return pylcm's canonical float dtype, derived from `jax_enable_x64`.

    Returns `jnp.float64` if `jax.config.jax_enable_x64` is True,
    otherwise `jnp.float32`. The value is read at call time, not at
    import, so toggling the JAX config (e.g. between tests) is honoured.
    """
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


def safe_to_int32(value: object, *, name: str) -> Array:
    """Cast a scalar, sequence, or array to `jnp.int32`, checking int32 range.

    Args:
        value: A Python int, numpy/JAX integer scalar, or array-like of
            integer values.
        name: Qualified name of the leaf — surfaced in the error message
            so the user can locate the offending input.

    Returns:
        A `jnp.int32` array (0-d if `value` was a scalar).

    Raises:
        ValueError: If any element of `value` is outside the int32 range
            `[-2**31, 2**31 - 1]`. The message names the leaf via `name`.

    """
    np_value = np.asarray(value)
    if np_value.size > 0:
        lo = int(np_value.min())
        hi = int(np_value.max())
        if lo < _INT32_MIN or hi > _INT32_MAX:
            msg = (
                f"{name}: int32 overflow — value range [{lo}, {hi}] "
                f"exceeds [{_INT32_MIN}, {_INT32_MAX}]."
            )
            raise ValueError(msg)
    return jnp.asarray(np_value, dtype=jnp.int32)


def safe_to_float_dtype(value: object, *, name: str) -> Array:
    """Cast a scalar, sequence, or array to the canonical float dtype.

    When the cast is *down* (float64 -> float32 under `jax_enable_x64=False`),
    check that no element exceeds `float32` magnitude — raising
    `OverflowError` if so rather than letting JAX silently saturate to
    `±inf`. Up-casts and same-width casts skip the range check; precision
    loss within range is *not* an error (it is an inherent consequence of
    `jax_enable_x64=False`).

    Args:
        value: A Python float, numpy/JAX scalar, or array-like.
        name: Qualified name of the leaf — surfaced in the error message.

    Returns:
        A JAX array at `canonical_float_dtype()` (0-d if `value` was a
        scalar).

    Raises:
        OverflowError: If down-casting to `float32` would saturate any
            element to `±inf`. The message names the leaf via `name`.

    """
    target_dtype = canonical_float_dtype()
    np_value = np.asarray(value)
    if target_dtype == jnp.float32 and np_value.size > 0:
        max_mag = float(np.max(np.abs(np_value)))
        if max_mag > _FLOAT32_MAX:
            msg = (
                f"{name}: float32 overflow — max |value| {max_mag:g} "
                f"exceeds float32 max {_FLOAT32_MAX:g}."
            )
            raise OverflowError(msg)
    return jnp.asarray(np_value, dtype=target_dtype)
