"""Boundary-cast helpers that pin user-supplied data to canonical pylcm dtypes.

Used at every API boundary that accepts user data (params, initial
conditions, regime-id arrays) — always called from Python, never inside
JIT. Each helper validates that the value fits the target dtype and
raises a clearly-named error if not.

Casts further down the simulate stack (e.g. transition outputs landing
in the state pool) use plain `.astype` and rely on the boundary cast
above them having already pinned the canonical dtype.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


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
