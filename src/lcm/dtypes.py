"""Boundary-cast helpers that pin user-supplied data to canonical pylcm dtypes.

These helpers run **outside JIT** at every API boundary (params, initial
conditions, regime-id arrays). They check the value fits the target dtype
and raise a clearly-named error if not. Inside-JIT casts (e.g. on
transition outputs landing in the simulate state pool) keep the silent
saturation/wrap semantics — overflow there means a broken user transition,
which is out of scope for the boundary helpers.
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
