"""Boundary-cast helpers that pin user-supplied data to canonical pylcm dtypes.

Used at every API boundary that accepts user data (params, initial
conditions, regime-id arrays) — always called from Python, never inside
JIT. Each helper validates that the value fits the target dtype and
raises a clearly-named error if not. The user-output adapters at the end
of the module are the one exception: they convert what a user function
returns inside traced code, without validation. Once an input has crossed the
boundary it carries the canonical dtype unchanged through the simulate
stack; downstream code does not re-cast.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from lcm.exceptions import InvalidFunctionOutputError
from lcm.typing import (
    BoolND,
    CanonicalArrayWriter,
    FloatND,
    IntND,
    ValueND,
)

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)

# A numeric value crossing the boundary: a Python or NumPy scalar, or a host or
# device array.
type _NumericValue = int | float | np.integer | np.floating | np.ndarray | ValueND

# What a user function may return where the engine expects one array: a Python
# scalar, a NumPy scalar or array, or a JAX array of any numeric kind.
type UserOutput = bool | int | float | np.generic | np.ndarray | ValueND

# A discrete code from a user function: `int32`, or wider when it arrives wider.
type Code = Int[Array, "..."]

# Names the user function an output-kind error refers to, e.g. "Constraint 'c'".
type OutputSource = str


def canonical_float_dtype() -> type:  # noqa: PAN006 - JAX scalar classes use a private metaclass exposed as callable values in its stubs
    """Return pylcm's canonical float dtype, derived from `jax_enable_x64`.

    Returns `jnp.float64` if `jax.config.jax_enable_x64` is True,
    otherwise `jnp.float32`. The return is a JAX scalar type, passable
    wherever JAX/NumPy accept a dtype-like specifier. The value is read
    at call time, not at import, so toggling the JAX config (e.g.
    between tests) is honoured.
    """
    return cast(
        "type",  # noqa: PAN006 - JAX's private scalar metaclass has no public instance type
        jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32,
    )


def safe_to_int_dtype(
    *, value: _NumericValue, name: str, array_writer: CanonicalArrayWriter | None = None
) -> IntND:
    """Cast a scalar, sequence, or array to `jnp.int32`, checking int32 range.

    Args:
        value: A Python int, numpy/JAX integer scalar, or array-like of
            integer values.
        name: Qualified name of the leaf — surfaced in the error message
            so the user can locate the offending input.
        array_writer: Optional allocation owner called after host range validation.
            Without it, retain the normal JAX constructor and default placement.

    Returns:
        A `jnp.int32` array (0-d if `value` was a scalar).

    Raises:
        ValueError: If any element of `value` is not a whole number or is outside
            the int32 range `[-2**31, 2**31 - 1]`. The message names the leaf via
            `name`.

    """
    np_value = np.asarray(value)
    if np_value.dtype.kind == "f":
        fractional = np_value != np.trunc(np_value)
        if fractional.any():
            msg = (
                f"{name}: integer values required, got {np_value[fractional].tolist()}."
            )
            raise ValueError(msg)
    if np_value.size > 0:
        lo = int(np_value.min())
        hi = int(np_value.max())
        if lo < _INT32_MIN or hi > _INT32_MAX:
            msg = (
                f"{name}: int32 overflow — value range [{lo}, {hi}] "
                f"exceeds [{_INT32_MIN}, {_INT32_MAX}]."
            )
            raise ValueError(msg)
    if array_writer is not None:
        return array_writer(value=np_value, dtype=np.dtype(np.int32), name=name)
    return jnp.asarray(np_value, dtype=jnp.int32)


def safe_to_float_dtype(
    *, value: _NumericValue, name: str, array_writer: CanonicalArrayWriter | None = None
) -> FloatND:
    """Cast a scalar, sequence, or array to the canonical float dtype.

    Range check fires only on a down-cast:

    - Down-cast (float64 → float32 under `jax_enable_x64=False`): raise
      `OverflowError` if any element exceeds float32 magnitude rather
      than letting JAX silently saturate to `±inf`.
    - Up-cast or same-width cast: skip the range check. Precision loss
      within range is not an error — it is an inherent consequence of
      `jax_enable_x64=False`.

    Args:
        value: A Python float, numpy/JAX scalar, or array-like.
        name: Qualified name of the leaf — surfaced in the error message.
        array_writer: Optional allocation owner called after host range validation.
            Without it, retain the normal JAX constructor and default placement.

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
    if array_writer is not None:
        return array_writer(value=np_value, dtype=np.dtype(target_dtype), name=name)
    return jnp.asarray(np_value, dtype=target_dtype)


# User-output adapters. Unlike the host helpers above, these run inside traced
# code: they are the one place a user function's return value is converted to
# the kind its engine consumer requires. Each checks only the dtype, which is
# static under tracing, so it never reads, validates or synchronizes a traced
# value. A Python scalar returned by a user function (`0`, `0.0`, `True`) takes
# the same route as an array of its kind.


def name_of(function: object) -> OutputSource:  # noqa: PAN001 - any user callable
    """Return a function's `__name__` for an output-kind error, else its repr."""
    return getattr(function, "__name__", repr(function))


def as_canonical_float(*, value: UserOutput, source: OutputSource) -> FloatND:
    """Return a float-role output (an integer or float) at the canonical dtype.

    JAX canonicalizes the Python `float` dtype to the active precision, so the
    adapter reads no configuration: model fingerprints walk its code.

    Raises:
        InvalidFunctionOutputError: `source` returned a Boolean or complex value.

    """
    array = _first_array(value)
    if array.dtype.kind not in "iuf":
        msg = (
            f"{source} returned dtype '{array.dtype}', but must return a number. "
            "A Boolean here is usually a comparison returned in place of its "
            "consequence: use `jnp.where(condition, a, b)`, or a constraint to "
            "rule cells out."
        )
        raise InvalidFunctionOutputError(msg)
    return jnp.asarray(array, dtype=float)


def as_bool(*, value: UserOutput, source: OutputSource) -> BoolND:
    """Return a predicate's output, which must be Boolean, as a JAX Boolean.

    Raises:
        InvalidFunctionOutputError: `source` returned a non-Boolean value.

    """
    array = jnp.asarray(value)
    if array.dtype != jnp.bool_:
        msg = (
            f"{source} returned dtype '{array.dtype}', but must return a Boolean. "
            "Compare a discrete code explicitly, e.g. `work == 1`; a probability "
            "belongs in a stochastic transition."
        )
        raise InvalidFunctionOutputError(msg)
    return array


def as_code(*, value: UserOutput, source: OutputSource) -> Code:
    """Return a discrete-code output (an integer or Boolean) as an integer array.

    A Boolean becomes the code 0 or 1. An integer keeps at least 32 bits and a
    wider width: narrowing `2**32 + 1` to `int32` here would wrap it onto the
    code `1` before the off-grid check could see it.

    Raises:
        InvalidFunctionOutputError: `source` returned a float or complex value.

    """
    array = _first_array(value)
    if array.dtype.kind == "b":
        return jnp.asarray(array, dtype=jnp.int32)
    if array.dtype.kind not in "iu":
        msg = (
            f"{source} returned dtype '{array.dtype}', but must return an integer "
            "code. A float would be truncated onto a code: compute the code as "
            "an integer, e.g. with `.astype(jnp.int32)` after rounding on purpose."
        )
        raise InvalidFunctionOutputError(msg)
    if not isinstance(array, np.ndarray):
        return array.astype(jnp.promote_types(array.dtype, jnp.int32))
    # A host constant, never a tracer: its value can be read.
    # JAX narrows a host array to the active precision without a warning.
    active = jnp.asarray(np.zeros((), dtype=array.dtype)).dtype
    dtype = jnp.promote_types(active, jnp.int32)
    code = array.astype(dtype)
    if not np.array_equal(code, array):
        msg = (
            f"{source} returned an integer code that the active precision's "
            f"'{code.dtype}' cannot hold."
        )
        raise InvalidFunctionOutputError(msg)
    return jnp.asarray(code)


def _first_array(value: UserOutput) -> np.ndarray | ValueND:
    """Return a user output as an array without narrowing it.

    JAX narrows a host integer to 32 bits at single precision, or refuses it, so
    a host value is read with NumPy: `2**32` is a float-role output like any
    other integer, and a wide code is still wide when it is checked.
    """
    return value if isinstance(value, Array) else np.asarray(value)
