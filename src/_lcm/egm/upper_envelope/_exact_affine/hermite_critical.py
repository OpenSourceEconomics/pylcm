"""Exact critical-polynomial control signs for a canonical Hermite piece.

Only original represented operands enter the native predicate. Its signs
describe a mathematical cubic, not rounded row/corner selection or log error.
The caller must retain piece authority, source status and uncertainty holes.
"""

import inspect

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope._exact_affine import ffi
from lcm.typing import FloatND, IntND

HERMITE_CRITICAL_F32_LIMBS = 35
HERMITE_CRITICAL_F64_LIMBS = 263


def hermite_critical_bernstein_signs(
    *,
    x0: FloatND,
    h: FloatND,
    v0: FloatND,
    v1: FloatND,
    a: FloatND,
    b: FloatND,
    L: FloatND,
    A: FloatND,
    lower: FloatND,
    upper: FloatND,
) -> tuple[IntND, IntND]:
    """Return exact scaled Bernstein signs and mandatory native status.

    The common float dtype must be float32 or float64. Inputs broadcast to
    one batch shape; signs append a cubic-control axis, while status keeps
    the batch shape. Status zero validates the signs, not uniform positivity.
    Native unresolved status covers nonfinite operands, nonpositive width or
    scale, and intervals outside the unclamped normalized piece.

    The canonical materializer supplies the original rounded width and
    correction operands. This function neither recomputes the limiter nor
    substitutes a rounded expansion of the polynomial coefficients.
    """
    operands = (x0, h, v0, v1, a, b, L, A, lower, upper)
    target = ffi._target_for(  # noqa: SLF001
        operands=operands,
        f32="ExactHermiteCriticalBernsteinF32",
        f64="ExactHermiteCriticalBernsteinF64",
    )
    operands = tuple(jnp.broadcast_arrays(*operands))
    shape = operands[0].shape
    limbs = (
        HERMITE_CRITICAL_F32_LIMBS
        if operands[0].dtype == jnp.float32
        else HERMITE_CRITICAL_F64_LIMBS
    )
    ffi._ensure_registered()  # noqa: SLF001
    # The accumulator pair is an explicit result allocation, not a CUDA stack.
    signs, status, _workspace = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct((*shape, 4), jnp.int32),
            jax.ShapeDtypeStruct(shape, jnp.int32),
            jax.ShapeDtypeStruct((*shape, 2, limbs), jnp.uint32),
        ),
        vmap_method="broadcast_all",
    )(*operands)
    return signs, status


# This is a numerical export, not authority for arbitrary functions in this module.
_ORIGINAL_NUMERICAL_EXPORTS = (
    (
        "hermite_critical_bernstein_signs",
        hermite_critical_bernstein_signs,
        hermite_critical_bernstein_signs.__code__,
        inspect.unwrap(hermite_critical_bernstein_signs),
        inspect.unwrap(hermite_critical_bernstein_signs).__code__,
    ),
)
