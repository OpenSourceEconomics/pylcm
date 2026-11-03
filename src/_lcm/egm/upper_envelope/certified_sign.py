"""Affine channel arithmetic and backend readability of subnormal operands."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul_float,
)
from lcm.typing import BoolND, FloatND


def affine_numerator(
    *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
) -> DoubleDouble:
    """Return `v0*(x1 - x) + v1*(x - x0)`, the width-scaled value at `x`."""
    return dd_add(
        dd_mul_float(dd_from_difference(x1, x_query), v0),
        dd_mul_float(dd_from_difference(x_query, x0), v1),
    )


@cache
def backend_flushes_subnormals(dtype: DTypeLike) -> bool:
    """Report whether this backend destroys a subnormal rather than reading it.

    The answer belongs to the compiled backend, not to the format: XLA:CPU
    flushes, CUDA reads the whole band. So it is measured by asking the backend
    to halve the smallest normal and seeing whether what comes back is the
    subnormal that step lands on.

    Every refusal built on "the arithmetic could not see this" is a claim about a
    backend that cannot represent it. Where the backend can, the same refusal
    withholds a verdict the arithmetic reached correctly — the mirror of the
    defect the refusals exist to prevent, and invisible to a battery run only
    where the flush happens.

    The probe runs on concrete inputs, so it resolves to a Python bool while the
    program is traced and the refusals it gates leave the compiled program
    entirely on a backend that reads the band.

    `ensure_compile_time_eval` is what makes "concrete inputs" true from inside a
    trace. Without it the halving stages into the enclosing jaxpr and hands back a
    tracer, and asking a tracer for a Python bool raises. The memo on this function
    hides that completely whenever an eager call for the same dtype happened to come
    first, so whether it raises depends on which test a process drew first — a
    failure that reads as random and is not.
    """
    with jax.ensure_compile_time_eval():
        smallest_normal = np.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
        halved = jax.jit(_halve)(smallest_normal)
        return bool(np.asarray(halved) == 0.0)


def _halve(value: FloatND) -> FloatND:
    """Multiply by one half — the step that takes a smallest normal subnormal."""
    return value * 0.5


def is_subnormal(value: FloatND) -> BoolND:
    """Report where a float is subnormal, and so reads as zero to every operation.

    A subnormal's *ordering* against a normal number still comes out right — the
    flush makes it zero, and zero is on the correct side of any nonzero normal.
    What is lost is its ordering against another subnormal, and against zero
    itself, where the flush makes distinct numbers compare equal.

    Decided on the bit pattern, which `bitcast_convert_type` preserves. Nothing
    arithmetic can answer it: the comparisons that would ask — against zero,
    against `tiny` — are themselves subject to the flushing they are meant to
    detect, and `frexp` reports the same exponent for every subnormal whatever
    its magnitude.
    """
    _unsigned, exponent_mask, mantissa_mask = _IEEE_FIELDS[jnp.dtype(value.dtype).name]
    bits = jax.lax.bitcast_convert_type(value, jnp.dtype(_unsigned))
    return ((bits & bits.dtype.type(exponent_mask)) == 0) & (
        (bits & bits.dtype.type(mantissa_mask)) != 0
    )


_IEEE_FIELDS = {
    "float32": ("uint32", 0x7F800000, 0x007FFFFF),
    "float64": ("uint64", 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}
