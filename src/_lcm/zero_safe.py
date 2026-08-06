"""Weighted arithmetic that treats an exactly-zero weight as a null event.

`-inf` is the ordinary value of a state at which every action is infeasible,
and an exactly-zero weight is equally ordinary — a `MarkovTransition` row with
a zero entry, a binned process with an empty tail bin, a regime that cannot be
reached from here. Where the two meet, floating-point arithmetic computes
`0.0 * -inf = nan` rather than `0.0`, and that NaN then destroys every
well-specified node beside it. Probability says the opposite: an event that
cannot occur contributes nothing to an expectation, whatever value it carries.

What this module guarantees, and what it deliberately does not:

- an exactly-zero weight annihilates any value, finite or not;
- a *positive* weight on `+-inf` keeps that infinity — it is the answer;
- a NaN weight stays poison, because it is not a probability;
- a negative weight stays visible rather than being absorbed into zero, so an
  invalid specification surfaces instead of being silently rescued.

Total-mass conventions are not settled here. They differ by call site — a
target represented with no mass contributes `0`, while a whole continuation
lottery with no mass anywhere is malformed and aggregates to NaN — so each
caller states its own denominator.

This module implements an arithmetic and nothing else, so its functions keep
the spelling their operation is known by and take their operands positionally
rather than keyword-only.
"""

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, FloatND

_FLOAT32_BYTES = 4


def has_nonzero_subnormal(values: FloatND) -> BoolND:
    """Return whether any entry is a represented subnormal other than zero.

    Such a value is outside the probability contract this module can honour.
    A subnormal survives in memory, but XLA:CPU treats it as zero in *both*
    the comparison that decides nullity and the multiplication that would form
    its contribution — so a weight of that size is silently dropped rather
    than either respected or refused. Reading the bits is the only way to see
    it: every arithmetic test is subject to the same flush, so `0 < p < tiny`
    evaluates as `0 < 0`.

    Args:
        values: Array to inspect, of any floating dtype.

    Returns:
        Scalar boolean, true if some entry is subnormal and not zero.

    """
    arr = jnp.asarray(values)
    int_dtype = jnp.int32 if arr.dtype.itemsize == _FLOAT32_BYTES else jnp.int64
    magnitude = jax.lax.bitcast_convert_type(arr, int_dtype) & jnp.asarray(
        (1 << (8 * arr.dtype.itemsize - 1)) - 1, dtype=int_dtype
    )
    smallest_normal = jax.lax.bitcast_convert_type(
        jnp.asarray(jnp.finfo(arr.dtype).tiny, dtype=arr.dtype), int_dtype
    )
    return jnp.any((magnitude > 0) & (magnitude < smallest_normal))


def zero_safe_weighted_term(weight: FloatND, value: FloatND) -> FloatND:
    """Return `weight * value`, exactly `0.0` wherever `weight` is zero.

    The mask sits on the *value*, an operand of the multiplication, rather than
    on the product. Both spellings neutralize a zero-weight infinity, but only
    this one leaves the multiplication as a bare operation feeding whatever
    reduction consumes it, which XLA can contract into a fused multiply-add. A
    `select` placed after the multiplication forces the product to round on its
    own before the sum rounds again, which moves an ordinary all-positive
    weighted average by several units in the last place — enough to reverse a
    discrete choice that was not close to tied.

    Args:
        weight: Probability, quadrature, or interpolation weight.
        value: The value being weighted, possibly `+-inf` at a zero-weight node.

    Returns:
        The elementwise product, broadcast as `weight * value` would be.

    """
    weight_arr = jnp.asarray(weight)
    value_arr = jnp.asarray(value)
    safe_value = jnp.where(weight_arr == 0, jnp.zeros((), value_arr.dtype), value_arr)
    return weight_arr * safe_value
