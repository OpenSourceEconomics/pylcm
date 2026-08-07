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
  invalid specification surfaces instead of being silently rescued;
- an event that can occur is never priced as one that cannot, even where its
  probability is too small for the dtype to hold — such a weight is refused
  rather than rounded down to impossible.

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


def usable_weight(weights: FloatND) -> FloatND:
    """Return each weight, with one too small to use raised to the smallest normal.

    A subnormal is representable but not usable, and what arithmetic does with
    it belongs to the backend rather than to the model: XLA:CPU flushes it, so
    it compares equal to zero and multiplies as zero, while CUDA carries it
    through. Zero is how this engine spells an event that cannot occur, so on a
    flushing backend an event of strictly positive probability becomes
    indistinguishable from an impossible one.

    That distinction only changes an answer when the value at the node is
    infinite. Dropping a node whose value is finite costs at most
    `tiny * |value|`, which is below every tolerance any model declares. Dropping
    one that carries `-inf` — the ordinary value of a state at which no action
    is feasible — replaces the true answer with a finite number, and the error
    is unbounded.

    Raising the weight to the smallest normal settles both cases with one
    substitution, and does it on an *operand* so the multiplication downstream
    stays a bare operation that can contract into a fused multiply-add:

    - against a finite value the contribution is `tiny * value`, which differs
      from dropping the node by less than the dropping error itself;
    - against `+-inf` the product is that infinity, which is the true answer;
    - a represented zero is untouched and remains the genuine null event;
    - the sign is preserved, so a negative weight stays visible to the
      distribution guard rather than being rescued into a positive one.

    Args:
        weights: Probabilities or quadrature weights, of any floating dtype.

    Returns:
        The weights, with every nonzero subnormal raised to the smallest normal
        magnitude of its dtype.

    """
    arr = jnp.asarray(weights)
    unusable = ~_is_represented_zero(arr) & _is_below_smallest_normal(arr)
    smallest_normal = jnp.asarray(jnp.finfo(arr.dtype).tiny, dtype=arr.dtype)
    return jnp.where(unusable, jnp.copysign(smallest_normal, arr), arr)


def joint_weight(factors: FloatND) -> FloatND:
    """Return the product of a node's probability factors, never rounded to zero.

    A joint node carries one factor per stochastic axis, and the product is its
    probability. Each factor can sit inside the dtype's normal range while the
    product falls below it — in float32, `sqrt(tiny)/2` squared is subnormal —
    at which point the hardware delivers exactly zero. That would make an event
    which can occur indistinguishable from one which cannot, and downstream
    nothing could tell them apart, because the two arrive as the same zero.

    An underflowed product therefore leaves here as the smallest normal
    magnitude instead, carrying the sign of the product. The magnitude is not
    the node's probability and is not meant to be: it is small enough that the
    node contributes nothing to a finite continuation, and nonzero so that an
    infinity standing at the node still reaches the answer. `usable_weight`
    states the same rule for a weight that arrives small rather than becoming
    small here.

    A factor of exactly zero is the genuine null event, so the product keeps its
    zero and the node contributes nothing whatever value stands at it. A NaN
    factor stays poison.

    Args:
        factors: The factors stacked along the leading axis, one entry per
            stochastic axis of the node. Trailing axes broadcast, so a scalar
            regime probability stacked against a vector of node weights
            reduces to one weight per node.

    Returns:
        Their product over the leading axis, raised to the smallest normal
        magnitude wherever every factor can occur but the product cannot be
        represented.

    """
    arr = usable_weight(jnp.asarray(factors))
    product = jnp.prod(arr, axis=0)
    every_factor_can_occur = ~jnp.any(_is_represented_zero(arr), axis=0)
    smallest_normal = jnp.asarray(jnp.finfo(product.dtype).tiny, dtype=product.dtype)
    return jnp.where(
        every_factor_can_occur & _is_below_smallest_normal(product),
        jnp.copysign(smallest_normal, product),
        product,
    )


def has_nonzero_subnormal(values: FloatND) -> BoolND:
    """Return whether any entry is a represented subnormal other than zero.

    Such a value is outside the probability contract this module can honour.
    A subnormal survives in memory, but XLA:CPU treats it as zero in *both*
    the comparison that decides nullity and the multiplication that would form
    its contribution — so a weight of that size is silently dropped rather
    than either respected or refused. Reading the bits is what makes the
    verdict the same on a backend that flushes and one that does not: where the
    value is flushed, every arithmetic test for it is subject to the same
    flush, so `0 < p < tiny` evaluates as `0 < 0`.

    Args:
        values: Array to inspect, of any floating dtype.

    Returns:
        Scalar boolean, true if some entry is subnormal and not zero.

    """
    arr = jnp.asarray(values)
    return jnp.any(~_is_represented_zero(arr) & _is_below_smallest_normal(arr))


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

    The weight is raised through `usable_weight` first, for the same reason: a
    weight too small for the dtype to use is settled on an operand rather than
    on the product, so a node that can occur keeps an infinity standing at it
    without any of this costing a rounding step.

    Args:
        weight: Probability, quadrature, or interpolation weight.
        value: The value being weighted, possibly `+-inf` at a zero-weight node.

    Returns:
        The elementwise product, broadcast as `weight * value` would be.

    """
    weight_arr = usable_weight(jnp.asarray(weight))
    value_arr = jnp.asarray(value)
    safe_value = jnp.where(weight_arr == 0, jnp.zeros((), value_arr.dtype), value_arr)
    return weight_arr * safe_value


def _is_represented_zero(values: FloatND) -> BoolND:
    """Whether each entry is `+0` or `-0`, read from its bits.

    `values == 0` cannot answer this: a subnormal compares equal to zero under
    the same flush that would drop it.
    """
    arr = jnp.asarray(values)
    int_dtype = jnp.int32 if arr.dtype.itemsize == _FLOAT32_BYTES else jnp.int64
    magnitude = jax.lax.bitcast_convert_type(arr, int_dtype) & jnp.asarray(
        (1 << (8 * arr.dtype.itemsize - 1)) - 1, dtype=int_dtype
    )
    return magnitude == 0


def _is_below_smallest_normal(values: FloatND) -> BoolND:
    """Whether each entry's magnitude is zero or subnormal, read from its bits.

    The two cases are one question here: a product arriving either way has lost
    the size it was meant to carry.
    """
    arr = jnp.asarray(values)
    int_dtype = jnp.int32 if arr.dtype.itemsize == _FLOAT32_BYTES else jnp.int64
    magnitude = jax.lax.bitcast_convert_type(arr, int_dtype) & jnp.asarray(
        (1 << (8 * arr.dtype.itemsize - 1)) - 1, dtype=int_dtype
    )
    smallest_normal = jax.lax.bitcast_convert_type(
        jnp.asarray(jnp.finfo(arr.dtype).tiny, dtype=arr.dtype), int_dtype
    )
    return magnitude < smallest_normal
