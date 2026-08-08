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
- an event that can occur is discarded only where discarding it cannot change
  the answer, and is never enlarged. A weight too small for the dtype to use
  keeps its own magnitude against a finite value, where a backend that flushes
  it drops the node and the omission is the smallest available; it is raised to
  the smallest normal only against `+-inf`, where every strictly positive weight
  gives the same answer and dropping would not.

The residual is an approximation, declared rather than hidden: a node whose
probability the format cannot hold may contribute less than its true share, by
at most `tiny * |value|`. That is below every declared tolerance for any value
function a model can also add utility to, and it is the floor for arithmetic
that cannot multiply subnormals. The bound is two-sided: a weight arriving
already representable is never enlarged, but a product that vanished is
substituted by the smallest representable magnitude, which overstates a product
that underflowed below even that. What is guaranteed is the size of the
disagreement, not its direction.

That residual is **backend-visible**, and deliberately so. Leaving the weight
alone against a finite value means a backend that flushes subnormals drops the
node while one that represents them prices it, so the two disagree by `p *
|value|`. The bound is the same one above and is negligible for any ordinary
continuation, but it is not zero: at a value near the top of the dtype's range
the disagreement reaches order one. Buying agreement instead would mean
adopting the flushing backend's answer everywhere — discarding a contribution
the other hardware computed correctly — so the arithmetic is left to be as
right as each machine can make it. Tests must therefore assert the invariants
rather than any particular value, because which of the two answers appears is a
property of the executing backend:

- the weight is nonzero in its bits;
- it agrees with its node to within the smallest representable magnitude;
- it stays below the normal range;
- a published quantity compared against an exactly-computed reference carries
  one representable step at the active precision. An exact inequality on a
  reported value asserts the rounding direction, which is not contractual — a
  backend that prices the node returns the nearest representable value to the
  exact answer, and that lies above it half the time.

Total-mass conventions are not settled here. They differ by call site — a
target represented with no mass contributes `0`, while a whole continuation
lottery with no mass anywhere is malformed and aggregates to NaN — so each
caller states its own denominator.

Every argument is keyword-only, as everywhere else in the package. These read
as arithmetic and arithmetic invites positional operands, but the operands are
not interchangeable and one of them carries a claim about what the caller has
already established, so the call site says which is which.
"""

import jax.numpy as jnp

from _lcm.probability import (
    balanced_product,
    is_below_smallest_normal,
    is_represented_zero,
    nonzero_exact_product,
    scaled_exact_product,
)
from lcm.typing import FloatND, IntND


def joint_weight(factors: FloatND) -> FloatND:
    """Return the product of a node's probability factors, never rounded to zero.

    A joint node carries one factor per stochastic axis, and the product is its
    probability. Each factor can sit inside the dtype's normal range while the
    product falls below it — in float32, `sqrt(tiny)/2` squared is subnormal —
    at which point the hardware delivers exactly zero. That would make an event
    which can occur indistinguishable from one which cannot, and downstream
    nothing could tell them apart, because the two arrive as the same zero.

    A product that vanishes this way leaves here as the **smallest representable
    magnitude** instead, carrying the sign of the product. It is the smallest
    value that is still nonzero, so a node that can occur keeps an infinity
    standing at it, and it is as close to the discarded product as the format
    allows. `zero_safe_weighted_term` decides which of those two properties is
    the operative one, because that depends on the value.

    The substitution is not one-sided. A product may underflow *below* the
    smallest representable magnitude — in float32, `sqrt(tiny)/2` squared does —
    and the substitute then overstates it. The overstatement is bounded by that
    magnitude itself, so the node contributes at most
    `smallest_subnormal * |value|`, which is smaller than the `tiny * |value|`
    omission bound by a factor of `2**23` in single precision and `2**52` in
    double. What is one-sided is the treatment of a weight that arrives
    representable: `zero_safe_weighted_term` never enlarges one.

    A product which is merely subnormal — the backend represented it rather than
    flushing it — keeps its own magnitude, which is more informative than the
    substitute.

    A factor of exactly zero is the genuine null event, so the product keeps its
    zero and the node contributes nothing whatever value stands at it. A NaN
    factor stays poison.

    Args:
        factors: The factors stacked along the leading axis, one entry per
            stochastic axis of the node. Trailing axes broadcast, so a scalar
            regime probability stacked against a vector of node weights
            reduces to one weight per node.

    Returns:
        Their product over the leading axis, with a product that vanished
        despite every factor being able to occur replaced by the smallest
        representable magnitude of the same sign.

    """
    return nonzero_exact_product(jnp.asarray(factors))


def scaled_joint_weight(factors: FloatND) -> tuple[FloatND, IntND]:
    """Return `joint_weight`'s product with its scale carried alongside it.

    The node's probability is `weights * 2**-shift`. A joint product that lands
    below the dtype's normal range cannot be carried as a plain float on this
    backend — it is representable, but only as a subnormal, and a subnormal
    that exists only as an intermediate reads back as zero once it is consumed
    inside a fused region. Keeping the scale separate leaves every returned
    number normal, so the event stays distinguishable from one that cannot
    occur no matter what the consumer does with it.

    One shift covers the whole array, so the weights keep their sizes relative
    to each other and a consumer that normalizes by their own total can ignore
    it entirely. A consumer that concatenates weights from several calls has to
    bring them onto a common scale first, which is what the shift is for.

    The shift is whatever the smallest product in the array needs, so nothing
    that can occur is left below the format and none of `joint_weight`'s
    substitution applies. A node that can occur therefore keeps its exact
    probability, not an approximation of it.

    Args:
        factors: The factors stacked along the leading axis, one entry per
            stochastic axis of the node. Trailing axes broadcast.

    Returns:
        Tuple of the scaled product and the base-two scale it carries.

    """
    return scaled_exact_product(jnp.asarray(factors))


def zero_safe_weighted_term(
    *, weight: FloatND, value: FloatND, subnormal_is_accounted_for: bool
) -> FloatND:
    """Return `weight * value`, exactly `0.0` wherever `weight` is zero.

    The mask sits on the *value*, an operand of the multiplication, rather than
    on the product. Both spellings neutralize a zero-weight infinity, but only
    this one leaves the multiplication as a bare operation feeding whatever
    reduction consumes it, which XLA can contract into a fused multiply-add. A
    `select` placed after the multiplication forces the product to round on its
    own before the sum rounds again, which moves an ordinary all-positive
    weighted average by several units in the last place — enough to reverse a
    discrete choice that was not close to tied.

    A weight the format cannot use — a nonzero subnormal — is treated according
    to the value it meets, because that is what decides whether its magnitude
    matters:

    - against a **finite** value the weight is left exactly as it is. A backend
      that flushes it then drops the node, and the omission is `p * |value|`,
      the smallest error available to anything that cannot do subnormal
      arithmetic. Enlarging the weight here would be strictly worse: the
      subnormal range spans a factor of `2**23` in single precision and `2**52`
      in double, so promoting the smallest of them to `tiny` would hand a
      negligible event the mass of the largest one and invent a contribution;
    - against `+-inf` it is raised to the smallest normal magnitude, keeping its
      sign. Every strictly positive weight yields the same infinity there, so no
      magnitude is lost by the substitution, while dropping the node would
      replace an infinite answer with a finite one.

    A represented zero is the genuine null event and annihilates any value,
    finite or not. A NaN weight stays poison.

    The value is replaced only where a zero weight meets a value that is not
    finite, which is the only case the multiplication itself gets wrong. Against
    a finite value `0 * value` is already `0`, and leaving the operand in place
    keeps the derivative with respect to the weight — `d(w * v)/dw = v` — which
    masking unconditionally would flatten to zero at exactly the coordinates
    where a weight vanishes.

    The exponent is moved from the value onto the weight so that a weight
    below the normal range multiplies exactly rather than being flushed as an
    operand. That runs at every term, and two callers can say in advance that
    no weight of theirs needs it, because a weight below the range there cannot
    change what they return:

    - the nodes of one target arrive on a common base-two scale, and one still
      below the range after it is one that scale's own cap has declared
      understated; moving its exponent would price a node the contract says is
      not priced;
    - an interpolation corner weight below the range means the coordinate lies
      within a subnormal of a node, so the corner beside it carries weight one
      and the whole read.

    Both pass `subnormal_is_accounted_for=True`. Every other caller passes
    `False` — a regime transition probability arrives at whatever size the
    model chose, and nothing upstream has accounted for a small one. There is
    no default: which of the two a call site is cannot be guessed from the
    operands, so it is stated.

    Args:
        weight: Probability, quadrature, or interpolation weight.
        value: The value being weighted, possibly `+-inf` at a zero-weight node.
        subnormal_is_accounted_for: Whether the caller has already established
            that a weight below the normal range cannot change its result.

    Returns:
        The elementwise product, broadcast as `weight * value` would be.

    """
    weight_arr = jnp.asarray(weight)
    value_arr = jnp.asarray(value)

    weight_is_null = is_represented_zero(weight_arr)
    weight_is_unusable = ~weight_is_null & is_below_smallest_normal(weight_arr)
    smallest_normal = jnp.asarray(
        jnp.finfo(weight_arr.dtype).tiny, dtype=weight_arr.dtype
    )
    effective_weight = jnp.where(
        weight_is_unusable & jnp.isinf(value_arr),
        jnp.copysign(smallest_normal, weight_arr),
        weight_arr,
    )
    safe_value = jnp.where(
        weight_is_null & ~jnp.isfinite(value_arr),
        jnp.zeros((), value_arr.dtype),
        value_arr,
    )
    if subnormal_is_accounted_for:
        return effective_weight * safe_value
    return balanced_product(effective_weight, safe_value)
