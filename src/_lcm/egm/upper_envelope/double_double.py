"""Double-double arithmetic built from error-free transforms.

A double-double represents a number as an unevaluated sum of floats. Knuth's
`two_sum` and Dekker's `two_prod` return the rounding error of an addition or a
multiplication exactly, so a sum or product can be carried at roughly twice the
working precision. That is what lets the upper envelope decide questions whose
answers live below the resolution of a single float — which branch is higher
where two of them round to the same value.

Values here are `(high, low, dropped)` triples. `high + low` is the estimate;
`dropped` bounds the low-order tail each renormalization discards, so it is
exactly zero whenever the whole evaluation was exact. That distinction is the
point: a certified tie and a tie that merely fits in the format are different
answers, and only the first is safe to act on.

Every function here is an operator surrogate on a double-double number, so its
operands are positional — the spelling each operation is known by. This module
contains nothing else, which is what makes that exemption from the project's
keyword-only rule auditable: a helper that is not an operator does not belong
here.

The transforms are exact only while their intermediates stay normal. Dekker's
splitting multiplies by `2**((nmant + 2) // 2) + 1`, so a product that underflows
or lands among the subnormals silently loses the tail. Callers that read a zero
as evidence must check that domain themselves; `certified_sign` does.

Everything is branch-free and elementwise, so it stays `jax.jit`- and
`jax.vmap`-compatible with static shapes.
"""

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, IntND

# A value as `(high, low, dropped)`: estimate `high + low`, tail bound `dropped`.
type DoubleDouble = tuple[FloatND, FloatND, FloatND]


def two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return `(s, e)` with `a + b == s + e` exactly (Knuth)."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def two_prod(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return Dekker's product pair inside the transform's normal-range domain.

    This primitive does not certify that domain. Structural consumers needing a
    full-range exact result must either establish it before calling or use the
    fixed-width exact kernels in ``_exact_affine``.
    """
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    error = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, error


def scale_by_power_of_two(value: FloatND, exponent: IntND) -> FloatND:
    """Return `value * 2**exponent`, exactly, wherever the result stays normal.

    Scaling by a power of two moves no information, and every decision built on
    the transforms here relies on that. `jnp.ldexp` does not deliver it
    uniformly: on a CUDA backend it is exact once XLA has compiled it and wrong
    for a substantial share of inputs when evaluated eagerly, so a primitive that
    is supposed to be exact would instead depend on whether it had been traced.

    Constructing the multiplier from the IEEE-754 exponent field removes the
    question. Writing `exponent + bias` into that field is the definition of
    `2**exponent`, so the multiplier is exact by construction and the product is
    a single correctly-rounded multiplication by a power of two.

    An `exponent` that would leave the normal range produces a multiplier that is
    zero, infinite, or otherwise not `2**exponent`. Nothing is clamped or
    guessed here — callers that need certainty compare against the unscaled
    operand and refuse the case, which is the only honest answer once the
    scaling itself has lost information.
    """
    info = jnp.finfo(value.dtype)
    bias = info.maxexp - 1
    field = (exponent.astype(f"int{info.bits}") + bias) << info.nmant
    multiplier = jax.lax.bitcast_convert_type(field, value.dtype)
    return value * multiplier


def normalizing_exponent(*values: FloatND) -> IntND:
    """Return `e` such that scaling by `2**-e` lands the largest term near one.

    Scaling by a power of two is exact, so it moves no information. Pulling the
    operands into the binade around one is what keeps products out of the range
    where the error-free transforms stop being error-free. Zero and non-finite
    terms are ignored; an all-zero group scales by `2**0`, which leaves it alone.
    """
    magnitude = jnp.zeros_like(values[0])
    for term in values:
        magnitude = jnp.maximum(magnitude, jnp.abs(term))
    usable = jnp.isfinite(magnitude) & (magnitude > 0.0)
    _mantissa, exponent = jnp.frexp(jnp.where(usable, magnitude, 1.0))
    return exponent


def is_stored_zero(value: FloatND) -> BoolND:
    """Report where a float is a true zero, decided on the stored bits.

    `value == 0.0` cannot answer this on a backend that flushes: a subnormal
    reads as zero there, so a quantity that is small but present becomes
    indistinguishable from one that is absent — and every guard keyed on the
    difference then takes the branch meant for the absent case. Shifting the
    sign bit off leaves zero exactly for `+0.0` and `-0.0`, and a bitcast is a
    reinterpretation rather than an operation, so nothing is destroyed reading
    it. Integers are not flushed.
    """
    unsigned = jnp.dtype(f"uint{jnp.finfo(value.dtype).bits}")
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    return (bits << bits.dtype.type(1)) == 0


def scale_tail_bound(*, bound: FloatND, factor: FloatND) -> FloatND:
    """Return `bound * |factor|`, floored where that product underflows.

    A tail bound only ever has to be an *upper* bound on what was discarded, and
    a product that underflows is below the smallest normal exactly when the true
    amount is — so replacing it with the smallest normal keeps the bound valid
    while making it something the format can hold. `maximum` cannot underflow,
    so a bound that reaches here nonzero leaves it nonzero, however many factors
    below one it still has to pass.

    Without that floor a bound is destroyed by the first small factor and
    arrives at the comparison as an exact zero, which is the certificate for
    having discarded nothing — the one reading it must never produce.

    A bound of exactly zero says nothing was discarded, and stays zero: that is
    what two links lying on one line rely on. Which case it is has to be asked of
    the stored bits, not of `!= 0.0` — a bound arriving subnormal reads as zero
    to that comparison, so the floor would be skipped on exactly the operands it
    exists for. A floored bound is never subnormal afterwards, but the sites that
    *mint* a bound are upstream of the first floor, so the invariant cannot be
    assumed by the thing that establishes it.

    The floor holds on both kinds of backend, by different routes: where the
    product flushes, the exact value was below the smallest normal, which is what
    made it flush; where it becomes a genuine subnormal instead, the smallest
    normal still bounds a subnormal.
    """
    tiny = jnp.finfo(bound.dtype).tiny
    scaled = bound * jnp.abs(factor)
    both_present = ~is_stored_zero(bound) & ~is_stored_zero(factor)
    return jnp.where(both_present, jnp.maximum(scaled, tiny), scaled)


def dd_from_difference(a: FloatND, b: FloatND) -> DoubleDouble:
    """Return the exact difference `a - b` as a double-double."""
    high, low = two_sum(a, -b)
    return high, low, jnp.zeros_like(high)


def dd_negate(value: DoubleDouble) -> DoubleDouble:
    """Return the negation of a double-double, preserving its error bound."""
    high, low, dropped = value
    return -high, -low, dropped


def dd_add_float(value: DoubleDouble, addend: FloatND) -> DoubleDouble:
    """Add a plain float to a double-double, accumulating the discarded tail."""
    high, low, dropped = value
    sum_high, error_high = two_sum(high, addend)
    low_sum, error_low = two_sum(low, error_high)
    new_high, new_low = two_sum(sum_high, low_sum)
    return new_high, new_low, dropped + jnp.abs(error_low)


def dd_add(left: DoubleDouble, right: DoubleDouble) -> DoubleDouble:
    """Add two double-doubles, accumulating both error bounds and the tail."""
    left_high, left_low, left_dropped = left
    right_high, right_low, right_dropped = right
    sum_high, error_high = two_sum(left_high, right_high)
    sum_low, error_low = two_sum(left_low, right_low)
    low_sum, tail = two_sum(error_high, sum_low)
    new_high, new_low = two_sum(sum_high, low_sum)
    dropped = left_dropped + right_dropped + jnp.abs(tail) + jnp.abs(error_low)
    return new_high, new_low, dropped


def dd_mul_float(value: DoubleDouble, factor: FloatND) -> DoubleDouble:
    """Multiply a double-double by a plain float."""
    high, low, dropped = value
    product_high, error_high = two_prod(high, factor)
    product_low, error_low = two_prod(low, factor)
    accumulated = (
        product_high,
        jnp.zeros_like(product_high),
        jnp.zeros_like(product_high),
    )
    for term in (error_high, product_low, error_low):
        accumulated = dd_add_float(accumulated, term)
    new_high, new_low, new_dropped = accumulated
    return (
        new_high,
        new_low,
        new_dropped + scale_tail_bound(bound=dropped, factor=factor),
    )


def dd_mul(left: DoubleDouble, right: DoubleDouble) -> DoubleDouble:
    """Multiply two double-doubles, accumulating both error bounds and the tail."""
    left_high, left_low, left_dropped = left
    right_high, right_low, right_dropped = right
    product, error = two_prod(left_high, right_high)
    cross_high, cross_high_error = two_prod(left_high, right_low)
    cross_low, cross_low_error = two_prod(left_low, right_high)
    tail, tail_error = two_prod(left_low, right_low)

    accumulated = (product, jnp.zeros_like(product), jnp.zeros_like(product))
    for term in (
        error,
        cross_high,
        cross_high_error,
        cross_low,
        cross_low_error,
        tail,
        tail_error,
    ):
        accumulated = dd_add_float(accumulated, term)

    left_scale = jnp.abs(left_high) + jnp.abs(left_low)
    right_scale = jnp.abs(right_high) + jnp.abs(right_low)
    new_high, new_low, new_dropped = accumulated
    dropped = (
        new_dropped
        + scale_tail_bound(bound=left_dropped, factor=right_scale)
        + scale_tail_bound(bound=right_dropped, factor=left_scale)
        + scale_tail_bound(bound=left_dropped, factor=right_dropped)
    )
    return new_high, new_low, dropped


def dd_quotient(
    numerator: DoubleDouble, denominator: DoubleDouble
) -> tuple[FloatND, FloatND]:
    """Return `numerator / denominator` as an unevaluated sum `(high, low)`.

    Division has no error-free transform, so this is the one operation here that
    carries no certificate: one Newton correction on the remainder leaves a
    result good to roughly twice the working precision, which is enough to *find*
    a structure but never enough to *prove* one. Decisions that must hold exactly
    go through `certified_sign`, or through `dd_quotient_bounded` where the
    quotient itself is the quantity being decided on.
    """
    denominator_high, _low, _dropped = denominator
    estimate = numerator[0] / denominator_high
    remainder = dd_add(numerator, dd_negate(dd_mul_float(denominator, estimate)))
    correction = (remainder[0] + remainder[1]) / denominator_high
    return two_sum(estimate, correction)


def dd_quotient_bounded(
    numerator: DoubleDouble, denominator: DoubleDouble
) -> DoubleDouble:
    """Return `numerator / denominator` with a bound on how far off it is.

    The bound is what separates a quotient that merely *looks* exact from one
    that is, so it is **measured rather than assumed**: the quotient is
    multiplied back by the denominator and what fails to reproduce the numerator
    is what is reported, referred back through the denominator. A division that
    reproduces its numerator exactly is exact, and this says so with a bound of
    zero.

    That distinction is the whole point. A crossing that lands on a representable
    abscissa — the ordinary case, since two lines routinely meet at a node — has
    an exact quotient, and a consumer asking which side of it the truth falls on
    can be told. A blanket second-order charge on every division would leave that
    consumer with a positive bound around an exactly located answer, and it would
    refuse precisely the cases that are easiest to be sure about.

    Referring the left-over back through the denominator is itself rounded, and
    a bound rounded down is not a bound, so the result is widened by a few
    rounding steps. The widening is multiplicative, which leaves an exact zero
    exactly zero: nothing needs covering where nothing was left over. It also
    absorbs dividing by the denominator's leading word rather than its full
    value, a relative slack of the same order.
    """
    high, low = dd_quotient(numerator, denominator)
    left_over = dd_add(
        numerator,
        dd_negate(dd_mul(denominator, (high, low, jnp.zeros_like(high)))),
    )
    unreproduced = jnp.abs(left_over[0] + left_over[1]) + left_over[2]
    referred = unreproduced / jnp.abs(denominator[0])
    return high, low, referred * (1.0 + 4.0 * jnp.finfo(referred.dtype).eps)


def _split(a: FloatND) -> tuple[FloatND, FloatND]:
    """Split `a` into two half-precision halves with `a == hi + lo` exactly.

    Dekker's split reaches its halves by multiplying the operand by roughly the
    square root of the format's precision. That intermediate overflows near the
    top of the range even where the product the split serves is an ordinary
    finite number, and an overflowed intermediate does not degrade gracefully:
    `inf - inf` is `nan`, so the operand comes back as a pair of `nan` and every
    certificate downstream inherits it.

    An operand that large is split from a scaled copy instead. Scaling by a power
    of two is exact in both directions wherever the result stays normal, and one
    that starts near the top of the range has room to spare below it, so the
    halves scaled back are the halves of the original — the range is borrowed,
    not the precision.
    """
    half = (jnp.finfo(a.dtype).nmant + 2) // 2
    factor = jnp.asarray(2.0**half + 1.0, dtype=a.dtype)
    # `factor` is just above `2**half`, so the intermediate stays finite while
    # the operand stays below `2**(maxexp - half)`; back off two further binades
    # rather than sit against that edge.
    threshold = jnp.asarray(
        2.0 ** (jnp.finfo(a.dtype).maxexp - 2 - half), dtype=a.dtype
    )
    down = jnp.asarray(2.0 ** -(half + 1), dtype=a.dtype)
    up = jnp.asarray(2.0 ** (half + 1), dtype=a.dtype)

    oversized = jnp.abs(a) >= threshold
    scaled = jnp.where(oversized, a * down, a)
    c = factor * scaled
    a_big = c - scaled
    a_hi = c - a_big
    a_lo = scaled - a_hi
    return (
        jnp.where(oversized, a_hi * up, a_hi),
        jnp.where(oversized, a_lo * up, a_lo),
    )
