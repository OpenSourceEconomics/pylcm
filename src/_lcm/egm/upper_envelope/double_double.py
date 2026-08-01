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

The transforms are exact only while their intermediates stay normal. Dekker's
splitting multiplies by `2**((nmant + 2) // 2) + 1`, so a product that underflows
or lands among the subnormals silently loses the tail. Callers that read a zero
as evidence must check that domain themselves; `certified_sign` does.

Everything is branch-free and elementwise, so it stays `jax.jit`- and
`jax.vmap`-compatible with static shapes.
"""

import jax
import jax.numpy as jnp

from lcm.typing import FloatND, IntND

# A value as `(high, low, dropped)`: estimate `high + low`, tail bound `dropped`.
type DoubleDouble = tuple[FloatND, FloatND, FloatND]


def two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return `(s, e)` with `a + b == s + e` exactly (Knuth)."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def two_prod(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return `(p, e)` with `a * b == p + e` exactly (Dekker)."""
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
    terms are ignored; a group holding nothing else scales by `2**0`, which
    leaves it alone.
    """
    magnitude = jnp.zeros_like(values[0])
    for term in values:
        magnitude = jnp.maximum(magnitude, jnp.abs(term))
    usable = jnp.isfinite(magnitude) & (magnitude > 0.0)
    _mantissa, exponent = jnp.frexp(jnp.where(usable, magnitude, 1.0))
    return jnp.where(usable, exponent, jnp.zeros_like(exponent))


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
    return new_high, new_low, new_dropped + dropped * jnp.abs(factor)


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
        + left_dropped * right_scale
        + right_dropped * left_scale
        + left_dropped * right_dropped
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
    """Split `a` into two half-precision halves with `a == hi + lo` exactly."""
    n_mantissa = jnp.finfo(a.dtype).nmant
    factor = jnp.asarray(2.0 ** ((n_mantissa + 2) // 2) + 1.0, dtype=a.dtype)
    c = factor * a
    a_big = c - a
    a_hi = c - a_big
    return a_hi, a - a_hi
