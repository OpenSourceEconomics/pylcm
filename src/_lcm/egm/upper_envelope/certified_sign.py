r"""Certified sign of the difference of two affine value lines.

Upper-envelope construction is a sequence of *structural* decisions — which link
owns a cell, whether two links cross inside it, whether a crossing coincides with
a node. A magnitude-scaled tolerance cannot make those decisions: it is not
invariant to a common additive value level, so it masks a strictly better branch
whenever values sit on a large constant, and it promotes a below-envelope
crossing after the same shift.

This module decides them from the sign of the exact difference instead. For two
links given by their endpoints the value at `x` is

```{math}
A(x) = \frac{a_{v0}(a_{x1} - x) + a_{v1}(x - a_{x0})}{a_{x1} - a_{x0}},
```

so with positive widths the sign of `A(x) - B(x)` is the sign of the
cross-multiplied determinant

```{math}
D = N_a w_b - N_b w_a .
```

`D` is evaluated in the double-double arithmetic of `double_double`, whose
error-free transforms are exact. The only inexactness is the low-order tail each
renormalization discards, and that tail is captured exactly and accumulated into
a `dropped` bound. Two properties follow, and they are what the envelope relies
on:

- `dropped` is exactly zero whenever the whole evaluation was exact, so a genuine
  tie (a crossing sitting exactly on a node, or a link compared with itself) is
  *certified* rather than inferred from a threshold;
- otherwise the true determinant is within `dropped` of the computed one, so a
  sign is published only when it is certain.

Two things can stop a sign being published, and they are not the same, so they
are reported apart:

- `BELOW_RESOLUTION_SIGN` — the determinant was computed, but it is smaller than
  its own error bound. The links are then within a rounding of each other, so no
  state between them is demonstrably better and a caller may choose either,
  provided it chooses deterministically.
- `UNRESOLVED_SIGN` — no determinant worth reading was produced at all: an input
  was non-finite, a product overflowed, or an operand did not survive the shared
  scaling intact. Nothing follows about the geometry, which may be far apart, so
  a caller must fail loud rather than choose.

A product that underflows is *not* one of those cases. It has a known magnitude
bound — below the smallest normal — which the error bound carries, so a group
spanning more of the exponent range than any one scaling can hold is still
decided by whichever term is an ordinary number.

Collapsing the two would be a fail-open: the second case is exactly the one where
a large true margin can be reported as no margin. Callers must mask dead
candidates and zero-width links before calling: a link of zero width has no
affine value line, and this predicate does not invent one.

Correctness is the design constraint here, not throughput: one comparison costs a
few hundred flops. The evaluation is branch-free and elementwise, so it stays
`jax.jit`- and `jax.vmap`-compatible with static shapes.
"""

import operator
from functools import reduce
from typing import NamedTuple

import jax.numpy as jnp

from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul,
    dd_mul_float,
    dd_negate,
    dd_quotient,
    normalizing_exponent,
    scale_by_power_of_two,
)
from lcm.typing import BoolND, FloatND, IntND

# Returned where no usable determinant was produced — a non-finite input, an
# overflowing product, or a positive width the shared scaling flattened. Nothing
# is known about the geometry; callers must fail loud.
UNRESOLVED_SIGN: int = 2

# Returned where the determinant is real but smaller than its own error bound:
# the links are within a rounding, so either may be chosen, deterministically.
BELOW_RESOLUTION_SIGN: int = 3


def certified_margin_sign(
    *,
    a_x0: FloatND,
    a_x1: FloatND,
    a_v0: FloatND,
    a_v1: FloatND,
    b_x0: FloatND,
    b_x1: FloatND,
    b_v0: FloatND,
    b_v1: FloatND,
    x_query: FloatND,
) -> IntND:
    """Return the certified sign of `A(x_query) - B(x_query)`.

    `A` and `B` are the affine lines through the given endpoints, extended beyond
    them: the caller decides which cell a comparison belongs to, this predicate
    only settles the sign there.

    Args:
        a_x0: Lower endpoint abscissa of the first link.
        a_x1: Upper endpoint abscissa of the first link; must exceed `a_x0`.
        a_v0: Value of the first link at `a_x0`.
        a_v1: Value of the first link at `a_x1`.
        b_x0: Lower endpoint abscissa of the second link.
        b_x1: Upper endpoint abscissa of the second link; must exceed `b_x0`.
        b_v0: Value of the second link at `b_x0`.
        b_v1: Value of the second link at `b_x1`.
        x_query: Abscissa at which the two lines are compared.

    Returns:
        `+1` where `A` is certainly above `B`, `-1` where it is certainly below,
        `0` where the difference is certified exactly zero, and
        `BELOW_RESOLUTION_SIGN` where the determinant is under its own error
        bound, and `UNRESOLVED_SIGN` where none could be computed (including any
        non-finite input).

    """
    finite = (
        jnp.isfinite(a_x0)
        & jnp.isfinite(a_x1)
        & jnp.isfinite(a_v0)
        & jnp.isfinite(a_v1)
        & jnp.isfinite(b_x0)
        & jnp.isfinite(b_x1)
        & jnp.isfinite(b_v0)
        & jnp.isfinite(b_v1)
        & jnp.isfinite(x_query)
    )

    # `D` is homogeneous of degree two in the abscissae and degree one in the
    # values, so scaling each group by a power of two multiplies `D` by a
    # positive power of two and leaves its sign alone. Pulling the operands into
    # the binade around one is what keeps the products out of the range where the
    # error-free transforms stop being error-free: a determinant that would
    # underflow to zero in the caller's units is an ordinary number in these.
    abscissa_exponent = _shared_exponent(a_x0, a_x1, b_x0, b_x1, x_query)
    value_exponent = _shared_exponent(a_v0, a_v1, b_v0, b_v1)
    source_abscissae = (a_x0, a_x1, b_x0, b_x1, x_query)
    source_values = (a_v0, a_v1, b_v0, b_v1)
    scaled_abscissae = tuple(
        scale_by_power_of_two(term, -abscissa_exponent) for term in source_abscissae
    )
    scaled_values = tuple(
        scale_by_power_of_two(term, -value_exponent) for term in source_values
    )

    # That homogeneity argument holds only while the scaling is exact. The shared
    # exponent is chosen to keep every operand normal, so an operand far below the
    # largest one ordinarily survives — but a backend that reads subnormals as zero
    # can still flatten one, and a group spanning more than the whole exponent range
    # leaves no exponent that suits every term. What follows from a flattened
    # operand is a true statement about geometry the caller never supplied — a
    # narrow link whose endpoints have rounded onto the same number, most
    # emphatically a certified tie licensing either link. Scaling back is exact
    # whenever the scaling was, so the round trip tests that premise itself rather
    # than any one way of breaking it.
    scaling_exact = _round_trips(
        scaled_abscissae, source_abscissae, abscissa_exponent
    ) & _round_trips(scaled_values, source_values, value_exponent)

    a_x0, a_x1, b_x0, b_x1, x_query = scaled_abscissae
    a_v0, a_v1, b_v0, b_v1 = scaled_values

    numerator_a = affine_numerator(x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1, x_query=x_query)
    numerator_b = affine_numerator(x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1, x_query=x_query)
    width_a = dd_from_difference(a_x1, a_x0)
    width_b = dd_from_difference(b_x1, b_x0)

    product_a = _bounded_product(numerator_a, width_b)
    product_b = _bounded_product(numerator_b, width_a)
    determinant = dd_add(product_a, dd_negate(product_b))

    # A product that leaves the top of the range says nothing about the determinant
    # it was meant to contribute to, so it stays unresolved. The bottom of the range
    # is not symmetric with it and is handled inside `_bounded_product`.
    products_finite = jnp.isfinite(product_a[0]) & jnp.isfinite(product_b[0])

    return _certified_sign_of(
        determinant, finite=finite & products_finite & scaling_exact
    )


class QuotientMargin(NamedTuple):
    """How far one quotient lies above another, and whether that is knowable."""

    value: FloatND
    """`left_numerator/left_divisor - right_numerator/right_divisor`."""
    bound: FloatND
    """The true margin lies within `bound` of `value`."""
    trustworthy: BoolND
    """Whether the evaluation stayed where the transforms — and so `bound` — hold."""


def certified_quotient_margin(
    *,
    left_numerator: DoubleDouble,
    left_divisor: DoubleDouble,
    right_numerator: DoubleDouble,
    right_divisor: DoubleDouble,
) -> QuotientMargin:
    """Return how far the left quotient lies above the right one, with a bound.

    Reading each quotient and subtracting the two results bounds their difference
    at the *values'* magnitude, which is the wrong scale to decide between them: on
    a large common value level two such bounds swamp a gap that is orders of
    magnitude above zero, and an ordering the format holds exactly is reported as
    a tie. That is not a defect of either read — each is as good as its own
    magnitude allows — but of asking a question about a difference by way of two
    separate answers.

    Cross-multiplying first asks it directly. `N_l w_r - N_r w_l` is formed in the
    double-double arithmetic of `double_double`, whose transforms are exact, so the
    common level cancels in arithmetic that loses nothing. What reaches the bound is
    only the tail the two multiplications discard — second order in the format's
    precision, against a first-order rounding of the level — so the margin stays
    decidable on a level many orders of magnitude above the gap, and a common
    additive shift of both value lines does not change the outcome until it exhausts
    that second-order headroom.

    Args:
        left_numerator: Numerator of the left quotient.
        left_divisor: Divisor of the left quotient; must be non-zero.
        right_numerator: Numerator of the right quotient.
        right_divisor: Divisor of the right quotient; must be non-zero.

    Returns:
        The margin, a bound on it, and whether the bound may be relied on. Where it
        may not, nothing follows about the geometry — the true margin may be large —
        so a caller must fail loud rather than treat it as a tie.

    """
    determinant = dd_add(
        _bounded_product(left_numerator, right_divisor),
        dd_negate(_bounded_product(right_numerator, left_divisor)),
    )
    divisor_product = dd_mul(left_divisor, right_divisor)
    high, low = dd_quotient(determinant, divisor_product)
    value = high + low

    # The bound is a residual, taken against the single float that is published
    # rather than against the pair it came from. How well a quotient *pair*
    # reproduces its numerator says nothing about the float the caller acts on,
    # and the two differ by more than the pair's own accuracy suggests; the
    # residual of the published value has no such gap by construction. It is also
    # exactly zero for a quotient that divides out exactly, which is what lets an
    # exact tie be certified rather than inferred.
    residual = dd_add(determinant, dd_negate(dd_mul_float(divisor_product, value)))
    unreproduced = jnp.abs(residual[0] + residual[1]) + residual[2]
    # Referring the residual back through the divisor must not understate it, so
    # it is divided by a *lower* bound on the divisor rather than its leading word.
    divisor_floor = (
        jnp.abs(divisor_product[0])
        - jnp.abs(divisor_product[1])
        - jnp.abs(divisor_product[2])
    )
    epsilon = jnp.finfo(value.dtype).eps
    # The residual's own sum, the division, and this widening each round once; the
    # widening is multiplicative, so a residual of exactly zero stays exactly zero.
    bound = (unreproduced / divisor_floor) * (1.0 + 8.0 * epsilon)

    # Dekker's transform is exact only while its products stay normal. Above that
    # range the determinant is not evidence of anything, least of all of a tie.
    # Below it the numerator products are bounded rather than unknown, which
    # `_bounded_product` has already carried into the error bound; the divisor
    # product is not, since a quotient cannot be referred back through a divisor
    # whose own magnitude is in doubt.
    in_domain = (
        jnp.isfinite(left_numerator[0] * right_divisor[0])
        & jnp.isfinite(right_numerator[0] * left_divisor[0])
        & _product_in_transform_domain(left_divisor[0], right_divisor[0])
    )
    return QuotientMargin(
        value=value,
        bound=bound,
        trustworthy=in_domain
        & jnp.isfinite(value)
        & jnp.isfinite(bound)
        & (divisor_floor > 0.0),
    )


class QuotientMargin(NamedTuple):
    """How far one quotient lies above another, and whether that is knowable."""

    value: FloatND
    """`left_numerator/left_divisor - right_numerator/right_divisor`."""
    bound: FloatND
    """The true margin lies within `bound` of `value`."""
    trustworthy: BoolND
    """Whether the evaluation stayed where the transforms — and so `bound` — hold."""


def certified_quotient_margin(
    *,
    left_numerator: DoubleDouble,
    left_divisor: DoubleDouble,
    right_numerator: DoubleDouble,
    right_divisor: DoubleDouble,
) -> QuotientMargin:
    """Return how far the left quotient lies above the right one, with a bound.

    Reading each quotient and subtracting the two results bounds their difference
    at the *values'* magnitude, which is the wrong scale to decide between them: on
    a large common value level two such bounds swamp a gap that is orders of
    magnitude above zero, and an ordering the format holds exactly is reported as
    a tie. That is not a defect of either read — each is as good as its own
    magnitude allows — but of asking a question about a difference by way of two
    separate answers.

    Cross-multiplying first asks it directly. `N_l w_r - N_r w_l` is formed in the
    double-double arithmetic of `double_double`, whose transforms are exact, so the
    common level cancels in arithmetic that loses nothing. What reaches the bound is
    only the tail the two multiplications discard — second order in the format's
    precision, against a first-order rounding of the level — so the margin stays
    decidable on a level many orders of magnitude above the gap, and a common
    additive shift of both value lines does not change the outcome until it exhausts
    that second-order headroom.

    Args:
        left_numerator: Numerator of the left quotient.
        left_divisor: Divisor of the left quotient; must be non-zero.
        right_numerator: Numerator of the right quotient.
        right_divisor: Divisor of the right quotient; must be non-zero.

    Returns:
        The margin, a bound on it, and whether the bound may be relied on. Where it
        may not, nothing follows about the geometry — the true margin may be large —
        so a caller must fail loud rather than treat it as a tie.

    """
    determinant = dd_add(
        dd_mul(left_numerator, right_divisor),
        dd_negate(dd_mul(right_numerator, left_divisor)),
    )
    divisor_product = dd_mul(left_divisor, right_divisor)
    high, low = dd_quotient(determinant, divisor_product)
    value = high + low

    # The bound is a residual, taken against the single float that is published
    # rather than against the pair it came from. How well a quotient *pair*
    # reproduces its numerator says nothing about the float the caller acts on,
    # and the two differ by more than the pair's own accuracy suggests; the
    # residual of the published value has no such gap by construction. It is also
    # exactly zero for a quotient that divides out exactly, which is what lets an
    # exact tie be certified rather than inferred.
    residual = dd_add(determinant, dd_negate(dd_mul_float(divisor_product, value)))
    unreproduced = jnp.abs(residual[0] + residual[1]) + residual[2]
    # Referring the residual back through the divisor must not understate it, so
    # it is divided by a *lower* bound on the divisor rather than its leading word.
    divisor_floor = (
        jnp.abs(divisor_product[0])
        - jnp.abs(divisor_product[1])
        - jnp.abs(divisor_product[2])
    )
    epsilon = jnp.finfo(value.dtype).eps
    # The residual's own sum, the division, and this widening each round once; the
    # widening is multiplicative, so a residual of exactly zero stays exactly zero.
    bound = (unreproduced / divisor_floor) * (1.0 + 8.0 * epsilon)

    # Dekker's transform is exact only while its products stay normal. Outside that
    # range the determinant is not evidence of anything, least of all of a tie.
    in_domain = (
        _product_in_transform_domain(left_numerator[0], right_divisor[0])
        & _product_in_transform_domain(right_numerator[0], left_divisor[0])
        & _product_in_transform_domain(left_divisor[0], right_divisor[0])
    )
    return QuotientMargin(
        value=value,
        bound=bound,
        trustworthy=in_domain
        & jnp.isfinite(value)
        & jnp.isfinite(bound)
        & (divisor_floor > 0.0),
    )


def affine_numerator(
    *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
) -> DoubleDouble:
    """Return `v0*(x1 - x) + v1*(x - x0)`, the width-scaled value at `x`."""
    return dd_add(
        dd_mul_float(dd_from_difference(x1, x_query), v0),
        dd_mul_float(dd_from_difference(x_query, x0), v1),
    )


def _round_trips(
    scaled: tuple[FloatND, ...], source: tuple[FloatND, ...], exponent: IntND
) -> BoolND:
    """Report whether scaling every operand by `2**-exponent` lost nothing.

    Multiplying by a power of two is exact unless the result leaves the normal
    range, and scaling back is exact under the same condition, so an operand that
    returns to where it started passed through the scaling untouched.
    """
    return reduce(
        operator.and_,
        (
            scale_by_power_of_two(scaled_term, exponent) == source_term
            for scaled_term, source_term in zip(scaled, source, strict=True)
        ),
    )


def _shared_exponent(*terms: FloatND) -> IntND:
    """Return the exponent by which a group of operands is scaled.

    `normalizing_exponent` lands the largest term near one, which is what keeps
    the products inside the domain where the transforms are exact. A group whose
    magnitudes span more than the format's exponent range cannot have that for
    every term at once: scaling the largest term down to one pushes the smallest
    into the subnormals, where the scaling stops being exact and the comparison
    is refused — although a pair that far apart is the easiest comparison there
    is. Backing the exponent off keeps every term normal instead. The largest
    term then sits further from one than it otherwise would, which costs only the
    headroom the product's own range check already polices.
    """
    largest = normalizing_exponent(*terms)
    # `frexp` reports the smallest normal as `0.5 * 2**(minexp + 1)`, so a term
    # whose scaled exponent stays at or above that bound stays normal.
    smallest_normal_exponent = jnp.finfo(terms[0].dtype).minexp + 1
    return jnp.minimum(largest, _smallest_exponent(*terms) - smallest_normal_exponent)


def _smallest_exponent(*terms: FloatND) -> IntND:
    """Return the `frexp` exponent of the smallest finite nonzero term.

    Zero and non-finite terms carry no scale of their own and are ignored; a
    group holding nothing else scales by `2**0`, which leaves it alone.
    """
    magnitude = jnp.full_like(terms[0], jnp.inf)
    for term in terms:
        usable = jnp.isfinite(term) & (term != 0.0)
        magnitude = jnp.minimum(magnitude, jnp.where(usable, jnp.abs(term), jnp.inf))
    _mantissa, exponent = jnp.frexp(
        jnp.where(jnp.isfinite(magnitude), magnitude, 1.0),
    )
    return exponent


def _bounded_product(left: DoubleDouble, right: DoubleDouble) -> DoubleDouble:
    """Return the product, or a certified bound where it underflows.

    Dekker's transform is exact only while the product and the splitting
    intermediates stay normal, so a product landing among the subnormals loses
    the tail the certificate reads and must never be mistaken for an exact zero.
    It is not unknown, though: its magnitude is below the smallest normal. Saying
    exactly that — an exact zero carrying that magnitude as its discarded tail —
    is a true statement the error bound already knows how to carry, and it is
    what lets a determinant whose other term is an ordinary number still be
    decided. Two negligible terms fall below resolution rather than certifying a
    tie they have not earned.
    """
    high, low, dropped = dd_mul(left, right)
    tiny = jnp.finfo(high.dtype).tiny
    both_nonzero = (left[0] != 0.0) & (right[0] != 0.0)
    negligible = both_nonzero & (jnp.abs(left[0] * right[0]) < tiny)
    zero = jnp.zeros_like(high)
    return (
        jnp.where(negligible, zero, high),
        jnp.where(negligible, zero, low),
        jnp.where(negligible, tiny, dropped),
    )


def _product_in_transform_domain(a: FloatND, b: FloatND) -> BoolND:
    """Report whether `two_prod(a, b)` stays inside its exact domain.

    Dekker's transform is exact only while the product and the splitting
    intermediates stay normal. A product that underflows to zero, or lands among
    the subnormals, silently loses the tail the certificate reads — so such a
    product must never be mistaken for an exact zero.
    """
    product = jnp.abs(a * b)
    tiny = jnp.finfo(product.dtype).tiny
    both_nonzero = (a != 0.0) & (b != 0.0)
    return jnp.isfinite(product) & (~both_nonzero | (product >= tiny))


def _certified_sign_of(value: DoubleDouble, *, finite: BoolND) -> IntND:
    """Turn a double-double with an error bound into a certified sign."""
    high, low, dropped = value
    estimate = high + low
    epsilon = jnp.finfo(estimate.dtype).eps
    # `dropped` bounds the discarded tail; the final sum adds one more rounding.
    tolerance = dropped + epsilon * jnp.abs(estimate)
    exactly_zero = (dropped == 0.0) & (estimate == 0.0)
    unresolved = jnp.asarray(UNRESOLVED_SIGN, dtype=jnp.int32)
    below_resolution = jnp.asarray(BELOW_RESOLUTION_SIGN, dtype=jnp.int32)
    sign = jnp.where(
        estimate > tolerance,
        jnp.int32(1),
        jnp.where(
            estimate < -tolerance,
            jnp.int32(-1),
            jnp.where(exactly_zero, jnp.int32(0), below_resolution),
        ),
    )
    return jnp.where(finite, sign, unresolved).astype(jnp.int32)
