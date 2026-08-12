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

import jax
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
        `0` where the difference is certified exactly zero,
        `BELOW_RESOLUTION_SIGN` where the determinant is under its own error
        bound, and `UNRESOLVED_SIGN` where none could be computed (including any
        non-finite input). Two cases are read straight off the operands and never
        reach the determinant: two lines given by the same endpoints are one
        line, and a query on an endpoint of both lines is decided by the two
        stored values there.

    """
    operands = (a_x0, a_x1, a_v0, a_v1, b_x0, b_x1, b_v0, b_v1, x_query)
    finite = reduce(operator.and_, (jnp.isfinite(term) for term in operands))

    # A subnormal operand is not a small number here, it is an unreadable one.
    # The backend flushes subnormals to zero in arithmetic and in comparison
    # alike, so every difference formed from one is taken against zero rather
    # than against the operand, and the determinant that results is a true
    # statement about geometry the caller never supplied. It is not a bounded
    # error either: with `x0 = 0`, `x1 = tiny` and a subnormal query the two
    # numerators both collapse to exactly zero, `dropped` is exactly zero
    # because nothing was discarded, and the predicate certifies an exact tie
    # between lines that are strictly ordered — the one outcome that licenses a
    # caller to choose freely.
    #
    # Nothing downstream can undo that, because the magnitude is gone before the
    # first subtraction. So it is refused here, where a caller must fail loud,
    # rather than bounded somewhere that would report it as a near-tie. Reading
    # such an operand at all needs an arithmetic over significands and exponents
    # rather than over floats.
    #
    # The refusal is silent at this point, so what a user sees is a NaN-bearing
    # solution and a generic `InvalidValueFunctionError`. Naming itself is the
    # ordinary poison-then-raise-at-debug strategy, but the trigger is a tracer,
    # so the flag has to be returned from JIT and raised host-side after
    # synchronization — and no function on the path from here to the solve
    # boundary carries a logger to raise from. What such an operand would take
    # to read, and why both halves are deferred rather than dropped, is recorded
    # in https://github.com/OpenSourceEconomics/pylcm/issues/425.
    readable = ~reduce(operator.or_, (is_subnormal(term) for term in operands))

    # Two lines given by the same four endpoints are one line, and one line is
    # exactly level with itself at every query. Read off the operands, this
    # costs four comparisons and is exactly true.
    #
    # The determinant cannot reach that conclusion on its own. Its two products
    # are then formed from identical operands, so they discard identical tails
    # and the tails cancel with everything else — but `dropped` accumulates them
    # as though they were independent errors, which is sound and, here, not
    # tight. What comes back is a zero estimate carrying a positive bound, which
    # is `BELOW_RESOLUTION_SIGN`: the arithmetic reporting it could not separate
    # a line from itself. Callers that compare a set against one of its own
    # members would then find no certified tie anywhere, including with the
    # member they took the reference from.
    same_line = (
        _same_bits(a_x0, b_x0)
        & _same_bits(a_x1, b_x1)
        & _same_bits(a_v0, b_v0)
        & _same_bits(a_v1, b_v1)
    )

    # A query sitting on an endpoint of both lines is settled by comparing the
    # two stored values there. An affine line takes exactly its endpoint value at
    # its own endpoint, so this is the difference itself rather than an estimate
    # of it, and a comparison of two stored floats decides it without arithmetic.
    #
    # This is the common case, not a corner: every candidate is also a zero-width
    # self-bracket at its own abscissa, and consecutive links share the node
    # between them. It is also where the determinant is weakest, because the
    # bound it carries is set by the largest operand in the group rather than by
    # anything near the query — one steep link to a far-away neighbour widens the
    # bound past a difference that is exactly zero.
    # Whether the query *is* a node is asked of the bit patterns, not of the
    # floats. Under flush-to-zero a subnormal query compares equal to a zero
    # node while being a strictly interior point of the link, so the float test
    # fires where the shortcut does not apply and then reads the wrong pair of
    # endpoint values — publishing a confident tie for lines that are strictly
    # ordered. Bit equality cannot say that. It declines two representations of
    # zero, which costs only the shortcut and falls back to the determinant.
    a_at_low = _same_bits(x_query, a_x0)
    b_at_low = _same_bits(x_query, b_x0)
    a_node_value = jnp.where(a_at_low, a_v0, a_v1)
    b_node_value = jnp.where(b_at_low, b_v0, b_v1)
    both_at_node = (a_at_low | _same_bits(x_query, a_x1)) & (
        b_at_low | _same_bits(x_query, b_x1)
    )
    # The values themselves are compared as floats, which is exact for an
    # ordering: two stored floats that differ compare as they differ. Only the
    # subnormal *magnitudes* are unreadable, and a magnitude is not what an
    # ordering needs — but two distinct subnormals do compare equal, so a tie
    # they report is not certified and is handed back to the determinant.
    both_readable = ~is_subnormal(a_node_value) & ~is_subnormal(b_node_value)
    node_sign = jnp.where(
        a_node_value > b_node_value,
        jnp.int32(1),
        jnp.where(a_node_value < b_node_value, jnp.int32(-1), jnp.int32(0)),
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

    # One shared abscissa exponent keeps the whole group normal, but it cannot
    # put two links of very different widths in the same part of the range: the
    # narrow one keeps its width and lands wherever that is. Everything the
    # determinant reads from a link — both distances to the query, and the width
    # — carries that link's own scale, so each link's contribution can sit far
    # below one while every operand remains an ordinary number, and what cancels
    # between the two contributions is smaller again by the ratio between them.
    # The subtraction then falls under the smallest normal.
    #
    # Nothing else sees that. The operands are readable, each product stays
    # inside the domain where the transforms are exact, and the transforms
    # discard nothing on the way — so a backend that flushes the difference
    # returns an estimate of exactly zero carrying an error bound of exactly
    # zero, which is the certificate for an exact tie, for links that are
    # strictly ordered.
    #
    # `D` is bilinear in the two links' distances: every term it is built from
    # multiplies one distance taken from `A` by one taken from `B`. So each link
    # may be measured on its own scale — the two factors multiply `D` by a
    # positive constant and leave its sign alone — and normalizing them
    # separately is what puts the cancellation where the format has precision to
    # hold it.
    distances_a, a_on_scale = _on_its_own_scale(
        (
            dd_from_difference(a_x1, x_query),
            dd_from_difference(x_query, a_x0),
            dd_from_difference(a_x1, a_x0),
        )
    )
    distances_b, b_on_scale = _on_its_own_scale(
        (
            dd_from_difference(b_x1, x_query),
            dd_from_difference(x_query, b_x0),
            dd_from_difference(b_x1, b_x0),
        )
    )
    numerator_a = dd_add(
        dd_mul_float(distances_a[0], a_v0), dd_mul_float(distances_a[1], a_v1)
    )
    numerator_b = dd_add(
        dd_mul_float(distances_b[0], b_v0), dd_mul_float(distances_b[1], b_v1)
    )
    width_a, width_b = distances_a[2], distances_b[2]

    product_a = _bounded_product(numerator_a, width_b)
    product_b = _bounded_product(numerator_b, width_a)
    determinant = dd_add(product_a, dd_negate(product_b))

    # A product that leaves the top of the range says nothing about the determinant
    # it was meant to contribute to, so it stays unresolved. The bottom of the range
    # is not symmetric with it and is handled inside `_bounded_product`.
    products_finite = jnp.isfinite(product_a[0]) & jnp.isfinite(product_b[0])

    sign = _certified_sign_of(
        determinant,
        finite=finite & products_finite & scaling_exact & a_on_scale & b_on_scale,
        readable=readable,
    )
    exact = jnp.where(same_line, jnp.int32(0), node_sign)
    settled_off_the_operands = same_line | (both_at_node & both_readable)
    return jnp.where(finite & settled_off_the_operands, exact, sign).astype(jnp.int32)


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


def affine_numerator(
    *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
) -> DoubleDouble:
    """Return `v0*(x1 - x) + v1*(x - x0)`, the width-scaled value at `x`."""
    return dd_add(
        dd_mul_float(dd_from_difference(x1, x_query), v0),
        dd_mul_float(dd_from_difference(x_query, x0), v1),
    )


def _same_bits(left: FloatND, right: FloatND) -> BoolND:
    """Report whether two floats have identical representations.

    Float equality is the wrong instrument wherever an operand may be
    subnormal. The backend flushes subnormals to zero in comparisons as well as
    in arithmetic, so `==` reports a subnormal equal to zero and equal to every
    other subnormal — statements about the backend, not about the operands. Two
    identical representations are identical numbers whatever the backend does
    with them.

    Signed zeros have different representations and are declined, which costs
    only a shortcut and never an answer.
    """
    unsigned = _IEEE_FIELDS[jnp.dtype(left.dtype).name][0]
    return jax.lax.bitcast_convert_type(left, jnp.dtype(unsigned)) == (
        jax.lax.bitcast_convert_type(right, jnp.dtype(unsigned))
    )


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


# The unsigned type each float bitcasts to, and the masks selecting its biased
# exponent and its significand. A subnormal is exactly a zero exponent field over
# a nonzero significand. The masks stay plain integers: the 64-bit ones do not fit
# a 32-bit word, and building them as arrays here would fail at import in the
# default configuration, which does not enable 64-bit types at all.
_IEEE_FIELDS = {
    "float32": ("uint32", 0x7F800000, 0x007FFFFF),
    "float64": ("uint64", 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}


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


def _on_its_own_scale(
    distances: tuple[DoubleDouble, ...],
) -> tuple[tuple[DoubleDouble, ...], BoolND]:
    """Return one link's distances scaled together into the binade around one.

    The three of them — the two distances from the query to the endpoints, and
    the width between the endpoints — are the only way this link enters the
    determinant, and every term multiplies exactly one of them by exactly one of
    the other link's. So they move together by one power of two, and what that
    contributes to the determinant is a positive constant factor.

    The exponent is the one that suits the whole triple rather than its largest
    member. A query sitting one ulp from an endpoint of a link that spans most of
    the range puts a distance near the top of the format alongside one near the
    bottom, and landing the first on one would flatten the second — refusing a
    comparison that the same three distances decide easily when left further from
    one.

    Scaling by a power of two is exact only while the result stays normal, and
    the two halves of a distance do not reach that limit together: the low half
    trails the high one by the format's precision, so a scaling that lands the
    high half near the bottom of the normal range flushes the low half on a
    backend that flushes at all. Refusing on that would refuse the comparison
    over a term the certificate already knows how to carry — a tail below the
    smallest normal is bounded by it, which is what the error bound is for, and
    the same statement `_bounded_product` makes about an underflowing product.
    So only the leading half has to survive, and the flag reports that; a tail
    that does not survive is dropped and its magnitude added to the bound.
    """
    exponent = _shared_exponent(*(term[0] for term in distances))
    tiny = jnp.finfo(distances[0][0].dtype).tiny
    rescaled: list[DoubleDouble] = []
    leading_survived: list[BoolND] = []
    for high, low, dropped in distances:
        scaled_high = scale_by_power_of_two(high, -exponent)
        scaled_low = scale_by_power_of_two(low, -exponent)
        scaled_dropped = scale_by_power_of_two(dropped, -exponent)
        leading_survived.append(scale_by_power_of_two(scaled_high, exponent) == high)
        tail_survived = (scale_by_power_of_two(scaled_low, exponent) == low) & (
            scale_by_power_of_two(scaled_dropped, exponent) == dropped
        )
        rescaled.append(
            (
                scaled_high,
                jnp.where(tail_survived, scaled_low, jnp.zeros_like(scaled_low)),
                jnp.where(
                    tail_survived, scaled_dropped, jnp.maximum(scaled_dropped, tiny)
                ),
            )
        )
    return tuple(rescaled), reduce(operator.and_, leading_survived)


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


def _certified_sign_of(
    value: DoubleDouble, *, finite: BoolND, readable: BoolND
) -> IntND:
    """Turn a double-double with an error bound into a certified sign.

    `readable` says whether every operand was one the arithmetic could see. Where
    it was not, only a *strict* verdict survives: the margin then exceeds the
    tolerance by more than a flushed operand could have contributed, since a
    flushed operand is below the smallest normal while the margin is not. What
    does not survive is the near-zero end, where the whole difference is of the
    order of what went missing — and that is the end where the collapse is
    total, leaving an estimate of exactly zero with an error bound of exactly
    zero, because nothing was rounded on the way to losing everything.
    """
    high, low, dropped = value
    estimate = high + low
    epsilon = jnp.finfo(estimate.dtype).eps
    # `dropped` bounds the discarded tail; the final sum adds one more rounding.
    tolerance = dropped + epsilon * jnp.abs(estimate)
    exactly_zero = (dropped == 0.0) & (estimate == 0.0)
    unresolved = jnp.asarray(UNRESOLVED_SIGN, dtype=jnp.int32)
    below_resolution = jnp.asarray(BELOW_RESOLUTION_SIGN, dtype=jnp.int32)
    undecided = jnp.where(
        readable,
        jnp.where(exactly_zero, jnp.int32(0), below_resolution),
        unresolved,
    )
    sign = jnp.where(
        estimate > tolerance,
        jnp.int32(1),
        jnp.where(estimate < -tolerance, jnp.int32(-1), undecided),
    )
    return jnp.where(finite, sign, unresolved).astype(jnp.int32)
