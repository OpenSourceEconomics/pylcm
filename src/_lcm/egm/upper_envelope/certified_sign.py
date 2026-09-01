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

`D` is evaluated exactly, in integers. Each finite operand is decoded from its
stored bits into a signed dyadic $(-1)^s m 2^e$, the sixteen signed triple
products the determinant expands into are accumulated at full width, and the
verdict is the comparison of the positive and negative accumulators. No
subtraction, product, or equality test in the decision is a floating operation,
so the sign does not depend on the backend and a determinant far below the
smallest normal is read as the ordinary number it is.

One property follows, and it is what the envelope relies on: a tie is published
only for exact equality of the two rational lines. A crossing sitting exactly on
a node, or a link compared with itself, is *certified*; a difference that
underflowed is not, because nothing underflows.

- `UNRESOLVED_SIGN` — the comparison was refused rather than lost: an operand is
  non-finite, or a width is not strictly positive. Nothing follows about the
  geometry, which may be far apart, so a caller must fail loud rather than
  choose.

`BELOW_RESOLUTION_SIGN` has no route out of an exact comparator — a determinant
is nonzero or it is not — and remains only as the vocabulary its consumers read.

Callers must mask dead candidates and zero-width links before calling: a link of
zero width has no affine value line, and this predicate does not invent one.

Correctness is the design constraint here, not throughput: one comparison costs a
few hundred flops. The evaluation is branch-free and elementwise, so it stays
`jax.jit`- and `jax.vmap`-compatible with static shapes.
"""

import operator
from functools import cache, reduce
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from _lcm.egm.upper_envelope._exact_affine import certified_affine_compare
from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul,
    dd_mul_float,
    dd_negate,
    dd_quotient,
    is_stored_zero,
    normalizing_exponent,
    scale_by_power_of_two,
    two_sum,
)
from lcm.typing import BoolND, FloatND, IntND

# Returned where no usable determinant was produced — a non-finite input, an
# overflowing product, an operand the scaling flattened, or a positive distance
# the subtraction that forms it flushed to zero. Nothing is known about the
# geometry; callers must fail loud.
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
        `+1` where `A` is above `B`, `-1` where it is below, `0` where the two
        rational lines determined by the stored operands are exactly equal at the
        query, and `UNRESOLVED_SIGN` where an operand is non-finite or a width is
        not strictly positive. The verdict is exact, so it is the same on every
        backend and at every batch partition.

    """
    # The exact comparator settles every finite, positive-width case on its own,
    # so no shortcut is read off the operands first. The bit-equality tests that
    # used to precede the determinant — two links given by identical endpoints, a
    # query sitting on a node of both — existed because the floating determinant
    # could not separate a line from itself, and they are subsumed rather than
    # dropped: exact equality of the two rational lines is precisely what returns
    # zero here.
    return certified_affine_compare(
        a_x0=a_x0,
        a_x1=a_x1,
        a_v0=a_v0,
        a_v1=a_v1,
        b_x0=b_x0,
        b_x1=b_x1,
        b_v0=b_v0,
        b_v1=b_v1,
        x_query=x_query,
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
        _bounded_product(left=left_numerator, right=right_divisor),
        dd_negate(_bounded_product(left=right_numerator, right=left_divisor)),
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
    # Referring the bound back through the divisor is the one division a bound
    # passes through, and any divisor above one can send it under the smallest
    # normal — where it arrives as exactly zero, the certificate for an exact
    # margin, resting on a quantity that was destroyed. The underflow is itself
    # the statement that the referred amount is below the smallest normal, so
    # that is what it becomes. A residual of exactly zero reproduced the
    # determinant exactly and keeps its zero.
    referred = unreproduced / divisor_floor
    tiny = jnp.finfo(referred.dtype).tiny
    floored = jnp.where(
        is_stored_zero(unreproduced), referred, jnp.maximum(referred, tiny)
    )
    # The residual's own sum, the division, and this widening each round once; the
    # widening is multiplicative, so a residual of exactly zero stays exactly zero.
    bound = floored * (1.0 + 8.0 * epsilon)

    # Dekker's transform is exact only while its products stay normal. Above that
    # range the determinant is not evidence of anything, least of all of a tie.
    # Below it the numerator products are bounded rather than unknown, which
    # `_bounded_product` has already carried into the error bound; the divisor
    # product is not, since a quotient cannot be referred back through a divisor
    # whose own magnitude is in doubt.
    in_domain = (
        jnp.isfinite(left_numerator[0] * right_divisor[0])
        & jnp.isfinite(right_numerator[0] * left_divisor[0])
        & _product_in_transform_domain(a=left_divisor[0], b=right_divisor[0])
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


def _same_bits(*, left: FloatND, right: FloatND) -> BoolND:
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
        halved = jax.jit(lambda value: value * 0.5)(smallest_normal)
        return bool(np.asarray(halved) == 0.0)


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
    *, scaled: tuple[FloatND, ...], source: tuple[FloatND, ...], exponent: IntND
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


def _bounded_product(*, left: DoubleDouble, right: DoubleDouble) -> DoubleDouble:
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
    both_present = ~is_stored_zero(left[0]) & ~is_stored_zero(right[0])
    negligible = both_present & (jnp.abs(left[0] * right[0]) < tiny)
    zero = jnp.zeros_like(high)
    return (
        jnp.where(negligible, zero, high),
        jnp.where(negligible, zero, low),
        jnp.where(negligible, tiny, dropped),
    )


def _bounded_mul_float(
    *, value: DoubleDouble, factor: FloatND
) -> tuple[DoubleDouble, BoolND]:
    """Return the product by a plain float, or a certified bound where it underflows.

    A product of two ordinary numbers can still land among the subnormals — a
    distance below one against an endpoint value near the bottom of the range, or
    a value of any size against a distance that small — and there the transform
    keeps nothing. What arrives is an exact zero whose discarded tail also reads
    as exactly zero, and that pair is the certificate for an exact zero, so a
    determinant assembled from such terms is the determinant of a different pair
    of links.

    The magnitude is not unknown, though: it is below the smallest normal, which
    is exactly what an underflowing product tells you. Recording that as the
    term's discarded tail keeps the statement true and lets a determinant whose
    other terms are ordinary numbers still be decided. `scale_tail_bound` is what
    makes the bound survive the multiplications still to come.

    A term that is genuinely zero loses nothing and keeps its exactness, so two
    links lying on one line still certify the tie they have earned.

    The second return says whether the bound is that fallback rather than a
    measured tail, which is what decides how an inconclusive determinant
    abstains — not whether it may be strict.
    """
    high, low, dropped = dd_mul_float(value, factor)
    tiny = jnp.finfo(high.dtype).tiny
    both_present = ~is_stored_zero(value[0]) & ~is_stored_zero(factor)
    negligible = both_present & (jnp.abs(value[0] * factor) < tiny)
    zero = jnp.zeros_like(high)
    return (
        jnp.where(negligible, zero, high),
        jnp.where(negligible, zero, low),
        jnp.where(negligible, jnp.maximum(dropped, tiny), dropped),
    ), negligible


def _link_distances(
    *, x0: FloatND, x1: FloatND, x_query: FloatND
) -> tuple[tuple[DoubleDouble, ...], BoolND]:
    """Return one link's three distances and whether they scaled.

    A link enters the determinant only through the two distances from the query
    to its endpoints and the width between them, and every term of `D` pairs one
    of those with one of the other link's. So the whole triple may be measured on
    this link's own scale, and the positive factor that introduces cancels out of
    the sign.

    That is done twice, because the two scalings answer different failures:

    - The abscissae are scaled before the differences are taken. A link sitting
      at the bottom of the normal range has readable endpoints separated by a
      subnormal step, so the subtraction — not the storage — is what destroys it,
      and no later rescaling can recover a bit the subtraction did not produce.
    - The distances are scaled after. A link whose endpoints are ordinary
      neighbours has differences far below its own abscissae, and it is the
      cancellation *between* the two links' contributions that then underflows.

    A link that still cannot be lifted clear — a query far outside a narrow link
    pins the exponent on the query, and a link whose own endpoints span the
    format has no exponent at all — is left with a difference of exactly zero
    between operands that are not equal. That zero is the flush itself, and on
    its own it is indistinguishable from a link the caller supplied as a point,
    which is the shape that licenses a certified tie.

    The magnitude is known, though — a difference that flushed is below the
    smallest normal — so the distance carries that as its discarded tail and the
    two cases stop looking alike. A determinant whose other terms are ordinary
    numbers is then still decided, and one of the order of what went missing
    falls below resolution on the bound rather than on a flag. That is only
    possible because `scale_tail_bound` keeps the tail from being flushed in turn
    by the multiplications still to come.
    """
    source = (x1, x_query, x0)
    exponent = _shared_exponent(x0, x1, x_query)
    scaled = tuple(scale_by_power_of_two(term, -exponent) for term in source)
    scaled_x1, scaled_query, scaled_x0 = scaled
    pairs = (
        (scaled_x1, scaled_query),
        (scaled_query, scaled_x0),
        (scaled_x1, scaled_x0),
    )
    distances = tuple(_bounded_difference(a=left, b=right) for left, right in pairs)
    rescaled, on_scale = _on_its_own_scale(distances)
    return rescaled, on_scale & _round_trips(
        scaled=scaled, source=source, exponent=exponent
    )


def _bounded_difference(*, a: FloatND, b: FloatND) -> DoubleDouble:
    """Return `a - b`, recording the smallest normal as its tail where it flushed.

    Two operands that differ cannot have an exact difference of zero, so a zero
    arriving from a subtraction of unequal operands is the flush and nothing
    else. What it destroyed is below the smallest normal — that is what made it
    flush — so the smallest normal bounds it, and a distance that says so is a
    true statement where an exact-zero tail would be a false one.
    """
    high, low = two_sum(a, -b)
    tiny = jnp.finfo(high.dtype).tiny
    zero = jnp.zeros_like(high)
    flushed = (high == 0.0) & (low == 0.0) & (a != b)
    return high, low, jnp.where(flushed, jnp.full_like(high, tiny), zero)


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


def _product_in_transform_domain(*, a: FloatND, b: FloatND) -> BoolND:
    """Report whether `two_prod(a, b)` stays inside its exact domain.

    Dekker's transform is exact only while the product and the splitting
    intermediates stay normal. A product that underflows to zero, or lands among
    the subnormals, silently loses the tail the certificate reads — so such a
    product must never be mistaken for an exact zero.
    """
    product = jnp.abs(a * b)
    tiny = jnp.finfo(product.dtype).tiny
    both_present = ~is_stored_zero(a) & ~is_stored_zero(b)
    return jnp.isfinite(product) & (~both_present | (product >= tiny))


def _any_bound_floored(distances: tuple[DoubleDouble, ...]) -> BoolND:
    """Report where any of one link's distances carries a floored tail bound.

    A distance starts out exact, so the only way it acquires a tail at all is a
    floor — the flush of the difference itself, or the tail the rescaling could
    not keep. A nonzero tail here therefore identifies a bound that stands in
    for a magnitude rather than measuring one.
    """
    return reduce(
        operator.or_, (~is_stored_zero(distance[2]) for distance in distances)
    )


def _certified_sign_of(
    *,
    value: DoubleDouble,
    finite: BoolND,
    readable: BoolND,
    bound_is_a_fallback: BoolND,
) -> IntND:
    """Turn a double-double with an error bound into a certified sign.

    `readable` says whether everything the determinant was built from — the
    operands themselves, and the distances the subtractions between them
    produced — was something the arithmetic could see. Where it was not, no
    verdict survives, strict ones included. A flushed term is below the smallest
    normal only where it was discarded; the determinant is bilinear in the two
    links' distances, so whatever went missing is multiplied by the other link's
    width on the way here. Against a width near the top of the range that
    amplifies the missing amount past the tolerance the estimate is then
    compared to, and the resulting margin is strict, sizeable, and pointing the
    wrong way. Both ends of the range therefore report the same thing, which is
    that this comparison was never posed to the arithmetic.
    """
    high, low, dropped = value
    estimate = high + low
    epsilon = jnp.finfo(estimate.dtype).eps
    # `dropped` bounds the discarded tail; the final sum adds one more rounding.
    tolerance = dropped + epsilon * jnp.abs(estimate)
    exactly_zero = (dropped == 0.0) & (estimate == 0.0)
    unresolved = jnp.asarray(UNRESOLVED_SIGN, dtype=jnp.int32)
    below_resolution = jnp.asarray(BELOW_RESOLUTION_SIGN, dtype=jnp.int32)
    # `BELOW_RESOLUTION_SIGN` promises that the two lines are within a rounding
    # of each other, which is what licenses a caller to choose between them on
    # any deterministic rule. A determinant that failed to clear a *floored*
    # bound has earned no such statement — the floor says only that a term was
    # somewhere below the smallest normal — so it abstains as unresolved, where
    # a caller must fail loud.
    inconclusive = jnp.where(bound_is_a_fallback, unresolved, below_resolution)
    sign = jnp.where(
        estimate > tolerance,
        jnp.int32(1),
        jnp.where(
            estimate < -tolerance,
            jnp.int32(-1),
            jnp.where(exactly_zero, jnp.int32(0), inconclusive),
        ),
    )
    return jnp.where(finite & readable, sign, unresolved).astype(jnp.int32)
