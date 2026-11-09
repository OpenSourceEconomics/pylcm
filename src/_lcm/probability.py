"""One reading of a probability's bits, shared by every consumer.

A probability arrives as a floating-point number, and three questions get
asked of it all over the engine: is it the null event, is it a valid
probability at all, and is it small enough that the arithmetic will lose it.
Asked with `==`, `<` and `>` the answers are wrong on any backend that
flushes subnormal operands, because a strictly positive probability below the
dtype's normal range then compares equal to zero. Asked here they are read
from the bit pattern and are the same on every backend.

The consequence of asking twice, in two different ways, is not a rounding
difference. A node the engine calls impossible has its value overwritten by a
neighbour's, so an event carrying `-inf` loses the infinity; a negative
probability that arithmetic reports as `-0` passes the distribution guard;
and under a nonlinear certainty equivalent a node's transformed value can be
large enough that dropping it moves the answer by its own magnitude. So every
consumer reads a weight through this module, and none re-derives these
properties from a comparison.

`rescaled_lottery_weights` is the other half. A weighted mean is invariant to
scaling every weight by a common factor, and a power of two scales exactly,
so a lottery carrying a weight the dtype cannot use is rescaled until every
live weight is a normal number. Downstream arithmetic — including a
user-written `aggregate`, which this module cannot see — then never has a
subnormal operand, and the ratio between any two weights is the one the
caller supplied, bit for bit.
"""

import dataclasses
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Int

from lcm.typing import BoolND, FloatND, ValueND

# The bit pattern of a float, as an integer of the same width. Genuinely
# width-polymorphic: `int32` beside `float32`, `int64` beside `float64`.
type _BitsND = Int[Array, "..."]

# The tangent JAX pairs with an integer primal: a `float0` zero array, held on
# the host or traced.
type _IntegerTangent = ValueND | np.ndarray

_FLOAT32_BYTES = 4
_FLOAT32_MANTISSA_BITS = 23
_FLOAT64_MANTISSA_BITS = 52

# Gross departures from unit regime mass are specification errors. Continuation
# aggregators normalize their weights, so missing mass would otherwise disappear
# from the solved value. The loose tolerance guards the model rather than rounding.
_MAX_REGIME_MASS_DEVIATION = 1.0e-3


def regime_mass_is_a_distribution(
    *, probability_mass: FloatND, has_negative_probability: BoolND
) -> BoolND:
    """Return whether retained targets carry unit mass and no negative weight.

    Unit mass is accepted within the regime-mass tolerance, at every log level.
    Non-negativity arrives as a decision read from each weight's own bits: an
    arithmetic reduction could flush a negative subnormal into signed zero.
    NaN or infinite mass fails the unit-mass decision.

    Args:
        probability_mass: The retained targets' probabilities, summed.
        has_negative_probability: Whether any target carried the sign bit on a
            nonzero magnitude.

    Returns:
        Whether the retained targets carry a probability distribution.

    """
    is_unit = jnp.abs(probability_mass - 1.0) <= _MAX_REGIME_MASS_DEVIATION
    return is_unit & ~has_negative_probability


def probabilities_form_distribution(
    *, probabilities: Iterable[FloatND], dtype: DTypeLike
) -> BoolND:
    """Accumulate target probabilities and test their elementwise distribution.

    Inputs are broadcast in their supplied order. Each sign is read before
    accumulation, preserving negative subnormal weights. An empty iterable
    carries zero mass and returns false.

    Args:
        probabilities: Target weights with broadcast-compatible shapes.
        dtype: Floating dtype of the initial scalar mass accumulator.

    Returns:
        Elementwise distribution validity over the broadcast target weights.

    """
    probability_mass = jnp.asarray(0.0, dtype=dtype)
    has_negative_probability = jnp.zeros((), dtype=bool)
    for prob in probabilities:
        probability_mass = probability_mass + prob
        has_negative_probability = has_negative_probability | is_negative(prob)
    return regime_mass_is_a_distribution(
        probability_mass=probability_mass,
        has_negative_probability=has_negative_probability,
    )


def log_of_nonnegative(values: FloatND) -> FloatND:
    """Return `log(values)` without reading a subnormal as floating data.

    Some XLA backends preserve a subnormal's IEEE representation through a
    gather or bitcast but flush it when a transcendental primitive consumes it.
    A direct `log` therefore reports `-inf` for a strictly positive number.
    Decompose every finite positive value from its bit pattern instead: its
    significand is rebuilt as a normal number in `[1, 2)` and its unbiased
    base-two exponent is carried as an ordinary integer.  Only the normal
    significand is passed to `log`.

    Zero, infinities, NaNs, and negative inputs retain the ordinary logarithm
    conventions.  The custom tangent is the mathematical `dx / x`; the bit
    decomposition is only a representation device for the primal.

    Args:
        values: Nonnegative floating values.

    Returns:
        Elementwise natural logarithms.

    """
    return _log_of_nonnegative_with_tangent(jnp.asarray(values))


def _log_of_nonnegative_bits(values: FloatND) -> FloatND:
    """Evaluate the nonnegative logarithm from significand and exponent bits."""
    arr = jnp.asarray(values)
    represented_zero = is_represented_zero(arr)
    subnormal = is_below_smallest_normal(arr) & ~represented_zero
    significand = _significand_in_unit_range(arr)
    exponent = _unbiased_exponent(arr).astype(arr.dtype)
    subnormal_log = jnp.log(significand) + exponent * jnp.log(
        jnp.asarray(2.0, dtype=arr.dtype)
    )
    # Keep the ordinary backend implementation for every normal or special
    # value.  Besides preserving its rounding, replacing subnormal inputs by
    # one prevents a dead direct-log branch from ever consuming the value.
    direct_log = jnp.log(jnp.where(subnormal, jnp.ones_like(arr), arr))
    return jnp.where(subnormal, subnormal_log, direct_log)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _log_of_nonnegative_jvp(
    primals: tuple[FloatND], tangents: tuple[FloatND]
) -> tuple[FloatND, FloatND]:
    """Differentiate the representation-aware logarithm as `dx / x`."""
    (values,) = primals
    (values_dot,) = tangents
    return _log_of_nonnegative_with_tangent(values), values_dot / values


_log_of_nonnegative_with_tangent = jax.custom_jvp(_log_of_nonnegative_bits)
_log_of_nonnegative_with_tangent.defjvp(_log_of_nonnegative_jvp)


def is_represented_zero(values: FloatND) -> BoolND:
    """Whether each entry is `+0` or `-0`, read from its bits.

    This is the null event, and the only thing that is: `values == 0` also
    catches every subnormal on a backend that flushes them, which is how a
    node that can occur gets treated as one that cannot.
    """
    arr = jnp.asarray(values)
    int_dtype = _int_dtype(arr)
    # Keep the zero test opaque where floating comparisons flush subnormals.
    mask = jnp.asarray((1 << (8 * arr.dtype.itemsize - 1)) - 1, dtype=int_dtype)
    tiny = jnp.finfo(arr.dtype).tiny
    sign_cleared = jnp.where(
        (arr >= tiny) | (arr <= -tiny),
        mask,
        jax.lax.optimization_barrier(mask),
    )
    return (jax.lax.bitcast_convert_type(arr, int_dtype) & sign_cleared) == 0


def is_below_smallest_normal(values: FloatND) -> BoolND:
    """Whether each entry's magnitude is zero or subnormal, read from its bits.

    The two cases are one question for a producer of weights: a product
    arriving either way has lost the size it was meant to carry.
    """
    arr = jnp.asarray(values)
    return _magnitude_bits(arr) < _smallest_normal_bits(arr)


def is_negative(values: FloatND) -> BoolND:
    """Whether each entry is a nonzero magnitude carrying the sign bit.

    A negative probability is a misspecification rather than an unlikely
    event, so it must stay visible. `values < 0` misses a negative subnormal,
    which the backend reports as `-0`, and `-0` is a zero rather than a
    negative number: unit mass and non-negativity would both hold for a
    transition that places negative mass on a target.
    """
    arr = jnp.asarray(values)
    unsigned_dtype = jnp.uint32 if arr.dtype.itemsize == _FLOAT32_BYTES else jnp.uint64
    bits = jax.lax.bitcast_convert_type(arr, unsigned_dtype)
    signed_zero_bits = jnp.asarray(
        1 << (8 * arr.dtype.itemsize - 1), dtype=unsigned_dtype
    )
    return bits > signed_zero_bits


def is_live(values: FloatND) -> BoolND:
    """Whether each entry is a probability that can occur.

    Live means a nonzero magnitude with the sign bit clear and not NaN — the
    complement of "contributes nothing" and "is not a probability". A
    subnormal is live: the dtype cannot multiply it, which is a fact about
    the arithmetic rather than about the event.
    """
    arr = jnp.asarray(values)
    unsigned_dtype = jnp.uint32 if arr.dtype.itemsize == _FLOAT32_BYTES else jnp.uint64
    bits = jax.lax.bitcast_convert_type(arr, unsigned_dtype)
    infinity = jax.lax.bitcast_convert_type(
        jnp.asarray(jnp.inf, dtype=arr.dtype), unsigned_dtype
    )
    # Positive IEEE words are ordered from the least subnormal through infinity.
    return (bits > 0) & (bits <= infinity)


def rescaled_lottery_weights(*, weights: FloatND, axis: int | None = -1) -> FloatND:
    """Return the weights scaled so every live one is a normal number.

    A weighted mean depends on the weights only through their ratios, so
    multiplying them all by one constant leaves it unchanged; a power of two
    does that exactly, with no rounding anywhere. The common factor is the
    smallest one that lifts every live weight of the lottery out of the
    subnormal range, so every consumer downstream — `LinearExpectation`,
    `weighted_power_mean`, or a user-written `aggregate` this module has never
    seen — multiplies only operands the dtype can use.

    Every live weight becomes normal, not merely the smallest: the factor is
    the largest of the per-entry factors, and each entry receives at least the
    one it needed — up to what the largest weight has room for. A lottery whose
    weights differ by more binades than the exponent field can express cannot
    be held on one scale at all, and stopping at the largest weight's headroom
    leaves the smallest understated rather than replacing the largest with an
    infinity that would destroy every node beside it.

    Zero weights, NaNs and infinities come back untouched — the first is the
    null event and stays exactly null, and the other two are not probabilities
    and have to stay visible.

    Args:
        weights: Weights of one lottery.
        axis: The lottery's axis, over which the common factor is chosen.
            `None` treats every axis of `weights` as one lottery, which is what
            a target whose nodes span several stochastic dimensions needs.

    Returns:
        The weights, multiplied by a common power of two chosen per lottery.

    """
    arr = jnp.asarray(weights)
    needed = jnp.max(_shift_to_normal(arr), axis=axis, keepdims=True)
    room = jnp.min(
        jnp.where(
            is_live(arr) & jnp.isfinite(arr),
            (_exponent_bias(arr) - 1) - _unbiased_exponent(arr),
            needed,
        ),
        axis=axis,
        keepdims=True,
    )
    return scaled_by_power_of_two(values=arr, shift=jnp.minimum(needed, room))


def normalized_scaled_weights(
    *, coefficients: FloatND, shifts: _BitsND
) -> tuple[FloatND, _BitsND]:
    """Return the same scaled measure with unit mass, still carrying its scales.

    Each entry stands for `coefficients * 2**-shifts`, which is how a joint
    product too small for the normal range travels. Dividing such entries by
    their mass as plain numbers is what a lottery cannot survive: the mass is
    the size of the largest entry, and the ratio of the smallest to it is
    exactly the quantity the format cannot hold — so the rarest node reaches
    the consumer as a represented zero, and the scale it still carries has
    nothing left to restore.

    The division is therefore taken between each entry's significand and the
    mass measured on that entry's own scale, and the exponents go where they
    can be held exactly. What every entry satisfies is

    ```{math}
    c_i^{out} \\, 2^{-s_i^{out}} = \\frac{c_i^{in} \\, 2^{-s_i^{in}}}
                                        {\\sum_j c_j^{in} \\, 2^{-s_j^{in}}},
    ```

    with every returned coefficient within a factor of the number of entries of
    one, so no spread of probabilities can push one out of the normal range.
    On a lottery the format can already hold, the coefficient is the same
    quotient the plain division returns, times the exact power of two its scale
    names.

    The mass is read on the scale of the largest entry, so an entry too far
    below it to register there contributes nothing to it — which is what a
    probability that many orders of magnitude down does contribute to a total
    of order one. Its own share is unaffected: it keeps its scale.

    A mass that is not strictly positive leaves the entries as they are rather
    than dividing. There is no distribution to normalize to, and every consumer
    of this reads the mass itself to decide what to do about that.

    Args:
        coefficients: The scaled weights' significands.
        shifts: Each one's own base-two scale.

    Returns:
        Tuple of the normalized coefficients and the scale each one carries.

    """
    arr = jnp.asarray(coefficients)
    shifts = jnp.asarray(shifts)
    own_exponent = _unbiased_exponent(arr)
    exponent = own_exponent - shifts
    live = is_live(arr) & jnp.isfinite(arr)
    # Both reductions run along the lottery's own axis: a batch holds unrelated
    # lotteries, and one row's likeliest node says nothing about what another
    # row's mass should be measured against.
    floor = jnp.min(exponent, axis=-1, keepdims=True)
    peak = jnp.max(jnp.where(live, exponent, floor), axis=-1, keepdims=True)
    mass = jnp.sum(
        jnp.ldexp(arr, -(shifts + peak).astype(jnp.int32)), axis=-1, keepdims=True
    )
    safe_mass = jnp.where(mass > 0.0, mass, jnp.ones_like(mass))
    carries_a_scale = ~is_represented_zero(arr) & jnp.isfinite(arr)
    return (
        jnp.ldexp(arr, -own_exponent.astype(jnp.int32)) / safe_mass,
        jnp.where(carries_a_scale, peak - exponent, jnp.zeros_like(shifts)).astype(
            jnp.int32
        ),
    )


def rescaled_weight_pair(
    *, first_weight: FloatND, second_weight: FloatND
) -> tuple[FloatND, FloatND]:
    """Return a two-node lottery's weights scaled by one common power of two.

    `rescaled_lottery_weights` for a pair carried as two arrays rather than as
    a trailing axis of length two.
    """
    first = jnp.asarray(first_weight)
    second = jnp.asarray(second_weight)
    shift = jnp.maximum(_shift_to_normal(first), _shift_to_normal(second))
    return (
        scaled_by_power_of_two(values=first, shift=shift),
        scaled_by_power_of_two(values=second, shift=shift),
    )


def scaled_exact_product(factors: FloatND) -> tuple[FloatND, _BitsND]:
    """Return the product over the leading axis, scaled to stay a normal number.

    The product is `result * 2**-shift`, and `result` is normal wherever that
    product is nonzero. A joint node's probability routinely lands below the
    dtype's normal range while every factor sits comfortably inside it — two
    probabilities of `2**-64` in single precision make one of `2**-128` — and
    that value cannot be carried as a plain float here. It is representable,
    but only as a subnormal, and a subnormal that exists solely as an
    intermediate is not reliable: XLA:CPU returns the right bits when the value
    is a fused region's output and zero when it is consumed inside one, so even
    a bit-level test of it answers differently depending on what the caller
    does with the result. The scale therefore travels beside the number instead
    of inside it.

    A weighted mean is invariant to a common factor on its weights, so the
    caller can ignore `shift` whenever it normalizes by the mass of exactly the
    weights it was given. Where weights from several calls are concatenated,
    the caller has to bring them onto one scale first.

    A zero factor is the genuine null event and a non-finite one is not a
    probability; both take the ordinary product, which says the right thing
    about them already.

    Args:
        factors: The factors stacked along the leading axis. Trailing axes
            broadcast.

    Returns:
        Tuple of the scaled product and the base-two scale it carries.

    """
    arr = jnp.asarray(factors)
    parts = _product_parts(arr)
    # One shift for the whole array, so every entry keeps its size relative to
    # every other, and large enough that the smallest of them clears the
    # subnormal range — but no larger than the biggest of them has room for.
    # An entry whose factors include a zero or a non-finite takes the ordinary
    # product, and every value that can be is scale-invariant, so it neither
    # needs the shift nor constrains it.
    bias = _exponent_bias(arr)
    zero = jnp.zeros_like(parts.exponent)
    smallest = jnp.min(jnp.where(parts.ordinary, zero, parts.exponent))
    largest = jnp.max(jnp.where(parts.ordinary, zero + (1 - bias), parts.exponent))
    shift = jnp.maximum(
        jnp.minimum((1 - bias) - smallest, bias - largest), jnp.zeros((), jnp.int32)
    ).astype(jnp.int32)
    scaled = _encoded(
        significand=parts.significand,
        exponent=parts.exponent + shift,
        negative=parts.negative,
    )
    return jnp.where(parts.ordinary, parts.plain, scaled), shift


def exact_product(factors: FloatND) -> FloatND:
    """Return the product over the leading axis, encoded rather than multiplied.

    A joint node's probability is the product of one factor per stochastic
    axis, and that product routinely lands below the dtype's normal range while
    every factor sits comfortably inside it — two probabilities of `2**-64` in
    single precision make one of `2**-128`. That number *is* representable; the
    multiplication is what cannot produce it, because a backend that flushes
    subnormal results delivers zero and the event becomes indistinguishable
    from one that cannot occur.

    The product is therefore assembled instead of computed: significands
    multiply, exponents add, and the result is written into its bit pattern,
    subnormal or not. Nothing is lost that the format can hold, and what the
    format cannot hold — a product below the smallest subnormal — comes back as
    zero.

    A zero factor is the genuine null event and a non-finite one is not a
    probability; both take the ordinary product, which says the right thing
    about them already.

    Args:
        factors: The factors stacked along the leading axis. Trailing axes
            broadcast.

    Returns:
        Their product over the leading axis.

    """
    parts = _product_parts(jnp.asarray(factors))
    return jnp.where(
        parts.ordinary,
        parts.plain,
        _encoded(
            significand=parts.significand,
            exponent=parts.exponent,
            negative=parts.negative,
        ),
    )


def balanced_product(*, weight: FloatND, value: FloatND) -> FloatND:
    """Return `weight * value` with the exponent moved onto the smaller operand.

    A weight below the normal range is flushed as an operand, so a node of
    probability `2**-128` against a value of `2**126` prices at zero instead of
    at a quarter. The product does not depend on how the exponent is shared
    between the two, though, so it is moved: the weight is scaled up by as many
    binades as the value can give away and still stay normal itself. Both
    operands then multiply as ordinary numbers and the product is the one the
    exact arithmetic gives.

    Where the value cannot give away enough — because it is small too — the
    product is genuinely below what the format can hold, and no arrangement of
    the operands recovers it.

    Args:
        weight: Probability or quadrature weight, possibly subnormal.
        value: The finite value being weighted.

    Returns:
        Their product.

    """
    return _balanced_with_tangent(jnp.asarray(weight), jnp.asarray(value))


# keyword-only-exempt: library-callback=jax.custom_jvp
def _balanced_bits(weight: FloatND, value: FloatND) -> FloatND:
    """Multiply with the exponent moved onto the weight, without a derivative.

    The weight travels through the general scaling, because moving it up is
    exactly the case where it is subnormal. The value only ever moves *down*,
    by no more binades than it can give away and stay normal, so its exponent
    field is decremented directly rather than through the general routine — a
    handful of integer operations instead of the branch structure a subnormal
    operand would need. This primitive runs at every weighted term, so the
    difference is visible in compile time rather than only in the graph.
    """
    weight_arr = jnp.asarray(weight)
    value_arr = jnp.asarray(value)
    int_dtype = _int_dtype(value_arr)
    give = jnp.maximum(
        _unbiased_exponent(value_arr) - (1 - _exponent_bias(value_arr)),
        jnp.zeros((), jnp.int32),
    )
    shift = jnp.minimum(_shift_to_normal(weight_arr), give)
    movable = jnp.isfinite(value_arr) & ~is_represented_zero(value_arr)
    shift = jnp.where(movable, shift, jnp.zeros_like(shift))
    value_bits = jax.lax.bitcast_convert_type(value_arr, int_dtype)
    lowered = value_bits - (shift.astype(int_dtype) << _mantissa_bits(value_arr))
    lowered_value = jax.lax.bitcast_convert_type(
        jnp.where(movable, lowered, value_bits), value_arr.dtype
    )
    return _scaled_bits(values=weight_arr, shift=shift) * lowered_value


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _balanced_jvp(
    primals: tuple[FloatND, FloatND], tangents: tuple[FloatND, FloatND]
) -> tuple[FloatND, FloatND]:
    """Differentiate the product against each operand as the other's slope.

    The slopes are the primals, which differentiation treats as constants, so
    the tangent is an ordinary multiplication — reverse mode has to transpose
    it, and neither a bitcast nor a `custom_jvp` call has a transpose. The
    balanced form is therefore the primal's alone. Where the weight is below
    the normal range the derivative with respect to the value flushes to zero;
    that derivative is the weight itself, so what is lost is a slope already
    smaller than the format can hold.
    """
    weight, value = primals
    weight_dot, value_dot = tangents
    return (
        _balanced_with_tangent(weight, value),
        weight_dot * value + value_dot * weight,
    )


# One differentiable boundary for the whole product, rather than one per
# operand: the bit manipulation inside carries no derivative and does not need
# to, and a wrapper per operand costs a second trace of the same machinery.
_balanced_with_tangent = jax.custom_jvp(_balanced_bits)
_balanced_with_tangent.defjvp(_balanced_jvp)


@dataclasses.dataclass(frozen=True)
class _ProductParts:
    """A product over the leading axis, held apart as significand and exponent."""

    significand: FloatND
    """The product of the factors' significands, reduced to `[1, 2)`."""
    exponent: _BitsND
    """Where that significand belongs, as an unbiased power of two."""
    negative: BoolND
    """Whether an odd number of factors carried a sign."""
    ordinary: BoolND
    """Whether some factor is a zero or is not finite, so `plain` is the answer."""
    plain: FloatND
    """The multiplied product, correct wherever `ordinary` holds."""


def _product_parts(values: FloatND) -> _ProductParts:
    """Return the product over the leading axis, before it is written to bits."""
    arr = jnp.asarray(values)
    raw_significand = jnp.prod(_significand_in_unit_range(arr), axis=0)
    # Each factor's significand lies in `[1, 2)`, so their product can carry a
    # binade of its own. It has to move to the exponent rather than be counted
    # in both places, which is what `_encoded` assumes of what it is handed.
    return _ProductParts(
        significand=_significand_in_unit_range(raw_significand),
        exponent=jnp.sum(_unbiased_exponent(arr), axis=0)
        + _unbiased_exponent(raw_significand),
        negative=jnp.sum(jnp.where(is_negative(arr), 1, 0), axis=0) % 2 == 1,
        ordinary=jnp.any(is_represented_zero(arr) | ~jnp.isfinite(arr), axis=0),
        plain=jnp.prod(arr, axis=0),
    )


def _encoded(*, significand: FloatND, exponent: _BitsND, negative: BoolND) -> FloatND:
    """Return `significand * 2**exponent` written straight into the bit pattern.

    `significand` is a positive number the caller has already reduced to
    `[1, 2)`; `exponent` is where it belongs. A result inside the normal range
    is the significand's own bits with the exponent field displaced. Below that
    range the implicit bit is restored and the whole significand shifted right,
    which is the subnormal encoding; shifted past the stored significand it is
    zero, the one case the format genuinely cannot hold.
    """
    arr = jnp.asarray(significand)
    int_dtype = _int_dtype(arr)
    mantissa_bits = _mantissa_bits(arr)
    bias = _exponent_bias(arr)
    magnitude = _magnitude_bits(arr)
    exponent = exponent.astype(int_dtype)

    normal_bits = magnitude + (exponent << mantissa_bits)
    full_significand = (magnitude & ((1 << mantissa_bits) - 1)) | (1 << mantissa_bits)
    binades_below = (1 - bias) - exponent
    subnormal_bits = jnp.where(
        binades_below <= mantissa_bits,
        full_significand >> jnp.clip(binades_below, 0, mantissa_bits),
        jnp.zeros_like(full_significand),
    )
    infinite_bits = jnp.asarray(
        ((1 << (8 * arr.dtype.itemsize - mantissa_bits - 1)) - 1) << mantissa_bits,
        dtype=int_dtype,
    )
    bits = jnp.where(exponent >= 1 - bias, normal_bits, subnormal_bits)
    bits = jnp.where(exponent > bias, infinite_bits, bits)
    # The sign bit set, and every other bit clear, is the signed minimum.
    sign_bit = jnp.asarray(jnp.iinfo(int_dtype).min, dtype=int_dtype)
    return jax.lax.bitcast_convert_type(
        jnp.where(negative, bits | sign_bit, bits), arr.dtype
    )


def _significand_in_unit_range(values: FloatND) -> FloatND:
    """Return each entry's significand as a positive number in `[1, 2)`."""
    arr = jnp.asarray(values)
    int_dtype = _int_dtype(arr)
    mantissa_bits = _mantissa_bits(arr)
    mantissa_mask = jnp.asarray((1 << mantissa_bits) - 1, dtype=int_dtype)
    magnitude = _magnitude_bits(arr)
    normalized = jnp.where(
        is_below_smallest_normal(arr),
        (magnitude << _shift_to_normal(arr)) & mantissa_mask,
        magnitude & mantissa_mask,
    )
    unit_exponent = jnp.asarray(_exponent_bias(arr) << mantissa_bits, dtype=int_dtype)
    return jax.lax.bitcast_convert_type(normalized | unit_exponent, arr.dtype)


def scaled_down_by_power_of_two(*, values: FloatND, shift: _BitsND) -> FloatND:
    """Return `values * 2**shift` for a non-positive integer `shift`.

    This is the downward-only counterpart of `scaled_by_power_of_two`.
    `jnp.ldexp` is deliberately general: it decomposes every value with
    `frexp` and rebuilds it through floating-point powers. In the target-node
    expectation the shift is known to be non-positive, and that general graph
    runs twice over the largest continuation surface.

    Scaling down needs only integer work on the IEEE representation. A normal
    result is obtained by subtracting from the exponent field. If the result
    crosses into the subnormal range, the explicit significand is shifted right
    with round-to-nearest, ties-to-even. Zeros, infinities and NaNs are returned
    unchanged. The result therefore agrees bit-for-bit with an IEEE `ldexp`
    for every non-positive shift, including the normal/subnormal boundary,
    without materialising a subnormal scaling operand.

    The map is linear in `values` away from the usual rounding boundaries, so
    its custom derivative multiplies by the same exact power of two. The tangent
    remains an ordinary multiplication and can be transposed by reverse mode.

    Args:
        values: Values to scale.
        shift: Integer powers of two. Every entry must be non-positive.

    Returns:
        `values * 2**shift` with IEEE round-to-nearest-even semantics.

    """
    return _scaled_down_with_tangent(jnp.asarray(values), shift)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _scaled_down_bits(values: FloatND, shift: _BitsND) -> FloatND:
    """Scale downward by editing the exponent/significand fields directly."""
    arr = jnp.asarray(values)
    int_dtype = _int_dtype(arr)
    mantissa_bits = _mantissa_bits(arr)
    shift_arr = jnp.asarray(shift, dtype=jnp.int32)

    bits = jax.lax.bitcast_convert_type(arr, int_dtype)
    magnitude = _magnitude_bits(arr)
    sign = bits - magnitude
    exponent = magnitude >> mantissa_bits
    mantissa_mask = jnp.asarray((1 << mantissa_bits) - 1, dtype=int_dtype)
    mantissa = magnitude & mantissa_mask

    target_exponent = exponent.astype(jnp.int32) + shift_arr
    stays_normal = target_exponent > 0
    normal_magnitude = magnitude + (shift_arr.astype(int_dtype) << mantissa_bits)

    # Below the normal range, restore a normal number's implicit leading bit and
    # shift the full significand. A subnormal already carries every bit
    # explicitly, so its shift starts from the stored mantissa.
    implicit_bit = jnp.asarray(1 << mantissa_bits, dtype=int_dtype)
    significand = jnp.where(exponent == 0, mantissa, mantissa | implicit_bit)
    distance = jnp.where(exponent == 0, -shift_arr, 1 - target_exponent)

    # Dynamic shifts larger than the significand width are clamped for the
    # integer operations below and are set to zero afterwards. At exactly one
    # half unit, ties-to-even depends on the retained least-significant bit.
    safe_distance = jnp.clip(distance, 1, mantissa_bits + 1).astype(int_dtype)
    quotient = significand >> safe_distance
    one = jnp.ones_like(significand)
    discarded_mask = (one << safe_distance) - one
    discarded = significand & discarded_mask
    halfway = one << (safe_distance - one)
    round_up = (discarded > halfway) | (
        (discarded == halfway) & ((quotient & one) != 0)
    )
    rounded = quotient + round_up.astype(int_dtype)
    rounded = jnp.where(distance > mantissa_bits + 1, jnp.zeros_like(rounded), rounded)
    subnormal_magnitude = jnp.where(distance == 0, magnitude, rounded)

    output_magnitude = jnp.where(stays_normal, normal_magnitude, subnormal_magnitude)
    exponent_all_ones = (1 << (8 * arr.dtype.itemsize - mantissa_bits - 1)) - 1
    keep_unchanged = is_represented_zero(arr) | (exponent == exponent_all_ones)
    output_bits = jnp.where(keep_unchanged, bits, sign | output_magnitude)
    return jax.lax.bitcast_convert_type(output_bits, arr.dtype)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _scaled_down_jvp(
    primals: tuple[FloatND, _BitsND], tangents: tuple[FloatND, _IntegerTangent]
) -> tuple[FloatND, FloatND]:
    """Differentiate downward scaling as multiplication by `2**shift`."""
    values, shift = primals
    values_dot, _ = tangents
    slope = _scaled_down_bits(values=jnp.ones_like(values), shift=shift)
    return (
        _scaled_down_with_tangent(values, shift),
        values_dot * slope,
    )


_scaled_down_with_tangent = jax.custom_jvp(_scaled_down_bits)
_scaled_down_with_tangent.defjvp(_scaled_down_jvp)


def scaled_by_power_of_two(*, values: FloatND, shift: _BitsND) -> FloatND:
    """Return `values * 2**shift`, exactly, for an integer `shift`.

    Written entirely on the bit pattern, with no floating-point arithmetic
    anywhere. Multiplying is not available: a subnormal is flushed as an
    operand, and carrying the factor as two normal ones does not survive
    either, because the compiler is free to reassociate them back into the
    single subnormal constant it started from.

    Bit manipulation carries no derivative of its own, so the one this stands
    for is supplied: the map is linear in `values` with slope `2**shift`, which
    is the same scaling applied to the tangent. Without it every consumer that
    rescales a weight — the interpolation read, the certainty equivalent, the
    power mean — would report a slope of exactly zero.

    A normal number scales by adding to its exponent field. A subnormal has no
    exponent field to add to, so it is re-encoded: its significand's leading
    set bit becomes the implicit bit, the rest becomes the mantissa field, and
    what is left of the shift becomes the exponent. `shift` is the largest the
    lottery needed, so it is at least what each entry needs and the result is
    always normal.

    Zeros, infinities and NaNs come back untouched — none of them has a scale
    to change, and adding to an all-ones exponent field would manufacture one.
    """
    return _scaled_with_tangent(jnp.asarray(values), shift)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _scaled_bits(values: FloatND, shift: _BitsND) -> FloatND:
    """Write `values * 2**shift` straight into the bit pattern."""
    arr = jnp.asarray(values)
    int_dtype = _int_dtype(arr)
    mantissa_bits = _mantissa_bits(arr)
    bits = jax.lax.bitcast_convert_type(arr, int_dtype)
    magnitude = _magnitude_bits(arr)
    sign = bits - magnitude

    shift = shift.astype(int_dtype)
    own_shift = _shift_to_normal(arr)
    # A normal number scales by adding to its exponent field. A subnormal scales
    # by shifting its whole magnitude, which stays correct right up to the normal
    # range: the significand's leading bit carries into the exponent field and
    # becomes the implicit bit, which is what the encoding is built to do. Beyond
    # that point the number is normal and the rest of the shift is an exponent
    # again.
    normal_bits = magnitude + (shift << mantissa_bits)
    to_normal = jnp.clip(jnp.minimum(shift, own_shift), 0, mantissa_bits + 1)
    beyond_normal = jnp.maximum(shift - own_shift, jnp.zeros_like(shift))
    subnormal_bits = (magnitude << to_normal) + (beyond_normal << mantissa_bits)
    is_subnormal = is_below_smallest_normal(arr) & ~is_represented_zero(arr)
    scaled = jnp.where(is_subnormal, subnormal_bits, normal_bits)
    keep = is_represented_zero(arr) | ~jnp.isfinite(arr)
    return jax.lax.bitcast_convert_type(jnp.where(keep, bits, sign | scaled), arr.dtype)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _scaled_bits_jvp(
    primals: tuple[FloatND, _BitsND], tangents: tuple[FloatND, _IntegerTangent]
) -> tuple[FloatND, FloatND]:
    """Scale the tangent by the same power of two the value is scaled by.

    Both spellings compute the same linear map, bit for bit. The tangent is an
    ordinary multiplication by the slope because that spelling is the one JAX
    can transpose: `bitcast_convert_type` carries no transpose rule, so a
    tangent written as the bit-level scaling would leave the map forward-mode
    only. The slope is exact wherever it is representable; past that it is the
    infinity the true slope has overflowed to, which is the honest answer for a
    map that has lifted a subnormal across the whole range.
    """
    values, shift = primals
    values_dot, _ = tangents
    arr = jnp.asarray(values)
    slope = jnp.where(
        shift <= _exponent_bias(arr),
        _scaled_bits(values=jnp.ones_like(arr), shift=shift),
        jnp.asarray(jnp.inf, dtype=arr.dtype),
    )
    return _scaled_with_tangent(values, shift), values_dot * slope


# Built by call rather than by decorator: `@jax.custom_jvp` produces a callable
# instance, and the package's beartype claw rebinds one of those to its own
# `__call__`, which loses `defjvp` along with everything else the object knows.
_scaled_with_tangent = jax.custom_jvp(_scaled_bits)
_scaled_with_tangent.defjvp(_scaled_bits_jvp)


def binades_above_smallest_normal(values: FloatND) -> _BitsND:
    """How far each entry can be halved before it leaves the normal range.

    A caller with a power of two to apply to a product decides how much of it
    the weight can absorb before it becomes a subnormal the backend may flush,
    and how much has to be applied to the product instead. An entry already
    below the normal range has no room at all.

    Args:
        values: Entries whose headroom is wanted.

    Returns:
        The number of binades each entry has above the smallest normal.

    """
    arr = jnp.asarray(values)
    return jnp.where(
        is_below_smallest_normal(arr),
        jnp.zeros((), jnp.int32),
        (_unbiased_exponent(arr) - (1 - _exponent_bias(arr))).astype(jnp.int32),
    )


def _unbiased_exponent(values: FloatND) -> _BitsND:
    """Return each entry's base-two exponent, subnormals included."""
    arr = jnp.asarray(values)
    magnitude = _magnitude_bits(arr)
    mantissa_bits = _mantissa_bits(arr)
    bias = _exponent_bias(arr)
    normal = (magnitude >> mantissa_bits) - bias
    int_bits = 8 * arr.dtype.itemsize
    subnormal = (int_bits - 1 - jax.lax.clz(magnitude)) + 1 - bias - mantissa_bits
    return jnp.where(is_below_smallest_normal(arr), subnormal, normal).astype(jnp.int32)


def _shift_to_normal(values: FloatND) -> _BitsND:
    """Return the smallest power of two lifting each entry into the normal range.

    Zero for anything already normal, zero, or not finite. For a subnormal it
    is the distance from the significand's leading set bit to where the normal
    range starts, which the leading-zero count gives directly.
    """
    arr = jnp.asarray(values)
    int_bits = 8 * arr.dtype.itemsize
    shift = jax.lax.clz(_magnitude_bits(arr)) - (int_bits - 1 - _mantissa_bits(arr))
    needs_shift = is_below_smallest_normal(arr) & ~is_represented_zero(arr)
    return jnp.where(needs_shift, shift, jnp.zeros_like(shift))


def _magnitude_bits(values: FloatND) -> _BitsND:
    """Return each entry's bit pattern with the sign bit cleared."""
    arr = jnp.asarray(values)
    int_dtype = _int_dtype(arr)
    return jax.lax.bitcast_convert_type(arr, int_dtype) & jnp.asarray(
        (1 << (8 * arr.dtype.itemsize - 1)) - 1, dtype=int_dtype
    )


def _smallest_normal_bits(values: FloatND) -> _BitsND:
    """Return the bit pattern of the dtype's smallest normal magnitude."""
    arr = jnp.asarray(values)
    return jax.lax.bitcast_convert_type(
        jnp.asarray(jnp.finfo(arr.dtype).tiny, dtype=arr.dtype), _int_dtype(arr)
    )


def _int_dtype(values: FloatND) -> DTypeLike:
    """Return the integer dtype of the same width as `values`."""
    if jnp.asarray(values).dtype.itemsize == _FLOAT32_BYTES:
        return jnp.int32
    return jnp.int64


def _mantissa_bits(values: FloatND) -> int:
    """Return the number of stored significand bits of the dtype."""
    if jnp.asarray(values).dtype.itemsize == _FLOAT32_BYTES:
        return _FLOAT32_MANTISSA_BITS
    return _FLOAT64_MANTISSA_BITS


def _exponent_bias(values: FloatND) -> int:
    """Return the dtype's exponent bias."""
    return 1 - int(jnp.finfo(jnp.asarray(values).dtype).minexp)
