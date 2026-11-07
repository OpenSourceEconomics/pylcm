"""Weighted arithmetic that treats an exactly-zero weight as a null event.

`-inf` is the ordinary value of a state at which every action is infeasible,
and an exactly-zero weight is equally ordinary — a `StochasticTransition` row with
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

import jax
import jax.numpy as jnp

from _lcm.probability import (
    _significand_in_unit_range,
    _unbiased_exponent,
    balanced_product,
    binades_above_smallest_normal,
    exact_product,
    is_below_smallest_normal,
    is_live,
    is_negative,
    is_represented_zero,
    nonzero_exact_product,
    scaled_by_power_of_two,
    scaled_down_by_power_of_two,
    scaled_exact_product,
)
from lcm.typing import BoolND, FloatND, IntND


def factored_weighted_term(*, factors: FloatND, values: FloatND) -> FloatND:
    """Form each literal weighted term before rounding its probability product.

    Leading-axis factors retain their represented support. Including the finite
    value in the exponent arithmetic lets a large value rescue a contribution
    whose probability alone is below the format. No mass normalization occurs.
    The product rule keeps finite ``dweight * value`` at zero primal factors.

    The value-channel derivative is carried in the active dtype. A probability
    product below its cotangent range can underflow before an upstream large
    payoff multiplier, even when the primal term is rescued. This does not
    qualify extreme-scale automatic sensitivities for certified accuracy; an
    economic marginal must instead be weighted as its explicit value channel.
    """
    return _factored_with_tangent(factors, values)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _factored_primal(factors: FloatND, values: FloatND) -> FloatND:
    support = jnp.all(~is_represented_zero(factors), axis=0)
    safe_values = jnp.where(support | jnp.isfinite(values), values, 0.0)
    product = exact_product(jnp.concatenate((factors, safe_values[None]), axis=0))
    negative = jnp.sum(is_negative(factors), axis=0) % 2 == 1
    nonfinite = jnp.where(negative, -values, values)
    nonfinite = jnp.where(jnp.any(jnp.isnan(factors), axis=0), jnp.nan, nonfinite)
    return jnp.where(support & ~jnp.isfinite(values), nonfinite, product)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _factored_jvp(
    primals: tuple[FloatND, FloatND], tangents: tuple[FloatND, FloatND]
) -> tuple[FloatND, FloatND]:
    factors, values = primals
    factors_dot, values_dot = tangents
    # Slopes depend only on primals. Ordinary tangent multiplication is linear
    # and transposable; applying bit operations to tangents would break reverse AD.
    # shortcut: active-dtype value cotangents; extending their exponent range
    # would require a different AD representation, beyond this weighted kernel.
    tangent = _factored_with_tangent(factors, jnp.ones_like(values)) * values_dot
    for column in range(factors.shape[0]):
        others = jnp.concatenate((factors[:column], factors[column + 1 :]), axis=0)
        slope = _factored_with_tangent(others, values)
        slope = jnp.where(
            is_represented_zero(factors[column]) & ~jnp.isfinite(values), 0.0, slope
        )
        tangent = tangent + slope * factors_dot[column]
    return _factored_with_tangent(factors, values), tangent


_factored_with_tangent = jax.custom_jvp(_factored_primal)
_factored_with_tangent.defjvp(_factored_jvp)


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
    return balanced_product(weight=effective_weight, value=safe_value)


#: Longest permutable axis whose reduction order cannot matter: a single entry
#: has no order, and two entries are one commutative addition.
_MAX_ORDER_FREE_AXIS_LENGTH = 2


def sum_in_value_order(*, values: FloatND, axis: int = 0) -> FloatND:
    """Sum `values` after canonicalising the floating-point reduction order.

    Floating-point addition is not associative.  When the entries are keyed by
    economically inert identifiers (regime labels, stakeholder names), reducing in
    declaration order can therefore make a value — and even a strict argmax — depend
    on those identifiers.  Sorting the NUMERIC contributions before the sum makes the
    reduction order a function of the contribution multiset alone.

    This is not a correctly-rounded summation algorithm; it is the shared invariant
    used where identifier permutations must not select a different reduction tree.

    Associativity is what fails; commutativity does not. An axis of one entry has
    no order to canonicalise, and an axis of two is a single addition, which
    IEEE-754 evaluates to the same bits in either operand order — signed zeros,
    infinities and NaN included. So the sort provably changes nothing there and is
    skipped, which is the ordinary path: a model whose regimes reach one or two
    targets pays for the canonicalisation only where it can matter.

    Args:
        values: Contributions to sum.
        axis: Axis containing the permutable contributions.

    Returns:
        The value-ordered sum over `axis`.
    """
    arr = jnp.asarray(values)
    if arr.shape[axis] <= _MAX_ORDER_FREE_AXIS_LENGTH:
        return jnp.sum(arr, axis=axis)
    return jnp.sum(jnp.sort(arr, axis=axis), axis=axis)


def center_lottery_shifts(*, coefficients: FloatND, shifts: IntND) -> IntND:
    """Remove a common weight gauge before coefficient exponent arithmetic.

    Only live finite coefficients select the gauge. Relative shift spans and
    subsequent exponent arithmetic must fit int32; private arbitrary endpoint
    spans are not an admitted scaled-measure representation.
    """
    coefficients = jnp.asarray(coefficients)
    shifts = jnp.broadcast_to(jnp.asarray(shifts), coefficients.shape)
    live = is_live(coefficients) & jnp.isfinite(coefficients)
    fallback = jnp.max(shifts, axis=-1, keepdims=True)
    reference = jnp.min(jnp.where(live, shifts, fallback), axis=-1, keepdims=True)
    return shifts - reference


def scaled_lottery_mean(
    *, values: FloatND, coefficients: FloatND, shifts: IntND
) -> tuple[FloatND, FloatND]:
    """Return a scaled lottery's mean over the last axis, and its scaled mass.

    Every finite operand is held as a significand in `[1, 2)` and an integer
    exponent. A node's weighted value is the product of two significands,
    which is always a normal number, and its exponent is the sum
    `coefficient - shift + value`. The numerator is placed so its largest
    term sits in `[1, 4)`, and the mass so its likeliest node sits in
    `[1, 2)`. Each term is lowered from there by its own distance to the
    largest one. When no live finite nonzero value contributes, the numerator
    uses the mass's anchor: a zero numerator then has zero relative exponent,
    while live infinities and NaNs retain their original products.

    A term that leaves the format on the way down is therefore smaller than
    the largest term by the whole normal range. Without cancellation it
    cannot be material to the numerator, whatever the scale of the values,
    the weights, or the mean.

    shortcut: cancellation can expose a small term that underflows on the
    shared scale. Use a wider accumulator if the application needs that case.

    Neither sum can overflow, since every term is
    below four. The two exponents meet once, in the final ratio. Their
    difference is at most the largest finite value exponent when such a value
    contributes, and zero otherwise, so the upward part is a normal power of
    two. If rounding makes reconstruction overflow,
    return the corresponding endpoint of the live support. A finite-support
    mean lies between those endpoints; genuine live infinities remain infinite.
    Finite estimates and NaNs pass through unchanged.

    A null event contributes nothing, even against an infinite value. A live
    node with a nonfinite value carries that value into the mean. A
    coefficient that is negative, infinite, or NaN poisons its whole row.
    Relative shift spans and the derived exponent sums must fit int32.

    Args:
        values: Continuation values of the lottery along the last axis.
        coefficients: The weights' significands, over the same axis.
        shifts: Each weight's own base-two scale, broadcast against them.

    Returns:
        Tuple of the mean and the mass on its own normalized scale. The mass
        is zero only when the row has no live node.

    """
    coefficients = jnp.asarray(coefficients)
    values = jnp.asarray(values)
    shifts = center_lottery_shifts(coefficients=coefficients, shifts=shifts)

    live = is_live(coefficients) & jnp.isfinite(coefficients)
    invalid = is_negative(coefficients) | (
        ~jnp.isfinite(coefficients) & ~is_represented_zero(coefficients)
    )
    weight_exponent = jnp.where(live, _unbiased_exponent(coefficients) - shifts, 0)
    unit_coefficients = jnp.where(
        live, _significand_in_unit_range(coefficients), jnp.zeros_like(coefficients)
    )

    finite_value = jnp.isfinite(values) & ~is_represented_zero(values)
    unit_values = jnp.where(
        finite_value,
        jnp.copysign(_significand_in_unit_range(values), values),
        values,
    )
    contributes = live & finite_value
    term_exponent = weight_exponent + jnp.where(
        finite_value, _unbiased_exponent(values), 0
    )
    mass_exponent = _largest_selected(selected=live, exponents=weight_exponent)
    numerator_exponent = _largest_selected(
        selected=contributes, exponents=term_exponent, fallback=mass_exponent
    )
    products = zero_safe_weighted_term(
        weight=unit_coefficients, value=unit_values, subnormal_is_accounted_for=True
    )
    value_terms = scaled_down_by_power_of_two(
        values=products,
        shift=jnp.where(contributes, term_exponent - numerator_exponent, 0),
    )

    weight_terms = scaled_down_by_power_of_two(
        values=unit_coefficients,
        shift=jnp.where(live, weight_exponent - mass_exponent, 0),
    )

    mass = jnp.sum(weight_terms, axis=-1)
    ratio = jnp.sum(value_terms, axis=-1) / mass
    exponent = (numerator_exponent - mass_exponent)[..., 0]
    upward = scaled_by_power_of_two(
        values=jnp.ones_like(ratio), shift=jnp.maximum(exponent, 0)
    )
    mean = scaled_down_by_power_of_two(
        values=ratio * upward, shift=jnp.minimum(exponent, 0)
    )
    # A normalized nonnegative mean cannot exceed its live support.
    lower = jnp.min(jnp.where(live, values, jnp.inf), axis=-1)
    upper = jnp.max(jnp.where(live, values, -jnp.inf), axis=-1)
    mean = jnp.where(jnp.isinf(mean), jnp.where(jnp.signbit(mean), lower, upper), mean)
    nan = jnp.asarray(jnp.nan, dtype=coefficients.dtype)
    invalid_row = jnp.any(invalid, axis=-1)
    return jnp.where(invalid_row, nan, mean), jnp.where(invalid_row, nan, mass)


def _largest_selected(
    *, selected: BoolND, exponents: IntND, fallback: IntND | None = None
) -> IntND:
    """Return each row's largest selected exponent, using the fallback only if empty."""
    absent = jnp.iinfo(exponents.dtype).min
    largest = jnp.max(jnp.where(selected, exponents, absent), axis=-1, keepdims=True)
    has_selected = jnp.any(selected, axis=-1, keepdims=True)
    return jnp.where(has_selected, largest, 0 if fallback is None else fallback)


def scaled_lottery_terms_with_scale(
    *, values: FloatND, coefficients: FloatND, shifts: IntND
) -> tuple[FloatND, FloatND, IntND]:
    """Return a lottery's value terms and weight terms on one shared scale.

    A node's contribution to a mean is `c * 2**-s * v`, and the order those
    three are combined in decides whether the node survives. Two orders each
    lose it at one end of the range:

    - forming the weight `c * 2**-s` first loses a rare node, because the
      weight of a node many binades below the likeliest one is not
      representable and becomes zero, taking its value with it — although the
      *product* it was heading for may be an ordinary number, which is
      precisely the case where the node changes the answer;
    - forming `c * v` first loses a node whose coefficient has been normalized
      above one and whose value sits near the top of the range, because that
      intermediate overflows to infinity before any scale is applied to bring
      it back.

    Splitting the scale covers both. The coefficient absorbs as much of the
    downward shift as it can while staying normal, the product absorbs the
    remainder. Neither an unrepresentable weight nor an overflowing product is
    ever materialized, and a rare node earns its place in the mean exactly when
    its value is large enough to offset its probability.

    Normalize each positive finite coefficient into a normal significand
    before any product; its exact binary exponent moves into its shift.
    Both returned terms carry the same shift, so their scale cancels in the
    ratio. The common shift is selected from those normalized encodings, per
    lottery rather than per batch. No unrelated lottery or dead node selects it.

    The values are not normalized, so a value near the bottom of the normal
    range can lose its product. The caller is the scaled mean's centered
    residual, whose values are normal significands; a mean reduces through
    `scaled_lottery_mean` instead.

    This physical-scale helper requires all relative shift and derived exponent
    arithmetic, including the returned exponent, to fit int32. Its caller uses
    bounded IEEE-exponent-derived shifts. This is not an arbitrary private-int32
    endpoint-span interface, and no live exponent is silently saturated.

    Headroom bounds each actual scaled contribution, as well as the weight
    total. An extremely rare large value must not force an unrelated normal
    small contribution below the arithmetic range. A cardinality bound keeps
    both sums below the top binade without materializing an overflowing product.

    A coefficient that is not a probability at all — negative, infinite, or
    NaN — is not a dead node and is not silently dropped. It poisons its whole
    row, so a malformed lottery is visible in the result rather than reduced to
    a plausible number.

    Args:
        values: Continuation values of the lottery along the last axis.
        coefficients: The weights' significands, over the same axis.
        shifts: Each weight's own base-two scale, broadcast against them.

    Returns:
        Tuple of the value terms and the weight terms, both over the last axis
        and both carrying the row's shared scale, followed by that binary exponent. This
        exponent multiplies the unscaled terms; subtract it to recover their scale.

    """
    coefficients = jnp.asarray(coefficients)
    values = jnp.asarray(values)
    shifts = jnp.broadcast_to(jnp.asarray(shifts), jnp.shape(coefficients))

    live = is_live(coefficients) & jnp.isfinite(coefficients)
    invalid = is_negative(coefficients) | (
        ~jnp.isfinite(coefficients) & ~is_represented_zero(coefficients)
    )
    # Normalize each coefficient before multiplying by a small value. Its
    # exact power of two moves into the accompanying shift, not into a float
    # product that may underflow even when the normalized mean is normal.
    coefficient_exponent = jnp.where(live, _unbiased_exponent(coefficients), 0)
    normalization = -coefficient_exponent
    coefficients = scaled_down_by_power_of_two(
        values=coefficients, shift=jnp.minimum(normalization, 0)
    )
    coefficients = scaled_by_power_of_two(
        values=coefficients, shift=jnp.maximum(normalization, 0)
    )
    shifts = shifts - coefficient_exponent
    unusable = jnp.max(shifts, axis=-1, keepdims=True)
    common = jnp.min(jnp.where(live, shifts, unusable), axis=-1, keepdims=True)
    scale = (common - shifts).astype(jnp.int32)

    contribution_exponent = (
        _unbiased_exponent(coefficients)
        + scale
        + jnp.maximum(_unbiased_exponent(jnp.abs(values)), 0)
    )
    largest_exponent = jnp.max(
        jnp.where(live, contribution_exponent, jnp.iinfo(jnp.int32).min // 2),
        axis=-1,
        keepdims=True,
    )
    # Each product's two significands are below two; n terms add ceil(log2(n))
    # binades. Keeping the bound below 2**(maxexp-1) leaves rounding headroom.
    sum_binades = (values.shape[-1] - 1).bit_length()
    headroom = jnp.maximum(
        largest_exponent + 2 + sum_binades - (jnp.finfo(values.dtype).maxexp - 1),
        0,
    )
    full = scale - headroom

    room = binades_above_smallest_normal(coefficients)
    on_weight = jnp.maximum(full, -room)
    scaled_coefficients = scaled_down_by_power_of_two(
        values=coefficients, shift=on_weight
    )
    products = zero_safe_weighted_term(
        weight=scaled_coefficients,
        value=values,
        subnormal_is_accounted_for=False,
    )
    value_terms = scaled_down_by_power_of_two(values=products, shift=full - on_weight)
    weight_terms = scaled_down_by_power_of_two(values=coefficients, shift=full)

    # Zero coefficients already annihilate their values; masking the terms
    # again would erase a finite node's derivative with respect to its weight.
    invalid_row = jnp.any(invalid, axis=-1, keepdims=True)
    nan = jnp.asarray(jnp.nan, dtype=coefficients.dtype)
    return (
        jnp.where(invalid_row, nan, value_terms),
        jnp.where(invalid_row, nan, weight_terms),
        common - headroom,
    )
