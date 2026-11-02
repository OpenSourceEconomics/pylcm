"""Private CE coefficient transport with an explicit transpose.

Primal coefficient metadata stays separate from the linear operand. IEEE bits
retain subnormal inputs and final outputs; the transpose uses the same scalar
transport rather than differentiating bit operations. Nonzero underflow and
nonfinite operands/results refuse. This does not accumulate opposing scales.
"""

from collections.abc import Sequence
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax._src.core import AbstractValue, ShapedArray
from jax._src.literals import TypedFloat, TypedNdArray
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jaxtyping import Float

from _lcm.egm.current_iid_power_math import (
    _materialize,
    _signed,
    _SignedLog,
    _sum_signed,
)
from _lcm.probability import (
    _significand_in_unit_range,
    _unbiased_exponent,
    is_below_smallest_normal,
    is_negative,
    is_represented_zero,
    log_of_nonnegative,
    scaled_by_power_of_two,
    scaled_down_by_power_of_two,
)
from lcm.typing import BoolND, FloatND, IntND

type _LinearPrimal = (
    FloatND | Float[TypedNdArray, "..."] | TypedFloat | ad.UndefinedPrimal
)
type _CoefficientFormat = Literal["log", "represented"]


def _select_payload(*, predicate: BoolND, left: FloatND, right: FloatND) -> FloatND:
    """Select original IEEE bits without consuming a subnormal floating operand."""
    integer = jnp.uint32 if left.dtype == jnp.float32 else jnp.uint64
    selected = jnp.where(
        predicate,
        jax.lax.bitcast_convert_type(left, integer),
        jax.lax.bitcast_convert_type(right, integer),
    )
    return jax.lax.bitcast_convert_type(selected, left.dtype)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _select_cotangent(predicate: BoolND, left: FloatND, right: FloatND) -> FloatND:
    """Select bits exactly, yet differentiate as the selection it is."""
    return _select_payload(predicate=predicate, left=left, right=right)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _select_cotangent_jvp(
    predicate: BoolND,
    primals: tuple[FloatND, FloatND],
    tangents: tuple[FloatND, FloatND],
) -> tuple[FloatND, FloatND]:
    # A bitcast has no derivative or transpose, so higher-order transposes would
    # vanish; the tangent is the ordinary linear selection.
    left, right = primals
    left_dot, right_dot = tangents
    return (
        _differentiable_select(predicate, left, right),
        jnp.where(predicate, left_dot, right_dot),
    )


# Construct by call so the method-only import hook retains the JVP registration.
_differentiable_select = jax.custom_jvp(_select_cotangent, nondiff_argnums=(0,))
_differentiable_select.defjvp(_select_cotangent_jvp)


def _finite_bits(value: FloatND) -> BoolND:
    """Classify the original float32/64 IEEE exponent without floating operands."""
    integer = jnp.uint32 if value.dtype == jnp.float32 else jnp.uint64
    mask = 0x7F800000 if value.dtype == jnp.float32 else 0x7FF0000000000000
    bits = jax.lax.bitcast_convert_type(value, integer)
    return (bits & jnp.asarray(mask, dtype=integer)) != mask


def scale_ce_direction(
    *, logabs: FloatND, sign: FloatND, direction: FloatND
) -> FloatND:
    """Apply an authentic nonnegative scaled CE coefficient to one direction."""
    return join_ce_direction(
        logabs=logabs, sign=sign, base=jnp.zeros_like(direction), direction=direction
    )


def join_ce_direction(
    *, logabs: FloatND, sign: FloatND, base: FloatND, direction: FloatND
) -> FloatND:
    """Join a scaled CE direction under the existing cancellation refusal."""
    logarithm, polarity, original, operand = jnp.broadcast_arrays(
        logabs, sign, base, direction
    )
    # The primitive's abstract rule checks equal float32/64 arrays and returns
    # exactly that numerical shape/dtype; it cannot produce another tree/type.
    return cast("FloatND", _scaled_ce_p.bind(logarithm, polarity, original, operand))


def join_represented_ce_direction(
    *, coefficient: FloatND, sign: FloatND, base: FloatND, direction: FloatND
) -> FloatND:
    """Join an ordinary caller's actual coefficient without a logarithmic roundtrip."""
    factor, polarity, original, operand = jnp.broadcast_arrays(
        coefficient, sign, base, direction
    )
    return cast(
        "FloatND",
        _scaled_ce_p.bind(
            factor, polarity, original, operand, coefficient_format="represented"
        ),
    )


def _coefficient_parts(*, logabs: FloatND) -> tuple[FloatND, IntND, BoolND]:
    """Retain a bounded binary coefficient without materializing its scale."""
    info = jnp.finfo(logabs.dtype)
    ln_two = jnp.log(jnp.asarray(2.0, dtype=logabs.dtype))
    limit = info.maxexp - info.minexp + info.nmant + 2
    bounded = jnp.clip(logabs, -limit * ln_two, limit * ln_two)
    bounded = jnp.where(jnp.isfinite(bounded), bounded, 0.0)
    exponent = jnp.floor(bounded / ln_two).astype(jnp.int32)
    mantissa = jnp.exp(bounded - exponent * ln_two)
    represented = (logabs >= -limit * ln_two) & (logabs <= limit * ln_two)
    return mantissa, exponent, represented


def _transport(*, logabs: FloatND, sign: FloatND, direction: FloatND) -> FloatND:
    """Scale the represented binary significand before final IEEE rounding."""
    info = jnp.finfo(logabs.dtype)
    mantissa, exponent, coefficient_bounded = _coefficient_parts(logabs=logabs)
    coefficient_valid = _valid_coefficient(logabs=logabs, sign=sign)
    finite_input = jnp.isfinite(direction)
    zero = is_represented_zero(direction) | (sign == 0.0)
    # Only normal significands multiply. Neither the coefficient nor a tiny
    # incoming direction becomes a vulnerable floating-point operand.
    product = mantissa * _significand_in_unit_range(direction)
    shift = exponent + _unbiased_exponent(direction)
    overflow = _unbiased_exponent(product) + shift >= info.maxexp
    downward = scaled_down_by_power_of_two(
        values=jnp.where(shift < 0, product, jnp.ones_like(product)),
        shift=jnp.where(shift < 0, shift, jnp.zeros_like(shift)),
    )
    upward_live = (shift >= 0) & ~overflow
    upward = scaled_by_power_of_two(
        values=jnp.where(upward_live, product, jnp.ones_like(product)),
        shift=jnp.where(upward_live, shift, jnp.zeros_like(shift)),
    )
    result = jnp.copysign(jnp.where(shift < 0, downward, upward), direction)
    retained = (
        coefficient_bounded
        & ~overflow
        & jnp.isfinite(result)
        & ~is_represented_zero(result)
    )
    return jnp.where(
        coefficient_valid & finite_input,
        jnp.where(
            zero,
            jnp.copysign(jnp.zeros_like(result), direction),
            jnp.where(retained, result, jnp.nan),
        ),
        jnp.nan,
    )


def _valid_coefficient(*, logabs: FloatND, sign: FloatND) -> BoolND:
    return ((sign == 1.0) & jnp.isfinite(logabs)) | (
        (sign == 0.0) & (logabs == -jnp.inf)
    )


# keyword-only-exempt: library-callback=jax.core.Primitive.def_impl
def _linear_join(
    logabs: FloatND,
    sign: FloatND,
    base: FloatND,
    direction: FloatND,
    *,
    coefficient_format: _CoefficientFormat = "log",
) -> FloatND:
    if coefficient_format == "represented":
        return _represented_join(
            coefficient=logabs, sign=sign, base=base, direction=direction
        )
    if coefficient_format != "log":
        raise TypeError("CE transport requires a declared coefficient format.")
    scaled = _transport(logabs=logabs, sign=sign, direction=direction)
    original, correction = _signed(value=base), _signed(value=scaled)
    joined = _materialize(
        value=_sum_signed(
            terms=_SignedLog(
                jnp.stack((original.logabs, correction.logabs)),
                jnp.stack((original.sign, correction.sign)),
            ),
            axis=0,
        )
    )
    # Add the actual represented operands before their rounded logs collide.
    ordinary = base + scaled
    normal_operands = ~is_below_smallest_normal(base) & ~is_below_smallest_normal(
        scaled
    )
    exact_zero = is_represented_zero(ordinary) & (base == -scaled)
    ordinary_retained = _finite_bits(ordinary) & (
        ~is_below_smallest_normal(ordinary) | exact_zero
    )
    joined = _select_payload(
        predicate=normal_operands & _finite_bits(joined),
        left=_select_payload(
            predicate=ordinary_retained,
            left=ordinary,
            right=jnp.full_like(ordinary, jnp.nan),
        ),
        right=joined,
    )
    result = _select_payload(
        predicate=is_represented_zero(base),
        left=scaled,
        right=_select_payload(
            predicate=is_represented_zero(scaled), left=base, right=joined
        ),
    )
    # Keep the final decision in integer bits: the original fused floating
    # finite guard lost subnormal identity in compiled joins.
    return _select_payload(
        predicate=_finite_bits(base) & _finite_bits(scaled),
        left=result,
        right=jnp.full_like(result, jnp.nan),
    )


def _represented_join(
    *, coefficient: FloatND, sign: FloatND, base: FloatND, direction: FloatND
) -> FloatND:
    """Use ordinary retained products and sums, preserving exceptional transport."""
    zero = jnp.zeros_like(direction)
    info = jnp.finfo(coefficient.dtype)
    coefficient_zero = is_represented_zero(coefficient)
    direction_zero = is_represented_zero(direction)
    valid_coefficient = _finite_bits(coefficient) & (
        ((sign == 1.0) & ~is_negative(coefficient) & ~coefficient_zero)
        | ((sign == 0.0) & coefficient_zero)
    )
    product_zero = (
        valid_coefficient
        & _finite_bits(direction)
        & (coefficient_zero | direction_zero)
    )
    product_exponent = _unbiased_exponent(coefficient) + _unbiased_exponent(direction)
    # Bound multiplication before selecting its operands, including under vmap.
    direct_product = (
        valid_coefficient
        & (sign == 1.0)
        & ~is_below_smallest_normal(coefficient)
        & _finite_bits(direction)
        & ~is_below_smallest_normal(direction)
        & ~direction_zero
        & (product_exponent >= info.minexp)
        & (product_exponent < info.maxexp - 1)
    )
    ordinary_product = jnp.where(direct_product, coefficient, 1.0) * _select_payload(
        predicate=direct_product, left=direction, right=zero
    )
    keep_product = (
        direct_product
        & _finite_bits(ordinary_product)
        & ~is_below_smallest_normal(ordinary_product)
        & ~is_represented_zero(ordinary_product)
    )
    use_product = keep_product | product_zero
    exceptional_product = _transport(
        logabs=log_of_nonnegative(jnp.where(use_product, 1.0, coefficient)),
        sign=jnp.where(use_product, 1.0, sign),
        direction=_select_payload(predicate=use_product, left=zero, right=direction),
    )
    scaled = _select_payload(
        predicate=use_product,
        left=_select_payload(predicate=product_zero, left=zero, right=ordinary_product),
        right=exceptional_product,
    )
    base_zero = is_represented_zero(base)
    scaled_zero = is_represented_zero(scaled)
    addable = (
        _finite_bits(base)
        & _finite_bits(scaled)
        & (~is_below_smallest_normal(base) | base_zero)
        & (~is_below_smallest_normal(scaled) | scaled_zero)
        & (
            base_zero
            | scaled_zero
            | (is_negative(base) != is_negative(scaled))
            | (
                jnp.maximum(_unbiased_exponent(base), _unbiased_exponent(scaled))
                <= info.maxexp - 2
            )
        )
    )
    ordinary = _select_payload(predicate=addable, left=base, right=zero) + (
        _select_payload(predicate=addable, left=scaled, right=zero)
    )
    exact_zero = is_represented_zero(ordinary) & (base == -scaled)
    use_ordinary = (
        addable
        & _finite_bits(ordinary)
        & (~is_below_smallest_normal(ordinary) | exact_zero)
    )
    exceptional = _linear_join(
        jnp.zeros_like(coefficient),
        jnp.ones_like(sign),
        _select_payload(predicate=use_ordinary, left=zero, right=base),
        _select_payload(predicate=use_ordinary, left=zero, right=scaled),
    )
    return _select_payload(predicate=use_ordinary, left=ordinary, right=exceptional)


# keyword-only-exempt: library-callback=jax.core.Primitive.def_abstract_eval
def _abstract(
    logabs: AbstractValue,
    sign: AbstractValue,
    base: AbstractValue,
    direction: AbstractValue,
    *,
    coefficient_format: _CoefficientFormat = "log",
) -> ShapedArray:
    if coefficient_format not in ("log", "represented"):
        raise TypeError("CE transport requires a declared coefficient format.")
    if not isinstance(logabs, ShapedArray) or not isinstance(sign, ShapedArray):
        raise TypeError("CE transport requires shaped coefficient metadata.")
    if not isinstance(direction, ShapedArray):
        raise TypeError("CE transport requires a shaped direction.")
    if not isinstance(base, ShapedArray):
        raise TypeError("CE transport requires a shaped original direction.")
    if (
        logabs.shape != direction.shape
        or sign.shape != direction.shape
        or base.shape != direction.shape
        or logabs.dtype != direction.dtype
        or sign.dtype != direction.dtype
        or base.dtype != direction.dtype
        or direction.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
    ):
        raise TypeError("CE transport requires matching float32/float64 arrays.")
    return direction.update(weak_type=False)


# keyword-only-exempt: library-callback=jax.interpreters.ad.primitive_jvps
def _jvp(
    primals: tuple[FloatND, FloatND, FloatND, FloatND],
    tangents: tuple[
        FloatND | ad.Zero, FloatND | ad.Zero, FloatND | ad.Zero, FloatND | ad.Zero
    ],
    *,
    coefficient_format: _CoefficientFormat = "log",
) -> tuple[FloatND, FloatND | ad.Zero]:
    logabs, sign, base, direction = primals
    # The sign is piecewise constant, so its tangent vanishes almost everywhere.
    log_dot, _sign_dot, base_dot, direction_dot = tangents
    if not isinstance(log_dot, ad.Zero):
        # A coefficient tangent (higher-order derivatives) folds into the same
        # transport: d(s e^L d) = s e^L (d dL) in log format, and the represented
        # coefficient is linear, so d(s R d) = (s d dR) passes through the base.
        if coefficient_format == "log":
            scaled = direction * log_dot
            direction_dot = (
                scaled if isinstance(direction_dot, ad.Zero) else direction_dot + scaled
            )
        else:
            linear = sign * direction * log_dot
            base_dot = linear if isinstance(base_dot, ad.Zero) else base_dot + linear
    primal = cast(
        "FloatND",
        _scaled_ce_p.bind(
            logabs, sign, base, direction, coefficient_format=coefficient_format
        ),
    )
    tangent = (
        ad.Zero(jax.typeof(primal).to_tangent_aval())
        if isinstance(direction_dot, ad.Zero) and isinstance(base_dot, ad.Zero)
        else cast(
            "FloatND",
            _scaled_ce_p.bind(
                logabs,
                sign,
                jnp.zeros_like(base) if isinstance(base_dot, ad.Zero) else base_dot,
                jnp.zeros_like(direction)
                if isinstance(direction_dot, ad.Zero)
                else direction_dot,
                coefficient_format=coefficient_format,
            ),
        )
    )
    return primal, tangent


# keyword-only-exempt: library-callback=jax.interpreters.ad.primitive_transposes
def _transpose(
    cotangent: FloatND | ad.Zero,
    logabs: _LinearPrimal,
    sign: _LinearPrimal,
    base: _LinearPrimal,
    direction: _LinearPrimal,
    *,
    coefficient_format: _CoefficientFormat = "log",
) -> tuple[None, None, FloatND | ad.Zero | None, FloatND | ad.Zero | None]:
    if isinstance(logabs, ad.UndefinedPrimal) or isinstance(sign, ad.UndefinedPrimal):
        raise NotImplementedError("CE transport transposes only its linear direction.")
    logabs = jnp.asarray(logabs)
    sign = jnp.asarray(sign)
    coefficient_log = (
        log_of_nonnegative(logabs) if coefficient_format == "represented" else logabs
    )
    if isinstance(cotangent, ad.Zero):
        return (
            None,
            None,
            ad.Zero(base.aval) if isinstance(base, ad.UndefinedPrimal) else None,
            ad.Zero(direction.aval)
            if isinstance(direction, ad.UndefinedPrimal)
            else None,
        )
    return (
        None,
        None,
        _differentiable_select(
            _valid_coefficient(logabs=coefficient_log, sign=sign)
            & jnp.isfinite(cotangent),
            cotangent,
            jnp.full_like(cotangent, jnp.nan),
        )
        if isinstance(base, ad.UndefinedPrimal)
        else None,
        cast(
            "FloatND",
            _scaled_ce_p.bind(
                logabs,
                sign,
                jnp.zeros_like(cotangent),
                cotangent,
                coefficient_format=coefficient_format,
            ),
        )
        if isinstance(direction, ad.UndefinedPrimal)
        else None,
    )


# keyword-only-exempt: library-callback=jax.interpreters.batching.primitive_batchers
def _batch(
    arguments: Sequence[FloatND],
    dimensions: Sequence[int | None],
    *,
    coefficient_format: _CoefficientFormat = "log",
) -> tuple[FloatND, int | None]:
    sizes = {
        argument.shape[dimension]
        for argument, dimension in zip(arguments, dimensions, strict=True)
        if dimension is not None
    }
    if not sizes:
        return cast(
            "FloatND",
            _scaled_ce_p.bind(*arguments, coefficient_format=coefficient_format),
        ), None
    if len(sizes) != 1:
        raise TypeError("CE transport requires one consistent mapped axis.")
    size = next(iter(sizes))
    moved = [
        batching.bdim_at_front(argument, dimension, size)
        for argument, dimension in zip(arguments, dimensions, strict=True)
    ]
    return cast(
        "FloatND", _scaled_ce_p.bind(*moved, coefficient_format=coefficient_format)
    ), 0


_scaled_ce_p = Primitive("lcm_ce_scaled_linear")
_scaled_ce_p.def_impl(_linear_join)
_scaled_ce_p.def_abstract_eval(_abstract)
ad.primitive_jvps[_scaled_ce_p] = _jvp
ad.primitive_transposes[_scaled_ce_p] = _transpose
batching.primitive_batchers[_scaled_ce_p] = _batch
mlir.register_lowering(
    _scaled_ce_p, mlir.lower_fun(_linear_join, multiple_results=False)
)
