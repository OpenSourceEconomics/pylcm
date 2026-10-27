"""Private first-order CE coefficient transport with an explicit transpose.

Primal coefficient metadata stays separate from the linear operand. IEEE bits
retain subnormal inputs and final outputs; the transpose uses the same scalar
transport rather than differentiating bit operations. Nonzero underflow and
nonfinite operands/results refuse. This does not accumulate opposing scales.
"""

from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
from jax._src.core import AbstractValue, ShapedArray
from jax._src.literals import TypedNdArray
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
    is_represented_zero,
    scaled_by_power_of_two,
    scaled_down_by_power_of_two,
)
from lcm.typing import BoolND, FloatND, IntND

type _LinearPrimal = FloatND | Float[TypedNdArray, "..."] | ad.UndefinedPrimal


def _select_payload(*, predicate: BoolND, left: FloatND, right: FloatND) -> FloatND:
    """Select original IEEE bits without consuming a subnormal floating operand."""
    integer = jnp.uint32 if left.dtype == jnp.float32 else jnp.uint64
    selected = jnp.where(
        predicate,
        jax.lax.bitcast_convert_type(left, integer),
        jax.lax.bitcast_convert_type(right, integer),
    )
    return jax.lax.bitcast_convert_type(selected, left.dtype)


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
    logabs: FloatND, sign: FloatND, base: FloatND, direction: FloatND
) -> FloatND:
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


# keyword-only-exempt: library-callback=jax.core.Primitive.def_abstract_eval
def _abstract(
    logabs: AbstractValue,
    sign: AbstractValue,
    base: AbstractValue,
    direction: AbstractValue,
) -> ShapedArray:
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
) -> tuple[FloatND, FloatND | ad.Zero]:
    logabs, sign, base, direction = primals
    log_dot, sign_dot, base_dot, direction_dot = tangents
    if not isinstance(log_dot, ad.Zero) or not isinstance(sign_dot, ad.Zero):
        raise NotImplementedError("CE transport supports first-order residuals only.")
    primal = join_ce_direction(logabs=logabs, sign=sign, base=base, direction=direction)
    tangent = (
        ad.Zero(jax.typeof(primal).to_tangent_aval())
        if isinstance(direction_dot, ad.Zero) and isinstance(base_dot, ad.Zero)
        else join_ce_direction(
            logabs=logabs,
            sign=sign,
            base=jnp.zeros_like(base) if isinstance(base_dot, ad.Zero) else base_dot,
            direction=jnp.zeros_like(direction)
            if isinstance(direction_dot, ad.Zero)
            else direction_dot,
        )
    )
    return primal, tangent


# keyword-only-exempt: library-callback=jax.interpreters.ad.primitive_transposes
def _transpose(
    cotangent: FloatND | ad.Zero,
    logabs: FloatND | ad.UndefinedPrimal,
    sign: FloatND | ad.UndefinedPrimal,
    base: _LinearPrimal,
    direction: _LinearPrimal,
) -> tuple[None, None, FloatND | ad.Zero | None, FloatND | ad.Zero | None]:
    if isinstance(logabs, ad.UndefinedPrimal) or isinstance(sign, ad.UndefinedPrimal):
        raise NotImplementedError("CE transport transposes only its linear direction.")
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
        _select_payload(
            predicate=_valid_coefficient(logabs=logabs, sign=sign)
            & jnp.isfinite(cotangent),
            left=cotangent,
            right=jnp.full_like(cotangent, jnp.nan),
        )
        if isinstance(base, ad.UndefinedPrimal)
        else None,
        scale_ce_direction(logabs=logabs, sign=sign, direction=cotangent)
        if isinstance(direction, ad.UndefinedPrimal)
        else None,
    )


# keyword-only-exempt: library-callback=jax.interpreters.batching.primitive_batchers
def _batch(
    arguments: Sequence[FloatND], dimensions: Sequence[int | None]
) -> tuple[FloatND, int | None]:
    sizes = {
        argument.shape[dimension]
        for argument, dimension in zip(arguments, dimensions, strict=True)
        if dimension is not None
    }
    if not sizes:
        return join_ce_direction(
            logabs=arguments[0],
            sign=arguments[1],
            base=arguments[2],
            direction=arguments[3],
        ), None
    if len(sizes) != 1:
        raise TypeError("CE transport requires one consistent mapped axis.")
    size = next(iter(sizes))
    moved = [
        batching.bdim_at_front(argument, dimension, size)
        for argument, dimension in zip(arguments, dimensions, strict=True)
    ]
    return join_ce_direction(
        logabs=moved[0], sign=moved[1], base=moved[2], direction=moved[3]
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
