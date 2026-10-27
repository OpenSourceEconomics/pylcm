"""Private divided differences and signed log-scaled economic arithmetic.

The small-region contract is absolute error <= 64*eps for the polynomial
evaluation, including coefficient rounding and its analytic omitted tail,
under ordinary round-to-nearest arithmetic without overflow. This is not a
library-transcendental ULP claim or a structural topology certificate.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp

from _lcm.probability import (
    _unbiased_exponent,
    is_negative,
    is_represented_zero,
    log_of_nonnegative,
    scaled_by_power_of_two,
    scaled_down_by_power_of_two,
)
from lcm.typing import FloatND

_SMALL_REGION = 0.125


class _SignedLog(NamedTuple):
    """A signed statistic retained without materializing its magnitude."""

    logabs: FloatND
    sign: FloatND


def _signed(*, value: FloatND) -> _SignedLog:
    zero = is_represented_zero(value)
    sign = jnp.where(zero, 0.0, jnp.where(is_negative(value), -1.0, 1.0))
    return _SignedLog(log_of_nonnegative(jnp.abs(value)), sign)


def _sum_signed(*, terms: _SignedLog, axis: int | None = None) -> _SignedLog:
    live = (terms.sign != 0.0) & (terms.logabs != -jnp.inf)
    peak = jnp.max(jnp.where(live, terms.logabs, -jnp.inf), axis=axis, keepdims=True)
    low = jnp.min(jnp.where(live, terms.logabs, jnp.inf), axis=axis, keepdims=True)
    # A same-sign inner sum may lose a small constituent before an outer sum
    # opposes its aggregate. Refuse wide span before ANY aggregation, so a
    # later merge cannot borrow a false equal-scale cancellation certificate.
    # Complete multi-scale accumulation remains required for this frontier.
    cancellation_unresolved = (peak - low) > jnp.finfo(
        terms.logabs.dtype
    ).nmant * jnp.log(2.0)
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    mantissa = jnp.sum(
        jnp.where(live, terms.sign * jnp.exp(terms.logabs - peak), 0.0),
        axis=axis,
        keepdims=True,
    )
    result = _signed(value=mantissa)
    logabs = peak + result.logabs
    invalid = cancellation_unresolved | jnp.any(
        jnp.isnan(terms.logabs), axis=axis, keepdims=True
    )
    logabs = jnp.where(invalid, jnp.nan, logabs)
    if axis is None:
        return _SignedLog(jnp.reshape(logabs, ()), jnp.reshape(result.sign, ()))
    return _SignedLog(jnp.squeeze(logabs, axis), jnp.squeeze(result.sign, axis))


def _materialize(*, value: _SignedLog) -> FloatND:
    """Apply the final scale once, including represented subnormal results."""
    dtype = value.logabs.dtype
    info = jnp.finfo(dtype)
    ln_two = jnp.log(jnp.asarray(2.0, dtype=dtype))
    # Keep exponent conversion bounded even on an invalid/overflowing result.
    bounded = jnp.clip(
        value.logabs,
        (info.minexp - info.nmant - 2) * ln_two,
        (info.maxexp + 1) * ln_two,
    )
    bounded = jnp.where(jnp.isfinite(bounded), bounded, 0.0)
    exponent = jnp.floor(bounded / ln_two).astype(jnp.int32)
    mantissa = value.sign * jnp.exp(bounded - exponent * ln_two)
    negative = exponent < 0
    overflow = (
        jnp.isfinite(mantissa)
        & ~is_represented_zero(mantissa)
        & (_unbiased_exponent(mantissa) + exponent >= info.maxexp)
    )
    # Inactive inputs must be safe even when vmap evaluates both branches.
    downward = scaled_down_by_power_of_two(
        values=jnp.where(negative, mantissa, jnp.ones_like(mantissa)),
        shift=jnp.where(negative, exponent, jnp.zeros_like(exponent)),
    )
    upward_live = ~negative & ~overflow
    upward = scaled_by_power_of_two(
        values=jnp.where(upward_live, mantissa, jnp.ones_like(mantissa)),
        shift=jnp.where(upward_live, exponent, jnp.zeros_like(exponent)),
    )
    result = jnp.where(negative, downward, upward)
    result = jnp.where(overflow, value.sign * jnp.inf, result)
    result = jnp.where(
        value.logabs > (info.maxexp + 1) * ln_two, value.sign * jnp.inf, result
    )
    result = jnp.where(
        value.logabs < (info.minexp - info.nmant - 2) * ln_two,
        jnp.zeros_like(result),
        result,
    )
    result = jnp.where(value.sign == 0.0, jnp.zeros_like(result), result)
    return jnp.where(jnp.isnan(value.logabs), jnp.nan, result)


def _polynomial(*, argument: FloatND, coefficients: tuple[float, ...]) -> FloatND:
    result = jnp.full_like(argument, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * argument + coefficient
    return result


def _series_degree(*, argument: FloatND, phi: bool = False) -> int:
    """Use a fixed dtype-derived, analytically bounded local polynomial.

    Domain A=1/8. exprel/psi degrees 8/14 and phi degrees 10/20 (f32/f64)
    have tails <= eps/16 by the integral bounds recorded in the private spec.
    Horner has at most 2m rounded operations and coefficients <=1; on this
    domain its absolute rounding bound is <(2m+3)*eps/(1-2m*eps)/(1-A),
    below 64*eps for the admitted dtypes. These bounds include coefficient
    conversion, assuming ordinary round-to-nearest arithmetic.
    """
    if argument.dtype == jnp.float32:
        return 10 if phi else 8
    if argument.dtype == jnp.float64:
        return 20 if phi else 14
    raise TypeError("PowerMean divided differences require float32 or float64.")


def _log_exprel(*, argument: FloatND) -> FloatND:
    """Log(expm1(t)/t), preserving large derivative-only positive arguments."""
    small = jnp.abs(argument) <= _SMALL_REGION
    degree = _series_degree(argument=argument)
    polynomial = _polynomial(
        argument=jnp.where(small, argument, 0.0),
        coefficients=tuple(1.0 / math.factorial(k + 1) for k in range(degree + 1)),
    )
    positive = argument > _SMALL_REGION
    positive_arg = jnp.where(positive, argument, 1.0)
    negative_arg = jnp.where(~positive & ~small, argument, -1.0)
    ordinary = jnp.where(
        positive,
        positive_arg + jnp.log(-jnp.expm1(-positive_arg)) - jnp.log(positive_arg),
        jnp.log(-jnp.expm1(negative_arg)) - jnp.log(-negative_arg),
    )
    return jnp.where(small, jnp.log(polynomial), ordinary)


def _log_psi(*, argument: FloatND) -> FloatND:
    """Log integral_0^1 u*exp(u*t) du with a safe small-argument branch."""
    small = jnp.abs(argument) <= _SMALL_REGION
    degree = _series_degree(argument=argument)
    polynomial = _polynomial(
        argument=jnp.where(small, argument, 0.0),
        coefficients=tuple(
            1.0 / (math.factorial(k) * (k + 2)) for k in range(degree + 1)
        ),
    )
    high = argument > 1.0
    high_arg = jnp.where(high, argument, 2.0)
    ordinary_arg = jnp.where(~high & ~small, argument, -1.0)
    ordinary = jnp.where(
        high,
        high_arg
        + jnp.log(high_arg - 1.0 + jnp.exp(-high_arg))
        - 2.0 * jnp.log(high_arg),
        jnp.log1p((ordinary_arg - 1.0) * jnp.exp(ordinary_arg))
        - 2.0 * jnp.log(jnp.abs(ordinary_arg)),
    )
    return jnp.where(small, jnp.log(polynomial), ordinary)


def _phi(*, argument: FloatND) -> FloatND:
    """Negative normalized log1p divided difference on the primal near regime."""
    small = jnp.abs(argument) <= _SMALL_REGION
    degree = _series_degree(argument=argument, phi=True)
    polynomial = _polynomial(
        argument=jnp.where(small, argument, 0.0),
        coefficients=tuple(
            (-1.0) ** (k + 1) * (k + 1) / (k + 2) for k in range(degree + 1)
        ),
    )
    ordinary_arg = jnp.where(small, -0.25, argument)
    ordinary = (
        ordinary_arg / (1.0 + ordinary_arg) - jnp.log1p(ordinary_arg)
    ) / ordinary_arg**2
    return jnp.where(small, polynomial, ordinary)
