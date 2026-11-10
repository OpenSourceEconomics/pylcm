"""Private divided differences and signed log-scaled economic arithmetic.

The small-region contract is absolute error <= 64*eps for the polynomial
evaluation, including coefficient rounding and its analytic omitted tail,
under ordinary round-to-nearest arithmetic without overflow. This is not a
library-transcendental ULP claim or a structural topology certificate.
"""

from typing import NamedTuple

import jax.numpy as jnp

from _lcm.probability import (
    _unbiased_exponent,
    is_represented_zero,
    scaled_by_power_of_two,
    scaled_down_by_power_of_two,
)
from lcm.typing import FloatND


class _SignedLog(NamedTuple):
    """A signed statistic retained without materializing its magnitude."""

    logabs: FloatND
    sign: FloatND


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
