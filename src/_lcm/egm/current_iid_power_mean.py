"""Private explicit economic directions of the genuine stable PowerMean.

This reducer consumes transient original child/coordinate records. Signed
statistics retain their scales through final multiplication by C. It neither
differentiates a stored level interpolant nor changes aggregate_scaled's AD
refusal. Complete-bank ownership/admission belongs to its actual producer.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_power_math import (
    _signed,
    _SignedLog,
    _sum_signed,
)
from _lcm.probability import (
    is_below_smallest_normal,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarFloat


def _join(*, parts: tuple[_SignedLog, ...]) -> _SignedLog:
    return _SignedLog(
        jnp.concatenate(tuple(jnp.ravel(part.logabs) for part in parts)),
        jnp.concatenate(tuple(jnp.ravel(part.sign) for part in parts)),
    )


def _unit_risk_exponent(
    *,
    values: FloatND,
    d: FloatND,
    log_w: FloatND,
    positive: BoolND,
    log_z: ScalarFloat,
    log_c: ScalarFloat,
    dp: ScalarFloat,
) -> _SignedLog:
    """Keep nonnegative pair variance terms scaled through the final CE.

    Var(ell)=sum_{i<j} wi*wj*(ell_i-ell_j)^2/Z^2. A rounded centroid can
    give the dominant node a false squared residual larger than a rare term.
    Pair differences avoid that subtraction. Only a transient row is retained.
    A representable pair ratio gives each difference as one rounded log; two
    rounded logs of extreme values would carry ulp(|log v|) into it.
    """
    signed_dp = _signed(value=dp)
    ln_two = jnp.log(jnp.asarray(2.0, dtype=d.dtype))
    indices = jnp.arange(d.shape[0], dtype=jnp.int32)
    zero = _SignedLog(
        jnp.asarray(-jnp.inf, dtype=d.dtype), jnp.zeros((), dtype=d.dtype)
    )

    # keyword-only-exempt: library-callback=jax.lax.scan
    def fold(state: _SignedLog, index: IntND) -> tuple[_SignedLog, None]:
        ratio = values[index] / values
        exact = jnp.isfinite(ratio) & ~is_below_smallest_normal(ratio)
        difference = _signed(
            value=jnp.where(exact, jnp.log(jnp.where(exact, ratio, 1.0)), d[index] - d)
        )
        active = positive[index] & positive & (indices > index)
        row = _SignedLog(
            log_c
            + signed_dp.logabs
            + log_w[index]
            + log_w
            - 2.0 * log_z
            + 2.0 * difference.logabs
            - ln_two,
            active.astype(d.dtype) * signed_dp.sign,
        )
        reduced = _sum_signed(terms=row)
        return _sum_signed(terms=_join(parts=(state, reduced))), None

    result, _ = jax.lax.scan(fold, zero, indices)
    return result
