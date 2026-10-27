"""Private explicit economic directions of the genuine stable PowerMean.

This reducer consumes transient original child/coordinate records. Signed
statistics retain their scales through final multiplication by C. It neither
differentiates a stored level interpolant nor changes aggregate_scaled's AD
refusal. Complete-bank ownership/admission belongs to its actual producer.
"""

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import PowerMean
from _lcm.egm.current_iid_power_math import (
    _log_exprel,
    _log_psi,
    _materialize,
    _phi,
    _signed,
    _SignedLog,
    _sum_signed,
)
from _lcm.probability import (
    is_below_smallest_normal,
    is_represented_zero,
    log_of_nonnegative,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarBool, ScalarFloat


class PowerMeanNodeBank(NamedTuple):
    """Transient actual joint lottery with unmultiplied economic components."""

    values: FloatND
    economic: FloatND
    savings_coordinates: FloatND
    source_coordinates: FloatND
    coefficients: FloatND
    shifts: IntND
    differentiated_coefficients: FloatND
    differentiated_shifts: IntND
    needed: BoolND
    valid: ScalarBool


def _join(*, parts: tuple[_SignedLog, ...]) -> _SignedLog:
    return _SignedLog(
        jnp.concatenate(tuple(jnp.ravel(part.logabs) for part in parts)),
        jnp.concatenate(tuple(jnp.ravel(part.sign) for part in parts)),
    )


def _choose(
    *, predicate: ScalarBool, left: _SignedLog, right: _SignedLog
) -> _SignedLog:
    return _SignedLog(
        jnp.where(predicate, left.logabs, right.logabs),
        jnp.where(predicate, left.sign, right.sign),
    )


def _bank_input_valid(*, bank: PowerMeanNodeBank) -> ScalarBool:
    positive = ~is_represented_zero(bank.coefficients)
    normal = ~is_below_smallest_normal(bank.coefficients)
    # Per-event product scaling supplies normal coefficients. Therefore the
    # canonical primal's common lifting is identity, and Dw/Z share units.
    valid = bank.valid & jnp.all(~positive | normal)
    valid &= jnp.all(jnp.isfinite(bank.coefficients) & (bank.coefficients >= 0.0))
    valid &= jnp.all(jnp.isfinite(bank.differentiated_coefficients))
    valid &= jnp.all(
        ~bank.needed
        | (
            jnp.isfinite(bank.values)
            & (bank.values > 0.0)
            & jnp.isfinite(bank.economic)
            & jnp.isfinite(bank.savings_coordinates)
            & jnp.isfinite(bank.source_coordinates)
        )
    )
    return valid


def _unit_risk_exponent(
    *,
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
    """
    signed_dp = _signed(value=dp)
    ln_two = jnp.log(jnp.asarray(2.0, dtype=d.dtype))
    indices = jnp.arange(d.shape[0], dtype=jnp.int32)
    zero = _SignedLog(
        jnp.asarray(-jnp.inf, dtype=d.dtype), jnp.zeros((), dtype=d.dtype)
    )

    # keyword-only-exempt: library-callback=jax.lax.scan
    def fold(state: _SignedLog, index: IntND) -> tuple[_SignedLog, None]:
        difference = _signed(value=d[index] - d)
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


def reduce_power_mean_economic(
    *,
    declared: PowerMean,
    params: Mapping[str, FloatND],
    parameter_direction: Mapping[str, FloatND],
    bank: PowerMeanNodeBank,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    """Price the declared joint lottery once and retain both economic directions."""
    if type(declared) is not PowerMean:
        raise TypeError("Stable economic context requires the actual exact PowerMean.")
    context = declared._scaled_context(  # noqa: SLF001
        values=bank.values,
        coefficients=bank.coefficients,
        shifts=bank.shifts,
        params=params,
    )
    p = 1.0 - params["risk_aversion"]
    dp = -parameter_direction["risk_aversion"]
    ln_two = jnp.log(jnp.asarray(2.0, dtype=bank.values.dtype))
    positive = ~is_represented_zero(bank.coefficients)
    valid = _bank_input_valid(bank=bank)
    # The primal's null-node masked logs are deliberately not reused here.
    values = jnp.where(bank.needed, bank.values, 1.0)
    log_v = log_of_nonnegative(values)
    d = log_v - context.anchor
    u = p * d
    valid &= jnp.all(~bank.needed | jnp.isfinite(u))
    log_w = log_of_nonnegative(bank.coefficients) - bank.shifts * ln_two
    log_z = context.log_mass
    log_t = log_z + context.log_moment
    log_c = log_of_nonnegative(context.value)
    valid &= jnp.all(jnp.isfinite(jnp.stack((p, dp, log_z, log_t, log_c))))
    economic = _signed(value=jnp.where(bank.needed, bank.economic, 0.0))

    def child_terms(*, coordinate: FloatND) -> _SignedLog:
        direction = _signed(value=jnp.where(bank.needed, coordinate, 0.0))
        return _SignedLog(
            log_c + log_w + u - log_t + economic.logabs + direction.logabs - log_v,
            positive.astype(values.dtype) * economic.sign * direction.sign,
        )

    savings_terms = child_terms(coordinate=bank.savings_coordinates)
    source_terms = child_terms(coordinate=bank.source_coordinates)
    q = _signed(value=d)
    q = _SignedLog(q.logabs + _log_exprel(argument=u), q.sign)
    mean_q = _sum_signed(
        terms=_SignedLog(
            log_w + q.logabs - log_z, positive.astype(values.dtype) * q.sign
        )
    )
    dw = _signed(value=bank.differentiated_coefficients)
    log_dw = dw.logabs - bank.differentiated_shifts * ln_two
    near_weight = _join(
        parts=(
            _SignedLog(log_c + log_dw + q.logabs - log_t, dw.sign * q.sign),
            _SignedLog(log_c + log_dw + mean_q.logabs - log_t, -dw.sign * mean_q.sign),
        )
    )
    safe_p = jnp.where(p == 0.0, 1.0, p)
    signed_p = _signed(value=safe_p)
    far_weight = _join(
        parts=(
            _SignedLog(
                log_c + log_dw + u - log_t - signed_p.logabs, dw.sign * signed_p.sign
            ),
            _SignedLog(
                log_c + log_dw - log_z - signed_p.logabs, -dw.sign * signed_p.sign
            ),
        )
    )
    weight_terms = _choose(
        predicate=context.near_geometric, left=near_weight, right=far_weight
    )

    signed_dp = _signed(value=dp)
    signed_d = _signed(value=d)
    # Use the original primal's branch and ratio; do not reclassify via p*m.
    z = jnp.where(context.near_geometric, context.deviation_ratio, 0.0)
    phi = _signed(value=_phi(argument=z))
    near_exponent = _join(
        parts=(
            _SignedLog(
                log_c
                + signed_dp.logabs
                + log_w
                - log_z
                + 2.0 * signed_d.logabs
                + _log_psi(argument=u)
                - jnp.log1p(z),
                positive.astype(values.dtype) * signed_dp.sign,
            ),
            _SignedLog(
                log_c + signed_dp.logabs + 2.0 * mean_q.logabs + phi.logabs,
                signed_dp.sign * phi.sign,
            ),
        )
    )
    signed_k = _signed(value=context.log_moment)
    far_exponent = _join(
        parts=(
            _SignedLog(
                log_c
                + signed_dp.logabs
                + log_w
                + u
                - log_t
                + signed_d.logabs
                - signed_p.logabs,
                positive.astype(values.dtype)
                * signed_dp.sign
                * signed_d.sign
                * signed_p.sign,
            ),
            _SignedLog(
                log_c + signed_dp.logabs + signed_k.logabs - 2.0 * signed_p.logabs,
                -signed_dp.sign * signed_k.sign,
            ),
        )
    )
    unit_exponent = jax.lax.cond(
        (p == 0.0) & ~is_represented_zero(dp),
        lambda: _unit_risk_exponent(
            d=d,
            log_w=log_w,
            positive=positive,
            log_z=log_z,
            log_c=log_c,
            dp=dp,
        ),
        lambda: _SignedLog(jnp.full_like(p, -jnp.inf), jnp.zeros_like(p)),
    )
    # Each regime has a different term count; sum in signed scale before choice.
    exponent = _choose(
        predicate=p == 0.0,
        left=unit_exponent,
        right=_choose(
            predicate=context.near_geometric,
            left=_sum_signed(terms=near_exponent),
            right=_sum_signed(terms=far_exponent),
        ),
    )
    saving = _materialize(value=_sum_signed(terms=savings_terms))
    source = _materialize(
        value=_sum_signed(terms=_join(parts=(source_terms, weight_terms, exponent)))
    )
    valid &= jnp.isfinite(context.value) & (context.value > 0.0)
    valid &= jnp.isfinite(saving) & jnp.isfinite(source)
    return context.value, saving, source, valid
