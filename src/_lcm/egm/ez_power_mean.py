"""Original EZ continuation primal with its missing unit-risk exponent direction.

The correction reuses the existing scaled pair variance. Tangents enter only
through ordinary multiplication, so first-order reverse mode can transpose it.
Exceptional coefficient scales use a private first-order linear transport;
unresolved multi-scale cancellation remains a separate required frontier.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.ce_scaled_linear import join_ce_direction
from _lcm.egm.current_iid_power_math import _materialize, _SignedLog
from _lcm.egm.current_iid_power_mean import _unit_risk_exponent
from _lcm.power_mean import _anchored_power_mean, weighted_power_mean
from _lcm.probability import (
    _IntegerTangent,
    is_below_smallest_normal,
    is_represented_zero,
    log_of_nonnegative,
    rescaled_lottery_weights,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarBool, ScalarFloat


def continuation_power_mean(
    *, values: FloatND, weights: FloatND, exponent: ScalarFloat, shifts: IntND
) -> FloatND:
    """Keep original continuation arithmetic and repair only its unit-risk AD."""
    return _mean_with_unit_direction(values, weights, exponent, shifts)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _original_mean(
    values: FloatND, weights: FloatND, exponent: ScalarFloat, shifts: IntND
) -> FloatND:
    return weighted_power_mean(
        values=values, weights=weights, exponent=exponent, shifts=shifts
    )


def _unit_exponent_data(
    *, values: FloatND, weights: FloatND, exponent: ScalarFloat, shifts: IntND
) -> tuple[FloatND, FloatND, FloatND]:
    """Compute C*Var(log V)/2 as a primal-only, transposable tangent slope."""
    context = _anchored_power_mean(
        values=values, weights=weights, exponent=exponent, shifts=shifts
    )
    lifted = rescaled_lottery_weights(weights=weights)
    positive = ~is_represented_zero(lifted)
    safe_values = jnp.where(positive, values, 1.0)
    log_v = log_of_nonnegative(safe_values)
    centered = log_v - context.anchor[..., None]
    log_w = log_of_nonnegative(lifted) - shifts * jnp.log(
        jnp.asarray(2.0, dtype=values.dtype)
    )
    log_w = jnp.broadcast_to(log_w, centered.shape)
    positive = jnp.broadcast_to(positive, centered.shape)
    needed_valid = jnp.all(
        ~positive | (jnp.isfinite(safe_values) & (safe_values > 0.0)), axis=-1
    )
    row_shape = centered.shape[:-1]
    width = centered.shape[-1]

    # keyword-only-exempt: library-callback=jax.vmap
    def row(
        d: FloatND,
        weights_log: FloatND,
        live: BoolND,
        log_mass: ScalarFloat,
        value: ScalarFloat,
        valid: ScalarBool,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
        scaled = _unit_risk_exponent(
            d=d,
            log_w=weights_log,
            positive=live,
            log_z=log_mass,
            log_c=log_of_nonnegative(value),
            dp=jnp.ones_like(value),
        )
        valid &= jnp.isfinite(log_mass) & jnp.isfinite(value) & (value > 0.0)
        info = jnp.finfo(d.dtype)
        normal = (
            (scaled.sign == 1.0)
            & jnp.isfinite(scaled.logabs)
            & (scaled.logabs >= jnp.log(jnp.asarray(info.tiny, dtype=d.dtype)))
            & (scaled.logabs <= jnp.log(jnp.asarray(info.max, dtype=d.dtype)))
        )
        zero = (scaled.sign == 0.0) & (scaled.logabs == -jnp.inf)
        # vmap may evaluate both cond branches. Bound the actual materializer
        # input first: refused/zero rows use the harmless normal unit value.
        safe = _SignedLog(
            jnp.where(valid & normal, scaled.logabs, jnp.zeros_like(value)),
            jnp.ones_like(value),
        )
        coefficient = _materialize(value=safe)
        retained = (
            jnp.isfinite(coefficient)
            & (coefficient > 0.0)
            & ~is_below_smallest_normal(coefficient)
        )
        return (
            jnp.where(
                valid & zero,
                jnp.zeros_like(value),
                jnp.where(valid & normal & retained, coefficient, jnp.nan),
            ),
            jnp.where(valid, scaled.logabs, jnp.nan),
            jnp.where(valid, scaled.sign, jnp.nan),
        )

    coefficients, logabs, sign = jax.vmap(row)(
        centered.reshape((-1, width)),
        log_w.reshape((-1, width)),
        positive.reshape((-1, width)),
        jnp.broadcast_to(context.log_mass, row_shape).reshape(-1),
        jnp.broadcast_to(context.value, row_shape).reshape(-1),
        needed_valid.reshape(-1),
    )
    return (
        coefficients.reshape(row_shape),
        logabs.reshape(row_shape),
        sign.reshape(row_shape),
    )


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _mean_jvp(
    primals: tuple[FloatND, FloatND, ScalarFloat, IntND],
    tangents: tuple[FloatND, FloatND, ScalarFloat, _IntegerTangent],
) -> tuple[FloatND, FloatND]:
    value, original_direction = jax.jvp(_original_mean, primals, tangents)
    values, weights, exponent, shifts = primals
    _values_dot, _weights_dot, exponent_dot, _shifts_dot = tangents
    coefficient, logabs, sign = jax.lax.cond(
        exponent == 0.0,
        lambda: _unit_exponent_data(
            values=values, weights=weights, exponent=exponent, shifts=shifts
        ),
        lambda: (jnp.zeros_like(value), jnp.zeros_like(value), jnp.zeros_like(value)),
    )
    ordinary = jnp.isfinite(coefficient)
    exceptional = join_ce_direction(
        logabs=jnp.where(ordinary, 0.0, logabs),
        sign=jnp.where(ordinary, 1.0, sign),
        base=jnp.where(ordinary, 0.0, original_direction),
        direction=jnp.where(ordinary, 0.0, exponent_dot),
    )
    ordinary_coefficient = jnp.where(ordinary, coefficient, 0.0)
    return value, jnp.where(
        ordinary, original_direction + ordinary_coefficient * exponent_dot, exceptional
    )


# Construct by call: beartype's method-only import hook must retain defjvp.
_mean_with_unit_direction = jax.custom_jvp(_original_mean)
_mean_with_unit_direction.defjvp(_mean_jvp)
