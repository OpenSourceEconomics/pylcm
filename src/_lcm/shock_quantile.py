"""Smooth two-normal quantiles with one common standard-normal innovation.

The bracket is bounded by the component quantiles at the same probability. Log-CDF
or log-survival comparison avoids rounding a tail probability to zero or unity.
An unresolved bracket yields NaN, which the public law's validity channel refuses.
Derivatives follow the implicit CDF equation, not the bisection branch history.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from lcm.typing import BoolND, FloatND, LoopIndex

_BISECTION_CAPACITY = 2200


def _log_density(value: FloatND) -> FloatND:
    return -0.5 * value * value - 0.5 * jnp.log(2 * jnp.pi)


def _solve(
    *,
    z: FloatND,
    p: FloatND,
    mu1: FloatND,
    sigma1: FloatND,
    mu2: FloatND,
    sigma2: FloatND,
) -> FloatND:
    z, p, mu1, sigma1, mu2, sigma2 = jnp.broadcast_arrays(
        z, p, mu1, sigma1, mu2, sigma2
    )
    first, second = mu1 + sigma1 * z, mu2 + sigma2 * z
    lower, upper = jnp.minimum(first, second), jnp.maximum(first, second)
    safe1, safe2 = (
        jnp.where(sigma1 > 0, sigma1, 1.0),
        jnp.where(sigma2 > 0, sigma2, 1.0),
    )
    lp, lq = jnp.log(p), jnp.log1p(-p)

    def comparison(q: FloatND) -> tuple[BoolND, BoolND]:
        t1, t2 = (q - mu1) / safe1, (q - mu2) / safe2
        first_above, second_above = t1 >= 0, t2 >= 0
        mass = jnp.where(first_above, p, 1 - p)
        lower_mass = mass <= 0.5  # noqa: PLR2004 - CDF/survival symmetry
        # Cancel the plateau mass against the target before adding tiny tails.
        offset, offset_sign = _log_difference(
            left=jnp.where(lower_mass, jnp.log(mass), log_ndtr(-z)),
            right=jnp.where(lower_mass, log_ndtr(z), jnp.log1p(-mass)),
        )
        tail1, tail2 = lp + log_ndtr(-jnp.abs(t1)), lq + log_ndtr(-jnp.abs(t2))
        positive = logsumexp(
            jnp.stack(
                (
                    jnp.where(offset_sign > 0, offset, -jnp.inf),
                    jnp.where(first_above, -jnp.inf, tail1),
                    jnp.where(second_above, -jnp.inf, tail2),
                )
            ),
            axis=0,
        )
        negative = logsumexp(
            jnp.stack(
                (
                    jnp.where(offset_sign < 0, offset, -jnp.inf),
                    jnp.where(first_above, tail1, -jnp.inf),
                    jnp.where(second_above, tail2, -jnp.inf),
                )
            ),
            axis=0,
        )
        between = first_above != second_above
        below_both = ~first_above & ~second_above
        log_cdf = jnp.logaddexp(lp + log_ndtr(t1), lq + log_ndtr(t2))
        log_sf = jnp.logaddexp(lp + log_ndtr(-t1), lq + log_ndtr(-t2))
        below = jnp.where(
            between,
            positive < negative,
            jnp.where(below_both, log_cdf < log_ndtr(z), log_sf > log_ndtr(-z)),
        )
        sound = jnp.where(
            between,
            ~(
                jnp.isnan(positive)
                | jnp.isnan(negative)
                | (jnp.isneginf(positive) & jnp.isneginf(negative))
            ),
            jnp.where(
                below_both,
                jnp.isfinite(log_cdf) | jnp.isfinite(log_ndtr(z)),
                jnp.isfinite(log_sf) | jnp.isfinite(log_ndtr(-z)),
            )
            & ~(jnp.isnan(log_cdf) | jnp.isnan(log_sf)),
        )
        return below, sound

    def finished(*, lo: FloatND, hi: FloatND) -> BoolND:
        return (jnp.nextafter(lo, jnp.inf) >= hi) | (p <= 0) | (p >= 1)

    def condition(carry: tuple[LoopIndex, FloatND, FloatND, BoolND]) -> BoolND:
        iteration, lo, hi, sound = carry
        return (iteration < _BISECTION_CAPACITY) & jnp.any(
            ~finished(lo=lo, hi=hi) & sound
        )

    def body(
        carry: tuple[LoopIndex, FloatND, FloatND, BoolND],
    ) -> tuple[LoopIndex, FloatND, FloatND, BoolND]:
        iteration, lo, hi, sound = carry
        delta = hi - lo
        mid = jnp.where(jnp.isfinite(delta), lo + delta * 0.5, lo * 0.5 + hi * 0.5)
        below, comparison_sound = comparison(mid)
        active = ~finished(lo=lo, hi=hi) & sound
        lo = jnp.where(active & below, mid, lo)
        hi = jnp.where(active & ~below, mid, hi)
        return iteration + 1, lo, hi, sound & (~active | comparison_sound)

    _, lo, hi, sound = jax.lax.while_loop(
        condition, body, (0, lower, upper, jnp.ones_like(lower, dtype=bool))
    )
    value = lo * 0.5 + hi * 0.5
    resolved = finished(lo=lo, hi=hi) & sound & jnp.isfinite(value)
    value = jnp.where(resolved, value, jnp.nan)
    return jnp.where(p <= 0, second, jnp.where(p >= 1, first, value))


def _log_difference(*, left: FloatND, right: FloatND) -> tuple[FloatND, FloatND]:
    """Log absolute difference and sign of two exponentials, cancelling first."""
    same = left == right
    large, small = jnp.maximum(left, right), jnp.minimum(left, right)
    magnitude = large + jnp.log(-jnp.expm1(small - large))
    return jnp.where(same, -jnp.inf, magnitude), jnp.where(
        same, 0.0, jnp.sign(left - right)
    )


# keyword-only-exempt: library-callback=jax.custom_jvp
def _quantile_primal(
    z: FloatND, p: FloatND, mu1: FloatND, sigma1: FloatND, mu2: FloatND, sigma2: FloatND
) -> FloatND:
    return _solve(z=z, p=p, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)


_quantile = jax.custom_jvp(_quantile_primal)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _quantile_jvp(
    primals: tuple[FloatND, ...], tangents: tuple[FloatND, ...]
) -> tuple[FloatND, FloatND]:
    z, p, mu1, sigma1, mu2, sigma2 = primals
    dz, dp, dm1, ds1, dm2, ds2 = tangents
    value = _quantile(*primals)
    t1, t2 = (value - mu1) / sigma1, (value - mu2) / sigma2
    ld1, ld2 = _log_density(t1) - jnp.log(sigma1), _log_density(t2) - jnp.log(sigma2)
    log_density = jnp.logaddexp(jnp.log(p) + ld1, jnp.log1p(-p) + ld2)
    r1, r2 = (
        jnp.exp(jnp.log(p) + ld1 - log_density),
        jnp.exp(jnp.log1p(-p) + ld2 - log_density),
    )
    high, low = jnp.maximum(t1, t2), jnp.minimum(t1, t2)
    upper_tail = low > 0
    log_larger = jnp.where(upper_tail, log_ndtr(-low), log_ndtr(high))
    log_smaller = jnp.where(upper_tail, log_ndtr(-high), log_ndtr(low))
    log_difference, _ = _log_difference(left=log_larger, right=log_smaller)
    probability_slope = jnp.sign(t1 - t2) * jnp.exp(log_difference - log_density)
    derivative = (
        jnp.exp(_log_density(z) - log_density) * dz
        - probability_slope * dp
        + r1 * (dm1 + t1 * ds1)
        + r2 * (dm2 + t2 * ds2)
    )
    return value, derivative


_quantile.defjvp(_quantile_jvp)


def mixture_quantile(
    *,
    z: FloatND,
    p: FloatND,
    mu1: FloatND,
    sigma1: FloatND,
    mu2: FloatND,
    sigma2: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return quantiles and convergence; smooth derivatives require positive scales.

    An inactive zero-scale component is valid for primal evaluation. Its boundary
    derivative is outside the smooth law contract and produces a nonfinite tangent,
    which a derivative consumer must reject at its numerical runtime boundary.
    """
    value = _quantile(z, p, mu1, sigma1, mu2, sigma2)
    return value, jnp.isfinite(value)
