"""Exponent-framed arithmetic for numerically safe EGM geometry.

The helpers in this module form differences without overflowing when finite
operands lie in opposite top binades. They are used by the live NB-EGM
interpolation path; envelope reference implementations belong in the tests.
"""

import jax
import jax.numpy as jnp

from lcm.typing import FloatND


def _dyadic_parts(x: FloatND) -> tuple[FloatND, jax.Array]:
    """Represent ``x`` as an exact mantissa and integer exponent pair."""
    mantissa, exponent = jnp.frexp(x)
    usable = jnp.isfinite(x) & (x != 0)
    return mantissa, jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _two_diff(*, a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return ``a - b`` and its exact rounding residual."""
    difference = a - b
    transfer = difference - a
    return difference, (a - (difference - transfer)) - (b + transfer)


def binade_exponent(magnitude: FloatND) -> jax.Array:
    """Return the integer exponent of a positive finite magnitude's binade."""
    _, exponent = jnp.frexp(magnitude)
    usable = (magnitude > 0) & jnp.isfinite(magnitude)
    return jnp.where(usable, exponent, jnp.zeros_like(exponent))


def framed_difference(*, a: FloatND, b: FloatND) -> tuple[FloatND, FloatND, jax.Array]:
    """Return ``a - b`` as head and tail in a shared integer exponent frame."""
    mantissa_a, exponent_a = _dyadic_parts(a)
    mantissa_b, exponent_b = _dyadic_parts(b)
    exponent = jnp.maximum(exponent_a, exponent_b)
    head, tail = _two_diff(
        a=jnp.ldexp(mantissa_a, exponent_a - exponent),
        b=jnp.ldexp(mantissa_b, exponent_b - exponent),
    )
    return head, tail, exponent
