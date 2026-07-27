r"""Certified sign of the difference of two affine value lines.

Upper-envelope construction is a sequence of *structural* decisions — which link
owns a cell, whether two links cross inside it, whether a crossing coincides with
a node. A magnitude-scaled tolerance cannot make those decisions: it is not
invariant to a common additive value level, so it masks a strictly better branch
whenever values sit on a large constant, and it promotes a below-envelope
crossing after the same shift.

This module decides them from the sign of the exact difference instead. For two
links given by their endpoints the value at `x` is

```{math}
A(x) = \frac{a_{v0}(a_{x1} - x) + a_{v1}(x - a_{x0})}{a_{x1} - a_{x0}},
```

so with positive widths the sign of `A(x) - B(x)` is the sign of the
cross-multiplied determinant

```{math}
D = N_a w_b - N_b w_a .
```

`D` is evaluated in double-double arithmetic built from error-free transforms
(Knuth's `two_sum`, Dekker's `two_prod`), which are exact. The only inexactness
is the low-order tail each renormalization discards, and that tail is captured
exactly and accumulated into a `dropped` bound. Two properties follow, and they
are what the envelope relies on:

- `dropped` is exactly zero whenever the whole evaluation was exact, so a genuine
  tie (a crossing sitting exactly on a node, or a link compared with itself) is
  *certified* rather than inferred from a threshold;
- otherwise the true determinant is within `dropped` of the computed one, so a
  sign is published only when it is certain.

Everything else — a nonzero determinant too small to separate from its own error
bound, or a non-finite input — returns `UNRESOLVED_SIGN`. That value is a
fail-loud signal, never a silently chosen branch. Callers must mask dead
candidates and zero-width links before calling: a link of zero width has no
affine value line, and this predicate does not invent one.

Correctness is the design constraint here, not throughput: one comparison costs a
few hundred flops. The evaluation is branch-free and elementwise, so it stays
`jax.jit`- and `jax.vmap`-compatible with static shapes.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, IntND

# Returned where the sign cannot be certified; callers must fail loud on it.
UNRESOLVED_SIGN: int = 2


def certified_margin_sign(
    *,
    a_x0: FloatND,
    a_x1: FloatND,
    a_v0: FloatND,
    a_v1: FloatND,
    b_x0: FloatND,
    b_x1: FloatND,
    b_v0: FloatND,
    b_v1: FloatND,
    x_query: FloatND,
) -> IntND:
    """Return the certified sign of `A(x_query) - B(x_query)`.

    `A` and `B` are the affine lines through the given endpoints, extended beyond
    them: the caller decides which cell a comparison belongs to, this predicate
    only settles the sign there.

    Args:
        a_x0: Lower endpoint abscissa of the first link.
        a_x1: Upper endpoint abscissa of the first link; must exceed `a_x0`.
        a_v0: Value of the first link at `a_x0`.
        a_v1: Value of the first link at `a_x1`.
        b_x0: Lower endpoint abscissa of the second link.
        b_x1: Upper endpoint abscissa of the second link; must exceed `b_x0`.
        b_v0: Value of the second link at `b_x0`.
        b_v1: Value of the second link at `b_x1`.
        x_query: Abscissa at which the two lines are compared.

    Returns:
        `+1` where `A` is certainly above `B`, `-1` where it is certainly below,
        `0` where the difference is certified exactly zero, and
        `UNRESOLVED_SIGN` where no sign can be certified (including any
        non-finite input).

    """
    numerator_a = _affine_numerator(x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1, x_query=x_query)
    numerator_b = _affine_numerator(x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1, x_query=x_query)
    width_a = _dd_from_difference(a_x1, a_x0)
    width_b = _dd_from_difference(b_x1, b_x0)

    determinant = _dd_add(
        _dd_mul(numerator_a, width_b),
        _dd_negate(_dd_mul(numerator_b, width_a)),
    )

    finite = (
        jnp.isfinite(a_x0)
        & jnp.isfinite(a_x1)
        & jnp.isfinite(a_v0)
        & jnp.isfinite(a_v1)
        & jnp.isfinite(b_x0)
        & jnp.isfinite(b_x1)
        & jnp.isfinite(b_v0)
        & jnp.isfinite(b_v1)
        & jnp.isfinite(x_query)
    )
    return _certified_sign_of(determinant, finite=finite)


def _affine_numerator(
    *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
) -> tuple[FloatND, FloatND, FloatND]:
    """Return `v0*(x1 - x) + v1*(x - x0)`, the width-scaled value at `x`."""
    return _dd_add(
        _dd_mul_float(_dd_from_difference(x1, x_query), v0),
        _dd_mul_float(_dd_from_difference(x_query, x0), v1),
    )


def _certified_sign_of(
    value: tuple[FloatND, FloatND, FloatND], *, finite: BoolND
) -> IntND:
    """Turn a double-double with an error bound into a certified sign."""
    high, low, dropped = value
    estimate = high + low
    epsilon = jnp.finfo(estimate.dtype).eps
    # `dropped` bounds the discarded tail; the final sum adds one more rounding.
    tolerance = dropped + epsilon * jnp.abs(estimate)
    exactly_zero = (dropped == 0.0) & (estimate == 0.0)
    unresolved = jnp.asarray(UNRESOLVED_SIGN, dtype=jnp.int32)
    sign = jnp.where(
        estimate > tolerance,
        jnp.int32(1),
        jnp.where(
            estimate < -tolerance,
            jnp.int32(-1),
            jnp.where(exactly_zero, jnp.int32(0), unresolved),
        ),
    )
    return jnp.where(finite, sign, unresolved).astype(jnp.int32)


def _two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return `(s, e)` with `a + b == s + e` exactly (Knuth)."""
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def _split(a: FloatND) -> tuple[FloatND, FloatND]:
    """Split `a` into two half-precision halves with `a == hi + lo` exactly."""
    n_mantissa = jnp.finfo(a.dtype).nmant
    factor = jnp.asarray(2.0 ** ((n_mantissa + 2) // 2) + 1.0, dtype=a.dtype)
    c = factor * a
    a_big = c - a
    a_hi = c - a_big
    return a_hi, a - a_hi


def _two_prod(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Return `(p, e)` with `a * b == p + e` exactly (Dekker)."""
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    error = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, error


def _dd_from_difference(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND, FloatND]:
    """Return the exact difference `a - b` as a double-double."""
    high, low = _two_sum(a, -b)
    return high, low, jnp.zeros_like(high)


def _dd_negate(
    value: tuple[FloatND, FloatND, FloatND],
) -> tuple[FloatND, FloatND, FloatND]:
    """Return the negation of a double-double, preserving its error bound."""
    high, low, dropped = value
    return -high, -low, dropped


def _dd_add_float(
    value: tuple[FloatND, FloatND, FloatND], addend: FloatND
) -> tuple[FloatND, FloatND, FloatND]:
    """Add a plain float to a double-double, accumulating the discarded tail."""
    high, low, dropped = value
    sum_high, error_high = _two_sum(high, addend)
    low_sum, error_low = _two_sum(low, error_high)
    new_high, new_low = _two_sum(sum_high, low_sum)
    return new_high, new_low, dropped + jnp.abs(error_low)


def _dd_add(
    left: tuple[FloatND, FloatND, FloatND],
    right: tuple[FloatND, FloatND, FloatND],
) -> tuple[FloatND, FloatND, FloatND]:
    """Add two double-doubles, accumulating both error bounds and the tail."""
    left_high, left_low, left_dropped = left
    right_high, right_low, right_dropped = right
    sum_high, error_high = _two_sum(left_high, right_high)
    sum_low, error_low = _two_sum(left_low, right_low)
    low_sum, tail = _two_sum(error_high, sum_low)
    new_high, new_low = _two_sum(sum_high, low_sum)
    dropped = left_dropped + right_dropped + jnp.abs(tail) + jnp.abs(error_low)
    return new_high, new_low, dropped


def _dd_mul_float(
    value: tuple[FloatND, FloatND, FloatND], factor: FloatND
) -> tuple[FloatND, FloatND, FloatND]:
    """Multiply a double-double by a plain float."""
    high, low, dropped = value
    product_high, error_high = _two_prod(high, factor)
    product_low, error_low = _two_prod(low, factor)
    accumulated = (
        product_high,
        jnp.zeros_like(product_high),
        jnp.zeros_like(product_high),
    )
    for term in (error_high, product_low, error_low):
        accumulated = _dd_add_float(accumulated, term)
    new_high, new_low, new_dropped = accumulated
    return new_high, new_low, new_dropped + dropped * jnp.abs(factor)


def _dd_mul(
    left: tuple[FloatND, FloatND, FloatND],
    right: tuple[FloatND, FloatND, FloatND],
) -> tuple[FloatND, FloatND, FloatND]:
    """Multiply two double-doubles, accumulating both error bounds and the tail."""
    left_high, left_low, left_dropped = left
    right_high, right_low, right_dropped = right
    product, error = _two_prod(left_high, right_high)
    cross_high, cross_high_error = _two_prod(left_high, right_low)
    cross_low, cross_low_error = _two_prod(left_low, right_high)
    tail, tail_error = _two_prod(left_low, right_low)

    accumulated = (product, jnp.zeros_like(product), jnp.zeros_like(product))
    for term in (
        error,
        cross_high,
        cross_high_error,
        cross_low,
        cross_low_error,
        tail,
        tail_error,
    ):
        accumulated = _dd_add_float(accumulated, term)

    left_scale = jnp.abs(left_high) + jnp.abs(left_low)
    right_scale = jnp.abs(right_high) + jnp.abs(right_low)
    new_high, new_low, new_dropped = accumulated
    dropped = (
        new_dropped
        + left_dropped * right_scale
        + right_dropped * left_scale
        + left_dropped * right_dropped
    )
    return new_high, new_low, dropped
