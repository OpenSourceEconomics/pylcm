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

`D` is evaluated in the double-double arithmetic of `double_double`, whose
error-free transforms are exact. The only inexactness is the low-order tail each
renormalization discards, and that tail is captured exactly and accumulated into
a `dropped` bound. Two properties follow, and they are what the envelope relies
on:

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

from _lcm.egm.upper_envelope.double_double import (
    DoubleDouble,
    dd_add,
    dd_from_difference,
    dd_mul,
    dd_mul_float,
    dd_negate,
    normalizing_exponent,
)
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

    # `D` is homogeneous of degree two in the abscissae and degree one in the
    # values, so scaling each group by a power of two multiplies `D` by a
    # positive power of two and leaves its sign alone. Both scalings are exact,
    # and pulling the operands into the binade around one is what keeps the
    # products out of the range where the error-free transforms stop being
    # error-free: a determinant that would underflow to zero in the caller's
    # units is an ordinary number in these.
    abscissa_exponent = normalizing_exponent(a_x0, a_x1, b_x0, b_x1, x_query)
    value_exponent = normalizing_exponent(a_v0, a_v1, b_v0, b_v1)
    a_x0, a_x1, b_x0, b_x1, x_query = (
        jnp.ldexp(term, -abscissa_exponent)
        for term in (a_x0, a_x1, b_x0, b_x1, x_query)
    )
    a_v0, a_v1, b_v0, b_v1 = (
        jnp.ldexp(term, -value_exponent) for term in (a_v0, a_v1, b_v0, b_v1)
    )

    numerator_a = affine_numerator(x0=a_x0, x1=a_x1, v0=a_v0, v1=a_v1, x_query=x_query)
    numerator_b = affine_numerator(x0=b_x0, x1=b_x1, v0=b_v0, v1=b_v1, x_query=x_query)
    width_a = dd_from_difference(a_x1, a_x0)
    width_b = dd_from_difference(b_x1, b_x0)

    product_a = dd_mul(numerator_a, width_b)
    product_b = dd_mul(numerator_b, width_a)
    determinant = dd_add(product_a, dd_negate(product_b))

    # Normalization makes an underflowed product unreachable for representable
    # inputs, but it is the premise the certificate rests on rather than an
    # observation, so it is checked: outside the domain where `two_prod` is
    # exact, a zero is not evidence of a tie and the sign stays unresolved.
    in_domain = _product_in_transform_domain(
        numerator_a[0], width_b[0]
    ) & _product_in_transform_domain(numerator_b[0], width_a[0])
    return _certified_sign_of(determinant, finite=finite & in_domain)


def affine_numerator(
    *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
) -> DoubleDouble:
    """Return `v0*(x1 - x) + v1*(x - x0)`, the width-scaled value at `x`."""
    return dd_add(
        dd_mul_float(dd_from_difference(x1, x_query), v0),
        dd_mul_float(dd_from_difference(x_query, x0), v1),
    )


def _product_in_transform_domain(a: FloatND, b: FloatND) -> BoolND:
    """Report whether `two_prod(a, b)` stays inside its exact domain.

    Dekker's transform is exact only while the product and the splitting
    intermediates stay normal. A product that underflows to zero, or lands among
    the subnormals, silently loses the tail the certificate reads — so such a
    product must never be mistaken for an exact zero.
    """
    product = jnp.abs(a * b)
    tiny = jnp.finfo(product.dtype).tiny
    both_nonzero = (a != 0.0) & (b != 0.0)
    return jnp.isfinite(product) & (~both_nonzero | (product >= tiny))


def _certified_sign_of(value: DoubleDouble, *, finite: BoolND) -> IntND:
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
