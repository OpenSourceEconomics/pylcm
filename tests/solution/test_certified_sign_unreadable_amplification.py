"""Unreadable link distances may not support a strict certified owner.

A scaled distance can flush to zero while its omitted contribution is amplified by a
large endpoint value.  The exact represented-input order must then be certified
correctly or refused, never replaced by the strict sign of the truncated determinant.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import X64_ENABLED


def _working_case():
    dtype = np.float64 if X64_ENABLED else np.float32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    x_exponent, value_exponent = (1020, 1023) if X64_ENABLED else (120, 127)
    one = dtype(1.0)
    query = np.nextafter(one, dtype(2.0), dtype=dtype)
    a_x1 = dtype(np.ldexp(1.0, x_exponent))
    a_v1 = dtype(np.ldexp(1.0, value_exponent))
    return dtype, jax_dtype, one, query, a_x1, a_v1


def _fraction(value) -> Fraction:
    return Fraction(float(value))


def _affine(*, x0, x1, v0, v1, query) -> Fraction:
    return (
        _fraction(v0) * (_fraction(x1) - _fraction(query))
        + _fraction(v1) * (_fraction(query) - _fraction(x0))
    ) / (_fraction(x1) - _fraction(x0))


@pytest.mark.parametrize("swap", [False, True])
def test_an_unreadable_distance_never_supports_the_wrong_strict_sign(swap):
    """The exact sign is returned, or the certificate visibly abstains."""
    dtype, jax_dtype, one, query, a_x1, a_v1 = _working_case()
    info = np.finfo(dtype)
    a = (one, a_x1, one, a_v1)
    b = (dtype(0), dtype(2), dtype(info.tiny), dtype(2))
    exact_a = _affine(x0=a[0], x1=a[1], v0=a[2], v1=a[3], query=query)
    exact_b = _affine(x0=b[0], x1=b[1], v0=b[2], v1=b[3], query=query)
    assert exact_a > exact_b
    if swap:
        a, b = b, a
        expected = -1
    else:
        expected = 1

    def compare(*args):
        return certified_margin_sign(
            a_x0=args[0],
            a_x1=args[1],
            a_v0=args[2],
            a_v1=args[3],
            b_x0=args[4],
            b_x1=args[5],
            b_v0=args[6],
            b_v1=args[7],
            x_query=args[8],
        )

    args = tuple(jnp.asarray(x, dtype=jax_dtype) for x in (*a, *b, query))
    observed = int(jax.jit(compare)(*args))
    assert observed in (expected, UNRESOLVED_SIGN)


@pytest.mark.parametrize("segment_block_size", [0, 1])
def test_unreadable_distance_publishes_the_winner_or_nan(segment_block_size):
    """Dense and blocked routes may not publish the finite losing triple."""
    dtype, jax_dtype, one, query, a_x1, a_v1 = _working_case()
    info = np.finfo(dtype)
    exact_a = _affine(x0=one, x1=a_x1, v0=one, v1=a_v1, query=query)
    exact_b = _affine(
        x0=dtype(0),
        x1=dtype(2),
        v0=dtype(info.tiny),
        v1=dtype(2),
        query=query,
    )
    assert exact_a > exact_b

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(0), dtype(2), one, a_x1], dtype=jax_dtype),
        policy=jnp.asarray([dtype(0), dtype(0), dtype(1), dtype(1)], dtype=jax_dtype),
        value=jnp.asarray([dtype(info.tiny), dtype(2), one, a_v1], dtype=jax_dtype),
        marginal=jnp.asarray(
            [dtype(22), dtype(22), dtype(11), dtype(11)], dtype=jax_dtype
        ),
        segment_id=jnp.asarray([dtype(0), dtype(0), dtype(1), dtype(1)]),
        x_query=jnp.asarray([query], dtype=jax_dtype),
        segment_block_size=segment_block_size,
        arithmetic="certified",
    )
    observed = tuple(
        float(np.asarray(channel)[0]) for channel in (value, policy, marginal)
    )
    exact = observed[0] == dtype(float(exact_a)) and observed[1:] == (1.0, 11.0)
    fail_loud = all(np.isnan(item) for item in observed)
    assert exact or fail_loud, f"finite losing publication: {observed}"
