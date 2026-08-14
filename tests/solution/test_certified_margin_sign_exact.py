"""The certified margin sign is the exact ordering of the two stored lines.

Every case here is a structural predicate — which of two affine links is above
the other at a query — so nothing is asserted to a tolerance. The expected sign
is computed from the operands with `fractions.Fraction`, which is exact and
shares no arithmetic with the implementation.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from tests import conftest


@pytest.fixture(name="dtype")
def _fixture_dtype():
    """Return the float type the suite's `--precision` selects."""
    return jnp.float64 if conftest.X64_ENABLED else jnp.float32


# Links whose ordering the floating determinant cannot settle: each pairs a link
# spanning the bottom of the exponent range with one of ordinary magnitude, so
# every difference formed from the small link underflows while the ordering
# itself is an ordinary, decidable fact. Given as (x0, x1, v0, v1) per link plus
# the query, in float32 units.
_UNDERFLOWING_PAIRS_F32 = [
    ((0.0, 9.403955e-38, 0.0, 1.0), (0.0, 1.0, 0.5, -1.0), 4.7019774e-38),
    ((0.0, 1.0, 0.5, -1.0), (0.0, 9.403955e-38, 0.0, 1.0), 4.7019774e-38),
    ((0.0, 2.0, 0.0, 1.0), (0.0, 2.1267648e37, 0.5, -1.0), 1.0),
    (
        (0.0, 2.0, -1.1754944e-38, 1.7014118e38),
        (1.1754944e-38, 1.880791e-37, 1.0, 1.0),
        1.1754945e-38,
    ),
    (
        (0.0, 2.0, 0.5, -1.0),
        (1.1754944e-38, 1.880791e-37, 0.0, 1.0),
        9.991702e-38,
    ),
]

# The same shapes one binade family down, in float64 units.
_UNDERFLOWING_PAIRS_F64 = [
    (
        (0.0, 1.7800590868057611e-307, 0.0, 1.0),
        (0.0, 1.0, 0.5, -1.0),
        8.900295434028806e-308,
    ),
    (
        (0.0, 1.0, 0.5, -1.0),
        (0.0, 1.7800590868057611e-307, 0.0, 1.0),
        8.900295434028806e-308,
    ),
    ((0.0, 2.0, 0.0, 1.0), (0.0, 1.1235582092889474e307, 0.5, -1.0), 1.0),
    (
        (0.0, 2.0, -2.2250738585072014e-308, 8.98846567431158e307),
        (2.2250738585072014e-308, 3.5601181736115222e-307, 1.0, 1.0),
        2.225073858507202e-308,
    ),
    (
        (0.0, 2.0, 0.5, -1.0),
        (2.2250738585072014e-308, 3.5601181736115222e-307, 0.0, 1.0),
        1.891312779731121e-307,
    ),
]


def _exact_sign(*, link_a, link_b, x_query, dtype) -> int:
    """Return the sign of `A(q) - B(q)` from exact rational arithmetic.

    Every operand is rounded to `dtype` *before* becoming a `Fraction`, so the
    oracle reads the same stored numbers the kernel is handed. Taking the
    fraction of the Python literal instead would compare two different problems
    whenever the literal is not representable in the narrower format.
    """
    values = []
    for x0, x1, v0, v1 in (link_a, link_b):
        query = _exact(x_query, dtype=dtype)
        numerator = _exact(v0, dtype=dtype) * (
            _exact(x1, dtype=dtype) - query
        ) + _exact(v1, dtype=dtype) * (query - _exact(x0, dtype=dtype))
        values.append(numerator / (_exact(x1, dtype=dtype) - _exact(x0, dtype=dtype)))
    difference = values[0] - values[1]
    return (difference > 0) - (difference < 0)


def _exact(term, *, dtype) -> Fraction:
    """Return the exact value of `term` after rounding it to `dtype`."""
    return Fraction(float(np.asarray(term, dtype=dtype)))


def _as_dtype(link, *, dtype):
    return [jnp.asarray(term, dtype=dtype) for term in link]


@pytest.mark.parametrize("case_index", range(5))
def test_certified_margin_sign_orders_links_that_underflow(case_index, dtype):
    """Links whose differences underflow are still ordered by their exact values."""
    pairs = _UNDERFLOWING_PAIRS_F32 if dtype == jnp.float32 else _UNDERFLOWING_PAIRS_F64
    link_a, link_b, x_query = pairs[case_index]
    expected = _exact_sign(link_a=link_a, link_b=link_b, x_query=x_query, dtype=dtype)
    # The witnesses are chosen for their arithmetic, so a case that degenerated
    # on the way into the narrower format would test nothing.
    assert expected != 0

    a_x0, a_x1, a_v0, a_v1 = _as_dtype(link_a, dtype=dtype)
    b_x0, b_x1, b_v0, b_v1 = _as_dtype(link_b, dtype=dtype)

    got = certified_margin_sign(
        a_x0=a_x0,
        a_x1=a_x1,
        a_v0=a_v0,
        a_v1=a_v1,
        b_x0=b_x0,
        b_x1=b_x1,
        b_v0=b_v0,
        b_v1=b_v1,
        x_query=jnp.asarray(x_query, dtype=dtype),
    )

    assert int(got) == expected


def test_certified_margin_sign_reports_a_tie_only_for_one_line(dtype):
    """A tie is published only where the two rational lines are exactly equal."""
    shared = _as_dtype((0.0, 2.0, 1.0, 3.0), dtype=dtype)
    got = certified_margin_sign(
        a_x0=shared[0],
        a_x1=shared[1],
        a_v0=shared[2],
        a_v1=shared[3],
        b_x0=shared[0],
        b_x1=shared[1],
        b_v0=shared[2],
        b_v1=shared[3],
        x_query=jnp.asarray(0.75, dtype=dtype),
    )
    assert int(got) == 0


def test_certified_margin_sign_is_unresolved_on_a_nonfinite_operand(dtype):
    """A non-finite operand yields the unresolved code, never a sign."""
    link = _as_dtype((0.0, 2.0, 1.0, 3.0), dtype=dtype)
    got = certified_margin_sign(
        a_x0=link[0],
        a_x1=link[1],
        a_v0=link[2],
        a_v1=jnp.asarray(np.inf, dtype=dtype),
        b_x0=link[0],
        b_x1=link[1],
        b_v0=link[2],
        b_v1=link[3],
        x_query=jnp.asarray(0.75, dtype=dtype),
    )
    assert int(got) == UNRESOLVED_SIGN
