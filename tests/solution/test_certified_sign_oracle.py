"""Every strict verdict the certificate publishes is the exact one.

A bound that is too small does not produce a failure — it produces a strict sign
that is wrong. So the property that has to be checked is soundness against
truth, not the absence of complaint: wherever `certified_margin_sign` commits to
`+1`, `-1` or `0`, exact rational arithmetic must agree. Where it abstains,
nothing is claimed and nothing is asserted.

The oracle is `fractions.Fraction` over the same stored operands. Every operand
is a power of two or a small integer, so `Fraction(float(...))` is exact and the
comparison carries no tolerance, no floating point, and none of the module's own
helpers — the two routes share only the geometry.
"""

import itertools
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    BELOW_RESOLUTION_SIGN,
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

ABSTENTIONS = (UNRESOLVED_SIGN, BELOW_RESOLUTION_SIGN)


def _dtype():
    return np.float64 if X64_ENABLED else np.float32


def _exponent_ladder():
    """Powers of two from the bottom of the normal range to the top, and near one.

    The magnitudes that matter are the ones no single scaling can serve at once,
    so the ladder reaches both ends of the format rather than clustering around
    one.
    """
    maxexp = int(np.finfo(_dtype()).maxexp) - 1
    minexp = int(np.finfo(_dtype()).minexp)
    return (minexp, minexp + 3, -20, -1, 0, 1, 20, maxexp - 3, maxexp)


def _geometries():
    """Yield `(a, b, query)` triples spanning the exponent range in both links.

    Each link is `(x0, x1, v0, v1)`. Widths are positive by construction, which
    is what the certificate requires of its caller.
    """
    dtype = _dtype()
    ladder = _exponent_ladder()
    scales = tuple(dtype(np.ldexp(1.0, power)) for power in ladder)
    abscissa_pairs = ((dtype(0.0), one) for one in scales)
    abscissa_pairs = tuple(abscissa_pairs) + tuple(
        (one, dtype(np.ldexp(float(one), 4))) for one in scales[:-4]
    )
    value_pairs = (
        *(
            (dtype(sign * float(low)), dtype(-sign * float(high)))
            for low, high in ((scales[0], scales[-1]), (scales[3], scales[4]))
            for sign in (1, -1)
        ),
        (dtype(0.0), dtype(1.0)),
        (dtype(1.0), dtype(1.0)),
    )

    for (a_x0, a_x1), (b_x0, b_x1) in itertools.product(abscissa_pairs, repeat=2):
        for a_values, b_values in itertools.product(value_pairs, repeat=2):
            a = (a_x0, a_x1, *a_values)
            b = (b_x0, b_x1, *b_values)
            lower = max(float(a_x0), float(b_x0))
            upper = min(float(a_x1), float(b_x1))
            if not lower < upper:
                continue
            for query in (
                dtype(lower),
                dtype(upper),
                dtype(lower + 0.5 * (upper - lower)),
                np.nextafter(dtype(lower), dtype(np.inf), dtype=dtype),
            ):
                if np.isfinite(query):
                    yield a, b, query


def _exact_sign(*, a, b, query):
    """Return the exact sign of `A(query) - B(query)` in rational arithmetic."""

    def numerator(link):
        x0, x1, v0, v1 = (Fraction(float(item)) for item in link)
        return v0 * (x1 - Fraction(float(query))) + v1 * (Fraction(float(query)) - x0)

    def width(link):
        return Fraction(float(link[1])) - Fraction(float(link[0]))

    determinant = numerator(a) * width(b) - numerator(b) * width(a)
    return (determinant > 0) - (determinant < 0)


def _observed_signs(cases):
    """Return the certificate's verdict for each case, evaluated as one batch."""
    columns = [
        jnp.asarray(np.asarray([case[index][field] for case in cases]))
        for index, field in itertools.product((0, 1), range(4))
    ]
    query = jnp.asarray(np.asarray([case[2] for case in cases]))
    return np.asarray(
        certified_margin_sign(
            a_x0=columns[0],
            a_x1=columns[1],
            a_v0=columns[2],
            a_v1=columns[3],
            b_x0=columns[4],
            b_x1=columns[5],
            b_v0=columns[6],
            b_v1=columns[7],
            x_query=query,
        )
    )


def _disagreements():
    """Return every geometry whose published verdict is not the exact one."""
    cases = list(_geometries())
    observed = _observed_signs(cases)
    strict = [
        (case, int(sign))
        for case, sign in zip(cases, observed, strict=True)
        if int(sign) not in ABSTENTIONS
    ]
    assert len(strict) > 100, (
        f"witness is vacuous: only {len(strict)} of {len(cases)} geometries "
        "produced a verdict to check"
    )
    return [
        (case, published, exact)
        for case, published, exact in (
            (case, published, _exact_sign(a=case[0], b=case[1], query=case[2]))
            for case, published in strict
        )
        if published != exact
    ]


def test_no_tie_is_certified_where_exact_arithmetic_finds_a_strict_sign():
    """A `0` verdict means the two lines meet, and licenses any choice between them."""
    fabricated = [item for item in _disagreements() if item[1] == 0]
    assert not fabricated, (
        f"{len(fabricated)} certified ties are strict in exact arithmetic; "
        f"first: links {fabricated[0][0][0]} and {fabricated[0][0][1]} at query "
        f"{fabricated[0][0][2]!r}, exact sign {fabricated[0][2]}"
    )


def test_no_strict_verdict_is_the_opposite_of_the_exact_one():
    """A published `+1` or `-1` states which line is higher, and must be right.

    Deciding an ordering the format itself cannot see is the certificate's
    whole purpose — a double-double exists to settle which branch is higher
    where both round to the same float — so a separation of one ulp is inside
    its contract rather than beyond it, and reversing one is a failure of the
    contract however small the level difference that follows.
    """
    inverted = [item for item in _disagreements() if item[1] in (1, -1)]
    assert not inverted, (
        f"{len(inverted)} strict verdicts have the wrong sign; first: links "
        f"{inverted[0][0][0]} and {inverted[0][0][1]} at query "
        f"{inverted[0][0][2]!r} published {inverted[0][1]}, exact "
        f"{inverted[0][2]}"
    )


def test_the_oracle_would_catch_a_wrong_strict_verdict():
    """The comparison above is able to fail, on a verdict known to be wrong.

    Without this, a run in which every geometry abstained and a run in which the
    arithmetic is sound report the same thing.
    """
    dtype = _dtype()
    a = (dtype(0.0), dtype(1.0), dtype(0.0), dtype(1.0))
    b = (dtype(0.0), dtype(1.0), dtype(1.0), dtype(0.0))
    query = dtype(0.25)
    exact = _exact_sign(a=a, b=b, query=query)
    assert exact == -1, "A is below B at a quarter of the shared span"
    assert exact != 1, "the oracle must separate the two strict verdicts"


def test_a_flattened_contribution_never_certifies_the_tie_it_did_not_earn():
    """A term the scaling drove under the format still forbids a certified tie.

    The two links here span more of the exponent range than one scaling can
    serve, so the flatter link's contribution underflows on the way to the
    determinant while the steeper link's numerator is exactly zero at this
    query. Their difference then reads as zero — and a zero whose discarded tail
    is also zero is the certificate for an exact tie, which is the one verdict
    nothing downstream questions.

    The true margin is a quarter, so the tie is fabricated. Carrying what the
    underflow discarded as a bound is what keeps the exact-tie verdict off the
    table; the certificate may abstain here, but it may not agree.
    """
    dtype = _dtype()
    top = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 2))
    a = (dtype(np.ldexp(1.0, -20)), dtype(1.0), dtype(0.25), dtype(0.25))
    b = (dtype(0.0), dtype(2.0), top, dtype(-float(top)))
    query = dtype(1.0)
    exact = _exact_sign(a=a, b=b, query=query)
    assert exact == 1, "the flat link is a quarter above the steep one here"

    observed = int(_observed_signs([(a, b, query)])[0])
    assert observed in (exact, *ABSTENTIONS), (
        f"published {observed} where exact arithmetic gives {exact}"
    )
    assert observed != 0, (
        "a certified exact tie between links a quarter apart licenses every "
        "consumer to choose freely"
    )


@pytest.mark.parametrize("case_index", [0, 1])
def test_a_link_read_at_its_own_node_is_certified_against_the_stored_value(case_index):
    """At a shared node the verdict is the ordering of the two stored values."""
    dtype = _dtype()
    huge = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1))
    a = (dtype(0.0), dtype(1.0), dtype(0.25), dtype(0.25))
    b = (dtype(0.0), dtype(1.0), huge, dtype(-huge))
    query = (dtype(0.0), dtype(1.0))[case_index]
    observed = int(_observed_signs([(a, b, query)])[0])
    assert observed == _exact_sign(a=a, b=b, query=query), (
        f"at the shared node {query!r} the certificate published {observed}"
    )
