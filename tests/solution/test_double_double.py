"""Contracts of the double-double arithmetic the upper envelope decides on.

The bound a division reports is not decoration: the handover placement publishes
a state when the bound fits inside one, so a bound that is merely conservative
withholds rows whose crossing was located exactly.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.double_double import (
    dd_quotient_bounded,
    normalizing_exponent,
)
from tests.conftest import X64_ENABLED


def _dtype():
    """The jax dtype matching the configured working precision."""
    return jnp.float64 if X64_ENABLED else jnp.float32


def _as_double_double(value: float):
    """A double-double holding one exactly representable float."""
    high = jnp.asarray(value, dtype=_dtype())
    zero = jnp.zeros((), dtype=_dtype())
    return high, zero, zero


def test_exact_division_reports_a_bound_of_zero() -> None:
    """A division that reproduces its numerator exactly is reported as exact.

    Two links meeting at a node put the crossing on a representable abscissa, so
    this is the ordinary case rather than a lucky one, and the consumer asking
    which side of that abscissa the truth falls on has to be told it is neither.
    """
    high, low, bound = dd_quotient_bounded(
        _as_double_double(21.0), _as_double_double(3.0)
    )
    assert float(bound) == 0.0
    assert float(high) == 7.0
    assert float(low) == 0.0


def test_inexact_division_reports_a_bound_that_holds() -> None:
    """A quotient with no representable value is bounded by what it reports.

    `1 / 3` is not representable at any binary precision, so the pair carries a
    residue and the reported bound has to cover the distance to the true value.
    """
    numerator, denominator = 1.0, 3.0
    high, low, bound = dd_quotient_bounded(
        _as_double_double(numerator), _as_double_double(denominator)
    )
    error = abs(
        Fraction(numerator) / Fraction(denominator)
        - Fraction(float(high))
        - Fraction(float(low))
    )
    assert float(bound) > 0.0
    assert error <= Fraction(float(bound))


def test_a_reported_bound_covers_the_error_across_scales() -> None:
    """Every division's reported bound covers its own distance to the truth.

    Swept across magnitudes and signs, because the bound is referred back
    through the denominator and so travels with the operands' scale.
    """
    rng = np.random.default_rng(seed=0)
    exponents = rng.integers(-30, 31, 500)
    numerators = (rng.uniform(-2.0, 2.0, 500) * 2.0**exponents).astype(float)
    denominators = (rng.choice([-1.0, 1.0], 500) * rng.uniform(0.5, 2.0, 500)).astype(
        float
    )

    uncovered = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        high, low, bound = dd_quotient_bounded(
            _as_double_double(numerator), _as_double_double(denominator)
        )
        if not np.isfinite(float(high)):
            continue
        error = abs(
            Fraction(float(jnp.asarray(numerator, dtype=_dtype())))
            / Fraction(float(jnp.asarray(denominator, dtype=_dtype())))
            - Fraction(float(high))
            - Fraction(float(low))
        )
        if error > Fraction(float(bound)):
            uncovered.append((numerator, denominator))
    assert uncovered == []


@pytest.mark.parametrize("unusable", [0.0, float("inf"), -float("inf"), float("nan")])
def test_a_group_with_no_usable_magnitude_scales_by_one(unusable: float) -> None:
    """`normalizing_exponent` returns `0` when no term has a magnitude to normalize.

    The exponent is what a caller scales by, so the answer for a group that holds
    nothing to scale has to be the one that leaves it alone. Zero and non-finite
    terms are ignored, and a group made only of them is that case.
    """
    term = jnp.asarray(unusable, dtype=_dtype())

    assert int(normalizing_exponent(term, term)) == 0
