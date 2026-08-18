"""Contracts of the double-double arithmetic the upper envelope decides on.

The bound a division reports is not decoration: the handover placement publishes
a state when the bound fits inside one, so a bound that is merely conservative
withholds rows whose crossing was located exactly.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.cell_hull import hull_owners
from _lcm.egm.upper_envelope.double_double import (
    dd_quotient_bounded,
    normalizing_exponent,
    two_prod,
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


def test_a_product_is_exact_up_to_the_top_of_the_range() -> None:
    """`two_prod` reproduces its product exactly for every finite operand.

    Dekker's split multiplies its operand by roughly the square root of the
    format's precision, which overflows near the top of the range even where the
    product it serves is an ordinary finite number. The tail it returns is then
    not merely inaccurate but `nan`, and the poison spreads through every
    certificate built on it — so the split has to hold wherever the product does,
    not merely wherever the split's own intermediates happen to fit.
    """
    dtype = _dtype()
    huge = np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1)
    left, right = 0.25, huge

    high, low = two_prod(
        jnp.asarray(left, dtype=dtype), jnp.asarray(right, dtype=dtype)
    )
    exact = Fraction(left) * Fraction(float(jnp.asarray(right, dtype=dtype)))

    assert np.isfinite(float(low)), "the split overflowed and poisoned the tail"
    assert Fraction(float(high)) + Fraction(float(low)) == exact


def test_a_product_is_exact_down_to_the_bottom_of_the_range() -> None:
    """`two_prod` is exact where the split's lower half lands among the subnormals.

    Splitting an operand just above the smallest normal puts the lower half of
    its significand below the smallest normal. That half is still a real part of
    the operand, and against a large second operand its contribution is an
    ordinary number — so a transform that loses it returns a decomposition that
    is not the product, while reporting nothing.

    The cell envelope reads its operands exactly and does not depend on this, but
    `query.py` still forms its products here, so the primitive owes exactness on
    its own account rather than on its busiest caller's.
    """
    dtype = jnp.float64 if X64_ENABLED else jnp.float32
    info = np.finfo(np.float64 if X64_ENABLED else np.float32)
    small = float(np.nextafter(info.tiny, np.float64(np.inf)))
    large = float(np.ldexp(1.0, int(info.maxexp) - 1))

    left = jnp.asarray(small, dtype=dtype)
    right = jnp.asarray(large, dtype=dtype)
    high, low = two_prod(left, right)
    exact = Fraction(float(left)) * Fraction(float(right))

    assert np.isfinite(float(low)), "the split's lower half poisoned the tail"
    assert Fraction(float(high)) + Fraction(float(low)) == exact


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_cell_hull_resolves_a_handover_at_the_bottom_of_the_range() -> None:
    """The production owner walk does not inherit Dekker's subnormal split limit.

    The incoming line is below at zero and above immediately afterwards. Its
    exact crossing rounds to a subnormal state whose stored bit pattern is 413.
    A floating error-free transform can lose that state while splitting the
    endpoint; the cell-level exact kernel must instead publish it bit-for-bit.
    """
    numpy_dtype = np.float64 if X64_ENABLED else np.float32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    uint_dtype = np.uint64 if X64_ENABLED else np.uint32
    event = np.asarray(413, dtype=uint_dtype).view(numpy_dtype)

    solve_cell = jax.jit(
        lambda grid, value: hull_owners(
            left=jnp.asarray(0.0, dtype=jax_dtype),
            right=jnp.asarray(1.0, dtype=jax_dtype),
            live=jnp.asarray([True, True]),
            low=jnp.asarray([0, 2], dtype=jnp.int32),
            high=jnp.asarray([1, 3], dtype=jnp.int32),
            endog_grid=grid,
            value=value,
            max_runs=2,
        )
    )
    bounds, owners, unresolved = solve_cell(
        jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jax_dtype),
        jnp.asarray([0.0, 0.0, -event, 1.0], dtype=jax_dtype),
    )

    assert not bool(unresolved)
    np.testing.assert_array_equal(np.asarray(owners), np.asarray([0, 1]))
    assert int(np.asarray(bounds[1]).view(uint_dtype)) == 413
