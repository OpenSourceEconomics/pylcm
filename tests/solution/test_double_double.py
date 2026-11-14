"""Contracts of the double-double arithmetic the upper envelope decides on."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.cell_hull import hull_owners
from _lcm.egm.upper_envelope.double_double import (
    two_prod,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED


def _dtype():
    """The jax dtype matching the configured working precision."""
    return jnp.float64 if X64_ENABLED else jnp.float32


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
