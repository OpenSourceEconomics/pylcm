"""The envelope publishes a link's exact value, including in the subnormal band.

Reading a link at a query is an affine quotient of stored operands, so it has one
correct answer in the target format. These tests assert that answer bit-for-bit —
a value is either the correctly rounded one or it is wrong, and no tolerance can
express the difference between the smallest subnormal and zero.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import _line_value
from tests import conftest
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.fixture(name="dtype")
def _fixture_dtype():
    """Return the float type the suite's `--precision` selects."""
    return jnp.float64 if conftest.X64_ENABLED else jnp.float32


def _bits(value, *, dtype) -> int:
    """Return the stored bit pattern of `value` in `dtype`."""
    unsigned = np.uint64 if dtype == jnp.float64 else np.uint32
    return int(np.asarray(value, dtype=dtype).view(unsigned))


def _cancelling_link(*, dtype):
    """Return endpoints whose midpoint is the smallest subnormal.

    Both ordinates are normal and nearly opposite, so the midpoint is their sum
    halved — a quantity the format holds exactly and a floating evaluation
    cancels away.
    """
    numpy_dtype = np.float64 if dtype == jnp.float64 else np.float32
    info = np.finfo(numpy_dtype)
    smallest_subnormal = np.nextafter(numpy_dtype(0), numpy_dtype(1))
    v0 = numpy_dtype(-info.tiny)
    v1 = numpy_dtype(info.tiny + numpy_dtype(2) * smallest_subnormal)
    return v0, v1, smallest_subnormal


def test_line_value_publishes_a_subnormal_result_from_normal_endpoints(dtype):
    """A link whose read cancels to the smallest subnormal publishes it, not zero."""
    v0, v1, smallest_subnormal = _cancelling_link(dtype=dtype)

    # The endpoints must survive the cast, or the witness is a different problem.
    assert np.isfinite([v0, v1]).all()
    assert (np.asarray([v0, v1]) != 0).all()
    exact = (Fraction(float(v0)) + Fraction(float(v1))) / 2
    assert exact == Fraction(float(smallest_subnormal))

    published = _line_value(
        low=jnp.asarray(0, dtype=jnp.int32),
        high=jnp.asarray(1, dtype=jnp.int32),
        x_query=jnp.asarray(0.5, dtype=dtype),
        endog_grid=jnp.asarray([0.0, 1.0], dtype=dtype),
        ordinate=jnp.asarray([v0, v1], dtype=dtype),
    )

    assert _bits(published, dtype=dtype) == _bits(smallest_subnormal, dtype=dtype)


def test_line_value_returns_the_stored_ordinate_at_a_node(dtype):
    """Read at an endpoint, a link takes exactly the value stored there."""
    endog_grid = jnp.asarray([0.0, 4.0], dtype=dtype)
    ordinate = jnp.asarray([2.5, -7.25], dtype=dtype)

    at_low = _line_value(
        low=jnp.asarray(0, dtype=jnp.int32),
        high=jnp.asarray(1, dtype=jnp.int32),
        x_query=endog_grid[0],
        endog_grid=endog_grid,
        ordinate=ordinate,
    )
    assert _bits(at_low, dtype=dtype) == _bits(ordinate[0], dtype=dtype)


def test_line_value_is_correctly_rounded_on_an_ordinary_link(dtype):
    """An ordinary read is the exact quotient rounded once to the target format."""
    endog_grid = jnp.asarray([1.0, 3.0], dtype=dtype)
    ordinate = jnp.asarray([1.0, 2.0], dtype=dtype)
    x_query = jnp.asarray(1.75, dtype=dtype)

    grid = [Fraction(float(term)) for term in np.asarray(endog_grid)]
    values = [Fraction(float(term)) for term in np.asarray(ordinate)]
    query = Fraction(float(np.asarray(x_query)))
    exact = (values[0] * (grid[1] - query) + values[1] * (query - grid[0])) / (
        grid[1] - grid[0]
    )
    numpy_dtype = np.float64 if dtype == jnp.float64 else np.float32
    expected = np.asarray(float(exact), dtype=numpy_dtype)

    published = _line_value(
        low=jnp.asarray(0, dtype=jnp.int32),
        high=jnp.asarray(1, dtype=jnp.int32),
        x_query=x_query,
        endog_grid=endog_grid,
        ordinate=ordinate,
    )
    assert _bits(published, dtype=dtype) == _bits(expected, dtype=dtype)
