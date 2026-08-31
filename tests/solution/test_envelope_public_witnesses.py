"""Public-API witnesses for query-side envelope certification.

Every case here drives only the public `envelope_at_query`. Nothing touches an
internal symbol, so the file states a requirement on the published triple, not
on how the triple is produced, and stays valid across changes of kernel.

The oracle is exact rational arithmetic (`fractions.Fraction`), so no case can
pass vacuously against a tolerance.

All four families pass, including the one whose abscissa span overflows the
working format. The exact winner decodes the stored endpoints rather than
differencing them, so the infinite divisor that made that family unreadable
never forms.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# Block sizes exercise the dense reduction (0) and the blocked scan (2, 3),
# which reach ownership through separate code paths.
BLOCK_SIZES = (0, 2, 3)
# Several cases are built from the dtype's own extremes (`_huge`), so a float64
# arm is only meaningful where float64 exists. Under `--precision=32` JAX
# truncates a requested float64 to float32, which would silently re-run the
# float32 arm under a float64 label and place the extremes outside the range
# they were constructed for.
DTYPES = (jnp.float32, jnp.float64) if X64_ENABLED else (jnp.float32,)


def _published_triple(*, grid, value, policy, marginal, x_query, dtype, block):
    """Return the published `(value, policy, marginal)` at `x_query`."""

    def array(seq):
        return jnp.asarray(np.asarray(seq, dtype=dtype))

    published = envelope_at_query(
        endog_grid=array(grid),
        policy=array(policy),
        value=array(value),
        marginal=array(marginal),
        segment_id=array([0, 0, 1, 1]),
        x_query=array([x_query]),
        segment_block_size=block,
    )
    return tuple(float(np.asarray(entry)[0]) for entry in published)


def _huge(dtype):
    """The largest power of two that is still a finite normal in `dtype`."""
    return np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_segment_whose_value_gap_overflows_still_publishes_its_owner(*, dtype, block):
    """The winning segment is the one that is higher, at any representable gap.

    The two candidates' values differ by `2 * huge`, which is not representable.
    Ownership is a comparison, so the unrepresentable gap must not reach it.
    """
    huge = _huge(dtype)
    published = _published_triple(
        grid=[0, 1, 0, 1],
        value=[huge, -huge, 0.25, 0.25],
        policy=[1, 1, 0, 0],
        marginal=[20, 20, 10, 10],
        x_query=dtype(0.75),
        dtype=dtype,
        block=block,
    )
    assert published == (0.25, 0.0, 10.0)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_segment_whose_abscissa_span_overflows_still_publishes_its_owner(
    *, dtype, block
):
    """`x1 - x0` is `+inf` here while both endpoints are finite normals.

    The certified winner decodes the stored abscissae exactly rather than forming
    their difference in the working format, so a span too wide to represent never
    becomes the `inf` divisor that would leave this link unreadable.
    """
    huge = _huge(dtype)
    published = _published_triple(
        grid=[-huge, huge, -huge, huge],
        value=[0.5, -0.5, 0.25, 0.25],
        policy=[1, 1, 0, 0],
        marginal=[20, 20, 10, 10],
        x_query=dtype(0.0),
        dtype=dtype,
        block=block,
    )
    assert published == (0.25, 0.0, 10.0)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_subnormal_interpolation_fraction_still_moves_the_outputs(*, dtype, block):
    """The fraction underflows, but `fraction * span` is a finite normal.

    Exact result is one ulp of 1; a materialized fraction publishes 0 instead.
    """
    one = dtype(1.0)
    wide = np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 2)
    x_query = np.nextafter(one, dtype(np.inf), dtype=dtype)
    exact = (
        (Fraction(float(x_query)) - Fraction(float(one)))
        * Fraction(float(wide))
        / (Fraction(float(wide)) - Fraction(float(one)))
    )
    expected = float(dtype(float(exact)))
    assert expected != 0.0, "witness is vacuous if the exact result is zero"

    published = _published_triple(
        grid=[one, wide, one, wide],
        value=[1, 1, 0, 0],
        policy=[0, wide, 99, 99],
        marginal=[0, wide, 77, 77],
        x_query=x_query,
        dtype=dtype,
        block=block,
    )
    assert published == (1.0, expected, expected)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_opposite_signed_outputs_publish_their_finite_midpoint(*, dtype, block):
    """`right - left` is `-inf` here while the exact midpoint is exactly 0."""
    huge = _huge(dtype)
    working = np.dtype(jnp.dtype(dtype))
    with np.errstate(over="ignore"):
        naive = working.type(-huge) - working.type(huge)
    assert not np.isfinite(naive), (
        "witness is vacuous if the naive difference is representable"
    )

    published = _published_triple(
        grid=[0, 1, 0, 1],
        value=[1, 1, 0, 0],
        policy=[huge, -huge, 99, 99],
        marginal=[huge, -huge, 77, 77],
        x_query=dtype(0.5),
        dtype=dtype,
        block=block,
    )
    assert published == (1.0, 0.0, 0.0)
