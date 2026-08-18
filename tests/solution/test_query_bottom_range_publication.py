"""What `envelope_at_query` publishes for the winner is the winner's exact value.

Deciding who owns a query and reading what the owner is worth there are two
questions, and the certified path answers them with different arithmetic. The
owner is settled by an exact comparison of the stored operands. The published
number has to be the affine quotient of those same operands, rounded once —
not a value assembled by an arithmetic whose intermediate products can fall out
of range on the way.

Near the bottom of the format the two come apart. For a link from `(0, small)`
to `(1, 0)` read at `small`, the exact quotient is `small * (1 - small)`, which
rounds back to `small`; an arithmetic that loses the second product publishes
the neighbouring float instead. The result is finite, so nothing refuses it —
it is simply the wrong number, one ULP low, in a channel a caller will compare
against a rival that genuinely is that low.

Publishing exactly is not an aspiration here: the exact reader used as the
control below answers every case in the family. Abstaining in all three
channels together is the only alternative the contract allows.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope._exact_affine.ffi import exact_affine_read
from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)


def _dtype() -> np.dtype:
    """Return the floating format the suite is running at."""
    return np.dtype(np.float64 if jax.config.jax_enable_x64 else np.float32)


def _uint(dtype: np.dtype) -> np.dtype:
    """Return the unsigned integer type of the same width, for bit comparison."""
    return np.dtype(np.uint64 if dtype == np.float64 else np.uint32)


def _exact_link_value(*, x0, x1, v0, v1, query, dtype):
    """Round the exact affine quotient of one link to `dtype`, once."""
    numerator = Fraction(float(v0)) * (Fraction(float(x1)) - Fraction(float(query)))
    numerator += Fraction(float(v1)) * (Fraction(float(query)) - Fraction(float(x0)))
    exact = numerator / (Fraction(float(x1)) - Fraction(float(x0)))
    # float() on a Fraction is correctly rounded to nearest-even at binary64; the
    # binary32 cast that follows is a second rounding, harmless here because every
    # value in this family is exactly representable after the first.
    return dtype.type(float(exact))


def _publish(*, x0, x1, v0, v1, query, dtype):
    """Read all three channels of a single-link envelope at one query."""
    jdtype = jnp.float64 if dtype == np.float64 else jnp.float32
    endpoints = jnp.asarray([x0, x1], dtype=jdtype)
    channel = jnp.asarray([v0, v1], dtype=jdtype)
    return envelope_at_query(
        endog_grid=endpoints,
        policy=channel,
        value=channel,
        marginal=channel,
        segment_id=jnp.asarray([0.0, 0.0], dtype=jdtype),
        x_query=jnp.asarray([query], dtype=jdtype),
        arithmetic="certified",
    )


def test_the_exact_reader_answers_the_bottom_of_range_witness():
    """The control: the exact reader publishes the witness, so the format can."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    small = np.nextafter(tiny, dtype.type(np.inf), dtype=dtype)
    expected = _exact_link_value(x0=0, x1=1, v0=small, v1=0, query=small, dtype=dtype)

    published, status = exact_affine_read(
        x0=jnp.asarray(dtype.type(0)),
        x1=jnp.asarray(dtype.type(1)),
        v0=jnp.asarray(small),
        v1=jnp.asarray(dtype.type(0)),
        x_query=jnp.asarray(small),
    )

    assert int(np.asarray(status)) == 0
    assert np.asarray(published).view(_uint(dtype)) == expected.view(_uint(dtype))


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_every_channel_publishes_the_winners_exact_value_at_the_bottom_binade(channel):
    """Value, policy and marginal each carry the owner's exactly-rounded quotient."""
    dtype = _dtype()
    tiny = dtype.type(np.finfo(dtype).tiny)
    small = np.nextafter(tiny, dtype.type(np.inf), dtype=dtype)
    expected = _exact_link_value(x0=0, x1=1, v0=small, v1=0, query=small, dtype=dtype)

    got = _publish(x0=0, x1=1, v0=small, v1=0, query=small, dtype=dtype)[channel]

    assert np.asarray(got)[0].view(_uint(dtype)) == expected.view(_uint(dtype))


def test_no_finite_wrong_value_across_the_first_normal_binade():
    """Across 512 adjacent normals from `tiny`, every publication is exact or NaN."""
    dtype = _dtype()
    uint = _uint(dtype)

    values = np.empty(512, dtype=dtype)
    current = dtype.type(np.finfo(dtype).tiny)
    for index in range(512):
        current = np.nextafter(current, dtype.type(np.inf), dtype=dtype)
        values[index] = current

    got = np.array(
        [
            np.asarray(
                _publish(x0=0, x1=1, v0=value, v1=0, query=value, dtype=dtype)[0]
            )[0]
            for value in values
        ],
        dtype=dtype,
    )
    expected = np.array(
        [
            _exact_link_value(x0=0, x1=1, v0=value, v1=0, query=value, dtype=dtype)
            for value in values
        ],
        dtype=dtype,
    )

    # NaN is a permitted answer; a finite number that is not the exact one is not.
    published = ~np.isnan(got)
    wrong = np.flatnonzero(published & (got.view(uint) != expected.view(uint)))
    assert not wrong.size, (
        f"{wrong.size} finite wrong publications; first at input bit "
        f"{int(values[wrong[0]].view(uint))}: published "
        f"{int(got[wrong[0]].view(uint))}, exact {int(expected[wrong[0]].view(uint))}"
    )
