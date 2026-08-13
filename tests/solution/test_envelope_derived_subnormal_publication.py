"""A normal-input affine read may not silently flush its represented result.

The exact affine value can be a positive subnormal even when every stored operand is
normal or zero.  The certified query must publish that represented value or abstain in
all channels; finite zero is neither outcome.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    backend_flushes_subnormals,
    is_subnormal,
)
from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import X64_ENABLED


def _working_case(power: int):
    dtype = np.float64 if X64_ENABLED else np.float32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    max_exponent = 1023 if X64_ENABLED else 127
    x0 = dtype(0)
    x1 = dtype(np.ldexp(1.0, max_exponent))
    query = dtype(np.ldexp(1.0, -power))
    expected = dtype(np.ldexp(1.0, -max_exponent - power))
    assert expected > 0
    assert all(
        x == 0 or abs(x) >= np.finfo(dtype).tiny for x in (x0, x1, query, dtype(1))
    )
    return dtype, jax_dtype, x0, x1, query, expected


@pytest.mark.parametrize("segment_block_size", [0, 1])
@pytest.mark.parametrize("power_selector", ["largest", "smallest"])
def test_derived_subnormal_is_published_or_refused(segment_block_size, power_selector):
    """Both ends of the positive-subnormal output band obey the same contract."""
    mantissa_bits = 52 if X64_ENABLED else 23
    power = 0 if power_selector == "largest" else mantissa_bits - 1
    dtype, jax_dtype, x0, x1, query, expected = _working_case(power)

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([x0, x1], dtype=jax_dtype),
        policy=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        value=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        marginal=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        segment_id=jnp.asarray([dtype(0), dtype(0)]),
        x_query=jnp.asarray([query], dtype=jax_dtype),
        segment_block_size=segment_block_size,
        arithmetic="certified",
    )
    observed = tuple(np.asarray(channel)[0] for channel in (value, policy, marginal))
    exact = all(item == expected for item in observed)
    fail_loud = all(np.isnan(item) for item in observed)
    assert exact or fail_loud, f"finite wrong derived-subnormal publication: {observed}"


@pytest.mark.parametrize("segment_block_size", [0, 1])
def test_a_small_row_mate_does_not_withhold_a_representable_read(segment_block_size):
    """A channel's own small magnitude is not a reason to abstain in the others.

    Reading near an endpoint makes the far term small in every channel at once,
    but each channel's result is the sum of its own two terms. Here all three
    sums are ordinary normals, so all three are owed.
    """
    dtype = np.float64 if X64_ENABLED else np.float32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    small = dtype(np.ldexp(1.0, -980 if X64_ENABLED else -110))
    query = dtype(np.ldexp(1.0, -60 if X64_ENABLED else -20))
    assert small >= np.finfo(dtype).tiny, "the row-mate must be a stored normal"
    assert 0.0 < float(small) * float(query) < float(np.finfo(dtype).tiny), (
        "witness is vacuous unless the row-mate's magnitude, weighted by the "
        "distance to the near endpoint, lands in the subnormal band"
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        policy=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        value=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        marginal=jnp.asarray([small, small], dtype=jax_dtype),
        segment_id=jnp.asarray([dtype(0), dtype(0)]),
        x_query=jnp.asarray([query], dtype=jax_dtype),
        segment_block_size=segment_block_size,
        arithmetic="certified",
    )
    observed = tuple(np.asarray(channel)[0] for channel in (value, policy, marginal))
    assert observed == (query, query, small)


@pytest.mark.parametrize("segment_block_size", [0, 1])
def test_a_stored_subnormal_channel_is_published_or_refused(segment_block_size):
    """A channel that is constant at a subnormal is owed that constant, or NaN.

    Every read of a constant channel is the constant itself, so the represented
    result is a stored input and no arithmetic can have lost anything on the way
    to it. A backend that cannot carry a subnormal through the read owes an
    abstention; a finite zero claims the channel is zero, which it is not.
    """
    dtype = np.float64 if X64_ENABLED else np.float32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    small = dtype(np.ldexp(1.0, -1030 if X64_ENABLED else -130))
    stored = jnp.asarray([small, small], dtype=jax_dtype)
    assert bool(np.all(np.asarray(is_subnormal(stored)))), (
        "the witness needs a stored subnormal; asking `!= 0.0` cannot confirm "
        "that, because the comparison reads zero for every subnormal on a "
        "flushing backend"
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        policy=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        value=jnp.asarray([dtype(0), dtype(1)], dtype=jax_dtype),
        marginal=stored,
        segment_id=jnp.asarray([dtype(0), dtype(0)]),
        x_query=jnp.asarray([dtype(0.5)], dtype=jax_dtype),
        segment_block_size=segment_block_size,
        arithmetic="certified",
    )
    observed = tuple(np.asarray(channel)[0] for channel in (value, policy, marginal))
    if backend_flushes_subnormals(jax_dtype):
        assert all(np.isnan(item) for item in observed), (
            "a backend that cannot carry a subnormal through the read has only "
            f"one honest answer left, and it published {observed}"
        )
    else:
        assert _same_bits(observed[2], small), (
            "a backend that reads the subnormal band owes the stored value, "
            f"and it published {observed[2]!r} instead of {small!r}"
        )


def _same_bits(left, right) -> bool:
    """Report whether two floats of one dtype hold the identical bit pattern.

    Equality cannot answer this in the subnormal band: on a flushing backend
    both operands read as zero and every subnormal compares equal to every
    other one, and to zero.
    """
    unsigned = jnp.uint64 if X64_ENABLED else jnp.uint32
    jax_dtype = jnp.float64 if X64_ENABLED else jnp.float32
    as_bits = [
        int(
            np.asarray(
                jax.lax.bitcast_convert_type(
                    jnp.asarray(item, dtype=jax_dtype), unsigned
                )
            )
        )
        for item in (left, right)
    ]
    return as_bits[0] == as_bits[1]
