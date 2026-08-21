"""Normal endpoints may not lose their ordering through subnormal distances.

The per-link scaling in ``certified_margin_sign`` is algebraically sign-preserving,
but it is useful only if the three link distances reach it.  On a flush-to-zero
backend, adjacent *normal* floats have positive subnormal differences.  Forming
those differences first and scaling them second turns a live link into three exact
zeros and can certify a tie that does not exist.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _working_dtypes():
    return (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)


def _adjacent_normal_geometry(dtype):
    x0 = dtype(np.finfo(dtype).tiny)
    query = np.nextafter(x0, dtype(np.inf), dtype=dtype)
    x1 = np.nextafter(query, dtype(np.inf), dtype=dtype)
    assert x0 < query < x1
    assert all(abs(float(x)) >= float(np.finfo(dtype).tiny) for x in (x0, query, x1))
    assert query - x0 > 0
    assert x1 - query > 0
    return x0, query, x1


def _raised_value(dtype):
    out = dtype(0.75)
    for _ in range(64):
        out = np.nextafter(out, dtype(np.inf), dtype=dtype)
    return out


@pytest.mark.parametrize("narrow_is_higher", [False, True])
def test_normal_endpoints_with_subnormal_distances_never_certify_a_false_tie(
    narrow_is_higher,
):
    """The strict represented-input ordering is certified or refused, never tied."""
    dtype, jax_dtype = _working_dtypes()
    x0, query, x1 = _adjacent_normal_geometry(dtype)
    lower = dtype(0.75)
    higher = _raised_value(dtype)
    wide_value, narrow_value = (lower, higher) if narrow_is_higher else (higher, lower)
    expected = -1 if narrow_is_higher else 1

    sign = int(
        certified_margin_sign(
            a_x0=jnp.asarray(dtype(-1), dtype=jax_dtype),
            a_x1=jnp.asarray(dtype(1), dtype=jax_dtype),
            a_v0=jnp.asarray(wide_value, dtype=jax_dtype),
            a_v1=jnp.asarray(wide_value, dtype=jax_dtype),
            b_x0=jnp.asarray(x0, dtype=jax_dtype),
            b_x1=jnp.asarray(x1, dtype=jax_dtype),
            b_v0=jnp.asarray(narrow_value, dtype=jax_dtype),
            b_v1=jnp.asarray(narrow_value, dtype=jax_dtype),
            x_query=jnp.asarray(query, dtype=jax_dtype),
        )
    )

    assert sign in (expected, UNRESOLVED_SIGN)


@pytest.mark.parametrize("segment_block_size", [0, 1])
def test_query_publishes_the_higher_normal_link_or_abstains(segment_block_size):
    """A lost distance may not turn a strict winner into a finite losing triple."""
    dtype, jax_dtype = _working_dtypes()
    x0, query, x1 = _adjacent_normal_geometry(dtype)
    lower = dtype(0.75)
    higher = _raised_value(dtype)

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(-1), dtype(1), x0, x1], dtype=jax_dtype),
        policy=jnp.asarray([dtype(0), dtype(0), dtype(1), dtype(1)], dtype=jax_dtype),
        value=jnp.asarray([lower, lower, higher, higher], dtype=jax_dtype),
        marginal=jnp.asarray(
            [dtype(22), dtype(22), dtype(11), dtype(11)], dtype=jax_dtype
        ),
        segment_id=jnp.asarray(
            [dtype(0), dtype(0), dtype(1), dtype(1)], dtype=jax_dtype
        ),
        x_query=jnp.asarray([query], dtype=jax_dtype),
        segment_block_size=segment_block_size,
        arithmetic="certified",
    )
    observed = tuple(
        float(np.asarray(channel)[0]) for channel in (value, policy, marginal)
    )
    exact = observed == (float(higher), 1.0, 11.0)
    fail_loud = all(np.isnan(item) for item in observed)

    assert exact or fail_loud, f"finite losing publication: {observed}"
