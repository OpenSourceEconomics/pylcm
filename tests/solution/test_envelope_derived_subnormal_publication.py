"""A normal-input affine read may not silently flush its represented result.

The exact affine value can be a positive subnormal even when every stored operand is
normal or zero.  The certified query must publish that represented value or abstain in
all channels; finite zero is neither outcome.
"""

import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.xfail(
    reason=(
        "A backend that flushes subnormals hands back finite zero, and the query "
        "publishes it in all three channels instead of the represented value or NaN. "
        "Expected to pass on a backend that reads the subnormal band, such as CUDA."
    ),
    strict=False,
)
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
