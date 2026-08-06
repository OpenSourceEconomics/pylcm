"""The MSS crossing's on-envelope band never falls below the working resolution.

A crossing is emitted only where the dense envelope, evaluated at the crossing
abscissa, agrees with the crossing's own value. The two are the same
mathematical quantity reached by different routes -- a maximum over segment
lines, and a two-line intersection solve -- so they agree only up to rounding,
and the comparison needs a band.

The band's absolute term is what decides the comparison near zero, where the
relative term vanishes. A fixed absolute term below the dtype's own epsilon
cannot decide anything there: two readings that agree to the last bit the
format has still fall outside it, so whether a crossing is emitted stops being
a property of the geometry and becomes a property of the precision.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.mss import _ON_ENVELOPE_ATOL, _on_envelope_atol


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_the_band_is_at_least_the_dtype_resolution(dtype):
    """At any precision the absolute band is wider than one epsilon."""
    band = float(_on_envelope_atol(jnp.zeros((), dtype=dtype)))

    assert band >= float(jnp.finfo(dtype).eps)


def test_the_band_at_float32_admits_readings_that_agree_to_the_last_bit():
    """Two float32 readings one ULP apart near zero are inside the band."""
    reference = jnp.zeros((), dtype=jnp.float32)
    band = float(_on_envelope_atol(reference))
    one_ulp = float(np.nextafter(np.float32(0.0), np.float32(1.0)))

    assert one_ulp <= band
    assert float(jnp.finfo(jnp.float32).eps) <= band


def test_the_float64_band_is_the_declared_absolute_tolerance():
    """At double precision the band is the declared constant, unchanged."""
    eps64 = float(np.finfo(np.float64).eps)

    assert eps64 < _ON_ENVELOPE_ATOL
