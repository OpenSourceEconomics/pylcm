"""Scaling by a power of two moves no information, on any backend.

The certified sign pulls its operands into the binade around one before
evaluating, and the whole certificate rests on that shift being reversible: an
operand that does not come back unchanged is one whose geometry was never
supplied. So the scaling has to be exact wherever the result stays normal — not
exact once compiled, or exact on one backend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.double_double import scale_by_power_of_two
from tests.conftest import X64_ENABLED


def _working_dtypes():
    """The numpy/jax dtype pair matching the configured working precision."""
    return (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)


def _mantissas_and_exponents(dtype):
    """Random significands in `[0.5, 1)` paired with safely normal exponents."""
    rng = np.random.default_rng(seed=1)
    limit = 60 if dtype is np.float64 else 12
    mantissas = rng.uniform(0.5, 1.0, size=2000).astype(dtype)
    exponents = rng.integers(-limit, limit, size=2000).astype(np.int32)
    return mantissas, exponents


@pytest.mark.parametrize("compiled", [False, True])
def test_scaling_by_a_power_of_two_is_exact(compiled):
    """`value * 2**exponent` is exact, eager and compiled alike."""
    dtype, jax_dtype = _working_dtypes()
    mantissas, exponents = _mantissas_and_exponents(dtype)
    scale = jax.jit(scale_by_power_of_two) if compiled else scale_by_power_of_two

    got = np.asarray(
        scale(jnp.asarray(mantissas, jax_dtype), jnp.asarray(exponents, jnp.int32))
    )

    np.testing.assert_array_equal(got, np.ldexp(mantissas, exponents))


@pytest.mark.parametrize("compiled", [False, True])
def test_scaling_round_trip_returns_the_original_bits(compiled):
    """Scaling down and back up recovers the operand exactly.

    This is the premise the certificate tests at runtime: an operand that
    survives the round trip carried its information through the shift.
    """
    dtype, jax_dtype = _working_dtypes()
    mantissas, exponents = _mantissas_and_exponents(dtype)
    scale = jax.jit(scale_by_power_of_two) if compiled else scale_by_power_of_two

    source = jnp.asarray(mantissas, jax_dtype)
    exponent = jnp.asarray(exponents, jnp.int32)
    back = scale(scale(source, -exponent), exponent)

    np.testing.assert_array_equal(np.asarray(back), mantissas)
