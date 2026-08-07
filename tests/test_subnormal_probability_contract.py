"""A probability between zero and the smallest normal number is refused."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.zero_safe import has_nonzero_subnormal


def _largest_subnormal() -> float:
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    return float(np.nextafter(np.finfo(dtype).tiny, dtype(0.0), dtype=dtype))


def test_a_subnormal_probability_is_seen_on_any_backend() -> None:
    """Bit inspection sees a subnormal probability whether or not arithmetic can.

    Whether arithmetic can see one is a property of the backend, not of the
    contract: XLA:CPU flushes a subnormal to zero, so an arithmetic test reports a
    genuine null event, while CUDA represents it. Reading the bits is what makes
    the refusal the same on both.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    weights = jnp.asarray([_largest_subnormal(), 1.0], dtype=dtype)

    assert bool(has_nonzero_subnormal(weights))


def test_ordinary_probabilities_are_not_flagged() -> None:
    """Zero and normal values pass, so the check adds no false refusal."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert not bool(
        has_nonzero_subnormal(jnp.asarray([0.0, 1.0, 0.5, -0.5], dtype=dtype))
    )
    assert not bool(
        has_nonzero_subnormal(jnp.asarray([jnp.finfo(dtype).tiny], dtype=dtype))
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_check_sees_either_sign(sign: float) -> None:
    """A signed subnormal is a represented nonzero value too."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert bool(
        has_nonzero_subnormal(jnp.asarray([sign * _largest_subnormal()], dtype=dtype))
    )
