"""Original CE join identity and finite-domain decisions at IEEE boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ce_scaled_linear import join_ce_direction


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ce_join_zero_correction_retains_finite_payloads_and_refuses_nonfinite() -> (
    None
):
    dtype = jnp.asarray(0.0).dtype
    info = np.finfo(dtype)
    original = np.asarray(
        [
            info.smallest_subnormal,
            -info.smallest_subnormal,
            float(info.tiny) / 4.0,
            -float(info.tiny) / 4.0,
            3.0,
            -3.0,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=dtype,
    )
    base = jnp.asarray(original)
    direction = jnp.zeros_like(base)
    logarithm = jnp.log(jnp.asarray(2.0, dtype=dtype))
    sign = jnp.ones_like(logarithm)
    finite = np.isfinite(original)
    nonzero = finite & (original != 0.0)
    zero = finite & ~nonzero
    for evaluate in (join_ce_direction, jax.jit(join_ce_direction)):
        actual = evaluate(logabs=logarithm, sign=sign, base=base, direction=direction)
        assert isinstance(actual, jax.Array)
        assert actual.shape == base.shape
        assert actual.dtype == base.dtype
        host = np.asarray(actual)
        np.testing.assert_array_equal(np.isfinite(host), finite)
        assert host[nonzero].tobytes() == original[nonzero].tobytes()
        # Adding represented zero promises numerical zero, not its sign bit.
        assert np.all(host[zero] == 0.0)
        assert np.all(np.isnan(host[~finite]))
