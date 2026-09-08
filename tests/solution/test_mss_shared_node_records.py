"""One outgoing/incoming pair at a crossing shared by candidate input nodes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.mss import refine_envelope


def _refine(*, endog_grid, policy, value):
    return refine_envelope(
        endog_grid=endog_grid, policy=policy, value=value, n_refined=32
    )


_jit_refine = jax.jit(_refine)
_batched_refine = jax.jit(jax.vmap(_refine))


@pytest.mark.parametrize("copies", [1, 2, 4])
@pytest.mark.parametrize("shift", [0.0, 16.0])
@pytest.mark.parametrize("value_scale", [0.5, 1.0, 256.0])
@pytest.mark.parametrize("transformed", ["jit", "vmap"])
def test_shared_crossing_has_one_record_per_owner(
    *, copies, shift, value_scale, transformed
):
    grid = jnp.asarray([9.0, 10.0, 9.5] + [10.0] * copies + [10.5]) + shift
    policy = jnp.asarray([8.0, 8.0, 2.0] + [2.0] * copies + [2.0])
    value = jnp.asarray([4.875, 5.0, 4.75] + [5.0] * copies + [5.25]) * value_scale
    arguments = {"endog_grid": grid, "policy": policy, "value": value}
    if transformed == "vmap":
        result = _batched_refine(
            **{name: jnp.stack([array, array]) for name, array in arguments.items()}
        )
        rows = [tuple(array[index] for array in result) for index in range(2)]
    else:
        rows = [_jit_refine(**arguments)]
    for refined_grid, refined_policy, refined_value, kept in rows:
        count = int(kept)
        assert count <= 32
        live_grid = np.asarray(refined_grid)[:count]
        assert np.all(np.diff(live_grid) >= 0.0)
        at_crossing = np.flatnonzero(live_grid == 10.0 + shift)
        assert len(at_crossing) == 2
        np.testing.assert_array_equal(
            np.asarray(refined_policy)[at_crossing], [8.0, 2.0]
        )
        np.testing.assert_array_equal(
            np.asarray(refined_value)[at_crossing], [5.0 * value_scale] * 2
        )
