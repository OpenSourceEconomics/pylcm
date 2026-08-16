"""Exact cell-level ownership stays outside the traced JAX arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope._exact_affine import exact_cell_hull
from _lcm.egm.upper_envelope.cell_hull import hull_owners
from tests.conftest import X64_ENABLED


def _dtype():
    """Return the JAX floating dtype selected for this test run."""
    return jnp.float64 if X64_ENABLED else jnp.float32


def test_an_empty_batch_has_empty_outputs() -> None:
    """Partition padding may call the kernel with no cells and remains valid."""
    dtype = _dtype()
    resolve = jax.jit(
        lambda left, right, live, low, high, grid, value: exact_cell_hull(
            left=left,
            right=right,
            live=live,
            low=low,
            high=high,
            endog_grid=grid,
            value=value,
            max_runs=3,
        )
    )
    outputs = resolve(
        jnp.empty((0,), dtype=dtype),
        jnp.empty((0,), dtype=dtype),
        jnp.empty((0, 3), dtype=jnp.bool_),
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0, 8), dtype=dtype),
        jnp.empty((0, 8), dtype=dtype),
    )
    assert [np.asarray(output).shape for output in outputs] == [(0, 4), (0, 3), (0,)]


def test_a_vmapped_hull_lowers_to_one_custom_call() -> None:
    """Batch width changes one call's shape, not the exact representation."""
    dtype = _dtype()
    resolve = jax.jit(
        jax.vmap(
            lambda left, right, grid, value: hull_owners(
                left=left,
                right=right,
                live=jnp.asarray([True, True]),
                low=jnp.asarray([0, 2], dtype=jnp.int32),
                high=jnp.asarray([1, 3], dtype=jnp.int32),
                endog_grid=grid,
                value=value,
                max_runs=2,
            )
        )
    )
    left = jnp.zeros((3,), dtype=dtype)
    right = jnp.ones((3,), dtype=dtype)
    grid = jnp.tile(jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=dtype), (3, 1))
    value = jnp.tile(jnp.asarray([0.0, 0.0, -0.25, 1.0], dtype=dtype), (3, 1))

    lowered = resolve.lower(left, right, grid, value).as_text()
    target = "ExactCellHullF64" if X64_ENABLED else "ExactCellHullF32"
    assert lowered.count("stablehlo.custom_call") == 1
    assert lowered.count(target) == 1
