"""The exact envelope partitions node cells at its execution-plan width.

Ownership is resolved independently per node cell. The width changes the
working set while preserving every published value and ownership decision.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp

_N_CANDIDATES = 24


def _wiggly_chain(dtype):
    """A savings-ordered chain that folds, so many cells carry several links."""
    rng = np.random.default_rng(seed=17)
    endog_grid = np.cumsum(rng.uniform(0.05, 0.5, size=_N_CANDIDATES)).astype(dtype)
    # Three sweeps back over the same resources, so runs overlap and compete.
    endog_grid[8:16] = endog_grid[8:16] - dtype(1.2)
    endog_grid[16:] = endog_grid[16:] - dtype(0.4)
    value = np.cumsum(rng.uniform(0.0, 1.0, size=_N_CANDIDATES)).astype(dtype)
    policy = rng.uniform(0.5, 3.0, size=_N_CANDIDATES).astype(dtype)
    return (
        jnp.asarray(endog_grid),
        jnp.asarray(policy),
        jnp.asarray(value),
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("cell_width", [1, 2, 5, 64])
def test_published_row_agrees_across_cell_widths(cell_width):
    """Every width publishes the same row as a serial cell scan."""
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    endog_grid, policy, value = _wiggly_chain(dtype)
    expected = refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=3 * _N_CANDIDATES,
        max_runs=8,
        cell_width=1,
    )
    got = refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=3 * _N_CANDIDATES,
        max_runs=8,
        cell_width=cell_width,
    )

    expected_grid, expected_policy, expected_value, expected_kept = expected
    got_grid, got_policy, got_value, got_kept = got
    assert int(got_kept) == int(expected_kept)
    keep = int(expected_kept)
    assert_agrees_to_ulp(got=got_grid[:keep], expected=expected_grid[:keep], n_ulp=4)
    assert_agrees_to_ulp(
        got=got_policy[:keep], expected=expected_policy[:keep], n_ulp=4
    )
    assert_agrees_to_ulp(got=got_value[:keep], expected=expected_value[:keep], n_ulp=4)
