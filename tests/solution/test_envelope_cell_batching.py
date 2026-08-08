"""Batching the node-cell axis trades the exact envelope's working set for width.

Ownership is resolved per node cell, and `DCEGM.envelope_cell_batch_size` sets
how many cells are in flight: `None` scans them one at a time and holds a single
cell, an integer resolves that many in parallel and holds that many. Either way
it is a pure partition of the work — it may change how much memory the solve
needs but may never change a published value or policy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from _lcm.solution.dcegm import DCEGM
from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError

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


@pytest.mark.parametrize("cell_batch_size", [1, 2, 5, 64])
def test_published_row_is_identical_across_cell_batch_sizes(cell_batch_size):
    """Every batch size publishes the same row as resolving all cells at once."""
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    endog_grid, policy, value = _wiggly_chain(dtype)
    expected = refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=3 * _N_CANDIDATES,
        max_runs=8,
        cell_batch_size=None,
    )
    got = refine_envelope_exact(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=3 * _N_CANDIDATES,
        max_runs=8,
        cell_batch_size=cell_batch_size,
    )

    expected_grid, expected_policy, expected_value, expected_kept = expected
    got_grid, got_policy, got_value, got_kept = got
    assert int(got_kept) == int(expected_kept)
    keep = int(expected_kept)
    assert_array_equal(np.asarray(got_grid[:keep]), np.asarray(expected_grid[:keep]))
    assert_array_equal(
        np.asarray(got_policy[:keep]), np.asarray(expected_policy[:keep])
    )
    assert_array_equal(np.asarray(got_value[:keep]), np.asarray(expected_value[:keep]))


def _solver(**overrides):
    """A minimal valid DC-EGM configuration, with fields overridden as given."""
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=8),
        **overrides,
    )


def test_cell_batch_size_defaults_to_the_smallest_working_set():
    """The default scans cells one at a time rather than putting several in flight."""
    assert _solver().envelope_cell_batch_size is None


def test_cell_batch_size_accepts_an_integer_to_resolve_cells_in_parallel():
    """Widening the step is available to a caller whose cells leave a device idle."""
    assert _solver(envelope_cell_batch_size=8).envelope_cell_batch_size == 8


@pytest.mark.parametrize("invalid", [0, -1])
def test_non_positive_cell_batch_size_is_rejected(invalid):
    """A batch size below one partitions nothing and is refused at construction."""
    with pytest.raises(RegimeInitializationError, match="envelope_cell_batch_size"):
        _solver(envelope_cell_batch_size=invalid)
