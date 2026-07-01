"""Chunked-interval batching preserves the per-interval BQSEGM value function.

`bqsegm_per_interval_continuation_step_savings` solves one EGM case per liquid
interval and merges the cases with the branch-aware upper envelope. It batches the
intervals in chunks — parallel (`vmap`) within a chunk, sequential (`lax.map`) across
chunks — padding the interval inputs up to a whole number of chunks. The merged
value/marginal/policy must not depend on the chunk size: whether all intervals solve
in one chunk, one per chunk, or anything between, the envelope is the same, because
the padding intervals contribute no live candidate and the segment ids stay unique.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import bqsegm_step
from _lcm.egm.bqsegm_step import (
    _CHUNK_SIZE,
    bqsegm_per_interval_continuation_step_savings,
)

_CRRA = 2.0
_DISCOUNT = 0.96
_N_SAVINGS = 90
_N_LIQUID = 80


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def _build_inputs(n_intervals: int) -> dict:
    """Build a monotone-continuation per-interval problem with `n_intervals` cases."""
    liquid_grid = jnp.linspace(0.1, 30.0, _N_LIQUID)
    savings_grid = jnp.linspace(0.0, 28.0, _N_SAVINGS)
    breakpoints = jnp.linspace(2.0, 27.0, n_intervals - 1)
    coh_slopes = jnp.linspace(1.0, 1.3, n_intervals)
    coh_intercepts = jnp.linspace(0.5, 2.0, n_intervals)
    base_value = -1.0 / jnp.linspace(0.5, 5.0, _N_SAVINGS)
    base_marginal = jnp.linspace(2.0, 0.05, _N_SAVINGS)
    shift = jnp.linspace(0.0, 1.0, n_intervals)[:, None]
    cont_value = base_value[None, :] + shift
    cont_marginal = base_marginal[None, :] + 0.1 * shift
    return {
        "cont_value": cont_value,
        "cont_marginal": cont_marginal,
        "liquid_grid": liquid_grid,
        "savings_grid": savings_grid,
        "discount_factor": jnp.asarray(_DISCOUNT),
        "utility_of_action": _utility_of_action,
        "inverse_marginal_utility": _inverse_marginal_utility,
        "coh_slopes": coh_slopes,
        "coh_intercepts": coh_intercepts,
        "breakpoints": breakpoints,
    }


def _solve_with_chunk_size(inputs: dict, chunk_size: int, monkeypatch) -> tuple:
    monkeypatch.setattr(bqsegm_step, "_CHUNK_SIZE", chunk_size)
    value, marginal, policy = bqsegm_per_interval_continuation_step_savings(**inputs)
    return np.asarray(value), np.asarray(marginal), np.asarray(policy)


@pytest.mark.parametrize("n_intervals", [4, 5, 7, 8, 11, 12])
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 5])
def test_value_is_invariant_to_chunk_size(n_intervals, chunk_size, monkeypatch):
    """The merged value/marginal/policy does not depend on the chunk size.

    A chunk size of one is the fully sequential per-interval merge (no padding, no
    within-chunk parallelism); any larger chunk size pads the interval inputs up to a
    whole number of chunks and solves them in parallel. Both must yield the identical
    value function — the padding intervals stay NaN-dead and the segment ids unique —
    so the solve at `chunk_size` matches the fully sequential reference to floating-
    point tolerance for interval counts that both do and do not divide the chunk size.
    """
    inputs = _build_inputs(n_intervals)
    ref_value, ref_marginal, ref_policy = _solve_with_chunk_size(inputs, 1, monkeypatch)
    value, marginal, policy = _solve_with_chunk_size(inputs, chunk_size, monkeypatch)
    np.testing.assert_allclose(value, ref_value, atol=1e-6, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(
        marginal, ref_marginal, atol=1e-6, rtol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(policy, ref_policy, atol=1e-6, rtol=1e-6, equal_nan=True)


def test_solved_value_is_finite_where_the_budget_is_covered():
    """The per-interval merge yields a finite value where the budget funds a solve.

    A chunk-size-invariant NaN would pass the invariance test but hide a solve that
    dropped a covered interval; over the low-to-mid liquid range the intervals'
    cash-on-hand funds an interior or no-save candidate, so the merged value is finite
    there. (High-liquid points can legitimately be NaN when no interval's coh reaches
    them — the invariance test still guards those.)
    """
    inputs = _build_inputs(n_intervals=12)
    value, _, _ = bqsegm_per_interval_continuation_step_savings(**inputs)
    liquid_grid = np.asarray(inputs["liquid_grid"])
    covered = (liquid_grid > 1.0) & (liquid_grid < 15.0)
    assert np.isfinite(np.asarray(value)[covered]).all()


def test_chunk_size_is_a_small_positive_constant():
    """The chunk size is a small, tunable module-level constant of at least two."""
    assert isinstance(_CHUNK_SIZE, int)
    assert 2 <= _CHUNK_SIZE <= 16
