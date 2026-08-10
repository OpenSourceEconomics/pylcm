"""A dead candidate is dead however it says so.

A candidate is absent from the chain when its abscissa is not a number, when
its value is not a number, or when it belongs to no monotone run. Which of
those holds is a property of the producer, not of the envelope: the published
row must be the same either way.

The producer reaches the envelope with a finite abscissa carrying a NaN value
on purpose -- a candidate whose continuation is genuine poison keeps its
abscissa so the poison propagates rather than being laundered into an absent
point. Folding such a candidate into a live run instead would put its NaN into
the run's own node list, where a link the hull query selects can read it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact

_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_VALUE = [0.0, 1.0, 1.8, 2.4, 2.8, 3.0]
_POLICY = [0.4, 0.8, 1.1, 1.3, 1.4, 1.45]
_N_REFINED = 24


def _refine(grid, value, policy):
    return refine_envelope_exact(
        endog_grid=jnp.asarray(grid),
        policy=jnp.asarray(policy),
        value=jnp.asarray(value),
        n_refined=_N_REFINED,
        segment_id=None,
        max_runs=8,
    )


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_a_nan_value_makes_a_candidate_absent_like_a_nan_abscissa(index):
    """Killing a candidate by its value matches killing it by its abscissa."""
    by_value_grid, by_value = list(_GRID), list(_VALUE)
    by_value[index] = np.nan

    by_both_grid, by_both = list(_GRID), list(_VALUE)
    by_both[index] = np.nan
    by_both_grid[index] = np.nan

    got_grid, got_policy, got_value, got_n = _refine(by_value_grid, by_value, _POLICY)
    want_grid, want_policy, want_value, want_n = _refine(by_both_grid, by_both, _POLICY)

    assert int(got_n) == int(want_n)
    np.testing.assert_array_equal(np.asarray(got_grid), np.asarray(want_grid))
    np.testing.assert_array_equal(np.asarray(got_value), np.asarray(want_value))
    np.testing.assert_array_equal(np.asarray(got_policy), np.asarray(want_policy))


def test_a_nan_valued_candidate_does_not_poison_the_published_row():
    """The row stays publishable when one candidate's value is not a number."""
    value = list(_VALUE)
    value[2] = np.nan

    refined_grid, _, refined_value, n_kept = _refine(_GRID, value, _POLICY)

    assert int(n_kept) <= _N_REFINED
    live = np.asarray(refined_grid)[: int(n_kept)]
    assert np.isfinite(live).all()
    assert np.isfinite(np.asarray(refined_value)[: int(n_kept)]).all()
