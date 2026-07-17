"""MSS crossing completeness: every envelope switch is represented exactly.

The off-grid simulation policy read interpolates the refined MSS rows between
nodes, so the rows must carry *every* breakpoint of the candidate
correspondence's upper envelope — not only switches the candidate abscissae
happen to straddle one at a time. The spec pins the three topologies a
single-switch-per-interval scan misses:

- two (or more) switches between adjacent candidate abscissae, where an
  intermediate branch wins only strictly inside the interval;
- a switch exactly at a candidate abscissa, which needs the duplicated-node
  convention (same abscissa, left then right branch record);
- a winning branch ending below the next branch's line, where the envelope
  value drops discontinuously and both one-sided limits must be published.

Intervals needing more switches than the per-interval enumeration budget must
overflow loudly via `n_kept > n_refined` (the existing overflow contract), not
silently truncate the switch sequence.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.mss import refine_envelope
from tests.conftest import X64_ENABLED

_ATOL = 1e-8 if X64_ENABLED else 1e-4


def _interp(grid: jnp.ndarray, field: jnp.ndarray, x_query: float) -> float:
    keep = ~np.isnan(np.asarray(grid))
    return float(np.interp(x_query, np.asarray(grid)[keep], np.asarray(field)[keep]))


def _three_branch_candidates() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Three log-utility-consistent branches on `[10, 11]`, folded twice.

    Slopes are `1 / policy` (the envelope relation); the branch value lines
    cross at `R = 10.3` (A-B) and `R = 10.7` (B-C), so the middle branch wins
    only strictly inside the single candidate interval `(10, 11)`.
    """
    slopes = np.array([1.0 / 3.0, 2.0 / 3.0, 4.0])
    policies = 1.0 / slopes
    intercepts = np.array([20.0, 19.9, 17.566666666666666])

    grid = jnp.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    policy = jnp.asarray(np.repeat(policies, 2))
    value = jnp.asarray(np.stack([intercepts, intercepts + slopes], axis=1).ravel())
    return grid, policy, value


@pytest.mark.parametrize(
    "segment_id",
    [None, jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])],
    ids=["inferred-segments", "explicit-labels"],
)
def test_two_switches_inside_one_candidate_interval_are_both_inserted(segment_id):
    """A branch winning only strictly inside one interval gets both crossings.

    The true envelope switches A->B at `R = 10.3` and B->C at `R = 10.7`; at
    `R = 10.4` the middle branch wins with value `20 + (2/3) * 0.4 - 0.1` and
    policy `1.5`. The refined rows must reproduce that read.
    """
    grid, policy, value = _three_branch_candidates()

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=24,
        segment_id=segment_id,
    )

    assert int(n_kept) <= 24
    clean = np.asarray(refined_grid)
    clean = clean[~np.isnan(clean)]
    assert np.isclose(clean, 10.3, atol=_ATOL).sum() == 2
    assert np.isclose(clean, 10.7, atol=_ATOL).sum() == 2
    np.testing.assert_allclose(
        _interp(refined_grid, refined_value, 10.4),
        19.9 + (2.0 / 3.0) * 0.4,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_policy, 10.4), 1.5, atol=_ATOL
    )


def test_a_switch_exactly_at_a_candidate_abscissa_is_duplicated():
    """A branch switch landing on a candidate abscissa keeps both records.

    The branches `v = 20 + (R - 10)` (policy 1) and `v = 20.5 + 2.5 (R - 10.5)`
    (policy 0.4) cross exactly at the candidate abscissa `R = 10.5`. The
    refined rows must carry that abscissa twice — left policy 1, right policy
    0.4 — so a read at `R = 10.51` returns the right branch (value 20.525,
    policy 0.4) instead of a blend.
    """
    grid = jnp.array([10.0, 11.0, 10.5, 11.5])
    policy = jnp.array([1.0, 1.0, 0.4, 0.4])
    value = jnp.array([20.0, 21.0, 20.5, 23.0])

    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    clean_grid = np.asarray(refined_grid)
    at_switch = np.isclose(clean_grid, 10.5, atol=_ATOL)
    assert at_switch.sum() == 2
    np.testing.assert_allclose(
        np.sort(np.asarray(refined_policy)[at_switch]), [0.4, 1.0], atol=_ATOL
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_value, 10.51), 20.525, atol=_ATOL
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_policy, 10.51), 0.4, atol=_ATOL
    )


def test_a_winner_branch_ending_below_the_next_publishes_both_limits():
    """An envelope value drop at a branch end keeps both one-sided records.

    Branch A (`v = 20 + (R - 10)`, policy 1) ends at `R = 11` strictly above
    the parallel branch B (`v = 18 + (R - 10)`, policy 0.5), so the envelope
    drops from 21 to 19 there. The refined rows must carry `R = 11` twice —
    values 21 then 19 — so a read at `R = 11.5` returns B's line (value 19.5,
    policy 0.5) instead of a blend toward A's endpoint.
    """
    grid = jnp.array([10.0, 11.0, 10.0, 12.0])
    policy = jnp.array([1.0, 1.0, 0.5, 0.5])
    value = jnp.array([20.0, 21.0, 18.0, 20.0])

    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    clean_grid = np.asarray(refined_grid)
    at_drop = np.isclose(clean_grid, 11.0, atol=_ATOL)
    assert at_drop.sum() == 2
    np.testing.assert_allclose(
        np.sort(np.asarray(refined_value)[at_drop]), [19.0, 21.0], atol=_ATOL
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_value, 11.5), 19.5, atol=_ATOL
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_policy, 11.5), 0.5, atol=_ATOL
    )


def test_more_switches_than_the_enumeration_budget_overflows_loudly():
    """An interval needing more switches than the budget reports overflow.

    Twelve tangent lines to the convex `exp(R - 10)` between adjacent candidate
    abscissae produce eleven envelope switches inside one interval — beyond any
    fixed per-interval enumeration budget. The kernel must signal via
    `n_kept > n_refined` (the overflow contract) rather than silently truncate
    the switch sequence.
    """
    tangent_points = np.linspace(10.05, 10.95, 12)
    grid_rows = []
    value_rows = []
    policy_rows = []
    for t in tangent_points:
        slope = np.exp(t - 10.0)
        grid_rows += [10.0, 11.0]
        value_rows += [slope * (1.0 + 10.0 - t), slope * (1.0 + 11.0 - t)]
        policy_rows += [1.0 / slope, 1.0 / slope]

    _, _, _, n_kept = refine_envelope(
        endog_grid=jnp.asarray(grid_rows),
        policy=jnp.asarray(policy_rows),
        value=jnp.asarray(value_rows),
        n_refined=200,
    )

    assert int(n_kept) > 200
