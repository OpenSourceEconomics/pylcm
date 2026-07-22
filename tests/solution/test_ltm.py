"""Spec for the LTM upper-envelope kernel (Druedahl's `consav` brute method).

Contract under test — `_lcm.egm.upper_envelope.ltm.refine_envelope`:

    refine_envelope(
        *,
        endog_grid: Float1D,   # candidate resources points (consecutive segments)
        policy: Float1D,
        value: Float1D,
        n_refined: int,        # static output length
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]

LTM is the local-upper-bound brute method: for each output abscissa it scans
every linear segment connecting two consecutive input candidates, linearly
interpolates the bracketing segments, and keeps the highest value. The cost is
`O(N_query x N_segments) = O(K^2)` by construction.

Returns NaN-padded arrays of length `n_refined` holding the upper envelope of
the candidate value correspondence (weakly ascending in the grid) plus the
number of kept points. `n_kept > n_refined` signals overflow.

Assertions are semantic — the refined arrays must *represent* the analytic upper
envelope under linear interpolation.

Skips until the kernel exists.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope import ltm
from tests.conftest import X64_ENABLED

# Tolerance for kernel-computed quantities (segment interpolation): float32
# carries a few ulp through the bracketing arithmetic.
_COMPUTED_ATOL = 1e-8 if X64_ENABLED else 1e-5


def _drop_nan(arr: jnp.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    return out[~np.isnan(out)]


def _envelope_interp(grid, value, x_query):
    keep = ~np.isnan(np.asarray(grid))
    return np.interp(x_query, np.asarray(grid)[keep], np.asarray(value)[keep])


def _crossing_segments_candidates():
    """Two linear choice-specific value segments crossing at R* = 2.25.

    - Segment A: v = 1.0 + 0.4 R, c = 0.30 R — optimal for R < 2.25.
    - Segment B: v = 0.1 + 0.8 R, c = 0.55 R — optimal for R > 2.25.

    The candidates arrive in two consecutive runs (one per segment), as the
    Euler inversion produces them in non-concave regions: the input order makes
    each segment a chain of consecutive candidates.
    """
    r_a = np.array([0.6, 1.2, 1.8, 2.4])
    r_b = np.array([0.8, 1.6, 2.6, 3.4])
    grid = np.concatenate([r_a, r_b])
    value = np.concatenate([1.0 + 0.4 * r_a, 0.1 + 0.8 * r_b])
    policy = np.concatenate([0.30 * r_a, 0.55 * r_b])
    return jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value)


R_STAR = 2.25


def test_concave_input_passes_through_unchanged():
    """A monotone-concave single segment is returned as-is (sorted, no edits)."""
    grid = jnp.linspace(1.0, 10.0, 10)
    policy = 0.3 * grid
    value = jnp.log(grid)

    got_grid, got_policy, got_value, n_kept = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=14
    )

    assert int(n_kept) == 10
    np.testing.assert_allclose(_drop_nan(got_grid), np.asarray(grid), atol=1e-12)
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, np.asarray(grid)),
        np.asarray(value),
        atol=_COMPUTED_ATOL,
    )
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_policy, np.asarray(grid)),
        np.asarray(policy),
        atol=_COMPUTED_ATOL,
    )


def test_envelope_value_at_each_candidate_node_is_the_segment_maximum():
    """At every output abscissa the value is the max over bracketing segments.

    The brute scan evaluates each candidate abscissa against every consecutive
    segment, so the refined value at a node is the highest interpolant among the
    segments bracketing it — exactly `max(v_A, v_B)` at the node, with no kink
    insertion. (Between nodes the linear read traces one segment, so the exact
    crossing is recovered only to within the node spacing.)
    """
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, got_value, _ = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    clean_grid = _drop_nan(got_grid)
    clean_value = np.asarray(got_value)[~np.isnan(np.asarray(got_grid))]
    expected = np.maximum(1.0 + 0.4 * clean_grid, 0.1 + 0.8 * clean_grid)
    np.testing.assert_allclose(clean_value, expected, atol=_COMPUTED_ATOL)


def test_envelope_policy_follows_the_winning_segment_at_a_node():
    """The refined policy at a node is the winning segment's interpolated policy.

    At a node well below the crossing segment A dominates, so the published
    policy is A's `c = 0.30 R`; at a node well above, B dominates with
    `c = 0.55 R`. The policy is read at the node abscissa so the kink-placement
    error between nodes does not enter.
    """
    grid, policy, value = _crossing_segments_candidates()

    got_grid, got_policy, _, _ = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    clean_grid = _drop_nan(got_grid)
    clean_policy = np.asarray(got_policy)[~np.isnan(np.asarray(got_grid))]

    # A node where A wins: m = 1.2 (v_A = 1.48 > v_B = 1.06).
    idx_low = int(np.argmin(np.abs(clean_grid - 1.2)))
    np.testing.assert_allclose(clean_policy[idx_low], 0.30 * 1.2, atol=_COMPUTED_ATOL)
    # A node where B wins: m = 2.6 (v_B = 2.18 > v_A = 2.04).
    idx_high = int(np.argmin(np.abs(clean_grid - 2.6)))
    np.testing.assert_allclose(clean_policy[idx_high], 0.55 * 2.6, atol=_COMPUTED_ATOL)


def test_refined_grid_is_weakly_ascending_with_nan_tail():
    """Non-NaN prefix is weakly ascending; NaNs appear only as a suffix."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, _, _ = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    raw = np.asarray(got_grid)
    nan_mask = np.isnan(raw)
    first_nan = nan_mask.argmax() if nan_mask.any() else raw.size
    assert not nan_mask[:first_nan].any()
    assert nan_mask[first_nan:].all()
    assert np.all(np.diff(raw[:first_nan]) >= -1e-12)


def test_overlapping_segment_dominates_a_bracketed_lower_node():
    """A later segment looping back over a node lifts that node to the envelope.

    The non-concavity an EGM cloud carries is a *backward jump* in the
    endogenous grid: a later segment re-covers an earlier abscissa at a higher
    value. The candidates `m = [1, 3, 2, 4]` chain two segments — `(1->3)` then
    `(2->4)` after the jump back to 2 — so the abscissa `m = 2` is bracketed
    both by the low first segment (value 1.5 at `m=2`) and by the high second
    segment (value 5.0 at `m=2`). LTM keeps the higher one.
    """
    grid = jnp.array([1.0, 3.0, 2.0, 4.0])
    policy = jnp.array([0.5, 1.5, 1.0, 2.0])
    value = jnp.array([1.0, 2.0, 5.0, 7.0])

    got_grid, _, got_value, _ = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=8
    )

    clean_grid = _drop_nan(got_grid)
    clean_value = np.asarray(got_value)[~np.isnan(np.asarray(got_grid))]
    at_two = int(np.argmin(np.abs(clean_grid - 2.0)))
    np.testing.assert_allclose(clean_value[at_two], 5.0, atol=_COMPUTED_ATOL)


def test_vmap_over_rows_matches_per_row_calls():
    """The kernel is vmappable; batched output equals per-row output."""
    grid, policy, value = _crossing_segments_candidates()
    batched = jax.vmap(
        lambda g, p, v: ltm.refine_envelope(
            endog_grid=g, policy=p, value=v, n_refined=12
        )
    )(
        jnp.stack([grid, grid]),
        jnp.stack([policy, policy]),
        jnp.stack([value, value]),
    )
    single = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    for batched_arr, single_arr in zip(batched, single, strict=True):
        np.testing.assert_array_equal(
            np.asarray(batched_arr[0]), np.asarray(single_arr)
        )


def test_overflow_is_reported_via_n_kept():
    """When the envelope needs more slots than `n_refined`, `n_kept` says so."""
    grid, policy, value = _crossing_segments_candidates()

    _, _, _, n_kept = ltm.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=4
    )

    assert int(n_kept) > 4
