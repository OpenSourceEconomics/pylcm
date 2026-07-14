"""Spec for the FUES upper-envelope kernel (Dobrescu & Shanker 2022).

Contract under test — `_lcm.egm.upper_envelope.fues.refine_envelope`:

    refine_envelope(
        *,
        endog_grid: Float1D,   # candidate resources points (any order)
        policy: Float1D,
        value: Float1D,
        n_refined: int,        # static output length
        jump_thresh: float = 2.0,
        n_points_to_scan: int = 10,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]

Returns NaN-padded arrays of length `n_refined` holding the upper envelope of the
candidate value correspondence (weakly ascending in the grid: intersection points
appear twice, with left- and right-extrapolated policies) plus the number of kept
points. `n_kept > n_refined` signals overflow; the caller decides how to surface
it.

Assertions are semantic — the refined arrays must *represent* the analytic upper
envelope under linear interpolation — rather than pinning which input points
survive, so the spec is robust to scan-mechanics details.

Skips until the kernel exists.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import X64_ENABLED

fues = pytest.importorskip(
    "_lcm.egm.upper_envelope.fues", reason="FUES kernel not yet implemented"
)

# Tolerance for quantities computed by the kernel (segment intersections and
# values interpolated through them): float32 carries a few ulp (~1e-7 at the
# fixtures' scale) through the intersection arithmetic.
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

    Candidate points interleave both segments, as the Euler inversion produces
    them in non-concave regions.
    """
    r_a = np.array([0.6, 1.2, 1.8, 2.4])
    r_b = np.array([0.8, 1.6, 2.6, 3.4])
    grid = np.concatenate([r_a, r_b])
    value = np.concatenate([1.0 + 0.4 * r_a, 0.1 + 0.8 * r_b])
    policy = np.concatenate([0.30 * r_a, 0.55 * r_b])
    return jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value)


R_STAR = 2.25
V_STAR = 1.0 + 0.4 * R_STAR


def test_concave_input_passes_through_unchanged():
    """A monotone-concave single segment is returned as-is (sorted, no edits)."""
    grid = jnp.linspace(1.0, 10.0, 10)
    policy = 0.3 * grid
    value = jnp.log(grid)

    got_grid, got_policy, got_value, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=14
    )

    assert int(n_kept) == 10
    np.testing.assert_allclose(_drop_nan(got_grid), np.asarray(grid), atol=1e-12)
    np.testing.assert_allclose(_drop_nan(got_policy), np.asarray(policy), atol=1e-12)
    np.testing.assert_allclose(_drop_nan(got_value), np.asarray(value), atol=1e-12)


def test_steep_but_continuous_policy_is_not_treated_as_a_switch():
    """`|ΔA/ΔR|` just below the threshold leaves a single segment untouched."""
    grid = jnp.linspace(1.0, 5.0, 9)
    # c = 5 - 0.9 R: implied savings A = R - c = 1.9 R - 5, so |ΔA/ΔR| = 1.9 < 2.
    policy = 5.0 - 0.9 * grid
    value = jnp.log(grid)

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    assert int(n_kept) == 9


def test_crossing_segments_yield_the_analytic_upper_envelope():
    """The refined value function equals max(v_A, v_B) under linear interpolation."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, got_value, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    r_query = np.linspace(0.7, 3.3, 53)
    expected = np.maximum(1.0 + 0.4 * r_query, 0.1 + 0.8 * r_query)
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, r_query), expected, atol=_COMPUTED_ATOL
    )


def test_crossing_segments_insert_intersection_twice_with_both_policies():
    """The kink abscissa appears twice, carrying the left and right policies."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, got_policy, _, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    clean_grid = _drop_nan(got_grid)
    clean_policy = np.asarray(got_policy)[~np.isnan(np.asarray(got_grid))]
    at_kink = np.isclose(clean_grid, R_STAR, atol=_COMPUTED_ATOL)
    assert at_kink.sum() == 2
    np.testing.assert_allclose(
        np.sort(clean_policy[at_kink]),
        np.sort([0.30 * R_STAR, 0.55 * R_STAR]),
        atol=_COMPUTED_ATOL,
    )


def test_crossing_segments_drop_dominated_points():
    """Candidates strictly below the envelope do not survive refinement."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, _, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    clean_grid = _drop_nan(got_grid)
    for dominated in [0.8, 1.6, 2.4]:
        assert not np.any(np.isclose(clean_grid, dominated, atol=1e-8))


def test_refined_grid_is_weakly_ascending_with_nan_tail():
    """Non-NaN prefix is weakly ascending; NaNs appear only as a suffix."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, _, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    raw = np.asarray(got_grid)
    nan_mask = np.isnan(raw)
    first_nan = nan_mask.argmax() if nan_mask.any() else raw.size
    assert not nan_mask[:first_nan].any()
    assert nan_mask[first_nan:].all()
    assert np.all(np.diff(raw[:first_nan]) >= -1e-12)


def test_value_decreasing_point_is_dropped():
    """A single interior dominated point (value dip) is removed."""
    grid = jnp.array([1.0, 2.0, 3.0])
    policy = jnp.array([0.5, 1.8, 1.5])
    value = jnp.array([1.0, 0.9, 2.0])

    got_grid, _, _, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=6
    )

    clean_grid = _drop_nan(got_grid)
    assert not np.any(np.isclose(clean_grid, 2.0, atol=1e-8))


def test_vmap_over_rows_matches_per_row_calls():
    """The kernel is vmappable; batched output equals per-row output."""
    grid, policy, value = _crossing_segments_candidates()
    batched = jax.vmap(
        lambda g, p, v: fues.refine_envelope(
            endog_grid=g, policy=p, value=v, n_refined=12
        )
    )(
        jnp.stack([grid, grid]),
        jnp.stack([policy, policy]),
        jnp.stack([value, value]),
    )
    single = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    for batched_arr, single_arr in zip(batched, single, strict=True):
        np.testing.assert_array_equal(
            np.asarray(batched_arr[0]), np.asarray(single_arr)
        )


@pytest.mark.parametrize("scan_unroll", [2, 4, 8])
def test_scan_unroll_leaves_the_refined_envelope_unchanged(scan_unroll):
    """`scan_unroll` is a pure performance knob: the output is bit-identical.

    Unrolling the candidate `lax.scan` only changes how the loop is lowered, not
    what it computes, so the refined `(grid, policy, value)` and the kept count
    must match the default `scan_unroll=1` run exactly.
    """
    grid, policy, value = _crossing_segments_candidates()
    baseline = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )
    unrolled = fues.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=12,
        scan_unroll=scan_unroll,
    )
    for baseline_arr, unrolled_arr in zip(baseline, unrolled, strict=True):
        np.testing.assert_array_equal(
            np.asarray(unrolled_arr), np.asarray(baseline_arr)
        )


def test_overflow_is_reported_via_n_kept():
    """When the envelope needs more slots than `n_refined`, `n_kept` says so."""
    grid, policy, value = _crossing_segments_candidates()

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=4
    )

    assert int(n_kept) > 4


def test_insert_node_crossings_selects_one_branch_at_a_three_way_tie():
    """A node where three branches coincide keeps one genuine branch policy.

    The on-node crossing repair reinserts the branch policy dropped by the
    coincident-node dedup. When three candidates share one abscissa with equal value
    (two coincident crossings at that node), the kept node must carry a single genuine
    branch policy, not the sum of both crossings' policies — a summed policy is not a
    consumption any branch prescribes.
    """
    presort_grid = jnp.array([1.0, 1.0, 1.0])
    presort_policy = jnp.array([10.0, 20.0, 30.0])
    presort_value = jnp.array([5.0, 5.0, 5.0])
    segment_sorted = jnp.array([0.0, 1.0, 2.0])
    n_refined = 8
    refined_grid = jnp.full(n_refined, jnp.nan).at[0].set(1.0)
    refined_policy = jnp.full(n_refined, jnp.nan).at[0].set(10.0)
    refined_value = jnp.full(n_refined, jnp.nan).at[0].set(5.0)

    out_grid, out_policy, _out_value, _n = fues._insert_node_crossings(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=jnp.asarray(1, dtype=jnp.int32),
        presort_grid=presort_grid,
        presort_policy=presort_policy,
        presort_value=presort_value,
        segment_sorted=segment_sorted,
        n_refined=n_refined,
        n_points_to_scan=1,
        jump_thresh=2.0,
    )

    at_tie = np.asarray(out_policy)[np.isclose(np.asarray(out_grid), 1.0)]
    genuine = {10.0, 20.0, 30.0}
    assert set(at_tie.tolist()) <= genuine
