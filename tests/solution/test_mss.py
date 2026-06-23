"""Spec for the MSS upper-envelope kernel (HARK's EGM upper envelope).

Contract under test — `_lcm.egm.upper_envelope.mss.refine_envelope`:

    refine_envelope(
        *,
        endog_grid: Float1D,   # candidate resources points (consecutive segments)
        policy: Float1D,
        value: Float1D,
        n_refined: int,        # static output length
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]

MSS is HARK's EGM upper envelope (Carroll et al. 2018): it sweeps the common
grid left-to-right, evaluates every currently overlapping linear segment, keeps
the max-value branch, and — unlike LTM — inserts the exact segment-crossing
abscissa (the kink) where the winning branch switches. The output therefore
tracks the FUES envelope tightly.

Returns NaN-padded arrays of length `n_refined` holding the upper envelope of
the candidate value correspondence (weakly ascending in the grid) plus the
number of kept points. `n_kept > n_refined` signals overflow.

Assertions are semantic — the refined arrays must *represent* the analytic upper
envelope under linear interpolation, with the crossing point present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope import get_upper_envelope
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM
from tests.conftest import X64_ENABLED

mss = pytest.importorskip(
    "_lcm.egm.upper_envelope.mss", reason="MSS kernel not yet implemented"
)


def _mss_solver():
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        upper_envelope="mss",
    )


# Tolerance for kernel-computed quantities (segment interpolation): float32
# carries a few ulp through the bracketing and crossing arithmetic.
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

    got_grid, got_policy, got_value, n_kept = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
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


def test_envelope_value_is_the_pointwise_segment_maximum():
    """The refined value traces `max(v_A, v_B)` of the two crossing segments.

    The interpolated envelope value at every abscissa is the higher of the two
    choice-specific value lines, evaluated under linear interpolation of the
    refined arrays.
    """
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, got_value, _ = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    test_points = np.linspace(0.8, 3.4, 25)
    interp = _envelope_interp(got_grid, got_value, test_points)
    expected = np.maximum(1.0 + 0.4 * test_points, 0.1 + 0.8 * test_points)
    np.testing.assert_allclose(interp, expected, atol=_COMPUTED_ATOL)


def test_inserts_the_exact_crossing_abscissa():
    """An envelope node lands exactly on the segment crossing R* = 2.25.

    Unlike LTM, MSS inserts the kink: the refined grid carries a node at the
    intersection of the two value lines, so the linear read does not smear the
    kink across the local grid spacing.
    """
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, _, _ = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    clean_grid = _drop_nan(got_grid)
    nearest = clean_grid[int(np.argmin(np.abs(clean_grid - R_STAR)))]
    np.testing.assert_allclose(nearest, R_STAR, atol=_COMPUTED_ATOL)


def test_policy_jumps_at_the_inserted_kink():
    """At the crossing the policy carries the left and right branch values.

    The kink abscissa is inserted twice — once with segment A's policy
    (`0.30 R*`) and once with segment B's (`0.55 R*`) — so the policy
    discontinuity at the discrete-choice switch is preserved exactly.
    """
    grid, policy, value = _crossing_segments_candidates()

    got_grid, got_policy, _, _ = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    clean_grid = np.asarray(got_grid)
    clean_policy = np.asarray(got_policy)
    at_star = np.isclose(clean_grid, R_STAR, atol=_COMPUTED_ATOL)
    policies_at_star = np.sort(clean_policy[at_star])

    assert at_star.sum() == 2
    np.testing.assert_allclose(
        policies_at_star, np.sort([0.30 * R_STAR, 0.55 * R_STAR]), atol=_COMPUTED_ATOL
    )


def test_refined_grid_is_weakly_ascending_with_nan_tail():
    """Non-NaN prefix is weakly ascending; NaNs appear only as a suffix."""
    grid, policy, value = _crossing_segments_candidates()

    got_grid, _, _, _ = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
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
    segment (value 5.0 at `m=2`). MSS keeps the higher one.
    """
    grid = jnp.array([1.0, 3.0, 2.0, 4.0])
    policy = jnp.array([0.5, 1.5, 1.0, 2.0])
    value = jnp.array([1.0, 2.0, 5.0, 7.0])

    got_grid, _, got_value, _ = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, 2.0), 5.0, atol=_COMPUTED_ATOL
    )


def test_vmap_over_rows_matches_per_row_calls():
    """The kernel is vmappable; batched output equals per-row output."""
    grid, policy, value = _crossing_segments_candidates()
    batched = jax.vmap(
        lambda g, p, v: mss.refine_envelope(
            endog_grid=g, policy=p, value=v, n_refined=16
        )
    )(
        jnp.stack([grid, grid]),
        jnp.stack([policy, policy]),
        jnp.stack([value, value]),
    )
    single = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    for batched_arr, single_arr in zip(batched, single, strict=True):
        np.testing.assert_array_equal(
            np.asarray(batched_arr[0]), np.asarray(single_arr)
        )


def test_overflow_is_reported_via_n_kept():
    """When the envelope needs more slots than `n_refined`, `n_kept` says so."""
    grid, policy, value = _crossing_segments_candidates()

    _, _, _, n_kept = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=3
    )

    assert int(n_kept) > 3


def test_mss_backend_is_selected_by_solver_config():
    """`upper_envelope="mss"` dispatches to the MSS backend.

    The backend selected for an `mss` solver must reproduce the standalone MSS
    kernel on a non-concave candidate set.
    """
    solver = _mss_solver()
    backend = get_upper_envelope(solver=solver, n_refined=16)

    grid, policy, value = _crossing_segments_candidates()
    marginal = jnp.ones_like(grid)
    via_backend = backend(
        endog_grid=grid, policy=policy, value=value, marginal_utility=marginal
    )
    direct = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )
    for via_arr, direct_arr in zip(via_backend, direct, strict=True):
        np.testing.assert_array_equal(np.asarray(via_arr), np.asarray(direct_arr))


def test_dead_candidates_are_excluded_from_the_envelope():
    """NaN-poisoned candidates never reach the envelope or delete a live node.

    A dead candidate arrives NaN-filled; it must sort to the tail and contribute
    no segment, leaving the live concave envelope intact.
    """
    grid = jnp.array([1.0, 2.0, jnp.nan, 3.0, 4.0])
    policy = jnp.array([0.3, 0.6, jnp.nan, 0.9, 1.2])
    value = jnp.array([0.0, 0.7, jnp.nan, 1.1, 1.4])

    got_grid, _, got_value, n_kept = mss.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    clean_grid = _drop_nan(got_grid)
    assert int(n_kept) == 4
    np.testing.assert_allclose(clean_grid, [1.0, 2.0, 3.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, np.asarray([1.0, 2.0, 3.0, 4.0])),
        [0.0, 0.7, 1.1, 1.4],
        atol=_COMPUTED_ATOL,
    )
