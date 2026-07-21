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

from _lcm.egm.upper_envelope import fues
from tests.conftest import X64_ENABLED

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


def test_tied_savings_keep_all_nodes_under_ulp_perturbation():
    """Rounding noise in exactly tied implied savings never drops an envelope node.

    On a rising concave segment where every candidate saves the same amount
    (`A = R - c` constant), the savings-monotonicity dominance test compares
    quantities that are equal in exact arithmetic; their floating-point
    difference is pure rounding noise whose sign varies with backend reduction
    order. A one-ulp perturbation of a single policy value must not change the
    kept set — all candidates lie on the envelope either way.
    """
    grid = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    policy = grid - 0.01
    value = jnp.log(grid)

    # Nudge one interior consumption up by one ulp: its implied savings drops
    # one ulp below the (exactly tied) predecessor's.
    policy_perturbed = policy.at[4].set(jnp.nextafter(policy[4], jnp.inf))

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )
    _, _, _, n_kept_perturbed = fues.refine_envelope(
        endog_grid=grid, policy=policy_perturbed, value=value, n_refined=12
    )

    assert int(n_kept) == 8
    assert int(n_kept_perturbed) == 8


def _cross_source_masked_decrease_candidates():
    """A genuine savings decrease the magnitude-scaled float32 floor masks.

    Three ascending-grid candidates with large resources so the noise floor
    `16 * eps * max(|R|, |c|)` (~0.19 in float32 at this magnitude) exceeds the
    real savings step. Exogenous source savings are `[2.0, 1.0, 0.875]`: the top
    candidate genuinely saves 0.125 less than its predecessor (from a distinct
    source), a decrease the floor swallows but provenance resolves exactly. Value
    keeps rising, so only the savings-monotonicity clause can drop the top node.
    """
    grid = jnp.asarray([100_000.0, 100_001.0, 100_002.0], dtype=jnp.float32)
    policy = jnp.asarray([99_998.0, 100_000.0, 100_001.125], dtype=jnp.float32)
    value = jnp.asarray([10.0, 10.5, 10.55], dtype=jnp.float32)
    savings = jnp.asarray([2.0, 1.0, 0.875], dtype=jnp.float32)
    return grid, policy, value, savings


def test_savings_floor_masks_a_cross_source_decrease_without_provenance():
    """Without source savings, the float32 floor keeps a genuinely dominated node.

    The magnitude-scaled noise floor treats the real 0.125 savings decrease as a
    tie, so the top candidate survives refinement even though its exogenous
    savings fell below its predecessor's.
    """
    grid, policy, value, _ = _cross_source_masked_decrease_candidates()

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=8
    )

    assert int(n_kept) == 3


def test_savings_provenance_drops_a_cross_source_decrease():
    """Source savings resolve a cross-source decrease the floor would mask.

    Passing the exogenous source savings makes the monotonicity clause compare
    true sources (`0.875 < 1.0`) exactly rather than the floor-masked implied
    difference, so the dominated top candidate is dropped.
    """
    grid, policy, value, savings = _cross_source_masked_decrease_candidates()

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=8, savings=savings
    )

    assert int(n_kept) == 2


def test_savings_provenance_protects_same_source_ties():
    """Candidates sharing one exogenous savings source are never dropped as a decrease.

    On a rising concave segment every candidate saves the same amount, so their
    implied savings are equal in exact arithmetic and their source savings are
    identical. Comparing sources (`s_i < s_j` is false at equality) keeps every
    node on the envelope regardless of rounding in the implied difference.
    """
    grid = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    policy = grid - 0.01
    value = jnp.log(grid)
    savings = jnp.full_like(grid, 0.01)

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12, savings=savings
    )

    assert int(n_kept) == 8


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


def _node_aligned_crossing_candidates():
    """Two linear value branches crossing exactly on the grid node R = 10.

    - Branch A: v = 1 + 0.4 R, c = 4 + 0.5 R — optimal for R < 10.
    - Branch B: v = -3 + 0.8 R, c = -1 + 0.5 R — optimal for R > 10.

    They meet at R = 10 with equal value 5.0 but different policy (9.0 vs 4.0).
    Candidates are supplied in exogenous-savings order (implied savings
    `A = R - c = [0.5, 1, 3, 6, 6.5]` ascending) on a non-monotone endogenous
    grid; the R = 12 point (v = -100) is dominated.
    """
    grid = jnp.asarray([9.0, 10.0, 12.0, 10.0, 11.0])
    policy = jnp.asarray([8.5, 9.0, 9.0, 4.0, 4.5])
    value = jnp.asarray([4.6, 5.0, -100.0, 5.0, 5.8])
    return grid, policy, value


def test_node_aligned_crossing_publishes_right_policy_on_default_path():
    """A branch switch exactly on a grid node keeps the right branch's policy.

    With `segment_id=None` (the production FUES dispatch) and candidates in
    exogenous-savings order, a crossing that lands on the grid node R = 10 must
    reinsert both branch policies, so the refined policy just right of the node
    is the right branch's `c = -1 + 0.5 R`: c(10.1) = 4.05. Collapsing the node
    to only its left copy would instead interpolate across the policy
    discontinuity — from the left copy 9.0 down to the next node 4.5 — giving a
    spurious 8.55. The envelope value is 5.08 either way.
    """
    grid, policy, value = _node_aligned_crossing_candidates()

    got_grid, got_policy, got_value, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_policy, 10.1), 4.05, atol=_COMPUTED_ATOL
    )
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, 10.1), 5.08, atol=_COMPUTED_ATOL
    )


def _two_crossing_chain_candidates():
    """Three affine value branches A|B|C crossing exactly on nodes 10 and 20.

    - Branch A: c = 8, value slope 0.125 — optimal on [9, 10].
    - Branch B: c = 4, value slope 0.25 — optimal on [10, 20].
    - Branch C: c = 2, value slope 0.5 — optimal on [20, 21].

    Candidates are in exogenous-savings order (`savings` ascending). The exact
    envelope keeps branch B across (10, 20), so at R = 15 the pair is
    (value, policy) = (6.25, 4). Losing B's middle segment interpolates the
    policy across the collapsed discontinuity instead.
    """
    grid = jnp.asarray([9.0, 10.0, 10.0, 20.0, 20.0, 21.0])
    policy = jnp.asarray([8.0, 8.0, 4.0, 4.0, 2.0, 2.0])
    value = jnp.asarray([4.875, 5.0, 5.0, 7.5, 7.5, 8.0])
    savings = jnp.asarray([1.0, 2.0, 6.0, 16.0, 18.0, 19.0])
    return grid, policy, value, savings


def test_multi_crossing_chain_preserves_the_middle_branch_policy():
    """Two node-aligned crossings keep the middle branch: c(15) = 4, V(15) = 6.25.

    A three-branch chain crossing on both R = 10 and R = 20 must retain branch B
    across (10, 20). The envelope has six records (two per crossing plus the two
    endpoints), so `n_kept == 6`.
    """
    grid, policy, value, savings = _two_crossing_chain_candidates()

    got_grid, got_policy, got_value, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=16
    )

    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_policy, 15.0), 4.0, atol=_COMPUTED_ATOL
    )
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, 15.0), 6.25, atol=_COMPUTED_ATOL
    )
    assert int(n_kept) == 6


def _dominated_coincident_candidates():
    """A coincident c=9 / c=4 pair (value 5) sits beneath the c=7 branch at R=10.

    The c = 7 branch strictly dominates at R = 10 (value 6 > 5), so the exact
    envelope is [9, 10, 11] with policy 7 throughout — the lower pair is not an
    envelope kink.
    """
    grid = jnp.asarray([9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0])
    policy = jnp.asarray([9.0, 9.0, 7.0, 7.0, 7.0, 4.0, 4.0])
    value = jnp.asarray([5 - 1 / 9, 5.0, 6 - 1 / 7, 6.0, 6 + 1 / 7, 5.0, 5.25])
    savings = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0])
    return grid, policy, value, savings


def test_dominated_coincident_pair_is_not_promoted_to_a_kink():
    """A dominated coincident pair beneath a superior branch is dropped.

    The c = 7 branch owns R = 10 (value 6), so the read there is (6, 7) and the
    envelope has exactly three nodes — no false kink, no spurious overflow.
    """
    grid, policy, value, savings = _dominated_coincident_candidates()

    got_grid, got_policy, got_value, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=16
    )

    assert int(n_kept) == 3
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_value, 10.0), 6.0, atol=_COMPUTED_ATOL
    )
    np.testing.assert_allclose(
        _envelope_interp(got_grid, got_policy, 10.0), 7.0, atol=_COMPUTED_ATOL
    )


def test_envelope_is_invariant_to_a_common_value_shift():
    """Adding a large constant to every value leaves the kept set and policy fixed.

    The maximizer tie test is invariant to `V -> V + K`: a relative `rtol * |V|`
    band would widen at large value level and manufacture kinks; the kept set and
    the refined policy must be identical to the un-shifted envelope.
    """
    grid, policy, value, savings = _dominated_coincident_candidates()

    # The shift must stay below the working precision's resolution-vs-margin
    # bound: the domination margins here are ~1, and once the shift's ULP exceeds
    # that margin (float32 near 1e8 has ULP ~8) the dominated members round up
    # into the tie and get promoted — a precision limit, not a tolerance bug. A
    # 3e5 shift keeps ULP well below 1 at both precisions while remaining large
    # enough that a relative `rtol * |V|` band (`1e-5 * 3e5 = 3 > 1`) would still
    # manufacture kinks.
    shift = 1e8 if X64_ENABLED else 3e5

    base = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=16
    )
    shifted = fues.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value + shift,
        savings=savings,
        n_refined=16,
    )

    assert int(base[3]) == int(shifted[3])
    np.testing.assert_allclose(
        np.asarray(base[1]), np.asarray(shifted[1]), atol=_COMPUTED_ATOL, equal_nan=True
    )


def test_same_source_coincident_candidates_are_not_a_jump():
    """Two candidates from one exogenous savings node are a duplicate, not a jump.

    At a coincident grid with equal value, candidates sharing an exogenous
    savings source are the same point: the refined row has three unique nodes and
    does not overflow, even when their implied `R - c` rounds differently in
    float32. Meaningful under 32-bit precision, where the subtraction noise
    appears.
    """
    grid = jnp.asarray([1.2, 1.3000002, 1.3000002, 1.4000001], dtype=jnp.float32)
    policy = jnp.asarray([0.9, 1.0000001, 1.0000002, 1.1], dtype=jnp.float32)
    value = (10 + jnp.log(policy)).astype(jnp.float32)
    savings = jnp.asarray([0.3, 0.3, 0.3, 0.3], dtype=jnp.float32)

    _, _, _, n_kept = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=8
    )

    assert int(n_kept) == 3


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
