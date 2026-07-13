"""Spec for the RFC upper-envelope kernel (Dobrescu & Shanker, Box 1).

Contract under test — `_lcm.egm.upper_envelope.rfc.refine_envelope`:

    refine_envelope(
        *,
        endog_grid: Float1D,   # candidate resources points (any order)
        policy: Float1D,
        value: Float1D,
        marginal_utility: Float1D,  # candidate supgradient mu = dv/dR
        n_refined: int,        # static output length
        search_radius: int = 10,
        jump_thresh: float = 2.0,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]

The Rooftop-Cut (RFC) algorithm builds, at each candidate $j$, the tangent
(value) line from the local supgradient $\\mu_j = \\partial v / \\partial R$
and deletes every neighbor $l$ within `search_radius` that lies *below* $j$'s
tangent *and* sits across a policy jump (the segment-switch test). The
survivors form the upper envelope.

Unlike FUES, RFC only *deletes* candidates — it never inserts the exact
segment-crossing point. So the refined output is a NaN-padded weakly-ascending
*subset* of the input candidates: the dominated points are gone, the
upper-envelope points are retained, and a kink lands between two retained
points (recovered by the downstream linear/Hermite carry read to within local
grid spacing).

Assertions are therefore about *which candidates survive*, not about an
inserted kink abscissa.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import get_bracket_finder, get_upper_envelope
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM
from tests.conftest import X64_ENABLED

# Interpolated bracket reads are float-eps-limited at the active precision.
_BRACKET_ATOL = 1e-10 if X64_ENABLED else 1e-5
_GRID_ATOL = 1e-12 if X64_ENABLED else 1e-5


def _rfc_solver():
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        upper_envelope="rfc",
    )


rfc = pytest.importorskip(
    "_lcm.egm.upper_envelope.rfc", reason="RFC kernel not yet implemented"
)


def _drop_nan(arr: jnp.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    return out[~np.isnan(out)]


def _crossing_segments_candidates():
    """Two linear choice-specific value segments crossing at R* = 2.25.

    - Segment A: v = 1.0 + 0.4 R, c = 0.30 R — optimal for R < 2.25.
    - Segment B: v = 0.1 + 0.8 R, c = 0.55 R — optimal for R > 2.25.

    Candidate points interleave both segments, as the Euler inversion produces
    them in non-concave regions. Below R*, the segment-B points are dominated;
    above R*, the segment-A points are. The supgradient $\\mu = \\partial v /
    \\partial R$ is each segment's constant value slope (0.4 / 0.8).
    """
    r_a = np.array([0.6, 1.2, 1.8, 2.4])
    r_b = np.array([0.8, 1.6, 2.6, 3.4])
    grid = np.concatenate([r_a, r_b])
    value = np.concatenate([1.0 + 0.4 * r_a, 0.1 + 0.8 * r_b])
    policy = np.concatenate([0.30 * r_a, 0.55 * r_b])
    marginal = np.concatenate([np.full_like(r_a, 0.4), np.full_like(r_b, 0.8)])
    return (
        jnp.asarray(grid),
        jnp.asarray(policy),
        jnp.asarray(value),
        jnp.asarray(marginal),
    )


R_STAR = 2.25


def test_concave_input_passes_through_unchanged():
    """A monotone-concave single segment is returned as-is (sorted, no deletions)."""
    grid = jnp.linspace(1.0, 10.0, 10)
    policy = 0.3 * grid
    value = jnp.log(grid)
    marginal = 1.0 / grid

    got_grid, got_policy, got_value, n_kept = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=14,
    )

    assert int(n_kept) == 10
    np.testing.assert_allclose(_drop_nan(got_grid), np.asarray(grid), atol=1e-12)
    np.testing.assert_allclose(_drop_nan(got_policy), np.asarray(policy), atol=1e-12)
    np.testing.assert_allclose(_drop_nan(got_value), np.asarray(value), atol=1e-12)


def test_steep_but_continuous_policy_is_not_treated_as_a_switch():
    """`|Δc/ΔR|` just below the threshold leaves a single segment untouched."""
    grid = jnp.linspace(1.0, 5.0, 9)
    # c = 5 - 0.9 R: |Δc/ΔR| = 0.9 < 2, so no segment switch is detected.
    policy = 5.0 - 0.9 * grid
    value = jnp.log(grid)
    marginal = 1.0 / grid

    _, _, _, n_kept = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
    )

    assert int(n_kept) == 9


def test_crossing_segments_retain_only_the_upper_envelope_subset():
    """RFC keeps exactly the upper-envelope candidates and deletes the dominated.

    No crossing point is inserted, so the retained grid is the subset of input
    points that lie on the upper envelope: segment-A points below R* and
    segment-B points above R*.
    """
    grid, policy, value, marginal = _crossing_segments_candidates()

    got_grid, _, _, _ = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
        jump_thresh=0.6,
    )

    clean_grid = np.sort(_drop_nan(got_grid))
    # Surviving upper-envelope points: A below R*, B above R*.
    expected = np.array([0.6, 1.2, 1.8, 2.6, 3.4])
    np.testing.assert_allclose(clean_grid, expected, atol=1e-8)


def test_crossing_segments_drop_dominated_points():
    """Candidates strictly below the envelope do not survive refinement."""
    grid, policy, value, marginal = _crossing_segments_candidates()

    got_grid, _, _, _ = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
        jump_thresh=0.6,
    )

    clean_grid = _drop_nan(got_grid)
    # Below R*: segment-B points 0.8, 1.6 are dominated. Above R*: segment-A
    # point 2.4 is dominated.
    for dominated in [0.8, 1.6, 2.4]:
        assert not np.any(np.isclose(clean_grid, dominated, atol=1e-8))


def test_no_crossing_point_is_inserted():
    """RFC never inserts the segment-crossing abscissa R* (it only deletes)."""
    grid, policy, value, marginal = _crossing_segments_candidates()

    got_grid, _, _, _ = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
        jump_thresh=0.6,
    )

    clean_grid = _drop_nan(got_grid)
    assert not np.any(np.isclose(clean_grid, R_STAR, atol=1e-6))


def test_value_decreasing_point_is_dropped():
    """A single interior dominated point (value dip) is removed."""
    grid = jnp.array([1.0, 2.0, 3.0])
    policy = jnp.array([0.5, 1.8, 1.5])
    value = jnp.array([1.0, 0.9, 2.0])
    # Supgradient: 1->3 the value rises (slope ~0.5); the dip at 2 sits below
    # both neighbours' tangents and across a policy jump.
    marginal = jnp.array([0.5, 0.5, 0.5])

    got_grid, _, _, _ = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=6,
        jump_thresh=0.5,
    )

    clean_grid = _drop_nan(got_grid)
    assert not np.any(np.isclose(clean_grid, 2.0, atol=1e-8))


def test_neg_inf_value_candidates_stay_deleted():
    """A `-inf`/NaN candidate is dominated by every finite point and dropped.

    Dead candidates arrive as NaN-filled triples (the EGM step poisons
    `-inf`-valued candidates to NaN before refinement); the dominance test
    must keep them out of the envelope and out of the kept count.
    """
    grid = jnp.array([1.0, 2.0, 3.0, 4.0])
    policy = jnp.array([0.5, 1.0, 1.5, 2.0])
    value = jnp.array([1.0, jnp.nan, 2.0, 2.5])
    marginal = jnp.array([0.5, 0.0, 0.5, 0.5])

    got_grid, _, got_value, n_kept = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=8,
    )

    clean_grid = _drop_nan(got_grid)
    assert not np.any(np.isclose(clean_grid, 2.0, atol=1e-8))
    assert int(n_kept) == 3
    assert not np.any(np.isnan(_drop_nan(got_value)))


def test_refined_grid_is_weakly_ascending_with_nan_tail():
    """Non-NaN prefix is weakly ascending; NaNs appear only as a suffix."""
    grid, policy, value, marginal = _crossing_segments_candidates()

    got_grid, _, _, _ = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
        jump_thresh=0.6,
    )

    raw = np.asarray(got_grid)
    nan_mask = np.isnan(raw)
    first_nan = nan_mask.argmax() if nan_mask.any() else raw.size
    assert not nan_mask[:first_nan].any()
    assert nan_mask[first_nan:].all()
    assert np.all(np.diff(raw[:first_nan]) >= -1e-12)


def test_vmap_over_rows_matches_per_row_calls():
    """The kernel is vmappable; batched output equals per-row output."""
    grid, policy, value, marginal = _crossing_segments_candidates()
    batched = jax.vmap(
        lambda g, p, v, m: rfc.refine_envelope(
            endog_grid=g, policy=p, value=v, marginal_utility=m, n_refined=12
        )
    )(
        jnp.stack([grid, grid]),
        jnp.stack([policy, policy]),
        jnp.stack([value, value]),
        jnp.stack([marginal, marginal]),
    )
    single = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
    )

    for batched_arr, single_arr in zip(batched, single, strict=True):
        np.testing.assert_array_equal(
            np.asarray(batched_arr[0]), np.asarray(single_arr)
        )


def test_overflow_is_reported_via_n_kept():
    """When the envelope needs more slots than `n_refined`, `n_kept` says so."""
    grid = jnp.linspace(1.0, 10.0, 10)
    policy = 0.3 * grid
    value = jnp.log(grid)
    marginal = 1.0 / grid

    _, _, _, n_kept = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=4,
    )

    assert int(n_kept) > 4


def test_rfc_backend_is_selected_by_solver_config():
    """`upper_envelope="rfc"` dispatches to the RFC backend.

    The backend selected for an `rfc` solver must reproduce the standalone
    RFC kernel on a non-concave candidate set, while `"fues"` stays on FUES.
    """
    solver = _rfc_solver()
    backend = get_upper_envelope(solver=solver, n_refined=12)

    grid, policy, value, marginal = _crossing_segments_candidates()
    via_backend = backend(
        endog_grid=grid, policy=policy, value=value, marginal_utility=marginal
    )
    direct = rfc.refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        n_refined=12,
        search_radius=solver.rfc_search_radius,
        jump_thresh=solver.rfc_jump_thresh,
    )
    for via_arr, direct_arr in zip(via_backend, direct, strict=True):
        np.testing.assert_array_equal(np.asarray(via_arr), np.asarray(direct_arr))


@pytest.mark.parametrize("x_query", [0.5, 1.0, 2.0, R_STAR, 3.0, 4.0])
def test_rfc_bracket_finder_matches_full_envelope_interpolation(x_query):
    """The non-streamed RFC bracket reads identically to the full refined row.

    The asset-row solve reads the refined envelope at one query per node. RFC
    has no streamed bracket finder, so `get_bracket_finder` materializes the
    full envelope and locates the `searchsorted(side="right")` bracket. The
    bracket's linear read must equal `interp_on_padded_grid` on the full
    refined row at every query — below the first node, on the kink, and above
    the last — so a full-envelope-then-interpolate and the bracket read publish
    the same `(value, policy)`. Below-support queries extrapolate along the
    first bracket's secant; above-support queries clamp to the last kept
    value, matching the padded-row read's boundary conventions.
    """
    solver = _rfc_solver()
    grid, policy, value, marginal = _crossing_segments_candidates()
    n_pad = 12

    refined_grid, refined_policy, refined_value, n_kept = get_upper_envelope(
        solver=solver, n_refined=n_pad
    )(endog_grid=grid, policy=policy, value=value, marginal_utility=marginal)

    query = jnp.asarray(x_query)
    ref_value = interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_value)
    ref_policy = interp_on_padded_grid(
        x_query=query, xp=refined_grid, fp=refined_policy
    )

    bracket = get_bracket_finder(solver=solver, n_refined=n_pad)(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal_utility=marginal,
        x_query=query,
    )

    width = jnp.where(
        bracket.upper_grid == bracket.lower_grid,
        1.0,
        bracket.upper_grid - bracket.lower_grid,
    )
    weight = jnp.minimum((query - bracket.lower_grid) / width, 1.0)
    bracket_value = bracket.lower_value + weight * (
        bracket.upper_value - bracket.lower_value
    )
    bracket_policy = bracket.lower_policy + weight * (
        bracket.upper_policy - bracket.lower_policy
    )

    np.testing.assert_allclose(
        float(bracket_value), float(ref_value), atol=_BRACKET_ATOL
    )
    np.testing.assert_allclose(
        float(bracket_policy), float(ref_policy), atol=_BRACKET_ATOL
    )
    assert int(bracket.n_kept) == int(n_kept)
    np.testing.assert_allclose(
        float(bracket.first_grid), float(refined_grid[0]), atol=_GRID_ATOL
    )
