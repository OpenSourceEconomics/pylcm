"""Regression locks for the external FUES correctness audit (findings F3, F4).

These pin two confirmed defects in `refine_envelope` so any repair is forced to
acknowledge them. Both `xfail(strict=True)` tests assert the *correct* envelope
behavior and fail on the current kernel; a fix flips them to XPASS, which strict
mode reports as a failure until the marker is removed.

The companion non-xfail tests lock the *boundary* of each finding — the
conditions under which the kernel is already correct — so a fix cannot regress
them.

Ground truth is the interpolated envelope *function*: the refined rows exist to
be read by `interp_on_padded_grid` downstream, so correctness is judged there,
not on which raw points survive. See `FUES_AUDIT_VERIFICATION_RESULTS.md`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.fues import refine_envelope
from tests.conftest import X64_ENABLED

_ATOL = 1e-8 if X64_ENABLED else 1e-5


def _kept(grid, *arrays):
    """Drop the NaN-padded tail, returning the kept prefix of each array."""
    keep = ~np.isnan(np.asarray(grid))
    return (np.asarray(grid)[keep], *(np.asarray(a)[keep] for a in arrays))


# ---------------------------------------------------------------------------
# F3 — a strictly dominated point at a shared abscissa must not survive
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="F3: the coincident-abscissa guard in _slope lets a strictly "
    "dominated duplicate point survive the scan (fues.py:469-472). Confirmed "
    "bug at the array level; remove this marker when the scan collapses "
    "equal-abscissa candidates to their max value.",
    strict=True,
)
def test_f3_dominated_duplicate_point_is_not_retained():
    """At a duplicated abscissa, no kept point carries a strictly lower value.

    A genuine envelope kink duplicates an abscissa with *equal* value and two
    policies; a dominated duplicate carries a *lower* value and must be dropped.
    """
    grid, _policy, value, n_kept = refine_envelope(
        endog_grid=jnp.array([0.0, 0.0, 1.0]),
        policy=jnp.array([0.0, 1.0, 1.0]),
        value=jnp.array([0.0, 1.0, 2.0]),
        n_refined=10,
    )
    g, v = _kept(grid[: int(n_kept)], value[: int(n_kept)])
    for abscissa in np.unique(g):
        at = np.isclose(g, abscissa, atol=_ATOL)
        if at.sum() > 1:
            spread = v[at].max() - v[at].min()
            assert spread <= _ATOL, (
                f"duplicated abscissa {abscissa} carries differing values "
                f"{sorted(v[at].tolist())} — a dominated point survived"
            )


def test_f3_interpolated_function_is_correct_despite_retained_duplicate():
    """The interpolated value/policy is exact even with the duplicate retained.

    `interp_on_padded_grid` skips the lower-index duplicate (`side="right"`), so
    the array-level F3 defect has no effect on the function the EGM step reads.
    This invariant is why F3 carries no practical impact; it must not regress.
    """
    grid, policy, value, _ = refine_envelope(
        endog_grid=jnp.array([0.0, 0.0, 1.0]),
        policy=jnp.array([0.0, 1.0, 1.0]),
        value=jnp.array([0.0, 1.0, 2.0]),
        n_refined=10,
    )
    x_query = jnp.array([-0.5, 0.0, 0.25, 0.5, 1.0])
    got_value = interp_on_padded_grid(x_query=x_query, xp=grid, fp=value)
    got_policy = interp_on_padded_grid(x_query=x_query, xp=grid, fp=policy)
    # Envelope over [0, 1] is the line through (0, 1)-(1, 2); policy is 1.
    expected_value = 1.0 + np.clip(np.asarray(x_query), 0.0, None)
    np.testing.assert_allclose(np.asarray(got_value), expected_value, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(got_policy), 1.0, atol=_ATOL)


# ---------------------------------------------------------------------------
# F4 — the bounded scan must not accept a run of suboptimal points
# ---------------------------------------------------------------------------


def _interleaved_segments():
    """Upper line A(x)=x with two anchors, plus 11 points 0.5 below it.

    The eleven below-envelope points interleave A's `(0.1, 0.1)` and `(12, 12)`
    anchors in grid order, so the bounded forward scan cannot see A's
    continuation at x=12 within its window.
    """
    a_x = [0.0, 0.1, 12.0]
    b_x = [float(i) for i in range(1, 12)]
    endog_grid = jnp.asarray(a_x + b_x)
    policy = jnp.asarray(a_x + [x - 100.0 for x in b_x])
    value = jnp.asarray(a_x + [i - 0.5 for i in range(1, 12)])
    return endog_grid, policy, value


@pytest.mark.xfail(
    reason="F4: with >n_points_to_scan interleaved off-segment candidates the "
    "bounded scan (fues.py:400) misses the same-segment witness and accepts a "
    "run of dominated points. Confirmed bug at the shipped default "
    "n_points_to_scan=10; remove this marker when the scan is made exhaustive "
    "in a correctness mode or keyed on a segment id.",
    strict=True,
)
def test_f4_interleaved_segments_give_analytic_envelope_at_default_scan():
    """The refined envelope equals the upper line A(x)=x at the default scan."""
    endog_grid, policy, value = _interleaved_segments()
    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid, policy=policy, value=value, n_refined=64
    )
    x_query = jnp.linspace(0.0, 12.0, 13)
    got = interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    np.testing.assert_allclose(np.asarray(got), np.asarray(x_query), atol=1e-6)


def test_f4_failure_resolves_when_scan_window_covers_all_candidates():
    """Widening the scan past the interleave count recovers the exact envelope.

    Locks the boundary of F4: the defect is the bounded window, not the
    refinement logic. A fix must keep this correct.
    """
    endog_grid, policy, value = _interleaved_segments()
    n_candidates = endog_grid.shape[0]
    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=64,
        n_points_to_scan=n_candidates,
    )
    x_query = jnp.linspace(0.0, 12.0, 13)
    got = interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    np.testing.assert_allclose(np.asarray(got), np.asarray(x_query), atol=1e-6)
