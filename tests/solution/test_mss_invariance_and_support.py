"""MSS numerical invariance and read-support contracts.

The refined row feeds an off-grid read, so the refinement's numerical
decisions must be invariant to representations the economics does not see:

- translating the resource coordinate (a unit or origin change) must not
  merge distinct envelope crossings — tie tolerances are interval-local, not
  relative to the absolute coordinate;
- adding a common constant to all values (a cardinal shift) must not change
  which branch wins where;
- a live candidate point whose value exceeds both one-sided link winners is
  unrepresentable in a linearly-read row (a spike at one abscissa) and must
  fail loudly through the `n_kept` overflow contract, never be dropped;
- an interval between adjacent emitted nodes that no live link covers (from
  NaN-dead candidates or an inferred finite-value-decrease split) is a
  coverage gap: the row's linear span there is fabricated, so the refinement
  reports `read_supported = False` and the published simulation rows are
  withheld from the off-grid read.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.egm.upper_envelope.mss import (
    refine_envelope,
    refine_envelope_with_support,
)
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM
from tests.conftest import X64_ENABLED

_ATOL = 1e-8 if X64_ENABLED else 1e-4


def _interp(grid: jnp.ndarray, field: jnp.ndarray, x_query: float) -> float:
    keep = ~np.isnan(np.asarray(grid))
    return float(np.interp(x_query, np.asarray(grid)[keep], np.asarray(field)[keep]))


def test_crossing_enumeration_is_invariant_to_the_resource_origin():
    """Large resource origins must not merge distinct local crossings.

    Three log-utility-consistent branches on `[origin, origin + 10]` switch at
    local offsets 1 and 2. In float32 at `origin = 1e6`, a tolerance relative
    to the absolute coordinate would span several whole resource units and
    collapse the two switches; interval-local comparisons keep them distinct,
    so the middle branch wins on `(1, 2)` and the read at local offset 1.25
    returns its value line and policy.
    """
    origin = 1_000_000.0
    slopes = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    policies = 1.0 / slopes
    intercepts = np.array([0.0, -1.0, -3.0], dtype=np.float32)
    grid, policy, value = [], [], []
    for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
        grid.extend([origin, origin + 10.0])
        policy.extend([action, action])
        value.extend([intercept, intercept + 10.0 * slope])

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=jnp.asarray(grid, dtype=jnp.float32),
        policy=jnp.asarray(policy, dtype=jnp.float32),
        value=jnp.asarray(value, dtype=jnp.float32),
        n_refined=32,
    )

    assert int(n_kept) <= 32
    local = np.asarray(refined_grid)[: int(n_kept)] - origin
    assert np.isclose(local, 1.0, atol=0.05).sum() == 2
    assert np.isclose(local, 2.0, atol=0.05).sum() == 2
    query = jnp.asarray(origin + 1.25, dtype=jnp.float32)
    got_value = interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_value)
    got_policy = interp_on_padded_grid(
        x_query=query, xp=refined_grid, fp=refined_policy
    )
    np.testing.assert_allclose(float(got_value), 1.5, rtol=1e-4)
    np.testing.assert_allclose(float(got_policy), 0.5, rtol=1e-4)


def test_crossing_enumeration_is_invariant_to_a_common_value_shift():
    """A cardinal value shift must not change crossings or winners.

    The three-branch multi-switch correspondence with `1e6` added to every
    value keeps its switches at `R = 10.3` and `R = 10.7` and the same policy
    read — the envelope is ordinal in value levels.
    """
    slopes = np.array([1.0 / 3.0, 2.0 / 3.0, 4.0])
    policies = 1.0 / slopes
    intercepts = np.array([20.0, 19.9, 17.566666666666666]) + 1_000_000.0

    grid = jnp.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    policy = jnp.asarray(np.repeat(policies, 2))
    value = jnp.asarray(np.stack([intercepts, intercepts + slopes], axis=1).ravel())

    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=24
    )

    clean = np.asarray(refined_grid)
    clean = clean[~np.isnan(clean)]
    assert np.isclose(clean, 10.3, atol=1e-6).sum() == 2
    assert np.isclose(clean, 10.7, atol=1e-6).sum() == 2
    np.testing.assert_allclose(
        _interp(refined_grid, refined_value, 10.4),
        1_000_000.0 + 19.9 + (2.0 / 3.0) * 0.4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _interp(refined_grid, refined_policy, 10.4), 1.5, atol=_ATOL
    )


def test_a_dominating_point_candidate_overflows_loudly():
    """A candidate spike no linear row can carry must not be dropped silently.

    The terminal candidate shares `R = 10` with its predecessor (a zero-width
    link) and strictly dominates the left limit (value 2 vs 1). No
    linearly-interpolated row can represent a single-abscissa spike, so the
    kernel must signal through `n_kept > n_refined` instead of publishing a
    row that understates the envelope at an actual candidate node.
    """
    _, _, _, n_kept = refine_envelope(
        endog_grid=jnp.array([9.0, 10.0, 10.0]),
        policy=jnp.array([1.0, 1.0, 0.5]),
        value=jnp.array([0.0, 1.0, 2.0]),
        n_refined=12,
    )

    assert int(n_kept) > 12


def test_a_redundant_duplicate_candidate_does_not_overflow():
    """A duplicate-abscissa candidate matching the link record stays quiet.

    When the repeated candidate carries the same value the covering link
    already publishes, nothing is lost — the row is exact and the kernel must
    not report overflow.
    """
    refined_grid, _, refined_value, n_kept = refine_envelope(
        endog_grid=jnp.array([9.0, 10.0, 10.0]),
        policy=jnp.array([1.0, 1.0, 1.0]),
        value=jnp.array([0.0, 1.0, 1.0]),
        n_refined=12,
    )

    assert int(n_kept) <= 12
    np.testing.assert_allclose(
        _interp(refined_grid, refined_value, 9.5), 0.5, atol=_ATOL
    )


def test_a_finite_value_decrease_split_reports_no_read_support():
    """A coverage gap from a finite value drop withholds the row from the read.

    The inferred-segment split kills the middle link of `[1, 2, 4, 5]` (its
    value drops 11 -> 0), leaving no live link on `(2, 4)`. The compacted row
    bridges that interval, so the refinement must report
    `read_supported = False`.
    """
    *_, read_supported = refine_envelope_with_support(
        endog_grid=jnp.array([1.0, 2.0, 4.0, 5.0]),
        policy=jnp.array([1.0, 1.0, 0.5, 0.5]),
        value=jnp.array([10.0, 11.0, 0.0, 2.0]),
        n_refined=16,
    )

    assert not bool(read_supported)


def test_a_dead_candidate_split_reports_no_read_support():
    """A coverage gap from NaN-dead candidates also withholds the read.

    Dead candidates split the live chain into `[1, 2]` and `[3, 4]`; the row
    compacts across `(2, 3)` (the solve-side contract), so the refinement
    reports `read_supported = False` for the off-grid read.
    """
    *_, read_supported = refine_envelope_with_support(
        endog_grid=jnp.array([1.0, 2.0, jnp.nan, 3.0, 4.0]),
        policy=jnp.array([0.3, 0.6, jnp.nan, 0.9, 1.2]),
        value=jnp.array([0.0, 0.7, jnp.nan, 1.1, 1.4]),
        n_refined=12,
    )

    assert not bool(read_supported)


def test_a_gapless_correspondence_reports_read_support():
    """A fully covered two-branch crossing correspondence supports the read."""
    r_a = np.array([0.6, 1.2, 1.8, 2.4])
    r_b = np.array([0.8, 1.6, 2.6, 3.4])
    grid = jnp.asarray(np.concatenate([r_a, r_b]))
    value = jnp.asarray(np.concatenate([1.0 + 0.4 * r_a, 0.1 + 0.8 * r_b]))
    policy = jnp.asarray(np.concatenate([0.30 * r_a, 0.55 * r_b]))

    *_, read_supported = refine_envelope_with_support(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    assert bool(read_supported)


def test_the_mss_backend_propagates_the_read_support_flag():
    """The configured MSS backend returns the refinement's support verdict."""
    solver = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        upper_envelope="mss",
    )
    backend = get_upper_envelope(solver=solver, n_refined=16)

    *_, gapped = backend(
        endog_grid=jnp.array([1.0, 2.0, 4.0, 5.0]),
        policy=jnp.array([1.0, 1.0, 0.5, 0.5]),
        value=jnp.array([10.0, 11.0, 0.0, 2.0]),
        marginal_utility=jnp.ones(4),
    )
    *_, clean = backend(
        endog_grid=jnp.array([1.0, 2.0, 3.0, 4.0]),
        policy=jnp.array([0.3, 0.6, 0.9, 1.2]),
        value=jnp.array([0.0, 0.7, 1.1, 1.4]),
        marginal_utility=jnp.ones(4),
    )

    assert not bool(gapped)
    assert bool(clean)


def test_non_certified_backends_report_no_read_support():
    """FUES rows are never certified for the off-grid read."""
    solver = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        upper_envelope="fues",
    )
    backend = get_upper_envelope(solver=solver, n_refined=16)

    *_, read_supported = backend(
        endog_grid=jnp.array([1.0, 2.0, 3.0, 4.0]),
        policy=jnp.array([0.3, 0.6, 0.9, 1.2]),
        value=jnp.array([0.0, 0.7, 1.1, 1.4]),
        marginal_utility=jnp.ones(4),
    )

    assert not bool(read_supported)
