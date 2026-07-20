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


def test_crossing_enumeration_is_invariant_to_a_common_value_shift(
    x64_enabled: None,
):
    """A cardinal value shift must not change crossings or winners.

    The three-branch multi-switch correspondence with `1e6` added to every
    value keeps its switches at `R = 10.3` and `R = 10.7` and the same policy
    read — the envelope is ordinal in value levels. The shifted geometry
    (sub-unit offsets at a `1e6` value level) is float64-representable only,
    so the test pins the x64 mode.
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


def test_a_common_value_shift_leaves_winner_and_policy_unchanged():
    """Adding a constant to every candidate value cannot change the read.

    The Bellman value is cardinal only up to a common additive normalization:
    two branch lines `0.5 + (R - 2000)/1000` (policy 1000) and
    `(R - 2000)/800` (policy 800) keep the same winner at `R = 2050` under
    any common value shift. In float32 a shift of `1e5` puts a few-ulp
    representational window at ~0.001 while the branch gap stays ~0.5, so the
    records must remain distinct and the read must track the true envelope.
    """

    def read_at_midpoint(shift: float) -> tuple[float, float, float, float]:
        origin, width = 2000.0, 100.0
        policies = np.array([1000.0, 800.0], dtype=np.float32)
        slopes = 1.0 / policies
        intercepts = np.array([0.5, 0.0], dtype=np.float32) + np.float32(shift)
        grid, policy, value = [], [], []
        for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
            grid += [origin, origin + width]
            policy += [action, action]
            value += [intercept, intercept + slope * width]
        got_grid, got_policy, got_value, _ = refine_envelope(
            endog_grid=jnp.asarray(grid, dtype=jnp.float32),
            policy=jnp.asarray(policy, dtype=jnp.float32),
            value=jnp.asarray(value, dtype=jnp.float32),
            n_refined=16,
        )
        query = jnp.asarray(origin + 50, dtype=jnp.float32)
        read_value = float(
            interp_on_padded_grid(x_query=query, xp=got_grid, fp=got_value)
        )
        read_policy = float(
            interp_on_padded_grid(x_query=query, xp=got_grid, fp=got_policy)
        )
        truth = intercepts + slopes * 50
        return (
            read_value,
            read_policy,
            float(truth.max()),
            float(policies[truth.argmax()]),
        )

    base = read_at_midpoint(0.0)
    shifted = read_at_midpoint(100000.0)
    np.testing.assert_allclose(base[0], base[2], rtol=0, atol=1e-6)
    np.testing.assert_allclose(base[1], base[3], rtol=0, atol=0)
    np.testing.assert_allclose(shifted[0], shifted[2], rtol=0, atol=0.02)
    np.testing.assert_allclose(shifted[1], shifted[3], rtol=0, atol=0)


def test_a_dominating_point_at_a_large_value_level_overflows_loudly():
    """The spike detector fires at any common value level.

    The zero-width dominating candidate `(10, 1000002)` exceeds both one-sided
    records by a full unit; at value level `1e6` in float32 a few-ulp margin
    is ~0.5, so the excess must still trigger `n_kept > n_refined`.
    """
    *_, n_kept, _ = refine_envelope_with_support(
        endog_grid=jnp.array([9.0, 10.0, 10.0], dtype=jnp.float32),
        policy=jnp.array([1.0, 1.0, 0.5], dtype=jnp.float32),
        value=jnp.array([1000000.0, 1000001.0, 1000002.0], dtype=jnp.float32),
        n_refined=12,
    )

    assert int(n_kept) > 12


def test_two_distinct_representable_crossings_are_both_emitted():
    """Close but representable crossings in a wide interval are all emitted.

    Three branches cross at local offsets 1 and 2 inside a `1e6`-wide float32
    interval. Both abscissae are exactly representable, so the switch
    sequence has two crossings (each duplicated) and the read strictly
    between them must follow the middle branch — a tie window proportional
    to the interval width would skip it.
    """
    origin, width = 4000000.0, 1000000.0
    policies = np.array([3100000.0, 2000000.0, 900000.0], dtype=np.float32)
    slopes = 1.0 / policies
    intercepts = np.array(
        [
            0.0,
            slopes[0] - slopes[1],
            slopes[0] - slopes[1] + 2.0 * (slopes[1] - slopes[2]),
        ],
        dtype=np.float32,
    )
    grid, policy, value = [], [], []
    for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
        grid += [origin, origin + width]
        policy += [action, action]
        value += [intercept, intercept + slope * width]

    got_grid, got_policy, _, n_kept = refine_envelope(
        endog_grid=jnp.asarray(grid, dtype=jnp.float32),
        policy=jnp.asarray(policy, dtype=jnp.float32),
        value=jnp.asarray(value, dtype=jnp.float32),
        n_refined=32,
    )

    local_grid = np.asarray(got_grid)[: int(n_kept)] - np.float32(origin)
    assert np.isclose(local_grid, 1.0, atol=0.25).sum() == 2
    assert np.isclose(local_grid, 2.0, atol=0.25).sum() == 2
    query = jnp.asarray(origin + 1.25, dtype=jnp.float32)
    read_policy = float(
        interp_on_padded_grid(x_query=query, xp=got_grid, fp=got_policy)
    )
    truth = intercepts + slopes * 1.25
    np.testing.assert_allclose(read_policy, float(policies[truth.argmax()]), rtol=1e-5)


def _read_two_branch_row_at_origin(shift: float) -> tuple[float, float]:
    """Refine a two-branch f32 correspondence and read value/policy at the origin.

    Branch values at the origin node differ by four units — a genuinely
    representable gap at every level used by the callers — so the published
    read must be the higher branch (value `4 + shift`, policy 1000) no matter
    what common cardinal level `shift` puts the values at.
    """
    origin = np.float32(2000.0)
    width = np.float32(100.0)
    policies = np.array([1000.0, 800.0], dtype=np.float32)
    slopes = np.float32(10000.0) / policies
    intercepts = np.array([4.0, 0.0], dtype=np.float32) + np.float32(shift)

    grid, policy, value = [], [], []
    for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
        grid.extend([origin, origin + width])
        policy.extend([action, action])
        value.extend([intercept, intercept + slope * width])

    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=jnp.asarray(np.asarray(grid, dtype=np.float32)),
        policy=jnp.asarray(np.asarray(policy, dtype=np.float32)),
        value=jnp.asarray(np.asarray(value, dtype=np.float32)),
        n_refined=16,
    )
    query = jnp.asarray(origin, dtype=jnp.float32)
    got_value = float(
        interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_value)
    )
    got_policy = float(
        interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_policy)
    )
    return got_value, got_policy


def test_a_large_common_value_shift_cannot_lower_the_published_maximum():
    """The published value is the exact stored maximum at any cardinal level.

    A representable four-unit branch gap survives a common shift of `1e7` in
    f32 (both shifted values are distinct floats), so the read must publish
    the higher branch's value and policy — never a near-maximal competitor's
    lower record.
    """
    base_value, base_policy = _read_two_branch_row_at_origin(0.0)
    shifted_value, shifted_policy = _read_two_branch_row_at_origin(10_000_000.0)

    np.testing.assert_allclose(base_value, 4.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(base_policy, 1000.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(shifted_value, 10_000_004.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(shifted_policy, 1000.0, rtol=0.0, atol=0.0)


def test_two_distinct_crossings_at_large_local_offsets_are_both_emitted():
    """Crossings one unit apart at local offset `1e6` are enumerated in order.

    At that offset f32 spacing is `0.0625`, so the two crossing abscissae are
    sixteen ulp apart — genuinely distinct points whose middle branch owns the
    envelope between them. A tie may coalesce crossings only when the lines
    actually meet at the same numerical point; here the third line sits about
    twenty ulp of the value scale below the winner at the first crossing, so
    the scan must emit both switches and the strictly-between read must
    follow the middle branch.
    """
    origin = np.float32(5_000_000.0)
    width = np.float32(1_000_010.0)
    policies = np.array([4_000_000.0, 2_500_000.0, 1_000_000.0], dtype=np.float32)
    slopes = 1.0 / policies
    first_crossing = np.float32(1_000_000.0)
    second_crossing = np.float32(1_000_001.0)
    intercepts = np.array(
        [
            0.0,
            (slopes[0] - slopes[1]) * first_crossing,
            (slopes[0] - slopes[1]) * first_crossing
            + (slopes[1] - slopes[2]) * second_crossing,
        ],
        dtype=np.float32,
    )

    grid, policy, value = [], [], []
    for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
        grid.extend([origin, origin + width])
        policy.extend([action, action])
        value.extend([intercept, intercept + slope * width])

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=jnp.asarray(np.asarray(grid, dtype=np.float32)),
        policy=jnp.asarray(np.asarray(policy, dtype=np.float32)),
        value=jnp.asarray(np.asarray(value, dtype=np.float32)),
        n_refined=32,
    )
    local_grid = np.asarray(refined_grid)[: int(n_kept)] - origin

    assert np.isclose(local_grid, first_crossing, rtol=0.0, atol=0.25).sum() == 2
    assert np.isclose(local_grid, second_crossing, rtol=0.0, atol=0.25).sum() == 2

    local_query = np.float32(1_000_000.5)
    query = jnp.asarray(origin + local_query, dtype=jnp.float32)
    got_value = float(
        interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_value)
    )
    got_policy = float(
        interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_policy)
    )
    truth = intercepts + slopes * local_query
    np.testing.assert_allclose(got_value, float(truth.max()), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(got_policy, float(policies[truth.argmax()]))


def test_two_distinct_crossings_survive_a_common_value_baseline():
    """A common value baseline must not merge distinct crossings.

    The two-switch correspondence with crossings at local offsets `1e6` and
    `1e6 + 1` carries a `1e6` baseline on every value. Simultaneity
    certification is translation-invariant — it is formed from local value
    gaps, where the baseline cancels before rounding — so both switches stay
    emitted and the strictly-between read follows the middle branch, exactly
    as in the unshifted geometry.
    """
    origin = np.float32(5_000_000.0)
    width = np.float32(1_000_010.0)
    policies = np.array([4_000_000.0, 2_500_000.0, 1_000_000.0], dtype=np.float32)
    slopes = np.float32(1_000_000.0) / policies
    first_crossing = np.float32(1_000_000.0)
    second_crossing = np.float32(1_000_001.0)
    intercepts = np.array(
        [
            0.0,
            (slopes[0] - slopes[1]) * first_crossing,
            (slopes[0] - slopes[1]) * first_crossing
            + (slopes[1] - slopes[2]) * second_crossing,
        ],
        dtype=np.float32,
    ) + np.float32(1_000_000.0)

    grid, policy, value = [], [], []
    for slope, intercept, action in zip(slopes, intercepts, policies, strict=True):
        grid.extend([origin, origin + width])
        policy.extend([action, action])
        value.extend([intercept, intercept + slope * width])

    refined_grid, refined_policy, _refined_value, n_kept = refine_envelope(
        endog_grid=jnp.asarray(np.asarray(grid, dtype=np.float32)),
        policy=jnp.asarray(np.asarray(policy, dtype=np.float32)),
        value=jnp.asarray(np.asarray(value, dtype=np.float32)),
        n_refined=32,
    )
    local_grid = np.asarray(refined_grid)[: int(n_kept)] - origin

    assert np.isclose(local_grid, first_crossing, rtol=0.0, atol=0.25).sum() == 2
    assert np.isclose(local_grid, second_crossing, rtol=0.0, atol=0.25).sum() == 2

    local_query = np.float32(1_000_000.5)
    query = jnp.asarray(origin + local_query, dtype=jnp.float32)
    got_policy = float(
        interp_on_padded_grid(x_query=query, xp=refined_grid, fp=refined_policy)
    )
    np.testing.assert_allclose(got_policy, 2_500_000.0)


def test_a_point_spike_a_few_ulp_above_an_exact_side_record_overflows():
    """A representable strict point maximum overflows, never vanishes.

    The terminal candidate at `R = 10` sits four float32 ULP (at the value
    level) above the covering link's stored endpoint record. The row cannot
    carry a single-abscissa spike, and the stored gap is representable, so
    the kernel signals `n_kept > n_refined` instead of publishing the lower
    side record as if the point did not exist.
    """
    *_, n_kept, _ = refine_envelope_with_support(
        endog_grid=jnp.array([9.0, 10.0, 10.0], dtype=jnp.float32),
        policy=jnp.array([1.0, 1.0, 0.5], dtype=jnp.float32),
        value=jnp.array([10_000_000.0, 10_000_100.0, 10_000_104.0], dtype=jnp.float32),
        n_refined=12,
    )

    assert int(n_kept) > 12


def test_a_point_spike_above_a_long_link_interior_read_overflows():
    """A stored point strictly above a covering link's chord overflows, never vanishes.

    The third candidate's abscissa lies two grid units inside a float32 link
    spanning `[2e7, 3e7]` with values `[-3e7, 3]`. The link's exact chord at
    that abscissa is below the stored point value `-3`, so the point is a
    strict unrepresented maximum. Evaluating the chord from the link's far
    endpoint cancels catastrophically and rounds the line read a full unit
    above the point; the interior read is anchored at the nearer stored
    endpoint instead, so the strict stored excess is detected and the kernel
    signals `n_kept > n_refined`.
    """
    *_, n_kept, _ = refine_envelope_with_support(
        endog_grid=jnp.array(
            [20_000_000.0, 30_000_000.0, 29_999_998.0], dtype=jnp.float32
        ),
        policy=jnp.array([19_000_000.0, 20_000_000.0, 1.0], dtype=jnp.float32),
        value=jnp.array([-30_000_000.0, 3.0, -3.0], dtype=jnp.float32),
        n_refined=12,
    )

    assert int(n_kept) > 12


def test_a_long_span_link_publishes_its_stored_endpoint_exactly():
    """A candidate endpoint is read from its stored record, bit-exactly.

    One live float32 link spans `[2e7, 3e7]` with values `[-3e7, 3]`.
    Reconstructing the upper endpoint as `anchor + slope * span` loses the
    stored `3.0` to cancellation, so endpoint queries snap to the stored
    endpoint records instead of the affine reconstruction.
    """
    grid = jnp.array([20_000_000.0, 30_000_000.0], dtype=jnp.float32)
    refined_grid, refined_policy, refined_value, _ = refine_envelope(
        endog_grid=grid,
        policy=jnp.array([10_000_000.0, 1.0], dtype=jnp.float32),
        value=jnp.array([-30_000_000.0, 3.0], dtype=jnp.float32),
        n_refined=8,
    )
    got_value = float(
        interp_on_padded_grid(x_query=grid[-1], xp=refined_grid, fp=refined_value)
    )
    got_policy = float(
        interp_on_padded_grid(x_query=grid[-1], xp=refined_grid, fp=refined_policy)
    )

    assert got_value == 3.0
    assert got_policy == 1.0
