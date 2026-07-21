"""Regression locks for the external FUES correctness audit (findings F1 to F4).

Each test asserts the corrected behavior directly: the duplicate-abscissae
collapse (audit F1), the `segment_id` switch hook (audit F2), and the
interleaved-segments case (audit F4) — where the default scan is exhaustive, so
however many off-segment candidates interleave between two points of one
segment, the scan still reaches the segment's continuation and rejects the
dominated interlopers. A narrower explicit `n_points_to_scan` is an opt-in speed
knob that gives up that guarantee.

Ground truth is the interpolated envelope *function*: the refined rows exist to
be read by `interp_on_padded_grid` downstream, so correctness is judged there,
not on which raw points survive.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.step_core import _publish_V_and_carry_rows
from _lcm.egm.upper_envelope.fues import (
    _intersect_lines,
    refine_envelope,
)
from _lcm.egm.upper_envelope.mss import refine_envelope as refine_envelope_mss
from tests.conftest import X64_ENABLED

_ATOL = 1e-8 if X64_ENABLED else 1e-5


def _kept(grid, *arrays):
    """Drop the NaN-padded tail, returning the kept prefix of each array."""
    keep = ~np.isnan(np.asarray(grid))
    return (np.asarray(grid)[keep], *(np.asarray(a)[keep] for a in arrays))


def _read_value(grid, policy, value, query):
    """Hermite value read with node slopes `1/c` (the production convention)."""
    slopes = jnp.where(jnp.isnan(policy), jnp.nan, 1.0 / policy)
    return float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=grid, fp=value, fp_slopes=slopes
        )
    )


def _read_policy(grid, policy, query):
    """Linear (right-continuous at a duplicated kink) policy read."""
    return float(interp_on_padded_grid(x_query=jnp.asarray(query), xp=grid, fp=policy))


# ---------------------------------------------------------------------------
# F3 — a strictly dominated point at a shared abscissa must not survive
# ---------------------------------------------------------------------------


def test_f3_dominated_duplicate_point_is_not_retained():
    """At a duplicated abscissa, no kept point carries a strictly lower value.

    A genuine envelope kink duplicates an abscissa with *equal* value and two
    policies; a dominated duplicate carries a *lower* value and must be dropped
    by the pre-scan collapse of equal-abscissa candidates to their max value.
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


def test_f3_interior_duplicate_collapses_to_max_value():
    """An interior duplicate abscissa keeps only its maximal-value candidate.

    The reviewer's counterexample: two candidates at `R=1` with values `0` and
    `1` (the lower one listed first). The refined envelope must keep `(1, 1)`
    and drop `(1, 0)`, so an interpolated read at `R=1` returns the envelope
    value `1`, not the dominated `0` that index-ordered interpolation would
    otherwise surface for queries just below the duplicate.
    """
    grid, _policy, value, n_kept = refine_envelope(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0]),
        policy=jnp.array([0.0, 0.5, 0.2, 1.0]),
        value=jnp.array([0.0, 0.0, 1.0, 2.0]),
        n_refined=10,
    )
    g, v = _kept(grid[: int(n_kept)], value[: int(n_kept)])
    at_one = np.isclose(g, 1.0, atol=_ATOL)
    assert at_one.sum() == 1, f"R=1 kept {at_one.sum()} times, expected 1"
    assert np.isclose(v[at_one][0], 1.0, atol=_ATOL), (
        f"R=1 kept value {v[at_one]} — dominated 0.0 not collapsed"
    )
    # The interpolated function just below the duplicate must not dip to the
    # dominated value.
    just_below = interp_on_padded_grid(x_query=jnp.array([0.999]), xp=grid, fp=value)
    assert float(just_below[0]) > 0.5, (
        f"interpolated value just below R=1 is {float(just_below[0])} — "
        "reads toward the dominated duplicate"
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
    # Envelope over [0, 1] is the line through (0, 1)-(1, 2); the read
    # continues its secant below the first node; policy is 1.
    expected_value = 1.0 + np.asarray(x_query)
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


def test_f4_interleaved_segments_give_analytic_envelope_at_default_scan():
    """The refined envelope equals the upper line A(x)=x at the default scan.

    The default scan is exhaustive, so it reaches segment A's continuation at
    `x=12` however many off-segment candidates interleave before it, and rejects
    every dominated point.
    """
    endog_grid, policy, value = _interleaved_segments()
    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid, policy=policy, value=value, n_refined=64
    )
    x_query = jnp.linspace(0.0, 12.0, 13)
    got = interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    np.testing.assert_allclose(np.asarray(got), np.asarray(x_query), atol=1e-6)


def test_f4_bounded_scan_underscans_when_window_too_small():
    """An explicit finite scan narrower than the interleave accepts the interlopers.

    The exhaustive default is the correctness guarantee; the finite window is an
    opt-in speed knob. On this fixture — 11 off-segment points between segment A's
    two anchors — a window of 10 cannot reach A's continuation, so it keeps the
    dominated points and the interpolated envelope sits a uniform 0.5 below the
    true line A(x)=x. This pins the documented tradeoff of the bounded mode.
    """
    endog_grid, policy, value = _interleaved_segments()
    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=64,
        n_points_to_scan=10,
    )
    x_query = jnp.linspace(0.0, 12.0, 13)
    got = interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    max_deviation = float(np.max(np.abs(np.asarray(got) - np.asarray(x_query))))
    np.testing.assert_allclose(max_deviation, 0.5, atol=1e-6)


def test_segment_id_label_forces_a_switch_a_flat_policy_notch_misses():
    """A `segment_id` label detects a switch the policy-slope test cannot.

    Two segments with the *same* policy law (`c = 0.5 R`, so no implied-savings
    jump anywhere) cross in value. Without labels the scan reads one smooth
    segment and drops the lower-value point. With labels it resolves the
    crossing — inserting the intersection twice as an equal-value kink, the
    correct upper envelope at a notch.
    """
    endog_grid = jnp.array([1.0, 1.5, 2.0, 2.5])
    policy = 0.5 * endog_grid
    value = jnp.array([2.0, 1.8, 2.2, 3.0])

    g0, _, _, n0 = refine_envelope(
        endog_grid=endog_grid, policy=policy, value=value, n_refined=12
    )
    g1, _, v1, n1 = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=12,
        segment_id=jnp.array([0.0, 1.0, 0.0, 1.0]),
    )

    # Unlabelled: one smooth segment, no kink inserted.
    grid0 = np.asarray(g0[: int(n0)])
    assert len(np.unique(grid0)) == len(grid0), "unlabelled output has a kink"

    # Labelled: a crossing kink — exactly one duplicated abscissa, equal values.
    grid1, val1 = np.asarray(g1[: int(n1)]), np.asarray(v1[: int(n1)])
    uniq, counts = np.unique(grid1, return_counts=True)
    kink = uniq[counts == 2]
    assert kink.size == 1, f"labelled output kinks: {grid1.tolist()}"
    at_kink = np.isclose(grid1, kink[0], atol=_ATOL)
    assert val1[at_kink].max() - val1[at_kink].min() <= _ATOL, (
        "inserted kink must carry equal values"
    )


def test_f5_exact_grid_aligned_branch_crossing_gives_v_shaped_envelope():
    """A branch crossing exactly on a grid node yields the true V-shaped envelope.

    Branch A is `V_A(R) = R` (policy 0) and branch B is `V_B(R) = 1 - R`
    (policy 10), sampled at `R in {0, 0.5, 1}`. They cross exactly at the existing
    node `R = 0.5`, where both have value `0.5` and carry different one-sided
    policies. The correct upper envelope is `max(R, 1 - R)` — a V with its minimum
    `0.5` at `R = 0.5` — not the flat `1` an envelope that deletes the crossing
    node produces.
    """
    endog_grid = jnp.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    value = jnp.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    policy = jnp.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    segment_id = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    grid, _, refined_value, _ = refine_envelope(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        n_refined=16,
        segment_id=segment_id,
    )
    x_query = jnp.linspace(0.0, 1.0, 5)
    got = interp_on_padded_grid(x_query=x_query, xp=grid, fp=refined_value)
    expected = np.maximum(np.asarray(x_query), 1.0 - np.asarray(x_query))
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-6)


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


def test_carry_value_row_is_the_unfloored_envelope_while_published_v_is_floored():
    """The carry value row stays the refined envelope; only `V_row` is floored.

    When the closed-form credit-constrained value at an exogenous node exceeds
    the refined-envelope read there, the published `V_row` is lifted to that
    constrained value (a feasible-policy floor). The carry value row — the object
    a parent interpolates as its continuation — must remain the raw refined
    envelope: flooring it would lift finite envelope nodes above the true
    envelope and overstate the parent's continuation at low wealth.
    """
    n_pad = 5
    refined_grid = jnp.array([1.0, 2.0, 3.0, jnp.nan, jnp.nan])
    refined_policy = jnp.array([1.0, 2.0, 3.0, jnp.nan, jnp.nan])
    # An envelope far below the constrained value at every node, so the floor
    # binds on `V_row` at the queried node.
    refined_value = jnp.array([-10.0, -10.0, -10.0, jnp.nan, jnp.nan])

    V_row, value_row, _marginal = _publish_V_and_carry_rows(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=jnp.int32(3),
        n_pad=n_pad,
        publish_resources=jnp.array([2.0]),
        borrowing_limit=jnp.asarray(0.0),
        utility_of_action=jnp.log,
        discounted_expected_value_at_limit=jnp.asarray(0.0),
    )

    # Published V is lifted to the constrained value u(R - limit) = log(2).
    np.testing.assert_allclose(np.asarray(V_row), [np.log(2.0)], atol=_ATOL)
    # The carry value row is the raw refined envelope, NOT floored: it stays the
    # under-floor envelope values, strictly below the published (floored) V.
    np.testing.assert_allclose(np.asarray(value_row)[:3], [-10.0, -10.0, -10.0])
    assert float(value_row[1]) < float(V_row[0])


def test_intersect_lines_parallel_branch_keeps_gradients_finite():
    """Reverse-mode gradients stay finite through the parallel-lines dead branch.

    Parallel candidate lines have no intersection, so the returned abscissa is
    NaN by contract. The ordinate is evaluated from a finite abscissa, so a
    gradient taken through this branch (e.g. an autodiff'd solve) does not get
    poisoned by the NaN the forward result intentionally carries.
    """

    zero, one, two, three = (jnp.asarray(v) for v in (0.0, 1.0, 2.0, 3.0))

    def ordinate(y_a):
        # slope_a == slope_b ⇒ parallel ⇒ the dead branch (abscissa is NaN).
        _x, y = _intersect_lines(
            x_a=zero, y_a=y_a, slope_a=two, x_b=one, y_b=three, slope_b=two
        )
        return y

    x, _y = _intersect_lines(
        x_a=zero, y_a=one, slope_a=two, x_b=one, y_b=three, slope_b=two
    )
    assert bool(jnp.isnan(x))  # the abscissa contract: NaN for parallel lines
    assert bool(jnp.isfinite(jax.grad(ordinate)(one)))


def test_segment_crossing_on_a_node_emits_both_branch_policies():
    """A branch crossing exactly on one branch's sampled node keeps both policies.

    Branch A (`c=8`, `V=4.875+.125(R-9)`) is sampled at R=9,10; branch B (`c=2`,
    `V=4.75+.5(R-9.5)`) at R=9.5,10.5. The two lines meet exactly at R=10, which
    is A's endpoint — B spans the node without a candidate there. The refined row
    must carry both R=10 policies (left owner 8, right owner 2), so the read just
    right of the node is the right branch's `c=2`, not an interpolation across the
    collapsed discontinuity.
    """
    grid = jnp.asarray([9.0, 10.0, 9.5, 10.5])
    policy = jnp.asarray([8.0, 8.0, 2.0, 2.0])
    value = jnp.asarray([4.875, 5.0, 4.75, 5.25])
    savings = grid - policy

    g, p, v, _n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=10
    )
    kept_grid, kept_policy = _kept(g, p)
    at_node = np.isclose(kept_grid, 10.0, atol=_ATOL)
    assert int(at_node.sum()) == 2
    np.testing.assert_allclose(sorted(kept_policy[at_node]), [2.0, 8.0], atol=_ATOL)
    np.testing.assert_allclose(_read_policy(g, p, 10.1), 2.0, atol=_ATOL)
    np.testing.assert_allclose(_read_value(g, p, v, 10.1), 5.05, atol=_ATOL)


def test_same_source_duplicates_collapse_across_an_interleaved_source():
    """A same-source duplicate collapses even with another source between the copies.

    At R=10 three candidates share value 5: two from source A (`c=8`, savings 2)
    and one from source B (`c=4`, savings 6), ordered A, B, A. The two A copies are
    the same exogenous source and collapse to one, so the group keeps A and B — the
    kept count is 4, not 5, and no false overflow occurs.
    """
    grid = jnp.asarray([9.0, 10.0, 10.0, 10.0, 11.0])
    policy = jnp.asarray([8.0, 8.0, 4.0, 8.0, 4.0])
    value = jnp.asarray([4.875, 5.0, 5.0, 5.0, 5.25])
    savings = jnp.asarray([1.0, 2.0, 6.0, 2.0, 7.0])

    _, _, _, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=10
    )
    assert int(n_kept) == 4


def test_exact_node_tie_ordering_is_invariant_to_input_order():
    """Two branches meeting at a node publish the same envelope under any input order.

    Branch A (`c=8`) and branch B (`c=4`) cross exactly at R=10 with equal value 5.
    A owns the interval to the left (shallower value slope), B to the right. The
    published value and policy read must not depend on which of the two coincident
    R=10 candidates appears first in the input: side ownership, not input order,
    fixes the left/right copies.
    """
    base_grid = jnp.asarray([9.0, 10.0, 10.0, 11.0])
    base_policy = jnp.asarray([8.0, 8.0, 4.0, 4.0])
    base_value = jnp.asarray([4.875, 5.0, 5.0, 5.25])

    reads = []
    for perm in ([0, 1, 2, 3], [0, 2, 1, 3]):
        order = jnp.asarray(perm)
        grid, policy, value = base_grid[order], base_policy[order], base_value[order]
        g, p, v, _ = refine_envelope(
            endog_grid=grid,
            policy=policy,
            value=value,
            savings=grid - policy,
            n_refined=10,
        )
        reads.append(
            (
                _read_value(g, p, v, 9.5),
                _read_policy(g, p, 9.5),
                _read_policy(g, p, 10.1),
            )
        )
    np.testing.assert_allclose(reads[1], reads[0], atol=_ATOL)


@pytest.mark.xfail(
    reason=(
        "Accepted known limitation of the FUES fast-scan (not a deferred fix): the "
        "coincident-group reducer keeps pointwise node maxima, so a branch lower at "
        "a shared node but owning the adjacent interval loses its slope anchor. The "
        "fast-scan lineage forbids coincident abscissae rather than solving them; "
        "the correct backend for exact coincident-node handling is `mss` — pinned by "
        "test_mss_resolves_the_coincident_interval_ownership. This strict xfail is a "
        "sentinel: it fails loudly if FUES ever silently changes here."
    ),
    strict=True,
)
def test_pointwise_lower_branch_that_owns_an_interval_is_retained():
    """FUES bridges a coincident-node crossing — the pinned fast-scan limitation.

    Branches A (`c=8`) and B (`c=4`) are both sampled at R=9,10 and cross at
    R=9.92. A is higher at R=9 and owns `[9, 9.92]`; B is higher at R=10 and owns
    `[9.92, ...]`. The exact read at R=9.5 is A's `(V,c)=(4.9375, 8)`. FUES keeps
    only the pointwise node maxima, drops A@10 and B@9, and bridges to B's read —
    so this assertion of the *correct* value is expected to fail on FUES. Use
    `upper_envelope="mss"` when exact coincident-node correctness is required.
    """
    grid = jnp.asarray([9.0, 10.0, 9.0, 10.0, 11.0])
    policy = jnp.asarray([8.0, 8.0, 4.0, 4.0, 4.0])
    value = jnp.asarray([4.875, 5.0, 4.76, 5.01, 5.26])
    savings = grid - policy

    g, p, v, _ = refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=12
    )
    np.testing.assert_allclose(_read_value(g, p, v, 9.5), 4.9375, atol=_ATOL)
    np.testing.assert_allclose(_read_policy(g, p, 9.5), 8.0, atol=_ATOL)


def test_mss_resolves_the_coincident_interval_ownership():
    """The `mss` backend reads the correct branch where FUES bridges the crossing.

    The escape hatch for the pinned FUES coincident-node limitation
    (`test_pointwise_lower_branch_that_owns_an_interval_is_retained`): the same
    two branches — A (`c=8`) owning `[9, 9.92]`, B (`c=4`) owning `[9.92, ...]`,
    both sampled at R=9,10 — read at R=9.5 through `mss` give A's exact
    `(V,c)=(4.9375, 8)`. The segment-envelope method keeps whole branches and
    resolves interval ownership by construction, so the pointwise-lower interval
    owner is never dropped.
    """
    grid = jnp.asarray([9.0, 10.0, 9.0, 10.0, 11.0])
    policy = jnp.asarray([8.0, 8.0, 4.0, 4.0, 4.0])
    value = jnp.asarray([4.875, 5.0, 4.76, 5.01, 5.26])

    g, p, v, _ = refine_envelope_mss(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )
    np.testing.assert_allclose(_read_value(g, p, v, 9.5), 4.9375, atol=_ATOL)
    np.testing.assert_allclose(_read_policy(g, p, 9.5), 8.0, atol=_ATOL)


def test_segment_crossing_on_the_later_node_emits_both_branch_policies():
    """The mirror of the endpoint crossing: the intersection lands on `grid_i`.

    Branch A (`c=8`) is sampled at R=9,9.5 and branch B (`c=2`) at R=10,11; their
    lines meet exactly at R=10, which is B's first sampled node. A owns the left,
    B the right. The refined row must carry both R=10 policies (left 8, right 2),
    so the read just left of the node is A's `c=8` and just right is B's `c=2`.
    """
    grid = jnp.asarray([9.0, 9.5, 10.0, 11.0])
    policy = jnp.asarray([8.0, 8.0, 2.0, 2.0])
    value = jnp.asarray([4.875, 4.9375, 5.0, 5.5])
    savings = grid - policy

    g, p, _v, _ = refine_envelope(
        endog_grid=grid, policy=policy, value=value, savings=savings, n_refined=12
    )
    kept_grid, kept_policy = _kept(g, p)
    at_node = np.isclose(kept_grid, 10.0, atol=_ATOL)
    assert int(at_node.sum()) == 2
    np.testing.assert_allclose(sorted(kept_policy[at_node]), [2.0, 8.0], atol=_ATOL)
    np.testing.assert_allclose(_read_policy(g, p, 9.9), 8.0, atol=_ATOL)
    np.testing.assert_allclose(_read_policy(g, p, 10.1), 2.0, atol=_ATOL)
