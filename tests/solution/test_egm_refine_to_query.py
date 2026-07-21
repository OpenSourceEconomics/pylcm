"""Spec for streamed refine-to-query in asset-row mode.

In asset-row mode the per-node solve refines a full NaN-padded upper envelope
and interpolates it at exactly one query point (`resources_at_node`) to publish
a scalar `(V_node, policy_node)`. Refine-to-query folds that single-query
interpolation into the upper-envelope scan: the scan keeps only the two
envelope points bracketing the query (plus the first point and the kept count)
and returns the scalar result, so the `n_pad` envelope rows are never
materialized.

Correctness is defined by the existing composition: for the *same* candidates,
query, and utility, the streamed pair

    refine_to_bracket(...) -> publish_node_from_bracket(...)

must equal the row-then-interpolate pair

    refine(...) -> _publish_node_V_and_policy(...)

to floating-point tolerance. This is a pure unit equivalence — no brute oracle.
The battery covers smooth, kinked, multi-crossing, all-dead, single-live, and
overflow candidate sets, with the query below the first point, above the last
point, and exactly on a duplicated kink abscissa (the `side="right"` tie-break,
the dominant correctness risk).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.asset_row import (
    _publish_node_V_and_policy,
    publish_node_from_bracket,
)
from _lcm.egm.upper_envelope import fues
from tests.conftest import X64_ENABLED

_ATOL = 1e-10 if X64_ENABLED else 1e-5

# CRRA utility with a node-dependent additive shift, so the value Hermite slope
# `grad(utility_of_action)` is a genuine non-trivial function of the policy.
_GAMMA = 1.5
_BORROWING_LIMIT = 0.0
_DISCOUNTED_EV_AT_LIMIT = -3.0


def _utility_of_action(action_value):
    safe = jnp.maximum(action_value, jnp.finfo(action_value.dtype).tiny)
    return safe ** (1.0 - _GAMMA) / (1.0 - _GAMMA)


def _crossing_segments_candidates():
    """Two crossing linear value segments (a discrete-choice kink at R* = 2.25)."""
    r_a = np.array([0.6, 1.2, 1.8, 2.4])
    r_b = np.array([0.8, 1.6, 2.6, 3.4])
    grid = np.concatenate([r_a, r_b])
    value = np.concatenate([1.0 + 0.4 * r_a, 0.1 + 0.8 * r_b])
    policy = np.concatenate([0.30 * r_a, 0.55 * r_b])
    return jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value)


def _smooth_candidates():
    """A single monotone-concave segment (no kink, passes through unchanged)."""
    grid = jnp.linspace(0.5, 6.0, 10)
    policy = 0.4 * grid
    value = jnp.log(grid)
    return grid, policy, value


def _multi_crossing_candidates():
    """Three linear segments with two crossings."""
    r_a = np.array([0.4, 0.9, 1.4])
    r_b = np.array([0.7, 1.2, 1.9, 2.4])
    r_c = np.array([2.1, 2.8, 3.5])
    grid = np.concatenate([r_a, r_b, r_c])
    value = np.concatenate([0.5 + 0.9 * r_a, 0.2 + 1.1 * r_b, -0.6 + 1.5 * r_c])
    policy = np.concatenate([0.2 * r_a, 0.45 * r_b, 0.7 * r_c])
    return jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value)


def _all_dead_candidates():
    """Every candidate infeasible: the value-correspondence is all `-inf`."""
    grid = jnp.linspace(0.5, 4.0, 6)
    policy = 0.3 * grid
    value = jnp.full_like(grid, -jnp.inf)
    return grid, policy, value


def _single_live_candidate():
    """Exactly one live candidate; the rest infeasible (`-inf`)."""
    grid = jnp.array([1.5, 2.0, 2.5, 3.0])
    policy = jnp.array([0.6, 0.7, 0.8, 0.9])
    value = jnp.array([2.0, -jnp.inf, -jnp.inf, -jnp.inf])
    return grid, policy, value


def _node_aligned_crossing_candidates():
    """Two linear branches crossing exactly on the grid node R = 10.

    Branch A (v = 1 + 0.4 R, c = 4 + 0.5 R) and branch B (v = -3 + 0.8 R,
    c = -1 + 0.5 R) meet at R = 10 with equal value 5.0 but different policy;
    supplied in exogenous-savings order on a non-monotone grid. The streamed
    bracket must reproduce the full row's reinserted right-branch policy.
    """
    grid = jnp.asarray([9.0, 10.0, 12.0, 10.0, 11.0])
    policy = jnp.asarray([8.5, 9.0, 9.0, 4.0, 4.5])
    value = jnp.asarray([4.6, 5.0, -100.0, 5.0, 5.8])
    return grid, policy, value


def _two_crossing_chain_candidates():
    """Three affine branches A|B|C crossing exactly on nodes 10 and 20.

    Branch B (c = 4) is optimal across (10, 20); the chain of two node-aligned
    crossings must keep B's middle segment, so at R = 15 the envelope policy is
    B's `4.0`. Supplied in exogenous-savings order on a non-monotone grid; the
    streamed bracket must reproduce the full row's middle branch.
    """
    grid = jnp.asarray([9.0, 10.0, 10.0, 20.0, 20.0, 21.0])
    policy = jnp.asarray([8.0, 8.0, 4.0, 4.0, 2.0, 2.0])
    value = jnp.asarray([4.875, 5.0, 5.0, 7.5, 7.5, 8.0])
    return grid, policy, value


_CANDIDATE_SETS = {
    "smooth": _smooth_candidates(),
    "kinked": _crossing_segments_candidates(),
    "multi_crossing": _multi_crossing_candidates(),
    "node_aligned_crossing": _node_aligned_crossing_candidates(),
    "two_crossing_chain": _two_crossing_chain_candidates(),
    "all_dead": _all_dead_candidates(),
    "single_live": _single_live_candidate(),
}


def _kink_abscissa(grid, policy, value, n_pad):
    """Return the lower duplicated-kink abscissa of a refined envelope, or None."""
    refined_grid, _, _, _ = fues.refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=n_pad
    )
    raw = np.asarray(refined_grid)
    clean = raw[~np.isnan(raw)]
    duplicates = clean[:-1][np.isclose(np.diff(clean), 0.0, atol=1e-9)]
    return None if duplicates.size == 0 else float(duplicates[0])


def _reference(grid, policy, value, x_query, n_pad):
    """Row-then-interpolate: refine to a full row, then publish the scalar."""
    dead = jnp.isneginf(value)
    refined_grid, refined_policy, refined_value, n_kept = fues.refine_envelope(
        endog_grid=jnp.where(dead, jnp.nan, grid),
        policy=jnp.where(dead, jnp.nan, policy),
        value=jnp.where(dead, jnp.nan, value),
        n_refined=n_pad,
    )
    return _publish_node_V_and_policy(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=n_kept,
        n_pad=n_pad,
        resources_at_node=jnp.asarray(x_query),
        borrowing_limit=jnp.asarray(_BORROWING_LIMIT),
        utility_of_action=_utility_of_action,
        discounted_expected_value_at_limit=jnp.asarray(_DISCOUNTED_EV_AT_LIMIT),
    )


def _streamed(grid, policy, value, x_query, n_pad):
    """Refine-to-bracket: fold the single query into the scan, then publish."""
    dead = jnp.isneginf(value)
    bracket = fues.refine_to_bracket(
        endog_grid=jnp.where(dead, jnp.nan, grid),
        policy=jnp.where(dead, jnp.nan, policy),
        value=jnp.where(dead, jnp.nan, value),
        x_query=jnp.asarray(x_query),
        n_refined=n_pad,
    )
    return publish_node_from_bracket(
        bracket=bracket,
        n_pad=n_pad,
        resources_at_node=jnp.asarray(x_query),
        borrowing_limit=jnp.asarray(_BORROWING_LIMIT),
        utility_of_action=_utility_of_action,
        discounted_expected_value_at_limit=jnp.asarray(_DISCOUNTED_EV_AT_LIMIT),
    )


def _assert_equivalent(grid, policy, value, x_query, n_pad):
    ref_v, ref_p = _reference(grid, policy, value, x_query, n_pad)
    got_v, got_p = _streamed(grid, policy, value, x_query, n_pad)
    np.testing.assert_allclose(float(got_v), float(ref_v), atol=_ATOL)
    # Policy is published as NaN exactly where the reference does (e.g. a poisoned
    # single-live / overflow read); compare NaN-for-NaN, value otherwise.
    if np.isnan(float(ref_p)):
        assert np.isnan(float(got_p))
    else:
        np.testing.assert_allclose(float(got_p), float(ref_p), atol=_ATOL)


@pytest.mark.parametrize("name", list(_CANDIDATE_SETS))
def test_streamed_matches_row_then_interp_on_interior_query(name):
    """The streamed publish equals row-then-interp at an interior query point."""
    grid, policy, value = _CANDIDATE_SETS[name]
    # The interior query is the midpoint of the candidate grid's span, which is a
    # finite point even for the all-dead set (whose envelope is empty).
    raw_grid = np.asarray(grid)
    x_query = float(0.5 * (raw_grid.min() + raw_grid.max()))
    _assert_equivalent(grid, policy, value, x_query, n_pad=16)


@pytest.mark.parametrize("name", list(_CANDIDATE_SETS))
def test_streamed_matches_row_then_interp_below_first_point(name):
    """A query below the lowest envelope point edge-clamps identically."""
    grid, policy, value = _CANDIDATE_SETS[name]
    x_query = float(np.min(np.asarray(grid))) - 1.0
    _assert_equivalent(grid, policy, value, x_query, n_pad=16)


@pytest.mark.parametrize("name", list(_CANDIDATE_SETS))
def test_streamed_matches_row_then_interp_above_last_point(name):
    """A query above the highest envelope point edge-clamps identically."""
    grid, policy, value = _CANDIDATE_SETS[name]
    x_query = float(np.max(np.asarray(grid))) + 1.0
    _assert_equivalent(grid, policy, value, x_query, n_pad=16)


@pytest.mark.parametrize(
    "name",
    ["kinked", "multi_crossing", "node_aligned_crossing", "two_crossing_chain"],
)
def test_streamed_matches_row_then_interp_exactly_on_kink(name):
    """A query exactly on a duplicated kink abscissa resolves the right-copy.

    The `side="right"` tie-break puts the bracket's lower node on the right
    copy of the kink, so the streamed publish must agree bit-for-bit with the
    row-then-interpolate publish there — the dominant correctness risk.
    """
    grid, policy, value = _CANDIDATE_SETS[name]
    x_query = _kink_abscissa(grid, policy, value, n_pad=16)
    assert x_query is not None, "fixture must produce a duplicated kink abscissa"
    _assert_equivalent(grid, policy, value, x_query, n_pad=16)


def test_streamed_matches_row_then_interp_on_overflow():
    """When the envelope overflows `n_pad`, both paths NaN-poison identically."""
    grid, policy, value = _crossing_segments_candidates()
    x_query = float(np.median(np.asarray(grid)))
    # `n_pad` below the envelope's kept-point count forces the overflow poison.
    got_v, got_p = _streamed(grid, policy, value, x_query, n_pad=4)
    ref_v, ref_p = _reference(grid, policy, value, x_query, n_pad=4)
    assert np.isnan(float(ref_v))
    assert np.isnan(float(got_v))
    assert np.isnan(float(ref_p))
    assert np.isnan(float(got_p))


def test_streamed_is_vmappable_over_nodes():
    """The streamed publish batches over query nodes like the per-node solve."""
    grid, policy, value = _crossing_segments_candidates()
    dead = jnp.isneginf(value)
    g = jnp.where(dead, jnp.nan, grid)
    p = jnp.where(dead, jnp.nan, policy)
    v = jnp.where(dead, jnp.nan, value)
    queries = jnp.linspace(0.7, 3.3, 9)

    def one(x_query):
        bracket = fues.refine_to_bracket(
            endog_grid=g, policy=p, value=v, x_query=x_query, n_refined=16
        )
        return publish_node_from_bracket(
            bracket=bracket,
            n_pad=16,
            resources_at_node=x_query,
            borrowing_limit=jnp.asarray(_BORROWING_LIMIT),
            utility_of_action=_utility_of_action,
            discounted_expected_value_at_limit=jnp.asarray(_DISCOUNTED_EV_AT_LIMIT),
        )

    batched_v, batched_p = jax.vmap(one)(queries)
    for k, x_query in enumerate(queries):
        single_v, single_p = one(x_query)
        np.testing.assert_allclose(float(batched_v[k]), float(single_v), atol=_ATOL)
        np.testing.assert_allclose(float(batched_p[k]), float(single_p), atol=_ATOL)
