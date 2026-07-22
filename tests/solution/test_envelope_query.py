"""The query-side envelope backend matches the host oracle exactly.

`envelope_at_query` evaluates the branch-aware upper envelope directly at query
abscissae. It must agree with the exact host oracle on value and policy across
the cases that distinguish the topology contract: a clean crossing, a folded
branch, and a non-bridging branch the inference backends get wrong.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.solution._envelope_oracle import exact_envelope


def _marginal(endog_grid, value, segment_id):
    """Per-node segment slope, the marginal a piecewise-linear branch carries."""
    grid = np.asarray(endog_grid)
    val = np.asarray(value)
    seg = np.asarray(segment_id)
    out = np.zeros_like(grid)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        order = idx[np.argsort(grid[idx])]
        if len(order) >= 2:
            slope = (val[order[1]] - val[order[0]]) / (grid[order[1]] - grid[order[0]])
            out[idx] = slope
    return jnp.asarray(out)


@pytest.mark.parametrize(
    ("endog_grid", "policy", "value", "segment_id", "x_query"),
    [
        # On-grid crossing of two branches at R=11.
        (
            [10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
            [3.0, 3.0, 3.0, 0.5, 0.5, 0.5],
            [5 / 3, 2.0, 7 / 3, 0.0, 2.0, 4.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [10.0, 10.5, 11.0, 11.1, 11.5, 12.0],
        ),
        # A and B disjoint, C a separate branch between them (the non-bridging case).
        (
            [0.0, 1.0, 2.0, 3.0, 1.5, 1.75],
            [0.0, 0.0, 10.0, 10.0, 5.0, 5.0],
            [0.0, 1.0, 4.0, 5.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [0.5, 1.5, 1.75, 2.5],
        ),
    ],
)
def test_query_envelope_matches_oracle(endog_grid, policy, value, segment_id, x_query):
    """Value and policy from the query backend equal the exact oracle."""
    endog_grid = jnp.asarray(endog_grid)
    policy = jnp.asarray(policy)
    value = jnp.asarray(value)
    segment_id = jnp.asarray(segment_id)
    x_query = jnp.asarray(x_query)

    got_value, got_policy, _got_marginal = envelope_at_query(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=_marginal(endog_grid, value, segment_id),
        segment_id=segment_id,
        x_query=x_query,
    )
    oracle_value, oracle_policy, _winner = exact_envelope(
        endog_grid=np.asarray(endog_grid),
        value=np.asarray(value),
        policy=np.asarray(policy),
        segment_id=np.asarray(segment_id),
        x_query=np.asarray(x_query),
    )

    np.testing.assert_allclose(np.asarray(got_value), oracle_value, atol=1e-9)
    np.testing.assert_allclose(np.asarray(got_policy), oracle_policy, atol=1e-9)


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_blocked_segment_scan_matches_the_dense_reduction(block_size):
    """`segment_block_size` is a memory knob: same value, policy, marginal.

    The two-pass blocked scan reproduces the dense `(n_query, n_segment)`
    reduction — same envelope value, same right-continuous tie-break winner — for
    any block size (divisor or not) below the segment count, up to floating-point
    reassociation between the two XLA lowerings.
    """
    rng = np.random.default_rng(20260626)
    # Three interleaved branches over a shared resource range, so several
    # segments bracket each query and the envelope max is contested.
    grids, values, policies, segments = [], [], [], []
    for seg, (intercept, slope_v) in enumerate([(1.0, 0.4), (0.1, 0.8), (0.6, 0.2)]):
        r = np.sort(rng.uniform(0.5, 3.5, size=6))
        grids.append(r)
        values.append(intercept + slope_v * r)
        policies.append(0.25 * (seg + 1) * r)
        segments.append(np.full_like(r, float(seg)))
    endog_grid = jnp.asarray(np.concatenate(grids))
    value = jnp.asarray(np.concatenate(values))
    policy = jnp.asarray(np.concatenate(policies))
    segment_id = jnp.asarray(np.concatenate(segments))
    marginal = _marginal(endog_grid, value, segment_id)
    x_query = jnp.asarray(np.linspace(0.7, 3.3, 41))

    dense = envelope_at_query(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        x_query=x_query,
    )
    blocked = envelope_at_query(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        x_query=x_query,
        segment_block_size=block_size,
    )
    for dense_arr, blocked_arr in zip(dense, blocked, strict=True):
        np.testing.assert_allclose(
            np.asarray(blocked_arr), np.asarray(dense_arr), rtol=1e-12, atol=1e-12
        )


def test_exact_node_tie_selects_the_segment_that_continues_right():
    """At a node where one segment ends and another starts, the right-continuous
    winner is the one that continues to the right, even if the ending segment is
    steeper.

    Segment A spans [0, 1] with the larger value-slope (10) and policy 0; segment B
    spans [1, 2] with slope 1 and policy 1. Both bracket the shared node q=1 and
    attain the same value there, but only B is defined immediately to the right, so
    a `side="right"` read must publish B's policy and marginal, not A's.
    """
    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0]),
        policy=jnp.array([0.0, 0.0, 1.0, 1.0]),
        value=jnp.array([0.0, 10.0, 10.0, 11.0]),
        marginal=jnp.array([10.0, 10.0, 1.0, 1.0]),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0]),
        x_query=jnp.array(1.0),
    )
    assert np.isclose(float(value), 10.0)
    assert np.isclose(float(policy), 1.0)
    assert np.isclose(float(marginal), 1.0)


def test_query_outside_all_branches_is_nan():
    """A query beyond every branch's support yields NaN value/policy/marginal."""
    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([1.0, 2.0]),
        marginal=jnp.array([1.0, 1.0]),
        segment_id=jnp.array([0.0, 0.0]),
        x_query=jnp.array([0.0, 5.0]),
    )
    assert bool(np.isnan(np.asarray(got_value)).all())
    assert bool(np.isnan(np.asarray(got_policy)).all())
    assert bool(np.isnan(np.asarray(got_marginal)).all())
