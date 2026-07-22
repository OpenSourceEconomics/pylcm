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


def test_inclusive_bracket_reads_the_boundary_owning_value_at_a_shared_abscissa():
    """At a duplicated abscissa the inclusive bracket picks the higher segment.

    Value jumps ride on carry rows as duplicated abscissae: two segments end and
    start at the same grid point, and reads at that point must see the segment
    whose value owns the boundary (the higher one). The bracket test is inclusive
    (`lower <= query <= upper`), so both segments are eligible there and the
    envelope maximum resolves the read.
    """
    # Two monotone segments meeting at the abscissa x = 1.0: segment 0 carries
    # value 1.0 there, segment 1 carries value 5.0 (the boundary-owning side).
    env_value, _, _ = envelope_at_query(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0]),
        policy=jnp.array([0.0, 1.0, 5.0, 6.0]),
        value=jnp.array([0.0, 1.0, 5.0, 6.0]),
        marginal=jnp.array([1.0, 1.0, 1.0, 1.0]),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0]),
        x_query=jnp.array([1.0]),
    )
    np.testing.assert_allclose(np.asarray(env_value), [5.0])


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


@pytest.mark.parametrize(
    ("dtype", "base", "gap"),
    [
        (jnp.float64, 1.0e4, 3.0e-11),
        (jnp.float32, 1.0e6, 1.0),
    ],
)
def test_large_magnitude_value_tie_is_precision_scaled(dtype, base, gap):
    """A value tie at large magnitude must still resolve right-continuously.

    Audit finding F6 (DC-1): the tie test used a fixed absolute half-width
    ``_VALUE_TIE_ATOL = 1e-12``. At magnitude ``base`` the representable ULP is
    ``eps(dtype) * base``, which for float32 at ``1e6`` (~0.06) — and even for
    float64 at ``1e4`` (~1.8e-12) — dwarfs ``1e-12``. Two branches that meet at
    a value tie then differ by more than the absolute band while being within a
    single ULP of each other, so the ending (steeper) segment A is picked as a
    strict winner and the right-continuous segment B is dropped: the published
    policy/marginal reverse. The dtype+magnitude-scaled band
    ``_TIE_BAND_ULPS * eps * max(|a|, |b|)`` recognizes the tie and B wins.

    ``gap`` sits in the window ``(1e-12, _TIE_BAND_ULPS * eps * base)``: above
    the old absolute band (so the old code saw A as a strict winner) and below
    the new scaled band (so the tie is honored). Segment A spans ``[0, 1]`` and
    ends at ``q=1`` with value ``base+gap``, policy 0, marginal 7; segment B
    spans ``[1, 2]`` and continues right of ``q=1`` with value ``base``, policy
    1, marginal 1. The right-continuous rule must publish B.
    """
    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        value=jnp.array([0.0, base + gap, base, base + 1.0], dtype=dtype),
        marginal=jnp.array([7.0, 7.0, 1.0, 1.0], dtype=dtype),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.array(1.0, dtype=dtype),
    )
    # Published value is the envelope max; policy/marginal are the
    # right-continuous winner B's.
    assert float(value) >= base
    assert np.isclose(float(policy), 1.0), "right-continuous segment B must win"
    assert np.isclose(float(marginal), 1.0), "B's marginal must be published"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_near_equal_slope_tie_picks_the_larger_slope_branch(dtype, block_size):
    """A value tie between two branches with near-equal slopes must resolve to the
    larger-slope branch in BOTH the dense and blocked paths.

    Audit finding F2 (round 4, second half): the tie-break folded the
    right-extends bit and the value-slope into one scalar
    (``arctan(slope)/pi + right_available``). For two genuinely-distinct but
    near-equal small slopes that fold rounds to the SAME value in float32, so
    ``argmax`` fell back to the lower index — the smaller-slope branch — reversing
    right-continuity. Comparing the slope directly at native precision picks the
    larger-slope branch.

    Both segments span ``[1, 2]`` and cross to ~0 at ``q=1.5``; segment A (policy
    20) carries the SMALLER slope, segment B (policy 10) the LARGER. Right-continuity
    publishes B.
    """
    s_small = np.asarray(3.06852285802961e-07, dtype=dtype)
    s_large = np.asarray(3.0687496632708644e-07, dtype=dtype)
    half = np.asarray(0.5, dtype=dtype)
    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([1.0, 2.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([20.0, 20.0, 10.0, 10.0], dtype=dtype),
        value=jnp.array(
            [-s_small * half, s_small * half, -s_large * half, s_large * half],
            dtype=dtype,
        ),
        marginal=jnp.array([200.0, 200.0, 100.0, 100.0], dtype=dtype),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.array([1.5], dtype=dtype),
        segment_block_size=block_size,
    )
    assert np.isclose(float(policy[0]), 10.0), "larger-slope branch B must win"
    assert np.isclose(float(marginal[0]), 100.0), "B's marginal must be published"

    # DC-3 counterpart: a genuine advantage on the SMALLER-slope branch A (far above
    # the operand-scaled rounding band) must NOT be swallowed by the tie band — A
    # then strictly dominates and wins in every path.
    adv = np.asarray(1e-4, dtype=dtype)
    _, policy_dc3, _ = envelope_at_query(
        endog_grid=jnp.array([1.0, 2.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([20.0, 20.0, 10.0, 10.0], dtype=dtype),
        value=jnp.array(
            [-s_small * half + adv, s_small * half + adv, -s_large * half, s_large * half],
            dtype=dtype,
        ),
        marginal=jnp.array([200.0, 200.0, 100.0, 100.0], dtype=dtype),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.array([1.5], dtype=dtype),
        segment_block_size=block_size,
    )
    assert np.isclose(float(policy_dc3[0]), 20.0), (
        "a genuine off-node advantage must not be swallowed by the tie band"
    )


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
