"""The query-side envelope backend matches the host oracle exactly.

`envelope_at_query` evaluates the branch-aware upper envelope directly at query
abscissae. It must agree with the exact host oracle on value and policy across
the cases that distinguish the topology contract: a clean crossing, a folded
branch, and a non-bridging branch the inference backends get wrong.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import (
    _candidate_terms,
    _exact_compare,
    _exact_slope_compare,
    envelope_at_query,
)
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

    The blocked scan reproduces the dense `(n_query, n_segment)` reduction —
    same envelope value, same exact-tie right-continuous winner — for any block
    size (divisor or not) below the segment count, up to floating-point
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
        (jnp.float64, 1.0e4, 1.0e-13),
        (jnp.float32, 1.0e6, 0.01),
    ],
)
def test_exact_stored_tie_at_a_node_is_right_continuous(dtype, base, gap):
    """Right-continuity applies to an EXACT stored tie — at any magnitude.

    This test descends from ``test_large_magnitude_value_tie_is_precision_
    scaled``, which asserted that a 16-ULP represented gap should be treated as
    a tie by a magnitude-scaled band. The round-5 audit identified that
    expectation as the defect itself: a stored node value carries ZERO rounding
    error, so genuinely distinct stored floats must never be declared tied. The
    tie half of the old test survives here with a ``gap`` BELOW half an ULP of
    ``base``, so ``base + gap`` rounds to exactly ``base``: the two branches
    carry bitwise-equal stored values at the shared node ``q=1`` and the
    right-continuous segment B (the one defined immediately to the right) wins.
    The strict-gap half lives in
    ``test_strict_represented_gap_selects_the_higher_branch``.
    """
    ending = float(jnp.asarray(base + gap, dtype=dtype))
    continuing = float(jnp.asarray(base, dtype=dtype))
    assert ending == continuing, "precondition: the stored node values tie exactly"

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        value=jnp.array([0.0, base + gap, base, base + 1.0], dtype=dtype),
        marginal=jnp.array([7.0, 7.0, 1.0, 1.0], dtype=dtype),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.array(1.0, dtype=dtype),
    )
    assert float(value) == continuing
    assert np.isclose(float(policy), 1.0), "right-continuous segment B must win"
    assert np.isclose(float(marginal), 1.0), "B's marginal must be published"


@pytest.mark.parametrize(
    ("dtype", "base", "gap"),
    [
        (jnp.float64, 1.0e4, 3.0e-11),
        (jnp.float32, 1.0e6, 1.0),
    ],
)
@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_strict_represented_gap_selects_the_higher_branch(dtype, base, gap, block_size):
    """A strict represented gap at a node is decisive — never a tie.

    The strict-gap half of the retired ``test_large_magnitude_value_tie_is_
    precision_scaled`` (same data: a ~16-ULP gap at large magnitude), with the
    expectation corrected per the round-5 audit. The stored node values are
    candidate DATA, compared exactly: segment A ends at ``q=1`` with value
    ``base+gap`` (policy 0, marginal 7), strictly above segment B's ``base``
    (policy 1), so A wins outright and value, policy, AND marginal are all
    published from A — right-continuity never enters, and the returned triple
    is coherent (no A-value/B-policy mix).
    """
    ending = float(jnp.asarray(base + gap, dtype=dtype))
    continuing = float(jnp.asarray(base, dtype=dtype))
    assert ending > continuing, "precondition: the gap survives storage"

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 1.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        value=jnp.array([0.0, base + gap, base, base + 1.0], dtype=dtype),
        marginal=jnp.array([7.0, 7.0, 1.0, 1.0], dtype=dtype),
        segment_id=jnp.array([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.array(1.0, dtype=dtype),
        segment_block_size=block_size,
    )
    assert float(value) == ending, "the winner's own stored value is published"
    assert np.isclose(float(policy), 0.0), "the strictly higher segment A must win"
    assert np.isclose(float(marginal), 7.0), "A's marginal must be published"


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_common_value_translation_does_not_change_a_strict_winner(block_size):
    """Adding a constant to every branch value cannot flip a strict winner.

    Round-5 audit regression (RT2): the retired magnitude-proportional tie band
    grew with ``|value|`` while a genuine represented gap does not, so a common
    translation flipped the selected branch. Segment A ends at the shared node
    ``q=1`` with a strict float32 gap of ``1e-5`` (~84 ULPs at 1.0) over the
    right-extending segment B; A must win at translation 0 and still win after
    translating every value by 1.0.
    """
    dtype = jnp.float32
    gap = 1.0e-5

    def selected_policy(translation: float) -> float:
        _, policy, _ = envelope_at_query(
            endog_grid=jnp.asarray([0.0, 1.0, 1.0, 2.0], dtype=dtype),
            policy=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            value=jnp.asarray(
                [translation, translation + gap, translation, translation + 1.0],
                dtype=dtype,
            ),
            marginal=jnp.asarray([gap, gap, 1.0, 1.0], dtype=dtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            x_query=jnp.asarray([1.0], dtype=dtype),
            segment_block_size=block_size,
        )
        return float(policy[0])

    assert selected_policy(0.0) == 0.0
    assert selected_policy(1.0) == 0.0


@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_node_event_selection_matches_the_exact_oracle_across_scales(order, block_size):
    """Compact round-5 mutation battery: node events resolve by exact comparison.

    Two branches share the node ``q=1``: the ending branch carries
    ``base + multiple*ULP(base)`` there (policy 0), the right-extending branch
    carries ``base`` (policy 1). For every dtype/scale/gap-multiple/branch-order/
    block-size combination the backend must agree with the exact host oracle at
    ``tol=0``: any strict represented gap (``multiple >= 1``) selects the higher
    ending branch; only the exact stored tie (``multiple == 0``) resolves
    right-continuously to the extending branch. The full 800-case battery lives
    in the round-5 audit artifacts; this compact grid pins the class.
    """
    configs = {np.float32: [1.0, 1.0e6], np.float64: [1.0, 1.0e12]}
    multiples = [0, 1, 16, 256]
    for dtype, bases in configs.items():
        for base in bases:
            ulp = float(np.spacing(np.asarray(base, dtype=dtype)))
            for multiple in multiples:
                gap = multiple * ulp
                if order == "AB":
                    grid = [0.0, 1.0, 1.0, 2.0]
                    value = [base, base + gap, base, base + 1.0]
                    policy = [0.0, 0.0, 1.0, 1.0]
                    segment = [0.0, 0.0, 1.0, 1.0]
                else:
                    grid = [1.0, 2.0, 0.0, 1.0]
                    value = [base, base + 1.0, base, base + gap]
                    policy = [1.0, 1.0, 0.0, 0.0]
                    segment = [1.0, 1.0, 0.0, 0.0]
                host = {
                    "endog_grid": np.asarray(grid, dtype=dtype),
                    "value": np.asarray(value, dtype=dtype),
                    "policy": np.asarray(policy, dtype=dtype),
                    "segment_id": np.asarray(segment, dtype=dtype),
                }
                got_value, got_policy, _ = envelope_at_query(
                    **{k: jnp.asarray(v) for k, v in host.items()},
                    marginal=jnp.zeros(4, dtype=dtype),
                    x_query=jnp.asarray([1.0], dtype=dtype),
                    segment_block_size=block_size,
                )
                oracle_value, oracle_policy, _ = exact_envelope(
                    **host, x_query=np.asarray([1.0], dtype=dtype), tol=0.0
                )
                context = f"{np.dtype(dtype)} base={base} multiple={multiple}"
                assert float(got_policy[0]) == float(oracle_policy[0]), context
                assert float(got_value[0]) == float(oracle_value[0]), context


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2])
def test_one_ulp_interior_gap_resolves_to_the_higher_branch(dtype, order, block_size):
    """The compensated interior evaluation certifies even a one-ULP gap.

    Branch H sits exactly one ULP above branch L at both stored endpoints of the
    shared span ``[0, 2]``, so the true piecewise-linear gap at any interior
    query is ~1 ULP — far below the retired magnitude-proportional tie band but
    strictly positive. The double-double interior evaluation carries each
    candidate to O(eps^2) relative accuracy, so H must win at an off-node query
    in both branch orders and both execution paths (the retired band declared a
    tie and let branch order pick the winner).
    """
    lo0 = np.asarray(1.0, dtype=dtype)
    lo1 = np.asarray(3.0, dtype=dtype)
    hi0 = np.asarray(np.nextafter(lo0, np.inf), dtype=dtype)
    hi1 = np.asarray(np.nextafter(lo1, np.inf), dtype=dtype)
    if order == "AB":
        value = [hi0, hi1, lo0, lo1]
        policy = [2.0, 2.0, 1.0, 1.0]
        segment = [0.0, 0.0, 1.0, 1.0]
    else:
        value = [lo0, lo1, hi0, hi1]
        policy = [1.0, 1.0, 2.0, 2.0]
        segment = [1.0, 1.0, 0.0, 0.0]
    _, got_policy, _ = envelope_at_query(
        endog_grid=jnp.asarray([0.0, 2.0, 0.0, 2.0], dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        marginal=jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=dtype),
        segment_id=jnp.asarray(segment, dtype=dtype),
        x_query=jnp.asarray([1.234567], dtype=dtype),
        segment_block_size=block_size,
    )
    assert float(got_policy[0]) == 2.0, "the one-ULP-higher branch must win"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_certified_radius_is_zero_at_nodes_and_second_order_interior(dtype):
    """The certified rounding radius tracks the arithmetic actually performed.

    At a node event the candidate value is stored data — the radius is exactly
    zero, so no tolerance can ever separate exactly-equal stored floats or
    merge distinct ones. At an interior query the compensated evaluation's
    residual radius is O(eps^2) of the operand scale — orders of magnitude
    below one ULP of the data, so it can never swallow a represented gap.
    """
    # One segment from (0, 1) to (2, 3): columns are (lx, rx, lv, rv, lp, rp,
    # lm, rm); queries hit the left node, the right node, and an interior point.
    block = jnp.asarray([[0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 0.5, 0.5]], dtype=dtype)
    terms = _candidate_terms(
        block=block,
        live=jnp.asarray([True]),
        flat=jnp.asarray([0.0, 2.0, 1.234567], dtype=dtype),
    )
    radius = np.asarray(terms.radius)[:, 0]
    value_hi = np.asarray(terms.value_hi)[:, 0]
    value_lo = np.asarray(terms.value_lo)[:, 0]
    assert radius[0] == 0.0, "left node event carries zero radius"
    assert radius[1] == 0.0, "right node event carries zero radius"
    assert value_hi[0] == 1.0, "left node value is stored data"
    assert value_hi[1] == 3.0, "right node value is stored data"
    assert value_lo[0] == 0.0
    assert value_lo[1] == 0.0
    eps = float(jnp.finfo(dtype).eps)
    assert 0.0 < radius[2] <= 64.0 * eps * eps * abs(value_hi[2]), (
        "interior radius is strictly positive but second-order small"
    )


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
    _value, policy, marginal = envelope_at_query(
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

    # DC-3 counterpart: a genuine advantage on the SMALLER-slope branch A breaks
    # the exact tie — A then strictly dominates and wins in every path.
    adv = np.asarray(1e-4, dtype=dtype)
    _, policy_dc3, _ = envelope_at_query(
        endog_grid=jnp.array([1.0, 2.0, 1.0, 2.0], dtype=dtype),
        policy=jnp.array([20.0, 20.0, 10.0, 10.0], dtype=dtype),
        value=jnp.array(
            [
                -s_small * half + adv,
                s_small * half + adv,
                -s_large * half,
                s_large * half,
            ],
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


def test_lone_candidate_wins_at_its_own_abscissa():
    """A branch holding a single candidate is visible where a query lands on it.

    Segment 1 carries one point at `R=1` worth `5.0`; segment 0 spans `[0, 2]` and
    is worth only `1.0` there. With no consecutive same-segment neighbour the lone
    point brackets nothing but itself, and a read at its own abscissa must publish
    its value and policy rather than collapsing onto the lower two-point branch.
    """
    value, policy, _marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 2.0, 1.0]),
        policy=jnp.array([1.0, 1.0, 9.0]),
        value=jnp.array([0.0, 2.0, 5.0]),
        marginal=jnp.array([1.0, 1.0, 1.0]),
        segment_id=jnp.array([0.0, 0.0, 1.0]),
        x_query=jnp.array(1.0),
    )
    assert np.isclose(float(value), 5.0)
    assert np.isclose(float(policy), 9.0)


def test_lone_candidate_does_not_displace_a_right_extending_chain():
    """Where a multi-point branch continues right of the query, it still wins a tie.

    Segment 0 spans `[0, 2]` and segment 1 holds one point at `R=1` with exactly
    segment 0's value there. The self-bracket is zero-width, so the right-continuous
    tie-break keeps segment 0's policy.
    """
    value, policy, _marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 2.0, 1.0]),
        policy=jnp.array([1.0, 1.0, 9.0]),
        value=jnp.array([0.0, 2.0, 1.0]),
        marginal=jnp.array([1.0, 1.0, 1.0]),
        segment_id=jnp.array([0.0, 0.0, 1.0]),
        x_query=jnp.array(1.0),
    )
    assert np.isclose(float(value), 1.0)
    assert np.isclose(float(policy), 1.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
@pytest.mark.parametrize("denominator", [3, 5, 7, 9, 11, 13, 17, 31, 63, 127])
def test_exact_interior_tie_takes_the_right_continuous_branch(
    dtype, order, block_size, denominator
):
    """A TRUE interior tie must be recognised as a tie, not as a strict win.

    Branch A runs `(0, 0) -> (d, 1)` and branch B runs `(-1, 1) -> (2d-1, 2-d)`.
    At the interior query `q = 1` both attain exactly `1/d` — the values are
    equal as rationals, not merely close — while A's right-hand slope `+1/d`
    exceeds B's `(1-d)/2d`. The documented rule therefore requires A's policy
    and marginal.

    `1/d` is not representable, so each branch's double-double pair carries a
    low word that depends on how ITS segment is parameterized: the two low
    words differ even though the exact values coincide. A selector that orders
    candidates lexicographically on `(hi, lo)` reads that difference as strict
    order, hands the query to B, and never runs the right-continuous rule —
    round-6 audit F2, which found 672 wrong policy/marginal choices in 1,680
    such ties. Only an exact comparison can classify this correctly, so this
    test is a direct check that one is being made.
    """
    end = np.asarray(denominator, dtype=dtype).item()
    far = np.asarray(2 * denominator - 1, dtype=dtype).item()
    tail = np.asarray(2 - denominator, dtype=dtype).item()
    branch_a = ([0.0, end], [0.0, 1.0], 0.0, 10.0)
    branch_b = ([-1.0, far], [1.0, tail], 1.0, 20.0)
    first, second = (branch_a, branch_b) if order == "AB" else (branch_b, branch_a)
    labels = [0.0, 0.0, 1.0, 1.0]
    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(first[0] + second[0], dtype=dtype),
        policy=jnp.asarray([first[2]] * 2 + [second[2]] * 2, dtype=dtype),
        value=jnp.asarray(first[1] + second[1], dtype=dtype),
        marginal=jnp.asarray([first[3]] * 2 + [second[3]] * 2, dtype=dtype),
        segment_id=jnp.asarray(labels, dtype=dtype),
        x_query=jnp.asarray([1.0], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size} d={denominator}"
    assert float(got_policy[0]) == 0.0, context
    assert float(got_marginal[0]) == 10.0, context
    # The tie means the published level is the shared value either way; it is
    # the branch attribution that the exact comparison fixes.
    assert float(got_value[0]) == pytest.approx(1.0 / denominator, rel=1e-6), context


def _slope_collision_pairs(dtype, *, seed, target=1.0 / 3.0, draws=40_000, want=3):
    """Segment pairs whose exact slopes differ but whose `fl(rise/run)` keys agree.

    Each candidate segment runs from the shared node `(1, 0)` to `(x1, rise)`. The
    exact slope is the rational `rise / (x1 - 1)` read off the STORED floats; the
    native key is the working-dtype division the selector used to compute. Drawing
    the run over a wide exponent range makes distinct rationals collapse onto one
    key routinely — the returned pairs are `(lower_exact, higher_exact)`.
    """
    rng = np.random.default_rng(seed)
    span = 12 if dtype is np.float32 else 24
    runs = np.asarray(
        rng.uniform(0.5, 2.0, size=draws) * 2.0 ** rng.integers(-span, span + 1, draws),
        dtype=dtype,
    )
    x1 = np.asarray(dtype(1.0) + runs, dtype=dtype)
    run = np.asarray(x1 - dtype(1.0), dtype=dtype)
    rise = np.asarray(dtype(target) * run, dtype=dtype)
    keep = np.isfinite(x1) & np.isfinite(rise) & (run > 0)
    x1, run, rise = x1[keep], run[keep], rise[keep]

    seen, pairs = {}, []
    keys = np.asarray(rise / run, dtype=dtype)
    for node, height, key in zip(x1, rise, keys, strict=True):
        exact = Fraction.from_float(float(height)) / (
            Fraction.from_float(float(node)) - Fraction(1)
        )
        previous = seen.setdefault(float(key), (exact, float(node), float(height)))
        if previous[0] == exact:
            continue
        pairs.append(
            tuple(
                sorted(
                    (previous, (exact, float(node), float(height))), key=lambda p: p[0]
                )
            )
        )
        if len(pairs) == want:
            break
    if len(pairs) < want:  # the generator, not the selector, has failed
        raise AssertionError(f"only {len(pairs)} slope collisions for {dtype}")
    return pairs


_SLOPE_COLLISIONS = {
    np.float32: _slope_collision_pairs(np.float32, seed=20260728),
    np.float64: _slope_collision_pairs(np.float64, seed=20260764),
}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", [0, 1, 2])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_exact_value_tie_orders_by_exact_slope_not_the_rounded_key(
    dtype, case, order, block_size
):
    """An exact VALUE tie must be broken by the exact slope, not a rounded key.

    Round-6 made the value comparison exact; round-7 audit F2 found the class
    reopened one operation later. Both branches leave the stored node `(1, 0)`,
    so the values are exactly tied and right-continuity decides — and the rule is
    "larger value-slope, then earliest candidate". The selector computed that
    slope as `fl((v1 - v0) / (x1 - x0))`, and two strictly ordered exact slopes
    can share one such float. `argmax` then fell through to candidate order, so a
    pure branch permutation flipped the published policy and marginal. The level
    is exactly zero either way; it is the attribution that is wrong, and the
    marginal it publishes feeds the parent Euler inversion.

    The lower-exact-slope branch carries policy 0, the higher policy 1. Only the
    higher may win, in every dtype, order and block layout.
    """
    (_, low_node, low_rise), (_, high_node, high_rise) = _SLOPE_COLLISIONS[dtype][case]
    one = dtype(1.0)
    # A probe that proves a negative must show it CAN fail: assert the keys really
    # do collide here, or the test would pass by never posing the question.
    low_key = dtype(dtype(low_rise) / dtype(dtype(low_node) - one))
    high_key = dtype(dtype(high_rise) / dtype(dtype(high_node) - one))
    assert low_key == high_key, "the two branches must share one rounded slope key"

    lower = ([1.0, low_node], [0.0, low_rise], 0.0, 10.0)
    higher = ([1.0, high_node], [0.0, high_rise], 1.0, 20.0)
    first, second = (lower, higher) if order == "AB" else (higher, lower)
    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(first[0] + second[0], dtype=dtype),
        policy=jnp.asarray([first[2]] * 2 + [second[2]] * 2, dtype=dtype),
        value=jnp.asarray(first[1] + second[1], dtype=dtype),
        marginal=jnp.asarray([first[3]] * 2 + [second[3]] * 2, dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray([1.0], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size} case={case}"
    assert float(got_value[0]) == 0.0, context
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 20.0, context


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_exactly_equal_slopes_fall_back_to_the_earliest_candidate(
    dtype, order, block_size
):
    """Candidate order decides only when the exact slopes are genuinely equal.

    The counterpart to the collision test above: branches `(1, 0) -> (4, 1)` and
    `(1, 0) -> (7, 2)` have the SAME exact slope `1/3` and the same exact value
    at the shared node, so no comparison can separate them and the documented
    fallback — earliest candidate — applies. This pins the other side of the
    predicate: an exact selector must not manufacture an order here either, so
    whichever branch is listed first wins.
    """
    lead = ([1.0, 4.0], [0.0, 1.0], 0.0, 10.0)
    trail = ([1.0, 7.0], [0.0, 2.0], 1.0, 20.0)
    first, second = (lead, trail) if order == "AB" else (trail, lead)
    _value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(first[0] + second[0], dtype=dtype),
        policy=jnp.asarray([first[2]] * 2 + [second[2]] * 2, dtype=dtype),
        value=jnp.asarray(first[1] + second[1], dtype=dtype),
        marginal=jnp.asarray([first[3]] * 2 + [second[3]] * 2, dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray([1.0], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size}"
    assert float(got_policy[0]) == first[2], context
    assert float(got_marginal[0]) == first[3], context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", [0, 1, 2])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_interior_exact_tie_also_orders_by_exact_slope(dtype, case, order, block_size):
    """The colliding-key class at a STRICTLY INTERIOR query, not just at a node.

    The audit witness put the tie on a stored node, where the value comparison
    short-circuits to the stored floats. Interior queries reach the tie through
    the interpolated double-double path instead, so the two routes deserve
    separate evidence even though they share one slope comparator.

    Each segment is `(-run, -rise) -> (run, rise)`, which passes exactly through
    the origin: both endpoints are the stored floats of the collision pair
    negated, so nothing rounds. The query `q = 0` is interior to both segments
    and both values there are exactly zero — a true interior tie whose slopes are
    strictly ordered as rationals yet share one `fl(rise/run)` key. Only the
    larger exact slope may win.
    """
    (_, low_node, low_rise), (_, high_node, high_rise) = _SLOPE_COLLISIONS[dtype][case]
    one = dtype(1.0)
    low_run, high_run = dtype(low_node) - one, dtype(high_node) - one
    low_rise, high_rise = dtype(low_rise), dtype(high_rise)

    def native_key(rise, run):
        """Exactly what the selector computes: `fl((v1 - v0) / (x1 - x0))`."""
        return dtype(dtype(rise - -rise) / dtype(run - -run))

    assert native_key(low_rise, low_run) == native_key(high_rise, high_run), (
        "the two branches must share one rounded slope key"
    )

    lower = ([-low_run, low_run], [-low_rise, low_rise], 0.0, 10.0)
    higher = ([-high_run, high_run], [-high_rise, high_rise], 1.0, 20.0)
    first, second = (lower, higher) if order == "AB" else (higher, lower)
    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(first[0] + second[0], dtype=dtype),
        policy=jnp.asarray([first[2]] * 2 + [second[2]] * 2, dtype=dtype),
        value=jnp.asarray(first[1] + second[1], dtype=dtype),
        marginal=jnp.asarray([first[3]] * 2 + [second[3]] * 2, dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray([0.0], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size} case={case}"
    assert float(got_value[0]) == 0.0, context
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 20.0, context


# Operand magnitude above which Dekker's TwoProd splitting itself overflows: the
# split multiplies by `2**s + 1`, so it needs `|a| < 2**(emax - s)`. These are the
# thresholds the round-8 selector silently crossed, and a scale test that stays
# below them cannot detect the defect at all.
_SPLIT_OVERFLOW_EXPONENT = {np.float32: 127 - 12, np.float64: 1023 - 27}


def _rescaling_exponents(entries, dtype):
    """Power-of-two exponents keeping every entry finite normal, spanning the range.

    Derived from the case's own magnitudes rather than hardcoded, so the ladder
    stretches to the true representable limits for whatever collision pair the
    generator produced instead of to a constant someone picked once.
    """
    info = np.finfo(dtype)
    magnitudes = [abs(float(x)) for x in entries if float(x) != 0.0]
    top = int(np.frexp(max(magnitudes))[1])
    bottom = int(np.frexp(min(magnitudes))[1])
    highest = int(info.maxexp) - 1 - top
    lowest = int(info.minexp) + 1 - bottom
    span = highest - lowest
    return sorted({lowest, lowest + span // 4, 0, highest - span // 4, highest})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 3])
def test_selection_survives_a_power_of_two_rescaling_of_the_whole_model(
    dtype, order, block_size
):
    """Choosing a unit must not choose a policy (round-8 audit F2).

    Multiplying every grid and value coordinate by one positive power of two is
    a change of units: exact in binary floating point, and it leaves the value
    ordering and every slope ordering algebraically untouched, so the published
    policy and marginal must be identical. Round 8 failed this three ways at
    once -- the cross products, the interpolant's products, and Dekker's own
    splitting each break at their own scale -- and each turned a strict exact
    ordering into a branch-order coin flip.

    The ladder is derived per case from the representable range, so it reaches
    the real limits, and the test asserts IN-LINE that its top rung crosses the
    splitting threshold. Without that check a green result would only say the
    scales happened to be comfortable.
    """
    (_, low_node, low_rise), (_, high_node, high_rise) = _SLOPE_COLLISIONS[dtype][0]
    one = dtype(1.0)
    low_run, high_run = dtype(low_node) - one, dtype(high_node) - one
    low_rise, high_rise = dtype(low_rise), dtype(high_rise)
    base = [low_run, high_run, low_rise, high_rise]

    exponents = _rescaling_exponents(base, dtype)
    reached = max(int(np.frexp(abs(float(x)))[1]) for x in base) + max(exponents)
    assert reached > _SPLIT_OVERFLOW_EXPONENT[dtype], (
        f"the ladder tops out at 2**{reached}, below the "
        f"2**{_SPLIT_OVERFLOW_EXPONENT[dtype]} splitting edge, so it could not "
        "have caught the round-8 defect"
    )

    tiny = np.finfo(dtype).tiny
    for exponent in exponents:
        scale = dtype(np.ldexp(1.0, exponent))
        lr, hr = dtype(low_run * scale), dtype(high_run * scale)
        lv, hv = dtype(low_rise * scale), dtype(high_rise * scale)
        assert all(np.isfinite(v) and abs(v) >= tiny for v in (lr, hr, lv, hv)), (
            f"rescaled input left the finite-normal range at 2**{exponent}"
        )
        # Rescaling is exact, so the exact slope order is the SAME order.
        assert (Fraction(float(hv)) / Fraction(float(hr))) > (
            Fraction(float(lv)) / Fraction(float(lr))
        )

        lower = ([-lr, lr], [-lv, lv], 0.0, 10.0)
        higher = ([-hr, hr], [-hv, hv], 1.0, 20.0)
        first, second = (lower, higher) if order == "AB" else (higher, lower)
        got_value, got_policy, got_marginal = envelope_at_query(
            endog_grid=jnp.asarray(first[0] + second[0], dtype=dtype),
            policy=jnp.asarray([first[2]] * 2 + [second[2]] * 2, dtype=dtype),
            value=jnp.asarray(first[1] + second[1], dtype=dtype),
            marginal=jnp.asarray([first[3]] * 2 + [second[3]] * 2, dtype=dtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            x_query=jnp.asarray([0.0], dtype=dtype),
            segment_block_size=block_size,
        )
        context = f"{np.dtype(dtype)} order={order} block={block_size} 2**{exponent}"
        assert float(got_value[0]) == 0.0, context
        assert float(got_policy[0]) == 1.0, context
        assert float(got_marginal[0]) == 20.0, context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_exact_predicates_match_a_rational_oracle_at_every_scale(dtype):
    """The VALUE predicate is rescaled too, and needs its own evidence.

    The audit's artifacts exercise the slope comparator only. The repair changed
    the value comparator as well, and a regression covering only the half that
    was reported is how this class has already come back twice. This drives both
    predicates directly against `Fraction` across the same ladder.
    """
    (_, low_node, low_rise), (_, high_node, high_rise) = _SLOPE_COLLISIONS[dtype][0]
    one = dtype(1.0)
    low_run, high_run = dtype(low_node) - one, dtype(high_node) - one
    low_rise, high_rise = dtype(low_rise), dtype(high_rise)
    base = [low_run, high_run, low_rise, high_rise]

    checked = 0
    for exponent in _rescaling_exponents(base, dtype):
        scale = dtype(np.ldexp(1.0, exponent))
        lr, hr = dtype(low_run * scale), dtype(high_run * scale)
        lv, hv = dtype(low_rise * scale), dtype(high_rise * scale)
        cols_low = jnp.asarray([[-lr, lr, -lv, lv]], dtype=dtype)
        cols_high = jnp.asarray([[-hr, hr, -hv, hv]], dtype=dtype)
        q = jnp.asarray([0.0], dtype=dtype)

        # Both segments pass exactly through the origin: a TRUE value tie.
        value_sign = float(
            np.asarray(_exact_compare(cols_a=cols_low, cols_b=cols_high, q=q))[0]
        )
        assert value_sign == 0.0, f"2**{exponent}: exact value tie not detected"

        slope_sign = float(
            np.asarray(_exact_slope_compare(cols_a=cols_low, cols_b=cols_high))[0]
        )
        expected = Fraction(float(lv)) / Fraction(float(lr)) - Fraction(
            float(hv)
        ) / Fraction(float(hr))
        assert slope_sign == float(np.sign(float(expected))), (
            f"2**{exponent}: slope sign {slope_sign} disagrees with the oracle"
        )
        checked += 1
    assert checked >= 5


def _far_segment_exponents(dtype):
    """Exponents for a remote segment that stay finite normal, spanning the range.

    Includes the top of the range deliberately: that is where a whole-array
    normalization does its damage, and a ladder that stops short of it cannot
    detect the defect at all.
    """
    info = np.finfo(dtype)
    return [int(info.minexp) + 4, -60, 0, 23, int(info.maxexp) - 2]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 3])
@pytest.mark.parametrize("far_first", [False, True])
def test_a_non_bracketing_segment_cannot_change_the_local_envelope(
    dtype, order, block_size, far_first
):
    """A candidate that cannot bracket the query is mathematically irrelevant.

    Round 9 normalized the whole problem by ONE grid exponent and ONE value
    exponent chosen from the largest candidate anywhere in the input. That map
    is not injective: a remote segment which does not bracket the query still
    selected the scale, and two adjacent local floats collapsed into the same
    subnormal BEFORE the certified comparator saw them. The comparator then
    correctly reported an exact tie between values that are distinct as stored,
    and the published value, policy and marginal all moved in response to a
    segment that cannot affect the answer (round-9 audit F2).

    Exact arithmetic cannot rebuild bits discarded upstream of it, so this is
    pinned at the public boundary: inserting the far segment must change
    nothing at all.
    """
    info = np.finfo(dtype)
    one = dtype(1)
    adjacent = np.nextafter(one, dtype(np.inf), dtype=dtype)
    # The strict local ordering the selector must preserve, as exact rationals.
    assert Fraction(float(adjacent)) > Fraction(float(one))

    query = 0.5
    far_bounds = [10.0, 11.0]
    lower = ([0.0, 1.0], [one, one], 1.0, 10.0)
    higher = ([0.0, 1.0], [adjacent, adjacent], 0.0, 20.0)
    local = [lower, higher] if order == "AB" else [higher, lower]

    for exponent in _far_segment_exponents(dtype):
        far_value = dtype(-np.ldexp(1.0, exponent))
        # The far segment must itself be finite normal, or the witness is about
        # non-representable input rather than about the normalization.
        assert np.isfinite(far_value)
        assert abs(far_value) >= info.tiny
        far = (far_bounds, [far_value, far_value], 9.0, 9.0)
        # It cannot bracket the query, so it cannot affect the answer there.
        assert not far_bounds[0] <= query <= far_bounds[1]

        rows = [far, *local] if far_first else [*local, far]
        grid, value, policy, marginal, segment_id = [], [], [], [], []
        for sid, (g, v, p, m) in enumerate(rows):
            grid += g
            value += list(v)
            policy += [p] * 2
            marginal += [m] * 2
            segment_id += [float(sid)] * 2

        got_value, got_policy, got_marginal = envelope_at_query(
            endog_grid=jnp.asarray(grid, dtype=dtype),
            policy=jnp.asarray(policy, dtype=dtype),
            value=jnp.asarray(value, dtype=dtype),
            marginal=jnp.asarray(marginal, dtype=dtype),
            segment_id=jnp.asarray(segment_id, dtype=dtype),
            x_query=jnp.asarray([query], dtype=dtype),
            segment_block_size=block_size,
        )
        context = (
            f"{np.dtype(dtype)} order={order} block={block_size} "
            f"far_first={far_first} far=2**{exponent}"
        )
        assert float(got_value[0]) == float(adjacent), context
        assert float(got_policy[0]) == 0.0, context
        assert float(got_marginal[0]) == 20.0, context


def _slope_overflow_cases(dtype):
    """(value_exponent, width_exponent) pairs whose rounded slope is not finite.

    Derived from the representable range rather than pinned: the screen key is
    `Δvalue / Δgrid`, so the condition is `value_exp - width_exp > maxexp`. Each
    pair is checked in-line below, so a case that stops overflowing fails loud
    instead of quietly testing nothing.
    """
    maxexp = int(np.finfo(dtype).maxexp)
    minexp = int(np.finfo(dtype).minexp)
    cases = []
    for width_exp in (minexp + 64, minexp + 40, minexp + 8, -6):
        # Overflow needs `value_exp - width_exp > maxexp`; +6 clears it with
        # margin, and the cap keeps the value itself finite normal.
        value_exp = min(maxexp - 2, maxexp + width_exp + 6)
        if value_exp - width_exp > maxexp:
            cases.append((value_exp, width_exp))
    # A generator that silently produced nothing would make the test vacuous.
    assert cases
    return cases


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("block_size", [0, 3])
def test_an_overflowing_slope_screen_defers_to_the_exact_comparison(dtype, block_size):
    """The rounded slope screen must never decide by falling silent.

    `slope` is computed in ORIGINAL units on purpose -- it is a key compared
    ACROSS candidates, so every candidate has to express it in the same units.
    That makes it overflow to an infinity when a large value gap meets a small
    grid width, while every stored input is still finite normal. The screen
    band `reach = 8*eps*(|slope| + |lead_slope|)` is then infinite,
    `lead_slope - reach` is NaN, and every `>=` against it is False -- so the
    contender set empties, `_exact_slope_compare` never runs, and the winner is
    whatever `argmax` over the rounded key returned: candidate order.

    That is the round-7 F2 signature exactly -- the value is tied either way and
    only the published policy and marginal are wrong -- so it is pinned the same
    way: the answer must not depend on the order the branches are listed in.
    """
    for value_exp, width_exp in _slope_overflow_cases(dtype):
        width = dtype(np.ldexp(1.0, width_exp))
        big = dtype(np.ldexp(1.0, value_exp))
        taller = np.nextafter(big, dtype(np.inf))
        # Both candidates are finite normal as STORED ...
        assert np.isfinite(big)
        assert np.isfinite(taller)
        assert abs(width) >= np.finfo(dtype).tiny
        # ... while the screen key they produce is not. The overflow is the
        # point of the case, so it is not a warning worth emitting.
        with np.errstate(over="ignore"):
            assert not np.isfinite(big / width)
        # and the two slopes are strictly ordered as exact rationals.
        assert Fraction(float(taller)) / Fraction(float(width)) > Fraction(
            float(big)
        ) / Fraction(float(width))

        a = ([0.0, float(width)], [dtype(0.0), big], 1.0, 10.0)
        b = ([0.0, float(width)], [dtype(0.0), taller], 2.0, 20.0)

        seen = {}
        for order in ("AB", "BA"):
            rows = [a, b] if order == "AB" else [b, a]
            grid, value, policy, marginal, segment_id = [], [], [], [], []
            for sid, (g, v, p, m) in enumerate(rows):
                grid += g
                value += list(v)
                policy += [p] * 2
                marginal += [m] * 2
                segment_id += [float(sid)] * 2
            got = envelope_at_query(
                endog_grid=jnp.asarray(grid, dtype=dtype),
                policy=jnp.asarray(policy, dtype=dtype),
                value=jnp.asarray(value, dtype=dtype),
                marginal=jnp.asarray(marginal, dtype=dtype),
                segment_id=jnp.asarray(segment_id, dtype=dtype),
                x_query=jnp.asarray([0.0], dtype=dtype),
                segment_block_size=block_size,
            )
            seen[order] = tuple(float(x[0]) for x in got)

        context = (
            f"{np.dtype(dtype)} block={block_size} "
            f"value=2**{value_exp} width=2**{width_exp}"
        )
        assert seen["AB"] == seen["BA"], context
        # The strictly steeper candidate wins, in both orders.
        assert seen["AB"][1] == 2.0, context
        assert seen["AB"][2] == 20.0, context
