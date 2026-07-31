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
    _dekker_split_factor,
    _dyadic_product,
    _exact_compare,
    _exact_difference,
    _exact_ratio,
    _exact_slope_compare,
    _exactly_maximal,
    _framed_difference,
    _right_continuous_winner,
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


def _smallest_scale_case(dtype):
    """A bracketing segment whose own WIDTH sits at the bottom of the range.

    `x0 = tiny`, `q = 1.5*tiny`, `x1 = 2*tiny` -- three distinct finite NORMAL
    floats whose differences are SUBNORMAL. That is the point of the case: XLA
    flushes subnormals to zero, so `q - x0` computed on the raw operands comes
    back exactly `0.0` and the interpolant collapses onto its left value.
    """
    tiny = np.finfo(dtype).tiny
    x0, q, x1 = dtype(tiny), dtype(1.5 * tiny), dtype(2.0 * tiny)
    assert x0 < q < x1, "the query must be strictly interior"
    for name, value in (("x0", x0), ("q", q), ("x1", x1)):
        assert abs(value) >= tiny, f"{name} must be finite NORMAL"
    # The reachability condition, asserted so the case cannot quietly stop
    # exercising the mechanism it exists for.
    assert 0 < abs(float(q) - float(x0)) < float(tiny), "the difference is subnormal"
    return x0, q, x1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_a_segment_at_the_smallest_scale_keeps_its_interpolation_fraction(
    dtype, order, block_size
):
    """A subnormal `q - x0` must not be flushed away before it is used.

    The mirror image of the wide-segment case, and a defect this repair
    introduced: "the difference of two finite floats cannot overflow" is true,
    but it can UNDERFLOW. At the bottom of the range the difference of two
    distinct normals is subnormal, XLA flushes it to zero, and the interpolant
    collapses onto its left value exactly as it did when the operands were
    pre-scaled DOWN (round-9 audit MT6, 48/144 mismatches).

    The two cases together pin the asymmetry the repair rests on: operands may be
    lifted UP before differencing, because a power of two is exact and injective,
    but never lowered, because lowering can merge two distinct floats.
    """
    x0, q, x1 = _smallest_scale_case(dtype)
    # Values, policies and marginals chosen so the exact answer at the midpoint
    # is 1.0 on all three channels, and a far competitor is strictly lower.
    local = ([float(x0), float(x1)], [dtype(0.0), dtype(2.0)], (0.0, 2.0), (1.0, 1.0))
    far = ([1.0, 1.5], [dtype(-100.0), dtype(-100.0)], (9.0, 9.0), (9.0, 9.0))
    rows = [local, far] if order == "AB" else [far, local]

    grid, value, policy, marginal, segment_id = [], [], [], [], []
    for sid, (g, v, p, m) in enumerate(rows):
        grid += g
        value += list(v)
        policy += list(p)
        marginal += list(m)
        segment_id += [float(sid)] * 2

    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(grid, dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        marginal=jnp.asarray(marginal, dtype=dtype),
        segment_id=jnp.asarray(segment_id, dtype=dtype),
        x_query=jnp.asarray([float(q)], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size}"
    assert float(got_value[0]) == 1.0, context
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 1.0, context


def _smallest_value_scale_case(dtype):
    """Two branches whose VALUES sit at the bottom of the range, on a unit grid.

    The grid axis is deliberately ordinary -- `[0, 1]` -- so nothing here can be
    blamed on grid arithmetic. Branch A rises by exactly one ULP across the
    segment, from `tiny` to `nextafter(tiny)`; branch B is constant at `tiny`.
    Every stored number is a finite NORMAL float.

    The mechanism: one ULP at `tiny` is the SMALLEST SUBNORMAL, so an interior
    fraction of it is below everything representable, and `r*d` flushes to zero
    unless the value operands are lifted first. A publishes `tiny`, ties B, and
    the strictly lower branch can take the policy and the marginal.

    Returns `(left_value, right_value)` for A; B is constant at `left_value`.
    """
    tiny = np.finfo(dtype).tiny
    lower = dtype(tiny)
    upper = np.nextafter(lower, dtype(np.inf), dtype=dtype)
    assert lower < upper, "the two endpoint values must be distinct floats"
    for name, value in (("lower", lower), ("upper", upper)):
        assert abs(value) >= tiny, f"{name} must be finite NORMAL"
    # The reachability condition. Asserting it here is what stops the case from
    # quietly degrading into an ordinary interpolation if `tiny` ever changes
    # meaning: the whole point is that the gap is NOT normal.
    gap = float(upper) - float(lower)
    assert 0 < gap < float(tiny), "the one-ULP value gap must be subnormal"
    return lower, upper


def _rounded_between(exact, lower, upper, dtype):
    """Round `exact` (a `Fraction` in `[lower, upper]`) to nearest, ties to even.

    `lower` and `upper` are consecutive floats, so they are the only candidates
    and the rule reduces to comparing the two exact distances.
    """
    below = exact - Fraction(float(lower))
    above = Fraction(float(upper)) - exact
    assert below >= 0, "the exact value must not sit below the lower neighbour"
    assert above >= 0, "the exact value must not sit above the upper neighbour"
    if below != above:
        return dtype(lower) if below < above else dtype(upper)
    # Halfway: ties-to-even on the significand. `lower` is `tiny`, whose
    # significand is zero, hence even.
    return dtype(lower)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.625, 0.75])
def test_a_one_ulp_value_gap_at_the_smallest_scale_reaches_the_published_value(
    dtype, order, block_size, fraction
):
    """The value axis needs the lift too, and for the OPPOSITE reason to the grid.

    The grid lift exists because differencing can lose a difference; the value
    lift exists because the increment `r*d` can be smaller than anything the
    format represents at that scale. Rounds 8 to 11 all reasoned that the value
    axis needed no scale because its intermediates stay BOUNDED, which is true
    and beside the point (round-11 audit F2, MT8: 234 of 234 generated cells).

    `fraction` sweeps both rounding directions on purpose. Above one half the
    correctly rounded value is A's upper endpoint, so the defect showed up in
    the LEVEL as well as the policy; at or below one half the correctly rounded
    value is A's lower endpoint, which B also publishes, so the level agrees
    either way and only the exact ordering separates them -- that cell tests the
    structural-exactness rule rather than the lift.
    """
    lower, upper = _smallest_value_scale_case(dtype)
    q = dtype(fraction)
    exact_a = Fraction(float(lower)) + Fraction(fraction) * (
        Fraction(float(upper)) - Fraction(float(lower))
    )
    assert exact_a > Fraction(float(lower)), "A must be STRICTLY above B at the query"
    expected_value = _rounded_between(exact_a, lower, upper, dtype)

    rising = ([0.0, 1.0], [lower, upper], (1.0, 1.0), (20.0, 20.0))
    flat = ([0.0, 1.0], [lower, lower], (0.0, 0.0), (10.0, 10.0))
    rows = [rising, flat] if order == "AB" else [flat, rising]

    grid, value, policy, marginal, segment_id = [], [], [], [], []
    for sid, (g, v, p, m) in enumerate(rows):
        grid += g
        value += list(v)
        policy += list(p)
        marginal += list(m)
        segment_id += [float(sid)] * 2

    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(grid, dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        marginal=jnp.asarray(marginal, dtype=dtype),
        segment_id=jnp.asarray(segment_id, dtype=dtype),
        x_query=jnp.asarray([float(q)], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size} r={fraction}"
    assert dtype(got_value[0]) == expected_value, context
    # The strictly higher branch owns the query, so its policy and marginal are
    # what must be published -- the channel the defect actually corrupted.
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 20.0, context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("block_size", [0, 2])
def test_a_power_of_two_value_rescaling_cannot_change_the_winner(dtype, block_size):
    """Scaling every value by an exact power of two is an ordering isomorphism.

    Multiplying all four stored values by `2**k` leaves the exact comparison
    between the branches unchanged, so the published policy and marginal must be
    unchanged and the published value must scale by exactly the same factor.
    That is a property of the model, not of the arithmetic, which is what makes
    it the right acceptance statement for this class: no scale is privileged.

    The unscaled leg runs at `1`, the rescaled leg at `tiny`. Both legs are
    asserted, so the test fails whether the defect is at the bottom of the range
    or -- were a future repair to trade one end for the other, which is how this
    class has moved twice -- at the top.
    """
    tiny = np.finfo(dtype).tiny
    results = {}
    for name, base in (("unit", dtype(1.0)), ("tiny", dtype(tiny))):
        lower = base
        upper = np.nextafter(lower, dtype(np.inf), dtype=dtype)
        got = envelope_at_query(
            endog_grid=jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=dtype),
            policy=jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=dtype),
            value=jnp.asarray([lower, upper, lower, lower], dtype=dtype),
            marginal=jnp.asarray([20.0, 20.0, 10.0, 10.0], dtype=dtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            x_query=jnp.asarray([0.75], dtype=dtype),
            segment_block_size=block_size,
        )
        results[name] = tuple(float(channel[0]) for channel in got)

    unit_value, unit_policy, unit_marginal = results["unit"]
    tiny_value, tiny_policy, tiny_marginal = results["tiny"]
    context = f"{np.dtype(dtype)} block={block_size} {results}"
    assert unit_policy == tiny_policy == 1.0, context
    assert unit_marginal == tiny_marginal == 20.0, context
    # The value is compared as a RATIO so the assertion is about invariance
    # rather than about either leg's absolute level, which the case above pins.
    assert tiny_value / float(tiny) == unit_value, context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_certified_tie_never_coexists_with_a_strict_exact_sign(dtype):
    """The selection's central invariant, checked where it previously broke.

    `_exactly_maximal` may skip `_exact_compare` only for candidates certified
    tied. If it ever certifies a tie between two candidates that `_exact_compare`
    strictly orders, the exact comparator has been bypassed on precisely the
    input it exists for, and the answer is whatever the approximate layer said.

    Round 11 read `radius == 0` as that certificate. A radius is a float: at the
    bottom of the range `eps**2 * |v|` underflows, and two candidates that were
    not tied at all presented as certifiably tied while `_exact_compare` returned
    the correct strict sign `+1` (round-11 audit F2/RT11). The certificate is now
    structural -- a node event, which publishes stored data and performs no
    arithmetic -- so no numerical accident can manufacture it.

    This is the INVERSE of the reviewer's localization artifact, which asserts the
    defect reproduces. Adopting that script as a regression would pin the broken
    state; what belongs in the suite is the invariant it violates.
    """
    lower, upper = _smallest_value_scale_case(dtype)
    q = dtype(0.75)
    # `[left_grid, right_grid, left_value, right_value, l/r policy, l/r marginal]`
    flat = [0.0, 1.0, lower, lower, 0.0, 0.0, 10.0, 10.0]
    rising = [0.0, 1.0, lower, upper, 1.0, 1.0, 20.0, 20.0]
    block = jnp.asarray([flat, rising], dtype=dtype)

    terms = _candidate_terms(
        block=block,
        live=jnp.asarray([True, True]),
        flat=jnp.asarray([q], dtype=dtype),
    )
    tied = _exactly_maximal(
        terms=terms, gather=lambda index: block[index], q=jnp.asarray([q], dtype=dtype)
    )
    sign = float(
        _exact_compare(cols_a=block[1], cols_b=block[0], q=jnp.asarray(q, dtype=dtype))
    )

    context = f"{np.dtype(dtype)} tied={np.asarray(tied)} sign={sign}"
    # Self-check: the invariant is only meaningful if the exact comparator does
    # separate these two. A vacuous pass here would hide the whole defect.
    assert sign == 1.0, f"exact comparator must strictly order this pair: {context}"
    assert not bool(np.asarray(tied)[0, 0]), (
        f"the strictly LOWER candidate was certified tied with the winner: {context}"
    )
    assert bool(np.asarray(tied)[0, 1]), (
        f"the strict winner must be selected: {context}"
    )
    # Pin WHICH repair is load bearing here. The interior lane's radius still
    # underflows to zero at this scale -- that is a fact about the format, not a
    # defect -- so the assertions above hold only because the certificate stopped
    # being numerical. If an interpolated lane ever reports itself structurally
    # exact, the shortcut is back, whatever the radius happens to be.
    assert not bool(np.asarray(terms.exact)[0, 1]), (
        f"an interpolated interior lane must NOT be structurally exact: {context}"
    )


def _wide_segment_case(dtype):
    """A bracketing segment whose own endpoints span most of the exponent range.

    `x0 = 1`, `q = nextafter(1)`, `x1 = 2**(maxexp-2)` with values `(0, x1)`, so
    the exact interpolant at `q` is `(q-x0)/(x1-x0) * x1`, which is `eps` to
    within a rounding -- far above the competitor placed two binades below it.
    Every stored input is finite normal.
    """
    maxexp = int(np.finfo(dtype).maxexp)
    nmant = int(np.finfo(dtype).nmant)
    x0 = dtype(1.0)
    q = np.nextafter(x0, dtype(np.inf))
    x1 = dtype(np.ldexp(1.0, maxexp - 2))
    competitor = dtype(np.ldexp(1.0, -(nmant + 2)))
    exact = (
        (Fraction(float(q)) - Fraction(float(x0)))
        / (Fraction(float(x1)) - Fraction(float(x0)))
    ) * Fraction(float(x1))
    return x0, q, x1, competitor, exact


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("block_size", [0, 2, 3])
def test_a_wide_segments_own_range_cannot_erase_its_interpolation_fraction(
    dtype, order, block_size
):
    """`q - x0` must survive a segment whose own endpoints span many binades.

    Rounds 9 and 10 both scaled the OPERANDS before differencing them -- round 9
    with one exponent for the whole array, round 10 with one per candidate. Both
    lose the same way: scaling `q` and `x0` before subtracting can map two
    distinct represented grid points onto the same float, after which `q - x0` is
    zero and no downstream exactness can rebuild it. Here the candidate exponent
    is set by its own far endpoint, so a candidate that is entirely relevant
    destroys its own interpolation fraction and collapses to its left value,
    handing value, policy and marginal to a strictly lower competitor
    (round-10 audit F2).

    The repair forms the differences first, on the raw operands where `_two_diff`
    is exact, and carries each scale as an integer exponent applied once at the
    end.
    """
    x0, q, x1, competitor, exact = _wide_segment_case(dtype)
    info = np.finfo(dtype)
    for name, val in (("x0", x0), ("q", q), ("x1", x1), ("competitor", competitor)):
        assert np.isfinite(val), name
        assert abs(val) >= info.tiny, f"{name} must be finite NORMAL"
    assert exact > Fraction(float(competitor)), "the witness needs a strict ordering"
    # The collision that used to cause the defect, asserted so the case cannot
    # quietly stop exercising it.
    wide_exp = int(np.frexp(x1)[1])
    assert dtype(np.ldexp(np.float64(x0), -wide_exp)) == dtype(
        np.ldexp(np.float64(q), -wide_exp)
    )

    wide = ([float(x0), float(x1)], [dtype(0.0), x1], 1.0, 20.0)
    flat = ([float(x0), 2.0], [competitor, competitor], 0.0, 10.0)
    rows = [wide, flat] if order == "AB" else [flat, wide]

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
        x_query=jnp.asarray([float(q)], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size}"
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 20.0, context
    # The published level is the correctly rounded exact interpolant.
    assert float(got_value[0]) == float(dtype(float(exact))), context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_compare_orders_a_wide_segment_against_a_lower_competitor(dtype):
    """The certified comparator must not need the screen's help to order these.

    Probes `_exact_compare` DIRECTLY, because the public path passes this witness
    for the wrong reason: the value screen resolves the gap before the comparator
    is consulted, so a comparator that reports a tie where the exact ordering is
    strict stays invisible there. Two rounds of the round-10 F2 repair passed the
    public test with this one still returning `0` (it was pinned `xfail(strict)`
    in between). A screen that rescues a broken comparator hides it.
    """
    x0, q, x1, competitor, exact = _wide_segment_case(dtype)
    cols_wide = jnp.asarray(
        [[x0, x1, dtype(0.0), x1, 1.0, 20.0, 1.0, 20.0]], dtype=dtype
    )
    cols_flat = jnp.asarray(
        [[x0, dtype(2.0), competitor, competitor, 0.0, 10.0, 0.0, 10.0]], dtype=dtype
    )
    sign = _exact_compare(
        cols_a=cols_wide, cols_b=cols_flat, q=jnp.asarray([float(q)], dtype=dtype)
    )
    assert exact > Fraction(float(competitor))
    assert float(sign[0]) == 1.0


# ---------------------------------------------------------------------------
# Round-13: the exponent-preserving exact ordering kernel.
#
# Rounds 6 to 12 each repaired the SITE a witness pointed at — a value
# comparison, a slope tie-break, an exponent, a normalization's scope, then its
# shape, then the last normalization inside the "exact" fallback — and every
# time the same reasoning error reappeared one layer down. The premise all nine
# repairs shared is that a rescaling can make a rounded evaluation safe. It
# cannot, and these tests pin the two things that replaced it: exponents carried
# as integers, and a fixed-point accumulator wide enough that no term can fall
# out of it.
# ---------------------------------------------------------------------------


def _cancelling_pair_case(dtype, value_exponent, grid_exponent, ulp_offset):
    """Branch A cancelling to exactly zero against a finite-normal constant B.

    `A` runs from `+a/2` to `-a/2` across the segment, so its exact value at the
    midpoint is zero — its two numerator products cancel COMPLETELY, which is
    what makes the comparison hinge entirely on `B`'s terms. `B` is the constant
    `-a * tiny`, a strictly lower finite normal whose cross products used to
    flush to zero before the exact summation saw them (round-12 F2).
    """
    info = np.finfo(dtype)
    tiny = dtype(info.tiny)
    amplitude = dtype(np.ldexp(1.0, value_exponent))
    left_value = dtype(amplitude * dtype(0.5))
    right_value = dtype(-amplitude * dtype(0.5))
    gap = dtype(amplitude * dtype(tiny + dtype(ulp_offset) * dtype(np.spacing(tiny))))
    width = dtype(np.ldexp(1.0, grid_exponent))
    midpoint = dtype(width * dtype(0.5))
    for name, value in (
        ("amplitude", amplitude),
        ("left", left_value),
        ("right", right_value),
        ("gap", gap),
        ("width", width),
        ("midpoint", midpoint),
    ):
        assert np.isfinite(value), name
        assert abs(value) >= info.tiny, f"{name} must be finite NORMAL"
    a_row = ([dtype(0.0), width], [left_value, right_value], 1.0, 20.0)
    b_row = ([dtype(0.0), width], [-gap, -gap], 0.0, 10.0)
    return a_row, b_row, midpoint, gap


def _assemble(rows):
    grid, value, policy, marginal, segment_id = [], [], [], [], []
    for sid, (g, v, p, m) in enumerate(rows):
        grid += list(g)
        value += list(v)
        policy += [p] * 2
        marginal += [m] * 2
        segment_id += [float(sid)] * 2
    return grid, value, policy, marginal, segment_id


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("block_size", [0, 2, 3])
@pytest.mark.parametrize("order", ["AB", "BA"])
@pytest.mark.parametrize("ulp_offset", [0, 1, 64])
def test_a_completely_cancelling_branch_still_outranks_a_lower_constant(
    dtype, block_size, order, ulp_offset
):
    """A strict gap must survive however far the decisive terms sit below the rest.

    Branch A's numerator cancels to exactly zero, so its own terms carry no
    information about the answer and the entire comparison rests on branch B's,
    which live `~2 * precision + emax` binades further down. Every normalization
    tried through round 12 pushed those terms out of the format — the products
    came back as `(-0.0, 0.0)` and `_exact_compare` reported a TIE where the
    exact ordering is strict, so the right-continuity rule handed value, policy
    and marginal to the lower branch.

    Carrying each term's exponent as an integer and depositing it into a
    fixed-point accumulator removes the frame the terms were falling out of.
    """
    a_row, b_row, midpoint, gap = _cancelling_pair_case(
        dtype,
        value_exponent=0 if ulp_offset else int(np.finfo(dtype).maxexp) - 1,
        grid_exponent=-60,
        ulp_offset=ulp_offset,
    )
    rows = [a_row, b_row] if order == "AB" else [b_row, a_row]
    grid, value, policy, marginal, segment_id = _assemble(rows)

    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=jnp.asarray(grid, dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        marginal=jnp.asarray(marginal, dtype=dtype),
        segment_id=jnp.asarray(segment_id, dtype=dtype),
        x_query=jnp.asarray([float(midpoint)], dtype=dtype),
        segment_block_size=block_size,
    )
    context = f"{np.dtype(dtype)} order={order} block={block_size} ulp={ulp_offset}"
    # The witness has to be a strict gap, or it asserts nothing.
    assert Fraction(0) > -Fraction(float(gap)), context
    assert float(got_value[0]) == 0.0, context
    assert float(got_policy[0]) == 1.0, context
    assert float(got_marginal[0]) == 20.0, context


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_every_cross_product_term_reaches_the_exact_accumulator(dtype):
    """No cross product may be lost, merged or rounded before the sign is read.

    Reads the dyadic term list straight out of `_dyadic_product` and sums it in
    `Fraction`. The result must equal the cross difference computed independently
    from the two ratios — and dropping the half that round 12 flushed must
    collapse that difference to zero, so the test cannot pass vacuously.
    """
    info = np.finfo(dtype)
    tiny = dtype(info.tiny)
    cols_a = jnp.asarray([0, 1, 0.5, -0.5, 1, 1, 20, 20], dtype=dtype)
    cols_b = jnp.asarray([0, 1, -tiny, -tiny, 0, 0, 10, 10], dtype=dtype)
    q = jnp.asarray(0.5, dtype=dtype)

    def total(dyadic):
        mantissa = np.asarray(dyadic.mantissa).reshape(-1)
        exponent = np.asarray(dyadic.exponent).reshape(-1)
        return sum(
            (
                Fraction(float(m)) * Fraction(2) ** int(e)
                for m, e in zip(mantissa, exponent, strict=True)
            ),
            Fraction(0),
        )

    ratio_a = _exact_ratio(cols=cols_a, q=q)
    ratio_b = _exact_ratio(cols=cols_b, q=q)
    expected = total(ratio_a.numerator) * total(ratio_b.denominator) - total(
        ratio_b.numerator
    ) * total(ratio_a.denominator)

    split = _dekker_split_factor(cols_a.dtype)
    forward = total(_dyadic_product(ratio_a.numerator, ratio_b.denominator, split))
    reverse = total(_dyadic_product(ratio_b.numerator, ratio_a.denominator, split))

    assert expected > 0, "the witness must be a strict gap"
    assert forward - reverse == expected
    # Round 12's behaviour: with `reverse` flushed the difference is exactly zero
    # and the comparator reports a tie. If that does not happen the assertion
    # above is not testing what it claims to.
    assert forward == 0
    assert float(_exact_compare(cols_a=cols_a, cols_b=cols_b, q=q)) == 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_the_winner_is_always_one_of_the_exactly_tied_candidates(dtype, order):
    """An overflowing slope key must not seed the lead outside the tie set.

    `_right_continuous_winner` masks non-competitors with `-inf`, and `-inf` is
    also a REACHABLE rounded slope key: `fl(v1-v0) / fl(x1-x0)` overflows when a
    large value gap meets a small grid width, with every stored input still
    finite normal. `argmax` then cannot tell the sole competitor from the masked
    candidates, returns index 0, and the loop seeds its lead there and never
    revisits it — so the published policy and marginal follow branch ORDER while
    the value stays right.

    The invariant is structural and survives any future screen: whatever wins
    must be a candidate the exact value comparison declared maximal.
    """
    a_row, b_row, midpoint, _ = _cancelling_pair_case(
        dtype,
        value_exponent=int(np.finfo(dtype).maxexp) - 1,
        grid_exponent=-60,
        ulp_offset=0,
    )
    a_columns = [0.0, float(a_row[0][1]), *(float(v) for v in a_row[1]), 1, 1, 20, 20]
    b_columns = [0.0, float(b_row[0][1]), *(float(v) for v in b_row[1]), 0, 0, 10, 10]
    rows = [a_columns, b_columns] if order == "AB" else [b_columns, a_columns]
    columns = jnp.asarray(rows, dtype=dtype)
    live = jnp.asarray([True, True])
    flat = jnp.asarray([float(midpoint)], dtype=dtype)

    terms = _candidate_terms(block=columns, live=live, flat=flat)
    tied = _exactly_maximal(terms=terms, gather=lambda index: columns[index], q=flat)
    _, winner = _right_continuous_winner(
        tied=tied, terms=terms, gather=lambda index: columns[index]
    )
    # The witness must actually exercise the collision, or it asserts nothing.
    assert bool(jnp.any(jnp.isinf(jnp.where(tied, terms.slope, 0.0)))), (
        "no tied candidate carries an overflowed slope key"
    )
    assert bool(jnp.any(tied)), "the exact comparison resolved nothing"
    assert bool(tied[0, int(winner[0])]), (
        f"{np.dtype(dtype)} order={order}: winner {int(winner[0])} is not in the "
        f"tie set {np.asarray(tied)[0]}"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_finite_input_overflows_the_exact_accumulator(dtype):
    """A randomised sweep across the WHOLE exponent range against a rational oracle.

    The accumulator is sized from the format, and a term that fell off its end
    would surface as a NaN sign rather than a wrong one — this asserts neither
    happens. Grid points, query and endpoint values are drawn from binades
    spanning most of the representable range, which is the regime every previous
    repair broke in.
    """
    info = np.finfo(dtype)
    limit = int(info.maxexp) - 8
    generator = np.random.default_rng(20260731)

    def draw():
        return dtype(
            np.ldexp(
                generator.uniform(-2.0, 2.0), int(generator.integers(-limit, limit))
            )
        )

    def rational(cols, q):
        x0, x1, v0, v1 = (Fraction(float(c)) for c in cols[:4])
        query = Fraction(float(q))
        if query == x0:
            return v0
        if query == x1:
            return v1
        return (v0 * (x1 - query) + v1 * (query - x0)) / (x1 - x0)

    checks = 0
    for _ in range(300):
        x0, x1 = sorted([draw(), draw()])
        if x0 == x1 or not np.isfinite(x1 - x0):
            continue
        q = dtype(x0 + (x1 - x0) * dtype(generator.uniform(0.0, 1.0)))
        if not x0 <= q <= x1:
            continue
        cols_a = [x0, x1, draw(), draw(), 1.0, 20.0, 1.0, 20.0]
        cols_b = [x0, x1, draw(), draw(), 0.0, 10.0, 0.0, 10.0]
        sign = float(
            _exact_compare(
                cols_a=jnp.asarray(cols_a, dtype=dtype),
                cols_b=jnp.asarray(cols_b, dtype=dtype),
                q=jnp.asarray(q, dtype=dtype),
            )
        )
        expected = rational(cols_a, q) - rational(cols_b, q)
        want = 0.0 if expected == 0 else (1.0 if expected > 0 else -1.0)
        assert sign == want, f"{cols_a} vs {cols_b} at {q}: {sign} != {want}"
        checks += 1
    assert checks > 200, f"only {checks} usable draws — the sweep degenerated"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_difference_is_never_formed_in_the_working_dtype(dtype):
    """`H - (-H)` must be represented, not computed.

    Round 13 built differences with `_two_diff` on operands lifted into
    mid-range, and justified it with the claim that subtraction of two finite
    floats cannot overflow. Opposite-signed top-binade operands refute that:
    for `H = 2**127` in float32 both operands are finite normals while their
    difference `2H` is not representable at all (round-13 audit F2).

    So the assertion is not that the difference is finite — it is that no float
    is ever asked to hold it. Every mantissa stays inside its own binade and the
    magnitude lives in the integer exponent.
    """
    top = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1))
    terms = _exact_difference(
        jnp.asarray(top, dtype=dtype), jnp.asarray(-top, dtype=dtype)
    )
    mantissa = np.asarray(terms.mantissa)
    exponent = np.asarray(terms.exponent)

    assert np.all(np.isfinite(mantissa)), f"a mantissa left the format: {mantissa}"
    assert np.all(np.abs(mantissa) < 1.0), f"a mantissa is not normalized: {mantissa}"
    total = sum(
        Fraction(float(m)) * Fraction(2) ** int(e)
        for m, e in zip(mantissa.ravel(), exponent.ravel(), strict=True)
    )
    assert total == 2 * Fraction(float(top)), f"terms sum to {total}, not 2H"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_framed_difference_is_exact_at_the_top_of_the_range(dtype):
    """The screen's difference must not overflow where the exact one does not.

    `_candidate_terms` formed its grid width, its endpoint gap and its slope key
    as working-dtype subtractions after a lift-only shift, so the public path
    carried the same defect as the exact kernel and published `(NaN, NaN, NaN)`.
    """
    top = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1))
    head, tail, exponent = _framed_difference(
        jnp.asarray(top, dtype=dtype), jnp.asarray(-top, dtype=dtype)
    )
    assert np.isfinite(np.asarray(head)), f"framed head is {head}"
    assert np.isfinite(np.asarray(tail)), f"framed tail is {tail}"
    recovered = (
        Fraction(float(np.asarray(head))) + Fraction(float(np.asarray(tail)))
    ) * Fraction(2) ** int(np.asarray(exponent))
    assert recovered == 2 * Fraction(float(top)), f"framed pair recovers {recovered}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", ["grid-width", "endpoint-value"])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("order", [0, 1])
def test_a_top_binade_segment_still_publishes_its_envelope(
    dtype, axis, fraction, order
):
    """A finite exact envelope must never be published as three NaNs.

    Two overflow axes are covered: a segment whose abscissa SPAN is not
    representable, and one whose endpoint VALUE gap is not. The expected triple
    is derived here in exact rationals rather than written as a literal — on the
    value axis the sloped branch legitimately WINS at some query positions, and
    a hand-written expectation gets that wrong.

    Note what the round-13 sweep did instead: it drew wide-range operands and
    then skipped any draw with `not np.isfinite(x1 - x0)`, which discards
    precisely the cells this test keeps. A filter written in the same arithmetic
    as the defect cannot witness it.
    """
    top = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1))
    if axis == "grid-width":
        x0, x1 = dtype(-top), dtype(top)
        v0, v1 = dtype(0.5), dtype(-0.5)
    else:
        x0, x1 = dtype(0.0), dtype(1.0)
        v0, v1 = dtype(top), dtype(-top)

    # In exact rationals: the obvious `x0 + f * (x1 - x0)` overflows on exactly
    # these inputs, so the harness would reproduce the defect it adjudicates.
    query = dtype(
        float(
            Fraction(float(x0))
            + Fraction(fraction) * (Fraction(float(x1)) - Fraction(float(x0)))
        )
    )
    sloped = ([x0, x1], [v0, v1], [1.0, 1.0], [20.0, 20.0])
    constant = ([x0, x1], [0.25, 0.25], [0.0, 0.0], [10.0, 10.0])
    branches = [sloped, constant] if order == 0 else [constant, sloped]

    grid, value, policy, marginal, segment = [], [], [], [], []
    for index, (xs, vs, ps, ms) in enumerate(branches):
        grid += xs
        value += vs
        policy += ps
        marginal += ms
        segment += [index, index]

    got = envelope_at_query(
        endog_grid=jnp.asarray(grid, dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        marginal=jnp.asarray(marginal, dtype=dtype),
        segment_id=jnp.asarray(segment, dtype=dtype),
        x_query=jnp.asarray([query], dtype=dtype),
    )
    # The exact envelope, in rationals: per branch the affine value at the
    # query, then the maximum, breaking an exact tie toward the larger slope as
    # the right-continuous rule does.
    weight = Fraction(fraction)
    best = None
    for xs, vs, ps, ms in branches:
        start, end = (Fraction(float(a)) for a in (vs[0], vs[1]))
        exact_value = start + weight * (end - start)
        span = Fraction(float(xs[1])) - Fraction(float(xs[0]))
        slope = (end - start) / span
        entry = (
            (exact_value, slope),
            (
                exact_value,
                Fraction(float(ps[0])) + weight * Fraction(float(ps[1]) - float(ps[0])),
                Fraction(float(ms[0])) + weight * Fraction(float(ms[1]) - float(ms[0])),
            ),
        )
        if best is None or entry[0] > best[0]:
            best = entry
    assert best is not None, "both branches bracket the query by construction"
    expected = tuple(float(dtype(float(item))) for item in best[1])

    published = tuple(float(np.asarray(item)[0]) for item in got)
    assert all(np.isfinite(item) for item in published), (
        f"{axis} at fraction {fraction}, branch order {order}: {published}"
    )
    assert published == expected, (
        f"{axis} at fraction {fraction}, branch order {order}: "
        f"{published} != {expected}"
    )
