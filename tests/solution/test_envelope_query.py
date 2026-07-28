"""The query-side envelope backend matches the host oracle exactly.

`envelope_at_query` evaluates the branch-aware upper envelope directly at query
abscissae. It must agree with the exact host oracle on value and policy across
the cases that distinguish the topology contract: a clean crossing, a folded
branch, and a non-bridging branch the inference backends get wrong.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import _candidate_terms, envelope_at_query
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
