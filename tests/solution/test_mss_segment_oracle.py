"""Exact-oracle contract for the MSS segment envelope.

These tests pin the geometric and numerical facts the optimized `refine_envelope`
must satisfy, expressed against the independent rational oracle in
`_mss_segment_oracle`: open-cell ownership (a middle branch that owns an open
interval while only tying at its boundaries), branch/run count vs discrete-action
count, node-aligned crossing classification by certified margin sign, one-sided
ownership at higher-order ties, translation invariance, and the collinear-overlap
tie interval.
"""

import itertools
import random
from fractions import Fraction

from tests.solution._mss_segment_oracle import (
    Branch,
    FloatSegment,
    all_breakpoints,
    brute_pair_crossings,
    classify_cell,
    exact_envelope,
    exact_margin,
    has_collinear_overlap,
    merge_pair_crossings,
    to_fraction,
)

F = Fraction


def test_f1_middle_owner_requires_open_cell_ownership_not_only_vertex_argmax():
    """A branch owning only an open interval is invisible to vertex-only argmax."""
    branches = (
        Branch("A", (F(0), F(1)), (F(9, 10), F(1)), (F(10), F(10))),
        Branch("B", (F(1), F(2)), (F(1), F(6, 5)), (F(5), F(5))),
        Branch("C", (F(2), F(3)), (F(6, 5), F(11, 5)), (F(1), F(1))),
    )
    points = all_breakpoints(branches)
    assert points == {F(0), F(1), F(2), F(3)}
    assert exact_envelope(branches, F(3, 2))[1] == ("B",)

    # A deterministic branch-id tie order can pick A and C at the two breakpoint
    # vertices even though B uniquely owns the open cell between them.
    rank = {"A": 0, "C": 1, "B": 2}
    assert min(exact_envelope(branches, F(1))[1], key=rank.__getitem__) == "A"
    assert min(exact_envelope(branches, F(2))[1], key=rank.__getitem__) == "C"


def test_one_action_does_not_bound_the_number_of_monotone_runs():
    """A single savings chain can fold into several x-monotone runs."""
    savings = tuple(map(F, range(8)))
    consumption = tuple(map(F, (10, 13, 9, 14, 8, 15, 7, 16)))
    resources = tuple(s + c for s, c in zip(savings, consumption, strict=True))
    n_runs = 1 + sum(b < a for a, b in itertools.pairwise(resources))
    assert n_runs == 4


def test_decreases_heuristic_would_bridge_a_boundary_that_raises_grid_and_value():
    """A run boundary that raises both grid and value escapes a value-decrease split."""
    grid = [F(0), F(1), F(2), F(3), F(3, 2), F(7, 4)]
    value = [F(0), F(1), F(4), F(5), F(1, 2), F(1, 2)]
    decreases = [
        (x1 < x0) or (v1 < v0)
        for x0, x1, v0, v1 in zip(grid, grid[1:], value, value[1:], strict=False)
    ]
    assert decreases[1] is False


def test_exact_endpoint_classification_uses_margin_signs_not_root_snapping():
    """A crossing exactly at a node is a right-endpoint event, from margin signs."""
    a = FloatSegment(9.0, 10.0, 4.875, 5.0, 8.0)
    b = FloatSegment(9.5, 10.5, 4.75, 5.25, 2.0)
    assert classify_cell(a, b, 9.5, 10.0) == ("right_endpoint", to_fraction(10.0))


def test_triple_crossing_has_only_two_one_sided_envelope_owners():
    """Three lines meeting at one abscissa still yield only left/right owners."""
    branches = (
        Branch("A", (F(0), F(2)), (F(-1, 10), F(1, 10)), (F(10), F(10))),
        Branch("B", (F(0), F(2)), (F(-1, 5), F(1, 5)), (F(5), F(5))),
        Branch("C", (F(0), F(2)), (F(-1), F(1)), (F(1), F(1))),
    )
    assert exact_envelope(branches, F(1, 2))[1] == ("A",)
    assert exact_envelope(branches, F(1))[1] == ("A", "B", "C")
    assert exact_envelope(branches, F(3, 2))[1] == ("C",)


def test_exact_margin_sign_does_not_use_absolute_value_level():
    """A strictly dominant branch stays strictly dominant under a large common shift."""
    a = FloatSegment(0.0, 1.0, 1.0e12, 1.0e12 + 0.001, 1000.0)
    b = FloatSegment(0.0, 1.0, 1.0e12 + 0.002, 1.0e12 + 0.003, 500.0)
    assert exact_margin(b, a, 0.5) > 0


def test_collinear_overlap_needs_a_tie_interval_convention():
    """Coincident value lines are a tie interval, not a finite list of crossings."""
    a = Branch("A", (F(-4), F(3)), (F(2), F(16)), (F(1), F(1)))
    b = Branch("B", (F(1), F(5)), (F(12), F(20)), (F(2), F(2)))
    assert brute_pair_crossings(a, b) == set()
    assert merge_pair_crossings(a, b) == {F(1), F(3)}
    assert exact_envelope((a, b), F(2))[1] == ("A", "B")


def test_consecutive_merge_matches_all_segment_pairs_on_random_branch_pairs():
    """The merged sweep enumerates exactly the brute-force crossing set."""
    rng = random.Random(20260727)  # noqa: S311  (reproducible test fixture, not crypto)
    checked = 0
    for _ in range(5000):
        n_a, n_b = rng.randint(2, 7), rng.randint(2, 7)
        x_a = sorted(rng.sample(range(-8, 13), n_a))
        x_b = sorted(rng.sample(range(-8, 13), n_b))
        v_a = [rng.randint(-20, 20) for _ in x_a]
        v_b = [rng.randint(-20, 20) for _ in x_b]
        a = Branch(
            "A", tuple(map(F, x_a)), tuple(map(F, v_a)), tuple(F(1) for _ in x_a)
        )
        b = Branch(
            "B", tuple(map(F, x_b)), tuple(map(F, v_b)), tuple(F(2) for _ in x_b)
        )
        if has_collinear_overlap(a, b):
            continue
        assert merge_pair_crossings(a, b) == brute_pair_crossings(a, b)
        checked += 1
    assert checked > 4900


def test_branch_order_and_common_value_translation_do_not_change_exact_geometry():
    """Envelope owner and value are invariant to branch order and a common shift."""
    branches = (
        Branch("A", (F(0), F(1)), (F(1), F(11, 10)), (F(10), F(10))),
        Branch("B", (F(0), F(1)), (F(21, 25), F(88, 75)), (F(3), F(3))),
        Branch("C", (F(3, 5), F(6, 5)), (F(4, 5), F(7, 5)), (F(1), F(1))),
    )
    query = F(4, 5)
    expected = exact_envelope(branches, query)
    for perm in itertools.permutations(branches):
        assert exact_envelope(perm, query) == expected
        shifted = tuple(
            Branch(b.name, b.x, tuple(v + F(10**12) for v in b.v), b.policy)
            for b in perm
        )
        shifted_value, shifted_owner = exact_envelope(shifted, query)
        assert shifted_owner == expected[1]
        assert shifted_value - F(10**12) == expected[0]
