"""Exact algebra tests for stable winner identity across interval blocks."""

import ast
from collections.abc import Callable
from fractions import Fraction
from itertools import permutations
from pathlib import Path

from _interval_block_winner_oracle import (
    Candidate,
    Identity,
    Ordering,
    blocked_winner,
    compositions,
    every_carry_slot,
    one_shot_winner,
    partition_by_sizes,
    winner_signature,
)

_CandidateFactory = Callable[..., tuple[Candidate, ...]]

ROOT = Path(__file__).resolve().parents[2]
QUERY_SOURCE = ROOT / "src/_lcm/egm/upper_envelope/query.py"


def _production_has_stable_index_field() -> bool:
    """Read the public repair seam without importing the Python-3.14 package."""
    tree = ast.parse(QUERY_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_TieBreakKey":
            return any(
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "stable_index"
                for statement in node.body
            )
    return False


def _ordinary_candidates(*, n: int, depth: str) -> tuple[Candidate, ...]:
    """Construct a family whose winner is decided at the requested rank depth."""
    midpoint = n // 2
    result: list[Candidate] = []
    for index in range(n):
        value = Fraction(10)
        right = True
        slope_high = Fraction(4)
        slope_low = Fraction(2)
        if depth == "value":
            value = Fraction(10 + (index == midpoint))
        elif depth == "right":
            right = index == midpoint
        elif depth == "slope_high":
            slope_high = Fraction(4 + (index == midpoint))
        elif depth == "slope_low":
            slope_low = Fraction(2 + (index == midpoint))
        elif depth != "stable_index":
            raise ValueError(depth)
        result.append(
            Candidate(
                stable_index=index,
                value=value,
                right_available=right,
                slope_high=slope_high,
                slope_low=slope_low,
                policy=Fraction(1000 + index),
                marginal=Fraction(2000 + index),
            )
        )
    return tuple(result)


def _certified_candidates(*, n: int, depth: str) -> tuple[Candidate, ...]:
    midpoint = n // 2
    result: list[Candidate] = []
    for index in range(n):
        value = Fraction(7, 3)
        right = True
        exact_slope = Fraction(11, 5)
        if depth == "value":
            value += Fraction(index == midpoint, 17)
        elif depth == "right":
            right = index == midpoint
        elif depth == "slope":
            exact_slope += Fraction(index == midpoint, 19)
        elif depth != "stable_index":
            raise ValueError(depth)
        result.append(
            Candidate(
                stable_index=index,
                value=value,
                right_available=right,
                slope_high=exact_slope,
                policy=Fraction(3000 + index),
                marginal=Fraction(4000 + index),
            )
        )
    return tuple(result)


def _first_partition_mismatch(*, identity: Identity) -> str | None:
    """Exhaust the small counterexample class and return the first mismatch."""
    ordinary_depths = ("value", "right", "slope_high", "slope_low", "stable_index")
    certified_depths = ("value", "right", "slope", "stable_index")
    scenarios: list[tuple[Ordering, str, _CandidateFactory]] = [
        ("ordinary", depth, _ordinary_candidates) for depth in ordinary_depths
    ] + [("certified", depth, _certified_candidates) for depth in certified_depths]
    for ordering, depth, factory in scenarios:
        for n_candidates in range(2, 8):
            candidates = factory(n=n_candidates, depth=depth)
            expected = winner_signature(
                one_shot_winner(candidates=candidates, ordering=ordering)
            )
            # Full permutations at n<=6 and a deterministic exhaustive set of
            # one-element relocations at n=7 keep the RED/GREEN command bounded
            # while still crossing every stable id with every block boundary.
            arrangements = (
                permutations(candidates)
                if n_candidates <= 6
                else (
                    tuple(candidates[offset:] + candidates[:offset])
                    for offset in range(n_candidates)
                )
            )
            for offered in arrangements:
                arrangement = tuple(offered)
                for sizes in compositions(n_candidates):
                    blocks = partition_by_sizes(candidates=arrangement, sizes=sizes)
                    for slots in every_carry_slot(len(blocks) - 1):
                        observed = winner_signature(
                            blocked_winner(
                                blocks=blocks,
                                ordering=ordering,
                                identity=identity,
                                carry_slots=slots,
                            )
                        )
                        if observed != expected:
                            return (
                                f"ordering={ordering} depth={depth} n={n_candidates} "
                                f"arrangement={[c.stable_index for c in arrangement]} "
                                f"sizes={sizes} slots={slots} expected={expected} "
                                f"observed={observed}"
                            )
    return None


def test_production_identity_is_partition_invariant() -> None:
    """RED at baseline: production's position identity changes under blocking."""
    identity = "stable" if _production_has_stable_index_field() else "position"
    mismatch = _first_partition_mismatch(identity=identity)
    assert mismatch is None, (
        "blocked winner differs from the one-shot stable-index oracle because "
        f"production currently uses {identity!r} identity: {mismatch}"
    )


def test_position_based_positive_control_is_rejected() -> None:
    """Prove the enumeration distinguishes the historical defective semantics."""
    assert _first_partition_mismatch(identity="position") is not None


def test_empty_and_excluded_blocks_are_identity_elements() -> None:
    winner = Candidate(
        stable_index=5,
        value=Fraction(1),
        right_available=True,
        slope_high=Fraction(0),
        policy=Fraction(9),
        marginal=Fraction(8),
    )
    excluded = Candidate(
        stable_index=0,
        value=Fraction(99),
        right_available=True,
        slope_high=Fraction(99),
        brackets=False,
    )
    observed = blocked_winner(
        blocks=((), (excluded,), (winner,), (), (excluded,)),
        ordering="ordinary",
        identity="stable",
        carry_slots=("head", "middle", "tail", "head"),
    )
    assert winner_signature(observed) == winner_signature(winner)
    assert (
        blocked_winner(
            blocks=((), (excluded,), ()),
            ordering="certified",
            identity="stable",
        )
        is None
    )
