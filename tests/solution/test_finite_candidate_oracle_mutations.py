"""The exact finite candidate reference rejects splices, stale floors and bad ties.

These tests falsify the reference's own sensitivity on synthetic tuples;
`test_finite_candidate_tuple_economics` binds the same reference to real
producer outputs.
"""

import math
from dataclasses import replace
from fractions import Fraction

import pytest

from tests.solution._finite_candidate_budget_oracle import (
    Candidate,
    assert_tuple,
    conditional_bank,
    owner,
)


def _check(row: Candidate, **changes: float) -> None:
    observed = {"action": row.action, "value": row.value, "target": row.target}
    observed.update(changes)
    assert_tuple(expected=row, **observed, abs_tol=1e-12)


def test_conditional_bank_ranks_zero_state_by_exact_mass() -> None:
    """At liquid=stock=0 the masses are 5/4, 5/4, 3/2 and a dead target 2."""
    bank = conditional_bank(liquid=Fraction(0), stock=0, dynamic_floor=True)
    assert tuple(row.mass for row in bank) == (
        Fraction(5, 4),
        Fraction(5, 4),
        Fraction(3, 2),
        None,
    )


def test_owner_at_zero_state_is_target_one() -> None:
    """Target 1 with consumption 3/4 owns liquid=stock=0."""
    bank = conditional_bank(liquid=Fraction(0), stock=0, dynamic_floor=True)
    assert (owner(bank), bank[2].action, bank[2].target) == (2, 0.75, 1.0)


def test_stock_one_keeper_ties_target_two_and_keeper_wins() -> None:
    """At stock 1 the keeper and target 2 tie exactly; the keeper owns the state."""
    bank = conditional_bank(liquid=Fraction(0), stock=1, dynamic_floor=True)
    assert (tuple(row.action for row in bank), owner(bank)) == (
        (1.0, 0.5, 1.0, 0.5),
        0,
    )


def test_owner_without_live_candidate_raises() -> None:
    """A bank of dead candidates has no owner."""
    dead = conditional_bank(liquid=Fraction(0), stock=0, dynamic_floor=True)[3]
    with pytest.raises(ValueError, match="No live candidate"):
        owner((dead,))


@pytest.mark.parametrize("field", ["action", "value", "target"])
def test_dead_tuple_rejects_one_live_component(*, field: str) -> None:
    """A dead candidate with any one live field is rejected."""
    dead = conditional_bank(liquid=Fraction(0), stock=0, dynamic_floor=True)[3]
    _check(dead)
    replacement = {"action": 0.125, "value": 0.0, "target": 2.0}[field]
    with pytest.raises(AssertionError):
        _check(dead, **{field: replacement})


@pytest.mark.parametrize("field", ["action", "value", "target"])
def test_live_neighbor_rejects_one_dead_component(*, field: str) -> None:
    """A live candidate with any one dead field is rejected."""
    live = conditional_bank(liquid=Fraction(0), stock=1, dynamic_floor=True)[3]
    _check(live)
    replacement = -math.inf if field == "value" else math.nan
    with pytest.raises(AssertionError):
        _check(live, **{field: replacement})


def test_direct_control_with_template_zero_floor_is_rejected() -> None:
    """A control run on a zero floor instead of the current floor 1/4 is rejected."""
    expected = conditional_bank(liquid=Fraction(0), stock=1, dynamic_floor=True)[1]
    contaminated = conditional_bank(liquid=Fraction(0), stock=1, dynamic_floor=False)[1]
    assert (expected.floor, contaminated.action, expected.action) == (
        Fraction(1, 4),
        0.75,
        0.5,
    )
    with pytest.raises(AssertionError):
        _check(expected, action=contaminated.action, value=contaminated.value)


@pytest.mark.parametrize("side", ["below", "exact"])
def test_budget_at_or_below_floor_is_dead(*, side: str) -> None:
    """Target 2 from stock 0 is dead at liquid 3/4 and its float predecessor."""
    liquid = (
        Fraction(math.nextafter(0.75, -math.inf)) if side == "below" else Fraction(3, 4)
    )
    row = conditional_bank(liquid=liquid, stock=0, dynamic_floor=True)[3]
    assert row.mass is None
    _check(row)


def test_budget_just_above_floor_is_live() -> None:
    """Target 2 from stock 0 is live at the float successor of 3/4."""
    above = Fraction(math.nextafter(0.75, math.inf))
    row = conditional_bank(liquid=above, stock=0, dynamic_floor=True)[3]
    assert row.mass is not None
    assert row.consumption > 0
    _check(row)
    with pytest.raises(AssertionError):
        _check(row, target=math.nan)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_exact_keeper_tie_survives_declared_outer_permutation(
    *, order: tuple[int, int, int]
) -> None:
    """The keeper wins its exact tie under every declared target order."""
    bank = conditional_bank(
        liquid=Fraction(0), stock=1, dynamic_floor=True, declared_targets=order
    )
    assert bank[0].mass == Fraction(2)
    assert bank[1 + order.index(2)].mass == bank[0].mass
    assert owner(bank) == 0


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_equal_adjusters_after_dead_keeper_keep_declared_order(
    *, order: tuple[int, int, int]
) -> None:
    """Equal legal candidates after a dead keeper resolve to the first declared."""
    bank = conditional_bank(
        liquid=Fraction(0), stock=1, dynamic_floor=True, declared_targets=order
    )
    dead_keeper = replace(
        bank[0], mass=None, value=-math.inf, action=math.nan, target=math.nan
    )
    assert owner((dead_keeper, bank[0], bank[1 + order.index(2)])) == 1


def test_positive_affine_value_change_preserves_exact_owner() -> None:
    """A positive affine value transformation leaves the exact owner unchanged."""
    bank = conditional_bank(liquid=Fraction(0), stock=1, dynamic_floor=True)
    transformed = tuple(
        replace(row, value=3 * row.value + 7) if row.mass is not None else row
        for row in bank
    )
    float_owner = max(range(len(transformed)), key=lambda i: transformed[i].value)
    assert float_owner == owner(bank) == 0
