"""Exact candidate-tuple reference for the finite nested beta-zero log toy.

The reference imports no project, JAX or NumPy code. It solves the toy of
`test_nested_dynamic_borrowing` by monotonicity and exact rational budget
arithmetic: with a zero discount factor every legal optimum saves exactly the
borrowing floor, so each candidate's consumption is `R - b`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Candidate:
    """An indivisible candidate tuple, including its exact ranking mass."""

    declared_target: int
    """Outer target the candidate stands for (the keeper's is the current stock)."""
    floor: Fraction
    """Borrowing floor of the current state, shared by every candidate."""
    resources: Fraction
    """Resources after the candidate's adjustment cost."""
    consumption: Fraction
    """Consumption `resources - floor`; legal only when positive."""
    value: float
    """`log(c) + target * log(2)` when legal, otherwise `-inf`."""
    action: float
    """Consumption when legal, otherwise NaN."""
    target: float
    """Published replay target when legal, otherwise NaN."""
    mass: Fraction | None
    """Exact ranking mass `c * 2**target`; `None` for a dead candidate."""


def conditional_bank(
    *,
    liquid: Fraction,
    stock: int,
    dynamic_floor: bool,
    declared_targets: tuple[int, ...] = (0, 1, 2),
) -> tuple[Candidate, ...]:
    """Return the keeper, then each declared target, with the floor held fixed.

    `R = x + 5/4 - (o - z)^2 / 2`. For integer `o`, maximizing
    `log(c) + o log(2)` is exactly maximizing `c * 2**o`, so ranking needs no
    floating-point tie band.

    Args:
        liquid: Current liquid state `x`.
        stock: Current illiquid stock `z`.
        dynamic_floor: Use the current-state floor `z / 4` instead of zero.
        declared_targets: Outer targets in their declared order.

    Returns:
        One candidate per keeper and declared target.

    """
    floor = Fraction(stock, 4) if dynamic_floor else Fraction(0)
    result = []
    for target in (stock, *declared_targets):
        resources = liquid + Fraction(5, 4) - Fraction((target - stock) ** 2, 2)
        consumption = resources - floor
        live = consumption > 0
        result.append(
            Candidate(
                declared_target=target,
                floor=floor,
                resources=resources,
                consumption=consumption,
                value=(
                    math.log(float(consumption)) + target * math.log(2)
                    if live
                    else -math.inf
                ),
                action=float(consumption) if live else math.nan,
                target=float(target) if live else math.nan,
                mass=consumption * Fraction(2) ** target if live else None,
            )
        )
    return tuple(result)


def owner(bank: Sequence[Candidate]) -> int:
    """Return the first exact maximum: keeper, then declared outer order."""
    eligible = [
        (index, row.mass) for index, row in enumerate(bank) if row.mass is not None
    ]
    if not eligible:
        msg = f"No live candidate exists in {bank!r}."
        raise ValueError(msg)
    best = max(mass for _, mass in eligible)
    return next(index for index, mass in eligible if mass == best)


def assert_tuple(
    *,
    expected: Candidate,
    action: float,
    value: float,
    target: float,
    abs_tol: float,
) -> None:
    """Check every association field; a dead action cannot carry a live target."""
    if expected.mass is None:
        assert math.isnan(action), (expected, "dead action", action)
        assert value == -math.inf, (expected, "dead value", value)
        assert math.isnan(target), (expected, "dead target", target)
        return
    assert math.isfinite(action), (expected, action)
    assert action > 0, (expected, action)
    assert math.isclose(action, expected.action, rel_tol=0, abs_tol=abs_tol)
    assert math.isclose(value, expected.value, rel_tol=0, abs_tol=abs_tol)
    assert target == expected.target, (expected, target)
    saving = float(expected.resources) - action
    assert math.isclose(saving, float(expected.floor), rel_tol=0, abs_tol=abs_tol)
