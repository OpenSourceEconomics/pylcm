"""Scalar reference for the finite nested mixed ordinary/corner query operator.

The operator blends each kind of alternative over the passive stencil and then
chooses at the actual query:

- the ordinary channel is the signed blend of the source rows' nodewise
  ordinary maxima over the outer candidates `O` (keeper first, then the
  declared outer nodes); it is supported only when every active row has a
  legal ordinary tuple, a supported `-inf` row makes it `-inf` at a positive
  weight and unsupported at a negative one;
- each candidate's corner is priced once at the actual budget, floor,
  preferences and discount on the signed blend of the rows' held floor
  continuations; it is finite only when every active row's record is known;
- the ordinary channel wins an exact tie, otherwise the first maximal finite
  corner in canonical outer order; without a supported ordinary channel and
  without a finite corner the read is uncovered.

`nodewise_reference` states the rejected order, which chooses on every source
row and blends the chosen tuples. Over nonnegative weights it never lies below
the blended read (Jensen on the row maxima), and it credits a different choice
on each row, which no single action realises.

The reference is written in plain Python scalars with `math`, without jax,
interpolation or any production helper. Its independence rests on three
choices: the economic laws of the canonical fixture are re-stated here from
their declarations, ordinary rows carry a constant value so no interpolant is
involved, and the folds are spelled out as loops rather than reductions.
Ordinary-only exact ties are outside this reference.
"""

import math
from dataclasses import dataclass

OUTER_NODES: tuple[float, ...] = (1.0, 2.0)
PASSIVE_NODES: tuple[float, ...] = (0.0, 1.0)
COORDINATE: float = 1.25
GROSS_OFFSET: float = 1.25
SAVINGS_SPAN: float = 2.0


@dataclass(frozen=True)
class OrdinaryRow:
    """One source row's constant ordinary tuple for one outer candidate."""

    value: float
    """Constant value of the row's bracket, read without interpolation."""
    marginal: float
    """Constant ordinary marginal of the row, before the unit row gradient."""
    savings: float
    """Associated saving of the row, checked against the actual floor and budget."""


@dataclass(frozen=True)
class MixedCornerCase:
    """One actual query against a two-row, three-candidate synthetic bank."""

    stock: float
    """Actual passive coordinate; the rows sit at `PASSIVE_NODES`."""
    liquid: float
    """Actual liquid state entering the budget `liquid + 1.25 - cost`."""
    ordinary: tuple[tuple[OrdinaryRow | None, ...], ...]
    """`ordinary[row][outer]`; `None` is an empty (padding) row."""
    corner: tuple[tuple[float | None, ...], ...]
    """`corner[row][outer]`: held `W` at the floor; `-inf` economic, `None` missing."""


@dataclass(frozen=True)
class Reference:
    """The reference read: a pair when covered, otherwise only the status."""

    value: float
    marginal: float
    coverage_missing: bool
    eligible_pairs: int
    """Raw value/marginal pair reads: one per active row and live candidate row."""


def utility(*, consumption: float, new_stock: float) -> float:
    """Flow utility of the canonical child declared in the finite germ fixture."""
    delta = consumption - 1.0
    return (
        math.log(consumption)
        + (delta**3 / 6.0 + delta**4 / 2.0) * math.exp(-100.0 * delta**2)
        - (new_stock - 1.0) ** 2
    )


def marginal_utility(*, consumption: float) -> float:
    """Hand derivative of `utility` in consumption."""
    delta = consumption - 1.0
    bump = math.exp(-100.0 * delta**2)
    polynomial = delta**3 / 6.0 + delta**4 / 2.0
    return 1.0 / consumption + bump * (
        (delta**2 / 2.0 + 2.0 * delta**3) - 200.0 * delta * polynomial
    )


def resources(*, liquid: float, stock: float, identity: float) -> float:
    """Budget of one outer candidate at a stock: gross net of the quadratic cost."""
    return liquid + GROSS_OFFSET - (identity - stock) ** 2 / 4.0


def floor(*, stock: float) -> float:
    """Callable floor of the canonical child."""
    return stock / 4.0


def identities(*, stock: float) -> tuple[float, ...]:
    """Canonical outer order: the keeper at the actual stock, then the nodes."""
    return (stock, *OUTER_NODES)


def row_identities(*, row: int) -> tuple[float, ...]:
    """The same candidates at a source row: the keeper holds the row's stock."""
    return (PASSIVE_NODES[row], *OUTER_NODES)


def passive_weights(*, stock: float) -> tuple[float, float]:
    """Nearest-segment weights on the two-node passive grid, signed off-grid."""
    return (1.0 - stock, stock)


def corner_value(*, case: MixedCornerCase, identity: float, held: float) -> float:
    """Actual-state corner `u(R - b) + beta W` with the fixture's unit discount."""
    consumption = resources(
        liquid=case.liquid, stock=case.stock, identity=identity
    ) - floor(stock=case.stock)
    return utility(consumption=consumption, new_stock=identity) + held


def _eligible(*, case: MixedCornerCase, outer: int) -> bool:
    identity = identities(stock=case.stock)[outer]
    budget = resources(liquid=case.liquid, stock=case.stock, identity=identity)
    return budget > floor(stock=case.stock)


def _legal(*, case: MixedCornerCase, row: int, outer: int) -> bool:
    """Legality of a row's ordinary tuple at the actual floor and budgets.

    The saving must clear the actual floor and the actual saving domain, leave
    positive consumption out of the candidate's actual budget, and the row's
    own budget for the candidate must clear the actual floor.
    """
    ordinary = case.ordinary[row][outer]
    if ordinary is None or not _eligible(case=case, outer=outer):
        return False
    actual_floor = floor(stock=case.stock)
    budget = resources(
        liquid=case.liquid,
        stock=case.stock,
        identity=identities(stock=case.stock)[outer],
    )
    row_budget = resources(
        liquid=case.liquid,
        stock=PASSIVE_NODES[row],
        identity=row_identities(row=row)[outer],
    )
    return (
        actual_floor <= ordinary.savings <= actual_floor + SAVINGS_SPAN
        and budget - ordinary.savings > 0.0
        and row_budget > actual_floor
    )


def _ordinary_winner(*, case: MixedCornerCase, row: int) -> OrdinaryRow | None:
    """The row's nodewise maximum over legal ordinary tuples, first in order."""
    winner: OrdinaryRow | None = None
    for outer in range(3):
        if not _legal(case=case, row=row, outer=outer):
            continue
        ordinary = case.ordinary[row][outer]
        assert ordinary is not None
        if winner is None or ordinary.value > winner.value:
            winner = ordinary
    return winner


@dataclass(frozen=True)
class _Channel:
    """One blended alternative: supported, and a pair when it is."""

    value: float
    marginal: float
    supported: bool


_UNSUPPORTED = _Channel(value=math.nan, marginal=math.nan, supported=False)
_MINUS_INF = _Channel(value=-math.inf, marginal=0.0, supported=True)


def _blend_ordinary(*, case: MixedCornerCase) -> _Channel:
    weights = passive_weights(stock=case.stock)
    active: list[tuple[float, OrdinaryRow]] = []
    for row, weight in enumerate(weights):
        if weight == 0.0:
            continue
        winner = _ordinary_winner(case=case, row=row)
        if winner is None:
            return _UNSUPPORTED
        active.append((weight, winner))
    if any(w < 0.0 and r.value == -math.inf for w, r in active):
        return _UNSUPPORTED
    if any(r.value == -math.inf for _, r in active):
        return _MINUS_INF
    return _Channel(
        value=sum(w * r.value for w, r in active),
        marginal=sum(w * r.marginal for w, r in active),
        supported=True,
    )


def _blend_corner(*, case: MixedCornerCase, outer: int) -> _Channel:
    """One candidate's actual corner on the blended held continuation."""
    if not _eligible(case=case, outer=outer):
        return _UNSUPPORTED
    weights = passive_weights(stock=case.stock)
    active: list[tuple[float, float]] = []
    for row, weight in enumerate(weights):
        if weight == 0.0:
            continue
        held = case.corner[row][outer]
        if held is None:
            return _UNSUPPORTED
        active.append((weight, held))
    if any(w < 0.0 and h == -math.inf for w, h in active):
        return _UNSUPPORTED
    if any(h == -math.inf for _, h in active):
        return _MINUS_INF
    identity = identities(stock=case.stock)[outer]
    blended = sum(w * h for w, h in active)
    consumption = resources(
        liquid=case.liquid, stock=case.stock, identity=identity
    ) - floor(stock=case.stock)
    return _Channel(
        value=corner_value(case=case, identity=identity, held=blended),
        marginal=marginal_utility(consumption=consumption),
        supported=True,
    )


def _eligible_pairs(*, case: MixedCornerCase) -> int:
    weights = passive_weights(stock=case.stock)
    return sum(
        1
        for row, weight in enumerate(weights)
        if weight != 0.0
        for outer in range(3)
        if case.ordinary[row][outer] is not None
    )


def reference(*, case: MixedCornerCase) -> Reference:
    """Blend each kind over the stencil, then choose at the actual query."""
    pairs = _eligible_pairs(case=case)
    if not any(_eligible(case=case, outer=outer) for outer in range(3)):
        return Reference(
            value=-math.inf, marginal=0.0, coverage_missing=False, eligible_pairs=pairs
        )
    ordinary = _blend_ordinary(case=case)
    corners = [_blend_corner(case=case, outer=outer) for outer in range(3)]
    finite = [c for c in corners if c.supported and math.isfinite(c.value)]
    best: _Channel | None = None
    for corner in finite:
        if best is None or corner.value > best.value:
            best = corner
    if ordinary.supported and (best is None or ordinary.value >= best.value):
        return Reference(
            value=ordinary.value,
            marginal=ordinary.marginal,
            coverage_missing=False,
            eligible_pairs=pairs,
        )
    if best is None:
        return Reference(
            value=math.nan,
            marginal=math.nan,
            coverage_missing=True,
            eligible_pairs=pairs,
        )
    return Reference(
        value=best.value,
        marginal=best.marginal,
        coverage_missing=False,
        eligible_pairs=pairs,
    )


def nodewise_reference(*, case: MixedCornerCase) -> float | None:
    """Value of the rejected order on nonnegative weights, `None` when uncovered.

    On every row the legal ordinary tuples and the row's corners (each priced
    at the actual budget on the row's own held continuation) compete; the row
    maxima are blended. Stated only for the Jensen comparison with `reference`.
    """
    weights = passive_weights(stock=case.stock)
    if any(w < 0.0 for w in weights):
        return None
    total = 0.0
    for row, weight in enumerate(weights):
        if weight == 0.0:
            continue
        values: list[float] = []
        for outer in range(3):
            if not _eligible(case=case, outer=outer):
                continue
            ordinary = case.ordinary[row][outer]
            if _legal(case=case, row=row, outer=outer):
                assert ordinary is not None
                values.append(ordinary.value)
            held = case.corner[row][outer]
            if held is not None:
                values.append(
                    corner_value(
                        case=case,
                        identity=identities(stock=case.stock)[outer],
                        held=held,
                    )
                )
        if not values or max(values) == -math.inf:
            return None
        total += weight * max(values)
    return total
