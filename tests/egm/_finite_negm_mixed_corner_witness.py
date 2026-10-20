"""Rational witness for the finite NEGM mixed ordinary/corner read.

The public model is the actual-corner route with a quadratic child utility
`u(c) = c - c^2 / 32`, inverse marginal `16 (1 - m)`, discount 1/2, terminal
slope 1/8 and a callable zero floor. The quadratic makes every constrained
Hermite source bracket exact and its separately interpolated ordinary marginal
exact, so the public read can be compared with exact fractions. The fixture
confines all child consumptions to `(0, 15]`, where `u' >= 1/16`.

The represented object blends each kind over the passive stencil and then
chooses at the actual query: the ordinary channel is the blend of the source
rows' nodewise ordinary maxima, each actual-state corner is priced on the
blended floor continuation, and the ordinary channel wins an exact tie.
`nodewise_alternative` states the rejected order (choose on every source row,
then blend); it credits a different choice at each passive node, which no
single action can realise, and by Jensen it is never below the blended read.
`exact_optimum` is the brute-force child optimum the blended read must not
exceed. No project imports; no interpolant.
"""

from dataclasses import dataclass
from fractions import Fraction

F = Fraction


@dataclass(frozen=True)
class Pair:
    value: F
    marginal: F


def utility(consumption: F) -> F:
    return consumption - consumption * consumption / 32


def marginal(consumption: F) -> F:
    return 1 - consumption / 16


def source_ordinary(*, source_stock: F, outer: F | None) -> Pair:
    """Native common-coordinate source row at child liquid 1 and saving 0."""
    identity = source_stock if outer is None else outer
    consumption = 2 + 8 * source_stock - (identity - source_stock) ** 2 / 4
    assert 0 < consumption <= 15
    return Pair(utility(consumption), marginal(consumption))


def actual_corner(*, stock: F, outer: F | None) -> Pair:
    """Actual floor 0, actual keeper/cost, held source-floor W 0, unit chain."""
    identity = stock if outer is None else outer
    consumption = 2 + 8 * stock - (identity - stock) ** 2 / 4
    assert 0 < consumption <= 15
    return Pair(utility(consumption), marginal(consumption))


def canonical_point_max(pairs: tuple[Pair, ...]) -> Pair:
    """Already ordered keeper then outer nodes; the first exact maximum owns M."""
    return max(pairs, key=lambda pair: pair.value)


NODES = (F(0), F(1))
IDENTITIES = (None, F(0), F(1))


def _blend(*, stock: F, pairs: tuple[Pair, Pair]) -> Pair:
    assert 0 <= stock <= 1
    weights = (1 - stock, stock)
    return Pair(
        sum((w * p.value for w, p in zip(weights, pairs, strict=True)), F(0)),
        sum((w * p.marginal for w, p in zip(weights, pairs, strict=True)), F(0)),
    )


def blend_then_choose(*, stock: F) -> Pair:
    """Blend the ordinary row maxima, then let the best actual corner compete.

    The held floor continuation is zero on every source row, so each corner's
    blended continuation is zero and the corner is its actual-state value. The
    ordinary channel wins an exact tie.
    """
    ordinary = _blend(
        stock=stock,
        pairs=(
            canonical_point_max(
                tuple(
                    source_ordinary(source_stock=NODES[0], outer=o) for o in IDENTITIES
                )
            ),
            canonical_point_max(
                tuple(
                    source_ordinary(source_stock=NODES[1], outer=o) for o in IDENTITIES
                )
            ),
        ),
    )
    best = canonical_point_max(
        tuple(actual_corner(stock=stock, outer=o) for o in IDENTITIES)
    )
    return ordinary if ordinary.value >= best.value else best


def nodewise_alternative(*, stock: F) -> Pair:
    """Choose across ordinary and corner on every source row, then blend."""
    selected = []
    for node in NODES:
        alternatives = tuple(
            pair
            for outer in IDENTITIES
            for pair in (
                source_ordinary(source_stock=node, outer=outer),
                actual_corner(stock=stock, outer=outer),
            )
        )
        selected.append(canonical_point_max(alternatives))
    return _blend(stock=stock, pairs=(selected[0], selected[1]))


def exact_optimum(*, stock: F) -> Pair:
    """Exact child optimum at liquid 1: `u(c) + (1/2)(R - c)/8` over `0 < c <= R`.

    The unconstrained Euler consumption is 15, so a candidate with `R < 15` sits
    at its corner `c = R`. Every outer candidate is scored at its own cost and
    the best one is returned with the marginal of its consumption.
    """
    best: Pair | None = None
    for outer in IDENTITIES:
        identity = stock if outer is None else outer
        budget = 2 + 8 * stock - (identity - stock) ** 2 / 4
        consumption = min(budget, F(15))
        value = utility(consumption) + F(1, 2) * (budget - consumption) / 8
        candidate = Pair(value, marginal(consumption))
        if best is None or candidate.value > best.value:
            best = candidate
    assert best is not None
    return best


def main() -> None:
    assert F(1, 2) * F(1, 8) == F(1, 16)
    assert 16 * (1 - F(1, 16)) == 15  # genuine Euler consumption
    assert source_ordinary(source_stock=F(0), outer=None) == Pair(F(15, 8), F(7, 8))
    assert source_ordinary(source_stock=F(1), outer=None) == Pair(F(55, 8), F(3, 8))
    assert actual_corner(stock=F(1, 2), outer=None) == Pair(F(39, 8), F(5, 8))
    assert blend_then_choose(stock=F(1, 2)) == Pair(F(39, 8), F(5, 8))
    assert nodewise_alternative(stock=F(1, 2)) == Pair(F(47, 8), F(1, 2))
    assert exact_optimum(stock=F(1, 2)) == Pair(F(39, 8), F(5, 8))
    for stock in (F(1, 4), F(1, 2), F(3, 4)):
        blended = blend_then_choose(stock=stock)
        nodewise = nodewise_alternative(stock=stock)
        exact = exact_optimum(stock=stock)
        assert nodewise.value > blended.value
        assert blended.value <= exact.value
        assert nodewise.value > exact.value
    for stock in (F(0), F(1)):
        assert blend_then_choose(stock=stock) == nodewise_alternative(stock=stock)
    # The ordinary channel owns an exact tie with a corner.
    assert canonical_point_max((Pair(F(0), F(17)), Pair(F(0), F(1)))).marginal == 17


if __name__ == "__main__":
    main()
