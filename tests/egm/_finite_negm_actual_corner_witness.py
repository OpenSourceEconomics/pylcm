"""Independent scalar witness for the represented actual-state NEGM corner.

This is a mathematical reference, not a solver or a copied interpolation rule.
The source-floor continuation is held at W(0)=0 by the adopted representation.
No project imports are needed.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Corner:
    savings: float
    consumption: float
    value: float
    marginal: float


def actual_corner(
    *, liquid: float = 1.0, stock: float = 0.5, resource_slope: float = 1.0
) -> Corner:
    """Apply the actual budget to the actual floor and held source W_B=0."""
    floor = 4.0 * stock * (1.0 - stock)
    resources = liquid + 1.0 + 8.0 * stock
    consumption = resources - floor
    if not (math.isfinite(resources) and math.isfinite(floor) and consumption > 0):
        raise ValueError(
            "This finite corner fixture requires a positive actual budget."
        )
    return Corner(
        floor, consumption, math.log(consumption), resource_slope / consumption
    )


def node_budget_substitution(
    *, liquid: float = 1.0, stock: float = 0.5, resource_slope: float = 1.0
) -> tuple[float, float]:
    """State the independently derived value of the disputed nodal substitution."""
    floor = 4.0 * stock * (1.0 - stock)
    c0 = liquid + 1.0 - floor
    c1 = liquid + 9.0 - floor
    if min(c0, c1) <= 0:
        raise ValueError("This comparison requires two positive node budgets.")
    return (
        (1.0 - stock) * math.log(c0) + stock * math.log(c1),
        resource_slope * ((1.0 - stock) / c0 + stock / c1),
    )


def source_production_facts() -> None:
    """Literal Euler and budget facts; no model constructor is claimed here."""
    beta = 0.5
    terminal_slope = 1.0 / 8.0
    euler_consumption = 1.0 / (beta * terminal_slope)
    assert euler_consumption == 16.0
    for stock in (0.0, 1.0):
        assert 4.0 * stock * (1.0 - stock) == 0.0
        for liquid in (0.0, 1.0, 2.0):
            resources = liquid + 1.0 + 8.0 * stock
            assert 0.0 < resources < euler_consumption
        # W is the terminal value of next liquid a.  It is sampled at source a=0.
        assert 0.0 / 8.0 == 0.0


def main() -> None:
    source_production_facts()
    expected = actual_corner()
    assert expected.savings == 1.0
    assert expected.consumption == 5.0
    assert expected.value == math.log(5.0)
    assert expected.marginal == 1.0 / 5.0
    disputed = node_budget_substitution()
    assert math.isclose(disputed[0], math.log(3.0), rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(disputed[1], 5.0 / 9.0, rel_tol=0.0, abs_tol=1e-15)
    assert disputed[0] < expected.value
    assert disputed[1] > expected.marginal
    for stock in (0.25, 0.5, 0.75):
        expected = actual_corner(stock=stock)
        changed_value, changed_marginal = node_budget_substitution(stock=stock)
        assert changed_value < expected.value
        assert changed_marginal > expected.marginal
        assert expected.consumption + expected.savings == 2.0 + 8.0 * stock
    for stock in (0.0, 1.0):
        expected = actual_corner(stock=stock)
        changed = node_budget_substitution(stock=stock)
        assert changed == (expected.value, expected.marginal)


if __name__ == "__main__":
    main()
