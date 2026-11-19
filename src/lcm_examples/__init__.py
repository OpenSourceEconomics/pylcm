"""Canonical example models for PyLCM.

Importable models for documentation, testing, and experimentation.
"""

from collections.abc import Mapping

from lcm import Regime
from lcm.solvers import GridSearch
from lcm.typing import RegimeName


def with_lottery_contraction(
    *, regimes: Mapping[RegimeName, Regime], lottery_contraction: bool
) -> dict[RegimeName, Regime]:
    """Return `regimes`, contracting lotteries in every grid-search regime if asked."""
    return {
        name: regime.replace(solver=GridSearch(lottery_contraction=True))
        if lottery_contraction and isinstance(regime.solver, GridSearch)
        else regime
        for name, regime in regimes.items()
    }
