"""A collective regime may carry a state whose solve leg is an imputation.

`Phased(solve=imputation, simulate=Grid)` is a derived function while the model
solves and a genuine state while it simulates. Keeping a collective regime's
folded shocks as stored node axes passes such a state through unchanged.
"""

from _lcm.regime_building.finalize import _unfold_collective_states
from lcm import LinSpacedGrid, Phased
from lcm.typing import ContinuousState

_YEARS = LinSpacedGrid(start=0.0, stop=20.0, n_points=5)


def _impute_years(wealth: ContinuousState) -> ContinuousState:
    return 5.0 * wealth


def test_a_carried_state_with_an_imputed_solve_leg_keeps_its_imputation() -> None:
    """The solve leg of a carried state in a collective regime is the imputation."""
    states = {"years": Phased(solve=_impute_years, simulate=_YEARS)}
    unfolded = _unfold_collective_states(states=states, collective=True)
    years = unfolded["years"]
    assert isinstance(years, Phased)
    assert years.solve is _impute_years
