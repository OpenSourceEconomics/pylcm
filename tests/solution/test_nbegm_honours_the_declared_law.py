"""`NBEGM` solves the accounting law the regime declares, whatever its form.

The endogenous grid method needs two things from the budget constraint: where a
given level of savings lands next period, and how that landing point moves when
savings move. Both are properties of the law the modeller wrote. A solver
restricted to one functional form cannot solve a regime whose law carries a term
outside it — and a model author who declares such a term is entitled to a
solution, not to a build error naming the parameter they added.

The witness is a per-period fixed cost, which no rearrangement of a
`return x balance + income` form can express. `GridSearch` maximizes over the
declared law directly, so it has nothing to assume and serves as the oracle: the
two solvers are handed the identical `Regime` and must agree on the interior.
"""

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    utility,
)

_FIXED_COST = 0.25
_N_PERIODS = 4
_N_LIQUID = 80
_LIQUID_MAX = 20.0
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=400)

# The comparison excludes the lowest states, where the borrowing corner binds
# and the two solvers resolve the same kink on different grids.
_INTERIOR = slice(8, None)


def feasible(*, liquid: ContinuousState, consumption: ContinuousAction) -> BoolND:
    """Consumption cannot exceed the directly declared liquid resources."""
    return consumption <= liquid


def savings(*, liquid: ContinuousState, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings subtract consumption from the liquid state."""
    return liquid - consumption


def next_liquid_net_of_a_fixed_cost(
    *,
    savings: FloatND,
    return_liquid: FloatND,
    income: FloatND,
    fixed_cost: float,
) -> ContinuousState:
    """The conventional law, less a charge levied once per period."""
    return (1.0 + return_liquid) * savings + income - fixed_cost


def _model(*, variant, n_consumption=120):
    """The one-asset lifecycle whose law charges a fixed cost each period."""
    return make_alive_dead_model(
        n_periods=_N_PERIODS,
        n_liquid=_N_LIQUID,
        liquid_max=_LIQUID_MAX,
        n_consumption=n_consumption,
        alive_functions={
            "utility": utility,
            "savings": savings,
        },
        liquid_law=next_liquid_net_of_a_fixed_cost,
        alive_solver=resolve_solver(variant=variant, savings_grid=_SAVINGS_GRID),
        constraints={} if variant == "nbegm" else {"feasible": feasible},
        liquid_resources="liquid",
    )


def _params():
    budget = {
        "return_liquid": 0.03,
        "income": 1.0,
        "fixed_cost": _FIXED_COST,
    }
    return {
        "alive": {
            "utility": {"crra": 2.0},
            "koopmans_aggregator": {"discount_factor": 0.95},
            "alive": {"next_liquid": budget, "next_regime": {"final_age_alive": 3.0}},
            "dead": {"next_liquid": budget, "next_regime": {"final_age_alive": 3.0}},
        },
        "dead": {"utility": {"crra": 2.0}},
    }


@pytest.mark.parametrize("period", [0, 1, 2])
def test_a_fixed_cost_in_the_law_reaches_the_nbegm_value(period):
    """`NBEGM` and a dense `GridSearch` agree on a law carrying a fixed cost."""
    params = _params()
    nbegm = _model(variant="nbegm").solve(params=params, log_level="off").values
    brute = (
        _model(variant="brute", n_consumption=1200)
        .solve(params=params, log_level="off")
        .values
    )

    got = np.asarray(nbegm[period]["alive"])[_INTERIOR]
    expected = np.asarray(brute[period]["alive"])[_INTERIOR]

    rel = np.abs(got - expected) / np.abs(expected)
    assert np.max(rel) < 1e-2
