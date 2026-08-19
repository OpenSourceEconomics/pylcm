"""`EGM` solves the accounting law the regime declares, whatever its form.

The endogenous grid method needs two things from the budget constraint: where a
given level of savings lands next period, and how that landing point moves when
savings move. Both are properties of the law the modeller wrote. A solver that
assumes a particular functional form instead — a gross return on the balance
plus an additive income, say — silently ignores every term outside that form,
and publishes a policy for a model the user did not write.

The witness here is a per-period fixed cost, which no rearrangement of a
`return x balance + income` form can express. `GridSearch` maximizes over the
declared law directly, so it has nothing to assume and serves as the oracle: the
two solvers are handed the identical `Regime` and must agree.
"""

import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model
from lcm.regime import ConsumptionSavingsRegime, LiquidMargin, Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import ContinuousState, FloatND
from tests.solution.test_egm_solver import (
    _CRRA,
    _DISCOUNT_FACTOR,
    _N_PERIODS,
    _RETURN,
    _SAVINGS_GRID,
    _UNCONSTRAINED,
    _WEALTH_GRID,
    RegimeId,
    feasible,
    prob_continue,
    prob_stop,
    resources,
    savings,
    terminal_utility,
    utility,
)

_FIXED_COST = 0.5


def next_wealth_net_of_a_fixed_cost(
    savings: FloatND,
    return_liquid: float,
    retirement_income: float,
    fixed_cost: float,
) -> ContinuousState:
    """The conventional law, less a charge levied once per period."""
    return (1.0 + return_liquid) * savings + retirement_income - fixed_cost


def _model(*, solver, n_consumption=200, law=next_wealth_net_of_a_fixed_cost):
    """The closed-form consumption--saving lifecycle, plus a fixed cost."""
    last_age = float(_N_PERIODS - 1)
    law = {"saving": law, "done": law}
    regime_type = ConsumptionSavingsRegime if isinstance(solver, EGM) else Regime
    saving = regime_type(
        actions={
            "consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=n_consumption)
        },
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": law},
        constraints={} if isinstance(solver, EGM) else {"feasible": feasible},
        transition={
            "saving": MarkovTransition(prob_continue),
            "done": MarkovTransition(prob_stop),
        },
        functions={"utility": utility, "resources": resources, "savings": savings},
        active=lambda age, la=last_age: age < la,
        solver=solver,
        **(
            {
                "liquid": LiquidMargin(
                    state="wealth",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                )
            }
            if isinstance(solver, EGM)
            else {}
        ),
    )
    done = Regime(
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age, la=last_age: age >= la,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _params():
    law = {
        "return_liquid": _RETURN,
        "retirement_income": 0.0,
        "fixed_cost": _FIXED_COST,
    }
    return {
        "saving": {
            "utility": {"crra": _CRRA},
            "koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR},
            "saving": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
            "done": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
        },
        "done": {"utility": {"crra": _CRRA}},
    }


@pytest.mark.parametrize("period", [0, 1, 2])
def test_a_fixed_cost_in_the_law_reaches_the_egm_value(period):
    """`EGM` and a dense `GridSearch` agree on a law carrying a fixed cost."""
    params = _params()
    egm = _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=params, log_level="debug"
    )
    brute = _model(solver=GridSearch(), n_consumption=1200).solve(
        params=params, log_level="debug"
    )

    got = np.asarray(egm[period]["saving"])[_UNCONSTRAINED]
    expected = np.asarray(brute[period]["saving"])[_UNCONSTRAINED]

    rel = np.abs(got - expected) / np.abs(expected)
    assert np.max(rel) < 1e-2
