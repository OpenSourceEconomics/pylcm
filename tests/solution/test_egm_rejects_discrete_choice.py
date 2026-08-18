"""`EGM` rejects a discrete action instead of solving as if there were none.

The one-asset EGM step is the specialization that needs no upper envelope: with
a single continuous state and no discrete choice the candidate value
correspondence is single-valued, so inverting the Euler equation on the savings
grid solves the period exactly. A discrete action breaks that premise — it makes
the correspondence fold, which is what `DCEGM` exists for — so the regime is
outside `EGM`'s contract and is refused where the contract is stated, at `Model`
construction.
"""

import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import ConsumptionSavingsRegime, LiquidMargin, Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import ContinuousAction, DiscreteAction, FloatND, ScalarInt
from tests.solution.test_egm_solver import (
    _N_PERIODS,
    _SAVINGS_GRID,
    _WEALTH_GRID,
    RegimeId,
    feasible,
    next_wealth,
    prob_continue,
    prob_stop,
    resources,
    savings,
    terminal_utility,
)

_LAST_AGE = float(_N_PERIODS - 1)


@categorical(ordered=False)
class Effort:
    low: ScalarInt
    high: ScalarInt


def utility(
    consumption: ContinuousAction, effort: DiscreteAction, crra: float
) -> FloatND:
    """CRRA felicity net of a flow cost of effort."""
    return consumption ** (1.0 - crra) / (1.0 - crra) - 0.1 * effort


def test_a_discrete_action_is_refused_at_model_construction() -> None:
    """A regime with a discrete action and `EGM` names the action and fails."""
    saving = ConsumptionSavingsRegime(
        actions={
            "consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=50),
            "effort": DiscreteGrid(Effort),
        },
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"saving": next_wealth, "done": next_wealth}},
        constraints={"feasible": feasible},
        transition={
            "saving": MarkovTransition(prob_continue),
            "done": MarkovTransition(prob_stop),
        },
        functions={"utility": utility, "resources": resources, "savings": savings},
        active=lambda age: age < _LAST_AGE,
        solver=EGM(savings_grid=_SAVINGS_GRID),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    done = Regime(
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age >= _LAST_AGE,
        solver=GridSearch(),
    )
    with pytest.raises(ModelInitializationError, match="effort"):
        Model(
            regimes={"saving": saving, "done": done},
            ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
            regime_id_class=RegimeId,
        )
