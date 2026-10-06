"""`EGM` rejects a discrete action instead of solving as if there were none.

The one-asset EGM step is the specialization that needs no upper envelope: with
a single continuous state and no discrete choice the candidate value
correspondence is single-valued, so inverting the Euler equation on the savings
grid solves the period exactly. A discrete action breaks that premise — it makes
the correspondence fold, which is what `DCEGM` exists for — so the regime is
outside `EGM`'s contract and is refused where the contract is stated, at `Model`
construction.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import ContinuousAction, DiscreteAction, FloatND, ScalarInt
from tests.solution.test_egm_solver import (
    _N_PERIODS,
    _SAVINGS_GRID,
    _WEALTH_GRID,
    RegimeId,
    feasible,
    next_wealth,
    savings,
    terminal_utility,
)


@categorical(ordered=False)
class Effort:
    low: ScalarInt
    high: ScalarInt


def utility(
    *, consumption: ContinuousAction, effort: DiscreteAction, crra: float
) -> FloatND:
    """CRRA felicity net of a flow cost of effort."""
    return consumption ** (1.0 - crra) / (1.0 - crra) - 0.1 * effort


def prob_continue(*, age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 < last_age, 1.0, 0.0)


def prob_stop(*, age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 >= last_age, 1.0, 0.0)


def test_a_discrete_action_is_refused_at_model_construction() -> None:
    """A regime with a discrete action and `EGM` names the action and fails."""
    saving = ConsumptionSavingsRegime(
        actions={
            "consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=50),
            "effort": DiscreteGrid(category_class=Effort),
        },
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"saving": next_wealth, "done": next_wealth}},
        constraints={"feasible": feasible},
        functions={"utility": utility, "savings": savings},
        solver=EGM(savings_grid=_SAVINGS_GRID),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    done = Regime(
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        solver=GridSearch(),
    )
    with pytest.raises(ModelInitializationError, match="effort"):
        Model(
            regimes={"saving": saving, "done": done},
            ages=AgeGrid(start=0, inclusive_stop=_N_PERIODS - 1, step="Y"),
            edges={
                "saving": Transition(
                    targets={
                        "saving": AgeRange(exclusive_stop=_N_PERIODS - 2),
                        "done": AgeRange(exclusive_stop=_N_PERIODS - 1),
                    },
                    law={
                        "saving": StochasticTransition(func=prob_continue),
                        "done": StochasticTransition(func=prob_stop),
                    },
                )
            },
            regime_id_class=RegimeId,
            initial_nodes={0: "saving"},
        )
