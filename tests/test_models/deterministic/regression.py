"""Regression test model — 2-regime subset of the mortality model.

Extends the mortality model with an age-dependent wage function and supports
configurable grid types for testing various grid classes.
"""

import jax.numpy as jnp

from _lcm.grids import UniformContinuousGrid
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    FloatND,
    ScalarInt,
    UserAge,
    UserParams,
)
from lcm_examples.mortality import (
    LaborSupply,
    borrowing_constraint,
    is_working,
    labor_income,
    next_wealth,
)
from lcm_examples.mortality import (
    utility_working as utility,
)
from tests.test_models.graph import with_fixture_graph
from tests.test_models.schedules import until_exit


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    dead: ScalarInt


def wage(age: float) -> float | FloatND:
    return 1 + 0.1 * age


def next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.working_life,
    )


START_AGE = 18
_DEFAULT_N_PERIODS = 5
_DEFAULT_LAST_ACTIVE_AGE = START_AGE + _DEFAULT_N_PERIODS - 2
DEFAULT_WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
DEFAULT_CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)


def working_life_transitions(*, last_age: UserAge | float) -> ByAge:
    """Work until the age before `last_age`, then die."""
    return until_exit(
        last_age,
        law=_SupportedDeterministicTransition(
            func=next_regime, targets=("working_life", "dead")
        ),
        exits=("dead",),
    )


working_life = UserRegime(
    actions={
        "labor_supply": DiscreteGrid(category_class=LaborSupply),
        "consumption": DEFAULT_CONSUMPTION_GRID,
    },
    states={
        "wealth": DEFAULT_WEALTH_GRID,
    },
    state_transitions={
        "wealth": next_wealth,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
    regime_transitions=working_life_transitions(last_age=_DEFAULT_LAST_ACTIVE_AGE + 1),
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "is_working": is_working,
        "wage": wage,
    },
)


dead = UserRegime(
    regime_transitions=None,
    functions={"utility": lambda: 0.0},
)


def get_model(
    *,
    n_periods: int,
    wealth_grid: UniformContinuousGrid
    | IrregSpacedGrid
    | PiecewiseLinSpacedGrid
    | PiecewiseLogSpacedGrid = DEFAULT_WEALTH_GRID,
    consumption_grid: UniformContinuousGrid
    | IrregSpacedGrid
    | PiecewiseLinSpacedGrid
    | PiecewiseLogSpacedGrid = DEFAULT_CONSUMPTION_GRID,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> Model:
    final_age_alive = START_AGE + n_periods - 2
    return with_fixture_graph(
        regimes={
            "working_life": working_life.replace(
                regime_transitions=working_life_transitions(
                    last_age=final_age_alive + 1
                ),
                states={"wealth": wealth_grid},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": consumption_grid,
                },
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, inclusive_stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=execution_config,
        initial_nodes={18: "working_life"},
    )


def get_params(
    *,
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
) -> UserParams:
    final_age_alive = START_AGE + n_periods - 2
    return {
        "discount_factor": discount_factor,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
        },
        "final_age_alive": final_age_alive,
    }
