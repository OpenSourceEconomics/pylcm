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
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    Transition,
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
DEFAULT_WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
DEFAULT_CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)


# Law of the `working_life` edges: keep working until `final_age_alive`, then die.
WORKING_LIFE_LAW = DeterministicTransition(func=next_regime)


def graph_bound_working_life_transitions(*, last_age: UserAge | float) -> ByAge:
    """`WORKING_LIFE_LAW` with the destinations a model graph would bind, by age.

    For tests that lower regime declarations directly, without a `Model` to bind
    the laws to its edges.
    """
    return ByAge.until(
        stop_age_exclusive=last_age,
        law=_SupportedDeterministicTransition(
            func=next_regime, targets=("working_life", "dead")
        ),
        then=_SupportedDeterministicTransition(func=next_regime, targets=("dead",)),
    )


def working_life_edges(
    ages: AgeGrid,
) -> dict[str, Transition]:
    """Keep working before the second-to-last age; die from every non-final age."""
    stays = tuple(ages.exact_values[:-2])
    return {
        "working_life": Transition(
            targets={
                "dead": tuple(ages.exact_values[:-1]),
                **({"working_life": stays} if stays else {}),
            },
            law=WORKING_LIFE_LAW,
        )
    }


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
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "is_working": is_working,
        "wage": wage,
    },
)


dead = UserRegime(
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
    ages = AgeGrid(start=START_AGE, inclusive_stop=final_age_alive + 1, step="Y")
    return Model(
        regimes={
            "working_life": working_life.replace(
                states={"wealth": wealth_grid},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": consumption_grid,
                },
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        execution_config=execution_config,
        initial_nodes={18: "working_life"},
        edges=working_life_edges(ages),
    )


def get_params(
    *,
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
) -> UserParams:
    return {
        "discount_factor": discount_factor,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
        },
        "final_age_alive": START_AGE + n_periods - 2,
    }
