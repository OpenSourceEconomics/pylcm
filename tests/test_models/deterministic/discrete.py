"""Example specifications of fully discrete deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic action models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643). See module `tests.test_models.deterministic` for the
continuous version.

"""

import functools

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    DiscreteGrid,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserAge,
    UserParams,
)
from tests.test_models.deterministic.regression import (
    LaborSupply,
    is_working,
    labor_income,
    next_wealth,
    utility,
)


@categorical(ordered=True)
class DiscreteConsumption:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class DiscreteWealth:
    low: ScalarInt
    medium: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    dead: ScalarInt


def utility_discrete(
    *,
    consumption: DiscreteAction,
    is_working: BoolND,
    disutility_of_work: float,
) -> FloatND:
    # In the discrete model, consumption is defined as "low" or "high". This can be
    # translated to the levels 1 and 2.
    consumption_level = 1 + (consumption == DiscreteConsumption.high)
    return utility(
        consumption=consumption_level,
        is_working=is_working,
        disutility_of_work=disutility_of_work,
    )


def next_wealth_discrete(
    *,
    wealth: DiscreteState,
    consumption: DiscreteAction,
    labor_income: FloatND,
    interest_rate: float,
) -> DiscreteState:
    # For discrete state variables, we need to assure that the next state is also a
    # valid state, i.e., it is a member of the discrete grid.
    continuous = next_wealth(
        wealth=wealth,
        consumption=consumption,
        labor_income=labor_income,
        interest_rate=interest_rate,
    )
    return jnp.clip(
        jnp.rint(continuous), DiscreteWealth.low, DiscreteWealth.high
    ).astype(jnp.int32)


def next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.working_life,
    )


def borrowing_constraint(
    *, consumption: DiscreteAction, wealth: DiscreteState
) -> BoolND:
    return consumption <= wealth


_DEFAULT_N_PERIODS = 4
_DEFAULT_LAST_ACTIVE_AGE = 50 + (_DEFAULT_N_PERIODS - 2) * 10


def working_life_transitions(*, last_age: UserAge | float) -> ByAge:
    """Work until the age before `last_age`, then die."""
    return ByAge.until(
        stop_age_exclusive=last_age,
        law=DeterministicTransition(func=next_regime),
        then=DeterministicTransition(func=next_regime),
    )


def working_life_edges(ages: AgeGrid) -> dict[str, dict[str, tuple[UserAge, ...]]]:
    """Keep working before the second-to-last age; die from every non-final age."""
    stays = tuple(ages.exact_values[:-2])
    return {
        "working_life": {
            **({"working_life": stays} if stays else {}),
            "dead": tuple(ages.exact_values[:-1]),
        }
    }


working_life = UserRegime(
    actions={
        "labor_supply": DiscreteGrid(category_class=LaborSupply),
        "consumption": DiscreteGrid(category_class=DiscreteConsumption),
    },
    states={
        "wealth": DiscreteGrid(category_class=DiscreteWealth),
    },
    state_transitions={
        "wealth": next_wealth_discrete,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
    },
    regime_transitions=working_life_transitions(last_age=_DEFAULT_LAST_ACTIVE_AGE + 10),
    functions={
        "utility": utility_discrete,
        "labor_income": labor_income,
        "is_working": is_working,
    },
)


dead = UserRegime(
    regime_transitions=None,
    functions={"utility": lambda: 0.0},
)


@functools.cache
def get_model(n_periods: int) -> Model:
    ages = AgeGrid(start=50, inclusive_stop=50 + (n_periods - 1) * 10, step="10Y")
    final_age_alive = 50 + (n_periods - 2) * 10
    return Model(
        regimes={
            "working_life": working_life.replace(
                regime_transitions=working_life_transitions(
                    last_age=final_age_alive + 10
                )
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        edges=working_life_edges(ages),
    )


def get_params(
    *,
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
    wage: float = 10.0,
) -> UserParams:
    final_age_alive = 50 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "next_wealth": {"interest_rate": interest_rate},
            "labor_income": {"wage": wage},
        },
        "final_age_alive": final_age_alive,
    }
