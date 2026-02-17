import jax.numpy as jnp
import pytest

from lcm import Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinSpacedGrid
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)


@pytest.fixture
def n_periods() -> int:
    return 2


@categorical
class RegimeId:
    non_terminal: int
    terminal: int


@pytest.fixture
def regimes_and_ages(n_periods: int) -> tuple[dict[str, Regime], AgeGrid]:
    def utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,  # noqa: ARG001
        health: ContinuousState,  # noqa: ARG001
    ) -> FloatND:
        return jnp.log(consumption)

    def next_wealth(
        wealth: ContinuousState,
        consumption: ContinuousAction,
    ) -> ContinuousState:
        return wealth - consumption

    def next_health(health: ContinuousState) -> ContinuousState:
        return health

    def next_regime(period: int, n_periods: int) -> ScalarInt:
        transition_into_terminal = period == (n_periods - 2)
        # 0 = non_terminal, 1 = terminal (based on dict order)
        return jnp.where(transition_into_terminal, 1, 0)

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    non_terminal = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=1,
                stop=2,
                n_points=3,
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=2,
                n_points=3,
                transition=next_wealth,
            ),
            "health": LinSpacedGrid(
                start=0,
                stop=1,
                n_points=3,
                transition=next_health,
            ),
        },
        functions={"utility": utility},
        constraints={
            "borrowing_constraint": borrowing_constraint,
        },
        transition=next_regime,
        active=lambda age, n=n_periods: age < n - 1,
    )

    terminal = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age, n=n_periods: age >= n - 1,
    )

    ages = AgeGrid(start=0, stop=n_periods, step="Y")

    return {"non_terminal": non_terminal, "terminal": terminal}, ages


@pytest.fixture
def nan_value_model(
    regimes_and_ages: tuple[dict[str, Regime], AgeGrid],
) -> Model:
    regimes, ages = regimes_and_ages

    def invalid_utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,
        health: ContinuousState,
    ) -> FloatND:
        nan_term = jnp.where(
            jnp.logical_and(wealth < 1.1, health < 0.1),
            jnp.nan,
            0.0,
        )
        return jnp.log(consumption) + nan_term

    invalid_regime = regimes["non_terminal"].replace(
        functions={**regimes["non_terminal"].functions, "utility": invalid_utility},
    )
    return Model(
        regimes={
            "non_terminal": invalid_regime,
            "terminal": regimes["terminal"],
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


@pytest.fixture
def inf_value_model(
    regimes_and_ages: tuple[dict[str, Regime], AgeGrid],
) -> Model:
    regimes, ages = regimes_and_ages

    def invalid_utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,
        health: ContinuousState,
    ) -> FloatND:
        inf_term = jnp.where(
            jnp.logical_and(wealth > 1.9, health > 0.9),
            jnp.inf,
            0.0,
        )
        return jnp.log(consumption) + inf_term

    inf_regime = regimes["non_terminal"].replace(
        functions={**regimes["non_terminal"].functions, "utility": invalid_utility},
    )
    return Model(
        regimes={
            "non_terminal": inf_regime,
            "terminal": regimes["terminal"],
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


@pytest.fixture
def params(n_periods: int) -> UserParams:
    return {
        "discount_factor": 0.95,
        "non_terminal": {
            "next_regime": {"n_periods": n_periods},
        },
        "terminal": {},
    }


def test_solve_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model, params: UserParams
) -> None:
    with pytest.raises(InvalidValueFunctionError):
        nan_value_model.solve(params)


def test_solve_model_with_inf_value_function_does_not_raise_error(
    inf_value_model: Model, params: UserParams
) -> None:
    # This should not raise an error
    inf_value_model.solve(params)


def test_simulate_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model, params: UserParams
) -> None:
    initial_states = {
        "wealth": jnp.array([0.9, 1.0]),
        "health": jnp.array([1.0, 1.0]),
    }

    with pytest.raises(InvalidValueFunctionError):
        nan_value_model.solve_and_simulate(
            params, initial_states=initial_states, initial_regimes=["non_terminal"] * 2
        )


def test_simulate_model_with_inf_value_function_array_does_not_raise_error(
    inf_value_model: Model, params: UserParams
) -> None:
    initial_states = {
        "wealth": jnp.array([0.9, 1.0]),
        "health": jnp.array([1.0, 1.0]),
    }

    # This should not raise an error
    inf_value_model.solve_and_simulate(
        params, initial_states=initial_states, initial_regimes=["non_terminal"] * 2
    )
