from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinspaceGrid
from lcm.user_model import Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        FloatND,
    )


@pytest.fixture
def valid_model() -> Model:
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

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    return Regime(
        name="valid_regime",
        functions={
            "utility": utility,
            "next_wealth": next_wealth,
            "next_health": next_health,
            "borrowing_constraint": borrowing_constraint,
        },
        actions={
            "consumption": LinspaceGrid(
                start=1,
                stop=2,
                n_points=3,
            ),
        },
        states={
            "wealth": LinspaceGrid(
                start=1,
                stop=2,
                n_points=3,
            ),
            "health": LinspaceGrid(
                start=0,
                stop=1,
                n_points=3,
            ),
        },
    ).to_model(n_periods=3)


@pytest.fixture
def nan_value_model(valid_model: Model) -> Model:
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

    updated_functions = valid_model.functions.copy()
    updated_functions["utility"] = invalid_utility

    return valid_model.replace(
        functions=updated_functions,
    )


@pytest.fixture
def inf_value_model(valid_model: Model) -> Model:
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

    updated_functions = valid_model.functions.copy()
    updated_functions["utility"] = invalid_utility

    return valid_model.replace(
        functions=updated_functions,
    )


def test_solve_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model,
) -> None:
    with pytest.raises(InvalidValueFunctionError):
        nan_value_model.solve({"beta": 0.95})


def test_solve_model_with_inf_value_function_does_not_raise_error(
    inf_value_model: Model,
) -> None:
    # This should not raise an error
    inf_value_model.solve({"beta": 0.95})


def test_simulate_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model,
) -> None:
    initial_states = {
        "wealth": jnp.array([0.9, 1.0]),
        "health": jnp.array([1.0, 1.0]),
    }

    with pytest.raises(InvalidValueFunctionError):
        nan_value_model.solve_and_simulate(
            {"beta": 0.95}, initial_states=initial_states
        )


def test_simulate_model_with_inf_value_function_array_does_not_raise_error(
    inf_value_model: Model,
) -> None:
    initial_states = {
        "wealth": jnp.array([0.9, 1.0]),
        "health": jnp.array([1.0, 1.0]),
    }

    # This should not raise an error
    inf_value_model.solve_and_simulate({"beta": 0.95}, initial_states=initial_states)
