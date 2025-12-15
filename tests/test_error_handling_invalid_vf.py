from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import Model, Regime
from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinspaceGrid

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        FloatND,
        ParamsDict,
        ScalarInt,
    )


@pytest.fixture
def n_periods() -> int:
    return 2


@pytest.fixture
def regimes_and_id_cls() -> tuple[dict[str, Regime], type]:
    @dataclass
    class RegimeID:
        non_terminal: int = 0
        terminal: int = 1

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
        return jnp.where(
            transition_into_terminal, RegimeID.terminal, RegimeID.non_terminal
        )

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    non_terminal = Regime(
        name="non_terminal",
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
        utility=utility,
        constraints={
            "borrowing_constraint": borrowing_constraint,
        },
        transitions={
            "next_wealth": next_wealth,
            "next_health": next_health,
            "next_regime": next_regime,
        },
    )

    terminal = Regime(
        name="terminal",
        terminal=True,
        states={
            "wealth": LinspaceGrid(start=1, stop=2, n_points=3),
        },
        utility=lambda wealth: jnp.array([0.0]),  # noqa: ARG005
    )

    return {"non_terminal": non_terminal, "terminal": terminal}, RegimeID


@pytest.fixture
def nan_value_model(
    regimes_and_id_cls: tuple[dict[str, Regime], type], n_periods: int
) -> Model:
    regimes, regime_id_cls = regimes_and_id_cls

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

    invalid_regime = regimes["non_terminal"].replace(utility=invalid_utility)
    return Model(
        regimes=[invalid_regime, regimes["terminal"]],
        n_periods=n_periods,
        regime_id_cls=regime_id_cls,
    )


@pytest.fixture
def inf_value_model(
    regimes_and_id_cls: tuple[dict[str, Regime], type], n_periods: int
) -> Model:
    regimes, regime_id_cls = regimes_and_id_cls

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

    inf_model = regimes["non_terminal"].replace(utility=invalid_utility)
    return Model(
        regimes=[inf_model, regimes["terminal"]],
        n_periods=n_periods,
        regime_id_cls=regime_id_cls,
    )


@pytest.fixture
def params(n_periods: int) -> ParamsDict:
    return {
        "non_terminal": {"beta": 0.95, "next_regime": {"n_periods": n_periods}},
        "terminal": {},
    }


def test_solve_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model, params: ParamsDict
) -> None:
    with pytest.raises(InvalidValueFunctionError):
        nan_value_model.solve(params)


def test_solve_model_with_inf_value_function_does_not_raise_error(
    inf_value_model: Model, params: ParamsDict
) -> None:
    # This should not raise an error
    inf_value_model.solve(params)


def test_simulate_model_with_nan_value_function_array_raises_error(
    nan_value_model: Model, params: ParamsDict
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
    inf_value_model: Model, params: ParamsDict
) -> None:
    initial_states = {
        "wealth": jnp.array([0.9, 1.0]),
        "health": jnp.array([1.0, 1.0]),
    }

    # This should not raise an error
    inf_value_model.solve_and_simulate(
        params, initial_states=initial_states, initial_regimes=["non_terminal"] * 2
    )
