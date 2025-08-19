from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

import lcm._config
from lcm.entry_point import get_lcm_function
from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinspaceGrid
from lcm.user_model import Model

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
    )


def test_solve_model_with_invalid_value_function_array(monkeypatch, tmp_path):
    test_log_dir = tmp_path / ".pylcm"
    monkeypatch.setattr(lcm._config, "LOG_DIRECTORY", str(test_log_dir), raising=False)

    def utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,
        health: ContinuousState,
    ) -> FloatND:
        bad_term = jnp.where(jnp.logical_and(wealth < 1.5, health >= 1.5), 0, jnp.nan)
        return jnp.log(consumption) + bad_term

    def next_wealth(
        wealth: ContinuousState,
        consumption: ContinuousAction,
    ) -> ContinuousState:
        return wealth - consumption

    def next_health(
        health: ContinuousState,
    ) -> ContinuousState:
        return health

    def borrowing_constraint(
        consumption: ContinuousAction | DiscreteAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    model = Model(
        n_periods=2,
        functions={
            "utility": utility,
            "next_wealth": next_wealth,
            "next_health": next_health,
            "borrowing_constraint": borrowing_constraint,
        },
        actions={
            "consumption": LinspaceGrid(
                start=0,
                stop=1,
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
    )
    solve_model, _ = get_lcm_function(
        model=model,
        targets="solve",
    )

    with pytest.raises(InvalidValueFunctionError):
        solve_model({"beta": 0.95})

    assert test_log_dir.exists()
    assert (test_log_dir / "invalid_nan_states.csv").exists()
