from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytest

import lcm._config
import lcm.error_handling
from lcm.entry_point import get_lcm_function
from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinspaceGrid
from lcm.user_model import Model

if TYPE_CHECKING:
    from lcm.typing import (
        ContinuousAction,
        ContinuousState,
        FloatND,
    )


def test_solve_model_with_invalid_value_function_array(monkeypatch, tmp_path):
    # Monkeypatch log directory and random filename suffix
    # ----------------------------------------------------------------------------------
    test_log_dir = tmp_path / ".pylcm"
    monkeypatch.setattr(lcm._config, "LOG_DIRECTORY", str(test_log_dir), raising=False)
    suffix = "123"
    monkeypatch.setattr(lcm.error_handling, "_generate_unique_suffix", lambda: suffix)

    # Testing
    # ----------------------------------------------------------------------------------
    def utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,
        health: ContinuousState,
    ) -> FloatND:
        nan_term = jnp.where(
            jnp.logical_and(wealth < 1.1, health < 0.1),
            jnp.nan,
            0.0,
        )
        inf_term = jnp.where(
            jnp.logical_and(wealth > 1.9, health > 0.9),
            jnp.inf,
            0.0,
        )
        return jnp.log(consumption) + nan_term + inf_term

    def next_wealth(
        wealth: ContinuousState,
        consumption: ContinuousAction,
    ) -> ContinuousState:
        return wealth - consumption

    def next_health(health: ContinuousState) -> ContinuousState:
        return health

    model = Model(
        n_periods=2,
        functions={
            "utility": utility,
            "next_wealth": next_wealth,
            "next_health": next_health,
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
    )
    solve_model, _ = get_lcm_function(
        model=model,
        targets="solve",
    )

    # Assert that the correct error is raised
    with pytest.raises(InvalidValueFunctionError):
        solve_model({"beta": 0.95})

    # Assert that the log directory and file were created
    assert test_log_dir.exists()
    assert (test_log_dir / f"invalid_states_{suffix}.csv").exists()

    # Assert that the logged data contains the expected values
    got = pd.read_csv(test_log_dir / f"invalid_states_{suffix}.csv")
    exp = pd.DataFrame(
        {
            "wealth": [1.0, 2.0],
            "health": [0.0, 1.0],
            "__value__": [jnp.nan, jnp.inf],
            "__period__": [1, 1],
        }
    )
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
