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
    from pathlib import Path

    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        FloatND,
    )


@pytest.fixture
def monkeypatched_log_and_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, str]:
    # Monkeypath the log directory to a temporary path
    test_log_dir = tmp_path / ".pylcm"
    monkeypatch.setattr(lcm._config, "LOG_DIRECTORY", str(test_log_dir))

    # Monkeypatch the suffix generation to return a fixed value, as it is otherwise
    # impossible to predict the suffix in tests.
    suffix = "123"
    monkeypatch.setattr(lcm.error_handling, "_generate_unique_suffix", lambda: suffix)

    return test_log_dir, suffix


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

    return Model(
        n_periods=2,
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
    )


@pytest.fixture
def invalid_value_model(valid_model: Model) -> Model:
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
        inf_term = jnp.where(
            jnp.logical_and(wealth > 1.9, health > 0.9),
            jnp.inf,
            0.0,
        )
        return jnp.log(consumption) + nan_term + inf_term

    updated_functions = valid_model.functions.copy()
    updated_functions["utility"] = invalid_utility

    return valid_model.replace(
        functions=updated_functions,
    )


def test_solve_model_with_invalid_value_function_array(
    monkeypatched_log_and_suffix: tuple[Path, str], invalid_value_model: Model
) -> None:
    test_log_dir, suffix = monkeypatched_log_and_suffix

    solve_model, _ = get_lcm_function(
        model=invalid_value_model,
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


def test_simulate_model_with_invalid_value_function_array(
    monkeypatched_log_and_suffix: tuple[Path, str], valid_model: Model
) -> None:
    test_log_dir, suffix = monkeypatched_log_and_suffix

    simulate_model, _ = get_lcm_function(
        model=valid_model,
        targets="solve_and_simulate",
    )

    # Create initial states that will lead to an invalid value function array.
    # Specifically, we will use some wealth values that are too low for positive
    # consumption in the second period. The minimal positive consumption value is 0.5,
    # which means that wealth must be at least 0.6 to allow for positive in the first
    # period. Additionally, wealth must be at least 1.0 to allow for positive
    # consumption in the second period. Because of the Bellman equation, this will lead
    # to an invalid value function array in the first period already, as the invalid
    # combinations will propagate through the periods.
    initial_states = {
        "wealth": jnp.array([0.6, 0.9, 1.0, 1.5]),
        "health": jnp.array([1.0, 1.0, 1.0, 1.0]),
    }

    # Assert that the correct error is raised
    with pytest.raises(InvalidValueFunctionError):
        simulate_model({"beta": 0.95}, initial_states=initial_states)

    # Assert that the log directory and file were created
    assert test_log_dir.exists()
    assert (test_log_dir / f"invalid_states_{suffix}.csv").exists()

    # Assert that the logged data contains the expected values
    got = pd.read_csv(test_log_dir / f"invalid_states_{suffix}.csv")
    exp = pd.DataFrame(
        {
            "wealth": [0.6, 0.9],
            "health": [1.0, 1.0],
            "__value__": [-jnp.inf, -jnp.inf],
            "__period__": [0, 0],
        }
    )
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
