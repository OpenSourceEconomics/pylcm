from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from lcm.entry_point import get_lcm_function
from lcm.input_processing import process_model
from lcm.logging import get_logger
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.simulation.simulate import (
    _lookup_optimal_continuous_actions,
    _lookup_values_from_indices,
    simulate,
)
from lcm.state_action_space import create_state_action_space, create_state_space_info
from tests.test_models import (
    get_model_config,
    get_params,
)

if TYPE_CHECKING:
    import pandas as pd


# ======================================================================================
# Test simulate using raw inputs
# ======================================================================================


@pytest.fixture
def simulate_inputs():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=1)
    actions = model_config.actions
    actions["consumption"] = actions["consumption"].replace(stop=100)  # type: ignore[attr-defined]
    model_config = model_config.replace(actions=actions)
    model = process_model(model_config)

    state_space_info = create_state_space_info(
        model=model,
        is_last_period=False,
    )
    state_action_space = create_state_action_space(
        model=model,
        is_last_period=False,
    )
    argmax_and_max_Q_over_a_functions = []
    for period in range(model.n_periods):
        Q_and_F = get_Q_and_F(
            model=model,
            next_state_space_info=state_space_info,
            period=period,
        )
        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=(*state_action_space.discrete_actions, "consumption"),
        )
        argmax_and_max_Q_over_a_functions.append(argmax_and_max_Q_over_a)

    return {
        "argmax_and_max_Q_over_a_functions": argmax_and_max_Q_over_a_functions,
        "model": model,
    }


def test_simulate_using_raw_inputs(simulate_inputs):
    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }

    got = simulate(
        params=params,
        V_arr_dict={0: jnp.empty(0)},
        initial_states={"wealth": jnp.array([1.0, 50.400803])},
        logger=get_logger(debug_mode=False),
        **simulate_inputs,
    )

    assert_array_equal(got.loc[0, :]["retirement"], 1)
    assert_array_almost_equal(got.loc[0, :]["consumption"], jnp.array([1.0, 50.400803]))


# ======================================================================================
# Test simulate using get_lcm_function
# ======================================================================================


@pytest.fixture
def iskhakov_et_al_2017_stripped_down_model_solution():
    def _model_solution(n_periods):
        model_config = get_model_config(
            "iskhakov_et_al_2017_stripped_down",
            n_periods=n_periods,
        )
        updated_functions = {
            # remove dependency on age, so that wage becomes a parameter
            name: func
            for name, func in model_config.functions.items()
            if name not in ["age", "wage"]
        }
        model_config = model_config.replace(functions=updated_functions)
        solve_model, _ = get_lcm_function(model_config, targets="solve")

        params = get_params()
        V_arr_dict = solve_model(params=params)
        return V_arr_dict, params, model_config

    return _model_solution


def test_simulate_using_get_lcm_function(
    iskhakov_et_al_2017_stripped_down_model_solution,
):
    n_periods = 3
    V_arr_dict, params, model = iskhakov_et_al_2017_stripped_down_model_solution(
        n_periods=n_periods,
    )

    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    res: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params,
        V_arr_dict=V_arr_dict,
        initial_states={
            "wealth": jnp.array([20.0, 150, 250, 320]),
        },
        additional_targets=["utility", "borrowing_constraint"],
    )

    assert {
        "_period",
        "value",
        "retirement",
        "consumption",
        "wealth",
        "utility",
        "borrowing_constraint",
    } == set(res.columns)

    # assert that everyone retires in the last period
    last_period_index = n_periods - 1
    assert_array_equal(res.loc[last_period_index, :]["retirement"], 1)

    for period in range(n_periods):
        # assert that higher wealth leads to higher consumption in each period
        assert (res.loc[period]["consumption"].diff()[1:] >= 0).all()  # type: ignore[operator]

        # assert that higher wealth leads to higher value function in each period
        assert (res.loc[period]["value"].diff()[1:] >= 0).all()  # type: ignore[operator]


def test_simulate_with_only_discrete_actions():
    model = get_model_config("iskhakov_et_al_2017_discrete", n_periods=2)
    params = get_params(wage=1.5, beta=1, interest_rate=0)

    simulate_model, _ = get_lcm_function(model=model, targets="solve_and_simulate")

    res: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params,
        initial_states={"wealth": jnp.array([0, 4])},
        additional_targets=["labor_income", "working"],
    )

    assert_array_equal(res["retirement"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["consumption"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["wealth"], jnp.array([0, 4, 2, 2]))


# ======================================================================================
# Testing effects of parameters
# ======================================================================================


def test_effect_of_beta_on_last_period():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=5)

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_config, targets="solve")

    # low beta
    params_low = get_params(beta=0.5, disutility_of_work=1.0)

    # high beta
    params_high = get_params(beta=0.99, disutility_of_work=1.0)

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model_config, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_low,
        V_arr_dict=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_high,
        V_arr_dict=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    last_period_index = 4
    assert (
        res_low.loc[last_period_index, :]["value"]
        <= res_high.loc[last_period_index, :]["value"]
    ).all()


def test_effect_of_disutility_of_work():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=5)

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_config, targets="solve")

    # low disutility_of_work
    params_low = get_params(beta=1.0, disutility_of_work=0.2)

    # high disutility_of_work
    params_high = get_params(beta=1.0, disutility_of_work=1.5)

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model_config, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_low,
        V_arr_dict=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_high,
        V_arr_dict=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    for period in range(5):
        # We expect that individuals with lower disutility of work, work (weakly) more
        # and thus consume (weakly) more
        assert (
            res_low.loc[period]["consumption"] >= res_high.loc[period]["consumption"]
        ).all()

        # We expect that individuals with lower disutility of work retire (weakly) later
        assert (
            res_low.loc[period]["retirement"] <= res_high.loc[period]["retirement"]
        ).all()


# ======================================================================================
# Helper functions
# ======================================================================================


def test_retrieve_actions():
    got = _lookup_values_from_indices(
        flat_indices=jnp.array([0, 3, 7]),
        grids={"a": jnp.linspace(0, 1, 5), "b": jnp.linspace(10, 20, 6)},
        grids_shapes=(5, 6),
    )
    assert_array_equal(got["a"], jnp.array([0, 0, 0.25]))
    assert_array_equal(got["b"], jnp.array([10, 16, 12]))


def test_get_continuous_action_argmax_given_discrete():
    argmax_and_max_Q_over_c_values = jnp.array(
        [
            [0, 1],
            [1, 0],
        ],
    )
    argmax = jnp.array([0, 1])
    vars_grid_shape = (2,)
    got = _lookup_optimal_continuous_actions(
        indices_argmax_Q_over_c=argmax_and_max_Q_over_c_values,
        discrete_argmax=argmax,
        discrete_actions_grid_shape=vars_grid_shape,
    )
    assert jnp.all(got == jnp.array([0, 0]))
