from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from lcm import Model
from lcm.input_processing import process_regimes
from lcm.logging import get_logger
from lcm.simulation.simulate import (
    _lookup_values_from_indices,
    simulate,
)

if TYPE_CHECKING:
    import pandas as pd


# ======================================================================================
# Test simulate using raw inputs
# ======================================================================================


@pytest.fixture
def simulate_inputs():
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        RegimeId,
        dead,
        working,
    )

    updated_working = working.replace(
        actions={
            **working.actions,
            "consumption": working.actions["consumption"].replace(stop=100),  # ty: ignore[unresolved-attribute]
        },
        active=[0],
    )
    updated_dead = dead.replace(active=[1])
    internal_regimes = process_regimes(
        [updated_working, updated_dead],
        n_periods=2,
        regime_id_cls=RegimeId,
        enable_jit=True,
    )

    return {
        "internal_regimes": internal_regimes,
        "regime_id_cls": RegimeId,
    }


def test_simulate_using_raw_inputs(simulate_inputs):
    params = {
        "working": {
            "discount_factor": 1.0,
            "utility": {"disutility_of_work": 1.0},
            "next_wealth": {
                "interest_rate": 0.05,
            },
            "next_regime": {},  # last_period is now a temporal context variable
            "borrowing_constraint": {},
            "labor_income": {},
        },
        "dead": {},
    }

    got = simulate(
        params=params,
        V_arr_dict={
            0: {"working": jnp.zeros(100), "dead": jnp.zeros(2)},
            1: {"working": jnp.zeros(100), "dead": jnp.zeros(2)},
        },
        initial_states={"wealth": jnp.array([1.0, 50.400803])},
        initial_regimes=["working"] * 2,
        logger=get_logger(debug_mode=False),
        **simulate_inputs,
    )["working"]

    assert_array_equal(got.loc[:]["labor_supply"], 1)
    assert_array_almost_equal(got.loc[:]["consumption"], jnp.array([1.0, 50.400803]))


# ======================================================================================
# Test simulate
# ======================================================================================


@pytest.fixture
def iskhakov_et_al_2017_stripped_down_model_solution():
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        RegimeId,
        dead,
        get_params,
        working,
    )

    def _model_solution(n_periods):
        updated_functions = {
            # remove dependency on age, so that wage becomes a parameter
            name: func
            for name, func in working.functions.items()
            if name not in ["age", "wage"]
        }
        updated_working = working.replace(
            functions=updated_functions, active=range(n_periods - 1)
        )
        updated_dead = dead.replace(active=[n_periods - 1])

        params = get_params()
        # Since wage function is removed, wage becomes a parameter for labor_income
        params["working"]["labor_income"] = {"wage": 1.5}
        model = Model(
            [updated_working, updated_dead], n_periods=n_periods, regime_id_cls=RegimeId
        )
        V_arr_dict = model.solve(params=params)
        return V_arr_dict, params, model

    return _model_solution


def test_simulate_using_model_methods(
    iskhakov_et_al_2017_stripped_down_model_solution,
):
    n_periods = 4
    V_arr_dict, params, model = iskhakov_et_al_2017_stripped_down_model_solution(
        n_periods=n_periods,
    )

    res: pd.DataFrame = model.simulate(
        params,
        V_arr_dict=V_arr_dict,
        initial_states={"wealth": jnp.array([20.0, 150, 250, 320])},
        initial_regimes=["working"] * 4,
        additional_targets={"working": ["utility", "borrowing_constraint"]},
    )["working"]

    assert {
        "period",
        "value",
        "labor_supply",
        "consumption",
        "wealth",
        "utility",
        "borrowing_constraint",
        "subject_id",
    } == set(res.columns)

    # assert that everyone retires in the last period
    last_period_index = n_periods - 1
    assert_array_equal(res.loc[last_period_index, :]["labor_supply"], 1)

    for period in range(n_periods):
        # assert that higher wealth leads to higher consumption in each period
        assert (res.loc[res["period"] == period]["consumption"].diff()[1:] >= 0).all()

        # assert that higher wealth leads to higher value function in each period
        assert (res.loc[res["period"] == period]["value"].diff()[1:] >= 0).all()


def test_simulate_with_only_discrete_actions():
    from tests.test_models.deterministic.discrete import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    model = get_model(n_periods=3)
    params = get_params(wage=1.5, discount_factor=1, interest_rate=0)

    res: pd.DataFrame = model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([0, 4])},
        initial_regimes=["working"] * 2,
    )["working"]

    assert_array_equal(res["labor_supply"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["consumption"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["wealth"], jnp.array([0, 4, 2, 2]))


# ======================================================================================
# Testing effects of parameters
# ======================================================================================


def test_effect_of_discount_factor_on_last_period():
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    n_periods = 6
    model = get_model(n_periods=n_periods)

    # low discount_factor
    params_low = get_params(
        discount_factor=0.9,
        disutility_of_work=1.0,
    )

    # high discount_factor
    params_high = get_params(
        discount_factor=0.99,
        disutility_of_work=1.0,
    )

    # solutions
    solution_low = model.solve(params_low)
    solution_high = model.solve(params_high)

    # Simulate
    # ==================================================================================
    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = model.simulate(
        params_low,
        V_arr_dict=solution_low,
        initial_states={"wealth": initial_wealth},
        initial_regimes=["working"] * 3,
    )["working"]

    res_high: pd.DataFrame = model.simulate(
        params_high,
        V_arr_dict=solution_high,
        initial_states={"wealth": initial_wealth},
        initial_regimes=["working"] * 3,
    )["working"]

    # Asserting
    # ==================================================================================
    last_period_index = 4
    assert (
        res_low.loc[last_period_index, :]["value"]
        <= res_high.loc[last_period_index, :]["value"]
    ).all()


def test_effect_of_disutility_of_work():
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    n_periods = 6
    model = get_model(n_periods=n_periods)

    # low disutility_of_work
    params_low = get_params(
        discount_factor=1.0,
        disutility_of_work=0.2,
    )

    # high disutility_of_work
    params_high = get_params(
        discount_factor=1.0,
        disutility_of_work=1.5,
    )

    # solutions
    solution_low = model.solve(params_low)
    solution_high = model.solve(params_high)

    # Simulate
    # ==================================================================================
    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = model.simulate(
        params_low,
        V_arr_dict=solution_low,
        initial_states={"wealth": initial_wealth},
        initial_regimes=["working"] * 3,
    )["working"]

    res_high: pd.DataFrame = model.simulate(
        params_high,
        V_arr_dict=solution_high,
        initial_states={"wealth": initial_wealth},
        initial_regimes=["working"] * 3,
    )["working"]

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
            res_low.loc[period]["labor_supply"] <= res_high.loc[period]["labor_supply"]
        ).all()


# ======================================================================================
# Helper functions
# ======================================================================================


def test_retrieve_actions():
    got = _lookup_values_from_indices(
        flat_indices=jnp.array([0, 3, 7]),
        grids={"a": jnp.linspace(0, 1, 5), "b": jnp.linspace(10, 20, 6)},
    )
    assert_array_equal(got["a"], jnp.array([0, 0, 0.25]))
    assert_array_equal(got["b"], jnp.array([10, 16, 12]))
