from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal

from lcm import Model
from lcm.ages import AgeGrid
from lcm.input_processing import process_regimes
from lcm.logging import get_logger
from lcm.simulation.result import SimulationResult
from lcm.simulation.simulate import (
    _lookup_values_from_indices,
    simulate,
)

if TYPE_CHECKING:
    from pathlib import Path

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

    n_periods = 2
    ages = AgeGrid(start=0, stop=n_periods, step="Y")
    updated_working = working.replace(
        actions={
            **working.actions,
            "consumption": working.actions["consumption"].replace(stop=100),  # ty: ignore[unresolved-attribute]
        },
        active=lambda age: age < n_periods - 1,
    )
    updated_dead = dead.replace(active=lambda age: age >= n_periods - 1)
    internal_regimes = process_regimes(
        [updated_working, updated_dead],
        ages=ages,
        regime_id_cls=RegimeId,
        enable_jit=True,
    )

    return {
        "internal_regimes": internal_regimes,
        "regime_id_cls": RegimeId,
        "ages": ages,
    }


def test_simulate_using_raw_inputs(simulate_inputs):
    params = {
        "working": {
            "discount_factor": 1.0,
            "utility": {"disutility_of_work": 1.0},
            "next_wealth": {"interest_rate": 0.05},
            "next_regime": {"final_age": 0},  # n_periods=2, so final_age=0
            "borrowing_constraint": {},
            "labor_income": {},
        },
        "dead": {},
    }

    result = simulate(
        params=params,
        V_arr_dict={
            0: {"working": jnp.zeros(100), "dead": jnp.zeros(2)},
            1: {"working": jnp.zeros(100), "dead": jnp.zeros(2)},
        },
        initial_states={"wealth": jnp.array([1.0, 50.400803])},
        initial_regimes=["working"] * 2,
        logger=get_logger(debug_mode=False),
        **simulate_inputs,
    )
    got = result.to_dataframe().query('regime == "working"')

    assert (got["labor_supply"] == "retire").all()
    assert_array_almost_equal(got["consumption"], [1.0, 50.400803])


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
            # remove dependency on agent_age, so that wage becomes a parameter
            name: func
            for name, func in working.functions.items()
            if name not in ["agent_age", "wage"]
        }
        ages = AgeGrid(start=0, stop=n_periods, step="Y")
        updated_working = working.replace(
            functions=updated_functions, active=lambda age, n=n_periods: age < n - 1
        )
        updated_dead = dead.replace(active=lambda age, n=n_periods: age >= n - 1)

        params = get_params(n_periods=n_periods)
        # Since wage function is removed, wage becomes a parameter for labor_income
        params["working"]["labor_income"] = {"wage": 1.5}
        # Override final_age since we use AgeGrid starting at 0, not START_AGE
        params["working"]["next_regime"] = {"final_age": n_periods - 2}
        model = Model(
            [updated_working, updated_dead], ages=ages, regime_id_cls=RegimeId
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

    result = model.simulate(
        params,
        V_arr_dict=V_arr_dict,
        initial_states={"wealth": jnp.array([20.0, 150, 250, 320])},
        initial_regimes=["working"] * 4,
    )
    df = result.to_dataframe(
        additional_targets=["utility", "borrowing_constraint"]
    ).query('regime == "working"')

    # Check expected columns
    expected_cols = {
        "period",
        "age",
        "value",
        "labor_supply",
        "consumption",
        "wealth",
        "utility",
        "borrowing_constraint",
        "subject_id",
        "regime",
    }
    assert expected_cols == set(df.columns)

    # Everyone retires in the last period
    assert (df.loc[df["period"] == n_periods - 1, "labor_supply"] == "retire").all()

    # Higher wealth leads to higher consumption and value in each period
    # (data is sorted by subject_id which corresponds to increasing initial wealth)
    for col in ["consumption", "value"]:
        is_monotonic = df.groupby("period")[col].apply(
            lambda x: x.is_monotonic_increasing
        )
        assert is_monotonic.all(), (
            f"{col} should increase with wealth within each period"
        )


def test_simulate_with_only_discrete_actions():
    from tests.test_models.deterministic.discrete import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    model = get_model(n_periods=3)
    params = get_params(n_periods=3, wage=1.5, discount_factor=1, interest_rate=0)

    result = model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([0, 4])},
        initial_regimes=["working"] * 2,
    )
    got = result.to_dataframe().query('regime == "working"')

    # Expected: sorted by (subject_id, period)
    # Subject 0: wealth=0 -> works, low; wealth=2 -> retires, high
    # Subject 1: wealth=4 -> retires, high; wealth=2 -> retires, high
    expected = pd.DataFrame(
        {
            "subject_id": [0, 0, 1, 1],
            "period": [0, 1, 0, 1],
            "wealth": [0.0, 2.0, 4.0, 2.0],
            "labor_supply": ["work", "retire", "retire", "retire"],
            "consumption": ["low", "high", "high", "high"],
        }
    )

    assert_frame_equal(
        got[
            ["subject_id", "period", "wealth", "labor_supply", "consumption"]
        ].reset_index(drop=True),
        expected,
        check_dtype=False,
        check_categorical=False,
    )


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
        n_periods=n_periods,
        discount_factor=0.9,
        disutility_of_work=1.0,
    )

    # high discount_factor
    params_high = get_params(
        n_periods=n_periods,
        discount_factor=0.99,
        disutility_of_work=1.0,
    )

    # Simulate
    # ==================================================================================
    initial_wealth = jnp.array([20.0, 50, 70])

    params_low = get_params(
        n_periods=n_periods, discount_factor=0.9, disutility_of_work=1.0
    )
    params_high = get_params(
        n_periods=n_periods, discount_factor=0.99, disutility_of_work=1.0
    )

    df_low = (
        model.solve_and_simulate(
            params_low,
            initial_states={"wealth": initial_wealth},
            initial_regimes=["working"] * 3,
        )
        .to_dataframe()
        .query('regime == "working"')
    )

    df_high = (
        model.solve_and_simulate(
            params_high,
            initial_states={"wealth": initial_wealth},
            initial_regimes=["working"] * 3,
        )
        .to_dataframe()
        .query('regime == "working"')
    )

    # Higher beta (more patient) should lead to higher value in later periods
    merged = df_low.merge(
        df_high, on=["subject_id", "period"], suffixes=("_low", "_high")
    )
    period_4 = merged.query("period == 4")
    assert (period_4["value_low"] <= period_4["value_high"]).all()


def test_effect_of_disutility_of_work():
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    n_periods = 6
    model = get_model(n_periods=n_periods)

    # low disutility_of_work
    params_low = get_params(
        n_periods=n_periods,
        discount_factor=1.0,
        disutility_of_work=0.2,
    )

    # high disutility_of_work
    params_high = get_params(
        n_periods=n_periods,
        discount_factor=1.0,
        disutility_of_work=1.5,
    )

    # Simulate
    # ==================================================================================
    initial_wealth = jnp.array([20.0, 50, 70])

    params_low = get_params(
        n_periods=n_periods, discount_factor=1.0, disutility_of_work=0.2
    )
    params_high = get_params(
        n_periods=n_periods, discount_factor=1.0, disutility_of_work=1.5
    )

    df_low = (
        model.solve_and_simulate(
            params_low,
            initial_states={"wealth": initial_wealth},
            initial_regimes=["working"] * 3,
        )
        .to_dataframe()
        .query('regime == "working"')
    )

    df_high = (
        model.solve_and_simulate(
            params_high,
            initial_states={"wealth": initial_wealth},
            initial_regimes=["working"] * 3,
        )
        .to_dataframe()
        .query('regime == "working"')
    )

    # Merge results for easy comparison
    merged = df_low.merge(
        df_high, on=["subject_id", "period"], suffixes=("_low", "_high")
    )

    # Lower disutility of work -> work more -> consume more
    assert (merged["consumption_low"] >= merged["consumption_high"]).all()

    # Lower disutility -> retire later (work=0, retire=1, lower code = more work)
    assert (
        merged["labor_supply_low"].cat.codes.to_numpy()
        <= merged["labor_supply_high"].cat.codes.to_numpy()
    ).all()


# ======================================================================================
# Test use_labels parameter
# ======================================================================================


def test_to_dataframe_use_labels_parameter():
    """Test that use_labels=True/False controls discrete column dtypes."""
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    model = get_model(n_periods=3)
    params = get_params(n_periods=3)
    result = model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([20.0, 50.0])},
        initial_regimes=["working"] * 2,
    )

    # use_labels=True (default): discrete columns are Categorical with string labels
    df_labels = result.to_dataframe()
    for col in ["regime", "labor_supply"]:
        assert df_labels[col].dtype.name == "category", f"{col} should be categorical"
    assert set(df_labels["labor_supply"].cat.categories) == {"work", "retire"}

    # use_labels=False: discrete columns have numeric codes
    df_codes = result.to_dataframe(use_labels=False)
    assert df_codes["labor_supply"].dtype.kind in "iuf"  # integer/unsigned/float
    assert set(df_codes["labor_supply"].dropna().unique()).issubset({0, 1})


# ======================================================================================
# Test available_targets and additional_targets="all"
# ======================================================================================


@pytest.fixture
def regression_simulation_result():
    """Shared fixture for available_targets tests."""
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    model = get_model(n_periods=3)
    params = get_params(n_periods=3)
    return model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([20.0, 50.0])},
        initial_regimes=["working"] * 2,
    )


def test_available_targets_property(regression_simulation_result):
    """Test that available_targets shows what can be computed."""
    result = regression_simulation_result
    assert isinstance(result.available_targets, list)
    assert {"utility", "borrowing_constraint"} <= set(result.available_targets)


def test_additional_targets_all(regression_simulation_result):
    """Test that additional_targets='all' computes all available targets."""
    result = regression_simulation_result
    df = result.to_dataframe(additional_targets="all")
    assert set(result.available_targets) <= set(df.columns)


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


# ======================================================================================
# Test pickling
# ======================================================================================


def test_simulation_result_pickle_roundtrip(tmp_path: Path):
    """Test that SimulationResult can be pickled and unpickled."""
    from tests.test_models.deterministic.regression import (  # noqa: PLC0415
        get_model,
        get_params,
    )

    # Create a SimulationResult
    model = get_model(n_periods=3)
    params = get_params(n_periods=3)
    result = model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([20.0, 50.0])},
        initial_regimes=["working"] * 2,
    )

    # Pickle and unpickle
    pickle_path = tmp_path / "result.pkl"
    result.to_pickle(pickle_path)
    loaded = SimulationResult.from_pickle(pickle_path)

    # Compare metadata attributes
    assert loaded.n_periods == result.n_periods
    assert loaded.n_subjects == result.n_subjects
    assert loaded.regime_names == result.regime_names
    assert loaded.state_names == result.state_names
    assert loaded.action_names == result.action_names
    assert loaded.available_targets == result.available_targets

    # Compare DataFrames
    assert_frame_equal(loaded.to_dataframe(), result.to_dataframe())
