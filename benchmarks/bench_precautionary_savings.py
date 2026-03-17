"""Precautionary savings benchmarks: solve, simulate, grid types."""

import jax.numpy as jnp
import pytest

from lcm_examples import precautionary_savings

_N_SUBJECTS = 1_000


def _make_model(*, wealth_grid_type="lin", wealth_n_points=10, consumption_n_points=10):
    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type=wealth_grid_type,
        wealth_n_points=wealth_n_points,
        consumption_n_points=consumption_n_points,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst",
        sigma=0.2,
        rho=0.9,
    )
    return model, params


def _make_initial_conditions(n_subjects):
    return {
        "age": jnp.full(n_subjects, 20.0),
        "wealth": jnp.full(n_subjects, 5.0),
        "income": jnp.full(n_subjects, 0.0),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }


@pytest.mark.parametrize("n_points", [50, 200, 500])
def test_solve(benchmark, n_points):
    model, params = _make_model(
        wealth_n_points=n_points,
        consumption_n_points=n_points,
    )
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")


@pytest.mark.parametrize("n_subjects", [1_000, 10_000])
def test_simulate(benchmark, n_subjects):
    model, params = _make_model()
    V_arr_dict = model.solve(params, log_level="off")
    initial_conditions = _make_initial_conditions(n_subjects)

    # Warm up JIT
    model.simulate(params, initial_conditions, V_arr_dict, log_level="off")
    benchmark(model.simulate, params, initial_conditions, V_arr_dict, log_level="off")


def test_solve_and_simulate(benchmark):
    model, params = _make_model(wealth_n_points=200, consumption_n_points=200)
    initial_conditions = _make_initial_conditions(_N_SUBJECTS)

    # Warm up JIT
    model.solve_and_simulate(params, initial_conditions, log_level="off")
    benchmark(
        model.solve_and_simulate,
        params,
        initial_conditions,
        log_level="off",
    )


@pytest.mark.parametrize("n_points", [500, 1000, 2000])
@pytest.mark.parametrize("grid_type", ["lin", "irreg"])
def test_grid_lookup(benchmark, n_points, grid_type):
    model, params = _make_model(
        wealth_grid_type=grid_type,
        wealth_n_points=n_points,
        consumption_n_points=n_points,
    )
    initial_conditions = _make_initial_conditions(100)

    # Warm up JIT
    model.solve_and_simulate(params, initial_conditions, log_level="off")
    benchmark(
        model.solve_and_simulate,
        params,
        initial_conditions,
        log_level="off",
    )
