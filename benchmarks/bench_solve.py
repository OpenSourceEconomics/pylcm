"""Solve benchmarks for different models and grid sizes."""

import jax.numpy as jnp
import pytest

from lcm_examples import mortality, precautionary_savings

_N_SUBJECTS = 100


@pytest.mark.parametrize("n_points", [50, 200, 500])
def test_solve(benchmark, n_points):
    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_n_points=n_points,
        consumption_n_points=n_points,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst",
        sigma=0.2,
        rho=0.9,
    )
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")


def test_mortality(benchmark):
    model = mortality.get_model(n_periods=4)
    params = mortality.get_params(n_periods=4)
    initial_conditions = {
        "age": jnp.full(_N_SUBJECTS, 40.0),
        "wealth": jnp.full(_N_SUBJECTS, 100.0),
        "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
    }
    # Warm up JIT
    model.solve_and_simulate(params, initial_conditions, log_level="off")
    benchmark(
        model.solve_and_simulate,
        params,
        initial_conditions,
        log_level="off",
    )
