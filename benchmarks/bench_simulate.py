"""Simulation benchmarks with varying numbers of subjects."""

import jax.numpy as jnp
import pytest

from lcm_examples import precautionary_savings


@pytest.mark.parametrize("n_subjects", [100, 1_000, 10_000])
def test_simulate_precautionary(benchmark, n_subjects):
    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_n_points=10,
        consumption_n_points=10,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst",
        sigma=0.2,
        rho=0.9,
    )
    V_arr_dict = model.solve(params, log_level="off")

    initial_conditions = {
        "age": jnp.full(n_subjects, 20.0),
        "wealth": jnp.full(n_subjects, 5.0),
        "income": jnp.full(n_subjects, 0.0),
        "regime_id": jnp.full(n_subjects, 0, dtype=jnp.int32),
    }

    # Warm up JIT
    model.simulate(params, initial_conditions, V_arr_dict, log_level="off")
    benchmark(model.simulate, params, initial_conditions, V_arr_dict, log_level="off")
