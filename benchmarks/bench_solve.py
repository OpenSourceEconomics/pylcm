"""Solve benchmarks for different models and grid sizes."""

import pytest

from lcm_examples import mortality, precautionary_savings, tiny


def test_solve_tiny(benchmark):
    model = tiny.get_model()
    params = tiny.get_params()
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")


@pytest.mark.parametrize("n_points", [10, 50, 200])
def test_solve_precautionary(benchmark, n_points):
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


def test_solve_mortality(benchmark):
    model = mortality.get_model(n_periods=4)
    params = mortality.get_params(n_periods=4)
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")
