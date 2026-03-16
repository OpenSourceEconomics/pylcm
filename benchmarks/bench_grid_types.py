"""Grid type comparison benchmarks (LinSpaced vs LogSpaced vs IrregSpaced).

Addresses #225: benchmark IrregSpacedGrid vs LinSpacedGrid coordinate lookups.
"""

import pytest

from lcm_examples import precautionary_savings


@pytest.mark.parametrize("n_points", [50, 200, 500])
@pytest.mark.parametrize("grid_type", ["lin", "log", "irreg"])
def test_solve_grid_types(benchmark, n_points, grid_type):
    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type=grid_type,
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
