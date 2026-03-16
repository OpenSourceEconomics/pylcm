"""Grid type comparison benchmarks (LinSpaced vs LogSpaced vs IrregSpaced).

Addresses #225: benchmark IrregSpacedGrid vs LinSpacedGrid coordinate lookups.
"""

import jax.numpy as jnp
import pytest

from lcm_examples import precautionary_savings

_N_SUBJECTS = 100


@pytest.mark.parametrize("n_points", [50, 200, 500])
@pytest.mark.parametrize("grid_type", ["lin", "log", "irreg"])
def test_grid_types(benchmark, n_points, grid_type):
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
    initial_conditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.full(_N_SUBJECTS, 5.0),
        "income": jnp.full(_N_SUBJECTS, 0.0),
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
