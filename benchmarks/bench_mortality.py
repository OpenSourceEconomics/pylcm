"""Mortality model benchmark: solve + simulate."""

import jax.numpy as jnp

from lcm_examples import mortality

_N_SUBJECTS = 1_000


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
