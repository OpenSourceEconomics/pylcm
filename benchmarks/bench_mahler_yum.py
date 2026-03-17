"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

import jax.numpy as jnp

from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)

_START_PARAMS_WITHOUT_BETA = {k: v for k, v in START_PARAMS.items() if k != "beta"}

_N_SUBJECTS = 100


def test_mahler_yum_2024(benchmark):
    model = MAHLER_YUM_MODEL
    common_params, initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=_N_SUBJECTS,
        **_START_PARAMS_WITHOUT_BETA,
    )
    params = {
        "alive": {
            "discount_factor": START_PARAMS["beta"]["mean"],
            **common_params,
        },
    }
    initial_conditions = {
        **initial_states,
        "regime_id": jnp.full(
            _N_SUBJECTS,
            model.regime_names_to_ids["alive"],
            dtype=jnp.int32,
        ),
    }
    # Warm up JIT
    model.solve_and_simulate(params, initial_conditions, log_level="off")
    benchmark(
        model.solve_and_simulate,
        params,
        initial_conditions,
        log_level="off",
    )
