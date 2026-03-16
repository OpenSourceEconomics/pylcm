"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL, START_PARAMS, create_inputs

_START_PARAMS_WITHOUT_BETA = {k: v for k, v in START_PARAMS.items() if k != "beta"}


def test_solve_mahler_yum(benchmark):
    model = MAHLER_YUM_MODEL
    common_params, _initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=10,
        **_START_PARAMS_WITHOUT_BETA,
    )
    params = {
        "alive": {
            "discount_factor": START_PARAMS["beta"]["mean"],
            **common_params,
        },
    }
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")
