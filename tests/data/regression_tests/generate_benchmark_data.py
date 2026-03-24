"""Generate regression test data for benchmark models.

Run with:
    pixi run -e tests-cuda13 python \\
        tests/data/regression_tests/generate_benchmark_data.py

Requires a GPU (Mahler & Yum is GPU-only). Regenerate when model internals change
intentionally (e.g., numerical algorithm improvements, grid changes). The stored
DataFrames pin the simulation output so accidental regressions are caught.
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from lcm_examples import mortality, precautionary_savings
from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)

DATA_DIR = Path(__file__).parent


def _generate_precautionary_savings() -> None:
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

    n_subjects = 4
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 20.0),
            "wealth": jnp.full(n_subjects, 5.0),
            "income": jnp.full(n_subjects, 0.0),
            "regime": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe()
    df.to_pickle(DATA_DIR / "precautionary_savings_simulation.pkl")
    print(f"Wrote precautionary_savings_simulation.pkl  ({len(df)} rows)")  # noqa: T201


def _generate_mortality() -> None:
    n_periods = 4
    model = mortality.get_model(n_periods=n_periods)
    params = mortality.get_params(n_periods=n_periods)

    n_subjects = 4
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_subjects, 40.0),
            "wealth": jnp.full(n_subjects, 100.0),
            "regime": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe()
    df.to_pickle(DATA_DIR / "mortality_simulation.pkl")
    print(f"Wrote mortality_simulation.pkl  ({len(df)} rows)")  # noqa: T201


def _generate_mahler_yum() -> None:
    n_subjects = 4
    start_params_without_beta = {k: v for k, v in START_PARAMS.items() if k != "beta"}
    common_params, initial_states, _discount_factor_type = create_inputs(
        seed=0,
        n_simulation_subjects=n_subjects,
        **start_params_without_beta,  # ty: ignore[invalid-argument-type]
    )
    model = MAHLER_YUM_MODEL
    params = {
        "alive": {
            "discount_factor": START_PARAMS["beta"]["mean"],  # ty: ignore[invalid-argument-type, not-subscriptable]
            **common_params,
        },
    }
    initial_conditions = {
        **initial_states,
        "regime": jnp.full(
            n_subjects,
            model.regime_names_to_ids["alive"],
            dtype=jnp.int32,
        ),
    }

    result = model.simulate(
        params=params,  # ty: ignore[invalid-argument-type]
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        log_level="off",
    )
    df = result.to_dataframe()
    df.to_pickle(DATA_DIR / "mahler_yum_simulation.pkl")
    print(f"Wrote mahler_yum_simulation.pkl  ({len(df)} rows)")  # noqa: T201


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", val=True)
    _generate_precautionary_savings()
    _generate_mortality()
    _generate_mahler_yum()
