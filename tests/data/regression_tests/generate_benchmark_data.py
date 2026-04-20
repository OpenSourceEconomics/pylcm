"""Generate regression test data for the small benchmark models.

Run with:
    pixi run -e tests-cuda13 python \
        tests/data/regression_tests/generate_benchmark_data.py --precision=64
    pixi run -e tests-cuda13 python \
        tests/data/regression_tests/generate_benchmark_data.py --precision=32

Covers `precautionary_savings` and `mortality`, whose per-subject simulation
output is reproducible across processes.

**Mahler & Yum is regenerated from pytest** because XLA kernel fusion differs
across processes and its stochastic mortality transitions amplify those
differences into trajectory-level drift. Use:

    LCM_UPDATE_FIXTURES=1 pixi run -e tests-cuda13 pytest \
        tests/test_regression_test.py::test_regression_mahler_yum
    LCM_UPDATE_FIXTURES=1 pixi run -e tests-cuda13 pytest \
        tests/test_regression_test.py::test_regression_mahler_yum --precision=32

This writes `mahler_yum_simulation_per_period.pkl` in each precision's
fixture directory from within the same process the regression test runs in,
which is what makes the comparison deterministic.
"""

import argparse
from pathlib import Path

import jax

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--precision", type=int, choices=[32, 64], default=64, help="32 or 64 bit"
)
_args = _parser.parse_args()

jax.config.update("jax_enable_x64", val=(_args.precision == 64))

import jax.numpy as jnp  # noqa: E402

from lcm_examples import mortality, precautionary_savings  # noqa: E402

DATA_DIR = Path(__file__).parent


def _generate_precautionary_savings(data_dir: Path) -> None:
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
    result.to_dataframe().to_pickle(data_dir / "precautionary_savings_simulation.pkl")


def _generate_mortality(data_dir: Path) -> None:
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
    result.to_dataframe().to_pickle(data_dir / "mortality_simulation.pkl")


if __name__ == "__main__":
    target = DATA_DIR / f"f{_args.precision}"
    target.mkdir(exist_ok=True)
    _generate_precautionary_savings(target)
    _generate_mortality(target)
