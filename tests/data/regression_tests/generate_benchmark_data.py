"""Generate regression test data for benchmark models.

Run with:
    pixi run -e tests-cuda13 python \
        tests/data/regression_tests/generate_benchmark_data.py --precision=64
    pixi run -e tests-cuda13 python \
        tests/data/regression_tests/generate_benchmark_data.py --precision=32

Regenerate when model internals change intentionally (e.g., numerical algorithm
improvements, grid changes). The stored DataFrames pin the simulation output so
accidental regressions are caught.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import jax
import jaxlib

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--precision", type=int, choices=[32, 64], default=64, help="32 or 64 bit"
)
_args = _parser.parse_args()

jax.config.update("jax_enable_x64", val=(_args.precision == 64))

import jax.numpy as jnp  # noqa: E402

from lcm_examples import mortality, precautionary_savings  # noqa: E402
from tests.test_models import processes  # noqa: E402

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
            "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        seed=12345,
        log_level="off",
    )
    result.to_dataframe(terminal_rows="all").to_pickle(
        data_dir / "precautionary_savings_simulation.pkl"
    )


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
            "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
        },
        seed=12345,
        log_level="off",
    )
    result.to_dataframe(terminal_rows="all").to_pickle(
        data_dir / "mortality_simulation.pkl"
    )


def _generate_process_panels(data_dir: Path) -> None:
    for distribution_type in (
        "uniform",
        "normal",
        "lognormal",
        "tauchen",
        "rouwenhorst",
    ):
        model = processes.get_model(n_periods=4, distribution_type=distribution_type)
        result = model.simulate(
            params=processes.get_params(distribution_type),
            initial_conditions={
                "health": jnp.asarray([0, 0], dtype=jnp.int32),
                "income": jnp.asarray([0.0, 0.0]),
                "wealth": jnp.asarray([1.0, 1.0]),
                "age": jnp.asarray([0.0, 0.0]),
                "regime_id": jnp.asarray([processes.RegimeId.alive] * 2),
            },
            seed=42,
            log_level="off",
        )
        assert result.randomness["address_version"] == "pylcm.random.v1"
        result.to_dataframe().to_pickle(
            data_dir / f"simulation_{distribution_type}.pkl"
        )


def _record_provenance(data_dir: Path) -> None:
    root = DATA_DIR.parents[2]
    git = shutil.which("git")
    assert git is not None
    source = subprocess.check_output(  # noqa: S603 - fixed Git inspection arguments
        [git, "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(  # noqa: S603 - fixed Git inspection arguments
        [git, "-C", str(root), "diff", "--binary"]
    )
    record = {
        "address_version": "pylcm.random.v1",
        "source_commit": source,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src").rglob("*.py"))
        },
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "precision": _args.precision,
        "prng_impl": "threefry2x32",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data_dir.glob("*.pkl"))
        },
    }
    (data_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    target = DATA_DIR / f"f{_args.precision}" / "random-v1"
    target.mkdir(exist_ok=False)
    _generate_precautionary_savings(target)
    _generate_mortality(target)
    _record_provenance(target)
    if _args.precision == 64:
        process_target = DATA_DIR.parent / "processes" / "random-v1"
        process_target.mkdir(exist_ok=False)
        _generate_process_panels(process_target)
        _record_provenance(process_target)
