"""Record the complete outputs of four representative models for regression tests.

The stored archive holds, for each of `tiny`, `precautionary_savings_health`,
`iskhakov_et_al_2017` and `collective_household`:

- every solved value array, at `<model>/V/<period>/<regime>`, with its own dtype
  and shape;
- every column of the simulated panel, at `<model>/simulation/<column>`, numeric
  columns with their own dtype and discrete columns as their string labels;
- the provenance, at `__provenance__/<field>`: the commit and dirty state of the
  source tree that produced it, the location `lcm` was imported from, the
  precision, the JAX backend and the JAX version.

Next to the archive, `host_file` names a text file holding `host_fingerprint` of
the machine that produced it. The last bits of a float output depend on the
instructions XLA's CPU backend emits, so the tests compare bytes only on a host
with the same fingerprint.

Each model is built from the `lcm_examples` of the tree being probed, with that
tree's own example parameters, so the archive records what that tree computes
for the same economic model. The same extraction serves the tests, which compare
the current tree's outputs with the archive key for key.

Regenerate at a given commit from a detached worktree of it, importing that
tree's sources through `PYTHONPATH` inside this checkout's environment, so the
environment is held fixed and only the source moves. The precision is set with
JAX's own `JAX_ENABLE_X64` switch, which takes effect when JAX is imported:

    git worktree add --detach <old> <sha>
    cp src/_lcm/version.py <old>/src/_lcm/version.py
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=<old>/src zsh -ic "cd <repo> && \
        cap pixi run -e type-checking python \
        tests/data/regression_tests/generate_representative_model_values.py \
        --source-root <old> --precision 64 \
        --output tests/data/regression_tests/f64/representative_model_values.npz"

and the same with `JAX_ENABLE_X64=0`, `--precision 32` and the `f32` directory.
The script refuses to write when `lcm` was not imported from `--source-root`, when
the precision JAX runs at differs from `--precision`, or when the backend is not
the CPU.
"""

import argparse
import os
import platform
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import lcm
from lcm import Model
from lcm_examples import (
    collective_household,
    iskhakov_et_al_2017,
    precautionary_savings_health,
    tiny,
)

REPRESENTATIVE_MODELS = (
    "tiny",
    "precautionary_savings_health",
    "iskhakov_et_al_2017",
    "collective_household",
)
PROVENANCE = "__provenance__"
SIMULATION_SEED = 12345
_N_SUBJECTS = 4
# CPU flags that change the vector width or the fused multiply-adds XLA emits for
# float arithmetic.
_ISA_FLAG_PREFIXES = ("avx", "fma", "sse4")


def representative_outputs(name: str) -> dict[str, np.ndarray]:
    """Return every solved value array and simulated column of one model.

    Args:
        name: One of `REPRESENTATIVE_MODELS`.

    Returns:
        The arrays keyed by `<name>/V/<period>/<regime>` and
        `<name>/simulation/<column>`.

    """
    model, params, initial_conditions = _representative_case(name)
    values = model.solve(params=params, log_level="off").values
    outputs = {
        key: np.asarray(leaf)
        for period, by_regime in values.items()
        for key, leaf in _leaves(prefix=f"{name}/V/{period}", value=by_regime)
    }
    frame = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        seed=SIMULATION_SEED,
        log_level="off",
    ).to_dataframe()
    outputs.update(
        {
            f"{name}/simulation/{column}": _column_array(frame[column])
            for column in frame.columns
        }
    )
    return outputs


def host_fingerprint() -> str:
    """Return the OS, machine, vector-ISA flags and XLA ISA cap of this host.

    The flags come from `/proc/cpuinfo`; a host without it records none, so its
    fingerprint never matches a Linux one.
    """
    # ponytail: a feature-flag subset, not XLA's full target description; record
    # XLA's host CPU features if two hosts with one fingerprint ever disagree.
    cpuinfo = Path("/proc/cpuinfo")
    flags: set[str] = set()
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    isa = sorted(flag for flag in flags if flag.startswith(_ISA_FLAG_PREFIXES))
    cap = [
        flag
        for flag in os.environ.get("XLA_FLAGS", "").split()
        if flag.startswith("--xla_cpu_max_isa")
    ]
    return " ".join([platform.system(), platform.machine().lower(), *isa, *cap])


def host_file(archive: Path) -> Path:
    """Return the text file next to `archive` that holds its host fingerprint."""
    return archive.with_suffix(".host.txt")


def main() -> None:
    """Write the representative outputs and their provenance to `--output`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    lcm_file = Path(lcm.__file__).resolve()
    _fail_if_not_probing(
        lcm_file=lcm_file, source_root=source_root, precision=args.precision
    )
    arrays: dict[str, np.ndarray] = {}
    for name in REPRESENTATIVE_MODELS:
        arrays.update(representative_outputs(name))
    provenance = {
        "source_sha": _git(source_root, "rev-parse", "HEAD"),
        "source_dirty": _git(source_root, "status", "--porcelain", "--", "src"),
        "lcm_file": str(lcm_file.relative_to(source_root)),
        "precision": str(args.precision),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
    }
    arrays.update(
        {
            f"{PROVENANCE}/{field}": np.asarray(value)
            for field, value in provenance.items()
        }
    )
    np.savez_compressed(args.output, allow_pickle=False, **arrays)
    host_file(args.output).write_text(host_fingerprint() + "\n")
    for field, value in provenance.items():
        print(f"{field}: {value!r}")  # noqa: T201
    print(f"wrote {len(arrays)} arrays to {args.output}")  # noqa: T201


def _fail_if_not_probing(*, lcm_file: Path, source_root: Path, precision: int) -> None:
    """Refuse a run that would record outputs of another tree, precision or backend."""
    if not lcm_file.is_relative_to(source_root):
        msg = f"lcm was imported from {lcm_file}, outside {source_root}."
        raise RuntimeError(msg)
    if bool(jax.config.read("jax_enable_x64")) != (precision == 64):
        msg = f"JAX runs with x64 {jax.config.read('jax_enable_x64')}, not {precision}."
        raise RuntimeError(msg)
    if jax.default_backend() != "cpu":
        msg = f"JAX runs on {jax.default_backend()!r}; the archive records CPU results."
        raise RuntimeError(msg)


def _git(source_root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(source_root), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _representative_case(name: str) -> tuple[Model, dict, dict]:
    """Build one representative model with its params and initial conditions."""
    starting_wealth = jnp.linspace(5.0, 40.0, _N_SUBJECTS)
    regime_ids = jnp.zeros(_N_SUBJECTS, dtype=jnp.int32)
    if name == "tiny":
        return (
            tiny.get_model(n_periods=3),
            tiny.get_params(n_periods=3),
            {
                "age": jnp.full(_N_SUBJECTS, 25.0),
                "wealth": starting_wealth,
                "regime_id": regime_ids,
            },
        )
    if name == "precautionary_savings_health":
        return (
            precautionary_savings_health.get_model(retirement_age=20),
            precautionary_savings_health.get_params(retirement_age=20),
            {
                "age": jnp.full(_N_SUBJECTS, 18.0),
                "wealth": starting_wealth,
                "health": jnp.linspace(0.2, 0.8, _N_SUBJECTS),
                "regime_id": regime_ids,
            },
        )
    if name == "iskhakov_et_al_2017":
        return (
            iskhakov_et_al_2017.get_model(n_periods=4),
            iskhakov_et_al_2017.get_params(n_periods=4),
            {
                "age": jnp.full(_N_SUBJECTS, 40.0),
                "wealth": jnp.linspace(20.0, 200.0, _N_SUBJECTS),
                "regime_id": regime_ids,
            },
        )
    model = collective_household.get_model(
        n_periods=3, wealth_n_points=6, consumption_n_points=6
    )
    return (
        model,
        collective_household.get_params(),
        collective_household.get_initial_conditions(
            n_subjects=_N_SUBJECTS, model=model
        ),
    )


def _leaves(*, prefix: str, value: object) -> Iterator[tuple[str, object]]:
    """Yield the leaves of a nested mapping with their `/`-joined key paths."""
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield from _leaves(prefix=f"{prefix}/{key}", value=inner)
    else:
        yield prefix, value


def _column_array(series: pd.Series) -> np.ndarray:
    """Return a numeric column as is and a discrete column as its labels."""
    if isinstance(series.dtype, pd.CategoricalDtype) or not (
        pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_numeric_dtype(series.dtype)
    ):
        return series.astype(str).to_numpy(dtype=str)
    return series.to_numpy()


if __name__ == "__main__":
    main()
