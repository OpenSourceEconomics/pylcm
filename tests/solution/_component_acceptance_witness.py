"""Describe a collected component-job result before and after a checkpoint reload.

Run as a fresh process:

- `python -m tests.solution._component_acceptance_witness produce <directory> ...`
  runs the single-process block-major reference and the component-job campaign,
  checks the collected result against the reference in values, raw bytes, public
  panel and placement, saves the collected solution and simulation under
  `<directory>`, and prints one JSON line describing the collected result.
- `python -m tests.solution._component_acceptance_witness reload <directory>`
  reads the saved solution and simulation back through the public readers and
  prints the same description of what it read.

The description holds, per raw leaf, its devices, sharding type, memory kind,
dtype, shape and a byte digest; digests of every value and of the public panel;
and the plain values of the raw fields the tests check analytically.
"""

import argparse
import dataclasses
import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import jax
import numpy as np
import pandas as pd

from lcm.component_jobs import (
    collect_component_jobs,
    plan_component_jobs,
    run_component_job,
)
from lcm.persistence import load_solution, save_solution
from lcm.result import SimulationResult
from tests.solution._component_placement_witness import (
    _bytes,
    _initial_population,
    _model,
    _placement_mismatches,
)


def produce(
    *,
    directory: Path,
    devices: tuple[int, ...],
    width: int,
    codes: tuple[int, ...],
    enable_jit: bool,
) -> dict[str, object]:
    """Collect a campaign, compare it with the reference, save it, describe it."""
    params = {"discount_factor": 0.9}
    initial = _initial_population(codes=codes)
    reference = _model(
        devices=devices,
        width=width,
        enable_jit=enable_jit,
        simulation_sharding="legacy",
    ).simulate(params=params, initial_conditions=initial, seed=7, log_level="off")
    model = _model(
        devices=devices,
        width=width,
        enable_jit=enable_jit,
        simulation_sharding="legacy",
    )
    with tempfile.TemporaryDirectory() as campaign:
        plan = plan_component_jobs(
            model=model,
            params=params,
            directory=Path(campaign),
            assignment=((0,), (1,), (2,)),
            initial_conditions=initial,
            seed=7,
        )
        for job in range(len(plan.jobs)):
            run_component_job(
                model=model,
                params=params,
                directory=Path(campaign),
                job=job,
                initial_conditions=initial,
                log_level="off",
            )
        collected = collect_component_jobs(
            model=model, params=params, directory=Path(campaign), log_level="off"
        )
    simulation = cast("SimulationResult", collected.simulation)
    report = {
        "misplaced": _placement_mismatches(reference=reference, collected=simulation),
        **_describe(simulation=simulation, values=collected.solution.values),
    }
    save_solution(solution=collected.solution, path=directory / "solution")
    simulation.save(directory=directory / "simulation")
    return report


def reload(*, directory: Path) -> dict[str, object]:
    """Describe the saved simulation as every public reader returns it."""
    loaded = SimulationResult.load(directory=directory / "simulation")
    return {
        **_describe(simulation=loaded, values=loaded.period_to_regime_to_V_arr),
        "simulation_load_solution_values": _value_digests(
            values=SimulationResult.load_solution(directory=directory / "simulation")
        ),
        "load_solution_values": _value_digests(
            values=load_solution(path=directory / "solution").values
        ),
    }


def _describe(
    *, simulation: SimulationResult, values: Mapping[int, Mapping[str, jax.Array]]
) -> dict[str, object]:
    raw = simulation.raw_results
    return {
        "raw": {
            path: {
                "devices": sorted(
                    [device.platform, device.id] for device in leaf.devices()
                ),
                "sharding": type(leaf.sharding).__name__,
                "memory_kind": leaf.sharding.memory_kind,
                "dtype": leaf.dtype.str,
                "shape": list(leaf.shape),
                "sha256": _digest(array=leaf),
            }
            for path, leaf in _named_leaves(tree=raw, path="")
        },
        "default_memory_kinds": {
            f"{device.platform}:{device.id}": device.default_memory().kind
            for backend in {"cpu", jax.default_backend()}
            for device in jax.local_devices(backend=backend)
        },
        "values": _value_digests(values=values),
        "panels": {
            terminal_rows: _panel_digest(
                panel=simulation.to_dataframe(terminal_rows=terminal_rows)
            )
            for terminal_rows in ("first", "all")
        },
        "fields": {
            "live_value": np.asarray(raw["live"][0].V_arr).tolist(),
            "live_choice": np.asarray(raw["live"][0].actions["choice"]).tolist(),
            "live_pref_type": np.asarray(raw["live"][0].states["pref_type"]).tolist(),
            "live_wealth": np.asarray(raw["live"][0].states["wealth"]).tolist(),
            "dead_value": np.asarray(raw["dead"][1].V_arr).tolist(),
        },
    }


def _named_leaves(*, tree: object, path: str) -> Iterator[tuple[str, jax.Array]]:
    """Yield every array leaf under a path of mapping keys and field names.

    The path names each leaf by regime, period, field and variable, so a leaf is
    matched by what it is, whatever order the mappings above it iterate in.
    """
    if isinstance(tree, Mapping):
        for key, subtree in tree.items():
            yield from _named_leaves(tree=subtree, path=f"{path}[{key!r}]")
    elif dataclasses.is_dataclass(tree):
        for field in dataclasses.fields(tree):
            yield from _named_leaves(
                tree=getattr(tree, field.name), path=f"{path}.{field.name}"
            )
    else:
        yield path, cast("jax.Array", tree)


def _digest(*, array: jax.Array) -> str:
    dtype, shape, data = _bytes(array)
    return hashlib.sha256(f"{dtype}{shape}".encode() + data).hexdigest()


def _value_digests(*, values: Mapping[int, Mapping[str, jax.Array]]) -> dict[str, str]:
    return {
        f"{period}/{regime}": _digest(array=value)
        for period, regimes in values.items()
        for regime, value in regimes.items()
    }


def _panel_digest(*, panel: pd.DataFrame) -> str:
    rows = pd.util.hash_pandas_object(panel, index=True).to_numpy().tobytes()
    schema = repr(list(panel.dtypes.astype(str).items())).encode()
    return hashlib.sha256(schema + rows).hexdigest()


def main() -> None:
    """Run one mode and print its description as one JSON line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("produce", "reload"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--codes", default="2,0,2")
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    options = parser.parse_args()
    if options.mode == "produce":
        report = produce(
            directory=options.directory,
            devices=tuple(int(device) for device in options.devices.split(",")),
            width=options.width,
            codes=tuple(int(code) for code in options.codes.split(",")),
            enable_jit=not options.eager,
        )
    else:
        report = reload(directory=options.directory)
    print(json.dumps(report, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
