"""Isolated four-cell execution-control companion for ASV simulation.

This diagnostic preserves the historical ASV benchmark and runs only when invoked
explicitly. Each cell owns a fresh process and a fixed validation realization.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
from jax import monitoring

from lcm.execution import ExecutionConfig
from lcm_examples import precautionary_savings

N_SUBJECTS = 1_000_000
SIMULATION_SEED = 20_250_915
WARM_REPETITIONS = 3
DEFAULT_BUDGET_BYTES = 8 * 1024**3
CELLS = (
    ("legacy-subject64", "legacy", 64),
    ("legacy-subject128", "legacy", 128),
    ("independent-subject64", "independent", 64),
    ("independent-subject128", "independent", 128),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _tree_identity(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {str(key): _tree_identity(tree[key]) for key in sorted(tree, key=str)}
    return _array_identity(tree)


def _frame_identity(frame: pd.DataFrame) -> dict[str, Any]:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return {
        "shape": list(frame.shape),
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "sha256": _sha256_bytes(hashed.tobytes()),
    }


def _memory_snapshot() -> list[dict[str, Any]]:
    rows = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except AttributeError, RuntimeError:
            stats = None
        rows.append(
            {
                "id": int(device.id),
                "platform": device.platform,
                "kind": device.device_kind,
                "peak_bytes_in_use": (
                    None if not stats else int(stats.get("peak_bytes_in_use", 0))
                ),
                "bytes_in_use": None
                if not stats
                else int(stats.get("bytes_in_use", 0)),
                "probe": "jax.Device.memory_stats",
                "coverage": "allocator counters at the sampling boundary",
            }
        )
    return rows


def _physical_device_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        return {
            "command": command,
            "returncode": None,
            "rows": [],
            "stderr": str(error),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": completed.stdout.splitlines() if completed.returncode == 0 else [],
        "stderr": completed.stderr.strip(),
    }


@contextlib.contextmanager
def _compilation_events():
    totals: dict[str, float] = {}

    # keyword-only-exempt: library-callback=jax.monitoring.duration_listener
    def listener(path: str, duration: float) -> None:
        if path.startswith("/jax/core/compile/"):
            totals[path] = totals.get(path, 0.0) + float(duration)

    monitoring.register_event_duration_secs_listener(listener)
    try:
        yield totals
    finally:
        monitoring.unregister_event_duration_secs_listener(listener)


def _plan_identity(prepared: Any) -> dict[str, Any]:
    profile = prepared.plan.profile
    receipt = prepared.plan.receipt
    return {
        "selected_outer_extent": profile.n_subjects,
        "padded_population": profile.padded_population,
        "resolved_axis_widths": dict(profile.axis_widths),
        "stages": [
            {
                "name": stage.name,
                "devices": [int(device.id) for device in stage.devices],
                "peak_bytes": stage.peak_bytes,
                "reservation_bytes": stage.reservation_bytes,
            }
            for stage in (*profile.stages, *profile.host_stages)
        ],
        "required_bytes": {
            str(device.id): value
            for device, value in prepared.plan.required_bytes.items()
        },
        "admission_receipt": None if receipt is None else dataclasses.asdict(receipt),
    }


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    lock = root / "pixi.lock"
    return {
        "repository_commit": commit,
        "pixi_lock_sha256": _sha256_bytes(lock.read_bytes()),
        "python": sys.version,
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "lcm": importlib.metadata.version("pylcm"),
        "source_files": {
            "companion": _sha256_bytes(Path(__file__).read_bytes()),
            "model_builder": _sha256_bytes(
                Path(precautionary_savings.__file__).read_bytes()
            ),
        },
    }


def _make_initial_conditions() -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        "age": jnp.full(N_SUBJECTS, 20.0),
        "wealth": jnp.full(N_SUBJECTS, 5.0),
        "income": jnp.full(N_SUBJECTS, 0.0),
        "regime_id": jnp.zeros(N_SUBJECTS, dtype=jnp.int32),
    }


def run_cell(
    *, name: str, _policy: str, width: int, budget_bytes: int
) -> dict[str, Any]:
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected exactly one GPU, got {devices!r}.")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    # `policy` no longer selects an ExecutionConfig field: the single budgeted
    # planner now runs unconditionally. The cell name and CLI argument stay for
    # provenance continuity across historical measurement records.
    controls = {
        "device_memory_bytes": budget_bytes,
        "sharded_states": [],
        "axis_widths": {"subject": width},
        "devices": [int(devices[0].id)],
        "donate_buffers": True,
        "max_compilation_workers": workers,
    }
    started = time.perf_counter()
    config = ExecutionConfig(
        device_memory_bytes=budget_bytes,
        sharded_states=(),
        axis_widths={"subject": width},
        devices=(int(devices[0].id),),
        donate_buffers=True,
    )
    model = precautionary_savings.create_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type="lin",
        wealth_n_points=10,
        consumption_n_points=10,
        execution_config=config,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst", sigma=0.2, rho=0.9
    )
    initial = _make_initial_conditions()
    setup_seconds = time.perf_counter() - started

    with _compilation_events() as solve_events:
        started = time.perf_counter()
        solution = model.solve(
            params=params, max_compilation_workers=workers, log_level="off"
        )
        jax.block_until_ready(solution)
        solve_seconds = time.perf_counter() - started

    import lcm.model as model_module

    original_prepare = model_module.prepare_simulation_chunks
    plans = []

    def capture_prepare(**kwargs: Any) -> Any:
        prepared = original_prepare(**kwargs)
        plans.append(_plan_identity(prepared))
        return prepared

    model_module.prepare_simulation_chunks = capture_prepare
    raw_seconds = []
    conversion_seconds = []
    compile_events = []
    panel_identity = None
    try:
        for _ in range(1 + WARM_REPETITIONS):
            with _compilation_events() as events:
                started = time.perf_counter()
                output = model.simulate(
                    params=params,
                    initial_conditions=initial,
                    solution=solution,
                    seed=SIMULATION_SEED,
                    log_level="off",
                )
                jax.block_until_ready(output.raw_results)
                raw_seconds.append(time.perf_counter() - started)
            compile_events.append(dict(events))
            started = time.perf_counter()
            frame = output.to_dataframe()
            conversion_seconds.append(time.perf_counter() - started)
            identity = _frame_identity(frame)
            if panel_identity is None:
                panel_identity = identity
            elif identity != panel_identity:
                raise AssertionError("Repeated simulations produced different panels.")
            del frame, output
            gc.collect()
    finally:
        model_module.prepare_simulation_chunks = original_prepare

    warm_median = statistics.median(raw_seconds[1:])
    return {
        "schema_version": "precautionary-savings-control-companion-1",
        "status": "complete",
        "exit_code": 0,
        "cell": name,
        "benchmark_target": "PrecautionarySavingsSimulate W10/C10 N1000000",
        "historical_series_modified": False,
        "fixed_realization": {
            "seed": SIMULATION_SEED,
            "role": "paired companion validation; not the historical ASV seed",
        },
        "source": _source_identity(),
        "environment": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
            "jax_visible_devices": [
                {"id": int(device.id), "kind": device.device_kind} for device in devices
            ],
            "physical_device": _physical_device_identity(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "allocated_cpus": workers,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "xla_python_client_allocator": os.environ.get(
                "XLA_PYTHON_CLIENT_ALLOCATOR"
            ),
            "xla_python_client_mem_fraction": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
        "controls": controls,
        "budget_rationale": (
            "8 GiB represented budget, conservatively below one A40's physical "
            "capacity; physical free memory is recorded separately and this is "
            "not a cap."
        ),
        "workload": {
            "periods": 5,
            "wealth_points": 10,
            "consumption_points": 10,
            "income_points": 5,
            "subjects": N_SUBJECTS,
            "dtype": str(np.asarray(initial["wealth"]).dtype),
        },
        "input_identity": {
            "initial": _tree_identity(initial),
            "parameters": _tree_identity(params),
        },
        "timings": {
            "setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "solve_compilation_events": dict(solve_events),
            "first_raw_simulation_seconds": raw_seconds[0],
            "warm_raw_simulation_seconds": raw_seconds[1:],
            "warm_median_seconds": warm_median,
            "first_call_minus_warm_median_seconds": raw_seconds[0] - warm_median,
            "dataframe_conversion_seconds": conversion_seconds,
            "simulation_compilation_events": compile_events,
        },
        "resolved_plans": plans,
        "panel_identity": panel_identity,
        "memory": {
            "device": _memory_snapshot(),
            "host_process_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "retained_object_state": (
            "solution retained; one output at a time; first panel hash retained"
        ),
    }


def _write_json(*, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run_all(*, output_dir: Path, budget_bytes: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    commands = []
    for name, policy, width in CELLS:
        output = output_dir / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.asv.precautionary_savings_control_companion",
            "--cell",
            name,
            "--policy",
            policy,
            "--width",
            str(width),
            "--budget-bytes",
            str(budget_bytes),
            "--output",
            str(output),
        ]
        commands.append(command)
        subprocess.run(command, check=True)
        results.append(json.loads(output.read_text()))
    reference = results[0]["panel_identity"]
    if any(result["panel_identity"] != reference for result in results[1:]):
        raise AssertionError("The four execution-control cells differ numerically.")
    summary = {
        "schema_version": "precautionary-savings-control-factorial-1",
        "status": "complete",
        "exit_code": 0,
        "cells": [result["cell"] for result in results],
        "commands": commands,
        "common_budget_bytes": budget_bytes,
        "cross_cell_exact_panel_identity": reference,
        "cross_cell_parity": "exact hash equality",
        "results": results,
    }
    _write_json(path=output_dir / "selected-factorial.json", payload=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--cell")
    parser.add_argument("--policy", choices=("legacy", "independent"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run_all:
        if args.output_dir is None:
            parser.error("--run-all requires --output-dir")
        run_all(output_dir=args.output_dir, budget_bytes=args.budget_bytes)
        return
    if None in (args.cell, args.policy, args.width, args.output):
        parser.error("cell mode requires --cell, --policy, --width, and --output")
    try:
        result = run_cell(
            name=args.cell,
            _policy=args.policy,
            width=args.width,
            budget_bytes=args.budget_bytes,
        )
    except Exception as error:
        _write_json(
            path=args.output,
            payload={
                "schema_version": "precautionary-savings-control-companion-1",
                "status": "failed",
                "exit_code": 1,
                "cell": args.cell,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    _write_json(path=args.output, payload=result)


if __name__ == "__main__":
    main()
