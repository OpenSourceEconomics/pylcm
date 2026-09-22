"""Measure source-sealed public solves in an isolated GPU process."""

# Precision must be set before importing numerical fixtures.
# Instrumentation deliberately observes private compiler and witness helpers.
# ruff: noqa: E402, SLF001

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch


def leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, jax.Array):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from leaves(child)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from leaves(getattr(value, field.name))


def synchronize(value: Any) -> None:
    for array in leaves(value):
        array.block_until_ready()


def emit(record: dict[str, Any]) -> None:
    with (args.output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def measure_call(
    *, model: Any, params: dict[str, float], label: str
) -> tuple[Any, dict]:
    gc.collect()
    counts = {event: {"count": 0, "duration_seconds": 0.0} for event in EVENTS}
    lock = threading.Lock()
    listener_seconds = 0.0

    # keyword-only-exempt: library-callback=jax.monitoring.duration_listener
    def listener(event: str, duration_secs: float, **_kwargs: Any) -> None:
        nonlocal listener_seconds
        start = time.perf_counter()
        if event in EVENTS:
            with lock:
                counts[event]["count"] += 1
                counts[event]["duration_seconds"] += duration_secs
        listener_seconds += time.perf_counter() - start

    before = [device.memory_stats() for device in selected]
    emit({"event": "start", "label": label, "allocator": before})
    orchestration = []
    original_compile = backward._compile_all_functions

    def observe_compile(*arguments: Any, **keywords: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_compile(*arguments, **keywords)
        finally:
            orchestration.append(time.perf_counter() - start)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        start, cpu = time.perf_counter(), time.process_time()
        with patch.object(backward, "_compile_all_functions", observe_compile):
            solution = model.solve(params=params, log_level="off")
        synchronize(solution)
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    after = [device.memory_stats() for device in selected]
    record = {
        "event": "complete",
        "label": label,
        "wall_seconds": wall,
        "process_seconds": cpu,
        "compiler_events": dict(counts),
        "compile_orchestration_inclusive_seconds": orchestration,
        "listener_bookkeeping_seconds": listener_seconds,
        "allocator_before": before,
        "allocator_after": after,
        "host_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_scope": (
            "Cumulative process allocator live high-water, includes "
            "initialization and retained first solution; not per-call or "
            "pool reservation"
        ),
    }
    emit(record)
    if label != "cold" and args.family == "continuous":
        assert counts[BACKEND]["count"] == 0, "Nonzero continuous warm backend requests"
    return solution, record


def validate_solution(solution: Any) -> dict[str, Any]:
    arrays = {}
    exact = witness._reference()[0] if args.family == "continuous" else None
    assert solution.values
    if exact is not None:
        assert {(t, r) for t, values in solution.values.items() for r in values} == set(
            exact
        )
        assert len(exact) == 5
    for period, regimes in solution.values.items():
        for regime, value in regimes.items():
            data = np.asarray(value)
            assert np.isfinite(data).all()
            assert {device.id for device in value.sharding.device_set} == set(
                range(args.devices)
            )
            if exact is not None:
                expected = np.asarray(
                    [float(x) for x in exact[period, regime].values()], dtype=data.dtype
                ).reshape(3, 3, 24)
                assert_agrees_to_ulp(got=data, expected=expected, n_ulp=8)
                if args.devices > 1:
                    witness._assert_assets_shards(value)
            arrays[f"{period}/{regime}"] = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                "devices": sorted(d.id for d in value.sharding.device_set),
                "shards": [
                    {
                        "device": shard.device.id,
                        "index": str(shard.index),
                        "shape": shard.data.shape,
                    }
                    for shard in value.addressable_shards
                ],
            }
    assert arrays
    return arrays


parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=("continuous",), default="continuous")
parser.add_argument("--devices", type=int, choices=(1, 3, 4, 6, 8), required=True)
parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
import jax

jax.config.update("jax_enable_x64", args.precision == 64)
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np

import _lcm
import lcm
from _lcm.solution import backward_induction as backward
from tests.conftest import assert_agrees_to_ulp

assert Path(lcm.__file__).resolve().is_relative_to(args.source.resolve())
assert Path(_lcm.__file__).resolve().is_relative_to(args.source.resolve())
assert jax.default_backend() == "gpu"
assert len(jax.devices()) == jax.local_device_count() == 8
assert all("A40" in d.device_kind for d in jax.devices())
selected = jax.devices()[: args.devices]
assert all(
    d.memory_stats() is not None and "peak_bytes_in_use" in d.memory_stats()
    for d in selected
)
EVENTS = (
    "/jax/core/compile/backend_compile_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/jaxpr_trace_duration",
)
BACKEND = EVENTS[0]
if args.family == "continuous":
    import gpu_assets as witness

    assert args.devices == witness.SCALING_DEVICES
    model = witness._model(
        devices=tuple(range(args.devices)), widths=(1, 1), sharded=args.devices > 1
    )
    params = {"discount_factor": 0.5}
emit(
    {
        "event": "identity",
        "family": args.family,
        "precision": args.precision,
        "source": str(args.source),
        "lcm_import": lcm.__file__,
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "pylcm")
        },
        "devices": [
            {"id": d.id, "kind": d.device_kind, "platform": d.platform}
            for d in jax.devices()
        ],
        "settings": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("JAX_", "XLA_", "OMP_", "OPENBLAS_"))
        },
        "precision_flags": {
            "x64": jax.config.jax_enable_x64,
            "matmul": str(jax.config.jax_default_matmul_precision),
        },
        "economic_widths": {"action_product": 1, "cell": 1, "subject": 432}
        if args.family == "continuous"
        else "fixture defaults",
    }
)
first, cold = measure_call(model=model, params=params, label="cold")
assert cold["compiler_events"][BACKEND]["count"] > 0, (
    "Cold listener positive control missing"
)
initial = validate_solution(first)
np.savez(
    args.output / "values.npz",
    **{
        f"{t}_{r}": np.asarray(a)
        for t, rs in first.values.items()
        for r, a in rs.items()
    },
)
samples = [cold]
for index in range(3):
    result, record = measure_call(model=model, params=params, label=f"warm{index + 1}")
    assert validate_solution(result) == initial
    assert validate_solution(first) == initial
    samples.append(record)
    del result
    gc.collect()
assert samples[0]["compiler_events"].get(BACKEND, {}).get("count", 0) > 0, (
    "Cold listener positive control missing"
)
(args.output / "result.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "samples": samples,
            "values": initial,
            "limitations": [
                "Two fresh processes per arm planned; three warm calls per process",
                "Allocator peaks cumulative per isolated process, not phase-reset",
                (
                    "JAX trace/lowering/backend durations can overlap across "
                    "workers; not additive"
                ),
                (
                    "Backend event counts compile-or-get-cached requests; cold "
                    "cache is fresh; orchestration span includes nested "
                    "compilation"
                ),
                (
                    "Simulation semantics measured separately; no simulation "
                    "timing claimed"
                ),
            ],
        },
        indent=2,
    )
    + "\n"
)
