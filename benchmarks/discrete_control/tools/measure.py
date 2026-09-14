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
    if label != "cold":
        assert counts[BACKEND]["count"] == 0, "Nonzero warm backend requests"
    return solution, record


def validate_solution(solution: Any) -> dict[str, Any]:
    """Check the original roster, real device coverage and finite published values."""
    roster = {t: ({"working", "retired"} if t < 4 else {"retired"}) for t in range(5)}
    assert {t: set(regimes) for t, regimes in solution.values.items()} == roster
    records = {}
    for period, regimes in solution.values.items():
        for regime, value in regimes.items():
            data = np.asarray(value)
            assert data.dtype == np.dtype(f"float{args.precision}")
            assert np.isfinite(data).all()
            expected = {0, 1, 2} if regime == "working" else {3}
            assert {device.id for device in value.sharding.device_set} == expected
            coverage = np.zeros(value.shape, dtype=np.int8)
            for shard in value.addressable_shards:
                assert shard.data.size > 0
                coverage[shard.index] += 1
            np.testing.assert_array_equal(coverage, np.ones(value.shape, dtype=np.int8))
            records[f"{period}/{regime}"] = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                "devices": sorted(expected),
                "shards": [
                    {
                        "device": s.device.id,
                        "index": str(s.index),
                        "shape": s.data.shape,
                    }
                    for s in value.addressable_shards
                ],
            }
    return records


parser = argparse.ArgumentParser()
parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
import jax

jax.config.update("jax_enable_x64", args.precision == 64)
jax.config.update("jax_default_matmul_precision", "highest")
import discrete_fixture as witness
import numpy as np

import _lcm
import lcm
from _lcm.solution import backward_induction as backward

assert Path(lcm.__file__).resolve().is_relative_to(args.source / "src")
assert Path(_lcm.__file__).resolve().is_relative_to(args.source / "src")
assert jax.default_backend() == "gpu"
assert len(jax.devices()) == jax.local_device_count() == 4
assert all("A40" in d.device_kind for d in jax.devices())
selected = jax.devices()
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
model = witness._make_three_type_model(
    distributed=True, devices=(0, 1, 2, 3), budget_bytes=134217728
)
params = witness._PARAMS
emit(
    {
        "event": "identity",
        "source": str(args.source),
        "precision": args.precision,
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "pylcm")
        },
        "settings": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("JAX_", "XLA_", "OMP_", "OPENBLAS_"))
        },
    }
)
first, cold = measure_call(model=model, params=params, label="cold")
assert all(cold["compiler_events"][event]["count"] > 0 for event in EVENTS)
assert cold["compile_orchestration_inclusive_seconds"]
assert all(duration > 0 for duration in cold["compile_orchestration_inclusive_seconds"])
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
for repeat in range(3):
    result, sample = measure_call(model=model, params=params, label=f"warm{repeat + 1}")
    assert validate_solution(result) == initial
    assert validate_solution(first) == initial
    samples.append(sample)
    del result
    gc.collect()
(args.output / "result.json").write_text(
    json.dumps({"status": "completed", "samples": samples, "values": initial}, indent=2)
    + "\n"
)
