"""Run one immutable harness against exact base and head GridSearch checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any

from benchmarks.grid_search_pair_scenarios import (
    EXTERNAL_HARNESS_SOURCES,
    SCENARIOS,
    TARGET_SCENARIO_SOURCES,
)

_FULL_REVISION = re.compile(r"[0-9a-f]{40}")
_AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"
_AUTOTUNE_PREFIX = "--xla_gpu_autotune_level"
_ROUTE_TOPOLOGY_FIELDS = (
    "folded_regimes",
    "collective_regimes",
    "distributed_regimes",
    "taste_shock_regimes",
    "gs_vd_regimes",
)
_KERNEL_ROUTE_FIELDS = (
    "regime",
    "period",
    "action_names",
    "action_extents",
    "streamed",
    "collective",
    "has_taste_shocks",
    "fold_state_names",
)
_BEHAVIORAL_ENV_FIELDS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure five GridSearch rows with one external base/head harness."
    )
    parser.add_argument("--base-checkout", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-checkout", type=Path, required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--repeats", type=int, default=1)
    return parser


def _git(*, checkout: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        msg = f"git {' '.join(args)} failed in {checkout}: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(*, root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for raw_relative in sorted(relative_paths):
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source path must be repo-relative: {raw_relative!r}.")
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256_file(root / relative)))
    return digest.hexdigest()


def _validate_checkout(*, checkout: Path, expected_revision: str) -> dict[str, Any]:
    checkout = checkout.resolve()
    if _FULL_REVISION.fullmatch(expected_revision) is None:
        raise ValueError(
            f"Expected a full lowercase revision, got {expected_revision!r}."
        )
    actual_revision = _git(checkout=checkout, args=("rev-parse", "HEAD^{commit}"))
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Checkout {checkout} is {actual_revision}, expected {expected_revision}."
        )
    dirty = _git(
        checkout=checkout,
        args=("status", "--porcelain", "--untracked-files=all"),
    )
    if dirty:
        raise RuntimeError(f"Checkout {checkout} is dirty:\n{dirty}")
    lock = checkout / "pixi.lock"
    if not lock.is_file():
        raise FileNotFoundError(lock)
    scenario_sources = {}
    for relative in TARGET_SCENARIO_SOURCES:
        path = checkout / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        scenario_sources[relative] = _sha256_file(path)
    return {
        "checkout": checkout,
        "revision": actual_revision,
        "lock_digest": _sha256_file(lock),
        "scenario_sources": scenario_sources,
    }


def _assert_pair_identity(*, base: Mapping[str, Any], head: Mapping[str, Any]) -> None:
    if base["revision"] == head["revision"]:
        raise RuntimeError("Base and head revisions must differ.")
    if base["lock_digest"] != head["lock_digest"]:
        raise RuntimeError("Base and head pixi.lock files differ.")
    if base["scenario_sources"] != head["scenario_sources"]:
        mismatched = sorted(
            key
            for key in TARGET_SCENARIO_SOURCES
            if base["scenario_sources"].get(key) != head["scenario_sources"].get(key)
        )
        raise RuntimeError(
            "Target-owned scenario dependencies differ across the pair: "
            f"{mismatched!r}."
        )


def _worker_env(
    *, harness_root: Path, checkout: Path, cache_dir: Path
) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath = [str(harness_root), str(checkout / "src")]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    flags = [
        flag
        for flag in env.get("XLA_FLAGS", "").split()
        if not flag.startswith(_AUTOTUNE_PREFIX)
    ]
    flags.append(_AUTOTUNE_OFF)
    env["XLA_FLAGS"] = " ".join(flags)
    return env


def _pixi_context(
    *, harness_root: Path, environment: Mapping[str, str]
) -> dict[str, str]:
    required = ("PIXI_ENVIRONMENT_NAME", "PIXI_EXE", "PIXI_PROJECT_ROOT")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise RuntimeError(
            "Run the paired controller inside a named pixi environment; missing "
            f"{missing!r}."
        )
    project_root = Path(environment["PIXI_PROJECT_ROOT"]).resolve()
    if project_root != harness_root.resolve():
        raise RuntimeError(
            f"Active pixi project is {project_root}, expected {harness_root.resolve()}."
        )
    pixi_exe = Path(environment["PIXI_EXE"]).resolve()
    if not pixi_exe.is_file():
        raise FileNotFoundError(pixi_exe)
    return {
        "environment_name": environment["PIXI_ENVIRONMENT_NAME"],
        "exe": str(pixi_exe),
        "project_root": str(project_root),
    }


def _worker_command(*, pixi: Mapping[str, str], arguments: Sequence[str]) -> list[str]:
    return [
        pixi["exe"],
        "run",
        "--frozen",
        "-e",
        pixi["environment_name"],
        "python",
        "-m",
        "benchmarks.grid_search_pair_worker",
        *arguments,
    ]


def _run_worker(
    *,
    harness_root: Path,
    harness_revision: str,
    pixi: Mapping[str, str],
    harness_digest: str,
    scenario_digest: str,
    identity: Mapping[str, Any],
    scenario: str,
    precision: str,
    backend: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="pylcm-grid-pair-cache-") as cache:
        worker_arguments = [
            "--checkout",
            str(identity["checkout"]),
            "--expected-revision",
            identity["revision"],
            "--expected-harness-revision",
            harness_revision,
            "--expected-harness-digest",
            harness_digest,
            "--expected-scenario-digest",
            scenario_digest,
            "--expected-lock-digest",
            identity["lock_digest"],
            "--expected-pixi-environment",
            pixi["environment_name"],
            "--expected-pixi-exe",
            pixi["exe"],
            "--expected-pixi-project-root",
            pixi["project_root"],
            "--scenario",
            scenario,
            "--precision",
            precision,
            "--backend",
            backend,
            "--output",
            str(output),
            "--warm-samples",
            "3",
        ]
        command = _worker_command(pixi=pixi, arguments=worker_arguments)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=pixi["project_root"],
            env=_worker_env(
                harness_root=harness_root,
                checkout=identity["checkout"],
                cache_dir=Path(cache),
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".stdout.log").write_text(completed.stdout)
    output.with_suffix(".stderr.log").write_text(completed.stderr)
    if completed.returncode:
        msg = (
            f"Worker failed for {scenario} at {identity['revision']} "
            f"(exit {completed.returncode}); see {output.with_suffix('.stderr.log')}."
        )
        raise RuntimeError(msg)
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file() or not (output / "values.npz").is_file():
        raise RuntimeError(f"Worker omitted required artifacts in {output}.")
    metrics = json.loads(metrics_path.read_text())
    expected = {
        "scenario": scenario,
        "revision": identity["revision"],
        "harness_revision": harness_revision,
        "harness_digest": harness_digest,
        "scenario_digest": scenario_digest,
        "lock_digest": identity["lock_digest"],
        "precision": int(precision),
        "pixi": dict(pixi),
    }
    observed = {key: metrics.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Worker artifact identity mismatch: expected {expected}, got {observed}."
        )
    if len(metrics["timing_ns"]["warm_solve"]) != 3:
        raise RuntimeError("Worker did not return exactly three warm samples.")
    _validate_worker_environment(metrics=metrics, expected_cache_dir=Path(cache))
    _validate_required_evidence(metrics)
    for core in metrics["compiled_cores"]:
        hlo_path = (output / core["hlo_file"]).resolve()
        if not hlo_path.is_relative_to(output.resolve()) or not hlo_path.is_file():
            raise RuntimeError(f"Worker omitted HLO artifact {core['hlo_file']!r}.")
        if _sha256_file(hlo_path) != core["hlo"]["sha256"]:
            raise RuntimeError(f"Worker HLO digest mismatch for {core['hlo_file']!r}.")
    return metrics


def _validate_worker_environment(
    *, metrics: Mapping[str, Any], expected_cache_dir: Path
) -> None:
    environment = metrics["environment"]
    cache = environment.get("JAX_COMPILATION_CACHE_DIR")
    if cache is None or Path(cache).resolve() != expected_cache_dir.resolve():
        raise RuntimeError(
            f"Worker compilation cache is {cache!r}, "
            f"expected {str(expected_cache_dir)!r}."
        )
    if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise RuntimeError("Worker did not disable XLA client preallocation.")
    if environment.get("XLA_PYTHON_CLIENT_MEM_FRACTION") is not None:
        raise RuntimeError("Worker retained XLA_PYTHON_CLIENT_MEM_FRACTION.")
    autotune_flags = [
        flag
        for flag in str(environment.get("XLA_FLAGS", "")).split()
        if flag.startswith(_AUTOTUNE_PREFIX)
    ]
    if autotune_flags != [_AUTOTUNE_OFF]:
        raise RuntimeError("Worker must disable GPU autotuning exactly once.")


def _validate_compiled_evidence(cores: Sequence[Mapping[str, Any]]) -> None:
    if not cores:
        raise RuntimeError("Worker returned no compiled-core/HLO evidence.")
    for core in cores:
        hlo = core["hlo"]
        if hlo["text_bytes"] <= 0 or hlo["instruction_count"] <= 0:
            raise RuntimeError(
                f"Compiled core {core['label']!r} has empty HLO evidence."
            )
        compiler_memory = core["compiler_memory"]
        expected_status = "measured" if compiler_memory is not None else "unavailable"
        if core.get("compiler_memory_status") != expected_status:
            raise RuntimeError("Compiler-memory evidence status is inconsistent.")
        reason = core.get("compiler_memory_reason")
        if (compiler_memory is None) != bool(reason):
            raise RuntimeError("Compiler-memory availability reason is inconsistent.")


def _validate_device_evidence(
    *,
    devices: Sequence[Mapping[str, Any]],
    device_memory: Sequence[Mapping[str, Any]],
) -> None:
    if len(devices) != len(device_memory):
        raise RuntimeError("Device-memory evidence does not cover every device.")
    for device, memory in zip(devices, device_memory, strict=True):
        identity = {key: memory.get(key) for key in ("id", "platform", "kind")}
        if identity != device:
            raise RuntimeError(
                f"Device-memory identity mismatch: {identity!r} != {device!r}."
            )
        platform = device["platform"]
        if platform == "gpu":
            if (
                memory.get("status") != "measured"
                or memory.get("peak_bytes_in_use") is None
                or memory.get("reason") is not None
            ):
                raise RuntimeError(
                    f"GPU device {device['id']} has no measured peak-memory evidence."
                )
        elif platform == "cpu":
            if (
                memory.get("status") != "not_applicable"
                or memory.get("peak_bytes_in_use") is not None
                or not memory.get("reason")
            ):
                raise RuntimeError(
                    f"CPU device {device['id']} lacks an explicit memory N/A record."
                )
        else:
            raise RuntimeError(f"Unsupported measurement platform: {platform!r}.")


def _validate_required_evidence(metrics: Mapping[str, Any]) -> None:
    _validate_compiled_evidence(metrics["compiled_cores"])
    hwm = metrics["memory"]["through_warm"]["hwm_bytes"]
    if hwm is None or hwm <= 0:
        raise RuntimeError("Worker returned no host VmHWM evidence.")
    _validate_device_evidence(
        devices=metrics["devices"],
        device_memory=metrics["memory"]["device"],
    )


def _route_identity(routes: Mapping[str, Any]) -> dict[str, Any]:
    expected_route_keys = {"kernels", "streamed_kernel_count", *_ROUTE_TOPOLOGY_FIELDS}
    if set(routes) != expected_route_keys:
        raise RuntimeError(
            "Unexpected route-evidence schema: "
            f"{sorted(set(routes) ^ expected_route_keys)!r}."
        )
    kernels = []
    expected_kernel_keys = set(_KERNEL_ROUTE_FIELDS)
    for row in routes["kernels"]:
        if set(row) != expected_kernel_keys:
            raise RuntimeError(
                "Unexpected kernel route-evidence schema: "
                f"{sorted(set(row) ^ expected_kernel_keys)!r}."
            )
        if len(row["action_names"]) != len(row["action_extents"]) or any(
            extent < 1 for extent in row["action_extents"]
        ):
            raise RuntimeError("Kernel action names/extents are inconsistent.")
        kernels.append(
            {field: row[field] for field in _KERNEL_ROUTE_FIELDS if field != "streamed"}
        )
    return {
        "kernels": kernels,
        **{field: routes[field] for field in _ROUTE_TOPOLOGY_FIELDS},
    }


def _behavioral_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"JAX_COMPILATION_CACHE_DIR", *_BEHAVIORAL_ENV_FIELDS}
    if set(environment) != expected:
        raise RuntimeError(
            "Unexpected worker-environment schema: "
            f"{sorted(set(environment) ^ expected)!r}."
        )
    return {field: environment[field] for field in _BEHAVIORAL_ENV_FIELDS}


def _compare_measurement_identity(
    *, base: Mapping[str, Any], head: Mapping[str, Any]
) -> dict[str, Any]:
    base_identity = {
        "dimensions": base["dimensions"],
        "routes": _route_identity(base["routes"]),
        "runtime": {
            field: base[field]
            for field in ("python", "jax_version", "jaxlib_version", "jax_enable_x64")
        },
        "devices": base["devices"],
        "behavioral_environment": _behavioral_environment(base["environment"]),
    }
    head_identity = {
        "dimensions": head["dimensions"],
        "routes": _route_identity(head["routes"]),
        "runtime": {
            field: head[field]
            for field in ("python", "jax_version", "jaxlib_version", "jax_enable_x64")
        },
        "devices": head["devices"],
        "behavioral_environment": _behavioral_environment(head["environment"]),
    }
    for field, base_value in base_identity.items():
        if base_value != head_identity[field]:
            raise RuntimeError(
                f"Base/head measurement {field} differs: "
                f"{base_value!r} != {head_identity[field]!r}."
            )
    return base_identity


def _is_scenario_target(
    *, scenario: str, row: Mapping[str, Any], routes: Mapping[str, Any]
) -> bool:
    if scenario == "singleton-ev1":
        return bool(row["has_taste_shocks"])
    if scenario == "folded-hard-max":
        return bool(row["fold_state_names"])
    if scenario == "collective-gs-vd":
        return bool(row["collective"]) and row["regime"] in routes["gs_vd_regimes"]
    if scenario == "distributed-co-map":
        return row["regime"] in routes["distributed_regimes"]
    if scenario == "singleton-hard-max":
        return (
            not row["collective"]
            and not row["has_taste_shocks"]
            and not row["fold_state_names"]
            and row["regime"] not in routes["gs_vd_regimes"]
            and row["regime"] not in routes["distributed_regimes"]
        )
    raise RuntimeError(f"Unknown paired scenario: {scenario!r}.")


def _assert_scenario_streaming_target(
    *, scenario: str, base_routes: Mapping[str, Any], head_routes: Mapping[str, Any]
) -> None:
    for side, routes in (("base", base_routes), ("head", head_routes)):
        observed = sum(bool(row["streamed"]) for row in routes["kernels"])
        if routes["streamed_kernel_count"] != observed:
            raise RuntimeError(f"{side} streamed-kernel count is inconsistent.")
    if base_routes["streamed_kernel_count"] != 0:
        raise RuntimeError("The F base unexpectedly contains streamed kernels.")

    eligible = [
        row
        for row in head_routes["kernels"]
        if row["action_names"]
        and prod(row["action_extents"]) > 1
        and _is_scenario_target(scenario=scenario, row=row, routes=head_routes)
    ]
    if not eligible:
        raise RuntimeError(
            f"Scenario {scenario!r} has no nontrivial named target kernel."
        )
    unstreamed = [
        (row["regime"], row["period"]) for row in eligible if not row["streamed"]
    ]
    if unstreamed:
        raise RuntimeError(
            f"Scenario {scenario!r} left named target kernels dense: {unstreamed!r}."
        )


def _float_tolerances(dtype: Any) -> tuple[float, float]:
    import numpy as np

    if dtype == np.dtype("float32"):
        return 1e-5, 1e-6
    if dtype == np.dtype("float64"):
        return 1e-12, 1e-12
    msg = f"Unsupported floating dtype in parity artifact: {dtype}."
    raise TypeError(msg)


def _max_ulp_distance(*, expected: Any, actual: Any, finite: Any) -> int:
    """Return exact ordered-bit distance over finite float32/float64 leaves."""
    import numpy as np

    if expected.dtype == np.dtype("float32"):
        unsigned_dtype = np.dtype("uint32")
    elif expected.dtype == np.dtype("float64"):
        unsigned_dtype = np.dtype("uint64")
    else:
        msg = f"Unsupported floating dtype in ULP metric: {expected.dtype}."
        raise TypeError(msg)
    if not np.any(finite):
        return 0

    def ordered_bits(values: Any) -> Any:
        # Signed zero is one numeric value. Map the remaining IEEE sign-magnitude
        # encodings into monotonically ordered unsigned integers.
        normalized = np.where(values == 0, values.dtype.type(0), values)
        bits = normalized.view(unsigned_dtype)
        sign = np.array(1 << (8 * unsigned_dtype.itemsize - 1), dtype=unsigned_dtype)
        return np.where(bits & sign, np.bitwise_not(bits), bits | sign)

    expected_bits = ordered_bits(expected[finite])
    actual_bits = ordered_bits(actual[finite])
    distances = np.maximum(expected_bits, actual_bits) - np.minimum(
        expected_bits, actual_bits
    )
    return int(distances.max(initial=0))


def _compare_value_artifacts(*, base_path: Path, head_path: Path) -> dict[str, Any]:
    import numpy as np

    records: list[dict[str, Any]] = []
    with (
        np.load(base_path, allow_pickle=False) as base,
        np.load(head_path, allow_pickle=False) as head,
    ):
        base_keys = set(base.files)
        head_keys = set(head.files)
        if base_keys != head_keys:
            raise AssertionError(
                "Value artifact keys differ: "
                f"base-only={sorted(base_keys - head_keys)!r}, "
                f"head-only={sorted(head_keys - base_keys)!r}."
            )
        for key in sorted(base_keys):
            expected = base[key]
            actual = head[key]
            if expected.shape != actual.shape:
                raise AssertionError(
                    f"{key} shape differs: {expected.shape} != {actual.shape}."
                )
            if expected.dtype != actual.dtype:
                raise AssertionError(
                    f"{key} dtype differs: {expected.dtype} != {actual.dtype}."
                )
            bitwise_equal = expected.tobytes(order="C") == actual.tobytes(order="C")
            if np.issubdtype(expected.dtype, np.inexact):
                rtol, atol = _float_tolerances(expected.dtype)
                parity = bool(
                    np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
                )
                finite = np.isfinite(expected) & np.isfinite(actual)
                absolute = np.abs(expected[finite] - actual[finite])
                denominator = np.maximum(
                    np.abs(expected[finite]), np.finfo(expected.dtype).tiny
                )
                relative = absolute / denominator
                max_abs = float(absolute.max(initial=0.0))
                max_rel = float(relative.max(initial=0.0))
                max_ulp = _max_ulp_distance(
                    expected=expected,
                    actual=actual,
                    finite=finite,
                )
            else:
                parity = bool(np.array_equal(expected, actual))
                rtol = atol = 0.0
                max_abs = max_rel = 0.0
                max_ulp = 0
            records.append(
                {
                    "key": key,
                    "shape": list(expected.shape),
                    "dtype": str(expected.dtype),
                    "bitwise_equal": bitwise_equal,
                    "parity": parity,
                    "rtol": rtol,
                    "atol": atol,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "max_ulp": max_ulp,
                }
            )
            if not parity:
                raise AssertionError(
                    f"{key} violates parity: max_abs={max_abs}, max_rel={max_rel}, "
                    f"rtol={rtol}, atol={atol}."
                )
    return {
        "all_passed": True,
        "all_bitwise_equal": all(record["bitwise_equal"] for record in records),
        "arrays": records,
    }


def _compare_array_manifests(
    *, base: Sequence[Mapping[str, Any]], head: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    base_by_key = {row["key"]: row for row in base}
    head_by_key = {row["key"]: row for row in head}
    if len(base_by_key) != len(base) or len(head_by_key) != len(head):
        raise AssertionError("Array manifests contain duplicate keys.")
    if base_by_key.keys() != head_by_key.keys():
        raise AssertionError(
            "Array manifest keys differ: "
            f"base-only={sorted(base_by_key.keys() - head_by_key.keys())!r}, "
            f"head-only={sorted(head_by_key.keys() - base_by_key.keys())!r}."
        )
    records = []
    for key in sorted(base_by_key):
        expected = base_by_key[key]
        actual = head_by_key[key]
        for field in ("shape", "dtype", "sharding"):
            if expected[field] != actual[field]:
                raise AssertionError(
                    f"{key} {field} differs: {expected[field]!r} != {actual[field]!r}."
                )
        records.append(
            {
                "key": key,
                "shape": expected["shape"],
                "dtype": expected["dtype"],
                "sharding": expected["sharding"],
            }
        )
    return {"all_passed": True, "arrays": records}


def _compiler_peak(*, metrics: Mapping[str, Any], field: str) -> int | None:
    values = [
        core["compiler_memory"][field]
        for core in metrics["compiled_cores"]
        if core["compiler_memory"] is not None
        and core["compiler_memory"].get(field) is not None
    ]
    return max(values) if values else None


def _device_peak(metrics: Mapping[str, Any]) -> int | None:
    values = [
        row["peak_bytes_in_use"]
        for row in metrics["memory"]["device"]
        if row["peak_bytes_in_use"] is not None
    ]
    return max(values) if values else None


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    compile_calls = metrics["timing_ns"]["aot_compile_calls"]
    if len(compile_calls) != 4:
        raise RuntimeError(
            "Expected one cold and three warm AOT orchestration calls, got "
            f"{len(compile_calls)}."
        )
    warm = metrics["timing_ns"]["warm_solve"]
    return {
        "cold_solve_ns": metrics["timing_ns"]["cold_solve"],
        "cold_aot_compile_ns": compile_calls[0],
        "warm_solve_ns": warm,
        "warm_solve_median_ns": int(statistics.median(warm)),
        "rss_hwm_through_warm_bytes": metrics["memory"]["through_warm"]["hwm_bytes"],
        "compiler_peak_bytes": _compiler_peak(
            metrics=metrics, field="peak_memory_in_bytes"
        ),
        "compiler_temp_bytes": _compiler_peak(
            metrics=metrics, field="temp_size_in_bytes"
        ),
        "device_peak_bytes": _device_peak(metrics),
        "hlo_text_bytes": sum(
            core["hlo"]["text_bytes"] for core in metrics["compiled_cores"]
        ),
        "hlo_instruction_count": sum(
            core["hlo"]["instruction_count"] for core in metrics["compiled_cores"]
        ),
        "communication_collective_count": sum(
            core["hlo"]["communication_collective_count"]
            for core in metrics["compiled_cores"]
        ),
        "streamed_kernel_count": metrics["routes"]["streamed_kernel_count"],
        "tile_plans": metrics["tile_plans"],
    }


def _ratio(*, head: int | None, base: int | None) -> float | None:
    if head is None or base in (None, 0):
        return None
    return head / base


def _pair_summary(
    *, base: Mapping[str, Any], head: Mapping[str, Any]
) -> dict[str, Any]:
    if base["streamed_kernel_count"] != 0:
        raise RuntimeError(
            "The F base unexpectedly contains streamed GridSearch kernels."
        )
    if head["streamed_kernel_count"] <= 0:
        raise RuntimeError("The F head contains no streamed GridSearch kernel.")
    keys = (
        "cold_solve_ns",
        "cold_aot_compile_ns",
        "warm_solve_median_ns",
        "rss_hwm_through_warm_bytes",
        "compiler_peak_bytes",
        "compiler_temp_bytes",
        "device_peak_bytes",
        "hlo_text_bytes",
        "hlo_instruction_count",
    )
    return {
        "base": dict(base),
        "head": dict(head),
        "head_over_base": {key: _ratio(head=head[key], base=base[key]) for key in keys},
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    harness_root = Path(__file__).resolve().parent.parent
    pixi = _pixi_context(harness_root=harness_root, environment=os.environ)
    harness = _validate_checkout(
        checkout=harness_root,
        expected_revision=args.head_revision,
    )
    harness_digest = _sha256_files(
        root=harness_root,
        relative_paths=EXTERNAL_HARNESS_SOURCES,
    )
    scenario_digest = _sha256_file(
        harness_root / "benchmarks/grid_search_pair_scenarios.py"
    )
    base = _validate_checkout(
        checkout=args.base_checkout,
        expected_revision=args.base_revision,
    )
    head = _validate_checkout(
        checkout=args.head_checkout,
        expected_revision=args.head_revision,
    )
    _assert_pair_identity(base=base, head=head)
    if (
        harness["lock_digest"] != head["lock_digest"]
        or harness["scenario_sources"] != head["scenario_sources"]
    ):
        raise RuntimeError("External harness checkout does not match the exact head.")
    for checkout in (harness_root, base["checkout"], head["checkout"]):
        if output.is_relative_to(checkout):
            raise ValueError(
                "Write paired artifacts outside the harness, base, and head checkouts."
            )
    output.mkdir(parents=True)

    raw: dict[tuple[str, int, str], dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    identities = {"base": base, "head": head}
    for repeat in range(args.repeats):
        order = ("base", "head") if repeat % 2 == 0 else ("head", "base")
        for scenario in SCENARIOS:
            for side in order:
                run_output = output / scenario / f"repeat-{repeat}" / side
                raw[(scenario, repeat, side)] = _run_worker(
                    harness_root=harness_root,
                    harness_revision=harness["revision"],
                    pixi=pixi,
                    harness_digest=harness_digest,
                    scenario_digest=scenario_digest,
                    identity=identities[side],
                    scenario=scenario,
                    precision=args.precision,
                    backend=args.backend,
                    output=run_output,
                )
            base_output = output / scenario / f"repeat-{repeat}" / "base"
            head_output = output / scenario / f"repeat-{repeat}" / "head"
            measurement_identity = _compare_measurement_identity(
                base=raw[(scenario, repeat, "base")],
                head=raw[(scenario, repeat, "head")],
            )
            _assert_scenario_streaming_target(
                scenario=scenario,
                base_routes=raw[(scenario, repeat, "base")]["routes"],
                head_routes=raw[(scenario, repeat, "head")]["routes"],
            )
            base_metrics = _metric_summary(raw[(scenario, repeat, "base")])
            head_metrics = _metric_summary(raw[(scenario, repeat, "head")])
            if (
                scenario == "distributed-co-map"
                and head_metrics["communication_collective_count"] != 0
            ):
                raise RuntimeError(
                    "Distributed co-map head HLO contains a communication collective."
                )
            pairs.append(
                {
                    "scenario": scenario,
                    "repeat": repeat,
                    "measurement_identity": measurement_identity,
                    "parity": _compare_value_artifacts(
                        base_path=base_output / "values.npz",
                        head_path=head_output / "values.npz",
                    ),
                    "layout_contract": _compare_array_manifests(
                        base=raw[(scenario, repeat, "base")]["arrays"],
                        head=raw[(scenario, repeat, "head")]["arrays"],
                    ),
                    "metrics": _pair_summary(
                        base=base_metrics,
                        head=head_metrics,
                    ),
                }
            )

    summary = {
        "schema_version": "1.0",
        "harness_digest": harness_digest,
        "harness_revision": harness["revision"],
        "scenario_digest": scenario_digest,
        "base_revision": base["revision"],
        "head_revision": head["revision"],
        "precision": int(args.precision),
        "requested_backend": args.backend,
        "pixi": pixi,
        "repeats": args.repeats,
        "execution_order": "AB on even repeats, BA on odd repeats",
        "pairs": pairs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
