"""Expose production component-job phases without initializing a device backend."""

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

_EXCLUSIVITY = None

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from lcm import Model, SimulationResult
    from lcm.component_jobs import CollectedComponentJobs, ComponentJobPlan
    from lcm.solver_api import SolutionResult
    from lcm.typing import UserParams


def main() -> None:  # noqa: PLR0915 - phases share one admission/failure boundary
    """Route component phases through the complete owner model builder."""
    args = parse_args()
    from datetime import UTC, datetime
    from time import perf_counter

    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    out = args.out / f"job-{args.job:04d}" if args.phase == "run" else args.out
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Phase output must be absent or empty")

    try:
        import dataclasses
        import sys

        sys.path.insert(0, str(args.aca_slurm_src))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage5b"))
        runtime = _native_preflight(aca_slurm_src=args.aca_slurm_src)
        contract = _production_inputs(aca_slurm_src=args.aca_slurm_src)
        from aca_slurm.config import make_execution_config
        from stage3_arms import _build

        from lcm import InvariantBlockSchedule

        execution = dataclasses.replace(
            make_execution_config(solver="brute_force", continuous_sharding=True),
            invariant_block_widths={"pref_type": 1},
            invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
            device_memory_bytes=None,
        )
        model, params, initial_conditions, _description = _build(
            workload="production",
            aca_slurm_src=args.aca_slurm_src,
            n_subjects=1,
            production_execution_config=execution,
        )
        from lcm.component_jobs import _initial_conditions_sha256

        raw_input_sha256 = _initial_conditions_sha256(
            initial_conditions=initial_conditions
        )
        provenance = {**dict(_description), "contract": contract, "runtime": runtime}

        def verify_inputs() -> None:
            """Recheck immutable source and input files before a successful receipt."""
            from _lcm.solution.component_fragments import canonical_json

            if canonical_json(
                _production_inputs(aca_slurm_src=args.aca_slurm_src)
            ) != canonical_json(contract):
                raise RuntimeError(  # noqa: TRY301 - preserve the shared phase failure receipt
                    "Production source or input files changed during the phase"
                )

        if args.phase in {"run", "collect"}:
            from lcm.component_jobs import load_component_job_plan

            bound_plan = load_component_job_plan(directory=args.plan_directory)
            _dense, original_ids = _canonical_population(
                model=model,
                initial_conditions=initial_conditions,
            )
            _validate_planned_ids(
                plan=bound_plan,
                receipt=args.planning_receipt,
                receipt_sha256=args.planning_receipt_sha256,
                original_ids=original_ids,
                production_contract=contract,
            )
            del _dense
        if args.phase == "plan":
            plan = _plan(
                model=model,
                params=params,
                initial_conditions=initial_conditions,
                directory=args.plan_directory,
            )
            dense, original_ids = _canonical_population(
                model=model,
                initial_conditions=initial_conditions,
            )
            verify_inputs()
            _publish_planned_receipt(
                out=out,
                plan=plan,
                original_ids=original_ids,
                raw_input_count=len(initial_conditions),
                canonical_input_count=len(dense),
                provenance=provenance,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=perf_counter() - started,
                raw_input_sha256=raw_input_sha256,
            )
        elif args.phase == "run":
            fragment, observation = _observe_call(
                call=lambda: _run(
                    model=model,
                    params=params,
                    initial_conditions=initial_conditions,
                    directory=args.plan_directory,
                    job=args.job,
                ),
                out=out,
                label="worker_call",
                gpu_uuids=runtime.get("gpu_uuids", []),
            )
            verify_inputs()
            from _lcm.solution.component_fragments import write_json_atomically

            out.mkdir(parents=True, exist_ok=True)
            write_json_atomically(
                path=out / "receipt.json",
                payload={
                    "format": "aca-stage8a-phase-receipt",
                    "format_version": 1,
                    "phase": "run",
                    "status": "completed",
                    "job": args.job,
                    "fragment": str(fragment),
                    "provenance": provenance,
                    "planning_receipt_sha256": args.planning_receipt_sha256,
                    "fragment_sha256": _file_sha256(path=fragment),
                    "observations": observation,
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": perf_counter() - started,
                },
            )
        elif args.phase == "reference":
            plan = _plan(
                model=model,
                params=params,
                initial_conditions=initial_conditions,
                directory=args.plan_directory,
            )
            (simulation, original_ids), _observation = _observe_call(
                call=lambda: _reference(
                    model=model,
                    params=params,
                    initial_conditions=initial_conditions,
                ),
                out=out,
                label="reference_call",
                gpu_uuids=runtime.get("gpu_uuids", []),
            )
            verify_inputs()
            _publish_reference(
                plan=plan,
                solution=simulation.solution,
                simulation=simulation,
                original_ids=original_ids,
                out=out,
                provenance=provenance,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=perf_counter() - started,
            )
        elif args.phase == "collect":
            worker_admission = _validate_worker_receipts(
                directory=args.worker_receipts,
                plan=bound_plan,
                planning_receipt_sha256=args.planning_receipt_sha256,
                production_contract=contract,
            )
            collected, _collection_observation = _observe_call(
                call=lambda: _collect(
                    model=model, params=params, directory=args.plan_directory
                ),
                out=out,
                label="collection_call",
                gpu_uuids=runtime.get("gpu_uuids", []),
            )
            _unused, _comparison_observation = _observe_call(
                call=lambda: _compare_reference(
                    collected=collected,
                    original_ids=original_ids,
                    reference=args.reference,
                    reference_receipt_sha256=args.reference_receipt_sha256,
                    production_contract=contract,
                ),
                out=out,
                label="comparison_call",
                gpu_uuids=runtime.get("gpu_uuids", []),
            )
            verify_inputs()
            _publish_reference(
                plan=collected.plan,
                solution=collected.solution,
                simulation=collected.simulation,
                original_ids=original_ids,
                out=out,
                phase="collect",
                provenance={**provenance, "worker_admission": worker_admission},
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=perf_counter() - started,
            )
    except Exception as error:
        import contextlib

        from _lcm.solution.component_fragments import write_json_atomically

        with contextlib.suppress(OSError):
            out.mkdir(parents=True, exist_ok=True)
            write_json_atomically(
                path=out / "receipt.failed.json",
                payload={
                    "format": "aca-stage8a-phase-receipt",
                    "format_version": 1,
                    "phase": args.phase,
                    "status": "failed",
                    **({"job": args.job} if args.phase == "run" else {}),
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        raise


def _numerical_intervals(*, path: Path) -> list[dict[str, object]]:
    """Read absolute numeric intervals with the maintained engine phase grammar."""
    from benchmarks.warm_solve_phases import _RECORD

    opened: dict[tuple[str, str], list[float]] = {}
    intervals = []
    for line in path.read_text().splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        stamp, _level, message = parts
        match = _RECORD.match(message)
        if match is None or match["name"] not in {
            "backward_induction",
            "simulation_chunk",
        }:
            continue
        key = (match["call"], match["name"])
        if match["edge"] == "begin":
            opened.setdefault(key, []).append(float(stamp))
        elif opened.get(key):
            begin = opened[key].pop(0)
            if match["status"] == "ok":
                intervals.append(
                    {
                        "call_id": key[0],
                        "phase": key[1],
                        "begin_epoch": begin,
                        "end_epoch": float(stamp),
                    }
                )
    return intervals


def _validate_worker_receipts(
    *,
    directory: Path,
    plan: ComponentJobPlan,
    planning_receipt_sha256: str,
    production_contract: Mapping[str, object],
) -> dict[str, object]:
    """Require source-matched distinct workers and observed numeric interval overlap."""
    import json
    import math

    from _lcm.solution.component_fragments import (
        FRAGMENT_DIRECTORY,
        canonical_json,
        fragment_name,
    )

    message = "Production worker receipts do not prove the matched three-node campaign"
    hosts = []
    allocations = []
    uuids = []
    events = []
    hashes = []
    for job in range(3):
        out = directory / f"job-{job:04d}"
        receipt = out / "receipt.json"
        if (out / "receipt.failed.json").exists() or not receipt.is_file():
            raise ValueError(message)
        record = json.loads(receipt.read_bytes())
        header = {
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": "run",
            "status": "completed",
            "job": job,
            "planning_receipt_sha256": planning_receipt_sha256,
        }
        fragment = plan.directory / FRAGMENT_DIRECTORY / fragment_name(job=job)
        if (
            canonical_json({key: record.get(key) for key in header})
            != canonical_json(header)
            or record.get("fragment") != str(fragment)
            or record.get("fragment_sha256") != _file_sha256(path=fragment)
            or canonical_json(record.get("provenance", {}).get("contract"))
            != canonical_json(production_contract)
        ):
            raise ValueError(message)
        observed = record["observations"]
        allocation = observed["allocation"]
        intervals = observed.get("numeric_intervals", [])
        if (
            allocation.get("SLURM_PROCID") != str(job)
            or allocation.get("SLURM_JOB_PARTITION") != "mlgpu_short"
            or not allocation.get("SLURM_JOB_ID")
            or not allocation.get("SLURM_STEP_ID")
            or not observed.get("host")
            or observed.get("gpu_exclusivity", {}).get("exclusive") is not True
            or observed.get("sampler_failed") is not False
            or len(observed.get("gpu_uuids", [])) != 8
            or not intervals
        ):
            raise ValueError(message)
        hosts.append(observed["host"])
        allocations.append((allocation["SLURM_JOB_ID"], allocation["SLURM_STEP_ID"]))
        uuids.extend(observed["gpu_uuids"])
        for interval in intervals:
            begin, end = interval["begin_epoch"], interval["end_epoch"]
            if (
                interval.get("phase") not in {"backward_induction", "simulation_chunk"}
                or type(begin) not in {int, float}
                or type(end) not in {int, float}
                or not math.isfinite(begin)
                or not math.isfinite(end)
                or begin >= end
            ):
                raise ValueError(message)
            events.extend(((begin, 1, job), (end, -1, job)))
        hashes.append(_file_sha256(path=receipt))
    active = [0, 0, 0]
    overlap = None
    ordered = sorted(events)
    for position, (stamp, delta, job) in enumerate(ordered):
        active[job] += delta
        if (
            all(active)
            and position + 1 < len(ordered)
            and stamp < ordered[position + 1][0]
        ):
            overlap = [stamp, ordered[position + 1][0]]
            break
    if (
        overlap is None
        or len(set(hosts)) != 3
        or len(set(allocations)) != 1
        or len(set(uuids)) != 24
    ):
        raise ValueError(message)
    return {
        "hosts": hosts,
        "allocation": allocations[0],
        "gpu_uuids": uuids,
        "numeric_overlap_epoch": overlap,
        "worker_receipt_sha256": hashes,
    }


# Keep primary-error preservation and all owned-resource cleanup in one scope.
def _observe_call[T](  # noqa: C901, PLR0912, PLR0915
    *,
    call: Callable[[], T],
    out: Path,
    label: str,
    gpu_uuids: list[str],
) -> tuple[T, dict[str, object]]:
    """Bracket one actual call with existing compile, log and resource observers."""
    import contextlib
    import dataclasses
    import logging
    import resource
    import socket
    import subprocess
    from datetime import UTC, datetime
    from time import perf_counter

    import jax
    from stage3_arms import _OWN_UUIDS, _Exclusivity, _Lines, _MemStats

    from _lcm.solution.component_fragments import write_json_atomically
    from benchmarks.asv._compile_counters import count_compile_requests
    from benchmarks.warm_solve_phases import parse_phase_records

    out.mkdir(parents=True, exist_ok=True)
    global _EXCLUSIVITY  # noqa: PLW0603 - one inherited daemon per process
    _OWN_UUIDS.update(gpu_uuids)
    subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runner = subprocess.run(
        ["pgrep", "-a", "-x", "Runner.Worker"],
        check=False,
        capture_output=True,
        text=True,
    )
    if runner.returncode not in {0, 1}:
        raise RuntimeError("GPU exclusivity probe is unavailable")
    if _EXCLUSIVITY is None:
        _EXCLUSIVITY = _Exclusivity()
    exclusivity = _EXCLUSIVITY
    if exclusivity.at_start:
        raise RuntimeError("GPU allocation is not exclusive")
    logger = logging.getLogger("lcm")
    handler = logging.FileHandler(out / f"{label}.log")
    handler.setFormatter(logging.Formatter("%(created).9f %(levelname)s %(message)s"))
    lines = _Lines()
    memory = _MemStats(out / f"{label}.memstats.jsonl")
    memory.label = label
    handlers = (handler, lines, memory)
    for observer in handlers:
        logger.addHandler(observer)
    record: dict[str, object] = {
        "label": label,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "allocation": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_JOB_PARTITION",
            )
        },
        "gpu_uuids": gpu_uuids,
        "started_at": datetime.now(UTC).isoformat(),
        "debug_core_records": None,
        "status": "failed",
    }
    begin = perf_counter()
    call_begin = None
    sampler = None
    counts = None
    error = None
    with (out / f"{label}.nvml.csv").open("w") as output:
        try:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    ",".join(gpu_uuids),
                    "--query-gpu=timestamp,uuid,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
            )
            if sampler.poll() is not None:
                raise RuntimeError("GPU sampler failed before the call")  # noqa: TRY301
            with count_compile_requests() as counts:
                call_begin = perf_counter()
                record["call_started_at"] = datetime.now(UTC).isoformat()
                result = call()
                for array in jax.live_arrays():
                    array.block_until_ready()
            record["status"] = "completed"
        except Exception as current:
            error = current
            record["error"] = {"type": type(current).__name__, "message": str(current)}
            raise
        finally:
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["call_wall_seconds"] = (
                None if call_begin is None else perf_counter() - call_begin
            )
            record["observation_wall_seconds"] = perf_counter() - begin
            for observer in handlers:
                logger.removeHandler(observer)
                observer.close()
            if sampler is not None:
                sampler_failed = sampler.poll() is not None
                with contextlib.suppress(ProcessLookupError):
                    sampler.terminate()
                try:
                    sampler.wait(timeout=30)
                    if sampler.stderr is not None:
                        record["sampler_stderr"] = sampler.stderr.read()
                        sampler.stderr.close()
                except Exception as cleanup_error:
                    record["cleanup_error"] = str(cleanup_error)
                    if error is None:
                        raise
                record["sampler_failed"] = sampler_failed
            record["compile_requests"] = (
                None if counts is None else dataclasses.asdict(counts)
            )
            record["phases"] = [
                dataclasses.asdict(phase)
                for phase in parse_phase_records(lines=lines.lines)
            ]
            try:
                record["numeric_intervals"] = _numerical_intervals(
                    path=out / f"{label}.log"
                )
            except Exception as cleanup_error:
                record["numeric_intervals"] = None
                record["interval_error"] = str(cleanup_error)
                if error is None:
                    raise
            try:
                record["allocator_after"] = [
                    device.memory_stats() for device in jax.local_devices()
                ]
            except Exception as cleanup_error:
                record["allocator_after"] = None
                record["allocator_error"] = str(cleanup_error)
                if error is None:
                    raise
            record["host_maxrss_kib"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            record["gpu_exclusivity"] = exclusivity.record()
            try:
                write_json_atomically(
                    path=out / f"{label}.observations.json", payload=record
                )
            except Exception:
                if error is None:
                    raise
            if error is None and (
                record.get("sampler_failed")
                or not record["gpu_exclusivity"]["exclusive"]
            ):
                raise RuntimeError(
                    "GPU telemetry or exclusivity failed during the call"
                )
    return result, record


# Preserve distinct source, payload and allocation refusal boundaries.
def _native_preflight(*, aca_slurm_src: Path) -> dict[str, Any]:  # noqa: C901
    """Require installed native readiness and the actual eight-A40 fp32 boundary."""
    import dataclasses
    import importlib.metadata
    import subprocess
    import sys

    import jax

    import hatch_build
    import lcm
    from _lcm.egm.upper_envelope._exact_affine import ffi
    from tests.ci.probe_native import probe

    root = Path(os.environ["PYLCM_DIR"]).resolve()
    if Path(lcm.__file__).resolve() != root / "src/lcm/__init__.py":
        raise RuntimeError("Stage 8A imported another pylcm source")
    native = probe(root=root)
    if native.exit_code != 0 or native.status != "ready":
        raise RuntimeError(f"Stage 8A native payload is not READY: {native.status}")
    installed_version = importlib.metadata.version("pylcm")
    if installed_version != lcm.__version__:
        raise RuntimeError("Stage 8A installed and source pylcm versions differ")
    if jax.default_backend() != "gpu":
        raise RuntimeError("Stage 8A requires an actual GPU backend")
    if Path(sys.prefix).resolve() != root / ".pixi/envs/benchmarks-cuda12":
        raise RuntimeError(
            "Stage 8A requires its preinstalled benchmarks-cuda12 prefix"
        )
    if jax.config.jax_enable_x64:
        raise RuntimeError("Stage 8A requires fp32")
    devices = jax.local_devices()
    if (
        len(devices) != 8
        or len({device.id for device in devices}) != 8
        or any(device.device_kind != "NVIDIA A40" for device in devices)
    ):
        raise RuntimeError("Stage 8A requires eight distinct local A40 devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = visible.split(",")
    if len(selected) != 8 or len(set(selected)) != 8 or not all(selected):
        raise RuntimeError("Stage 8A requires eight distinct visible A40 GPU UUIDs")
    output = subprocess.run(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = [
        tuple(part.strip() for part in line.split(",")) for line in output.splitlines()
    ]
    if (
        len(rows) != 8
        or any(len(row) != 2 for row in rows)
        or len({row[0] for row in rows}) != 8
        or any(not row[0].startswith("GPU-") or row[1] != "NVIDIA A40" for row in rows)
    ):
        raise RuntimeError("Stage 8A requires eight distinct visible A40 GPU UUIDs")
    cuda_library = ffi._CUDA_LIBRARY  # noqa: SLF001 - maintained native authority
    if cuda_library is None:
        raise RuntimeError("Stage 8A CUDA native library is unavailable")
    return {
        "source_version": lcm.__version__,
        "installed_version": installed_version,
        "native": dataclasses.asdict(native),
        "native_library": str(cuda_library),
        "native_library_sha256": _file_sha256(path=cuda_library),
        "native_manifest_sha256": _file_sha256(
            path=ffi._DIRECTORY / hatch_build.NATIVE_MANIFEST  # noqa: SLF001
        ),
        "pylcm_source": str(root),
        "aca_slurm_src": str(aca_slurm_src),
        "prefix": sys.prefix,
        "backend": "gpu",
        "precision": "fp32",
        "device_ids": [device.id for device in devices],
        "gpu_uuids": [row[0] for row in rows],
    }


def _production_inputs(*, aca_slurm_src: Path) -> dict[str, object]:
    """Authenticate the maintained production loader, owner sources and input files."""
    import subprocess
    import sys

    import aca_model.simulation
    import jax
    import jaxlib
    from aca_slurm import _simulate as production_io
    from aca_slurm._simulate import _production_input_paths

    from _lcm.egm.upper_envelope._exact_affine import ffi

    roots = {
        name: Path(os.environ[f"{name}_DIR"]).resolve()
        for name in ("PYLCM", "ACA_MODEL", "ACA_SLURM")
    }
    if (
        aca_slurm_src.resolve() != roots["ACA_SLURM"] / "src"
        or Path(production_io.__file__).resolve()
        != aca_slurm_src.resolve() / "aca_slurm/_simulate.py"
        or Path(aca_model.simulation.__file__).resolve()
        != roots["ACA_MODEL"] / "src/aca_model/simulation.py"
        or os.environ["ACA_MODEL_COMMIT"] != "ad38653696ec366e318ac61b9a81b597a4ecb700"
    ):
        raise RuntimeError("Stage 8A imported another frozen ACA source")
    commits = {}
    for name, root in roots.items():
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if head != os.environ[f"{name}_COMMIT"] or dirty:
            raise RuntimeError("Stage 8A requires clean exact owner source commits")
        commits[name] = head
    driver = Path(__file__).resolve()
    helper = driver.parents[1] / "stage5b/stage3_arms.py"
    files = {
        "lock": (roots["PYLCM"] / "pixi.lock", "PIXI_LOCK_SHA256"),
        "driver": (driver, "DRIVER_SHA256"),
        "builder": (helper, "BENCHMARK_HELPER_SHA256"),
    }
    hashes = {}
    for name, (path, variable) in files.items():
        digest = _file_sha256(path=path)
        if digest != os.environ[variable]:
            raise RuntimeError("Stage 8A source or frozen lock hash differs")
        hashes[name] = digest
    paths = _production_input_paths()
    if len(paths) != 11:
        raise RuntimeError("Stage 8A requires all eleven maintained production inputs")
    # The maintained payload module is the installed-library authority.
    library = (
        ffi._CUDA_LIBRARY if jax.default_backend() == "gpu" else ffi._CPU_LIBRARY  # noqa: SLF001
    )
    if library is None:
        raise RuntimeError("Stage 8A native library is unavailable")
    return {
        "commits": commits,
        "source_hashes": hashes,
        "imports": {
            "aca_model": str(aca_model.simulation.__file__),
            "aca_slurm": str(production_io.__file__),
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": _file_sha256(path=path)}
            for key, path in paths.items()
        },
        "native_library_sha256": _file_sha256(path=library),
        "prefix": str(Path(sys.prefix).resolve()),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "precision": "fp32",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one full-production ACA phase without importing its runtime."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan-directory", type=Path, required=True)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--aca-slurm-src", type=Path, required=True)
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("plan", parents=[common])
    worker = phases.add_parser("run", parents=[common])
    worker.add_argument("--job-from-slurm-procid", action="store_true", required=True)
    worker.add_argument("--planning-receipt", type=Path, required=True)
    worker.add_argument("--planning-receipt-sha256", required=True)
    collector = phases.add_parser("collect", parents=[common])
    collector.add_argument("--planning-receipt", type=Path, required=True)
    collector.add_argument("--planning-receipt-sha256", required=True)
    collector.add_argument("--reference", type=Path, required=True)
    collector.add_argument("--reference-receipt-sha256", required=True)
    collector.add_argument("--worker-receipts", type=Path, required=True)
    phases.add_parser("reference", parents=[common])
    args = parser.parse_args(argv)
    for name in ("planning_receipt_sha256", "reference_receipt_sha256"):
        digest = getattr(args, name, None)
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            parser.error(
                f"--{name.replace('_', '-')} must have 64 lowercase "
                "hexadecimal characters"
            )
    if args.phase == "run":
        process_id = os.environ.get("SLURM_PROCID")
        if process_id not in {"0", "1", "2"}:
            parser.error("SLURM_PROCID must be 0, 1, or 2")
        args.job = int(process_id)
    return args


def _canonical_population(
    *, model: Model, initial_conditions: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep initial-regime state columns for every owner-admitted row and ID."""
    import numpy as np
    from aca_model.simulation import select_admissible_starts

    admitted = select_admissible_starts(
        model=model, initial_conditions=initial_conditions
    )
    original_ids = np.asarray(admitted.index).copy()
    columns = {"age", "regime_name"} | {
        name
        for _age, regime_name in model.initial_nodes
        for name in model.user_regimes[regime_name].states
    }
    projected = admitted.loc[:, [name for name in admitted.columns if name in columns]]
    return projected.reset_index(drop=True), original_ids


def _plan(
    *,
    model: Model,
    params: UserParams,
    initial_conditions: pd.DataFrame,
    directory: Path,
) -> ComponentJobPlan:
    """Plan all original preference codes with the canonical owner population."""
    from aca_slurm.config import SIMULATION_SEED

    from lcm.component_jobs import plan_component_jobs

    dense, _original_ids = _canonical_population(
        model=model, initial_conditions=initial_conditions
    )
    return plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        assignment=((0,), (1,), (2,)),
        initial_conditions=dense,
        seed=SIMULATION_SEED,
    )


def _run(
    *,
    model: Model,
    params: UserParams,
    initial_conditions: pd.DataFrame,
    directory: Path,
    job: int,
) -> Path:
    """Forward the whole canonical population to its planned component worker."""
    from lcm.component_jobs import load_component_job_plan, run_component_job

    load_component_job_plan(directory=directory)
    dense, _original_ids = _canonical_population(
        model=model, initial_conditions=initial_conditions
    )
    return run_component_job(
        model=model,
        params=params,
        directory=directory,
        job=job,
        initial_conditions=dense,
        log_level="progress",
    )


def _reference(
    *, model: Model, params: UserParams, initial_conditions: pd.DataFrame
) -> tuple[SimulationResult, np.ndarray]:
    """Run the owner adapter on the same canonical rows and external IDs."""
    from aca_model.simulation import simulate_with_dense_index
    from aca_slurm.config import SIMULATION_SEED

    dense, original_ids = _canonical_population(
        model=model, initial_conditions=initial_conditions
    )
    admitted = dense.set_axis(original_ids)
    admitted.index.name = initial_conditions.index.name
    return simulate_with_dense_index(
        model=model,
        params=params,
        initial_conditions=admitted,
        seed=SIMULATION_SEED,
        log_level="progress",
    )


def _collect(
    *, model: Model, params: UserParams, directory: Path
) -> CollectedComponentJobs:
    """Preserve the engine's complete solution and simulation collection."""
    from lcm.component_jobs import collect_component_jobs

    return collect_component_jobs(
        model=model, params=params, directory=directory, log_level="progress"
    )


def _publish_planned_receipt(
    *,
    out: Path,
    plan: ComponentJobPlan,
    original_ids: np.ndarray,
    raw_input_count: int,
    canonical_input_count: int,
    provenance: Mapping[str, object],
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    raw_input_sha256: str | None = None,
) -> Path:
    """Publish the plan record and its separately bound original subject IDs."""
    import numpy as np

    from _lcm.solution.component_fragments import (
        array_checksum,
        write_atomically,
        write_json_atomically,
    )

    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "original_ids.npy"

    def write_ids(path: Path) -> None:
        """Write only the original-ID array without Python object serialization."""
        with path.open("wb") as handle:
            np.save(handle, original_ids, allow_pickle=False)

    write_atomically(path=ids_path, write=write_ids)
    receipt = out / "receipt.json"
    write_json_atomically(
        path=receipt,
        payload={
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": "plan",
            "status": "planned",
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.digest,
            "state_name": plan.state_name,
            "codes": list(plan.codes),
            "jobs": [list(job) for job in plan.jobs],
            "seed": plan.seed,
            "raw_input_count": raw_input_count,
            "raw_input_sha256": raw_input_sha256,
            "canonical_input_count": canonical_input_count,
            "initial_conditions_sha256": plan.initial_conditions_sha256,
            "provenance": dict(provenance),
            "artifacts": {
                "original_ids.npy": {
                    "sha256": _file_sha256(path=ids_path),
                    "array_checksum": array_checksum(
                        identity={"field": "original_subject_ids"},
                        array=original_ids,
                    ),
                    "dtype": original_ids.dtype.str,
                    "shape": list(original_ids.shape),
                },
            },
        },
    )
    return receipt


def _publish_reference(
    *,
    plan: ComponentJobPlan,
    solution: object,
    simulation: SimulationResult | None,
    original_ids: np.ndarray,
    out: Path,
    provenance: Mapping[str, object],
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    phase: str = "reference",
) -> Path:
    """Save complete public archives before publishing their immutable receipt."""
    from datetime import UTC, datetime
    from time import perf_counter

    import numpy as np

    from _lcm.solution.component_fragments import (
        array_checksum,
        write_atomically,
        write_json_atomically,
    )
    from lcm.solver_api import SolutionResult

    if not isinstance(solution, SolutionResult) or simulation is None:
        raise ValueError("Full production results require a solution and simulation")

    if any((out / name).exists() for name in ("receipt.json", "receipt.failed.json")):
        raise FileExistsError("Reference output has a previous phase receipt")
    archive_started = perf_counter()
    retention = {
        "retained_continuations": list(solution.retained_continuations),
        "replay_artifacts": list(solution.replay_artifacts),
        "auxiliary_artifacts": list(solution.auxiliary_artifacts),
        "diagnostics": list(solution.diagnostics),
        "omissions": dict(solution.omissions),
    }
    if any(retention.values()):
        raise ValueError(
            "Production archive requires the complete values-only GridSearch target"
        )
    value_catalog = _value_catalog(solution=solution)
    raw_coordinates, raw_schemas, raw_catalog = _raw_catalog(simulation=simulation)
    out.mkdir(parents=True, exist_ok=True)
    # Simulation persistence consumes its attached solution.
    solution.save(path=out / "solution.h5")
    simulation.save(directory=out / "simulation")
    ids_path = out / "original_ids.npy"

    def write_ids(path: Path) -> None:
        """Preserve original subject IDs without Python object serialization."""
        with path.open("wb") as handle:
            np.save(handle, original_ids, allow_pickle=False)

    write_atomically(path=ids_path, write=write_ids)
    receipt = out / "receipt.json"
    artifacts: dict[str, dict[str, Any]] = {
        payload.relative_to(out).as_posix(): {"sha256": _file_sha256(path=payload)}
        for payload in sorted(out.rglob("*"))
        if payload.is_file() and payload != receipt
    }
    artifacts["original_ids.npy"].update(
        {
            "array_checksum": array_checksum(
                identity={"field": "original_subject_ids"},
                array=original_ids,
            ),
            "dtype": original_ids.dtype.str,
            "shape": list(original_ids.shape),
        }
    )
    archive_wall_seconds = perf_counter() - archive_started
    write_json_atomically(
        path=receipt,
        payload={
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": phase,
            "status": "completed",
            "started_at": started_at,
            "phase_finished_at": finished_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds + archive_wall_seconds,
            "archive_wall_seconds": archive_wall_seconds,
            "retention": retention,
            "provenance": dict(provenance),
            "campaign": _campaign_contract(plan=plan),
            "value_catalog": value_catalog,
            "raw_coordinates": raw_coordinates,
            "raw_schemas": raw_schemas,
            "raw_catalog": raw_catalog,
            "artifacts": artifacts,
        },
    )
    return receipt


def _campaign_contract(*, plan: ComponentJobPlan) -> dict[str, object]:
    """Use the public plan's shared contract without its independent UUID or path."""
    return {
        "identity": dict(plan.identity),
        "state_name": plan.state_name,
        "codes": list(plan.codes),
        "jobs": [list(job) for job in plan.jobs],
        "seed": plan.seed,
        "n_subjects": plan.n_subjects,
        "initial_conditions_sha256": plan.initial_conditions_sha256,
        "job_rows_sha256": (
            list(plan.job_rows_sha256) if plan.job_rows_sha256 is not None else None
        ),
    }


def _value_catalog(*, solution: SolutionResult) -> list[dict[str, Any]]:
    """Frame every value in its original logical publication order."""
    import numpy as np

    from _lcm.solution.component_fragments import array_checksum

    catalog: list[dict[str, Any]] = []
    for period, regimes in solution.values.items():
        for regime, value in regimes.items():
            array = np.asarray(value)
            catalog.append(
                {
                    "period": period,
                    "regime": regime,
                    "axis_names": list(
                        solution.metadata.value_schemas[(period, regime)].axis_names,
                    ),
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "array_checksum": array_checksum(
                        identity={"period": period, "regime": regime},
                        array=array,
                    ),
                }
            )
            del array
    return catalog


def _raw_catalog(
    *,
    simulation: SimulationResult,
) -> tuple[list[list[str | list[int]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Frame every raw address and retain mapping schemas including empty fields."""
    import numpy as np

    from _lcm.solution.component_fragments import array_checksum
    from lcm.component_jobs import _data_leaves

    raw_coordinates = [
        [regime, list(periods)] for regime, periods in simulation.raw_results.items()
    ]
    raw_schemas = []
    raw_catalog = []
    for regime, periods in simulation.raw_results.items():
        for period, data in periods.items():
            raw_schemas.append(
                {
                    "regime": regime,
                    "period": period,
                    "actions": list(data.actions),
                    "states": list(data.states),
                }
            )
            for address, leaf in _data_leaves(regime=regime, period=period, data=data):
                array = np.asarray(leaf)
                raw_catalog.append(
                    {
                        "address": list(address),
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "array_checksum": array_checksum(
                            identity={
                                "regime": address[0],
                                "period": address[1],
                                "field": address[2],
                                "key": address[3],
                            },
                            array=array,
                        ),
                    }
                )
                del array
    return raw_coordinates, raw_schemas, raw_catalog


# Keep the ordered checksum, schema and durable-identity protocol together.
def _compare_reference(  # noqa: C901, PLR0912, PLR0915
    *,
    collected: CollectedComponentJobs,
    original_ids: np.ndarray,
    reference: Path,
    reference_receipt_sha256: str,
    production_contract: Mapping[str, object] | None = None,
) -> None:
    """Compare complete ordered results with an authenticated public archive."""
    import dataclasses
    import json

    import numpy as np
    import pandas as pd
    from aca_model.simulation import restore_subject_ids

    from _lcm.solution.component_fragments import array_checksum, canonical_json
    from lcm.persistence import load_solution
    from lcm.result import SimulationResult

    if collected.simulation is None:
        raise ValueError("Full production results require a solution and simulation")

    receipt = reference / "receipt.json"
    if _file_sha256(path=receipt) != reference_receipt_sha256:
        raise ValueError("Reference receipt SHA-256 differs")
    record = json.loads(receipt.read_bytes())
    header = {
        "format": "aca-stage8a-phase-receipt",
        "format_version": 1,
        "phase": "reference",
        "status": "completed",
    }
    if not isinstance(record, dict) or canonical_json(
        {key: record.get(key) for key in header},
    ) != canonical_json(header):
        raise ValueError("Reference receipt header differs")
    if canonical_json(record.get("campaign")) != canonical_json(
        _campaign_contract(plan=collected.plan),
    ):
        raise ValueError("Reference campaign differs from the component plan")
    if production_contract is not None and canonical_json(
        record.get("provenance", {}).get("contract"),
    ) != canonical_json(production_contract):
        raise ValueError("Reference production input/source contract differs")
    files = {
        path.relative_to(reference).as_posix(): path
        for path in reference.rglob("*")
        if path.is_file() and path != receipt
    }
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(files) != set(artifacts):
        raise ValueError("Reference artifact inventory differs")
    for name, path in files.items():
        if _file_sha256(path=path) != artifacts[name].get("sha256"):
            raise ValueError(f"Reference artifact differs: {name}")
    ids = np.load(reference / "original_ids.npy", allow_pickle=False)
    if (
        ids.dtype != original_ids.dtype
        or ids.shape != original_ids.shape
        or ids.tobytes() != original_ids.tobytes()
        or artifacts["original_ids.npy"].get("array_checksum")
        != array_checksum(
            identity={"field": "original_subject_ids"},
            array=ids,
        )
        or artifacts["original_ids.npy"].get("dtype") != ids.dtype.str
        or artifacts["original_ids.npy"].get("shape") != list(ids.shape)
    ):
        raise ValueError("Reference comparison differs: original_ids")
    solution = load_solution(path=reference / "solution.h5", verify_checksums=True)
    expected_values = record["value_catalog"]
    actual_values = _value_catalog(solution=collected.solution)
    archived_values = _value_catalog(solution=solution)
    archived_by_address = {
        (row["period"], row["regime"]): row for row in archived_values
    }
    expected_addresses = {(row["period"], row["regime"]) for row in expected_values}
    if (
        set(archived_by_address) != expected_addresses
        or len(expected_addresses) != len(expected_values)
        or canonical_json(actual_values) != canonical_json(expected_values)
        or any(
            canonical_json(archived_by_address[(row["period"], row["regime"])])
            != canonical_json(row)
            for row in expected_values
        )
        or dataclasses.replace(
            solution.metadata,
            source=collected.solution.metadata.source,
            model_instance_id=collected.solution.metadata.model_instance_id,
        )
        != collected.solution.metadata
    ):
        raise ValueError("Reference comparison differs: values")
    simulation = SimulationResult.load(directory=reference / "simulation")
    coordinates, schemas, raw = _raw_catalog(simulation=simulation)
    candidate_coordinates, candidate_schemas, candidate_raw = _raw_catalog(
        simulation=collected.simulation,
    )
    expected_raw = record["raw_catalog"]
    archived_raw = {tuple(row["address"]): row for row in raw}
    raw_addresses = {tuple(row["address"]) for row in expected_raw}
    expected_schemas = {
        (row["regime"], row["period"]): (set(row["actions"]), set(row["states"]))
        for row in record["raw_schemas"]
    }
    archived_schemas = {
        (row["regime"], row["period"]): (set(row["actions"]), set(row["states"]))
        for row in schemas
    }
    if (
        set(archived_raw) != raw_addresses
        or len(raw_addresses) != len(expected_raw)
        or archived_schemas != expected_schemas
        or {(r, p) for r, periods in coordinates for p in periods}
        != {(r, p) for r, periods in record["raw_coordinates"] for p in periods}
        or canonical_json(candidate_coordinates)
        != canonical_json(record["raw_coordinates"])
        or canonical_json(candidate_schemas) != canonical_json(record["raw_schemas"])
        or canonical_json(candidate_raw) != canonical_json(expected_raw)
        or any(
            canonical_json(archived_raw[tuple(row["address"])]) != canonical_json(row)
            for row in expected_raw
        )
    ):
        raise ValueError("Reference comparison differs: raw_results")
    try:
        actual_frame = collected.simulation.to_dataframe()
        reference_frame = simulation.to_dataframe()
        pd.testing.assert_frame_equal(
            actual_frame,
            reference_frame,
            check_exact=True,
            check_dtype=True,
            check_categorical=True,
            check_index_type=True,
            check_column_type=True,
        )
    except AssertionError as error:
        raise ValueError("Reference comparison differs: panel") from error
    if _numeric_panel_checksums(frame=actual_frame) != _numeric_panel_checksums(
        frame=reference_frame,
    ):
        raise ValueError("Reference comparison differs: panel")
    actual_external = restore_subject_ids(panel=actual_frame, original_ids=original_ids)
    reference_external = restore_subject_ids(panel=reference_frame, original_ids=ids)
    pd.testing.assert_frame_equal(
        actual_external,
        reference_external,
        check_exact=True,
        check_dtype=True,
        check_categorical=True,
        check_index_type=True,
        check_column_type=True,
    )
    if _numeric_panel_checksums(frame=actual_external) != _numeric_panel_checksums(
        frame=reference_external
    ):
        raise ValueError("Reference comparison differs: external_ids")


def _numeric_panel_checksums(*, frame: pd.DataFrame) -> list[str]:
    """Frame numeric panel buffers; object labels use pandas' logical comparison."""
    import numpy as np
    import pandas as pd

    from _lcm.solution.component_fragments import array_checksum

    checksums = []

    def add(*, identity: dict[str, object], array: object) -> None:
        """Include a numeric buffer without hashing Python object addresses."""
        value = np.asarray(array)
        if not value.dtype.hasobject:
            checksums.append(array_checksum(identity=identity, array=value))

    for position in range(len(frame.columns)):
        column = frame.iloc[:, position]
        add(
            identity={"boundary": "column", "position": position},
            array=column.to_numpy(),
        )
        if isinstance(column.dtype, pd.CategoricalDtype):
            add(
                identity={"boundary": "category_codes", "position": position},
                array=column.cat.codes.to_numpy(),
            )
            add(
                identity={"boundary": "categories", "position": position},
                array=column.cat.categories.to_numpy(),
            )
    for name, index in (("index", frame.index), ("columns", frame.columns)):
        for level in range(index.nlevels):
            add(
                identity={"boundary": name, "level": level},
                array=index.get_level_values(level).to_numpy(),
            )
        if isinstance(index, pd.MultiIndex):
            for level, (values, codes) in enumerate(
                zip(index.levels, index.codes, strict=True)
            ):
                add(
                    identity={"boundary": name, "level_values": level},
                    array=values.to_numpy(),
                )
                add(identity={"boundary": name, "level_codes": level}, array=codes)
    return checksums


def _validate_planned_ids(
    *,
    plan: ComponentJobPlan,
    receipt: Path,
    receipt_sha256: str,
    original_ids: np.ndarray,
    production_contract: Mapping[str, object] | None = None,
) -> None:
    """Authenticate the planned external-ID mapping before a numeric phase."""
    import json

    import numpy as np

    from _lcm.solution.component_fragments import array_checksum, canonical_json

    if canonical_json(
        {
            "state_name": plan.state_name,
            "codes": list(plan.codes),
            "jobs": [list(job) for job in plan.jobs],
            "seed": plan.seed,
        }
    ) != canonical_json(
        {
            "state_name": "pref_type",
            "codes": [0, 1, 2],
            "jobs": [[0], [1], [2]],
            "seed": 20_260_903,
        }
    ):
        raise ValueError("Component plan differs from the production campaign")
    if _file_sha256(path=receipt) != receipt_sha256:
        raise ValueError("Planning receipt SHA-256 differs")
    record = json.loads(receipt.read_bytes())
    expected = {
        "format": "aca-stage8a-phase-receipt",
        "format_version": 1,
        "phase": "plan",
        "status": "planned",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.digest,
        "state_name": plan.state_name,
        "codes": list(plan.codes),
        "jobs": [list(job) for job in plan.jobs],
        "seed": plan.seed,
        "canonical_input_count": plan.n_subjects,
        "initial_conditions_sha256": plan.initial_conditions_sha256,
    }
    if not isinstance(record, dict) or canonical_json(
        {key: record.get(key) for key in expected},
    ) != canonical_json(expected):
        raise ValueError("Planning receipt differs from the component plan")
    if type(record.get("raw_input_count")) is not int or record[
        "raw_input_count"
    ] < len(original_ids):
        raise ValueError("Planning receipt population counts differ")
    ids_path = receipt.parent / "original_ids.npy"
    saved_ids = np.load(ids_path, allow_pickle=False)
    expected_artifact = {
        "sha256": _file_sha256(path=ids_path),
        "array_checksum": array_checksum(
            identity={"field": "original_subject_ids"},
            array=saved_ids,
        ),
        "dtype": saved_ids.dtype.str,
        "shape": list(saved_ids.shape),
    }
    if canonical_json(record.get("artifacts")) != canonical_json(
        {"original_ids.npy": expected_artifact},
    ):
        raise ValueError("Planning original-ID artifact differs")
    if (
        saved_ids.ndim != 1
        or saved_ids.shape != original_ids.shape
        or saved_ids.dtype != original_ids.dtype
        or saved_ids.tobytes() != original_ids.tobytes()
    ):
        raise ValueError("Planned original subject IDs differ")
    if production_contract is not None and canonical_json(
        record.get("provenance", {}).get("contract")
    ) != canonical_json(production_contract):
        raise ValueError("Planned production source or inputs differ")


def _file_sha256(*, path: Path) -> str:
    """Hash a persisted payload without reading it into one host buffer."""
    from hashlib import file_digest

    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    main()
