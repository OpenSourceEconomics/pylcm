"""Measure GPU peak memory for a benchmark in an isolated subprocess.

ASV runs all benchmarks in one process, so ``peak_bytes_in_use`` accumulates
across runs and warm-up calls.  This module spawns a fresh Python process that
builds the model, runs it once cold (compilation + execution), and reports the
peak.  A single cold run in a fresh process is the production footprint: the
production launcher solves + simulates exactly once per process.

XLA autotuning is disabled in the subprocess (``--xla_gpu_autotune_level=0``).
Autotuning benchmarks candidate kernels at compile time, allocating large,
run-to-run-variable scratch buffers; for big models that transient dwarfs and
masks the execution working set, so the reported peak swings several-fold
between otherwise-identical runs. Turning it off makes the compile footprint
deterministic and matches how the model is run in production (the sbatch
already sets ``--xla_gpu_autotune_level=0``).

The ``GpuPeakMem`` base class provides a ready-made ASV benchmark with a no-op
``setup`` so the parent process does not touch the GPU before spawning the
subprocess.  Subclass it and set ``bench_module`` / ``bench_class``::

    class MahlerYumGpuPeakMem(GpuPeakMem):
        bench_module = "benchmarks.bench_mahler_yum"
        bench_class = "MahlerYum"

The subprocess calls ``setup_for_gpu_measurement()`` (model + params only, no
warm-up) followed by ``time_execution()`` (cold = compile + run), then prints
``peak_bytes_in_use``.

ACA's long-running benchmarks also use ``measure_combined``. Its subprocess
performs one cold execution, captures cold elapsed time plus CPU peak memory,
then performs and times one warm execution. GPU attribution is deliberately
separate: ``measure_gpu_memory_profile`` runs the three named solve/persistence/
simulate phases sequentially in fresh, phase-isolated processes. Every child
reports its own peak and exact invocation provenance; peaks are never combined
arithmetically.
"""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

# Project root: the directory containing the benchmarks/ package. This file lives
# at benchmarks/asv/_gpu_mem.py, so the repo root is three parents up — the cwd the
# `python -m benchmarks.asv._gpu_mem` subprocess needs for `benchmarks` to import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Marks the peak-memory line on the subprocess's stdout. The subprocess imports
# lcm, whose beartype claw can emit diagnostics to stdout, so the parent locates
# this line instead of parsing stdout wholesale.
_PEAK_MARKER = "__PEAK_BYTES_IN_USE__"
_COMBINED_MARKER = "__COMBINED_MEASUREMENTS__"
_PROFILE_MARKER = "__GPU_MEMORY_PROFILE_PHASE__"
_PROFILE_PROTOCOL_VERSION = 1

# Stable protocol labels. ASV tracker suffixes, parent/child messages, and benchmark
# phase dispatch all use these exact strings.
GPU_MEMORY_PHASES = (
    "automatic_solve_simulate",
    "solve_save_all_persistable",
    "load_supplied_solution_simulate",
)
(
    AUTOMATIC_SOLVE_SIMULATE,
    SOLVE_SAVE_ALL_PERSISTABLE,
    LOAD_SUPPLIED_SOLUTION_SIMULATE,
) = GPU_MEMORY_PHASES

_PROFILE_ENV_KEYS = (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "JAX_COMPILATION_CACHE_DIR",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
)


def _subprocess_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """Build the GPU-mem subprocess environment from a base mapping.

    - Drops ``XLA_PYTHON_CLIENT_MEM_FRACTION`` so the isolated subprocess can
      use all device memory (the parent ASV process may cap itself).
    - Disables preallocation so ``peak_bytes_in_use`` tracks real demand.
    - Appends ``--xla_gpu_autotune_level=0`` to ``XLA_FLAGS`` (preserving any
      existing flags) so the compile footprint is deterministic.
    """
    env = {k: v for k, v in base_env.items() if k != "XLA_PYTHON_CLIENT_MEM_FRACTION"}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    autotune_off = "--xla_gpu_autotune_level=0"
    existing = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{existing} {autotune_off}".strip()
    return env


def _file_provenance(path: Path) -> dict[str, object]:
    """Return identity for one persisted solution archive, without loading it."""
    if not path.is_file():
        msg = f"GPU memory profile archive is not a regular file: {path}"
        raise RuntimeError(msg)
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _profile_subprocess_env(
    *, base_env: Mapping[str, str], phase_root: Path
) -> dict[str, str]:
    """Give one profile phase fresh temporary and compilation-cache directories."""
    env = _subprocess_env(base_env)
    directories = {
        "TMPDIR": phase_root / "tmp",
        "XDG_CACHE_HOME": phase_root / "xdg-cache",
        "JAX_COMPILATION_CACHE_DIR": phase_root / "jax-cache",
    }
    for key, path in directories.items():
        path.mkdir(parents=True)
        env[key] = str(path.resolve())
    return env


def _profile_subprocess_error(
    *,
    result: subprocess.CompletedProcess[str],
    phase: str,
) -> RuntimeError:
    return RuntimeError(
        f"GPU memory profile phase {phase!r} failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )


def _validate_profile_child_result(
    *,
    record: object,
    phase: str,
    bench_module: str,
    bench_class: str,
    invocation_token: str,
    env: Mapping[str, str],
    archive: dict[str, object] | None,
) -> tuple[int, int]:
    """Validate an exact child result before exposing its peak to ASV."""
    expected_keys = {
        "protocol_version",
        "phase",
        "bench_module",
        "bench_class",
        "invocation_token",
        "pid",
        "project_root",
        "executable",
        "environment",
        "device",
        "archive",
        "peak_bytes_in_use",
    }
    if type(record) is not dict or set(record) != expected_keys:
        msg = (
            "GPU memory profile child result has non-exact fields; "
            f"expected {sorted(expected_keys)}, got "
            f"{sorted(record) if type(record) is dict else type(record).__name__}."
        )
        raise RuntimeError(msg)

    expected_scalars = {
        "protocol_version": _PROFILE_PROTOCOL_VERSION,
        "phase": phase,
        "bench_module": bench_module,
        "bench_class": bench_class,
        "invocation_token": invocation_token,
        "project_root": str(_PROJECT_ROOT),
        "executable": str(Path(sys.executable).resolve()),
    }
    mismatches = {
        key: {"expected": expected, "observed": record[key]}
        for key, expected in expected_scalars.items()
        if record[key] != expected or type(record[key]) is not type(expected)
    }
    if mismatches:
        msg = f"GPU memory profile child provenance mismatch: {mismatches}."
        raise RuntimeError(msg)

    pid = record["pid"]
    if type(pid) is not int or pid <= 0:
        msg = f"GPU memory profile child pid must be a positive exact int, got {pid!r}."
        raise RuntimeError(msg)

    expected_environment = {key: env[key] for key in _PROFILE_ENV_KEYS}
    if (
        type(record["environment"]) is not dict
        or record["environment"] != expected_environment
        or any(type(value) is not str for value in record["environment"].values())
    ):
        msg = (
            "GPU memory profile child environment provenance mismatch: "
            f"expected {expected_environment!r}, got {record['environment']!r}."
        )
        raise RuntimeError(msg)

    device = record["device"]
    expected_device_keys = {"id", "kind", "platform"}
    if (
        type(device) is not dict
        or set(device) != expected_device_keys
        or type(device["id"]) is not int
        or type(device["kind"]) is not str
        or not device["kind"]
        or device["platform"] != "gpu"
        or type(device["platform"]) is not str
    ):
        msg = f"GPU memory profile child device provenance is invalid: {device!r}."
        raise RuntimeError(msg)

    if record["archive"] != archive or (
        record["archive"] is not None and type(record["archive"]) is not dict
    ):
        msg = (
            "GPU memory profile child archive provenance mismatch: "
            f"expected {archive!r}, got {record['archive']!r}."
        )
        raise RuntimeError(msg)

    peak = record["peak_bytes_in_use"]
    if type(peak) is not int or peak < 0:
        msg = (
            "GPU memory profile child peak_bytes_in_use must be a non-negative "
            f"exact int, got {peak!r}."
        )
        raise RuntimeError(msg)
    return peak, pid


def _parse_profile_child_result(
    *,
    result: subprocess.CompletedProcess[str],
    phase: str,
    bench_module: str,
    bench_class: str,
    invocation_token: str,
    env: Mapping[str, str],
    archive: dict[str, object] | None,
) -> tuple[int, int]:
    """Parse exactly one marker and reject incomplete or ambiguous child output."""
    if result.returncode != 0:
        raise _profile_subprocess_error(result=result, phase=phase)

    prefix = f"{_PROFILE_MARKER} "
    payloads = [
        line.removeprefix(prefix)
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        msg = (
            f"GPU memory profile phase {phase!r} produced {len(payloads)} "
            f"result markers, expected exactly one.\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg)
    try:
        record = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        msg = (
            f"GPU memory profile phase {phase!r} produced invalid JSON.\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg) from error
    return _validate_profile_child_result(
        record=record,
        phase=phase,
        bench_module=bench_module,
        bench_class=bench_class,
        invocation_token=invocation_token,
        env=env,
        archive=archive,
    )


def _register_profile_child_pid(*, child_pid: int, seen_child_pids: set[int]) -> None:
    """Reject process reuse across nominally isolated profile phases."""
    if child_pid in seen_child_pids:
        msg = (
            "GPU memory profile phases did not run in distinct child "
            f"processes; reused child pid {child_pid}."
        )
        raise RuntimeError(msg)
    seen_child_pids.add(child_pid)


def measure_gpu_memory_profile(
    *, bench_module: str, bench_class: str
) -> dict[str, int]:
    """Measure three solve/persistence/simulate peaks in fresh sequential processes."""
    peaks: dict[str, int] = {}
    seen_child_pids: set[int] = set()
    with tempfile.TemporaryDirectory(prefix="pylcm-gpu-memory-profile-") as tmp:
        profile_root = Path(tmp).resolve()
        archive_path = profile_root / "solution.lcm"
        saved_archive: dict[str, object] | None = None

        for phase in GPU_MEMORY_PHASES:
            if phase in (AUTOMATIC_SOLVE_SIMULATE, SOLVE_SAVE_ALL_PERSISTABLE):
                if archive_path.exists():
                    msg = f"GPU memory profile archive exists before phase {phase!r}."
                    raise RuntimeError(msg)
            elif (
                saved_archive is None or _file_provenance(archive_path) != saved_archive
            ):
                raise RuntimeError(
                    "GPU memory profile archive changed before the supplied-solution "
                    "simulation phase."
                )

            phase_root = profile_root / phase
            phase_root.mkdir()
            env = _profile_subprocess_env(
                base_env=os.environ,
                phase_root=phase_root,
            )
            invocation_token = secrets.token_hex(16)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.asv._gpu_mem",
                    "--profile-phase",
                    phase,
                    "--archive",
                    str(archive_path),
                    "--invocation-token",
                    invocation_token,
                    bench_module,
                    bench_class,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=_PROJECT_ROOT,
                env=env,
            )
            if result.returncode != 0:
                raise _profile_subprocess_error(result=result, phase=phase)

            if phase == AUTOMATIC_SOLVE_SIMULATE:
                if archive_path.exists():
                    raise RuntimeError(
                        "Automatic solve+simulate profile phase unexpectedly created "
                        "the shared solution archive."
                    )
                archive = None
            else:
                archive = _file_provenance(archive_path)
                if phase == SOLVE_SAVE_ALL_PERSISTABLE:
                    saved_archive = archive
                elif archive != saved_archive:
                    raise RuntimeError(
                        "Supplied-solution simulation modified the persisted archive."
                    )

            peak, child_pid = _parse_profile_child_result(
                result=result,
                phase=phase,
                bench_module=bench_module,
                bench_class=bench_class,
                invocation_token=invocation_token,
                env=env,
                archive=archive,
            )
            _register_profile_child_pid(
                child_pid=child_pid, seen_child_pids=seen_child_pids
            )
            peaks[phase] = peak

    return peaks


def measure_gpu_peak(*, bench_module: str, bench_class: str) -> int:
    """Run a benchmark in a subprocess and return peak GPU bytes.

    Args:
        bench_module: Dotted module path (e.g. ``"benchmarks.bench_mahler_yum"``).
        bench_class: Class name within the module (e.g. ``"MahlerYum"``).

    Returns:
        Peak GPU memory in bytes over a single cold run (compile + execute),
        with autotuning disabled.

    """
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.asv._gpu_mem", bench_module, bench_class],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
        env=_subprocess_env(os.environ),
    )
    if result.returncode != 0:
        msg = (
            f"GPU memory subprocess failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg)
    for line in result.stdout.splitlines():
        if line.startswith(_PEAK_MARKER):
            return int(line.removeprefix(_PEAK_MARKER).strip())
    msg = (
        "GPU memory subprocess produced no peak-bytes line.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    raise RuntimeError(msg)


def measure_combined(*, bench_module: str, bench_class: str) -> dict[str, float]:
    """Collect cold/warm timing and CPU peak memory in one isolated process."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.asv._gpu_mem",
            "--combined",
            bench_module,
            bench_class,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
        env=_subprocess_env(os.environ),
    )
    if result.returncode != 0:
        msg = (
            f"Combined measurement subprocess failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg)
    for line in result.stdout.splitlines():
        if line.startswith(_COMBINED_MARKER):
            measurements = json.loads(line.removeprefix(_COMBINED_MARKER).strip())
            return {key: float(value) for key, value in measurements.items()}
    msg = (
        "Combined measurement subprocess produced no result line.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    raise RuntimeError(msg)


def _collect_combined_measurements(instance) -> dict[str, float]:
    """Measure one cold execution and one immediately following warm execution."""
    instance.setup_for_gpu_measurement()

    start = time.perf_counter()
    instance.execute_for_measurement()
    compilation_time = time.perf_counter() - start
    peak_cpu_mem = _get_cpu_peak_bytes()

    start = time.perf_counter()
    instance.execute_for_measurement()
    execution_time = time.perf_counter() - start

    return {
        "compilation_time": compilation_time,
        "execution_time": execution_time,
        "peak_cpu_mem": peak_cpu_mem,
    }


def _collect_gpu_memory_phase(
    *,
    instance,
    phase: str,
    archive_path: Path,
    bench_module: str,
    bench_class: str,
    invocation_token: str,
) -> dict[str, object]:
    """Execute exactly one attributed phase and report its local peak/provenance."""
    if phase in (AUTOMATIC_SOLVE_SIMULATE, SOLVE_SAVE_ALL_PERSISTABLE):
        if archive_path.exists():
            msg = f"GPU memory profile archive exists before phase {phase!r}."
            raise RuntimeError(msg)
    elif phase == LOAD_SUPPLIED_SOLUTION_SIMULATE:
        if not archive_path.is_file():
            msg = (
                "GPU memory profile supplied-solution phase requires the archive "
                f"created by the solve+save phase: {archive_path}"
            )
            raise RuntimeError(msg)
    else:
        msg = f"Unknown GPU memory profile phase: {phase!r}."
        raise ValueError(msg)

    instance.setup_for_gpu_measurement()
    instance.execute_gpu_memory_phase(
        phase=phase,
        archive_path=archive_path,
    )

    if phase == AUTOMATIC_SOLVE_SIMULATE:
        if archive_path.exists():
            raise RuntimeError(
                "Automatic solve+simulate profile phase unexpectedly created the "
                "shared solution archive."
            )
        archive = None
    else:
        archive = _file_provenance(archive_path)

    device = _get_single_local_gpu()
    peak = _get_gpu_peak_bytes(device=device)
    return {
        "protocol_version": _PROFILE_PROTOCOL_VERSION,
        "phase": phase,
        "bench_module": bench_module,
        "bench_class": bench_class,
        "invocation_token": invocation_token,
        "pid": os.getpid(),
        "project_root": str(_PROJECT_ROOT),
        "executable": str(Path(sys.executable).resolve()),
        "environment": {key: os.environ[key] for key in _PROFILE_ENV_KEYS},
        "device": {
            "id": int(device.id),
            "kind": str(device.device_kind),
            "platform": str(device.platform),
        },
        "archive": archive,
        "peak_bytes_in_use": peak,
    }


def _get_cpu_peak_bytes() -> float:
    """Return this process's peak resident set size in bytes on Linux.

    The reading is `ru_maxrss` from `resource.getrusage`, which Linux reports
    in kibibytes. Where the `resource` module is unavailable (Windows) the peak
    is unknown and reported as NaN rather than a fabricated number; ASV treats
    a NaN sample as absent, and the value survives the JSON round trip through
    `measure_combined`.
    """
    try:
        import resource
    except ImportError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _get_single_local_gpu():
    """Return the sole local GPU, refusing ambiguous or CPU-only measurements."""
    import jax

    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        msg = (
            "GPU memory measurement requires exactly one local GPU, got "
            f"{[(device.id, device.platform) for device in devices]!r}."
        )
        raise RuntimeError(msg)
    return devices[0]


def _get_gpu_peak_bytes(*, device=None) -> int:
    """Return one synchronized local GPU allocator peak in bytes."""
    import jax

    jax.effects_barrier()
    measured_device = _get_single_local_gpu() if device is None else device
    stats = measured_device.memory_stats()
    if not isinstance(stats, Mapping) or "peak_bytes_in_use" not in stats:
        msg = f"GPU device returned no peak-memory statistics: {stats!r}."
        raise RuntimeError(msg)
    peak = stats["peak_bytes_in_use"]
    if type(peak) is not int or peak < 0:
        msg = (
            "GPU device peak_bytes_in_use must be a non-negative exact int, "
            f"got {peak!r}."
        )
        raise RuntimeError(msg)
    return peak


def _track_gpu_peak_mem(self):
    return measure_gpu_peak(
        bench_module=self.bench_module, bench_class=self.bench_class
    )


_track_gpu_peak_mem.unit = "bytes"


class GpuPeakMem:
    """ASV benchmark base class for GPU peak memory measurement.

    Subclasses only need to set ``bench_module`` and ``bench_class``.  The
    ``setup`` is intentionally a no-op so the parent ASV process does not
    allocate GPU memory before the subprocess runs.

    The ``track_gpu_peak_mem`` method is injected into subclasses (not the base)
    so that ASV does not discover the base class as a runnable benchmark.
    """

    bench_module: str
    bench_class: str
    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
    timeout = 1200

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.track_gpu_peak_mem = _track_gpu_peak_mem

    def setup(self):
        pass


def _profile_setup_cache(self) -> dict[str, int]:
    return measure_gpu_memory_profile(
        bench_module=self.bench_module,
        bench_class=self.bench_class,
    )


# keyword-only-exempt: library-callback=asv
def _track_peak_gpu_mem_automatic_solve_simulate(
    self, cache: dict[str, int] | None = None
) -> int:
    measurements = self._measurements if cache is None else cache
    return measurements[AUTOMATIC_SOLVE_SIMULATE]


_track_peak_gpu_mem_automatic_solve_simulate.unit = "bytes"


# keyword-only-exempt: library-callback=asv
def _track_peak_gpu_mem_solve_save_all_persistable(
    self, cache: dict[str, int] | None = None
) -> int:
    measurements = self._measurements if cache is None else cache
    return measurements[SOLVE_SAVE_ALL_PERSISTABLE]


_track_peak_gpu_mem_solve_save_all_persistable.unit = "bytes"


# keyword-only-exempt: library-callback=asv
def _track_peak_gpu_mem_load_supplied_solution_simulate(
    self, cache: dict[str, int] | None = None
) -> int:
    measurements = self._measurements if cache is None else cache
    return measurements[LOAD_SUPPLIED_SOLUTION_SIMULATE]


_track_peak_gpu_mem_load_supplied_solution_simulate.unit = "bytes"


class GpuPeakMemProfile:
    """ASV base for the exact three-process solution-lifecycle memory profile."""

    bench_module: str
    bench_class: str
    version = "1"
    timeout = 3600

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.setup_cache = _profile_setup_cache
        cls.track_peak_gpu_mem_automatic_solve_simulate = (
            _track_peak_gpu_mem_automatic_solve_simulate
        )
        cls.track_peak_gpu_mem_solve_save_all_persistable = (
            _track_peak_gpu_mem_solve_save_all_persistable
        )
        cls.track_peak_gpu_mem_load_supplied_solution_simulate = (
            _track_peak_gpu_mem_load_supplied_solution_simulate
        )

    def setup(self, cache: dict[str, int]) -> None:
        self._measurements = cache


if __name__ == "__main__":
    import importlib

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--combined", action="store_true")
    mode.add_argument("--profile-phase", choices=GPU_MEMORY_PHASES)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--invocation-token")
    parser.add_argument("bench_module")
    parser.add_argument("bench_class")
    args = parser.parse_args()

    is_profile = args.profile_phase is not None
    if is_profile and (args.archive is None or not args.invocation_token):
        parser.error("--profile-phase requires --archive and --invocation-token")
    if not is_profile and (
        args.archive is not None or args.invocation_token is not None
    ):
        parser.error("--archive and --invocation-token require --profile-phase")

    module = importlib.import_module(args.bench_module)
    cls = getattr(module, args.bench_class)
    instance = cls()

    if args.combined:
        measurements = _collect_combined_measurements(instance)
        print(f"{_COMBINED_MARKER} {json.dumps(measurements, sort_keys=True)}")
    elif is_profile:
        assert args.profile_phase is not None
        assert args.archive is not None
        assert args.invocation_token is not None
        record = _collect_gpu_memory_phase(
            instance=instance,
            phase=args.profile_phase,
            archive_path=args.archive.resolve(),
            bench_module=args.bench_module,
            bench_class=args.bench_class,
            invocation_token=args.invocation_token,
        )
        print(f"{_PROFILE_MARKER} {json.dumps(record, sort_keys=True)}")
    else:
        instance.setup_for_gpu_measurement()
        instance.time_execution()
        print(f"{_PEAK_MARKER} {_get_gpu_peak_bytes()}")
