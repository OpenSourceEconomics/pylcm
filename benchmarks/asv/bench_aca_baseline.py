"""End-to-end benchmark for the aca baseline model (benchmark-sized grids).

Uses `aca_model.benchmark.create_benchmark_model()` — the full 18-regime
aca baseline with tiny continuous grids (`BENCHMARK_GRID_CONFIG`) and a
2-type `BenchmarkPrefType` (half the compile + execution volume of the
production 3-type `PrefType`). The kernel exercised here keeps the
expensive parts of aca-baseline's cost structure (compile pipeline
over 19 regimes, DAG resolution, pref_type batching) while shrinking
per-call numerical work so the benchmark fits in an asv invocation.

Two simulate variants run as separate benchmark classes:

- `AcaBaseline` — `log_level="off"`, `log_path=None`: runtime validation
  and diagnostic logging disabled.
- `AcaBaselineDebugLog` — `log_level="debug"` with snapshots written to a
  temporary directory: the slow path that runs every validation check
  and persists diagnostic snapshots. The gap to `AcaBaseline` is the
  validation + logging overhead.

Requires the `aca_model` package to be importable. Use the
`benchmarks-cuda12` pixi environment, which pulls aca-model from its
public git URL. Inside the aca-dev monorepo the editable path install
takes precedence. Benchmark params are loaded from a frozen pickle
shipped in aca-model — no aca-data pipeline run required.

ASV wiring notes:

- Each class's `setup_cache` launches one isolated subprocess through
  `_gpu_mem.measure_combined`. That process builds once, runs one cold
  simulation while recording compilation time and CPU/GPU peak memory,
  then immediately times one warm simulation. Four cheap `track_*`
  methods read the shared result. This avoids recompiling the model once
  per metric while retaining four independent ASV series.
- `AcaBaselineDebugLog` has its own `setup_cache` definition so ASV gives
  the debug configuration a separate combined subprocess.
- XLA autotuning is disabled and preallocation is off in the measurement
  subprocess, preserving the previous GPU-memory benchmark semantics.
"""

import atexit
import pathlib
import shutil
import tempfile
import time

from . import _gpu_mem

_N_SUBJECTS = 1000

_LOG_DIR_PREFIX = "aca-bench-debug-log-"

# Longer than the combined subprocess timeout, so the sweep below can never
# remove a directory belonging to a live run.
_STALE_LOG_DIR_AGE_SECONDS = 24 * 3600


def _sweep_stale_log_dirs() -> None:
    """Remove debug-log directories orphaned by a process that died before cleanup.

    `atexit` covers a normal exit, but not `SIGKILL` -- an OOM kill, an ASV
    timeout, or a cancelled CI job. Without this backstop those directories
    accumulate at ~455 MB each until something else fills the disk, which is
    how the benchmark runner ran out of space (186 directories, 63 GB, the
    oldest 30 days old).
    """
    root = pathlib.Path(tempfile.gettempdir())
    cutoff = time.time() - _STALE_LOG_DIR_AGE_SECONDS
    for path in root.glob(f"{_LOG_DIR_PREFIX}*"):
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _make_log_dir() -> str:
    """Create a debug-log directory that cleans itself up when the process exits."""
    _sweep_stale_log_dirs()
    path = tempfile.mkdtemp(prefix=_LOG_DIR_PREFIX)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _build() -> tuple[object, object, object]:
    """Build the aca-baseline model, params, and initial conditions.

    aca_model and lcm imports are deferred to the function body — ASV's
    forkserver runs `preimport` to discover benchmarks across every
    `bench_*.py` module before forking workers. Importing JAX at module
    top loads the multithreaded XLA backend into the forkserver; every
    subsequent `os.fork()` inherits a corrupted CUDA context and the
    first device op in the worker aborts with
    `CUDA_ERROR_NOT_INITIALIZED`. Per-call imports keep JAX out of the
    forkserver and confine it to the worker process.
    """
    from aca_model.agent.preferences import BenchmarkPrefType
    from aca_model.benchmark import (
        create_benchmark_model,
        get_benchmark_initial_conditions,
        get_benchmark_params,
    )

    from lcm import DiscreteGrid

    model = create_benchmark_model(
        n_subjects=_N_SUBJECTS,
        pref_type_grid=DiscreteGrid(category_class=BenchmarkPrefType),
    )
    edge_periods = model.reachability.solution.periods_for_edge(
        source="retiree_oamc_forced_forcedout",
        target="nongroup_dimc_choose_canwork",
    )
    if edge_periods:
        msg = (
            "ACA activity windows must exclude the forced-out to can-work edge; "
            f"retained periods: {edge_periods}."
        )
        raise AssertionError(msg)
    model_params = get_benchmark_params(model=model)[2]
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=_N_SUBJECTS, seed=0
    )
    return model, model_params, initial_conditions


class AcaBaseline:
    """aca-baseline simulate with runtime validation and logging off."""

    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
    timeout = 3600
    # Simulate logging configuration; `AcaBaselineDebugLog` overrides both.
    log_level = "off"
    log_path: str | None = None

    def setup_cache(self) -> dict[str, float]:
        return _gpu_mem.measure_combined(
            bench_module="benchmarks.asv.bench_aca_baseline",
            bench_class="AcaBaseline",
        )

    def setup(self, cache: dict[str, float]) -> None:
        self._measurements = cache

    def setup_for_gpu_measurement(self) -> None:
        # Called inside the isolated combined-measurement subprocess. This only
        # builds the model and inputs; the collector owns the cold and warm runs.
        self.model, self.model_params, self.initial_conditions = _build()

    def execute_for_measurement(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level=self.log_level,
            log_path=self.log_path,
        )

    def track_execution_time(self, cache: dict[str, float] | None = None) -> float:
        return self._measurements["execution_time"]

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(self, cache: dict[str, float] | None = None) -> float:
        return self._measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_peak_gpu_mem(self, cache: dict[str, float] | None = None) -> float:
        return self._measurements["peak_gpu_mem"]

    track_peak_gpu_mem.unit = "bytes"

    def track_compilation_time(self, cache: dict[str, float] | None = None) -> float:
        return self._measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class AcaBaselineDebugLog(AcaBaseline):
    """aca-baseline simulate at `log_level="debug"` with snapshot logging.

    Runs every runtime-validation check and persists diagnostic
    snapshots to a temporary directory. Measured against `AcaBaseline`
    (`log_level="off"`), the difference is the validation + logging
    overhead.
    """

    log_level = "debug"

    def setup_cache(self) -> dict[str, float]:
        return _gpu_mem.measure_combined(
            bench_module="benchmarks.asv.bench_aca_baseline",
            bench_class="AcaBaselineDebugLog",
        )

    def setup_for_gpu_measurement(self) -> None:
        # Mirror `setup`'s log_path setup so the measurement subprocess
        # exercises snapshot writing too. It exits without ASV teardown, so
        # cleanup rides on `atexit` inside `_make_log_dir` instead.
        self.log_path = _make_log_dir()
        super().setup_for_gpu_measurement()
