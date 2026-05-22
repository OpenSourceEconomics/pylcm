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

- We use `setup_cache` with a cloudpickle-bytes wrapper. ASV's cache
  machinery serialises `setup_cache`'s return value through stdlib
  `pickle`, which can't handle the `MappingProxyType` leaves or
  user-defined callables inside a pylcm `Model`. `setup_cache` returns
  `cloudpickle.dumps(...)` of the `(model, params, initial_conditions)`
  triple; each method's `setup(cache)` calls `cloudpickle.loads` and
  runs a warm simulate. This amortises Python-level model construction
  across `time_execution`, `peakmem_execution`, and
  `track_compilation_time` (~60-120 s saved per ASV run). JAX
  compilation is still per-method — the JIT cache is process-local —
  but the persistent XLA disk cache keeps second and third compiles
  fast.
- `AcaBaselineDebugLog` subclasses `AcaBaseline`, overriding only the
  `log_level` and the per-run temporary `log_path`; it reuses the same
  `setup_cache` / metric methods.
- `AcaBaselineGpuPeakMem` runs in a separate subprocess via `_gpu_mem`
  that does not go through ASV's `setup_cache` pipeline. It calls
  `setup_for_gpu_measurement()` (rebuild fresh, no warm-up) then
  `time_execution()` to measure cold peak memory. Both methods
  accept `cache=None` so the same callable serves ASV (cache passed
  in) and the subprocess (cache omitted).
"""

import gc
import shutil
import tempfile
import time

import cloudpickle

from benchmarks import _gpu_mem

_N_SUBJECTS = 1000


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
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    model_params = get_benchmark_params(model=model)[2]
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=_N_SUBJECTS, seed=0
    )
    return model, model_params, initial_conditions


class AcaBaseline:
    """aca-baseline simulate with runtime validation and logging off."""

    timeout = 3600
    # Pin every ASV sample knob to 1 so setup runs once per subprocess
    # and one warm call is timed. `timeout=3600` gives headroom for the
    # cold compile that happens inside setup(cache).
    rounds = 1
    repeat = 1
    number = 1
    warmup_time = 0

    # Simulate logging configuration; `AcaBaselineDebugLog` overrides both.
    log_level = "off"
    log_path: str | None = None

    def setup_cache(self) -> bytes:
        # Build once per ASV benchmark class run and hand the result to
        # every method via ASV's setup_cache mechanism. ASV pickles the
        # return value with stdlib `pickle`, which can't handle the
        # `MappingProxyType` leaves or user callables inside a pylcm
        # `Model` — so wrap the triple in cloudpickle bytes. ASV then
        # ships plain bytes; each method's setup(cache) reconstructs.
        return cloudpickle.dumps(_build())

    def setup(self, cache: bytes) -> None:
        self.model, self.model_params, self.initial_conditions = cloudpickle.loads(
            cache
        )
        # Warm-trigger compilation so time_execution runs on a hot kernel.
        start = time.perf_counter()
        self._simulate()
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self) -> None:
        # Called by the _gpu_mem subprocess; bypasses ASV's setup_cache
        # pipeline so the subprocess can measure cold peak memory
        # (build + compile + run, no warm-up).
        self.model, self.model_params, self.initial_conditions = _build()

    def _simulate(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level=self.log_level,
            log_path=self.log_path,
        )

    def time_execution(self, cache: bytes | None = None) -> None:
        self._simulate()

    def peakmem_execution(self, cache: bytes | None = None) -> None:
        self._simulate()

    def teardown(self, cache: bytes | None = None) -> None:
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self, cache: bytes | None = None) -> float:
        return self._compile_time

    track_compilation_time.unit = "seconds"


class AcaBaselineDebugLog(AcaBaseline):
    """aca-baseline simulate at `log_level="debug"` with snapshot logging.

    Runs every runtime-validation check and persists diagnostic
    snapshots to a temporary directory. Measured against `AcaBaseline`
    (`log_level="off"`), the difference is the validation + logging
    overhead.
    """

    log_level = "debug"

    def setup(self, cache: bytes) -> None:
        self.log_path = tempfile.mkdtemp(prefix="aca-bench-debug-log-")
        super().setup(cache)

    def teardown(self, cache: bytes | None = None) -> None:
        super().teardown(cache)
        if self.log_path is not None:
            shutil.rmtree(self.log_path, ignore_errors=True)
            self.log_path = None


class AcaBaselineGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_aca_baseline"
    bench_class = "AcaBaseline"
    timeout = 3600
