"""End-to-end benchmark for the aca baseline model (benchmark-sized grids).

Uses `aca_model.benchmark.create_benchmark_model()` — the full 18-regime
aca baseline with tiny continuous grids (`BENCHMARK_GRID_CONFIG`) and a
2-type `BenchmarkPrefType` dispatched via `DispatchStrategy.FUSED_VMAP`
(the pylcm default). This matches the kernel that PR #328 benchmarks —
pref_type stays inside the state-action space and is fused with the
other state / action axes into one XLA program — so PR #331's numbers
isolate the infrastructure changes (lazy diagnostics, compile/run
split, `DispatchStrategy` plumbing) from the dispatch-strategy switch.

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
- `AcaBaselineGpuPeakMem` runs in a separate subprocess via `_gpu_mem`
  that does not go through ASV's `setup_cache` pipeline. It calls
  `setup_for_gpu_measurement()` (rebuild fresh, no warm-up) then
  `time_execution()` to measure cold peak memory. Both methods
  accept `cache=None` so the same callable serves ASV (cache passed
  in) and the subprocess (cache omitted).
"""

import gc
import time

import cloudpickle

from benchmarks import _gpu_mem

_N_SUBJECTS = 1000


def _build() -> tuple[object, object, object]:
    """Build the aca-baseline model, params, and initial conditions."""
    from aca_model.agent.preferences import BenchmarkPrefType
    from aca_model.benchmark import (
        create_benchmark_model,
        get_benchmark_initial_conditions,
        get_benchmark_params,
    )

    from lcm import DiscreteGrid, DispatchStrategy

    # Explicit `FUSED_VMAP` to match PR #328, which pre-dates
    # `DispatchStrategy` and always fuses pref_type into the
    # state-action space. Same kernel, same memory footprint;
    # only the infrastructure around it differs between branches.
    pref_type_grid = DiscreteGrid(
        BenchmarkPrefType, dispatch=DispatchStrategy.FUSED_VMAP
    )
    model = create_benchmark_model(pref_type_grid=pref_type_grid)
    _, model_params = get_benchmark_params()
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=_N_SUBJECTS, seed=0
    )
    return model, model_params, initial_conditions


class AcaBaseline:
    timeout = 3600
    # Pin every ASV sample knob to 1 so setup runs once per subprocess
    # and one warm call is timed. `timeout=3600` gives headroom for the
    # cold compile that happens inside setup(cache).
    rounds = 1
    repeat = 1
    number = 1
    warmup_time = 0

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
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self) -> None:
        # Called by the _gpu_mem subprocess; bypasses ASV's setup_cache
        # pipeline so the subprocess can measure cold peak memory
        # (build + compile + run, no warm-up).
        self.model, self.model_params, self.initial_conditions = _build()

    def time_execution(self, cache: bytes | None = None) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def peakmem_execution(self, cache: bytes | None = None) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def teardown(self, cache: bytes | None = None) -> None:
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self, cache: bytes | None = None) -> float:
        return self._compile_time

    track_compilation_time.unit = "seconds"


class AcaBaselineGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_aca_baseline"
    bench_class = "AcaBaseline"
    timeout = 3600
