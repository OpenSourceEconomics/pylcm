"""End-to-end benchmark for the aca baseline model (benchmark-sized grids).

Uses `aca_model.benchmark.create_benchmark_model()` — the full 18-regime
aca baseline with tiny continuous grids (`BENCHMARK_GRID_CONFIG`) and a
2-type `BenchmarkPrefType` lifted via
`DispatchStrategy.PARTITION_SCAN`. The kernel exercised here keeps the
expensive parts of aca-baseline's cost structure (compile pipeline
over 19 regimes, DAG resolution, partition sweep) while shrinking
per-call numerical work so the benchmark fits in an asv invocation.

Requires the `aca_model` package to be importable. Use the
`benchmarks-cuda12` pixi environment, which pulls aca-model from its
public git URL. Inside the aca-dev monorepo the editable path install
takes precedence. Benchmark params are loaded from a frozen pickle
shipped in aca-model — no aca-data pipeline run required.

ASV wiring:

- `setup_cache` builds the model, measures compile time, and returns a
  state dict that is shared across every benchmark method in the class.
  The expensive compile therefore runs **once** per class, not once per
  method — this alone halves aca-baseline's total wall time.
- GPU peak memory is reported by the separate `AcaBaselineGpuPeakMem`
  subprocess class below; CPU `peakmem_execution` is intentionally
  omitted because it would force a second compile for a low-value
  metric on this GPU-heavy workload.
"""

import gc
import time
from typing import Any

from benchmarks import _gpu_mem

_N_SUBJECTS = 100


class AcaBaseline:
    timeout = 3600
    # setup_cache compiles once per class; pin every ASV sample
    # parameter to 1 so a single warm call is timed. `timeout=3600`
    # gives headroom for setup + the timed call.
    rounds = 1
    repeat = 1
    number = 1
    warmup_time = 0

    def setup_cache(self) -> dict[str, Any]:
        """Build the model and compile once; shared across benchmark methods.

        Returns a dict with the compiled model, params, initial
        conditions, and the measured compile time. ASV pickles this
        value and passes it as the first argument to every benchmark
        method in the class.
        """
        from aca_model.benchmark import (
            create_benchmark_model,
            get_benchmark_initial_conditions,
            get_benchmark_params,
        )

        model = create_benchmark_model()
        _, params = get_benchmark_params()
        initial_conditions = get_benchmark_initial_conditions(
            model=model, n_subjects=_N_SUBJECTS, seed=0
        )
        start = time.perf_counter()
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )
        compile_time = time.perf_counter() - start
        return {
            "model": model,
            "params": params,
            "initial_conditions": initial_conditions,
            "compile_time": compile_time,
        }

    def setup_for_gpu_measurement(self) -> None:
        """Build the model on `self` for the GPU-peak-mem subprocess path.

        `AcaBaselineGpuPeakMem` spawns its own Python process that
        bypasses ASV's `setup_cache` machinery; it calls
        `setup_for_gpu_measurement()` and then `time_execution()`
        directly. So we build the model on `self` here and let
        `time_execution` fall back to `self.model` / `self.params`
        when it is called without a cached `state`.
        """
        from aca_model.benchmark import (
            create_benchmark_model,
            get_benchmark_initial_conditions,
            get_benchmark_params,
        )

        self._model = create_benchmark_model()
        _, self._params = get_benchmark_params()
        self._initial_conditions = get_benchmark_initial_conditions(
            model=self._model, n_subjects=_N_SUBJECTS, seed=0
        )

    def time_execution(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            model = self._model
            params = self._params
            initial_conditions = self._initial_conditions
        else:
            model = state["model"]
            params = state["params"]
            initial_conditions = state["initial_conditions"]
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def teardown(self, state: dict[str, Any] | None = None) -> None:
        del state
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self, state: dict[str, Any]) -> float:
        return state["compile_time"]

    track_compilation_time.unit = "seconds"


class AcaBaselineGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_aca_baseline"
    bench_class = "AcaBaseline"
    timeout = 3600
