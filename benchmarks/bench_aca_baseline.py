"""End-to-end benchmark for the aca baseline model (benchmark-sized grids).

Uses `aca_model.benchmark.create_benchmark_model()` — the full 18-regime
aca baseline with tiny continuous grids (`BENCHMARK_GRID_CONFIG`). This
keeps the expensive parts of aca-baseline's cost structure (compile
pipeline over 19 regimes, DAG resolution, pref_type batching) while
shrinking per-call numerical work so the benchmark fits in an asv
invocation.

Requires the `aca_model` package to be importable. Use the
`benchmarks-cuda12` pixi environment, which pulls aca-model from its
public git URL. Inside the aca-dev monorepo the editable path install
takes precedence. Benchmark params are loaded from a frozen pickle
shipped in aca-model — no aca-data pipeline run required.
"""

import gc
import time

from benchmarks import _gpu_mem

_N_SUBJECTS = 1000


class AcaBaseline:
    timeout = 3600
    # setup() compiles (~28 min on GPU with partition-fixed-states);
    # pin everything to 1 so setup runs once per subprocess and one
    # warm call is timed. `timeout=3600` gives headroom for setup +
    # the timed call.
    rounds = 1
    repeat = 1
    number = 1
    warmup_time = 0

    def _build(self) -> None:
        from aca_model.benchmark import (
            create_benchmark_model,
            get_benchmark_initial_conditions,
            get_benchmark_params,
        )

        self.model = create_benchmark_model()
        _, self.model_params = get_benchmark_params()
        self.initial_conditions = get_benchmark_initial_conditions(
            model=self.model, n_subjects=_N_SUBJECTS, seed=0
        )

    def setup(self) -> None:
        self._build()
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
        self._build()

    def time_execution(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def peakmem_execution(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def teardown(self) -> None:
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self) -> float:
        return self._compile_time

    track_compilation_time.unit = "seconds"


class AcaBaselineGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_aca_baseline"
    bench_class = "AcaBaseline"
    timeout = 3600
