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

ASV wiring notes:

- ASV re-runs `setup()` (and the full AOT compile it carries) before
  every benchmark method. The class therefore only defines
  `time_execution` and `track_compilation_time` — two setups, two
  compiles, already a saving versus the historical three-method shape
  that also paid a third compile for `peakmem_execution`.
- GPU peak memory comes from the separate `AcaBaselineGpuPeakMem`
  subprocess class; CPU peakmem for this GPU-heavy workload does not
  justify a third compile.
- We deliberately do *not* use `setup_cache`: ASV's cache machinery
  serialises the cached value through plain `pickle`, which cannot
  handle the `MappingProxyType` leaves that pylcm uses throughout
  `Model`.
"""

import gc
import time

from benchmarks import _gpu_mem

_N_SUBJECTS = 100


class AcaBaseline:
    timeout = 3600
    # Pin every ASV sample knob to 1 so setup runs once per subprocess
    # and one warm call is timed. `timeout=3600` gives headroom.
    rounds = 1
    repeat = 1
    number = 1
    warmup_time = 0

    def _build(self) -> None:
        from aca_model.agent.preferences import BenchmarkPrefType
        from aca_model.benchmark import (
            create_benchmark_model,
            get_benchmark_initial_conditions,
            get_benchmark_params,
        )

        from lcm import DiscreteGrid, DispatchStrategy

        # Partition-lifted pref_type so the benchmark kernel runs one
        # Bellman compile per partition point with a JAX-visible sweep.
        # aca-model's default is fused vmap (for compatibility with
        # pylcm versions that pre-date `DispatchStrategy`); the PR #331
        # benchmark intentionally exercises the partition path.
        pref_type_grid = DiscreteGrid(
            BenchmarkPrefType, dispatch=DispatchStrategy.PARTITION_SCAN
        )
        self.model = create_benchmark_model(pref_type_grid=pref_type_grid)
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
