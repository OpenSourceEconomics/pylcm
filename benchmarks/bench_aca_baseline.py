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

- We deliberately do *not* use `setup_cache`: ASV's cache machinery
  serialises the cached value through plain `pickle`, which cannot
  handle the `MappingProxyType` leaves that pylcm uses throughout
  `Model`.
"""

import gc
import time

from benchmarks import _gpu_mem

_N_SUBJECTS = 1000


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

        # Explicit `FUSED_VMAP` to match PR #328, which pre-dates
        # `DispatchStrategy` and always fuses pref_type into the
        # state-action space. Same kernel, same memory footprint;
        # only the infrastructure around it differs between branches.
        pref_type_grid = DiscreteGrid(
            BenchmarkPrefType, dispatch=DispatchStrategy.FUSED_VMAP
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
