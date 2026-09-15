"""Precautionary savings benchmarks: solve, simulate, grid types."""

import statistics

from . import _gpu_mem

_N_SUBJECTS = 1_000

# Warm samples each class's setup_cache collects per commit, shared by
# track_execution_time and track_peak_cpu_mem: build once, one cold call,
# then this many timed warm calls -- not one independent build+warm cycle
# per metric (mirrors bench_mahler_yum.py's MahlerYumBudgetedGpu).
_WARM_SAMPLES = 3


def _make_model(*, wealth_grid_type="lin", wealth_n_points=10, consumption_n_points=10):
    from lcm_examples import precautionary_savings

    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type=wealth_grid_type,
        wealth_n_points=wealth_n_points,
        consumption_n_points=consumption_n_points,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst",
        sigma=0.2,
        rho=0.9,
    )
    return model, params


def _make_initial_conditions(n_subjects):
    import jax.numpy as jnp

    return {
        "age": jnp.full(n_subjects, 20.0),
        "wealth": jnp.full(n_subjects, 5.0),
        "income": jnp.full(n_subjects, 0.0),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }


class PrecautionarySavingsSolve:
    """Solve-only timing/CPU-memory, one shared measured producer.

    `time_execution`/`peakmem_execution` were previously separate ASV-native
    benchmarks; each independently calls `setup()`, so measuring both paid
    for two independent build+solve cycles for the same underlying
    operation. `setup_cache` now collects one cold call and `_WARM_SAMPLES`
    warm calls in one isolated subprocess; the cheap `track_*` methods below
    read the shared result.
    """

    # Version bumped: setup_cache-based combined producer replaces the
    # separate ASV-native time_execution/peakmem_execution benchmarks.
    version = "2"
    timeout = 600

    def _build(self):
        self.model, self.model_params = _make_model(
            wealth_n_points=500,
            consumption_n_points=500,
        )

    def setup_for_gpu_measurement(self):
        self._build()

    def execute_for_measurement(self):
        self.model.solve(params=self.model_params, log_level="off")

    def setup_cache(self):
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_precautionary_savings",
            bench_class="PrecautionarySavingsSolve",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache):
        self._measurements = cache

    def track_execution_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_compilation_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulate:
    """Simulate-only timing/CPU-memory, one shared measured producer."""

    version = "2"
    timeout = 600

    def _build(self):
        self.model, self.model_params = _make_model()
        self.period_to_regime_to_V_arr = self.model.solve(
            params=self.model_params, log_level="off"
        )
        self.initial_conditions = _make_initial_conditions(1_000_000)

    def setup_for_gpu_measurement(self):
        self._build()

    def execute_for_measurement(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            solution=self.period_to_regime_to_V_arr,
            log_level="off",
        )

    def setup_cache(self):
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_precautionary_savings",
            bench_class="PrecautionarySavingsSimulate",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache):
        self._measurements = cache

    def track_execution_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_compilation_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulateGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_precautionary_savings"
    bench_class = "PrecautionarySavingsSimulate"


class PrecautionarySavingsSimulateWithSolve:
    """Combined solve+simulate timing/CPU-memory, one shared measured producer."""

    version = "2"
    timeout = 600

    def _build(self):
        self.model, self.model_params = _make_model(
            wealth_n_points=200,
            consumption_n_points=200,
        )
        self.initial_conditions = _make_initial_conditions(500_000)

    def setup_for_gpu_measurement(self):
        self._build()

    def execute_for_measurement(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )

    def setup_cache(self):
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_precautionary_savings",
            bench_class="PrecautionarySavingsSimulateWithSolve",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache):
        self._measurements = cache

    def track_execution_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_compilation_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulateWithSolveGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_precautionary_savings"
    bench_class = "PrecautionarySavingsSimulateWithSolve"


class PrecautionarySavingsSimulateWithSolveIrreg:
    """Combined solve+simulate on an irregular grid, one shared measured producer."""

    version = "2"
    timeout = 600

    def _build(self):
        self.model, self.model_params = _make_model(
            wealth_grid_type="irreg",
            wealth_n_points=200,
            consumption_n_points=200,
        )
        self.initial_conditions = _make_initial_conditions(500_000)

    def setup_for_gpu_measurement(self):
        self._build()

    def execute_for_measurement(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )

    def setup_cache(self):
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_precautionary_savings",
            bench_class="PrecautionarySavingsSimulateWithSolveIrreg",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache):
        self._measurements = cache

    def track_execution_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_compilation_time(self, cache=None):
        measurements = self._measurements if cache is None else cache
        return measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulateWithSolveIrregGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_precautionary_savings"
    bench_class = "PrecautionarySavingsSimulateWithSolveIrreg"
