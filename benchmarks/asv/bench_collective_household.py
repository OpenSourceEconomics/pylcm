"""Collective household workloads: build, compile, solve, simulate.

The declarations a collective regime makes cost something no singleton model
pays for, and each cost is measured on its own axis here:

- **model construction** — the phase scan, the lowering of the collective
  declarations, and the per-edge parameter discovery, with nothing compiled;
- **cold compilation** — the first solve, which traces and compiles every
  regime's kernels and its exit programs;
- **warm solve** — the same solve with the compilation cache hot, which is what
  an estimation loop pays per parameter vector;
- **simulation over cohort size** — exits route original subjects before their
  final action each period, so this is where a cohort-size term would show.

Host memory is tracked by `track_peak_cpu_mem` or `peakmem_*` and device memory
by the `GpuPeakMem` companions, on the same workloads.
"""

import gc
import statistics

from . import _gpu_mem

# Timed warm calls behind every `track_execution_time` here, which reports their
# median. The construct, solve and simulate classes collect them in `setup_cache`,
# once per commit and shared by every metric of the class: build once, one cold
# call, then this many timed warm calls -- not one independent build+warm cycle
# per metric (mirrors bench_mahler_yum.py's MahlerYumBudgetedGpu).
_WARM_SAMPLES = 3

_N_PERIODS = 6

_WEALTH_N_POINTS = 40
_CONSUMPTION_N_POINTS = 40


def _make_model():
    from lcm_examples import collective_household

    model = collective_household.get_model(
        n_periods=_N_PERIODS,
        wealth_n_points=_WEALTH_N_POINTS,
        consumption_n_points=_CONSUMPTION_N_POINTS,
        **_gpu_mem.default_budget_execution_kwargs(),
        **_gpu_mem.lottery_contraction_kwargs(collective_household.get_model),
    )
    return model, collective_household.get_params()


def _clear_gpu_memory():
    import jax

    jax.clear_caches()
    gc.collect()


class CollectiveHouseholdConstruct:
    """What `Model(...)` costs before anything is traced.

    `time_execution`/`peakmem_execution` were previously separate ASV-native
    benchmarks, each independently paying for a construction call.
    `setup_cache` now collects one cold call and `_WARM_SAMPLES` warm calls
    in one isolated subprocess; the cheap `track_*` methods below read the
    shared result.
    """

    # Version bumped: setup_cache-based combined producer replaces the
    # separate ASV-native time_execution/peakmem_execution benchmarks.
    version = "3"
    timeout = 600

    def setup_for_gpu_measurement(self):
        # Warm the imports so the measured call is construction, not the
        # first import of jax and pylcm.
        pass

    def execute_for_measurement(self):
        _make_model()

    def setup_cache(self) -> dict[str, float | list[float]]:
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_collective_household",
            bench_class="CollectiveHouseholdConstruct",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache: dict[str, float | list[float]]) -> None:
        self._measurements = cache

    def track_execution_time(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"


class CollectiveHouseholdSolve:
    """Backward induction over six periods of the marriage market.

    `time_execution`/`peakmem_execution` were previously separate ASV-native
    benchmarks; each independently calls `setup()`, so measuring both paid
    for two independent build+solve cycles for the same underlying
    operation. `setup_cache` now collects one cold call and `_WARM_SAMPLES`
    warm calls in one isolated subprocess; the cheap `track_*` methods below
    read the shared result.
    """

    # Version bumped: setup_cache-based combined producer replaces the
    # separate ASV-native time_execution/peakmem_execution benchmarks.
    version = "3"
    timeout = 900

    def _build(self):
        self.model, self.model_params = _make_model()

    def setup_for_gpu_measurement(self):
        self._build()

    def execute_for_measurement(self):
        self.model.solve(
            params=self.model_params, log_level="off", runtime_checks=False
        )

    def setup_cache(self) -> dict[str, float | list[float]]:
        return _gpu_mem.measure_combined_with_warm_samples(
            bench_module="benchmarks.asv.bench_collective_household",
            bench_class="CollectiveHouseholdSolve",
            warm_samples=_WARM_SAMPLES,
        )

    def setup(self, cache: dict[str, float | list[float]]) -> None:
        self._measurements = cache

    def track_execution_time(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return statistics.median(measurements["warm_samples"])

    track_execution_time.unit = "seconds"

    def track_peak_cpu_mem(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return measurements["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    def track_compilation_time(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return measurements["compilation_time"]

    track_compilation_time.unit = "seconds"


class _SimulateCohort:
    """One cohort size, as `_gpu_mem`'s measurement subprocess drives it."""

    n_subjects: int

    def setup_for_gpu_measurement(self):
        from lcm_examples import collective_household

        self.model, self.model_params = _make_model()
        self.period_to_regime_to_V_arr = self.model.solve(
            params=self.model_params, log_level="off", runtime_checks=False
        )
        self.initial_conditions = collective_household.get_initial_conditions(
            n_subjects=self.n_subjects, model=self.model
        )

    def execute_for_measurement(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            solution=self.period_to_regime_to_V_arr,
            log_level="off",
            runtime_checks=False,
            seed=0,
        )


class _Simulate1000(_SimulateCohort):
    n_subjects = 1_000


class _Simulate100000(_SimulateCohort):
    n_subjects = 100_000


_SIMULATE_COHORTS = {1_000: "_Simulate1000", 100_000: "_Simulate100000"}

type _CohortMeasurements = dict[int, dict[str, float | list[float]]]


class _SimulateMeasurements:
    """Measure every cohort once per commit, for every metric below.

    Each cohort runs in one isolated subprocess: build and solve, one cold
    simulation (first-call seconds, host peak, device peak), then
    `_WARM_SAMPLES` timed warm simulations. The first simulation of a cohort
    takes minutes, so no metric pays for one of its own.
    """

    def setup_cache(self) -> _CohortMeasurements:
        return {
            n_subjects: _gpu_mem.measure_combined_with_warm_samples_and_gpu_peak(
                bench_module="benchmarks.asv.bench_collective_household",
                bench_class=bench_class,
                warm_samples=_WARM_SAMPLES,
            )
            for n_subjects, bench_class in _SIMULATE_COHORTS.items()
        }

    # Two first simulations and their warm calls, one after the other.
    setup_cache.timeout = 3600


class CollectiveHouseholdSimulate(_SimulateMeasurements):
    """Forward simulation, over cohort sizes.

    Exits route each original subject before its final action.
    A cohort-size term that grew faster than linearly would show up between
    these points.
    """

    # Version bumped: the shared `setup_cache` replaces one build, solve and
    # first simulation per metric and cohort, and `track_peak_cpu_mem` replaces
    # ASV-native `peakmem_execution`.
    version = "3"
    # 10,000 (the intermediate point) added little coverage over the two
    # endpoints in the PR#440 benchmark review's measured run (warm
    # simulation ~337/355/354 ms at 1,000/10,000/100,000) and cost 3m24s;
    # dropped rather than kept as a third routine point.
    params = list(_SIMULATE_COHORTS)
    param_names = ["n_subjects"]

    # keyword-only-exempt: library-callback=asv
    def track_execution_time(
        self, cache: _CohortMeasurements, n_subjects: int
    ) -> float:
        return statistics.median(cache[n_subjects]["warm_samples"])

    track_execution_time.unit = "seconds"

    # keyword-only-exempt: library-callback=asv
    def track_peak_cpu_mem(self, cache: _CohortMeasurements, n_subjects: int) -> float:
        return cache[n_subjects]["peak_cpu_mem"]

    track_peak_cpu_mem.unit = "bytes"

    # keyword-only-exempt: library-callback=asv
    def track_compilation_time(
        self, cache: _CohortMeasurements, n_subjects: int
    ) -> float:
        return cache[n_subjects]["compilation_time"]

    track_compilation_time.unit = "seconds"


class CollectiveHouseholdSimulateGpuPeakMem(_SimulateMeasurements):
    """Device peak of the largest cohort's build, solve and first simulation."""

    # Version bumped: read from the shared `setup_cache` instead of a cold run
    # of its own.
    version = "3"

    def track_gpu_peak_mem(self, cache: _CohortMeasurements) -> float:
        return cache[max(cache)]["peak_gpu_mem_automatic_solve_simulate"]

    track_gpu_peak_mem.unit = "bytes"
