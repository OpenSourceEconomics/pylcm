"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

import gc
import pathlib
import statistics
import time

from . import _gpu_mem
from ._mahler_execution import create_mahler_gpu_model

_N_SUBJECTS = 100

# Warm samples MahlerYumBudgetedGpu's setup_cache collects per commit, shared
# by track_execution_time and track_peak_cpu_mem: build once, one cold call,
# then this many timed warm calls -- not one independent build+warm cycle
# per metric.
_WARM_SAMPLES = 3


class _MahlerYum:
    """Retired default configuration retained only as a historical reference."""

    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
    timeout = 1200
    simulation_seed: int | None = None

    def _build(self):
        from lcm_examples.mahler_yum_2024 import (
            MAHLER_YUM_MODEL,
        )

        self.model = MAHLER_YUM_MODEL
        self._build_inputs()

    def _build_inputs(self):
        start_params, create_inputs = _load_inputs_api()

        self.model_params, self.initial_conditions = create_inputs(
            seed=0,
            n_simulation_subjects=_N_SUBJECTS,
            params=start_params,
        )

    def setup(self):
        self._build()
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            seed=self.simulation_seed,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self.execute_for_measurement()

    def peakmem_execution(self):
        self.execute_for_measurement()

    def execute_for_measurement(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            seed=self.simulation_seed,
            log_level="off",
        )

    def execute_gpu_memory_phase(
        self,
        *,
        phase: str,
        archive_path: pathlib.Path,
    ) -> None:
        """Run one exact solution-lifecycle phase in its dedicated child process."""
        if phase == _gpu_mem.AUTOMATIC_SOLVE_SIMULATE:
            self.execute_for_measurement()
            return
        if phase == _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE:
            from lcm.solver_api import ResultRetention

            solution = self.model.solve(
                params=self.model_params,
                log_level="off",
                retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
            )
            solution.save(path=archive_path)
            return
        if phase == _gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE:
            from lcm.persistence import load_solution

            solution = load_solution(path=archive_path)
            self.model.simulate(
                params=self.model_params,
                initial_conditions=self.initial_conditions,
                seed=self.simulation_seed,
                solution=solution,
                log_level="off",
            )
            return
        msg = f"Unknown GPU memory profile phase: {phase!r}."
        raise ValueError(msg)

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class _MahlerYumGpuPeakMem(_gpu_mem.GpuPeakMemProfile):
    """Retired legacy lifecycle tracker, excluded from routine ASV discovery."""

    bench_module = "benchmarks.asv.bench_mahler_yum"
    bench_class = "MahlerYum"


class MahlerYumBudgetedGpu(_MahlerYum):
    """Distinct fp64 ASV series with capacity-admitted GPU execution.

    Unlike the retired `_MahlerYum` base, timing is not ASV-native: ASV calls
    `setup()` once per discovered benchmark (and again per round), so a
    class exposing `time_execution`, `peakmem_execution`, and
    `track_compilation_time` as three separate native/track benchmarks paid
    for three (or more, across rounds) independent build+compile+solve
    cycles for what is the same underlying measurement. `setup_cache`
    collects one cold call and `_WARM_SAMPLES` warm calls in one isolated
    subprocess (mirroring `AcaBaseline`'s combined producer); the cheap
    `track_*` methods below read the shared result. `time_execution` and
    `peakmem_execution` are unset so ASV does not also discover the
    inherited native-timing identities for this subclass.

    `setup_cache` also reads the automatic solve+simulate GPU peak from that
    same cold call (before any warm call). `GpuPeakMemProfile`'s
    `automatic_solve_simulate` phase previously spent its own fully
    independent isolated cold run just to measure that peak;
    `MahlerYumBudgetedGpuPeakMem` below now only covers the two phases that
    still need a genuinely separate process.
    """

    version = "4"
    simulation_seed = 0
    time_execution = None
    peakmem_execution = None

    def _build(self):
        self.model, self.capacity_receipt = create_mahler_gpu_model()
        self._build_inputs()

    def setup_cache(self) -> dict[str, float | list[float]]:
        return _gpu_mem.measure_combined_with_warm_samples_and_gpu_peak(
            bench_module="benchmarks.asv.bench_mahler_yum",
            bench_class="MahlerYumBudgetedGpu",
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

    def track_peak_gpu_mem_automatic_solve_simulate(
        self, cache: dict[str, float | list[float]] | None = None
    ) -> float:
        measurements = self._measurements if cache is None else cache
        return measurements["peak_gpu_mem_automatic_solve_simulate"]

    track_peak_gpu_mem_automatic_solve_simulate.unit = "bytes"


def _load_inputs_api():
    from lcm_examples.mahler_yum_2024 import START_PARAMS, create_inputs

    return START_PARAMS, create_inputs


class MahlerYumBudgetedGpuPeakMem(_gpu_mem.GpuPeakMemProfile):
    """Lifecycle GPU peaks for the capacity-admitted fp64 ASV series.

    `automatic_solve_simulate` is deliberately absent from `phases`:
    `MahlerYumBudgetedGpu.setup_cache` now captures that peak from its own
    cold call, so this class only spends isolated processes on the two
    phases -- solve+save and load+simulate -- that genuinely need one.
    """

    version = "3"
    phases = (
        _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE,
        _gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE,
    )
    bench_module = "benchmarks.asv.bench_mahler_yum"
    bench_class = "MahlerYumBudgetedGpu"
