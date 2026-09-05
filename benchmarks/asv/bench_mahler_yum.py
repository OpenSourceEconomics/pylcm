"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

import gc
import pathlib
import time

from . import _gpu_mem

_N_SUBJECTS = 100


class MahlerYum:
    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
    timeout = 1200

    def _build(self):
        from lcm_examples.mahler_yum_2024 import (
            MAHLER_YUM_MODEL,
            START_PARAMS,
            create_inputs,
        )

        self.model = MAHLER_YUM_MODEL
        self.model_params, self.initial_conditions = create_inputs(
            seed=0,
            n_simulation_subjects=_N_SUBJECTS,
            params=START_PARAMS,
        )

    def setup(self):
        self._build()
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )

    def peakmem_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
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
            self.time_execution()
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


class MahlerYumGpuPeakMem(_gpu_mem.GpuPeakMemProfile):
    bench_module = "benchmarks.asv.bench_mahler_yum"
    bench_class = "MahlerYum"
