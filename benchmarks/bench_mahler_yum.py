"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

import gc
import time

from benchmarks import _gpu_mem

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
        common_params, self.initial_conditions = create_inputs(
            seed=0,
            n_simulation_subjects=_N_SUBJECTS,
            params=START_PARAMS,
        )
        self.model_params = {"alive": common_params}

    def setup(self):
        self._build()
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )

    def peakmem_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class MahlerYumGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_mahler_yum"
    bench_class = "MahlerYum"
