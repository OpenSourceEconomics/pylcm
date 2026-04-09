"""Mortality model benchmark: solve + simulate."""

import gc
import time

from benchmarks import _gpu_mem

_N_SUBJECTS = 100_000


class Mortality:
    timeout = 600

    def _build(self):
        import jax.numpy as jnp

        from lcm_examples import mortality

        self.model = mortality.get_model(n_periods=4)
        self.model_params = mortality.get_params(n_periods=4)
        self.initial_conditions = {
            "age": jnp.full(_N_SUBJECTS, 40.0),
            "wealth": jnp.full(_N_SUBJECTS, 100.0),
            "regime": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        }

    def setup(self):
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

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def peakmem_execution(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            check_initial_conditions=False,
        )

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()

    def track_compilation_time(self):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class MortalityGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_mortality"
    bench_class = "Mortality"
