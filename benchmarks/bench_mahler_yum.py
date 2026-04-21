"""End-to-end benchmark for the Mahler & Yum (2024) replication model.

Loops over the two discount-factor types externally: one solve+simulate per
type, covering disjoint subsets of the subject population. Total simulated
subjects across the two runs equal `_N_SUBJECTS`.
"""

import gc
import time

from benchmarks import _gpu_mem

_N_SUBJECTS = 100


class MahlerYum:
    timeout = 1200

    def _build(self):
        import jax.numpy as jnp

        from lcm_examples.mahler_yum_2024 import (
            MAHLER_YUM_MODEL,
            START_PARAMS,
            create_inputs,
        )

        start_params_without_beta = {
            k: v for k, v in START_PARAMS.items() if k != "beta"
        }

        self.model = MAHLER_YUM_MODEL
        common_params, initial_states, discount_factor_type = create_inputs(
            seed=0,
            n_simulation_subjects=_N_SUBJECTS,
            **start_params_without_beta,
        )

        beta_mean = START_PARAMS["beta"]["mean"]
        beta_std = START_PARAMS["beta"]["std"]
        alive_id = self.model.regime_names_to_ids["alive"]

        self._runs = []
        for type_idx, beta in enumerate([beta_mean - beta_std, beta_mean + beta_std]):
            subset_ids = jnp.flatnonzero(discount_factor_type == type_idx)
            subset_initial_states = {
                state: values[subset_ids] for state, values in initial_states.items()
            }
            params = {
                "alive": {
                    "discount_factor": beta,
                    **common_params,
                },
            }
            initial_conditions = {
                **subset_initial_states,
                "regime": jnp.full(
                    subset_ids.shape[0],
                    alive_id,
                    dtype=jnp.int32,
                ),
            }
            self._runs.append((params, initial_conditions))

    def _simulate_all(self):
        for params, initial_conditions in self._runs:
            self.model.simulate(
                params=params,
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=None,
                log_level="off",
                check_initial_conditions=False,
            )

    def setup(self):
        self._build()
        start = time.perf_counter()
        self._simulate_all()
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self._simulate_all()

    def peakmem_execution(self):
        self._simulate_all()

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
