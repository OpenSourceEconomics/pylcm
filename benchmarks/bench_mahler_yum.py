"""End-to-end benchmark for the Mahler & Yum (2024) replication model."""

import gc
import time

_N_SUBJECTS = 100


class TimeMahlerYum:
    timeout = 1200

    def setup(self):
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
        common_params, initial_states, _discount_factor_type = create_inputs(
            seed=0,
            n_simulation_subjects=_N_SUBJECTS,
            **start_params_without_beta,
        )
        self.model_params = {
            "alive": {
                "discount_factor": START_PARAMS["beta"]["mean"],
                **common_params,
            },
        }
        self.initial_conditions = {
            **initial_states,
            "regime": jnp.full(
                _N_SUBJECTS,
                self.model.regime_names_to_ids["alive"],
                dtype=jnp.int32,
            ),
        }
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )
        self._warmup_time = time.perf_counter() - start

    def time_mahler_yum(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )

    def peakmem_mahler_yum(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()

    def track_warmup(self):
        return self._warmup_time

    track_warmup.unit = "seconds"
