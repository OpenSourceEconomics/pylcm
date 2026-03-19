"""Mortality model benchmark: solve + simulate."""

import gc
import time

_N_SUBJECTS = 1_000


class TimeMortality:
    timeout = 600

    def setup(self):
        import jax.numpy as jnp

        from lcm_examples import mortality

        self.model = mortality.get_model(n_periods=4)
        self.model_params = mortality.get_params(n_periods=4)
        self.initial_conditions = {
            "age": jnp.full(_N_SUBJECTS, 40.0),
            "wealth": jnp.full(_N_SUBJECTS, 100.0),
            "regime": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        }
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            log_level="off",
        )
        self._warmup_time = time.perf_counter() - start

    def time_mortality(self):
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
