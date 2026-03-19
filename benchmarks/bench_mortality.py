"""Mortality model benchmark: solve + simulate."""

import time

import jax.numpy as jnp

from lcm_examples import mortality

_N_SUBJECTS = 1_000


class TimeMortality:
    timeout = 600

    def setup(self):
        self.model = mortality.get_model(n_periods=4)
        self.model_params = mortality.get_params(n_periods=4)
        self.initial_conditions = {
            "age": jnp.full(_N_SUBJECTS, 40.0),
            "wealth": jnp.full(_N_SUBJECTS, 100.0),
            "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        }
        start = time.perf_counter()
        self.model.solve_and_simulate(
            self.model_params, self.initial_conditions, log_level="off"
        )
        self._warmup_time = time.perf_counter() - start

    def time_mortality(self):
        self.model.solve_and_simulate(
            self.model_params, self.initial_conditions, log_level="off"
        )

    def track_warmup(self):
        return self._warmup_time

    track_warmup.unit = "seconds"
