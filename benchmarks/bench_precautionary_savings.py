"""Precautionary savings benchmarks: solve, simulate, grid types."""

import gc
import time

_N_SUBJECTS = 1_000


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
        "regime": jnp.zeros(n_subjects, dtype=jnp.int32),
    }


def _clear_gpu_memory():
    import jax

    jax.clear_caches()
    gc.collect()


class PrecautionarySavingsSolve:
    params = [200, 500]
    param_names = ["n_points"]
    timeout = 600

    def setup(self, n_points):
        self.model, self.model_params = _make_model(
            wealth_n_points=n_points,
            consumption_n_points=n_points,
        )
        start = time.perf_counter()
        self.model.solve(params=self.model_params, log_level="off")
        self._compile_time = time.perf_counter() - start

    def time_execution(self, n_points):
        self.model.solve(params=self.model_params, log_level="off")

    def peakmem_execution(self, n_points):
        self.model.solve(params=self.model_params, log_level="off")

    def teardown(self, n_points):
        _clear_gpu_memory()

    def track_compilation_time(self, n_points):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulate:
    params = [[10_000, 100_000]]
    param_names = ["n_subjects"]
    timeout = 600

    def setup(self, n_subjects):
        self.model, self.model_params = _make_model()
        self.period_to_regime_to_V_arr = self.model.solve(
            params=self.model_params, log_level="off"
        )
        self.initial_conditions = _make_initial_conditions(n_subjects)
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def time_execution(self, n_subjects):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )

    def peakmem_execution(self, n_subjects):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )

    def teardown(self, n_subjects):
        _clear_gpu_memory()

    def track_compilation_time(self, n_subjects):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsSimulateWithSolve:
    timeout = 600

    def setup(self):
        self.model, self.model_params = _make_model(
            wealth_n_points=200,
            consumption_n_points=200,
        )
        self.initial_conditions = _make_initial_conditions(_N_SUBJECTS)
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

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
        _clear_gpu_memory()

    def track_compilation_time(self):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class PrecautionarySavingsGridLookup:
    params = [[2000], ["lin", "irreg"]]
    param_names = ["n_points", "grid_type"]
    timeout = 600

    def setup(self, n_points, grid_type):
        self.model, self.model_params = _make_model(
            wealth_grid_type=grid_type,
            wealth_n_points=n_points,
            consumption_n_points=n_points,
        )
        self.initial_conditions = _make_initial_conditions(100)
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def time_execution(self, n_points, grid_type):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )

    def peakmem_execution(self, n_points, grid_type):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )

    def teardown(self, n_points, grid_type):
        _clear_gpu_memory()

    def track_compilation_time(self, n_points, grid_type):
        return self._compile_time

    track_compilation_time.unit = "seconds"
