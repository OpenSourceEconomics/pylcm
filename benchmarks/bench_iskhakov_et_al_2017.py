"""Iskhakov et al. (2017) retirement model benchmarks: solve and simulate.

The deterministic consumption-retirement model of Iskhakov, Jørgensen, Rust &
Schjerning (2017, Quantitative Economics 8(2), 317-365): working life with a
binary work/retire choice and EV1 taste shocks on that choice, an absorbing
retirement regime, and a terminal death regime. This is the canonical
benchmark model for discrete-continuous solution methods, so solver variants
(brute force, consumption-grid-free methods) are benchmarked here on the same
model. The solve benchmark uses a dense consumption grid — exactly the cost a
consumption-grid-free solver removes.

Model building blocks (utility, budget, labor income) are shared with
`lcm_examples.mortality`; regime transitions are deterministic (retirement is
chosen, death occurs at a fixed age) as in the original paper.
"""

import gc
import time

from benchmarks import _gpu_mem

_N_PERIODS = 10
_TASTE_SHOCK_SCALE = 0.1
_N_SUBJECTS = 100_000

_SOLVE_WEALTH_N_POINTS = 1_000
_SOLVE_CONSUMPTION_N_POINTS = 5_000

_SIMULATE_WEALTH_N_POINTS = 500
_SIMULATE_CONSUMPTION_N_POINTS = 500


def _make_model_and_params(*, wealth_n_points: int, consumption_n_points: int):
    """Build the taste-shock retirement model and its parameters.

    lcm and jax imports are deferred to the function body so ASV's forkserver
    never loads the XLA backend before forking workers (see
    `bench_aca_baseline._build` for the failure mode).
    """
    import jax.numpy as jnp

    from lcm import (
        AgeGrid,
        DiscreteGrid,
        ExtremeValueTasteShocks,
        LinSpacedGrid,
        Model,
        Regime,
        categorical,
    )
    from lcm.typing import DiscreteAction, ScalarInt
    from lcm_examples.mortality import (
        LaborSupply,
        borrowing_constraint,
        is_working,
        labor_income,
        next_wealth,
        utility_retirement,
        utility_working,
    )

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt
        dead: ScalarInt

    def next_regime_from_working(
        labor_supply: DiscreteAction,
        age: float,
        final_age_alive: float,
    ) -> ScalarInt:
        return jnp.where(
            age >= final_age_alive,
            RegimeId.dead,
            jnp.where(
                labor_supply == LaborSupply.retire,
                RegimeId.retirement,
                RegimeId.working_life,
            ),
        )

    def next_regime_from_retirement(age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.retirement)

    ages = AgeGrid(start=40, stop=40 + _N_PERIODS - 1, step="Y")
    last_age = ages.exact_values[-1]

    wealth_grid = LinSpacedGrid(start=1, stop=400, n_points=wealth_n_points)
    consumption_grid = LinSpacedGrid(start=1, stop=400, n_points=consumption_n_points)

    working_life = Regime(
        actions={
            "labor_supply": DiscreteGrid(LaborSupply),
            "consumption": consumption_grid,
        },
        states={"wealth": wealth_grid},
        state_transitions={"wealth": next_wealth},
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime_from_working,
        functions={
            "utility": utility_working,
            "labor_income": labor_income,
            "is_working": is_working,
        },
        taste_shocks=ExtremeValueTasteShocks(),
        active=lambda age: age < last_age,
    )

    retirement = Regime(
        transition=next_regime_from_retirement,
        actions={"consumption": consumption_grid},
        states={"wealth": wealth_grid},
        state_transitions={"wealth": next_wealth},
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={"utility": utility_retirement},
        active=lambda age: age < last_age,
    )

    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda _age: True,
    )

    model = Model(
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "final_age_alive": 40 + _N_PERIODS - 2,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "labor_income": {"wage": 10.0},
            "taste_shocks": {"scale": _TASTE_SHOCK_SCALE},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }
    return model, params


def _make_initial_conditions(n_subjects: int):
    import jax.numpy as jnp

    # Spread initial wealth so simulated paths cover early and late retirement
    # and the simulate benchmark exercises every regime's kernel.
    return {
        "age": jnp.full(n_subjects, 40.0),
        "wealth": jnp.linspace(1.0, 150.0, n_subjects),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }


def _clear_gpu_memory() -> None:
    import jax

    jax.clear_caches()
    gc.collect()


class IskhakovEtAl2017Solve:
    """Brute-force solve with EV1 taste shocks on the work/retire choice."""

    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
    timeout = 600

    def _build(self) -> None:
        self.model, self.model_params = _make_model_and_params(
            wealth_n_points=_SOLVE_WEALTH_N_POINTS,
            consumption_n_points=_SOLVE_CONSUMPTION_N_POINTS,
        )

    def setup(self) -> None:
        self._build()
        start = time.perf_counter()
        self.model.solve(params=self.model_params, log_level="off")
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self) -> None:
        self._build()

    def time_execution(self) -> None:
        self.model.solve(params=self.model_params, log_level="off")

    def peakmem_execution(self) -> None:
        self.model.solve(params=self.model_params, log_level="off")

    def teardown(self) -> None:
        _clear_gpu_memory()

    def track_compilation_time(self) -> float:
        return self._compile_time

    track_compilation_time.unit = "seconds"


class IskhakovEtAl2017SolveGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_iskhakov_et_al_2017"
    bench_class = "IskhakovEtAl2017Solve"


class IskhakovEtAl2017Simulate:
    """Simulate with Gumbel-max discrete choices from a pre-solved model."""

    version = "1"
    timeout = 600

    def _build(self) -> None:
        self.model, self.model_params = _make_model_and_params(
            wealth_n_points=_SIMULATE_WEALTH_N_POINTS,
            consumption_n_points=_SIMULATE_CONSUMPTION_N_POINTS,
        )
        self.period_to_regime_to_V_arr = self.model.solve(
            params=self.model_params, log_level="off"
        )
        self.initial_conditions = _make_initial_conditions(_N_SUBJECTS)

    def setup(self) -> None:
        self._build()
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self) -> None:
        self._build()

    def time_execution(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )

    def peakmem_execution(self) -> None:
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            log_level="off",
        )

    def teardown(self) -> None:
        _clear_gpu_memory()

    def track_compilation_time(self) -> float:
        return self._compile_time

    track_compilation_time.unit = "seconds"


class IskhakovEtAl2017SimulateGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_iskhakov_et_al_2017"
    bench_class = "IskhakovEtAl2017Simulate"
