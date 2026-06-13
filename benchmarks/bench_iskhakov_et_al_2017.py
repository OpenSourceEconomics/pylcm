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
# Matched-precision calibration for the head-to-head: 200 cubically clustered
# savings nodes solve this model at least as accurately as the dense
# consumption grid does wherever the consumption grid's truncation at its
# start point does not bind, so the timing comparison is apples-to-apples
# (against the retirement leg's closed form, DC-EGM at this grid is at
# max error ~1e-3 while the brute leg floors at ~2e-2 from grid bias and
# is unboundedly wrong below its consumption-grid start). More savings
# nodes buy further accuracy only DC-EGM can reach — and lengthen the
# sequential envelope scan — so they would unbalance the comparison.
_SOLVE_SAVINGS_N_POINTS = 200

_SIMULATE_WEALTH_N_POINTS = 500
_SIMULATE_CONSUMPTION_N_POINTS = 500


def _make_model_and_params(
    *,
    wealth_n_points: int,
    consumption_n_points: int,
    solver: str = "brute_force",
):
    """Build the taste-shock retirement model and its parameters.

    The `"dcegm"` solver variant is the same economic model in the DC-EGM
    contract: the wealth transition consumes the post-decision savings, the
    borrowing constraint is dropped (the savings grid's lower bound enforces
    it), and `resources`/`savings`/`inverse_marginal_utility` are declared as
    regime functions. The consumption grid plays no role in the DC-EGM solve.

    lcm and jax imports are deferred to the function body so ASV's forkserver
    never loads the XLA backend before forking workers (see
    `bench_aca_baseline._build` for the failure mode).
    """
    import jax.numpy as jnp

    from lcm import (
        AgeGrid,
        DiscreteGrid,
        ExtremeValueTasteShocks,
        IrregSpacedGrid,
        LinSpacedGrid,
        Model,
        Regime,
        categorical,
    )
    from lcm.solvers import DCEGM
    from lcm.typing import (
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
        ScalarInt,
    )
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

    def resources(wealth: ContinuousState) -> FloatND:
        return wealth

    def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
        return resources - consumption

    def next_wealth_from_savings(
        savings: FloatND, labor_income: FloatND, interest_rate: float
    ) -> ContinuousState:
        return (1 + interest_rate) * savings + labor_income

    def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
        return 1.0 / marginal_continuation

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

    if solver == "dcegm":
        dcegm_solver = DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            # Cubically clustered toward the borrowing limit: the value
            # function curves hardest where the constraint starts to bind,
            # and a uniform grid under-resolves the lowest wealth nodes by
            # orders of magnitude.
            savings_grid=IrregSpacedGrid(
                points=tuple(
                    400.0 * (i / (_SOLVE_SAVINGS_N_POINTS - 1)) ** 3
                    for i in range(_SOLVE_SAVINGS_N_POINTS)
                )
            ),
        )
        dcegm_functions = {
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        }
        working_life = working_life.replace(
            state_transitions={"wealth": next_wealth_from_savings},
            constraints={},
            functions={**dict(working_life.functions), **dcegm_functions},
            solver=dcegm_solver,
        )
        retirement = retirement.replace(
            state_transitions={"wealth": next_wealth_from_savings},
            constraints={},
            functions={**dict(retirement.functions), **dcegm_functions},
            solver=dcegm_solver,
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


class IskhakovEtAl2017DCEGMSolve:
    """DC-EGM solve of the same model: Euler inversion replaces the grid search.

    Calibrated to the brute-force benchmark's precision (see the savings-grid
    constant). Reading the head-to-head: the upper-envelope refinement is a
    sequential `lax.scan` over the savings nodes, so on a GPU — which thrives
    on the brute solver's one huge parallel reduction — DC-EGM trades parallel
    width for a shorter critical path and can lose on wall clock while using a
    fraction of the memory. On CPU the same configuration beats brute force
    outright.
    """

    version = "2"
    timeout = 600

    def _build(self) -> None:
        self.model, self.model_params = _make_model_and_params(
            wealth_n_points=_SOLVE_WEALTH_N_POINTS,
            consumption_n_points=_SOLVE_CONSUMPTION_N_POINTS,
            solver="dcegm",
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


class IskhakovEtAl2017DCEGMSolveGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.bench_iskhakov_et_al_2017"
    bench_class = "IskhakovEtAl2017DCEGMSolve"


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
