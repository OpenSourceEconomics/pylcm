"""Collective and value-dependent workloads: build, compile, solve, simulate.

The declarations a collective regime makes cost something no singleton model
pays for, and each cost is measured on its own axis here:

- **model construction** — the phase scan, the lowering of the collective
  declarations, and the per-edge parameter discovery, with nothing compiled;
- **cold compilation** — the first solve, which traces and compiles every
  regime's kernels plus one fold per gated edge;
- **warm solve** — the same solve with the compilation cache hot, which is what
  an estimation loop pays per parameter vector;
- **simulation over cohort size** — the router evaluates each gate once per
  edge per period over the whole population, so this is where a cohort-size
  term would show;
- **transitive reference depth** — a value constraint may read a regime that
  itself reads another, and the solver orders each period's regimes by that
  chain. Depth is the axis on which an accidental quadratic would appear.

Host memory is tracked by `peakmem_*` and device memory by the `GpuPeakMem`
companions, on the same workloads.
"""

import gc
import time

from . import _gpu_mem

_N_PERIODS = 6

# Slack so wide that no participation constraint of the reference chain ever
# binds: depth is the only thing that varies between its parameterizations.
_CHAIN_SLACK = 1e9
_WEALTH_N_POINTS = 40
_CONSUMPTION_N_POINTS = 40

# `_gpu_mem` drives a wrapped class with NO arguments, so a parameterized
# workload has to name the single point its device-memory companion measures.
# Both pick the largest case, which is the one a device peak is interesting for.
_GPU_N_SUBJECTS = 100_000
_GPU_DEPTH = 8


def _make_model(*, n_subjects=None):
    from lcm_examples import collective_household

    model = collective_household.get_model(
        n_periods=_N_PERIODS,
        wealth_n_points=_WEALTH_N_POINTS,
        consumption_n_points=_CONSUMPTION_N_POINTS,
        n_subjects=n_subjects,
    )
    return model, collective_household.get_params()


def _make_initial_conditions(*, model, n_subjects):
    from lcm_examples import collective_household

    return collective_household.get_initial_conditions(
        n_subjects=n_subjects, model=model
    )


def _clear_gpu_memory():
    import jax

    jax.clear_caches()
    gc.collect()


class CollectiveHouseholdConstruct:
    """What `Model(...)` costs before anything is traced."""

    version = "1"
    timeout = 600

    def setup(self):
        # Warm the imports so the measured call is construction, not the first
        # import of jax and pylcm.
        _make_model()

    def time_execution(self):
        _make_model()

    def peakmem_execution(self):
        _make_model()

    def teardown(self):
        _clear_gpu_memory()


class CollectiveHouseholdSolve:
    """Backward induction over six periods of the marriage market."""

    version = "1"
    timeout = 900

    def _build(self):
        self.model, self.model_params = _make_model()

    def setup(self):
        self._build()
        start = time.perf_counter()
        self.model.solve(params=self.model_params, log_level="off")
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self):
        self._build()

    def time_execution(self):
        self.model.solve(params=self.model_params, log_level="off")

    def peakmem_execution(self):
        self.model.solve(params=self.model_params, log_level="off")

    def teardown(self):
        _clear_gpu_memory()

    def track_compilation_time(self):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class CollectiveHouseholdSolveGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_collective_household"
    bench_class = "CollectiveHouseholdSolve"


class CollectiveHouseholdSimulate:
    """Forward simulation, over cohort sizes.

    Routing is the part of simulation that is specific to a gated model: one
    gate evaluation per edge per period over the whole population. A
    cohort-size term that grew faster than linearly would show up between
    these three points.
    """

    version = "1"
    timeout = 900
    params = [1_000, 10_000, 100_000]
    param_names = ["n_subjects"]

    def _build(self, n_subjects):
        self.model, self.model_params = _make_model()
        self.period_to_regime_to_V_arr, self.dissolution_flags = self.model.solve(
            params=self.model_params, log_level="off", return_dissolution_flags=True
        )
        self.initial_conditions = _make_initial_conditions(
            model=self.model, n_subjects=n_subjects
        )

    def setup(self, n_subjects):
        self._build(n_subjects)
        start = time.perf_counter()
        self._simulate()
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self, n_subjects=_GPU_N_SUBJECTS):
        self._build(n_subjects)

    def _simulate(self):
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self.period_to_regime_to_V_arr,
            period_to_regime_to_dissolution_flags=self.dissolution_flags,
            log_level="off",
            seed=0,
        )

    def time_execution(self, n_subjects=_GPU_N_SUBJECTS):
        self._simulate()

    def peakmem_execution(self, n_subjects):
        self._simulate()

    def teardown(self, n_subjects):
        _clear_gpu_memory()

    def track_compilation_time(self, n_subjects):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class CollectiveHouseholdSimulateGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_collective_household"
    bench_class = "CollectiveHouseholdSimulate"


class ReferenceChainSolve:
    """Solve a chain of regimes each reading the previous one's value.

    Link `k` carries a participation constraint whose reference is link `k-1`
    in the SAME period, so the solver has to order each period's regimes by
    that chain before it can solve any of them.

    What depth actually varies is the ordering work and the number of regimes
    built and compiled — `2 * depth` of them, a link and its terminal. It is
    NOT the number of value arrays threaded into a decision: processing
    deduplicates references by regime, and every link names exactly one
    predecessor, so each link's kernel receives a single same-period array at
    every depth.

    Construction happens in `setup`, before the timer, so the tracked seconds
    are a first solve with no model building in them.
    """

    version = "1"
    timeout = 900
    params = [1, 2, 4, 8]
    param_names = ["depth"]

    def _build(self, depth):
        self.model, self.model_params = _make_reference_chain(depth=depth)

    def setup(self, depth):
        self._build(depth)
        start = time.perf_counter()
        self.model.solve(params=self.model_params, log_level="off")
        self._compile_time = time.perf_counter() - start

    def setup_for_gpu_measurement(self, depth=_GPU_DEPTH):
        self._build(depth)

    def time_execution(self, depth=_GPU_DEPTH):
        self.model.solve(params=self.model_params, log_level="off")

    def peakmem_execution(self, depth):
        self.model.solve(params=self.model_params, log_level="off")

    def teardown(self, depth):
        _clear_gpu_memory()

    def track_compilation_time(self, depth):
        return self._compile_time

    track_compilation_time.unit = "seconds"


class ReferenceChainSolveGpuPeakMem(_gpu_mem.GpuPeakMem):
    bench_module = "benchmarks.asv.bench_collective_household"
    bench_class = "ReferenceChainSolve"


def _make_reference_chain(*, depth):
    """Build `depth` collective regimes, each reading the previous one's value.

    Link `k` carries a participation constraint whose reference is link `k-1`
    in the SAME period, so the solver has to order the period's regimes by that
    chain and carry each link's value array into the next one's decision. Link
    0 has no reference, so the closure is exactly `depth - 1` edges deep.

    The constraints are declared with enormous slack so no cell is ever ruled
    out: the point is what the closure costs to resolve, not what it decides,
    and a chain that emptied a feasible set would measure a different model at
    each depth.

    Args:
        depth: Number of links in the chain.

    Returns:
        Tuple of the model and the params dict that solves it.

    """
    from lcm import AgeGrid, Model, categorical
    from lcm.typing import ScalarInt

    link_names = [f"link_{index}" for index in range(depth)]
    terminal_names = [f"{name}_terminal" for name in link_names]
    regime_id_class = categorical(ordered=False)(
        type(
            "ChainRegimeId",
            (),
            {
                "__annotations__": dict.fromkeys(
                    [*link_names, *terminal_names], ScalarInt
                )
            },
        )
    )

    regimes = {}
    params = {}
    for index, (name, terminal_name) in enumerate(
        zip(link_names, terminal_names, strict=True)
    ):
        reference_regime = link_names[index - 1] if index else None
        regimes[name] = _chain_link(
            terminal_name=terminal_name, reference_regime=reference_regime
        )
        regimes[terminal_name] = _chain_link_terminal()
        params[name] = {"koopmans_aggregator": {"discount_factor": 0.95}}
        if reference_regime is not None:
            params[name]["participation_f"] = {"slack": _CHAIN_SLACK}
            params[name]["participation_m"] = {"slack": _CHAIN_SLACK}
        params[terminal_name] = {}

    return (
        Model(
            regimes=regimes,
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=regime_id_class,
        ),
        params,
    )


def _chain_link(*, terminal_name, reference_regime):
    """Build one collective link of the reference chain."""
    from lcm import CollectiveUtility, Regime
    from lcm.transition import MarkovTransition

    kernels = _chain_kernels()
    return Regime(
        transition={terminal_name: MarkovTransition(kernels["to_terminal"])},
        active=lambda age: age < 1,
        states={"wealth": _chain_wealth_grid()},
        state_transitions={"wealth": kernels["next_wealth"]},
        actions={"consumption": _chain_consumption_grid()},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": kernels["utility"], "m": kernels["utility"]}
            )
        },
        constraints=_chain_constraints(reference_regime=reference_regime),
    )


def _chain_link_terminal():
    """Build one link's terminal regime."""
    from lcm import CollectiveUtility, Regime

    kernels = _chain_kernels()
    return Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _chain_wealth_grid()},
        actions={"consumption": _chain_consumption_grid()},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": kernels["utility"], "m": kernels["utility"]}
            )
        },
        constraints={"affordable": kernels["affordable"]},
    )


def _chain_constraints(*, reference_regime):
    """Build one link's constraints, with or without a reference to the previous."""
    from lcm import ProjectedRegimeValue, ValueDependentConstraint

    kernels = _chain_kernels()
    constraints = {"affordable": kernels["affordable"]}
    if reference_regime is None:
        return constraints
    # The reference regime is collective, so each partner reads her or his own
    # component of it, under a name of her or his own.
    for stakeholder in ("f", "m"):
        constraints[f"participation_{stakeholder}"] = ValueDependentConstraint(
            predicate=kernels[f"participation_{stakeholder}"],
            references={
                f"reference_{stakeholder}": ProjectedRegimeValue(
                    regime=reference_regime,
                    projection={"wealth": kernels["identity"]},
                    stakeholder=stakeholder,
                )
            },
        )
    return constraints


def _chain_wealth_grid():
    from lcm import LinSpacedGrid

    return LinSpacedGrid(start=1.0, stop=20.0, n_points=_WEALTH_N_POINTS)


def _chain_consumption_grid():
    from lcm import LinSpacedGrid

    return LinSpacedGrid(start=1.0, stop=20.0, n_points=_CONSUMPTION_N_POINTS)


def _chain_kernels():
    """Return the model functions every link of the chain shares.

    One dict of module-level closures, so two links built separately hold the
    SAME leaf callables and differ only in their references and their terminal
    target. That keeps the model definition honest about what varies with
    depth; it does not make the links share a compiled program. The solve-side
    dedup unit is a per-regime core built inside each regime's own
    `build_period_kernels` call, so each of the `2 * depth` links lowers and
    compiles its own regardless.
    """
    return _CHAIN_KERNELS


def _build_chain_kernels():
    import jax.numpy as jnp

    def utility(consumption):
        return jnp.log(0.5 * consumption)

    def affordable(wealth, consumption):
        return consumption <= wealth

    def next_wealth(wealth, consumption):
        return wealth - consumption

    def identity(wealth):
        return wealth

    def to_terminal(age):
        """A link is active for one period only, so it always hands over."""
        return jnp.ones_like(age, dtype=float)

    def participation_f(Q_f, reference_f, slack):
        return Q_f >= reference_f - slack

    def participation_m(Q_m, reference_m, slack):
        return Q_m >= reference_m - slack

    return {
        "utility": utility,
        "affordable": affordable,
        "next_wealth": next_wealth,
        "identity": identity,
        "to_terminal": to_terminal,
        "participation_f": participation_f,
        "participation_m": participation_m,
    }


_CHAIN_KERNELS = _build_chain_kernels()
