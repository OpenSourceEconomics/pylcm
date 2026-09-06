"""Revision-independent workloads for paired GridSearch measurements.

This module is imported by the external worker before JAX is configured.  Keep JAX,
``lcm``, and model imports inside the builders so the worker can select the backend,
precision, and (for the distributed row) CPU topology first.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

ScenarioName = Literal[
    "singleton-hard-max",
    "singleton-ev1",
    "collective-gs-vd",
    "distributed-co-map",
    "folded-hard-max",
    "aca-a3-c16",
    "aca-a3-c64",
    "aca-a3-c256",
    "aca-a6-c16",
    "aca-a6-c64",
    "aca-a6-c256",
]


@dataclass(frozen=True, kw_only=True)
class ScenarioSpec:
    """Static identity and topology requirements of one paired workload."""

    name: ScenarioName
    description: str
    topology: Literal["selected-backend", "cpu-4"]
    expected_folded: bool = False
    expected_collective: bool = False
    expected_distributed: bool = False
    expected_head_disposition: Literal["planned", "dense"] = "planned"
    expected_head_disposition_reason: str | None = None
    expected_taste_shocks: bool = False
    expected_gs_vd: bool = False
    aca_assets_n_points: int | None = None
    aca_consumption_n_points: int | None = None


SCENARIOS = MappingProxyType(
    {
        "singleton-hard-max": ScenarioSpec(
            name="singleton-hard-max",
            description="Precautionary-savings singleton hard maximum.",
            topology="selected-backend",
        ),
        "singleton-ev1": ScenarioSpec(
            name="singleton-ev1",
            description="Consumption-retirement singleton EV1 expected maximum.",
            topology="selected-backend",
            expected_head_disposition="dense",
            expected_head_disposition_reason=(
                "deliberately_dense:ev1_canonical_reduction_order"
            ),
            expected_taste_shocks=True,
        ),
        "collective-gs-vd": ScenarioSpec(
            name="collective-gs-vd",
            description=(
                "Collective household with value-dependent constraints and edges."
            ),
            topology="selected-backend",
            expected_collective=True,
            expected_head_disposition="dense",
            expected_head_disposition_reason=(
                "deliberately_dense:collective_resource_regression"
            ),
            expected_gs_vd=True,
        ),
        "distributed-co-map": ScenarioSpec(
            name="distributed-co-map",
            description=(
                "Singleton hard maximum co-mapped over one four-way fixed state."
            ),
            topology="cpu-4",
            expected_distributed=True,
        ),
        "folded-hard-max": ScenarioSpec(
            name="folded-hard-max",
            description="Singleton hard maximum with a full folded IID shock axis.",
            topology="selected-backend",
            expected_folded=True,
        ),
        **{
            f"aca-a{assets}-c{consumption}": ScenarioSpec(
                name=f"aca-a{assets}-c{consumption}",
                description=(
                    "Full 18-regime ACA baseline with "
                    f"{assets} asset and {consumption} consumption points."
                ),
                topology="selected-backend",
                aca_assets_n_points=assets,
                aca_consumption_n_points=consumption,
            )
            for assets in (3, 6)
            for consumption in (16, 64, 256)
        },
    }
)

# Target-checkout sources used by otherwise HEAD-owned workload definitions.  The
# controller requires these files to be byte-identical across the pair.  Production
# ``lcm`` and ``_lcm`` sources are intentionally absent: they are what is compared.
TARGET_SCENARIO_SOURCES = (
    "src/lcm_examples/collective_household.py",
    "src/lcm_examples/mortality.py",
    "src/lcm_examples/precautionary_savings.py",
)

# Every HEAD-owned source that can alter the worker or workload.  This is deliberately
# broader than the three new modules: the first three rows reuse established ASV model
# builders, so those builders are part of the immutable external harness too.
EXTERNAL_HARNESS_SOURCES = (
    "benchmarks/asv/_gpu_mem.py",
    "benchmarks/asv/bench_collective_household.py",
    "benchmarks/asv/bench_iskhakov_et_al_2017.py",
    "benchmarks/asv/bench_precautionary_savings.py",
    "benchmarks/grid_search_pair.py",
    "benchmarks/grid_search_pair_scenarios.py",
    "benchmarks/grid_search_pair_worker.py",
)


def build_scenario(*, name: ScenarioName) -> tuple[Any, dict[str, Any]]:
    """Build one workload after the worker has configured JAX."""
    spec = SCENARIOS[name]
    if spec.aca_assets_n_points is not None:
        if spec.aca_consumption_n_points is None:
            raise RuntimeError("ACA scenario omitted its consumption-grid width.")
        return _build_aca_baseline(
            assets_n_points=spec.aca_assets_n_points,
            consumption_n_points=spec.aca_consumption_n_points,
        )

    builders = {
        "singleton-hard-max": _build_singleton_hard_max,
        "singleton-ev1": _build_singleton_ev1,
        "collective-gs-vd": _build_collective_gs_vd,
        "distributed-co-map": _build_distributed_co_map,
        "folded-hard-max": _build_folded_hard_max,
    }
    return builders[name]()


def _build_aca_baseline(
    *, assets_n_points: int, consumption_n_points: int
) -> tuple[Any, dict[str, Any]]:
    from dataclasses import replace

    import aca_model.benchmark as aca_benchmark
    from aca_model.agent.preferences import BenchmarkPrefType

    from lcm import DiscreteGrid

    original_grid = aca_benchmark.BENCHMARK_GRID_CONFIG
    aca_benchmark.BENCHMARK_GRID_CONFIG = replace(
        original_grid,
        n_assets_gridpoints=assets_n_points,
        n_consumption_dollars_gridpoints=consumption_n_points,
    )
    try:
        model = aca_benchmark.create_benchmark_model(
            n_subjects=1,
            pref_type_grid=DiscreteGrid(category_class=BenchmarkPrefType),
        )
    finally:
        aca_benchmark.BENCHMARK_GRID_CONFIG = original_grid
    params = aca_benchmark.get_benchmark_params(model=model)[2]
    return model, params


def _build_singleton_hard_max() -> tuple[Any, dict[str, Any]]:
    from benchmarks.asv.bench_precautionary_savings import _make_model

    return _make_model(wealth_n_points=500, consumption_n_points=500)


def _build_singleton_ev1() -> tuple[Any, dict[str, Any]]:
    from benchmarks.asv.bench_iskhakov_et_al_2017 import _make_model_and_params

    return _make_model_and_params(
        wealth_n_points=1_000,
        consumption_n_points=5_000,
        solver="brute_force",
    )


def _build_collective_gs_vd() -> tuple[Any, dict[str, Any]]:
    from benchmarks.asv.bench_collective_household import _make_model

    return _make_model()


def _build_distributed_co_map() -> tuple[Any, dict[str, Any]]:
    import jax.numpy as jnp

    from lcm import (
        AgeGrid,
        DiscreteGrid,
        LinSpacedGrid,
        Model,
        Regime,
        categorical,
        fixed_transition,
    )
    from lcm.typing import ScalarInt

    @categorical(ordered=False)
    class RegimeId:
        working: ScalarInt
        retired: ScalarInt

    @categorical(ordered=True)
    class PermanentType:
        lowest: ScalarInt
        low: ScalarInt
        high: ScalarInt
        highest: ScalarInt

    def working_utility(*, wealth, consumption, permanent_type):
        return jnp.log(consumption) + 0.001 * wealth + 0.1 * permanent_type

    def retired_utility(*, wealth, permanent_type):
        return 0.5 * wealth + 0.1 * permanent_type

    def next_wealth(*, wealth, consumption):
        return wealth - consumption

    def affordable(*, wealth, consumption):
        return consumption <= wealth

    def next_regime(*, age):
        return jnp.where(age >= 4, RegimeId.retired, RegimeId.working)

    wealth = LinSpacedGrid(start=1.0, stop=100.0, n_points=500)
    consumption = LinSpacedGrid(start=0.1, stop=100.0, n_points=500)
    model = Model(
        regimes={
            "working": Regime(
                transition=next_regime,
                active=lambda age: age < 5,
                states={"wealth": wealth},
                state_transitions={"wealth": next_wealth},
                actions={"consumption": consumption},
                functions={"utility": working_utility},
                constraints={"affordable": affordable},
            ),
            "retired": Regime(
                transition=None,
                active=lambda age: age >= 5,
                states={"wealth": wealth},
                functions={"utility": retired_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
        states={
            "permanent_type": DiscreteGrid(
                category_class=PermanentType,
                distributed=True,
            )
        },
        state_transitions={"permanent_type": fixed_transition("permanent_type")},
    )
    return model, {"discount_factor": 0.95}


def _build_folded_hard_max() -> tuple[Any, dict[str, Any]]:
    import jax.numpy as jnp

    from lcm import (
        AgeGrid,
        LinSpacedGrid,
        Model,
        NormalIIDProcess,
        Regime,
        categorical,
    )
    from lcm.typing import ScalarInt

    @categorical(ordered=False)
    class RegimeId:
        working: ScalarInt
        retired: ScalarInt

    def working_utility(*, consumption, wage_shock):
        return jnp.log(consumption) + wage_shock

    def retired_utility(*, wealth):
        return jnp.log(wealth)

    def next_wealth(*, wealth, consumption):
        return wealth - consumption

    def affordable(*, wealth, consumption):
        return consumption <= wealth

    def next_regime(*, age):
        return jnp.where(age >= 3, RegimeId.retired, RegimeId.working)

    wealth = LinSpacedGrid(start=1.0, stop=100.0, n_points=500)
    consumption = LinSpacedGrid(start=0.1, stop=100.0, n_points=500)
    model = Model(
        regimes={
            "working": Regime(
                transition=next_regime,
                active=lambda age: age < 4,
                states={
                    "wealth": wealth,
                    "wage_shock": NormalIIDProcess(
                        n_points=9,
                        gauss_hermite=True,
                        mu=0.0,
                        sigma=0.25,
                        fold=True,
                    ),
                },
                state_transitions={"wealth": next_wealth},
                actions={"consumption": consumption},
                functions={"utility": working_utility},
                constraints={"affordable": affordable},
            ),
            "retired": Regime(
                transition=None,
                active=lambda age: age >= 4,
                states={"wealth": wealth},
                functions={"utility": retired_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=RegimeId,
    )
    return model, {"discount_factor": 0.95}
