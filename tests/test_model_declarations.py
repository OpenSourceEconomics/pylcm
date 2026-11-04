"""Inspect the declarations used by construction without evaluating the model."""

from unittest.mock import patch

import pytest
from beartype import beartype

from lcm import (
    DiscreteGrid,
    InitialNodes,
    Model,
    Periods,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import Phase, ReferenceName, ScalarInt


@categorical(ordered=False)
class Regimes:
    alive: ScalarInt
    final: ScalarInt


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    good: ScalarInt


def utility(*, health: ScalarInt) -> float:
    del health
    raise AssertionError("Declaration inspection must not evaluate utility.")


def simulation_utility(*, health: ScalarInt, coefficient: float) -> float:
    del health, coefficient
    raise AssertionError("Declaration inspection must not evaluate utility.")


def terminal() -> float:
    return 0.0


@pytest.fixture
def model() -> Model:
    return Model(
        n_periods=2,
        regime_id_class=Regimes,
        regimes={
            "alive": Regime(
                states={"health": DiscreteGrid(category_class=Health)},
                state_transitions={"health": fixed_transition("health")},
                functions={
                    "utility": Phased(
                        solve=beartype(utility), simulate=beartype(simulation_utility)
                    )
                },
            ),
            "final": Regime(functions={"utility": terminal}),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"final": Periods(values=(0,))}},
        enable_jit=False,
    )


def test_get_declarations_preserves_the_constructed_graph(model: Model) -> None:
    assert model.get_declarations().graph is model.graph


def test_get_declarations_retains_production_domains(model: Model) -> None:
    declarations = model.get_declarations()
    assert (
        declarations.phases[0, "alive", "solve"].grid_states["health"]
        is (model.user_regimes["alive"].states["health"])
    )


@pytest.mark.parametrize(
    ("phase", "arg_names"),
    [("solve", ("health",)), ("simulate", ("health", "coefficient"))],
)
def test_get_declarations_publishes_phase_specific_inputs(
    *, model: Model, phase: Phase, arg_names: tuple[ReferenceName, ...]
) -> None:
    declarations = model.get_declarations()
    func = declarations.phases[0, "alive", phase].functions["utility"]
    assert (
        next(
            record.arg_names
            for record in declarations.functions
            if record.function is func
        )
        == arg_names
    )


def test_get_declarations_never_prepares_a_runtime_generation(model: Model) -> None:
    with patch.object(Model, "_acquire_generation", side_effect=AssertionError):
        assert model.get_declarations().schema_version == 1


def test_get_declarations_mappings_are_read_only(model: Model) -> None:
    with pytest.raises(TypeError):
        model.get_declarations().regimes_by_period[0] = {}  # ty: ignore[invalid-assignment]


def test_get_declarations_resolves_cached_age_constraints() -> None:
    from lcm import AgeGrid, AgeSpecializedFunction  # noqa: PLC0415
    from lcm.typing import UserFunction  # noqa: PLC0415

    built: dict[float, UserFunction] = {}

    def build(age: float) -> UserFunction:
        def eligible(*, health: ScalarInt) -> bool:
            del health
            raise AssertionError("Constraint inspection must not evaluate its body.")

        built[age] = eligible
        return eligible

    candidate = Model(
        ages=AgeGrid(start=20, inclusive_stop=22, step="Y"),
        regime_id_class=Regimes,
        regimes={
            "alive": Regime(
                states={"health": DiscreteGrid(category_class=Health)},
                state_transitions={"health": fixed_transition("health")},
                functions={"utility": utility},
                constraints={
                    "eligible": AgeSpecializedFunction(
                        build=build, signature=lambda age: age
                    )
                },
            ),
            "final": Regime(functions={"utility": terminal}),
        },
        initial_nodes=InitialNodes(by_age={20: "alive"}),
        edges={"alive": {"alive": 20, "final": 21}},
        enable_jit=False,
    )
    before = dict(built)
    declarations = candidate.get_declarations()
    assert built == before
    for period, age in ((0, 20.0), (1, 21.0)):
        for phase in ("solve", "simulate"):
            constraint = declarations.phases[period, "alive", phase].constraints[
                "eligible"
            ]
            assert constraint.declaration is built[age]
            assert constraint.arg_names == ("health",)
            assert next(
                record.arg_names
                for record in declarations.functions
                if record.function is built[age]
            ) == ("health",)


def test_get_declarations_skips_undemanded_simulation_age_factory() -> None:
    from lcm import AgeGrid, AgeSpecializedFunction  # noqa: PLC0415
    from lcm.typing import UserFunction  # noqa: PLC0415

    built: dict[float, UserFunction] = {}

    def build(age: float) -> UserFunction:
        def simulated(*, health: ScalarInt) -> float:
            del health
            raise AssertionError("Inspection must not evaluate simulated utility.")

        built[age] = simulated
        return simulated

    candidate = Model(
        ages=AgeGrid(start=20, inclusive_stop=22, step="Y"),
        regime_id_class=Regimes,
        regimes={
            "alive": Regime(
                states={"health": DiscreteGrid(category_class=Health)},
                state_transitions={"health": fixed_transition("health")},
                functions={
                    "utility": Phased(
                        solve=utility,
                        simulate=AgeSpecializedFunction(
                            build=build, signature=lambda age: age
                        ),
                    )
                },
            ),
            "final": Regime(functions={"utility": terminal}),
        },
        initial_nodes=InitialNodes(by_age={20: "alive"}),
        edges=Phased(
            solve={"alive": {"alive": 20, "final": 21}},
            simulate={"alive": {"final": 20}},
        ),
        enable_jit=False,
    )
    assert tuple(built) == (20.0,)
    before = dict(built)
    declarations = candidate.get_declarations()
    assert built == before
    assert (1, "alive", "solve") in declarations.phases
    assert (1, "alive", "simulate") not in declarations.phases
    assert (
        declarations.phases[0, "alive", "simulate"].functions["utility"] is built[20.0]
    )
