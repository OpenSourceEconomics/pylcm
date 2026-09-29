"""Parameters and laws are lowered only for the problems the starts require.

A `ByAge` case selected only at ages no required problem solves contributes no
parameter, and a regime without any demanded pair reads no parameter at all.
The values at the demanded pairs do not depend on which undemanded cases exist.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.regime_building.schedules as schedules_module
from lcm import (
    AgeGrid,
    AgeRange,
    AgeSpecializedFunction,
    ByAge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _retirement_utility(*, wealth: ContinuousState, bonus: float) -> FloatND:
    return wealth + bonus


def _early_stay(early_rate: float) -> FloatND:
    return jnp.asarray(early_rate)


def _early_die(early_rate: float) -> FloatND:
    return 1 - jnp.asarray(early_rate)


def _late_stay(late_rate: float) -> FloatND:
    return jnp.asarray(late_rate)


def _late_die(late_rate: float) -> FloatND:
    return 1 - jnp.asarray(late_rate)


_RETIREMENT_EXIT = ByAge(cases={AgeRange(start=55, stop=75): "dead"})


@categorical(ordered=False)
class LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _model(
    *,
    initial_regimes: Any,
    retirement_transitions: Any = _RETIREMENT_EXIT,
) -> Model:

    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(start=25, stop=45): {
                            "working": MarkovTransition(func=_early_stay),
                            "dead": MarkovTransition(func=_early_die),
                        },
                        AgeRange(start=45, stop=65): {
                            "working": MarkovTransition(func=_late_stay),
                            "dead": MarkovTransition(func=_late_die),
                        },
                    },
                    default="dead",
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "retirement": Regime(
                regime_transitions=retirement_transitions,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _retirement_utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=LifeId,
        initial_regimes=initial_regimes,
    )


def _leaves(*, tree: Any, prefix: str = "") -> frozenset[str]:
    if not isinstance(tree, dict):
        return frozenset({prefix})
    return frozenset(
        leaf
        for key, value in tree.items()
        for leaf in _leaves(tree=value, prefix=f"{prefix}/{key}")
    )


def _param_names(*, model: Model, regime: str) -> frozenset[str]:
    return frozenset(
        path.rsplit("/", 1)[-1]
        for path in _leaves(tree=model.get_params_template()[regime])
        if path
    )


@pytest.mark.parametrize(
    ("initial_regimes", "regime", "expected"),
    [
        ({25: "working"}, "working", {"discount_factor", "early_rate", "late_rate"}),
        ({55: "working"}, "working", {"discount_factor", "late_rate"}),
        ({55: "working"}, "retirement", set()),
        ({45: "dead"}, "working", set()),
        ({55: "retirement"}, "retirement", {"discount_factor", "bonus"}),
    ],
    ids=["both-cases", "late-case-only", "zero-node", "terminal-root", "late-root"],
)
def test_params_template_holds_only_the_parameters_demand_reads(
    *, initial_regimes: Any, regime: str, expected: set[str]
) -> None:
    """A regime's template lists exactly the parameters its demanded laws read."""
    assert (
        _param_names(model=_model(initial_regimes=initial_regimes), regime=regime)
        == expected
    )


def test_late_root_values_equal_the_first_age_root_values_at_shared_pairs() -> None:
    """Dropping an undemanded case leaves the demanded values bit-identical."""
    full = _model(initial_regimes={25: "working"}).solve(
        params={"discount_factor": 0.9, "early_rate": 0.7, "late_rate": 0.8},
        log_level="off",
    )
    late = _model(initial_regimes={55: "working"}).solve(
        params={"discount_factor": 0.9, "late_rate": 0.8}, log_level="off"
    )
    for period in (3, 4):
        np.testing.assert_array_equal(
            np.asarray(late.values[period]["working"]),
            np.asarray(full.values[period]["working"]),
        )


def test_simulate_runs_on_a_late_root_with_only_the_demanded_parameters() -> None:
    """A late start simulates with the late case's parameter alone."""
    params = {"discount_factor": 0.9, "late_rate": 0.8}
    model = _model(initial_regimes={55: "working"})
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.asarray([0.0, 1.0]),
            "age": jnp.asarray([55.0, 55.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["working"]),
        },
        solution=model.solve(params=params, log_level="off"),
        log_level="off",
        seed=0,
    )
    assert set(result.to_dataframe()["age"]) <= {55, 65, 75}


def test_zero_node_regime_builds_no_transition_programs() -> None:
    """A regime no start demands gets no kernel and no transition program."""
    model = _model(initial_regimes={45: "dead"})
    working = model._regimes["working"]
    assert (
        working.solution.compute_regime_transition_probs,
        working.solution.validation_regime_transition_probs,
        working.simulation.compute_regime_transition_probs,
        dict(working.solution.period_kernels),
        dict(working.simulation.Q_and_F),
    ) == (None, None, None, {}, {})


def test_zero_node_regime_keeps_a_declared_law_without_period_dispatch() -> None:
    """An undemanded schedule is not lowered into a period-dispatched union."""
    law = (
        _model(initial_regimes={45: "dead"})
        ._engine_user_regimes["working"]
        .regime_transitions
    )
    assert isinstance(law, Mapping)
    cell = law["working"]
    assert isinstance(cell, MarkovTransition)
    assert cell.func is _early_stay


def _age_specialized_model(*, built_ages: list[float], initial_regimes: Any) -> Model:
    def build_utility(age: float) -> Any:
        built_ages.append(float(age))
        return _utility

    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge(cases={AgeRange(start=25, stop=65): "dead"}),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={
                    "utility": AgeSpecializedFunction(
                        build=build_utility, signature=lambda age: age
                    )
                },
            ),
            "retirement": Regime(
                regime_transitions=ByAge(cases={AgeRange(start=55, stop=75): "dead"}),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=LifeId,
        initial_regimes=initial_regimes,
    )


def test_age_specialized_factory_is_built_only_at_demanded_ages() -> None:
    """A per-age factory is called at the regime's solved ages and no other."""
    built_ages: list[float] = []
    _age_specialized_model(built_ages=built_ages, initial_regimes={55: "working"})
    assert set(built_ages) == {55.0}


def test_age_specialized_factory_of_a_zero_node_regime_is_never_built() -> None:
    """A regime no start demands never calls its per-age factory."""
    built_ages: list[float] = []
    _age_specialized_model(built_ages=built_ages, initial_regimes={55: "retirement"})
    assert built_ages == []


def _health_stay(*, health: ContinuousState, early_rate: float) -> FloatND:
    return jnp.asarray(early_rate) + 0 * health


def _health_die(*, health: ContinuousState, early_rate: float) -> FloatND:
    return 1 - jnp.asarray(early_rate) + 0 * health


def _broadcast_health_model(initial_regimes: Any) -> Model:
    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(start=25, stop=45): {
                            "working": MarkovTransition(func=_health_stay),
                            "dead": MarkovTransition(func=_health_die),
                        },
                        AgeRange(start=45, stop=65): {
                            "working": MarkovTransition(func=_late_stay),
                            "dead": MarkovTransition(func=_late_die),
                        },
                    },
                    default="dead",
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        states={"health": _WEALTH},
        state_transitions={"health": fixed_transition("health")},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=_WorkingDeadId,
        initial_regimes=initial_regimes,
    )


@categorical(ordered=False)
class _WorkingDeadId:
    working: ScalarInt
    dead: ScalarInt


@pytest.mark.parametrize(
    ("initial_regimes", "expected"),
    [({25: "working"}, False), ({55: "working"}, True)],
    ids=["early-case-demanded", "early-case-undemanded"],
)
def test_a_state_read_only_by_an_undemanded_case_is_not_live(
    *, initial_regimes: Any, expected: bool
) -> None:
    """Liveness counts only the laws of demanded cases."""
    model = _broadcast_health_model(initial_regimes)
    assert ("health" in model.pruned_variables["working"]) is expected


def _gated_fold_model(
    *,
    initial_regimes: Any,
    later_source_law: object | None = None,
) -> Model:
    fallback = ProjectedRegimeValue(regime="fallback", projection={"wealth": _identity})
    target_law = {
        "target": ValueDependentTransition(
            probability=MarkovTransition(func=_prob_one),
            gate=_gate,
            routes={"only": StakeholderRoute(fallback=fallback)},
            gate_references={
                "V_reference": ProjectedRegimeValue(
                    regime="reference", projection={"wealth": _identity}
                )
            },
        )
    }
    terminal = Regime(
        regime_transitions=None,
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )
    return Model(
        regimes={
            "source": Regime(
                regime_transitions=ByAge(
                    cases={
                        40: target_law,
                        **({} if later_source_law is None else {45: later_source_law}),
                    }
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "target": terminal,
            "reference": terminal,
            "fallback": terminal,
        },
        ages=AgeGrid(start=40, stop=50, step="5Y"),
        regime_id_class=_GatedId,
        initial_regimes=initial_regimes,
    )


@categorical(ordered=False)
class _GatedId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _gate(*, V_target: FloatND, V_reference: FloatND) -> BoolND:
    return V_target > V_reference


def _identity(wealth: ContinuousState) -> ContinuousState:
    return wealth


def test_a_gate_fold_exists_only_where_its_source_lands() -> None:
    """A target solved at a later age for another start gets no fold there."""
    model = _gated_fold_model(initial_regimes={40: "source", 50: "target"})
    folds = model._regimes["source"].gated_edges["target"].folds_by_period
    assert set(folds) == {1}


def test_a_gate_fold_exists_only_where_the_selected_case_declares_it() -> None:
    """A source solved at a later age whose case there declares no gate adds no
    fold and requires no gate reference at the next age."""
    model = _gated_fold_model(
        initial_regimes={40: "source", 45: "source", 50: "target"},
        later_source_law="fallback",
    )
    folds = model._regimes["source"].gated_edges["target"].folds_by_period
    assert set(folds) == {1}


def test_a_gated_target_solved_where_its_source_never_stands_solves() -> None:
    """A target solved for another start is not folded for this source there."""
    model = _gated_fold_model(
        initial_regimes={
            40: ("source", "target", "reference", "fallback"),
            50: "target",
        }
    )
    values = model.solve(params={"discount_factor": 0.9}, log_level="off").values
    assert {(period, name) for period, by in values.items() for name in by} == {
        (0, "source"),
        (0, "target"),
        (0, "reference"),
        (0, "fallback"),
        (1, "target"),
        (1, "reference"),
        (1, "fallback"),
        (2, "target"),
    }


def test_an_undemanded_case_is_never_lowered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construction lowers only the laws a required problem selects."""
    lowered: list[object] = []
    original = schedules_module._lower_side

    def recording_lower_side(**kwargs: Any) -> object:
        lowered.extend(kwargs["law_by_period"].values())
        return original(**kwargs)

    monkeypatch.setattr(schedules_module, "_lower_side", recording_lower_side)
    model = _model(initial_regimes={55: "working"})
    early = model.user_regimes["working"].regime_transitions.laws[0]  # ty: ignore[unresolved-attribute]
    assert all(law is not early for law in lowered)


def test_a_regime_whose_laws_cover_no_available_age_builds_while_unrequired() -> None:
    """A law declared only at the last age is unused, not an error."""
    model = _model(
        initial_regimes={25: "working"},
        retirement_transitions=ByAge(cases={75: "dead"}),
    )
    assert model.reachability.nodes.isdisjoint(
        {(age, "retirement") for age in model.ages.exact_values}
    )


def test_requiring_a_regime_whose_laws_cover_no_available_age_fails() -> None:
    """A required problem where the regime supplies no law raises and names it."""
    with pytest.raises(
        ModelInitializationError, match="requires 'retirement' at age 55"
    ):
        _model(
            initial_regimes={55: "retirement"},
            retirement_transitions=ByAge(cases={75: "dead"}),
        )
