"""Collective and value-dependent choice declared inside the slots a regime has.

A regime says who its stakeholders are in `functions["utility"]`, what a
value-reading feasibility constraint is in `constraints`, and where a
value-dependent transition routes in `transition`. Nothing about the model
changes: the declarations are lowered onto the same stakeholders, value
constraints, references and gated edges the engine has always run, so a model
written this way solves to the numbers the same model written the long way
solves to.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    GatedEdge,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.transition import MarkovTransition
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _WAGE_3,
    _identity_wage,
    _ir_f,
    _ir_m,
    _make_dissolution_regimes,
    _no_dissolution_gate,
    _prob_one,
    _u_married_ir_f,
    _u_married_ir_m,
    _u_single_f_ir,
    _u_single_m_ir,
    _u_zero,
    _u_zero_collective,
)

_AGES = AgeGrid(start=0, stop=3, step="Y")

_PARAMS = {
    "married": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "married_ir": {
        "koopmans_aggregator": {"discount_factor": _BETA},
        "ir_f": {"delta_f": 0.5},
        "ir_m": {"delta_m": 0.2},
    },
    "married_terminal": {},
    "single_f": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "single_f_terminal": {},
    "single_m": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "single_m_terminal": {},
}


@categorical(ordered=True)
class Work:
    """The binary action every regime of the dissolution miniature takes."""

    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the dissolution miniature."""

    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def test_collective_utility_declares_the_regimes_stakeholders():
    """The `utilities` keys are the stakeholders, in the order they are written."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert regime.stakeholders == ("f", "m")


def test_collective_utility_becomes_one_utility_function_per_stakeholder():
    """Each stakeholder's flow utility lands under the name the engine reads."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert regime.functions["utility_f"] is _u_married_ir_f
    assert regime.functions["utility_m"] is _u_married_ir_m


def test_value_dependent_constraint_keeps_its_references_local():
    """A constraint's own references reach the regime under the names it reads."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert set(regime.value_constraints) == {"ir_f", "ir_m"}
    assert set(regime.same_period_refs) == {"V_single_f_ref", "V_single_m_ref"}
    assert regime.same_period_refs["V_single_f_ref"].regime == "single_f"


def test_value_dependent_transition_keeps_the_ordinary_transition_entry():
    """The target still carries its selection probability, gate or no gate."""
    regime = _new_vocabulary_regimes()["married"]

    transition = regime.transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"married_ir"}
    assert isinstance(transition["married_ir"], MarkovTransition)


def test_value_dependent_transition_routes_each_stakeholder_to_her_own_fallback():
    """Each source stakeholder's route keeps all four of its destinations."""
    regime = _new_vocabulary_regimes()["married"]
    edge = regime.gated_edges["married_ir"]

    assert edge.legs["f"].target_stakeholder == "f"
    assert edge.legs["f"].solve_fallback.regime == "single_f"
    assert edge.legs["m"].target_stakeholder == "m"
    assert edge.legs["m"].solve_fallback.regime == "single_m"


def test_the_two_vocabularies_solve_to_the_same_values():
    """The same dissolution model, written both ways, has one solution.

    The new declarations are a consolidation of the old ones, so every regime's
    value function has to agree array for array — including `married_ir`,
    whose participation constraints empty at the middle wage node.
    """
    new = _solve(_new_vocabulary_regimes())
    old = _solve(_make_dissolution_regimes())

    assert set(new) == set(old)
    for period, regime_to_V in new.items():
        assert set(regime_to_V) == set(old[period])
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_almost_equal(
                np.asarray(V_arr),
                np.asarray(old[period][regime_name]),
                decimal=DECIMAL_PRECISION,
                err_msg=f"period {period}, regime {regime_name}",
            )


def _solve(regimes):
    """Solve the dissolution miniature built from `regimes`."""
    model = Model(regimes=regimes, ages=_AGES, regime_id_class=RegimeId)
    return model.solve(params=_PARAMS, log_level="off")


def _new_vocabulary_regimes() -> dict[str, Regime]:
    """The dissolution miniature, declared in the value-dependent vocabulary."""
    married = Regime(
        transition={
            "married_ir": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_no_dissolution_gate,
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f", projection={"wage": _identity_wage}
                        ),
                    ),
                    "m": StakeholderRoute(
                        target_stakeholder="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m", projection={"wage": _identity_wage}
                        ),
                    ),
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    married_ir = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_ir_f, "m": _u_married_ir_m}
            )
        },
        constraints={
            "ir_f": ValueDependentConstraint(
                predicate=_ir_f,
                references={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f", projection={"wage": _identity_wage}
                    )
                },
            ),
            "ir_m": ValueDependentConstraint(
                predicate=_ir_m,
                references={
                    "V_single_m_ref": ProjectedRegimeValue(
                        regime="single_m", projection={"wage": _identity_wage}
                    )
                },
            ),
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )
    single_m = single_f.replace(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        functions={"utility": _u_single_m_ir},
    )
    return {
        "married": married,
        "married_ir": married_ir,
        "married_terminal": married_terminal,
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m": single_m,
        "single_m_terminal": single_f_terminal.replace(),
    }


def test_an_edge_declared_twice_must_agree_in_full():
    """Two spellings of one edge disagreeing on routes is refused, not merged.

    A `ValueDependentTransition` lowers onto a `gated_edges` entry, so a regime
    that carries both for one target is naming one edge twice. Agreeing on the
    gate is not enough: the routes, the gate references and the off-grid
    contract are the edge too, and a disagreement in any of them is two
    different edges for one target.
    """
    route_f = StakeholderRoute(
        target_stakeholder="f",
        fallback=ProjectedRegimeValue(
            regime="single_f", projection={"wage": _identity_wage}
        ),
    )
    route_m = StakeholderRoute(
        target_stakeholder="m",
        fallback=ProjectedRegimeValue(
            regime="single_m", projection={"wage": _identity_wage}
        ),
    )

    with pytest.raises(RegimeInitializationError, match="disagree"):
        Regime(
            transition={
                "married_ir": ValueDependentTransition(
                    probability=MarkovTransition(_prob_one),
                    gate=_no_dissolution_gate,
                    routes={"f": route_f, "m": route_m},
                )
            },
            gated_edges={
                "married_ir": GatedEdge(
                    gate=_no_dissolution_gate,
                    legs={"f": route_f},
                )
            },
            active=lambda age: age < 1,
            states={"wage": _WAGE_3},
            state_transitions={"wage": fixed_transition("wage")},
            actions={"work": DiscreteGrid(Work)},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _u_zero_collective, "m": _u_zero_collective}
                )
            },
        )


def test_a_bare_probability_callable_is_wrapped_for_the_lowered_grammar():
    """`probability` accepts what a plain target transition entry accepts.

    Lowering places the probability in a per-target cell, where the grammar
    requires a `MarkovTransition`. A bare callable is wrapped on the way
    through, so the declared type and the resulting grammar agree.
    """
    route_f = StakeholderRoute(
        target_stakeholder="f",
        fallback=ProjectedRegimeValue(
            regime="single_f", projection={"wage": _identity_wage}
        ),
    )
    route_m = StakeholderRoute(
        target_stakeholder="m",
        fallback=ProjectedRegimeValue(
            regime="single_m", projection={"wage": _identity_wage}
        ),
    )

    regime = Regime(
        transition={
            "married_ir": ValueDependentTransition(
                probability=_prob_one,
                gate=_no_dissolution_gate,
                routes={"f": route_f, "m": route_m},
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )

    transition = regime.transition
    assert isinstance(transition, Mapping)
    lowered = transition["married_ir"]
    assert isinstance(lowered, MarkovTransition)
    assert lowered.func is _prob_one


def _derived_snapshot(regime: Regime) -> dict[str, object]:
    """The five engine-facing facts a regime's declarations determine.

    Flattened into plain data so that two regimes built by different code paths
    — or by the same code path before and after a refactoring — compare by
    value, with the declared callables compared by identity.
    """
    return {
        "stakeholders": regime.stakeholders,
        "pareto_objective": regime.pareto_objective,
        "value_constraints": dict(regime.value_constraints),
        "same_period_refs": dict(regime.same_period_refs),
        "gated_edges": {
            target: (
                edge.gate,
                {
                    source: (leg.target_stakeholder, leg.solve_fallback)
                    for source, leg in edge.legs.items()
                },
                dict(edge.gate_refs),
                edge.off_grid,
            )
            for target, edge in regime.gated_edges.items()
        },
    }


def _expected_snapshots() -> dict[str, dict[str, object]]:
    """What each shape of the dissolution miniature must derive, spelled out."""
    fallback_f = ProjectedRegimeValue(
        regime="single_f", projection={"wage": _identity_wage}
    )
    fallback_m = ProjectedRegimeValue(
        regime="single_m", projection={"wage": _identity_wage}
    )
    return {
        "married": {
            "stakeholders": ("f", "m"),
            "pareto_objective": None,
            "value_constraints": {},
            "same_period_refs": {},
            "gated_edges": {
                "married_ir": (
                    _no_dissolution_gate,
                    {
                        "f": ("f", fallback_f),
                        "m": ("m", fallback_m),
                    },
                    {},
                    "pointwise",
                )
            },
        },
        "married_ir": {
            "stakeholders": ("f", "m"),
            "pareto_objective": None,
            "value_constraints": {"ir_f": _ir_f, "ir_m": _ir_m},
            "same_period_refs": {
                "V_single_f_ref": ProjectedRegimeValue(
                    regime="single_f", projection={"wage": _identity_wage}
                ),
                "V_single_m_ref": ProjectedRegimeValue(
                    regime="single_m", projection={"wage": _identity_wage}
                ),
            },
            "gated_edges": {},
        },
        "married_terminal": {
            "stakeholders": ("f", "m"),
            "pareto_objective": None,
            "value_constraints": {},
            "same_period_refs": {},
            "gated_edges": {},
        },
        "single_f": {
            "stakeholders": None,
            "pareto_objective": None,
            "value_constraints": {},
            "same_period_refs": {},
            "gated_edges": {},
        },
    }


@pytest.mark.parametrize(
    "regime_name", ["married", "married_ir", "married_terminal", "single_f"]
)
def test_the_declarations_determine_every_engine_facing_fact(regime_name):
    """One regime's declarations fix all five facts the engine reads off it.

    A gated regime, a value-constrained one, a plain collective one and a
    singleton: between them they cover every shape a declaration can take. The
    snapshot is the contract that survives any change to how the decomposition
    is performed, because it names what the decomposition must produce rather
    than how.
    """
    regime = _new_vocabulary_regimes()[regime_name]

    assert _derived_snapshot(regime) == _expected_snapshots()[regime_name]
