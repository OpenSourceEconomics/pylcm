"""Collective and value-dependent choice declared inside the slots that carry them.

A regime says who its stakeholders are in `functions["utility"]` and what a
value-reading feasibility constraint is in `constraints`; where a
value-dependent transition routes is a `Gate` beside the source's law in
`Model(edges=...)`.
Nothing about the model changes: the declarations are lowered onto the same
stakeholders, value constraints, references and gated edges the engine has
always run, so a model written this way solves to the numbers the same model
written the long way solves to.
"""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.door import is_bearable

from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from _lcm.regime_law import RegimeLaw, bind_regime_law
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import FloatND, ScalarInt, UserFunction
from tests.conftest import DECIMAL_PRECISION, bind_laws
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _WAGE_3,
    _identity_wage,
    _ir_f,
    _ir_m,
    _make_dissolution_regimes,
    _married_dissolution_transition,
    _no_dissolution_gate,
    _prob_one,
    _u_married_ir_f,
    _u_married_ir_m,
    _u_single_f_ir,
    _u_single_m_ir,
    _u_zero,
    _u_zero_collective,
)

_AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")

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

    assert regime.decomposed_functions["utility_f"] is _u_married_ir_f
    assert regime.decomposed_functions["utility_m"] is _u_married_ir_m


def test_value_dependent_constraint_keeps_its_references_local():
    """A constraint's own references reach the regime under the names it reads."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert set(regime.value_constraints) == {"ir_f", "ir_m"}
    assert set(regime.same_period_refs) == {"V_single_f_ref", "V_single_m_ref"}
    assert regime.same_period_refs["V_single_f_ref"].regime == "single_f"


def test_value_dependent_transition_keeps_the_ordinary_transition_entry():
    """The target still carries its selection probability, gate or no gate."""
    transition = _bound_laws()["married"].decomposed_transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"married_ir"}
    assert isinstance(transition["married_ir"], StochasticTransition)


def test_value_dependent_transition_routes_each_stakeholder_to_her_own_fallback():
    """Each source stakeholder's route keeps all four of its destinations."""
    edge = _bound_laws()["married"].gated_edges["married_ir"]

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
    new = _solve(regimes=_new_vocabulary_regimes(), married=_married_transition())
    old = _solve(
        regimes=_make_dissolution_regimes(), married=_married_dissolution_transition()
    )

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


def test_an_edge_inside_a_phased_transition_solves_to_the_unphased_values():
    """Declaring one edge per phase, identically, changes no number.

    `Phased` is a wedge between what a household is solved against and what
    simulation realizes. Writing the same edge on both sides declares no
    wedge, so the model has to solve to exactly the values the unphased
    declaration solves to.
    """
    married = _married_transition()
    (law,) = cast("ByAge", married.law).laws
    phased = _solve(
        regimes=_new_vocabulary_regimes(),
        married=Transition(
            law=ByAge(
                cases={AgeRange(exclusive_stop=1): Phased(solve=law, simulate=law)}
            ),
            gates=married.gates,
        ),
    )
    unphased = _solve(regimes=_new_vocabulary_regimes(), married=married)

    for period, regime_to_V in phased.items():
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_almost_equal(
                np.asarray(V_arr),
                np.asarray(unphased[period][regime_name]),
                decimal=DECIMAL_PRECISION,
                err_msg=f"period {period}, regime {regime_name}",
            )


def _solve(*, regimes, married: Transition):
    """Solve the dissolution miniature built from `regimes` and `married`'s law."""
    model = Model(
        regimes=regimes,
        ages=_AGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "married"},
        edges={
            "married": Transition(
                targets={"married_ir": 0, "single_f": 0, "single_m": 0},
                law=married.law,
                gates=married.gates,
            ),
            "married_ir": {"married_terminal": 1},
            "single_f": {"single_f_terminal": 1},
            "single_m": {"single_m_terminal": (0, 1, 2)},
        },
    )
    return model.solve(params=_PARAMS, log_level="off").values


def _new_vocabulary_regimes() -> dict[str, Regime]:
    """The dissolution miniature, declared in the value-dependent vocabulary.

    The regimes carry no law: `_married_transition` is the only one a model
    needs, and `_bound_laws` binds every regime's law for inspection.
    """
    married = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    married_ir = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
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
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    single_f = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )
    single_m = single_f.replace(functions={"utility": _u_single_m_ir})
    return {
        "married": married,
        "married_ir": married_ir,
        "married_terminal": married_terminal,
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m": single_m,
        "single_m_terminal": single_f_terminal.replace(),
    }


def _married_transition() -> Transition:
    """`married`'s age-0 transition: a gated edge into `married_ir`.

    Each stakeholder's route falls back to her own single regime, so a model
    declares `married` with `married_ir`, `single_f` and `single_m` as targets.
    """
    return Transition(
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_ir": StochasticTransition(func=_prob_one)
                }
            }
        ),
        gates={"married_ir": _dissolution_gate()},
    )


def _dissolution_gate() -> Gate:
    """`married`'s gate into `married_ir`, each route to her own single regime."""
    return Gate(
        predicate=_no_dissolution_gate,
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


def _bound_laws() -> dict[str, RegimeLaw]:
    """The dissolution miniature's laws, bound as a model binds them."""
    laws = {
        "married": _married_transition(),
        "married_ir": ByAge(
            cases={
                AgeRange(start=1, exclusive_stop=2): {
                    "married_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
        "single_f": ByAge(
            cases={
                AgeRange(start=1, exclusive_stop=2): {
                    "single_f_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
        "single_m": {"single_m_terminal": StochasticTransition(func=_prob_one)},
    }
    return dict(bind_laws({name: laws.get(name) for name in _new_vocabulary_regimes()}))


def _prob_half(age: FloatND) -> FloatND:
    """Half the mass onto the target: the realized meeting rate of the wedge."""
    return 0.5 * jnp.ones_like(age, dtype=float)


def _phased_edge_law() -> RegimeLaw:
    """A married regime's law whose meeting probability differs by phase.

    One gate, declared beside the `Phased` law, holds in both phases.
    """
    return bind_laws(
        {
            "married": Transition(
                law=ByAge(
                    cases={
                        AgeRange(exclusive_stop=1): Phased(
                            solve={"married_ir": StochasticTransition(func=_prob_one)},
                            simulate={
                                "married_ir": StochasticTransition(func=_prob_half)
                            },
                        )
                    }
                ),
                gates={"married_ir": _dissolution_gate()},
            )
        }
    )["married"]


def test_a_gate_beside_a_phased_law_is_one_edge():
    """A law may differ by phase, and the gate beside it is one edge.

    The perceived meeting probability and the realized one are a legitimate
    wedge, so the law may differ by phase. The gate, the routes, the
    references and the off-grid contract are declared once beside it, so the
    law carries one `gated_edges` entry and each phase keeps its own
    probability.
    """
    law = _phased_edge_law()

    edge = law.gated_edges["married_ir"]
    assert edge.gate is _no_dissolution_gate
    assert set(edge.legs) == {"f", "m"}

    transition = law.decomposed_transition
    assert isinstance(transition, Phased)
    assert transition.solve["married_ir"].func is _prob_one
    assert transition.simulate["married_ir"].func is _prob_half


def _derived_snapshot(*, regime: Regime, law: RegimeLaw) -> dict[str, object]:
    """The five engine-facing facts a regime's declarations and law determine.

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
            for target, edge in law.gated_edges.items()
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
    snapshot = _derived_snapshot(
        regime=_new_vocabulary_regimes()[regime_name],
        law=_bound_laws()[regime_name],
    )

    assert snapshot == _expected_snapshots()[regime_name]


@pytest.mark.parametrize(
    "slot",
    [
        "stakeholders",
        "pareto_objective",
        "value_constraints",
        "same_period_refs",
        "gated_edges",
    ],
)
def test_a_derived_slot_cannot_be_declared(slot):
    """The five engine-facing slots are read off a regime, never written to it.

    Each is what one of the three declarations decomposes into, so naming one
    at construction would be a second way to say the same thing — and a way
    that could disagree with the first.
    """
    declared = {slot: None}

    with pytest.raises(TypeError, match=slot):
        Regime(
            states={"wage": _WAGE_3},
            functions={"utility": _u_zero},
            **declared,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    "slot",
    [
        "stakeholders",
        "pareto_objective",
        "value_constraints",
        "same_period_refs",
        "gated_edges",
    ],
)
def test_a_derived_slot_cannot_be_replaced(slot):
    """`replace` reaches the declarations, not what they decompose to."""
    regime = Regime(
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )

    with pytest.raises(RegimeInitializationError, match=slot):
        regime.replace(**{slot: None})


def test_decomposed_transition_of_an_age_schedule_is_its_engine_view():
    """A deterministic dated regime decomposes to one engine routing function."""
    law = bind_regime_law(
        ByAge.until(
            stop_age_exclusive=2,
            law=_SupportedDeterministicTransition(
                func=lambda: 0, targets=("alive", "dead")
            ),
            then="dead",
        )
    )

    assert is_bearable(law.decomposed_transition, UserFunction)
