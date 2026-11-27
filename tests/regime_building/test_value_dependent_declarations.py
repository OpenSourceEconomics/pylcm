"""Collective utilities, gates and the household's own dissolution composition action.

Current dissolution belongs to the household's own Composition action. The public
declaration paths determine roles, phases, parameter namespaces and the canonical policy
problems the model evaluates.
"""

import dataclasses
from collections.abc import Mapping
from typing import TypedDict, cast

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
    MemberRoute,
    Model,
    ParetoObjective,
    Phased,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.composition_actions import CompositionAction, Destination, PCOnActions
from lcm.exceptions import RegimeInitializationError
from lcm.regime import RegimeReplacement
from lcm.transition import PhaseTransitionLaw, StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, bind_laws
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _WAGE_3,
    _dissolution_cost_f,
    _dissolution_cost_m,
    _identity_wage,
    _make_dissolution_regimes,
    _married_dissolution_transition,
    _prob_one,
    _u_effort_collective,
    _u_married_ir_f,
    _u_married_ir_m,
    _u_single_f_ir,
    _u_single_m_ir,
    _u_zero,
)


class _Snapshot(TypedDict):
    members: tuple[str, ...]
    pareto_objective: ParetoObjective | None
    composition_action: tuple[str, dict[str, Destination]] | None


_AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")

_PARAMS = {
    "married": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "married_ir": {
        "koopmans_aggregator": {"discount_factor": _BETA},
        "composition_action": {
            "destinations": {
                "f": {"utility_adjustment": {"delta_f": 0.5}},
                "m": {"utility_adjustment": {"delta_m": 0.2}},
            }
        },
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


def test_collective_utility_declares_the_regimes_members() -> None:
    """The `utilities` keys are the members, in the order they are written."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert regime.members == ("f", "m")


def test_collective_utility_becomes_one_utility_function_per_member() -> None:
    """Each member's flow utility lands under the name the engine reads."""
    regime = _new_vocabulary_regimes()["married_ir"]

    assert regime.decomposed_functions["utility_f"] is _u_married_ir_f
    assert regime.decomposed_functions["utility_m"] is _u_married_ir_m


def test_value_dependent_transition_keeps_the_ordinary_transition_entry() -> None:
    """The target still carries its selection probability, gate or no gate."""
    transition = _bound_laws()["married"].decomposed_transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"married_ir"}
    assert isinstance(transition["married_ir"], StochasticTransition)


def test_separation_routes_each_member_to_her_own_single_regime() -> None:
    """Each member's composition action target is her own single regime."""
    married_composition_action = _new_vocabulary_regimes()[
        "married_ir"
    ].composition_action
    assert married_composition_action is not None

    assert set(married_composition_action.destinations) == {"f", "m"}
    assert married_composition_action.destinations["f"].regime == "single_f"
    assert married_composition_action.destinations["m"].regime == "single_m"


def test_the_two_vocabularies_solve_to_the_same_values() -> None:
    """The same dissolution model, written both ways, has one solution.

    The new declarations are a consolidation of the old ones, so every regime's
    value function has to agree array for array — including `married_ir`,
    whose kept stay actions empty at the middle wage node.
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


def test_an_edge_inside_a_phased_transition_solves_to_the_unphased_values() -> None:
    """Declaring one edge per phase, identically, changes no number.

    `Phased` is a wedge between what a household is solved against and what
    simulation realizes. Writing the same edge on both sides declares no
    wedge, so the model has to solve to exactly the values the unphased
    declaration solves to.
    """
    married = _married_transition()
    (law,) = cast("tuple[PhaseTransitionLaw]", cast("ByAge", married.law).laws)
    phased = _solve(
        regimes=_new_vocabulary_regimes(),
        married=Transition(
            law=ByAge(
                cases={AgeRange(exclusive_stop=1): Phased(solve=law, simulate=law)}
            ),
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


def _solve(
    *, regimes: Mapping[RegimeName, Regime], married: Transition
) -> Mapping[int, Mapping[RegimeName, FloatND]]:
    """Solve the dissolution miniature built from `regimes` and `married`'s law."""
    model = Model(
        regimes=regimes,
        ages=_AGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "married"},
        edges={
            "married": Transition(targets={"married_ir": 0}, law=married.law),
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
                utilities={"f": _u_effort_collective, "m": _u_effort_collective}
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
        composition_action=CompositionAction(
            destinations={
                "f": Destination(
                    regime="single_f",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_f,
                ),
                "m": Destination(
                    regime="single_m",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_m,
                ),
            },
            rule=PCOnActions(),
        ),
    )
    married_terminal = Regime(
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_effort_collective, "m": _u_effort_collective}
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
    """`married`'s move into `married_ir` after age 0.

    `married_ir` dissolves through its own composition action, each member to her own
    single regime.
    """
    return Transition(
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_ir": StochasticTransition(func=_prob_one)
                }
            }
        )
    )


def _always_open(wage: ContinuousState) -> BoolND:
    """A state-only gate: every landing wage keeps the household."""
    return jnp.ones_like(wage, dtype=bool)


def _gate_into_married_ir() -> Gate:
    """A gate into `married_ir`, each route to its own single regime."""
    return Gate(
        predicate=_always_open,
        routes={
            "f": MemberRoute(
                target_member="f",
                fallback=ProjectedRegimeValue(
                    regime="single_f", projection={"wage": _identity_wage}
                ),
            ),
            "m": MemberRoute(
                target_member="m",
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
                gates={"married_ir": _gate_into_married_ir()},
            )
        }
    )["married"]


def test_a_gate_beside_a_phased_law_is_one_edge() -> None:
    """A law may differ by phase, and the gate beside it is one edge.

    The perceived meeting probability and the realized one are a legitimate
    wedge, so the law may differ by phase. The gate, the routes and the off-grid
    contract are declared once beside it, so the law carries one gated edge and
    each phase keeps its own probability.
    """
    law = _phased_edge_law()

    edge = law.gated_edges["married_ir"]
    assert edge.gate is _always_open
    assert set(edge.legs) == {"f", "m"}

    transition = law.decomposed_transition
    assert isinstance(transition, Phased)
    assert transition.solve["married_ir"].func is _prob_one
    assert transition.simulate["married_ir"].func is _prob_half


def _derived_snapshot(*, regime: Regime) -> _Snapshot:
    """The canonical engine-facing facts a regime's declarations determine.

    Flattened into plain data so that two regimes built by different code paths
    — or by the same code path before and after a refactoring — compare by
    value, with the declared callables compared by identity.
    """
    return {
        "members": regime.members,
        "pareto_objective": regime.pareto_objective,
        "composition_action": None
        if regime.composition_action is None
        else (
            type(regime.composition_action.rule).__name__,
            dict(regime.composition_action.destinations),
        ),
    }


def _expected_snapshots() -> dict[RegimeName, _Snapshot]:
    """A plain collective source, a couple that can separate and a one-member regime."""
    plain: _Snapshot = {
        "members": ("self",),
        "pareto_objective": None,
        "composition_action": None,
    }
    return {
        "married": {**plain, "members": ("f", "m")},
        "married_ir": {
            **plain,
            "members": ("f", "m"),
            "composition_action": (
                "PCOnActions",
                {
                    "f": Destination(
                        regime="single_f",
                        projection={"wage": _identity_wage},
                        utility_adjustment=_dissolution_cost_f,
                    ),
                    "m": Destination(
                        regime="single_m",
                        projection={"wage": _identity_wage},
                        utility_adjustment=_dissolution_cost_m,
                    ),
                },
            ),
        },
        "married_terminal": {**plain, "members": ("f", "m")},
        "single_f": plain,
    }


@pytest.mark.parametrize(
    "regime_name", ["married", "married_ir", "married_terminal", "single_f"]
)
def test_the_declarations_determine_every_engine_facing_fact(
    regime_name: RegimeName,
) -> None:
    """One regime's declarations fix the canonical facts the engine reads off it.

    A collective source of a transition, one with a composition action, a plain
    collective one and a singleton: between them they cover every shape a declaration
    can take. The snapshot is the contract that survives any change to how the
    decomposition is performed, because it names what the decomposition must produce
    rather than how.
    """
    snapshot = _derived_snapshot(regime=_new_vocabulary_regimes()[regime_name])

    assert snapshot == _expected_snapshots()[regime_name]


@pytest.mark.parametrize(
    "slot",
    [
        "members",
        "pareto_objective",
        "gated_edges",
    ],
)
def test_a_derived_slot_cannot_be_declared(slot: str) -> None:
    """The engine-facing slots are read off a regime, never written to it.

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
        "members",
        "pareto_objective",
        "gated_edges",
    ],
)
def test_a_derived_slot_cannot_be_replaced(slot: str) -> None:
    """`replace` reaches the declarations, not what they decompose to."""
    regime = Regime(
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )

    with pytest.raises(RegimeInitializationError, match=slot):
        regime.replace(**{slot: None})  # ty: ignore[invalid-argument-type]


def test_replace_refuses_an_unknown_slot() -> None:
    """A keyword that names no declared slot is refused, naming it."""
    regime = Regime(
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )

    with pytest.raises(RegimeInitializationError, match="no_such_slot"):
        regime.replace(no_such_slot=None)


def test_regime_replacement_names_exactly_the_declared_slots() -> None:
    """`replace` takes every slot a regime is constructed from, and no other."""
    declared = {field.name for field in dataclasses.fields(Regime) if field.init}

    assert RegimeReplacement.__optional_keys__ == declared


def _stay_alive() -> int:
    return 0


def test_decomposed_transition_of_an_age_schedule_is_its_engine_view() -> None:
    """A deterministic dated regime decomposes to one engine routing function."""
    law = bind_regime_law(
        ByAge.until(
            stop_age_exclusive=2,
            law=_SupportedDeterministicTransition(
                func=_stay_alive, targets=("alive", "dead")
            ),
            then="dead",
        )
    )

    assert is_bearable(law.decomposed_transition, UserFunction)
