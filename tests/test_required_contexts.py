"""Required contexts: missing prerequisites, zero-node regimes, folds and schemas.

A problem some start requires by value must have a law at the age it is read,
whatever kind of read requires it, and adding a start never repairs a missing
prerequisite. A registered regime no start reaches keeps its code, its place in
the regime vector and its absence from every period. Gate folds belong to the
source case that declares the gate. Parameters are collected over the required
cases, including value-only problems, and one parameter name has one schema.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.regime_building.test_same_period_ref_period_axes import (
    COUPLE_GRID,
    Work,
    _couple_utility_f,
    _couple_utility_m,
    _participation_f,
    _project_wealth,
    _single_utility,
    _zero_collective_utility,
    _zero_utility,
)

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _nonterminal(*, utility: Any = _utility) -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": utility},
    )


def _terminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )


@categorical(ordered=False)
class _GatedId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    priced: ScalarInt
    fallback: ScalarInt


_GATED_AGES = AgeGrid(start=40, inclusive_stop=50, step="5Y")


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _gate(*, V_target: FloatND, V_reference: FloatND) -> BoolND:
    return V_target > V_reference


def _identity(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _projected(regime: str) -> ProjectedRegimeValue:
    return ProjectedRegimeValue(regime=regime, projection={"wealth": _identity})


def _gated_law(*, fallback: Any = None) -> dict:
    return {
        "target": ValueDependentTransition(
            probability=StochasticTransition(func=_prob_one),
            gate=_gate,
            routes={
                "only": StakeholderRoute(fallback=fallback or _projected("fallback"))
            },
            gate_references={"V_reference": _projected("reference")},
        )
    }


def _with_source_law(*, phase: Any, law: Any) -> dict:
    return {**phase, "source": Transition(targets=phase["source"], law=law)}


def _gated_model(
    *,
    initial_nodes: Any = None,
    source: Any = None,
    reference: Regime | None = None,
    priced: Regime | None = None,
    fallback: Any = None,
    edges: Any = None,
) -> Model:
    law = source or ByAge(cases={40: _gated_law(fallback=fallback)})
    structure = edges or {"source": {"target": 40, "fallback": 40}}
    return Model(
        regimes={
            "source": _nonterminal(),
            "target": _terminal(),
            "reference": reference or _terminal(),
            "priced": priced or _terminal(),
            "fallback": _terminal(),
        },
        ages=_GATED_AGES,
        regime_id_class=_GatedId,
        initial_nodes=initial_nodes or {40: "source"},
        edges=(
            Phased(
                solve=_with_source_law(phase=structure.solve, law=law),
                simulate=_with_source_law(phase=structure.simulate, law=law),
            )
            if isinstance(structure, Phased)
            else _with_source_law(phase=structure, law=law)
        ),
    )


def _gate_reference_without_law(initial_nodes: Any) -> Model:
    return _gated_model(
        initial_nodes=initial_nodes,
        reference=_nonterminal(),
        edges={
            "source": {"target": 40, "fallback": 40},
            "reference": {"fallback": 40},
        },
    )


def _solve_fallback_without_law(initial_nodes: Any) -> Model:
    return _gated_model(
        initial_nodes=initial_nodes,
        priced=_nonterminal(),
        fallback=Phased(solve=_projected("priced"), simulate=_projected("fallback")),
        edges=Phased(
            solve={
                "source": {"target": 40, "priced": 40},
                "priced": {"fallback": 40},
            },
            simulate={
                "source": {"target": 40, "fallback": 40},
                "priced": {"fallback": 40},
            },
        ),
    )


@categorical(ordered=False)
class _CoupleId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    couple: ScalarInt
    couple_terminal: ScalarInt


def _same_period_reference_without_law(initial_nodes: Any) -> Model:
    """The couple reads the single value at age 0, where single has no law."""
    single_grid = LinSpacedGrid(start=0.0, stop=100.0, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": single_grid},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _single_utility},
            ),
            "single_f_terminal": Regime(
                states={"wealth": single_grid},
                functions={"utility": _zero_utility},
            ),
            "couple": Regime(
                states={"wealth": COUPLE_GRID},
                state_transitions={"wealth": fixed_transition("wealth")},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _couple_utility_f, "m": _couple_utility_m}
                    )
                },
                constraints={
                    "participation_f": ValueDependentConstraint(
                        predicate=_participation_f,
                        references={
                            "V_single_f": ProjectedRegimeValue(
                                regime="single_f",
                                projection={"wealth": _project_wealth},
                            )
                        },
                    )
                },
            ),
            "couple_terminal": Regime(
                states={"wealth": COUPLE_GRID},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _zero_collective_utility,
                            "m": _zero_collective_utility,
                        }
                    )
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_CoupleId,
        initial_nodes=initial_nodes,
        edges={
            "single_f": {"single_f_terminal": 1},
            "couple": {"couple_terminal": 0},
        },
    )


# Each match names the kind of read, the requesting pair and the missing pair,
# in any order.
@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            _gate_reference_without_law,
            (
                r"(?is)(?=.*gate.reference)(?=.*\(40, 'source'\))"
                r"(?=.*'reference' at age 45)"
            ),
        ),
        (
            _solve_fallback_without_law,
            r"(?is)(?=.*fallback)(?=.*\(40, 'source'\))(?=.*'priced' at age 45)",
        ),
    ],
    ids=["gate-reference", "solve-fallback"],
)
def test_a_required_value_read_without_a_law_names_its_kind(
    *, build: Any, match: str
) -> None:
    """A gate reference or valuation fallback lacking a law is named as such."""
    with pytest.raises(ModelInitializationError, match=match):
        build({40: "source"})


def test_a_same_period_reference_without_a_law_names_its_kind() -> None:
    """A same-period outside option lacking a law names the reading pair."""
    with pytest.raises(
        ModelInitializationError,
        match=r"same-period reference of \(0, 'couple'\) requires 'single_f' at age 0",
    ):
        _same_period_reference_without_law({0: "couple"})


@pytest.mark.parametrize(
    ("build", "initial_nodes"),
    [
        (_gate_reference_without_law, {40: "source", 45: "reference"}),
        (_solve_fallback_without_law, {40: "source", 45: "priced"}),
    ],
    ids=["gate-reference", "solve-fallback"],
)
def test_an_added_root_does_not_repair_a_missing_prerequisite(
    *, build: Any, initial_nodes: Any
) -> None:
    """Declaring the missing pair as a start still fails on its missing law."""
    missing = next(name for age, name in initial_nodes.items() if age == 45)
    with pytest.raises(ModelInitializationError, match=rf"'{missing}' at age 45"):
        build(initial_nodes)


def test_an_unknown_target_in_an_unused_case_is_a_global_error() -> None:
    """An unknown regime name fails even in a case no start requires."""
    with pytest.raises(ModelInitializationError, match="'nowhere'"):
        _gated_model(
            source=ByAge(
                cases={
                    40: _gated_law(),
                    45: {"nowhere": StochasticTransition(func=_prob_one)},
                }
            ),
            edges={"source": {"target": 40, "fallback": 40, "nowhere": 45}},
        )


@categorical(ordered=False)
class _LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


_LIFE_AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


_LIFE_EDGES = {
    "working": Transition(
        targets={
            "working": (25, 35, 45),
            "dead": (25, 35, 45),
            "retirement": 55,
        },
        law=ByAge.until(
            stop_age_exclusive=65,
            law={
                "working": StochasticTransition(func=_stay),
                "dead": StochasticTransition(func=_die),
            },
            then="retirement",
        ),
    ),
    "retirement": {"dead": 65},
}


def _life_model(initial_nodes: Any) -> Model:
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges=_LIFE_EDGES,
    )


_ROOTS = pytest.mark.parametrize(
    "initial_nodes",
    [{25: "working"}, {65: "retirement"}],
    ids=["first-age-root", "late-root-with-zero-node-regime"],
)


@_ROOTS
def test_regime_codes_do_not_depend_on_the_starts(initial_nodes: Any) -> None:
    """Every registered regime keeps its code, reached or not."""
    assert _life_model(initial_nodes).regime_names_to_ids == {
        "working": 0,
        "retirement": 1,
        "dead": 2,
    }


@_ROOTS
def test_retirement_solves_to_the_dead_value_under_any_start(
    initial_nodes: Any,
) -> None:
    """Retirement moving along its only edge solves to the dead value."""
    values = _life_model(initial_nodes).solve(params=_PARAMS, log_level="off")
    np.testing.assert_allclose(
        np.asarray(values.values[4]["retirement"]), np.asarray([0.0, 1.9])
    )


def test_late_root_with_an_empty_first_period_simulates() -> None:
    """With no problem at the first age, subjects start at 65 and die at 75."""
    model = _life_model({65: "retirement"})
    panel = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([0.0, 1.0]),
            "age": jnp.asarray([65.0, 65.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["retirement"]),
        },
        solution=model.solve(params=_PARAMS, log_level="off"),
        log_level="off",
        seed=0,
    ).to_dataframe()
    assert sorted(zip(panel["age"], panel["regime_name"], strict=True)) == [
        (65, "retirement"),
        (65, "retirement"),
        (75, "dead"),
        (75, "dead"),
    ]


def test_first_period_of_a_late_root_has_no_values() -> None:
    """Periods keep their global index; those before the start publish no key."""
    values = _life_model({65: "retirement"}).solve(params=_PARAMS, log_level="off")
    assert {period: set(by) for period, by in values.values.items()} == {
        4: {"retirement"},
        5: {"dead"},
    }


def _source_owned_fold_model() -> Model:
    return _gated_model(
        initial_nodes={40: "source", 45: "source"},
        source=ByAge(cases={40: "fallback", 45: _gated_law()}),
        edges={"source": {"fallback": (40, 45), "target": 45}},
    )


def test_a_gate_fold_belongs_to_the_case_that_declares_the_gate() -> None:
    """A source gated only at 45 folds its gate only at the landing age 50."""
    folds = _source_owned_fold_model()._regimes["source"].gated_edges["target"]
    assert set(folds.folds_by_period) == {2}


@categorical(ordered=False)
class _MaritalId:
    single: ScalarInt
    married: ScalarInt
    dead: ScalarInt


def _gated_move(*, target: str, stay: str) -> dict:
    return {
        target: ValueDependentTransition(
            probability=StochasticTransition(func=_prob_one),
            gate=_marital_gate,
            routes={"only": StakeholderRoute(fallback=_projected(stay))},
            gate_references={"V_stay": _projected(stay)},
        )
    }


def _marital_gate(*, V_target: FloatND, V_stay: FloatND) -> BoolND:
    return V_target > V_stay


def _marital_model() -> Model:
    return Model(
        regimes={
            "single": _nonterminal(),
            "married": _nonterminal(),
            "dead": _terminal(),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_MaritalId,
        initial_nodes={0: ("single", "married")},
        edges={
            "single": Transition(
                targets={"married": 0, "single": 0, "dead": 1},
                law=ByAge.until(
                    stop_age_exclusive=2,
                    law=_gated_move(target="married", stay="single"),
                    then="dead",
                ),
            ),
            "married": Transition(
                targets={"single": 0, "married": 0, "dead": 1},
                law=ByAge.until(
                    stop_age_exclusive=2,
                    law=_gated_move(target="single", stay="married"),
                    then="dead",
                ),
            ),
        },
    )


def test_reciprocal_marriage_and_divorce_gates_build_without_a_cycle() -> None:
    """Gates reading each other's next-period value add no same-period ordering."""
    assert _marital_model().reachability.nodes == frozenset(
        {(age, name) for age in (0, 1) for name in ("single", "married")}
        | {(2, "dead")}
    )


def _rate_as_float(rate: float) -> FloatND:
    return jnp.asarray(rate)


def _rest_as_float(rate: float) -> FloatND:
    return 1 - jnp.asarray(rate)


def _rate_as_int(rate: ScalarInt) -> FloatND:
    return jnp.asarray(rate) * 0.5


def _rest_as_int(rate: ScalarInt) -> FloatND:
    return 1 - jnp.asarray(rate) * 0.5


def _two_schema_model(initial_nodes: Any) -> Model:
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _terminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges={
            "working": Transition(
                targets={
                    "working": AgeRange(exclusive_stop=65),
                    "dead": AgeRange(exclusive_stop=75),
                },
                law=ByAge(
                    cases={
                        AgeRange(start=25, exclusive_stop=45): {
                            "working": StochasticTransition(func=_rate_as_float),
                            "dead": StochasticTransition(func=_rest_as_float),
                        },
                        AgeRange(start=45, exclusive_stop=65): {
                            "working": StochasticTransition(func=_rate_as_int),
                            "dead": StochasticTransition(func=_rest_as_int),
                        },
                    },
                    default="dead",
                ),
            )
        },
    )


def test_conflicting_schemas_of_one_required_parameter_fail() -> None:
    """`rate` read as `float` in one required case and `ScalarInt` in another fails."""
    with pytest.raises(ModelInitializationError, match=r"'rate'.*float.*ScalarInt"):
        _two_schema_model({25: "working"})


def test_a_schema_conflict_in_an_unrequired_case_is_not_required() -> None:
    """With only the late case required, `rate` has the late case's schema."""
    template = _two_schema_model({55: "working"}).get_params_template()
    assert template["edges"]["working"]["working"] == {"rate": "ScalarInt"}


@categorical(ordered=False)
class _PhasedId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt


def _bonus_utility(*, wealth: ContinuousState, bonus: float) -> FloatND:
    return wealth + bonus


def _value_only_model() -> Model:
    return Model(
        regimes={
            "source": _nonterminal(),
            "perceived": _nonterminal(utility=_bonus_utility),
            "realized": _nonterminal(),
            "end": _terminal(),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_PhasedId,
        initial_nodes={0: "source"},
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
            simulate={
                "source": {"realized": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
        ),
    )


def test_a_parameter_of_a_value_only_regime_is_kept() -> None:
    """The perceived regime is never visited, yet its `bonus` is required."""
    assert _value_only_model().get_params_template()["perceived"]["utility"] == {
        "bonus": "float"
    }


def _grown(*, wealth: ContinuousState, growth: float) -> ContinuousState:
    return wealth * growth


def _producer_model(initial_nodes: Any) -> Model:
    return Model(
        regimes={
            "working": Regime(
                states={"wealth": _WEALTH},
                state_transitions={
                    "wealth": {
                        "working": _identity,
                        "retirement": _grown,
                        "dead": _identity,
                    }
                },
                functions={"utility": _utility},
            ),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges=_LIFE_EDGES,
    )


def test_a_late_root_keeps_the_producer_of_its_exit_target() -> None:
    """A start at 55 still requires the wealth producer into retirement."""
    template = _producer_model({55: "working"}).get_params_template()
    assert template["working"]["retirement"]["next_wealth"] == {"growth": "float"}
