"""Solve demand follows the declared starts: physical visits and value reads.

`model.initial_nodes` is the declared set of starts. From it the engine derives
the `(age, regime)` pairs a subject can physically visit
(`reachability.visited_nodes`) and every pair whose value some required
problem reads (`reachability.nodes`). A value read adds a solved problem, not a
visit, and simulation-only routes out of a value-only problem are not followed.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from _lcm.reachability import candidate_targets_from_transition
from _lcm.regime_building.transition_support import (
    _SupportedStochasticTransition,
)
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.regime_building.test_same_period_ref_period_axes import (
    _make_model as _make_outside_option_model,
)

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _nonterminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _terminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )


@categorical(ordered=False)
class LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


_LIFE_AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")
_BROKEN_EXIT_EDGES = {
    "working": {"working": (25, 35, 45), "retirement": 55},
    "retirement": {"dead": 45},
}


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


def _life_model(initial_nodes: Any) -> Model:
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=LifeId,
        initial_nodes=initial_nodes,
        edges={
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
        },
    )


_WORKING_LIFE = frozenset(
    {(age, "working") for age in (25, 35, 45, 55)}
    | {(65, "retirement")}
    | {(age, "dead") for age in (35, 45, 55, 75)}
)


@pytest.mark.parametrize(
    ("initial_nodes", "expected"),
    [
        ({25: "working"}, _WORKING_LIFE),
        (
            {55: "working"},
            frozenset({(55, "working"), (65, "retirement"), (75, "dead")}),
        ),
        ({65: "retirement"}, frozenset({(65, "retirement"), (75, "dead")})),
        ({45: "dead"}, frozenset({(45, "dead")})),
        (
            {
                25: "working",
                65: "retirement",
                AgeRange(start=25, exclusive_stop=26): "dead",
            },
            _WORKING_LIFE | {(25, "dead")},
        ),
    ],
    ids=["first-age", "late-root", "zero-node-regime", "terminal-only", "islands"],
)
def test_reachability_nodes_are_the_closure_of_the_starts(
    *, initial_nodes: Any, expected: frozenset
) -> None:
    """Solved pairs are exactly those a start can reach; no other age is solved."""
    assert _life_model(initial_nodes).reachability.nodes == expected


@pytest.mark.parametrize(
    "initial_nodes",
    [{25: "working"}, {55: "working"}, {65: "retirement"}, {45: "dead"}],
    ids=["first-age", "late-root", "zero-node-regime", "terminal-only"],
)
def test_solve_returns_values_exactly_at_the_demanded_pairs(
    initial_nodes: Any,
) -> None:
    """Every demanded pair has a value array, and no other pair has one."""
    model = _life_model(initial_nodes)
    # A model whose starts reach no nonterminal regime reads no parameter.
    params = {} if initial_nodes == {45: "dead"} else _PARAMS
    values = model.solve(params=params, log_level="off").values
    assert model.ages is not None
    solved = frozenset(
        (model.ages.exact_values[period], name)
        for period, by_regime in values.items()
        for name in by_regime
    )
    assert solved == model.reachability.nodes


def test_initial_nodes_stay_the_declared_starts() -> None:
    """Demand expansion never adds a start."""
    model = _life_model({55: "working"})
    assert model.initial_nodes == frozenset({(55, "working")})


def test_coverage_of_an_unreached_regime_is_empty() -> None:
    """A regime no start reaches keeps its id but is solved at no age."""
    model = _life_model({65: "retirement"})
    assert all(
        "working" not in active
        for active in model.reachability.solution.active_regimes_by_period
    )


@pytest.mark.parametrize(
    ("initial_nodes", "match"),
    [({75: "retirement"}, "75"), ({45: "retirement"}, "45")],
    ids=["nonterminal-at-last-age", "no-law-at-root-age"],
)
def test_a_start_without_an_available_problem_fails(
    *, initial_nodes: Any, match: str
) -> None:
    """A start needs a local law and, if nonterminal, a next age."""
    with pytest.raises(ModelInitializationError, match=match):
        _life_model(initial_nodes)


def test_a_required_target_without_a_law_names_the_source() -> None:
    """A demanded transition into an age with no law is an error, not a drop."""
    with pytest.raises(
        ModelInitializationError, match=r"\(55, 'working'\).*'retirement' at age 65"
    ):
        Model(
            regimes={
                "working": _nonterminal(),
                "retirement": _nonterminal(),
                "dead": _terminal(),
            },
            ages=_LIFE_AGES,
            regime_id_class=LifeId,
            initial_nodes={25: "working"},
            edges=_BROKEN_EXIT_EDGES,
        )


def test_an_unrequired_broken_target_does_not_fail() -> None:
    """The same broken exit is harmless when no start requires it."""
    model = Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=LifeId,
        initial_nodes={45: "retirement"},
        edges=_BROKEN_EXIT_EDGES,
    )
    assert model.reachability.nodes == frozenset({(45, "retirement"), (55, "dead")})


@categorical(ordered=False)
class PhasedId:
    source: ScalarInt
    other_source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt
    realized_end: ScalarInt


_PHASED_AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")


def _phased_model(initial_nodes: Any) -> Model:
    return Model(
        regimes={
            "source": _nonterminal(),
            "other_source": _nonterminal(),
            "perceived": _nonterminal(),
            "realized": _nonterminal(),
            "end": _terminal(),
            "realized_end": _terminal(),
        },
        ages=_PHASED_AGES,
        regime_id_class=PhasedId,
        initial_nodes=initial_nodes,
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "other_source": {"perceived": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
            simulate={
                "source": {"realized": 0},
                "other_source": {"perceived": 0},
                "perceived": {"realized_end": 1},
                "realized": {"end": 1},
            },
        ),
    )


def test_perceived_target_is_solved_but_not_visited() -> None:
    """A solve-only target is valued; its simulation-only routes are not followed."""
    reachability = _phased_model({0: "source"}).reachability
    assert (reachability.nodes, reachability.visited_nodes) == (
        frozenset({(0, "source"), (1, "perceived"), (1, "realized"), (2, "end")}),
        frozenset({(0, "source"), (1, "realized"), (2, "end")}),
    )


def test_a_valued_pair_that_is_also_visited_follows_its_realized_routes() -> None:
    """Once a valued pair is physically reachable, its realized targets are too."""
    reachability = _phased_model({0: ("source", "other_source")}).reachability
    assert (reachability.nodes, reachability.visited_nodes) == (
        frozenset(
            {
                (0, "source"),
                (0, "other_source"),
                (1, "perceived"),
                (1, "realized"),
                (2, "end"),
                (2, "realized_end"),
            }
        ),
        frozenset(
            {
                (0, "source"),
                (0, "other_source"),
                (1, "perceived"),
                (1, "realized"),
                (2, "end"),
                (2, "realized_end"),
            }
        ),
    )


@categorical(ordered=False)
class GatedId:
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


def _gated_model(*, phased_fallback: bool) -> Model:
    fallback = (
        Phased(
            solve=ProjectedRegimeValue(
                regime="priced", projection={"wealth": _identity}
            ),
            simulate=ProjectedRegimeValue(
                regime="fallback", projection={"wealth": _identity}
            ),
        )
        if phased_fallback
        else ProjectedRegimeValue(regime="fallback", projection={"wealth": _identity})
    )
    law = ByAge(cases={40: {"target": StochasticTransition(func=_prob_one)}})
    gates = {
        "target": Gate(
            predicate=_gate,
            routes={"only": StakeholderRoute(fallback=fallback)},
            references={
                "V_reference": ProjectedRegimeValue(
                    regime="reference",
                    projection={"wealth": _identity},
                )
            },
        )
    }
    return Model(
        regimes={
            "source": _nonterminal(),
            "target": _terminal(),
            "reference": _terminal(),
            "priced": _terminal(),
            "fallback": _terminal(),
        },
        ages=_GATED_AGES,
        regime_id_class=GatedId,
        initial_nodes={40: "source"},
        edges=Phased(
            solve={
                "source": Transition(
                    targets={"target": 40, "priced": 40}, law=law, gates=gates
                )
            },
            simulate={
                "source": Transition(
                    targets={"target": 40, "fallback": 40}, law=law, gates=gates
                )
            },
        )
        if phased_fallback
        else {
            "source": Transition(
                targets={"target": 40, "fallback": 40}, law=law, gates=gates
            )
        },
    )


def test_gate_references_and_fallbacks_are_demanded_at_the_landing_age() -> None:
    """The gate's reference is valued only; the fallback is also a landing."""
    reachability = _gated_model(phased_fallback=False).reachability
    assert (reachability.nodes, reachability.visited_nodes) == (
        frozenset(
            {(40, "source"), (45, "target"), (45, "reference"), (45, "fallback")}
        ),
        frozenset({(40, "source"), (45, "target"), (45, "fallback")}),
    )


def test_a_phased_fallback_prices_one_regime_and_lands_in_another() -> None:
    """The solve leg is valued only; the simulate leg is a physical landing."""
    reachability = _gated_model(phased_fallback=True).reachability
    assert (reachability.nodes, reachability.visited_nodes) == (
        frozenset(
            {
                (40, "source"),
                (45, "target"),
                (45, "reference"),
                (45, "priced"),
                (45, "fallback"),
            }
        ),
        frozenset({(40, "source"), (45, "target"), (45, "fallback")}),
    )


def test_a_local_outside_option_demands_its_island_by_value_only() -> None:
    """A same-period reference and its own continuation are solved, not visited."""
    reachability = _make_outside_option_model(
        later_ceiling=10.0, initial_nodes={0: "couple"}
    ).reachability
    assert (reachability.nodes, reachability.visited_nodes) == (
        frozenset(
            {
                (0, "couple"),
                (1, "couple_terminal"),
                (0, "single_f"),
                (1, "single_f"),
                (1, "single_f_terminal"),
                (2, "single_f_terminal"),
            }
        ),
        frozenset({(0, "couple"), (1, "couple_terminal")}),
    )


def test_simulate_visits_only_visited_pairs_with_zero_node_regimes() -> None:
    """Subjects starting at a start stay on visited pairs; unused regimes are empty."""
    model = _phased_model({0: "source"})
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([0.0, 1.0]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["source"]),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    panel = result.to_dataframe()
    visited = frozenset(zip(panel["age"], panel["regime_name"], strict=True))
    assert visited == frozenset({(0, "source"), (1, "realized"), (2, "end")})


def test_an_unreached_regime_with_an_age_specialized_grid_builds_and_solves() -> None:
    """An age-varying grid of a regime solved at no age needs no age to resolve."""
    model = _make_outside_option_model(
        later_ceiling=10.0, initial_nodes={2: "single_f_terminal"}
    )
    # Only the terminal root is demanded, so no regime reads a parameter.
    values = model.solve(params={}, log_level="off").values
    solved = {
        (period, name) for period, by_regime in values.items() for name in by_regime
    }
    assert solved == {(2, "single_f_terminal")}


def _probabilities() -> FloatND:
    return jnp.array([1.0, 0.0, 0.0])


def test_candidate_targets_of_a_vector_law_are_its_declared_targets() -> None:
    """A vector `StochasticTransition` names its candidates by `targets`, not by the
    whole regime vocabulary."""
    law = _SupportedStochasticTransition(func=_probabilities, targets=("retirement",))
    assert candidate_targets_from_transition(
        transition=law, all_regime_names=("working", "retirement", "dead")
    ) == ("retirement",)
