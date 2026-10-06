"""Regime transitions declared only on the model graph: structure and law."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

AGES = AgeGrid(start=60, inclusive_stop=65, step="Y")
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=25)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=50.0, n_points=25)
PARAMS = {"discount_factor": 0.95}


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption < wealth


def _alive() -> Regime:
    return Regime(
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
        constraints={"feasible": _feasible},
    )


DEAD = Regime(states={"wealth": WEALTH_GRID}, functions={"utility": _bequest})
# Retired stays retired through 63 and dies at 64. Its edges at 60-62 let a law
# that may retire early land on a regime with an outgoing edge at every age.
RETIRED_EDGES = {"retired": (60, 61, 62, 63), "dead": 64}
LAW_FREE_EDGES = {
    "working": {"working": (60, 61), "retired": 62},
    "retired": RETIRED_EDGES,
}


def _survive_perceived() -> FloatND:
    return jnp.asarray(0.95)


def _die_perceived() -> FloatND:
    return jnp.asarray(0.05)


def _survive_realized() -> FloatND:
    return jnp.asarray(0.9)


def _die_realized() -> FloatND:
    return jnp.asarray(0.1)


PERCEIVED = {
    "working": StochasticTransition(func=_survive_perceived),
    "dead": StochasticTransition(func=_die_perceived),
}
REALIZED = {
    "working": StochasticTransition(func=_survive_realized),
    "dead": StochasticTransition(func=_die_realized),
}
MORTAL_TARGETS = {"working": (60, 61), "dead": (60, 61), "retired": 62}


def _retire_at_62(age: float) -> ScalarInt:
    return jnp.where(age < 62, RegimeId.working, RegimeId.retired)


def _model(*, edges: object) -> Model:
    return Model(
        regimes={"working": _alive(), "retired": _alive(), "dead": DEAD},
        ages=AGES,
        regime_id_class=RegimeId,
        edges=edges,
        initial_nodes=((60, "working"),),
    )


def _values(model: Model) -> dict[tuple[int, str], Any]:
    solved = model.solve(params=PARAMS, log_level="off").values
    return {
        (period, regime): np.asarray(values)
        for period, by_regime in solved.items()
        for regime, values in by_regime.items()
    }


def test_law_free_edges_route_each_source_age_to_its_only_destination() -> None:
    """A source with one outgoing edge per age moves along that edge."""
    model = _model(edges=LAW_FREE_EDGES)
    assert model.reachability.solution.active_regimes_by_period == (
        frozenset({"working"}),
        frozenset({"working"}),
        frozenset({"working"}),
        frozenset({"retired"}),
        frozenset({"retired"}),
        frozenset({"dead"}),
    )


def test_law_free_edges_solve_like_an_explicit_age_selector() -> None:
    """The graph as law gives the values of a selector choosing the same edges."""
    selector_edges = {
        "working": Transition(
            targets={"working": (60, 61), "retired": (60, 61, 62)},
            law=DeterministicTransition(func=_retire_at_62),
        ),
        "retired": RETIRED_EDGES,
    }
    law_free = _values(_model(edges=LAW_FREE_EDGES))
    selected = _values(_model(edges=selector_edges))
    keys = sorted(law_free)
    np.testing.assert_array_equal(
        [law_free[key] for key in keys], [selected[key] for key in keys]
    )


def test_a_regime_without_outgoing_edges_is_terminal() -> None:
    """A regime no edge leaves is valued by its utility alone."""
    values = _values(_model(edges=LAW_FREE_EDGES))
    np.testing.assert_allclose(
        values[(5, "dead")], np.log(np.asarray(WEALTH_GRID.to_jax()))
    )


def test_transition_with_a_probability_mapping_declares_a_lottery() -> None:
    """A per-target probability law splits the source over its edges."""
    model = _model(
        edges={
            "working": Transition(
                targets=MORTAL_TARGETS,
                law=ByAge(cases={(60, 61): PERCEIVED}),
            ),
            "retired": RETIRED_EDGES,
        }
    )
    targets = model.reachability.solution.targets_by_period
    assert (set(targets[0]["working"]), targets[2]["working"]) == (
        {"working", "dead"},
        ("retired",),
    )


def test_transition_by_age_must_cover_every_age_with_several_edges() -> None:
    """A law that leaves an age with two outgoing edges unselected is rejected."""
    with pytest.raises(ModelInitializationError, match="age 61"):
        _model(
            edges={
                "working": Transition(
                    targets=MORTAL_TARGETS, law=ByAge(cases={60: PERCEIVED})
                ),
                "retired": RETIRED_EDGES,
            }
        )


def test_several_edges_without_a_law_are_rejected() -> None:
    """A source age with more than one destination needs a `Transition` law."""
    with pytest.raises(ModelInitializationError, match="Transition"):
        _model(edges={"working": MORTAL_TARGETS, "retired": RETIRED_EDGES})


def test_a_law_on_a_source_with_one_edge_per_age_is_rejected() -> None:
    """The graph is the law of a source with one destination at every age."""
    with pytest.raises(ModelInitializationError, match="graph is its law"):
        _model(
            edges={
                "working": LAW_FREE_EDGES["working"],
                "retired": Transition(targets=RETIRED_EDGES, law="dead"),
            }
        )


def test_phased_edges_solve_with_the_perceived_law() -> None:
    """Each phase's `Transition` carries that phase's law; solve uses the perceived."""
    phased = _model(
        edges=Phased(
            solve={
                "working": Transition(
                    targets=MORTAL_TARGETS, law=ByAge(cases={(60, 61): PERCEIVED})
                ),
                "retired": RETIRED_EDGES,
            },
            simulate={
                "working": Transition(
                    targets=MORTAL_TARGETS, law=ByAge(cases={(60, 61): REALIZED})
                ),
                "retired": RETIRED_EDGES,
            },
        )
    )
    perceived = _model(
        edges={
            "working": Transition(
                targets=MORTAL_TARGETS, law=ByAge(cases={(60, 61): PERCEIVED})
            ),
            "retired": RETIRED_EDGES,
        }
    )
    phased_values = _values(phased)
    perceived_values = _values(perceived)
    keys = sorted(phased_values)
    np.testing.assert_array_equal(
        [phased_values[key] for key in keys], [perceived_values[key] for key in keys]
    )


LOTTERY_EDGES = {
    "working": Transition(
        targets=MORTAL_TARGETS, law=ByAge(cases={(60, 61): PERCEIVED})
    ),
    "retired": RETIRED_EDGES,
}


@pytest.mark.parametrize(
    ("solve", "simulate"),
    [(LAW_FREE_EDGES, LOTTERY_EDGES), (LOTTERY_EDGES, LAW_FREE_EDGES)],
    ids=["lone_solve", "lone_simulate"],
)
def test_phased_lone_edge_pairs_with_a_probability_mapping(
    *, solve: dict[str, object], simulate: dict[str, object]
) -> None:
    """A lone edge in one phase is a probability-one lottery; solve uses its law."""
    phased_values = _values(_model(edges=Phased(solve=solve, simulate=simulate)))
    solve_values = _values(_model(edges=solve))
    keys = sorted(solve_values)
    np.testing.assert_array_equal(
        [phased_values[key] for key in keys], [solve_values[key] for key in keys]
    )


def _mortality_vector() -> FloatND:
    return jnp.asarray([0.95, 0.0, 0.05])


def test_phased_mapping_and_regime_vector_laws_are_rejected() -> None:
    """A per-target mapping and a regime-code vector are mismatched phase forms."""
    vector_edges = {
        "working": Transition(
            targets=MORTAL_TARGETS,
            law=ByAge(cases={(60, 61): StochasticTransition(func=_mortality_vector)}),
        ),
        "retired": RETIRED_EDGES,
    }
    with pytest.raises(
        (ModelInitializationError, RegimeInitializationError), match="matching forms"
    ):
        _model(edges=Phased(solve=LOTTERY_EDGES, simulate=vector_edges))


def test_regime_rejects_regime_transitions() -> None:
    """Regime laws are declared on the graph, never on the regime."""
    declaration: dict[str, Any] = {
        "regime_transitions": None,
        "functions": {"utility": _bequest},
    }
    with pytest.raises(RegimeInitializationError, match=r"Model\(edges=\.\.\.\)"):
        Regime(**declaration)
