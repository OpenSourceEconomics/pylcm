"""Editable temporal graph views preserve canonical declaration metadata."""

from fractions import Fraction
from importlib import import_module
from importlib.util import find_spec
from unittest.mock import patch

import jax.numpy as jnp
import networkx as nx
import pytest

from _lcm.reachability import EdgeStatus
from lcm import (
    AgeGrid,
    InitialNodes,
    Model,
    Periods,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.typing import FloatND, Phase, ScalarInt


@categorical(ordered=False)
class GraphRegimes:
    alive: ScalarInt
    final: ScalarInt


@categorical(ordered=False)
class BranchRegimes:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt


def _sentinel() -> FloatND:
    raise AssertionError("A graph view must not evaluate an economic callback.")


def _other_sentinel(*, coefficient: float) -> FloatND:
    del coefficient
    raise AssertionError("A graph view must not evaluate phased utility.")


def _view(*, model: Model, phase: Phase | None = None) -> nx.DiGraph:
    assert find_spec("lcm.graphs") is not None, "Missing public lcm.graphs module"
    module = import_module("lcm.graphs")
    assert hasattr(module, "digraph"), "Missing public lcm.graphs.digraph callable"
    return module.digraph(model, phase=phase)


@pytest.fixture
def period_model() -> Model:
    return Model(
        n_periods=2,
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(functions={"utility": _sentinel}),
            "final": Regime(functions={"utility": _sentinel}),
        },
        initial_nodes=InitialNodes(by_period={0: ("alive", "final")}),
        edges={"alive": {"final": Periods(values=(0,))}},
        enable_jit=False,
    )


def test_digraph_preserves_period_nodes_terminal_and_isolated_starts(
    period_model: Model,
) -> None:
    graph = _view(model=period_model)
    assert isinstance(graph, nx.DiGraph)
    assert set(graph) == {(0, "alive"), (0, "final"), (1, "final")}
    assert set(graph.edges) == {((0, "alive"), (1, "final"))}
    assert graph.graph["phase"] is None
    assert graph.graph["coordinate_kind"] == "period"
    assert graph.graph["clock"] == (0, 1)
    assert graph.nodes[0, "alive"]["period"] == 0
    assert graph.nodes[0, "alive"]["declaration"] is period_model.user_regimes["alive"]
    assert (
        graph.edges[(0, "alive"), (1, "final")]["law"]
        is period_model.graph.laws["alive"]
    )
    assert graph.edges[(0, "alive"), (1, "final")]["status"] is EdgeStatus.CONDITIONAL
    assert graph.graph["declared_edges"] is period_model.graph.edges
    assert graph.graph["declarations"].graph is period_model.graph


def test_digraph_retains_exact_fraction_ages() -> None:
    ages = AgeGrid(exact_values=(Fraction(20), Fraction(61, 3)))
    model = Model(
        ages=ages,
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(functions={"utility": _sentinel}),
            "final": Regime(functions={"utility": _sentinel}),
        },
        initial_nodes=InitialNodes(by_age={20: "alive"}),
        edges={"alive": {"final": 20}},
        enable_jit=False,
    )
    graph = _view(model=model)
    assert graph.graph["coordinate_kind"] == "age"
    assert graph.graph["clock"] == ages.exact_values
    assert set(graph.edges) == {((20, "alive"), (Fraction(61, 3), "final"))}
    target = next(node for node in graph if node[1] == "final")
    assert type(target[0]) is Fraction


@pytest.mark.parametrize("phase", ["solve", "simulate"])
def test_digraph_selects_perceived_or_realized_support(phase: Phase) -> None:
    model = Model(
        n_periods=2,
        regime_id_class=BranchRegimes,
        regimes={
            name: Regime(functions={"utility": _sentinel})
            for name in ("source", "perceived", "realized")
        },
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges=Phased(
            solve={"source": {"perceived": Periods(values=(0,))}},
            simulate={"source": {"realized": Periods(values=(0,))}},
        ),
        enable_jit=False,
    )
    with pytest.raises(ValueError, match="phase"):
        _view(model=model)
    graph = _view(model=model, phase=phase)
    target = "perceived" if phase == "solve" else "realized"
    assert set(graph) == (
        {(0, "source"), (1, "perceived"), (1, "realized")}
        if phase == "solve"
        else {(0, "source"), (1, "realized")}
    )
    assert set(graph.edges) == {((0, "source"), (1, target))}
    assert graph.graph["phase"] == phase


def test_digraph_does_not_select_one_phased_utility() -> None:
    model = Model(
        n_periods=2,
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(
                functions={"utility": Phased(solve=_sentinel, simulate=_other_sentinel)}
            ),
            "final": Regime(functions={"utility": _sentinel}),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"final": Periods(values=(0,))}},
        enable_jit=False,
    )
    graph = _view(model=model)
    declarations = graph.graph["declarations"]
    assert declarations.phases[0, "alive", "solve"].functions["utility"] is _sentinel
    assert (
        declarations.phases[0, "alive", "simulate"].functions["utility"]
        is _other_sentinel
    )


def test_digraph_is_editable_without_mutating_canonical_graph(
    period_model: Model,
) -> None:
    before = period_model.graph
    with patch.object(Model, "_acquire_generation", side_effect=AssertionError):
        first = _view(model=period_model)
        second = _view(model=period_model)
    first.remove_node((1, "final"))
    first.graph["phase"] = "edited"
    second.nodes[0, "alive"]["period"] = 99
    assert period_model.graph is before
    assert (1, "final") in _view(model=period_model)
    assert _view(model=period_model).nodes[0, "alive"]["period"] == 0


def test_digraph_retains_declared_fixed_zero_edge_metadata() -> None:
    def one() -> FloatND:
        return jnp.asarray(1.0)

    def probability(*, probability: float) -> FloatND:
        return jnp.asarray(probability)

    model = Model(
        n_periods=2,
        regime_id_class=BranchRegimes,
        regimes={
            name: Regime(functions={"utility": _sentinel})
            for name in ("source", "perceived", "realized")
        },
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={
            "source": Transition(
                targets={
                    "perceived": Periods(values=(0,)),
                    "realized": Periods(values=(0,)),
                },
                law={
                    "perceived": StochasticTransition(func=one),
                    "realized": StochasticTransition(func=probability),
                },
            )
        },
        fixed_params={"probability": 0.0},
        enable_jit=False,
    )
    graph = _view(model=model, phase="solve")
    assert set(graph.edges) == {((0, "source"), (1, "perceived"))}
    assert "realized" in graph.graph["declared_edges"].solve["source"]
    assert (0, "source", "realized") in graph.graph["pruned_edges"]
    assert graph.graph["pruned_edges"][0, "source", "realized"]


def test_digraph_retains_distinct_bound_laws_with_the_same_topology() -> None:
    model = Model(
        n_periods=2,
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(functions={"utility": _sentinel}),
            "final": Regime(functions={"utility": _sentinel}),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges=Phased(
            solve={
                "alive": Transition(
                    targets={"final": Periods(values=(0,))},
                    law={"final": StochasticTransition(func=_sentinel)},
                )
            },
            simulate={
                "alive": Transition(
                    targets={"final": Periods(values=(0,))},
                    law={"final": StochasticTransition(func=_other_sentinel)},
                )
            },
        ),
        enable_jit=False,
    )
    graph = _view(model=model)
    assert graph.graph["phase"] is None
    assert graph.edges[(0, "alive"), (1, "final")]["law"] is model.graph.laws["alive"]
    assert isinstance(graph.edges[(0, "alive"), (1, "final")]["law"].transition, Phased)


def test_digraph_reads_cached_age_factories_without_rebuilding() -> None:
    from lcm import AgeSpecializedFunction  # noqa: PLC0415
    from lcm.typing import UserFunction  # noqa: PLC0415

    calls: list[float] = []

    def build(age: float) -> UserFunction:
        calls.append(age)
        return _sentinel

    model = Model(
        ages=AgeGrid(start=20, inclusive_stop=22, step="Y"),
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(
                functions={
                    "utility": AgeSpecializedFunction(
                        build=build, signature=lambda age: age
                    )
                }
            ),
            "final": Regime(functions={"utility": _sentinel}),
        },
        initial_nodes=InitialNodes(by_age={20: "alive"}),
        edges={"alive": {"alive": 20, "final": 21}},
        enable_jit=False,
    )
    assert set(calls) == {20.0, 21.0}
    before = list(calls)
    graph = _view(model=model)
    assert calls == before
    assert (
        graph.graph["declarations"].phases[1, "alive", "solve"].functions["utility"]
        is _sentinel
    )
