"""Editable temporal graph views preserve canonical declaration metadata."""

import inspect
from collections.abc import Mapping, Sequence
from fractions import Fraction
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import networkx as nx
import pytest

from _lcm.reachability import EdgeStatus
from _lcm.typing import JSONValue
from lcm import (
    AgeGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.graphs import find_regimes
from lcm.processes import RouwenhorstAR1Process, UniformIIDProcess
from lcm.typing import ContinuousState, FloatND, Phase, ScalarInt
from lcm_examples.stochastic_volatility import get_model as stochastic_volatility_model


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


def test_find_regimes_checks_a_native_model(period_model: Model) -> None:
    """The public checker accepts a Model without evaluating its utility."""
    report = find_regimes(period_model)
    layers = report["layers"]
    assert isinstance(layers, dict)
    assert layers["frontier"] == "RETAINED_CANDIDATES_CHECKED"


def test_find_regimes_writes_html(*, period_model: Model, tmp_path: Path) -> None:
    """Users can read the result without running another command."""
    output = tmp_path / "report"
    find_regimes(period_model, output=output)
    assert (output / "report.html").is_file()


def test_find_regimes_refuses_existing_output(
    *, period_model: Model, tmp_path: Path
) -> None:
    """An existing report directory is protected from replacement."""
    with pytest.raises(FileExistsError):
        find_regimes(period_model, output=tmp_path)


@pytest.mark.parametrize("tampered", [False, True])
def test_regime_checker_binds_transition_authority(*, tampered: bool) -> None:
    """The checker recognizes the native class and rejects altered sources."""
    contract = import_module("_lcm_regime_finder.current_api_contract")
    authority = contract.inspect_authority(Transition)
    if tampered:
        authority["file_sha256"] = "0" * 64
    assert contract.matches_authority(
        name="lcm.transition.Transition", observed=authority
    ) is (not tampered)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("tampered", [False, True])
def test_regime_checker_authority_preserves_file_tamper_checks_across_newlines(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    newline: str,
    tampered: bool,
) -> None:
    """Line endings preserve authority, while changes outside the class do not."""
    contract = import_module("_lcm_regime_finder.current_api_contract")
    baseline = contract.inspect_authority(Transition)
    assert baseline is not None
    original_sourcefile = inspect.getsourcefile
    filename = original_sourcefile(Transition)
    assert filename is not None
    source = Path(filename).read_bytes().replace(b"\r\n", b"\n")
    if tampered:
        source += b"\n# changed source authority\n"
    contents = source.replace(b"\n", newline.encode())
    snapshot = tmp_path / "transition.py"
    snapshot.write_bytes(contents)
    assert snapshot.read_bytes() == contents
    assert (b"\r\n" in contents) is (newline == "\r\n")
    monkeypatch.setattr(
        inspect,
        "getsourcefile",
        lambda cls: str(snapshot) if cls is Transition else original_sourcefile(cls),
    )

    authority = contract.inspect_authority(Transition)

    assert authority is not None
    assert authority["source_sha256"] == baseline["source_sha256"]
    assert contract.matches_authority(
        name="lcm.transition.Transition", observed=authority
    ) is (not tampered)


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


def _draw_value(x: FloatND) -> FloatND:
    return x


def _next_x(next_z: ContinuousState) -> ContinuousState:
    return next_z


def _zero_utility() -> float:
    return 0.0


def _record(value: JSONValue) -> Mapping[str, JSONValue]:
    assert isinstance(value, Mapping)
    return value


def _records(value: JSONValue) -> tuple[Mapping[str, JSONValue], ...]:
    assert isinstance(value, Sequence)
    assert not isinstance(value, str)
    return tuple(_record(item) for item in value)


@pytest.mark.parametrize("persistent", [True, False])
def test_analysis_preserves_source_only_conditional_draw(*, persistent: bool) -> None:
    """A source-only process law determines whether its current state is needed."""
    process = (
        RouwenhorstAR1Process(n_points=2, rho=0.5, sigma=1.0)
        if persistent
        else UniformIIDProcess(n_points=2, start=-1.0, stop=1.0)
    )
    model = Model(
        n_periods=2,
        regimes={
            "alive": Regime(
                states={"z": process},
                functions={"utility": _zero_utility},
                state_transitions={"x": _next_x},
            ),
            "final": Regime(
                states={"x": LinSpacedGrid(start=-2.0, stop=2.0, n_points=3)},
                functions={"utility": _draw_value},
            ),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"final": Periods(values=(0,))}},
        regime_id_class=GraphRegimes,
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )
    report = find_regimes(model)
    analysis = report["analysis"]
    assert isinstance(analysis, dict)
    declarations = analysis["declarations"]
    assert isinstance(declarations, dict)
    cells = declarations["cells"]
    assert isinstance(cells, list)
    source = next(cell for cell in _records(cells) if cell["regime"] == "alive")
    interfaces = analysis["interfaces"]
    assert isinstance(interfaces, dict)
    identifier = source["id"]
    assert isinstance(identifier, str)
    options = _records(_record(interfaces[identifier])["options"])
    assert options
    assert all(("z" in _record(option["states"])) is persistent for option in options)


def test_analysis_preserves_conditioned_current_shock() -> None:
    """Income-risk conditioning remains in the alive value domain."""
    report = find_regimes(
        stochastic_volatility_model(
            n_periods=3,
            wealth_n_points=8,
            consumption_n_points=8,
            income_n_points=3,
        )
    )
    analysis = report["analysis"]
    assert isinstance(analysis, dict)
    declarations = analysis["declarations"]
    assert isinstance(declarations, dict)
    cells = declarations["cells"]
    assert isinstance(cells, list)
    interfaces = analysis["interfaces"]
    assert isinstance(interfaces, dict)
    alive = [cell for cell in _records(cells) if cell["regime"] == "alive"]
    assert alive
    for cell in alive:
        options = _records(_record(interfaces[str(cell["id"])])["options"])
        assert options
        assert all(
            "lagged_uncertainty" in _record(option["states"]) for option in options
        )


def test_analyze_regime_structure_returns_typed_recommendations(
    period_model: Model,
) -> None:
    """Recommendations link directly to immutable candidates and original costs."""
    module = import_module("lcm.graphs")
    assert hasattr(module, "analyze_regime_structure")
    result = module.analyze_regime_structure(period_model)
    assert isinstance(result, module.RegimeAnalysisResult)
    assert result.original.cost is not None
    assert result.recommendations
    assert all(
        candidate is result.original or candidate in result.candidates
        for candidate in result.recommendations
    )
    assert result.checks
    with pytest.raises(AttributeError):
        result.status = "invented"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        result.data["status"] = "invented"  # ty: ignore[invalid-assignment]
    copied = result.to_dict()
    copied["status"] = "invented"
    assert result.status != "invented"


def test_analyze_regime_structure_writes_an_explicit_report(
    *, period_model: Model, tmp_path: Path
) -> None:
    """Report writing accepts a string and protects an existing directory."""
    module = import_module("lcm.graphs")
    assert hasattr(module, "analyze_regime_structure")
    result = module.analyze_regime_structure(period_model)
    output = tmp_path / "typed-report"
    result.write_report(str(output))
    text = (output / "report.html").read_text()
    assert "Original representation" in text
    assert "Recommendation" in text
    assert "parent run wrapper" not in text
    assert 'href="resource.json"' not in text
    assert "dates Not supplied" not in text
    with pytest.raises(FileExistsError):
        result.write_report(output)


def test_model_to_networkx_matches_temporal_view(period_model: Model) -> None:
    """The model method exposes the same independent temporal graph."""
    assert hasattr(period_model, "to_networkx")
    actual = period_model.to_networkx()
    expected = _view(model=period_model)
    assert nx.utils.graphs_equal(actual, expected)


def test_analysis_custom_cost_is_used_during_search(period_model: Model) -> None:
    """A declared linear objective is recorded by the actual candidate search."""
    module = import_module("lcm.graphs")
    assert hasattr(module, "LinearRegimeCostModel")
    model = module.LinearRegimeCostModel(
        work_weight=7.0,
        storage_weight=3.0,
        complexity_weight=2.0,
        reference_scales=(1, 1, 1),
        minimum_improvement=0.25,
    )
    result = module.analyze_regime_structure(period_model, cost_model=model)
    assert result.cost_model is model
    scenario = _records(result.to_dict()["scenarios"])[0]
    objective = _record(scenario["objective"])
    assert objective["weights"] == {"work": "7", "storage": "3", "clarity": "2"}
    assert objective["reference_scales"] == {"work": 1, "storage": 1, "clarity": 1}
    assert objective["minimum_improvement"] == "1/4"


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_analysis_cost_rejects_invalid_weights(weight: float) -> None:
    """The supported positive linear class rejects ambiguous objective weights."""
    module = import_module("lcm.graphs")
    assert hasattr(module, "LinearRegimeCostModel")
    with pytest.raises(ValueError, match="work_weight"):
        module.LinearRegimeCostModel(work_weight=weight)
