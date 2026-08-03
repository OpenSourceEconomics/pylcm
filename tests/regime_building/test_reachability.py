import ast
from pathlib import Path

import pytest

from _lcm.reachability import (
    EdgeStatus,
    active_periods_from_predicates,
    build_model_reachability,
    build_phase_reachability,
)


def test_activity_is_source_t_and_target_t_plus_one() -> None:
    """An edge exists only when its source and next-period target are active."""
    graph = build_phase_reachability(
        n_periods=3,
        active_periods_by_regime={"source": {0}, "target": {1}},
        candidate_targets_by_source={"source": {"target"}},
    )

    assert (
        graph.edge_status(period=0, source="source", target="target")
        == EdgeStatus.CONDITIONAL
    )
    assert not graph.has_edge(period=1, source="source", target="target")


def test_forcedout_cannot_transition_back_to_canwork() -> None:
    """A later-life regime cannot transition into an earlier-life regime."""
    ages = tuple(range(60, 69))
    cutoff = 65
    active = active_periods_from_predicates(
        ages=ages,
        active_by_regime={
            "canwork": lambda age: age < cutoff,
            "forcedout": lambda age: age >= cutoff,
        },
    )
    graph = build_phase_reachability(
        n_periods=len(ages),
        active_periods_by_regime=active,
        candidate_targets_by_source={
            "forcedout": {"canwork", "forcedout"},
            "canwork": {"canwork", "forcedout"},
        },
    )

    assert graph.periods_for_edge(source="forcedout", target="canwork") == ()
    assert graph.periods_for_edge(source="canwork", target="forcedout") == (4,)


def test_conditional_edge_is_retained_without_runtime_resolution() -> None:
    """A declared conditional edge remains part of the static graph."""
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"a": {0}, "b": {1}},
        candidate_targets_by_source={"a": {"b"}},
    )

    assert graph.targets(period=0, source="a") == ("b",)
    assert graph.edge_status(period=0, source="a", target="b") == EdgeStatus.CONDITIONAL
    assert not hasattr(graph, "resolve")


def test_coarse_transition_retains_all_activity_compatible_regimes() -> None:
    """A coarse transition treats every regime as a conditional candidate."""
    regime_names = ("source", "low", "high")
    coarse_transition = object()
    graph = build_model_reachability(
        ages=(20, 21),
        active_by_regime={
            regime_name: lambda _age: True for regime_name in regime_names
        },
        transitions_by_phase={
            "solution": {
                "source": coarse_transition,
                "low": None,
                "high": None,
            },
            "simulation": {
                "source": coarse_transition,
                "low": None,
                "high": None,
            },
        },
        terminal_regimes={"low", "high"},
    )

    assert graph.solution.targets(period=0, source="source") == (
        "high",
        "low",
        "source",
    )
    assert all(
        graph.solution.edge_status(period=0, source="source", target=target)
        == EdgeStatus.CONDITIONAL
        for target in regime_names
    )


def test_terminal_source_has_no_edge() -> None:
    """A terminal source has no outgoing edge even when support is declared."""
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"dead": {0}, "alive": {1}},
        candidate_targets_by_source={"dead": {"alive"}},
        terminal_regimes={"dead"},
    )

    assert not graph.has_edge(period=0, source="dead", target="alive")


def test_forward_closure_is_derived_from_the_static_graph() -> None:
    """Initial support propagates only through retained static edges."""
    graph = build_phase_reachability(
        n_periods=3,
        active_periods_by_regime={"a": {0}, "b": {1}, "c": {2}, "x": {1}},
        candidate_targets_by_source={"a": {"b"}, "b": {"c"}, "x": {"c"}},
    )

    assert graph.reachable_from({"a"}) == (
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    )


def test_unknown_target_is_rejected() -> None:
    """Every declared target belongs to the graph's regime universe."""
    with pytest.raises(ValueError, match="unknown regimes"):
        build_phase_reachability(
            n_periods=2,
            active_periods_by_regime={"a": {0}},
            candidate_targets_by_source={"a": {"missing"}},
        )


def test_solution_and_simulation_graphs_use_the_same_builder() -> None:
    """Phase-specific declarations produce phase-specific temporal graphs."""
    graph = build_model_reachability(
        ages=(20, 21),
        active_by_regime={
            "source": lambda _age: True,
            "solve_target": lambda _age: True,
            "simulate_target": lambda _age: True,
        },
        transitions_by_phase={
            "solution": {
                "source": {"solve_target": object()},
                "solve_target": None,
                "simulate_target": None,
            },
            "simulation": {
                "source": {"simulate_target": object()},
                "solve_target": None,
                "simulate_target": None,
            },
        },
        terminal_regimes={"solve_target", "simulate_target"},
    )

    assert graph.solution.targets(period=0, source="source") == ("solve_target",)
    assert graph.simulation.targets(period=0, source="source") == ("simulate_target",)


def test_engine_has_no_period_target_inference_helper() -> None:
    """Engine modules consume the graph instead of inferring period targets."""
    package_root = Path(__file__).parents[2] / "src" / "_lcm"
    definitions = [
        (path, node.lineno)
        for path in package_root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_period_targets", "_active_regimes_at_period"}
    ]

    assert definitions == []


def test_solver_runtime_does_not_import_regime_declaration_topology() -> None:
    """Solver runtime modules depend on the graph, not user declarations."""
    package_root = Path(__file__).parents[2] / "src" / "_lcm"
    runtime_paths = [
        *sorted((package_root / "solution").rglob("*.py")),
        package_root / "simulation" / "compile.py",
        package_root / "simulation" / "simulate.py",
        package_root / "simulation" / "transitions.py",
    ]
    forbidden_modules = {
        "lcm.regime",
        "_lcm.regime_building.canonicalize",
        "_lcm.regime_building.phases",
    }
    imports = [
        (path, node.lineno, node.module)
        for path in runtime_paths
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules
    ]

    assert imports == []
