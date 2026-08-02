from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "src" / "_lcm" / "reachability.py"
SPEC = importlib.util.spec_from_file_location("_reachability_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EdgeStatus = MODULE.EdgeStatus
build_model_reachability = MODULE.build_model_reachability
build_phase_reachability = MODULE.build_phase_reachability


def test_activity_is_source_t_and_target_t_plus_one() -> None:
    graph = build_phase_reachability(
        n_periods=3,
        active_periods_by_regime={"source": {0}, "target": {1}},
        candidate_targets_by_source={"source": {"target"}},
    )
    assert graph.edge_status(period=0, source="source", target="target") == (
        EdgeStatus.CONDITIONAL
    )
    assert not graph.has_edge(period=1, source="source", target="target")


def test_forcedout_cannot_transition_back_to_canwork() -> None:
    graph = build_phase_reachability(
        n_periods=9,
        active_periods_by_regime={
            "canwork": {0, 1, 2, 3, 4},
            "forcedout": {5, 6, 7, 8},
        },
        candidate_targets_by_source={
            "forcedout": {"canwork", "forcedout"},
            "canwork": {"canwork", "forcedout"},
        },
    )
    assert graph.periods_for_edge(source="forcedout", target="canwork") == ()
    assert graph.periods_for_edge(source="canwork", target="forcedout") == (4,)


def test_conditional_is_retained_without_runtime_resolution() -> None:
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"a": {0}, "b": {1}},
        candidate_targets_by_source={"a": {"b"}},
    )
    assert graph.targets(period=0, source="a") == ("b",)
    assert graph.edge_status(period=0, source="a", target="b") == (
        EdgeStatus.CONDITIONAL
    )
    assert not hasattr(graph, "resolve")


def test_coarse_transition_defaults_to_all_activity_compatible_targets() -> None:
    all_names = ("a", "b", "c")
    coarse = object()
    graph = build_model_reachability(
        ages=(20, 21),
        active_by_regime={name: (lambda _age: True) for name in all_names},
        transitions_by_phase={
            "solution": {"a": coarse, "b": None, "c": None},
            "simulation": {"a": coarse, "b": None, "c": None},
        },
        terminal_regimes={"b", "c"},
    )
    assert graph.solution.targets(period=0, source="a") == all_names
    assert all(
        graph.solution.edge_status(period=0, source="a", target=target)
        == EdgeStatus.CONDITIONAL
        for target in all_names
    )


def test_terminal_source_has_no_edge() -> None:
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"dead": {0}, "alive": {1}},
        candidate_targets_by_source={"dead": {"alive"}},
        terminal_regimes={"dead"},
    )
    assert not graph.has_edge(period=0, source="dead", target="alive")


def test_forward_closure_is_only_a_derived_static_view() -> None:
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
    with pytest.raises(ValueError, match="unknown regimes"):
        build_phase_reachability(
            n_periods=2,
            active_periods_by_regime={"a": {0}},
            candidate_targets_by_source={"a": {"missing"}},
        )
