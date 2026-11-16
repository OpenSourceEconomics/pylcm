"""Constructed-Model graph guarantees do not require runtime calibration or buffers."""

from collections.abc import Mapping
from copy import deepcopy

from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder import verify_production_result
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_declaration_ingestion import (
    json_entries,
    json_member,
    json_rows,
)
from tests.regime_finder.test_model_adapter import snapshot, function


def _analyze(source: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = analyze_production(
        source, limits={"seconds": 10}, search_mode="exact", facts_backend="metadata"
    )
    assert (result["status"]) == ("COMPLETE"), str(result.get("semantic_check"))[:2500]
    assert (
        verify_production_result(source, result, limits={"seconds": 10})["status"]
    ) == ("PASS")
    return result


def test_constructed_graph_uses_regime_domains_without_calibration_or_allocation_contract() -> (
    None
):
    source = snapshot()
    result = _analyze(source)
    declaration = json_member(result, "analysis", "declarations")
    assert (declaration["analysis_scope"]) == ("pylcm-structural-graph/v1")
    scenarios = json_rows(result, "scenarios")
    assert (len(scenarios)) == (1)
    assert (json_member(scenarios[0], "allocation_contract")["kind"]) == ("per_block")
    assert not (
        any(
            item["kind"] == "allocation_contract"
            for item in json_rows(declaration, "obligations")
        )
    )


def test_runtime_zero_default_never_proves_state_irrelevant() -> None:
    source = snapshot()
    utility = function("def utility(x, theta=0):\n    return theta*x\n", ["x", "theta"])
    utility["defaults"] = {"theta": 0}
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = utility
    result = _analyze(source)
    closure = json_member(result, "analysis", "dependency_closure")
    assert all(isinstance(axes, list) and "x" in axes for axes in closure.values())


def test_opaque_function_preserves_its_declared_inputs() -> None:
    source = snapshot()
    utility = function("", ["x", "history"])
    utility["source_unavailable"] = True
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = utility
    result = _analyze(source)
    closure = json_member(result, "analysis", "dependency_closure")
    assert all(
        isinstance(axes, list) and {"x", "history"} <= set(axes)
        for axes in closure.values()
    )


def test_mutating_a_declared_input_fails_independent_replay() -> None:
    source = snapshot()
    result = _analyze(source)
    corrupt = deepcopy(result)
    utility = next(
        cb
        for cb in json_entries(corrupt, "analysis", "declarations", "callbacks")
        if cb.get("arguments") == ["x", "history"]
    )
    utility["arguments"] = []
    utility["reads"] = []
    assert (verify_production_result(source, corrupt)["status"]) == ("FAIL")


def test_global_union_cannot_replace_pylcm_regime_storage() -> None:
    source = snapshot()
    source["allocation_contract"] = {
        "kind": "global_union",
        "status": "declared",
        "identity": "foreign-allocation",
    }
    result = _analyze(source)
    allocation_path = ("analysis", "declarations", "allocation")
    assert (json_member(result, *allocation_path)["kind"]) == ("per_block")
    corrupt = deepcopy(result)
    json_member(corrupt, *allocation_path)["kind"] = "global_union"
    assert (verify_production_result(source, corrupt)["status"]) == ("FAIL")
