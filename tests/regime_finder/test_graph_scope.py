"""Constructed-Model graph guarantees do not require runtime calibration or buffers."""

from copy import deepcopy

from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder import verify_production_result
from tests.regime_finder.test_model_adapter import snapshot, function


def _analyze(source):
    result = analyze_production(source, limits={"seconds": 10}, search_mode="exact")
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
    declaration = result["analysis"]["declarations"]
    assert (declaration["analysis_scope"]) == ("pylcm-structural-graph/v1")
    assert (len(result["scenarios"])) == (1)
    assert (result["scenarios"][0]["allocation_contract"]["kind"]) == ("per_block")
    assert not (
        any(
            item["kind"] == "allocation_contract" for item in declaration["obligations"]
        )
    )


def test_runtime_zero_default_never_proves_state_irrelevant() -> None:
    source = snapshot()
    utility = function("def utility(x, theta=0):\n    return theta*x\n", ["x", "theta"])
    utility["defaults"] = {"theta": 0}
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = utility
    result = _analyze(source)
    assert all(
        "x" in axes for axes in result["analysis"]["dependency_closure"].values()
    )


def test_opaque_function_preserves_its_declared_inputs() -> None:
    source = snapshot()
    utility = function("", ["x", "history"])
    utility["source_unavailable"] = True
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = utility
    result = _analyze(source)
    assert all(
        {"x", "history"} <= set(axes)
        for axes in result["analysis"]["dependency_closure"].values()
    )


def test_mutating_a_declared_input_fails_independent_replay() -> None:
    source = snapshot()
    result = _analyze(source)
    corrupt = deepcopy(result)
    utility = next(
        cb
        for cb in corrupt["analysis"]["declarations"]["callbacks"].values()
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
    assert (result["analysis"]["declarations"]["allocation"]["kind"]) == ("per_block")
    corrupt = deepcopy(result)
    corrupt["analysis"]["declarations"]["allocation"]["kind"] = "global_union"
    assert (verify_production_result(source, corrupt)["status"]) == ("FAIL")
