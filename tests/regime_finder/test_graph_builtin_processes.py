"""Intrinsic process dependencies and query identity survive graph-only scope."""

from copy import deepcopy

import pytest
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder import verify_production_result
from tests.regime_finder.test_model_adapter import snapshot, function
from tests.regime_finder.test_query_owners import policy_snapshot


def process(name):
    module = "ar1" if "AR1" in name else "iid"
    return {
        "kind": "dataclass",
        "type": f"_lcm.processes.{module}.{name}",
        "fields": {
            "n_points": 3,
            "state_conditioned": None,
            "sigma": 1.0,
            "mu": 0.0,
            "rho": 0.8,
        },
    }


def _check_analysis(source):
    result = analyze_production(source, limits={"seconds": 15}, search_mode="exact")
    assert (result["status"]) == ("COMPLETE"), str(result.get("semantic_check"))[
        :2000
    ] + str(result["analysis"]["obligations"])
    assert (verify_production_result(source, result)["status"]) == ("PASS")
    return result


def test_persistent_process_keeps_its_current_state_dependency() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    bad = deepcopy(result)
    cb = next(
        c
        for c in bad["analysis"]["declarations"]["callbacks"].values()
        if c.get("original", {}).get("type", "").endswith("RouwenhorstAR1Process")
    )
    cb.update(arguments=[], reads=[])
    assert (verify_production_result(source, bad)["status"]) == ("FAIL")


def test_new_iid_is_drawn_in_arrival_regime_without_a_current_value() -> None:
    source = snapshot()
    arrival = deepcopy(source["user_regimes"]["original"])
    arrival["fields"]["states"]["fresh"] = process("NormalIIDProcess")
    arrival["fields"]["functions"]["utility"] = function(
        "def utility(x,fresh):\n    return x+fresh\n", ["x", "fresh"]
    )
    source["user_regimes"]["arrival"] = arrival
    source["reachability"]["fields"]["nodes"] = [[0, "original"]] + [
        [t, "arrival"] for t in range(1, 6)
    ]
    rows = source["reachability"]["fields"]["solution"]["fields"]["targets_by_period"]
    for t in range(5):
        rows[t] = {"original" if t == 0 else "arrival": ["arrival"]}
    _check_analysis(source)


def test_query_identity_is_preserved_without_numerical_factorization_claim() -> None:
    result = _check_analysis(policy_snapshot())
    assert result["analysis"]["declarations"]["query_owners"]


def test_unknown_process_class_is_not_dispatched_as_a_library_primitive() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    fields["states"]["x"]["type"] = "foreign.RouwenhorstAR1Process"
    del fields["state_transitions"]["x"]
    result = analyze_production(source, limits={"seconds": 15}, search_mode="exact")
    assert (result["status"]) != ("COMPLETE")


def test_lowered_joint_normal_coordinate_is_an_iid_latent_axis() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    coordinate = process("NormalIIDProcess")
    coordinate["type"] = "_lcm.processes.joint_normal_lowering._JointNormalCoordinate"
    coordinate["fields"]["group"] = "income"
    coordinate["fields"]["joint"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.joint_normal.JointNormalIIDProcess",
        "fields": {
            "states": ["income", "return"],
            "cov": [[1, 0.5], [0.5, 1]],
            "mean": [0, 0],
        },
    }
    fields["states"]["x"] = coordinate
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    callbacks = [
        cb
        for cb in result["analysis"]["declarations"]["callbacks"].values()
        if cb.get("original", {}).get("type") == coordinate["type"]
    ]
    assert callbacks
    assert all(cb["arguments"] == [] for cb in callbacks)


def test_empty_target_map_does_not_create_a_next_state_provider() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["state_transitions"]["absent"] = {}
    result = _check_analysis(source)
    assert all(
        "next_absent" not in c["functions"]
        for c in result["analysis"]["declarations"]["cells"]
    )


def test_canonical_identity_preserves_a_categorical_characteristic() -> None:
    from _lcm_regime_finder.semantic_check import _identity_callback

    raw = {
        "arguments": ["history"],
        "bound_self": {
            "type": "_lcm.identity_transition._IdentityTransition",
            "fields": {"_state_name": "history"},
        },
    }
    assert _identity_callback({"original": raw}, "history")
    assert not (_identity_callback({"original": raw}, "other"))
    raw["arguments"] = []
    assert not (_identity_callback({"original": raw}, "history"))


def test_equal_persistent_process_specs_keep_distinct_state_owners() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {
        name: process("RouwenhorstAR1Process") for name in ("x", "history")
    }
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(history,period):\n    return history if period>=3 else 0\n",
        ["history", "period"],
    )
    result = _check_analysis(source)
    assert all(
        "history" in axes and "x" not in axes
        for axes in result["analysis"]["dependency_closure"].values()
    )
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    declaration = deepcopy(result["analysis"]["declarations"])
    relation = next(r for r in declaration["relations"] if r["kind"] == "transition")
    relation["state_updates"]["history"] = relation["state_updates"]["x"]
    changed = MetadataAnalysis(declaration).run()
    checked = check_semantics(source, declaration, changed)
    assert any(
        e["kind"] == "original process dependency owner changed"
        for e in checked["errors"]
    ), checked


def test_phased_process_owner_is_the_axis_not_the_phase_slot() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    process_grid = {
        "kind": "dataclass",
        "type": "lcm.phased.Phased",
        "fields": {
            "solve": process("RouwenhorstAR1Process"),
            "simulate": process("TauchenAR1Process"),
        },
    }
    fields["states"]["x"] = process_grid
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    callbacks = [
        cb
        for cb in result["analysis"]["declarations"]["callbacks"].values()
        if cb.get("original", {}).get("type", "").startswith("_lcm.processes.ar1.")
    ]
    assert callbacks
    assert all(cb["arguments"] == ["x"] for cb in callbacks)


def test_independent_check_rejects_miswired_intrinsic_draw_providers() -> None:
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    analysis = deepcopy(result["analysis"])
    declaration = analysis["declarations"]
    for cell in declaration["cells"]:
        assert "next_x" in cell["functions"]
        cell["functions"]["next_x"] = cell["functions"]["utility"]
        for providers in cell["phase_functions"].values():
            assert "next_x" in providers
            providers["next_x"] = providers["utility"]
    checked = check_semantics(source, declaration, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked
    assert any(
        error["kind"] == "original intrinsic draw provider changed"
        for error in checked["errors"]
    ), checked


def test_independent_check_rejects_erased_conditioned_process_inputs() -> None:
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("NormalIIDProcess")
    fields["states"]["x"]["fields"]["sigma"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.base.StateConditioned",
        "fields": {"on": "history", "by": {"0": 0.5, "1": 1.0}},
    }
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    analysis = deepcopy(result["analysis"])
    declaration = analysis["declarations"]
    callbacks = [
        cb
        for cb in declaration["callbacks"].values()
        if cb.get("original", {}).get("type", "").endswith("NormalIIDProcess")
    ]
    assert callbacks
    for callback in callbacks:
        assert "history" in callback["arguments"]
        callback.update(arguments=[], reads=[])
    checked = check_semantics(source, declaration, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked


def test_missing_conditioning_input_is_rejected_before_discovery() -> None:
    from _lcm_regime_finder.declarations import DeclarationError, normalize_declarations

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("NormalIIDProcess")
    fields["states"]["x"]["fields"]["sigma"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.base.StateConditioned",
        "fields": {"by": {"0": 0.5, "1": 1.0}},
    }
    del fields["state_transitions"]["x"]
    with pytest.raises(DeclarationError, match="no captured conditioning state"):
        normalize_declarations(source)


@pytest.mark.parametrize("process_name", ["RouwenhorstAR1Process", "NormalIIDProcess"])
def test_equal_specs_do_not_authorize_source_provider_owner_swaps(
    process_name: str,
) -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {axis: process(process_name) for axis in ("x", "history")}
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(next_x):\n    return next_x\n", ["next_x"]
    )
    result = _check_analysis(source)
    declaration = deepcopy(result["analysis"]["declarations"])
    for cell in declaration["cells"]:
        for providers in cell["phase_functions"].values():
            assert providers["next_x"] != providers["next_history"]
            providers["next_x"] = providers["next_history"]
        cell["functions"]["next_x"] = cell["functions"]["next_history"]
    analysis = MetadataAnalysis(declaration).run()
    checked = check_semantics(source, declaration, analysis)
    assert checked["status"] == "FAIL", checked
    assert any(
        error["kind"] == "original intrinsic draw provider owner changed"
        for error in checked["errors"]
    ), checked


def test_equal_iid_specs_keep_distinct_source_draw_providers() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {axis: process("NormalIIDProcess") for axis in ("x", "history")}
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(next_x, next_history):\n    return next_x + next_history\n",
        ["next_x", "next_history"],
    )
    result = _check_analysis(source)
    for cell in result["analysis"]["declarations"]["cells"]:
        for providers in cell["phase_functions"].values():
            assert providers["next_x"] != providers["next_history"]
