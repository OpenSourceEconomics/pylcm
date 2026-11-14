"""The actual source factory and actual API guard determine routing decisions."""

import pytest
import ast
import json
from pathlib import Path
import tempfile

from _lcm_regime_finder.routing_adapter import (
    analyze_routing_manifest,
    analyze_routing_source,
    _read_api_function,
)
from _lcm_regime_finder.source_adapter import SourceAdapterError


from .source_data import source_fixture_project

PROJECT = source_fixture_project()
MANIFEST = PROJECT / "examples/foerster-routing-source.json"


def test_original_factory_generates_per_leg_retain_witnesses() -> None:
    report = analyze_routing_manifest(MANIFEST)
    assert (report["original_api_admission"]["status"]) == ("ACCEPTED")
    assert (report["counts"]) == (
        {
            "routes": 8,
            "literal_terminal_routes": 6,
            "equal_payload_pairs": 15,
            "retained_pairs": 15,
        }
    )
    assert not (report["predicate_literal"]["literal"])
    assert (len({r["routing_owner"] for r in report["routes"]})) == (8)
    for candidate in report["candidate_checks"]:
        assert (candidate["decision"]) == ("RETAIN")
        assert (candidate["implementation_admission"]["exception"]) == (
            "ModelInitializationError"
        )
        assert ("share the same fallback regime") in (
            candidate["implementation_admission"]["message"]
        )


def test_ordinary_resolution_input_changes_discovered_candidate_family() -> None:
    config = json.loads(MANIFEST.read_text())
    source = MANIFEST.parent / config["source"]
    api = MANIFEST.parent / config["api_source"]
    config["symbols"]["tables.resolution.n_periods"] = 2
    config["factory_parameters"]["period"] = 1
    report = analyze_routing_source(source, api, config)
    assert (report["counts"]) == (
        {
            "routes": 6,
            "literal_terminal_routes": 4,
            "equal_payload_pairs": 6,
            "retained_pairs": 6,
        }
    )
    assert all("_1_" in r["fallback_regime"] for r in report["routes"])


def test_source_name_renaming_leaves_the_admission_decision_unchanged() -> None:
    config = json.loads(MANIFEST.read_text())
    source = MANIFEST.parent / config["source"]
    api = MANIFEST.parent / config["api_source"]
    text = source.read_text().replace(
        'f"RZERO_{period}_{name}"', 'f"OTHER_{period}_{name}"'
    )
    assert (text) != (source.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "not_a_citekey.py"
        path.write_text(text)
        report = analyze_routing_source(path, api, config)
    assert (report["counts"]["retained_pairs"]) == (15)
    assert all(
        n.startswith("OTHER_")
        for c in report["candidate_checks"]
        for n in c["fallback_regimes"]
    )


def test_mutating_the_actual_guard_changes_its_result() -> None:
    config = json.loads(MANIFEST.read_text())
    source = MANIFEST.parent / config["source"]
    api = MANIFEST.parent / config["api_source"]
    _, guard = _read_api_function(api, config["api_guard"])

    class RemoveRaise(ast.NodeTransformer):
        def visit_Raise(self, node):
            return ast.copy_location(ast.Pass(), node)

    guard = RemoveRaise().visit(guard)
    with tempfile.TemporaryDirectory() as tmp:
        api = Path(tmp) / "different_api.py"
        api.write_text(ast.unparse(ast.fix_missing_locations(guard)))
        report = analyze_routing_source(source, api, config)
    assert (report["counts"]["equal_payload_pairs"]) == (15)
    assert (report["counts"]["retained_pairs"]) == (0)
    assert all(c["decision"] == "UNRESOLVED" for c in report["candidate_checks"])


def test_invalid_original_shared_fallback_is_not_certified() -> None:
    config = json.loads(MANIFEST.read_text())
    source = MANIFEST.parent / config["source"]
    api = MANIFEST.parent / config["api_source"]
    text = source.read_text().replace('f"RZERO_{period}_{name}"', 'f"SHARED_{period}"')
    assert (text) != (source.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.py"
        path.write_text(text)
        with pytest.raises(SourceAdapterError):
            analyze_routing_source(path, api, config)


def test_answer_bearing_pairs_are_rejected() -> None:
    config = json.loads(MANIFEST.read_text())
    source = MANIFEST.parent / config["source"]
    api = MANIFEST.parent / config["api_source"]
    config["pairs"] = [["one", "two"]]
    with pytest.raises(SourceAdapterError):
        analyze_routing_source(source, api, config)
