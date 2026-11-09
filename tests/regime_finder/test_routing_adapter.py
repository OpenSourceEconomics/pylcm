"""The actual source factory and actual API guard determine routing decisions."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.routing_adapter import (analyze_routing_manifest, analyze_routing_source,
                                          _read_api_function)
from _lcm_regime_finder.source_adapter import SourceAdapterError


from .source_data import source_fixture_project

PROJECT = source_fixture_project()
MANIFEST = PROJECT / "examples/foerster-routing-source.json"


class RoutingSourceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(MANIFEST.read_text())
        self.source = MANIFEST.parent / self.config["source"]
        self.api = MANIFEST.parent / self.config["api_source"]

    def test_original_factory_generates_per_leg_retain_witnesses(self):
        report = analyze_routing_manifest(MANIFEST)
        self.assertEqual(report["original_api_admission"]["status"],"ACCEPTED")
        self.assertEqual(report["counts"],{"routes":8,"literal_terminal_routes":6,
                                           "equal_payload_pairs":15,"retained_pairs":15})
        self.assertFalse(report["predicate_literal"]["literal"])
        self.assertEqual(len({r["routing_owner"] for r in report["routes"]}),8)
        for candidate in report["candidate_checks"]:
            self.assertEqual(candidate["decision"],"RETAIN")
            self.assertEqual(candidate["implementation_admission"]["exception"],"ModelInitializationError")
            self.assertIn("share the same fallback regime",candidate["implementation_admission"]["message"])

    def test_ordinary_resolution_input_changes_discovered_candidate_family(self):
        self.config["symbols"]["tables.resolution.n_periods"] = 2
        self.config["factory_parameters"]["period"] = 1
        report = analyze_routing_source(self.source,self.api,self.config)
        self.assertEqual(report["counts"],{"routes":6,"literal_terminal_routes":4,
                                           "equal_payload_pairs":6,"retained_pairs":6})
        self.assertTrue(all("_1_" in r["fallback_regime"] for r in report["routes"]))

    def test_source_name_renaming_leaves_the_admission_decision_unchanged(self):
        text = self.source.read_text().replace('f"RZERO_{period}_{name}"','f"OTHER_{period}_{name}"')
        self.assertNotEqual(text,self.source.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"not_a_citekey.py";path.write_text(text)
            report=analyze_routing_source(path,self.api,self.config)
        self.assertEqual(report["counts"]["retained_pairs"],15)
        self.assertTrue(all(n.startswith("OTHER_") for c in report["candidate_checks"] for n in c["fallback_regimes"]))

    def test_mutating_the_actual_guard_changes_its_result(self):
        _, guard = _read_api_function(self.api,self.config["api_guard"])
        class RemoveRaise(ast.NodeTransformer):
            def visit_Raise(self,node): return ast.copy_location(ast.Pass(),node)
        guard=RemoveRaise().visit(guard)
        with tempfile.TemporaryDirectory() as tmp:
            api=Path(tmp)/"different_api.py";api.write_text(ast.unparse(ast.fix_missing_locations(guard)))
            report=analyze_routing_source(self.source,api,self.config)
        self.assertEqual(report["counts"]["equal_payload_pairs"],15)
        self.assertEqual(report["counts"]["retained_pairs"],0)
        self.assertTrue(all(c["decision"]=="UNRESOLVED" for c in report["candidate_checks"]))

    def test_invalid_original_shared_fallback_is_not_certified(self):
        text=self.source.read_text().replace('f"RZERO_{period}_{name}"','f"SHARED_{period}"')
        self.assertNotEqual(text,self.source.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(text)
            with self.assertRaises(SourceAdapterError):
                analyze_routing_source(path,self.api,self.config)

    def test_answer_bearing_pairs_are_rejected(self):
        self.config["pairs"]=[["one","two"]]
        with self.assertRaises(SourceAdapterError):
            analyze_routing_source(self.source,self.api,self.config)


if __name__ == "__main__": unittest.main()
