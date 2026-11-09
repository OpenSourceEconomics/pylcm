"""Local source safety and finite-family coverage have distinct receipt scopes."""
from copy import deepcopy
import unittest

from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics


def original_declarations():
    """A literal one-cell finite problem; no capture or numerical model call."""
    return {
        "schema": "regime-declarations/v1",
        "source_identity": "literal-one-cell-scope-reporting-fixture",
        "domains": {"binary": {
            "kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1],
        }},
        "callbacks": {"utility": {
            "arguments": ["x"], "reads": ["x"], "bindings": {},
            "source": "def utility(x):\n    return x\n",
            "expression": "x", "status": "proved", "identity": "literal-utility",
        }},
        "cells": [{
            "id": "c0", "regime": "life", "period": 0, "terminal": True,
            "states": {"x": "binary"}, "actions": {},
            "functions": {"utility": "utility"},
            "consumers": [{"kind": "reward", "callback": "utility", "phase": "both"}],
        }],
        "relations": [], "characteristics": {}, "obligations": [],
        "allocation": {"kind": "per_block", "status": "declared", "boundary_unit": 1},
    }


class SemanticScopeReportingTests(unittest.TestCase):
    def setUp(self):
        self.original = original_declarations()
        self.declarations = deepcopy(self.original)
        self.analysis = MetadataAnalysis(self.declarations).run()

    def test_local_replay_does_not_borrow_complete_family_wording(self):
        complete = check_semantics(self.original, self.declarations, self.analysis)
        local = check_semantics(
            self.original, self.declarations, self.analysis, check_coverage=False)
        self.assertEqual(complete["status"], "PASS", complete)
        self.assertTrue(complete["projection_family_complete_verified"])
        self.assertIn("independently checked finite family", complete["guarantee"])
        self.assertEqual(local["errors"], [])
        self.assertEqual(local["status"], "UNRESOLVED", local)
        self.assertFalse(local["projection_family_complete_verified"])
        self.assertTrue(all(item["scope"] == "family" for item in local["unresolved"]))
        self.assertIn("supplied coherent projection policies only", local["guarantee"])
        self.assertIn("full-family coverage is not established", local["guarantee"])
        self.assertNotIn("every coherent", local["guarantee"])

    def test_requested_coverage_is_not_equivalent_to_verified_coverage(self):
        incomplete = deepcopy(self.analysis)
        incomplete["discovery_complete"] = False
        checked = check_semantics(
            self.original, self.declarations, incomplete, check_coverage=True)
        self.assertEqual(checked["errors"], [])
        self.assertEqual(checked["status"], "UNRESOLVED", checked)
        self.assertFalse(checked["projection_family_complete_verified"])
        self.assertIn("full-family coverage is not established", checked["guarantee"])
        self.assertNotIn("every coherent", checked["guarantee"])


if __name__ == "__main__":
    unittest.main()
