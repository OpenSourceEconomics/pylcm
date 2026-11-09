"""Public native normalization must preserve locality of interpretation gaps."""
from copy import deepcopy
from itertools import product
import unittest

from _lcm_regime_finder import find_regimes as _find_regimes
from functools import partial

# These regressions exercise the optional stronger semantic audit, not the
# Model-graph contract whose primitives retain their declared dependencies.
find_regimes = partial(_find_regimes, analysis_scope='full_semantics')
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.proposal_decision import build_decisions, verify_decisions
from .round5_locality_oracle import unchanged_applications, literal_loop_payoff
from .test_model_adapter import grid


def opaque_utility(x, history):
    result = x
    for weight in (1, 2):
        result = result + weight * history
    return result


def safe_utility(x):
    return x


def variadic_utility(x, history, **extra):
    result = x
    for weight in (1, 2):
        result = result + weight * history
    return result


def native_fixture(*, callback=opaque_utility, values=(0, 1)):
    """Two independently admitted original regimes, no reduced answer supplied."""
    regimes = {}
    for name, function in (("opaque", callback), ("safe", safe_utility)):
        regimes[name] = {"kind": "dataclass", "type": "lcm.Regime", "fields": {
            "states": {"x": grid(list(values)), "history": grid(list(values))},
            "actions": {}, "functions": {"utility": source_function(function)},
            "constraints": {}, "state_transitions": {},
            "semantic_role": {"actor": name, "information": ["x", "history"],
                              "commitments": [], "discount_owner": name}}}
    phase = {"kind": "dataclass", "fields": {"targets_by_period": []}}
    nodes = [[0, name] for name in regimes]
    reachability = {"kind": "dataclass", "fields": {
        "nodes": nodes, "solution": phase, "simulation": phase}}
    return {"schema": "constructed-pylcm-model-graph-1", "n_periods": 1,
            "ages": None, "user_regimes": regimes, "reachability": reachability,
            "graph": {"kind": "dataclass", "fields": {
                "coordinate_kind": "period", "reachability": reachability,
                "initial_nodes": nodes, "laws": {}}},
            "semantic_contract": {"allocation": {"kind": "per_block", "status": "declared"},
                "production_metadata": {"status": "explicit-small-source-regression",
                                        "native_model_constructed": False}}}


class LocalInterpretationTests(unittest.TestCase):
    def produce(self, source=None):
        source = native_fixture() if source is None else source
        report = find_regimes(source, search_mode="exact")
        self.assertIn("proposal_decisions", report, report)
        self.assertFalse(report["semantic_check"]["errors"], report["semantic_check"])
        return source, report

    def test_actual_capture_gap_is_local_on_public_native_path(self):
        source, report = self.produce()
        obligations = report["analysis"]["obligations"]
        self.assertEqual([o["kind"] for o in obligations], ["callback_interpretation"])
        self.assertEqual(report["status"], "UNRESOLVED")
        changed = [r for r in report["proposal_decisions"][1:] if r["changed_interfaces"]]
        self.assertTrue(changed)
        for row in changed:
            self.assertEqual(row["disposition"], "adopt", row)
            self.assertEqual({x["original_regime"] for x in row["changed_interfaces"]}, {"safe"})
            self.assertTrue(any(d["obligation"]["kind"] == "callback_interpretation"
                                for d in row["discharged_unchanged_obligations"]))
        self.assertEqual(verify_decisions(source, report["analysis"], report["scenarios"],
                                         report["proposal_decisions"])["status"], "PASS")

    def test_independent_oracle_and_domain_mutations(self):
        for values in ((-2, 0), (1, 3, 8), (8, 3, 1)):
            with self.subTest(values=values):
                _, report = self.produce(native_fixture(values=values))
                self.assertTrue(all(r["disposition"] == "adopt"
                                    for r in report["proposal_decisions"][1:]))
                for x, history in product(values, repeat=2):
                    actual = {"opaque": ("actual-capture", (x, history))}
                    self.assertTrue(unchanged_applications(actual, deepcopy(actual)))
                    self.assertEqual(opaque_utility(x, history), literal_loop_payoff(x, history))
                    changed = {"opaque": ("actual-capture", (x, history + 1))}
                    self.assertFalse(unchanged_applications(actual, changed))
                    self.assertNotEqual(opaque_utility(x, history), literal_loop_payoff(x, history + 1))

    def test_capture_variadic_and_query_premises_are_never_interpretation(self):
        for kind in ("capture_identity", "variadic_dependency", "policy_query_evaluation",
                     "role_timing_contract", "unrecognized_future_obligation"):
            with self.subTest(kind=kind):
                source, report = self.produce()
                analysis = deepcopy(report["analysis"])
                item = dict(analysis["obligations"][0], id="negative-control", kind=kind)
                analysis["obligations"].append(item)
                rows = build_decisions(source, analysis, report["scenarios"])[1:]
                self.assertTrue(all(r["disposition"] == "investigate" for r in rows))
                self.assertTrue(all(item in r["residual_obligations"] for r in rows))
        _, report = self.produce(native_fixture(callback=variadic_utility))
        self.assertTrue(all(r["disposition"] == "investigate"
                            for r in report["proposal_decisions"][1:]))

    def test_source_binding_and_pointer_mutations_cannot_be_discharged(self):
        source, report = self.produce()
        for field, value in (("source_path", "/user_regimes/safe/fields/functions/utility"),
                             ("identity", "different-source"),
                             ("bindings", {"changed": 1}),
                             ("arguments", ["x"])):
            with self.subTest(field=field):
                analysis = deepcopy(report["analysis"])
                callback = next(c for c in analysis["declarations"]["callbacks"].values()
                                if c.get("status") == "opaque")
                callback[field] = value
                rows = build_decisions(source, analysis, report["scenarios"])[1:]
                self.assertTrue(all(r["disposition"] != "adopt" for r in rows))
                self.assertFalse(any(d["obligation"]["kind"] == "callback_interpretation"
                                     for r in rows for d in r["discharged_unchanged_obligations"]))

    def test_altered_opaque_interface_requires_own_factorization(self):
        source, report = self.produce()
        analysis = deepcopy(report["analysis"])
        cell = next(c for c in analysis["declarations"]["cells"] if c["regime"] == "opaque")
        for option in analysis["interfaces"][cell["id"]]["options"]:
            option["states"].pop("history")
            option["state_projection"]["history"] = {"kind": "drop"}
        rows = build_decisions(source, analysis, report["scenarios"])[1:]
        self.assertTrue(all(r["disposition"] != "adopt" for r in rows))
        self.assertTrue(all(any(o["kind"] == "callback_interpretation"
                               for o in r["residual_obligations"]) for r in rows))

    def test_all_uses_including_second_changed_cell_must_be_unchanged(self):
        source = native_fixture()
        second = source["user_regimes"]["safe"]["fields"]
        second["functions"]["utility"] = deepcopy(
            source["user_regimes"]["opaque"]["fields"]["functions"]["utility"])
        second["states"]["unused"] = grid([0, 1])
        _, report = self.produce(source)
        self.assertTrue(all(r["disposition"] == "investigate"
                            for r in report["proposal_decisions"][1:]))
        self.assertTrue(all(any(o["kind"] == "callback_interpretation"
                               for o in r["residual_obligations"])
                            for r in report["proposal_decisions"][1:]))


if __name__ == "__main__":
    unittest.main()
