"""Public declaration specs must preserve the native query/visit distinction."""
from copy import deepcopy
import unittest

from _lcm_regime_finder import find_regimes
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.graph_view import typed_incidence_graph
from _lcm_regime_finder.model_snapshot import read_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from tests.regime_finder.query_admission_oracle import physical_closure, demanded_owners
from tests.regime_finder.test_model_adapter import function, grid
from tests.regime_finder.test_query_owners import policy_snapshot


def record(kind, **fields):
    return {"kind": "dataclass", "type": "lcm." + kind, "fields": fields}


def reference(target, kind="ProjectedRegimeValue"):
    return record(kind, regime=target,
                  projection={"x": function("def project(x):\n    return x\n", ["x"])})


def spec(*, names=("A", "B", "Q", "R"), reference_kind="ProjectedRegimeValue",
         slot="same_period_refs", dates=None):
    a, b, q, r = names
    base = record("Regime", states={"x": grid([0, 1])}, actions={},
                  functions={"utility": function("def utility(x):\n    return x\n", ["x"])},
                  constraints={}, state_transitions={"x": function("def next_x(x):\n    return x\n", ["x"])})
    regimes = {name: deepcopy(base) for name in names}
    for name in (a, b):
        regimes[name]["fields"][slot] = {"quoted": reference(q, reference_kind)}
    regimes[q]["fields"]["same_period_refs"] = {"nested": reference(r)}
    result = {"regimes": regimes, "n_periods": 3, "initial_nodes": [[0, a]],
              "edges": {a: record("Transition", law=b),
                        b: record("Transition", law=b)}}
    if dates is not None:
        result["ages"] = record("AgeGrid", exact_values=list(dates))
        result["initial_nodes"] = [[dates[0], a]]
    return result


class QuerySpecAdmissionTests(unittest.TestCase):
    def check_source(self, raw):
        normalized = normalize_declarations(raw)
        analysis = MetadataAnalysis(normalized).run()
        checked = check_semantics(raw, normalized, analysis)
        self.assertNotEqual(checked["status"], "FAIL", checked)
        return normalized, analysis, checked

    def test_public_spec_route_matches_literal_physical_closure(self):
        raw = read_declarations(spec())
        result = find_regimes(raw, budget={"seconds": 15})
        normalized = result["analysis"]["declarations"]
        expected = physical_closure({(0, "A")}, [((0, "A"), (1, "B")),
                                                        ((1, "B"), (2, "B"))])
        actual = {(c["period"], c["regime"]) for c in normalized["cells"]}
        self.assertEqual(actual, expected)
        self.assertEqual({c["regime"] for c in normalized["cells"]}, {"A", "B"})
        reads = [((period, owner, phase), (period, "Q", phase))
                 for period, owner in expected for phase in ("solve", "simulation")]
        reads += [((period, "Q", phase), (period, "R", phase))
                  for period in range(3) for phase in ("solve", "simulation")]
        self.assertEqual({(q["period"], q["regime"], q["phase"])
                          for q in normalized["query_owners"].values()},
                         demanded_owners(expected, reads))
        self.assertTrue(all(q["storage"] == "not_asserted"
                            for q in normalized["query_owners"].values()))

    def test_projected_reads_and_native_exit_shapes_preserve_owner_kind(self):
        for form in ("projected", "exit"):
            for names in (("A", "B", "Q", "R"), ("reader", "later", "partner", "outside")):
                with self.subTest(form=form, names=names):
                    raw = spec(names=names)
                    if form == "exit":
                        for name in names[:2]:
                            declaration = raw["regimes"][name]["fields"]
                            declaration["same_period_refs"] = {}
                            target = reference(names[2], "ExitTarget")
                            target["fields"].update(role="f", utility_adjustment=None)
                            counterparty = reference(names[3], "Counterparty")
                            counterparty["fields"]["role"] = "m"
                            declaration["exits"] = {"marry": record("Exit", available=None,
                                observes=[], targets={"self": target},
                                rule=record("Consent", counterparty=counterparty))}
                    normalized = normalize_declarations(raw)
                    self.assertEqual({c["regime"] for c in normalized["cells"]}, set(names[:2]))
                    self.assertEqual({q["regime"] for q in normalized["query_owners"].values()}, set(names[2:]))

    def test_phase_and_exact_fractional_date_are_preserved(self):
        raw = spec(dates=[20, 20.5, 21])
        for name in ("A", "B"):
            raw["regimes"][name]["fields"]["same_period_refs"] = {
                "quoted": record("Phased", solve=reference("Q"), simulate=reference("R"))}
        raw["regimes"]["Q"]["fields"]["same_period_refs"] = {}
        normalized, _, _ = self.check_source(raw)
        owners = normalized["query_owners"].values()
        self.assertEqual({(q["regime"], q["phase"]) for q in owners},
                         {("Q", "solve"), ("R", "simulation")})
        self.assertEqual({q["coordinate"] for q in owners}, {"20", "41/2", "21"})

    def test_explicit_initial_or_scheduled_target_remains_physical(self):
        raw = spec()
        raw["initial_nodes"].append([2, "Q"])
        raw["edges"]["B"] = record("Transition", law="R")
        normalized, _, _ = self.check_source(raw)
        physical = {(c["period"], c["regime"]) for c in normalized["cells"]}
        self.assertEqual(physical, {(0, "A"), (1, "B"), (2, "R"), (2, "Q")})
        self.assertFalse(any(q["period"] == 2 and q["regime"] in {"Q", "R"}
                             for q in normalized["query_owners"].values()))

    def test_independent_checker_rejects_manufactured_physical_visit(self):
        raw = spec()
        normalized, analysis, _ = self.check_source(raw)
        changed = deepcopy(normalized)
        false_cell = deepcopy(changed["cells"][0])
        false_cell.update(id="extra-query-cell", regime="Q")
        changed["cells"].append(false_cell)
        checked = check_semantics(raw, changed, analysis, check_coverage=False)
        self.assertEqual(checked["status"], "FAIL")
        self.assertTrue(any(e["kind"] == "original sparse date/regime admission changed"
                            for e in checked["errors"]))

    def test_gate_read_is_query_but_explicit_fallback_is_physical(self):
        raw = spec()
        for name in ("A", "B"):
            raw["regimes"][name]["fields"]["same_period_refs"] = {}
        raw["regimes"]["Q"]["fields"]["same_period_refs"] = {}
        gate = record("Gate", predicate=function("def allowed(x):\n    return x > 0\n", ["x"]),
                      references={"price": reference("Q")},
                      routes={"self": record("StakeholderRoute", fallback=reference("R"))})
        raw["edges"]["A"] = record("Transition", law="B", gates={"B": gate})
        normalized, _, _ = self.check_source(raw)
        self.assertEqual({c["regime"] for c in normalized["cells"]}, {"A", "B", "R"})
        self.assertEqual({q["regime"] for q in normalized["query_owners"].values()}, {"Q"})
        self.assertEqual({q["period"] for q in normalized["query_owners"].values()}, {1})

    def test_unproved_fallback_type_is_not_invented_as_settlement(self):
        raw = spec()
        gate = record("Gate", routes={"self": record("StakeholderRoute",
                                                      fallback=reference("R", "Counterparty"))})
        raw["edges"]["A"] = record("Transition", law="B", gates={"B": gate})
        result = find_regimes(raw, budget={"seconds": 15})
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIn("settlement contract unavailable", result.get("error", ""))
        self.assertEqual(result["scenarios"], [])


class QueryIncidenceTests(unittest.TestCase):
    def test_nested_owned_reads_are_visible_to_topological_traversal(self):
        raw = policy_snapshot()
        raw["user_regimes"]["query_partner"]["fields"]["same_period_refs"] = {
            "nested": reference("unused_owner")}
        normalized = normalize_declarations(raw)
        # Two equal reads remain two distinct owned incidences.
        owner = next(q for q in normalized["query_owners"].values()
                     if q["regime"] == "query_partner")
        owner["dependencies"].append(deepcopy(owner["dependencies"][0]))
        owner["dependencies"].append({"unavailable_owner": "missing",
                                       "source_path": "/deliberate-missing-owner-control",
                                       "reference_sha256": "control"})
        view = typed_incidence_graph(normalized)
        node_ids = {node["id"] for node in view["nodes"]}
        self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids
                            for edge in view["edges"]))
        relations = [node for node in view["nodes"] if node["node_kind"] == "policy_query_dependency"]
        self.assertEqual(len(relations), sum(len(q["dependencies"])
                                             for q in normalized["query_owners"].values()))
        owned = [node for node in relations if node["query_owner"] == owner["id"]]
        self.assertEqual(len(owned), 3)
        self.assertEqual(len({node["id"] for node in owned}), 3)
        self.assertTrue(any(node["node_kind"] == "unavailable_policy_query_owner"
                            for node in view["nodes"]))


if __name__ == "__main__":
    unittest.main()
