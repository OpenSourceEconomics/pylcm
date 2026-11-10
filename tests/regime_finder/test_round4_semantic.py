"""Independent semantic regressions without native Model evaluation."""
from copy import deepcopy
import hashlib
import json
import unittest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics, _closed_native_constant


def callback(name, arguments, expression, *, status="proved"):
    source = f"def {name}({', '.join(arguments)}):\n    return {expression}\n"
    original = {"kind": "callable", "arguments": arguments,
                "raw_arguments": arguments, "source": source, "bindings": {},
                "call_targets": {}, "variadic": False}
    return {"arguments": arguments, "reads": arguments, "source": source,
            "bindings": {}, "expression": expression if status == "proved" else None,
            "status": status, "original": original,
            "identity": hashlib.sha256(source.encode()).hexdigest()}


def one_cell(callbacks, functions, consumers, *, states=None, actions=None, domains=None, period=0):
    return {"schema": "regime-declarations/v1", "source_identity": "explicit round4 fixture",
            "domains": domains or {}, "callbacks": callbacks,
            "cells": [{"id": "c", "regime": "life", "age": 20 + period, "period": period,
                       "states": states or {}, "actions": actions or {}, "functions": functions,
                       "consumers": consumers, "stage": "decision", "role": "one-person",
                       "terminal": True, "admissible_start": True}],
            "relations": [], "characteristics": {}, "obligations": [],
            "allocation": {"status": "known", "kind": "per_block"}}


def produce(source):
    declarations = normalize_declarations(source)
    analysis = MetadataAnalysis(declarations).run()
    return declarations, analysis, check_semantics(source, declarations, analysis)


def grid(values):
    return {"kind": "object", "type": "lcm.DiscreteGrid",
            "fields": {"values": values, "codes": list(range(len(values)))}}


def captured_gate_graph(*, dormant_admitted=False, simulation_edge=True):
    def raw(name, args, body): return callback(name, args, body)["original"]
    def regime():
        return {"kind": "dataclass", "type": "lcm.Regime", "fields": {
            "states": {"history": grid([0, 1])}, "actions": {},
            "functions": {"utility": raw("utility", [], "0")}, "constraints": {},
            "state_transitions": {"history": raw("history", ["history"], "history")}}}
    fallback = {"kind": "dataclass", "type": "lcm.ProjectedRegimeValue", "fields": {
        "regime": "outside", "projection": {"history": raw("project", ["history"], "history")}}}
    gate = {"kind": "dataclass", "type": "_lcm.gated_edge.GatedEdge", "fields": {
        "gate": raw("accept", ["history"], "history > 0"), "gate_refs": {},
        "legs": {"person": {"kind": "dataclass", "type": "lcm.StakeholderRoute",
                               "fields": {"fallback": fallback}}}, "off_grid": "extrapolate"}}
    nodes = [[0, "source"], [1, "landing"], [1, "outside"]]
    starts = [[0, "source"]]
    if dormant_admitted:
        nodes.append([1, "dormant"]); starts.append([1, "dormant"])
    reach = {"kind": "dataclass", "fields": {"nodes": nodes,
        "solution": {"kind": "dataclass", "fields": {"targets_by_period": [{"source": ["landing"]}]}},
        "simulation": {"kind": "dataclass", "fields": {"targets_by_period": [{"source": ["landing"] if simulation_edge else []}]}}}}
    regimes = {name: regime() for name in ("source", "landing", "outside", "dormant")}
    return {"schema": "constructed-pylcm-model-graph-1", "n_periods": 2,
            "ages": {"kind": "object", "fields": {"exact_values": [0, 1]}},
            "user_regimes": regimes,
            "graph": {"kind": "dataclass", "fields": {"coordinate_kind": "age",
                "reachability": reach, "initial_nodes": starts,
                "laws": {"source": {"kind": "dataclass", "fields": {
                    "transition": "landing", "gated_edges": {"landing": gate, "dormant": deepcopy(gate)}}}}}},
            "semantic_contract": {"roles": dict.fromkeys(regimes, "one-person"),
                "allocation": {"status": "known", "kind": "per_block"},
                "production_metadata": {"status": "regression", "no_production_claim": True}}}


def native_constant():
    return {"kind": "dataclass", "type": "_lcm.regime_building.schedules._RegimeCodeConstant",
            "fields": {"value": 1}, "callable": {
                "kind": "callable", "arguments": [], "raw_arguments": ["self"], "variadic": False,
                "capture_contract": "inspection-only-frozen-bindings-1",
                "bindings": {"jnp": {"kind": "module", "name": "jax.numpy"}},
                "bound_self": {"type": "_lcm.regime_building.schedules._RegimeCodeConstant", "fields": {"value": 1}},
                "source": "def __call__(self):\n    return jnp.asarray(self.value)\n"}}


class Round4SemanticTests(unittest.TestCase):
    def test_integral_float_provider_binding_has_same_exact_numeric_image(self):
        for period in (0, 1, 20, 99):
            cbs = {"experience": callback("experience", ["period"], "period + 1.0"),
                   "utility": callback("utility", ["experience"], "experience")}
            source = one_cell(cbs, {"experience": "experience", "utility": "utility"},
                              [{"kind": "reward", "callback": "utility", "phase": "simulation"}], period=period)
            _, analysis, check = produce(source)
            self.assertEqual(check["status"], "PASS", check)
            record = analysis["interfaces"]["c"]["options"][0]["function_interfaces"]["simulation"]["utility"]
            self.assertEqual(record["bound_arguments"]["experience"], period + 1)

    def test_wrong_numeric_or_boolean_binding_is_rejected_without_tolerance(self):
        cbs = {"experience": callback("experience", ["period"], "period + 1.0"),
               "utility": callback("utility", ["experience"], "experience")}
        source = one_cell(cbs, {"experience": "experience", "utility": "utility"},
                          [{"kind": "reward", "callback": "utility", "phase": "simulation"}])
        declarations, analysis, _ = produce(source)
        for replacement in (True, "1", 1.0000000000000002):
            bad = deepcopy(analysis)
            bad["interfaces"]["c"]["options"][0]["function_interfaces"]["simulation"]["utility"]["bound_arguments"]["experience"] = replacement
            check = check_semantics(source, declarations, bad)
            self.assertTrue(any(e["kind"] == "unproved bound original argument" for e in check["errors"]), check)

    def test_consumed_opaque_provider_keeps_proved_singleton_binding(self):
        cbs = {"report": callback("report", ["alpha"], "opaque(alpha)", status="unresolved")}
        source = one_cell(cbs, {"report": "report"},
                          [{"kind": "reporting", "callback": "report", "phase": "simulation"}],
                          states={"alpha": "singleton"},
                          domains={"singleton": {"kind": "categorical", "size": 1, "values": [0.0], "codes": [7]}})
        _, analysis, check = produce(source)
        option = analysis["interfaces"]["c"]["options"][0]
        record = option["function_interfaces"]["simulation"]["report"]
        self.assertEqual(option["state_projection"]["alpha"]["kind"], "drop")
        self.assertEqual(record["implementation"]["kind"], "source_bound_partial_application")
        self.assertEqual(record["bound_arguments"], {"alpha": 0.0})
        self.assertFalse(check["errors"], check)
        self.assertEqual(check["status"], "UNRESOLVED")

    def test_reporting_root_retains_history_and_rejects_provider_omission(self):
        cbs = {"report": callback("report", ["history"], "history"),
               "utility": callback("utility", [], "0")}
        source = one_cell(cbs, {"report": "report", "utility": "utility"},
                          [{"kind": "reward", "callback": "utility", "phase": "simulation"},
                           {"kind": "reporting", "callback": "report", "phase": "simulation"}],
                          states={"history": "two"}, domains={"two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}})
        declarations, analysis, check = produce(source)
        self.assertEqual(check["status"], "PASS", check)
        self.assertEqual(analysis["dependency_closure"]["c"], ["history"])
        bad = deepcopy(analysis)
        record = bad["interfaces"]["c"]["options"][0]["function_interfaces"]["simulation"]["report"]
        record["implementation"] = {"kind": "omit_unconsumed_provider"}; record["arguments"] = []
        errors = check_semantics(source, declarations, bad)["errors"]
        self.assertTrue(any(e["kind"] == "consumed original provider was omitted" for e in errors))

    def test_distinct_represented_feasibility_phases_do_not_intersect_actions(self):
        cbs = {"utility": callback("utility", ["a"], "a"),
               "solve": callback("solve", ["a"], "True"),
               "simulation": callback("simulation", ["a"], "a == 0")}
        consumers = [{"kind": "reward", "callback": "utility", "phase": "both"},
                     {"kind": "feasible", "callback": "solve", "phase": "solve"},
                     {"kind": "feasible", "callback": "simulation", "phase": "simulation"}]
        source = one_cell(cbs, {"utility": "utility"}, consumers, actions={"a": "two"},
                          domains={"two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}})
        _, analysis, check = produce(source)
        self.assertEqual(analysis["interfaces"]["c"]["options"][0]["action_reconstruction"]["a"], {"kind": "identity", "source": "a"})
        self.assertEqual(check["status"], "PASS", check)

    def test_single_truth_feasibility_channel_still_eliminates_forced_action(self):
        cbs = {"utility": callback("utility", ["a"], "a"), "truth": callback("truth", ["a"], "a == 0")}
        source = one_cell(cbs, {"utility": "utility"},
                          [{"kind": "reward", "callback": "utility", "phase": "simulation"},
                           {"kind": "feasible", "callback": "truth", "phase": "simulation",
                            "decision_phases": ["solve", "simulation"]}],
                          actions={"a": "two"}, domains={"two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}})
        _, analysis, check = produce(source)
        self.assertEqual(analysis["interfaces"]["c"]["options"][0]["actions"], {})
        self.assertEqual(check["status"], "PASS", check)

    def test_absent_phase_constraint_does_not_restrict_that_phase(self):
        cbs = {"utility": callback("utility", ["a"], "a"), "solve": callback("solve", ["a"], "a == 0")}
        source = one_cell(cbs, {"utility": "utility"},
                          [{"kind": "reward", "callback": "utility", "phase": "both"},
                           {"kind": "feasible", "callback": "solve", "phase": "solve"}],
                          actions={"a": "two"}, domains={"two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}})
        _, analysis, check = produce(source)
        self.assertEqual(analysis["interfaces"]["c"]["options"][0]["actions"], {"a": "two"})
        self.assertEqual(check["status"], "PASS", check)

    def test_constant_routing_bridge_requires_source_and_receiver_agreement(self):
        original = native_constant()
        self.assertTrue(_closed_native_constant(original))
        mutations = []
        bad = deepcopy(original); bad["fields"]["value"] = 2; mutations.append(bad)
        bad = deepcopy(original); bad["callable"]["arguments"] = ["history"]; mutations.append(bad)
        bad = deepcopy(original); bad["callable"]["source"] = "def __call__(self):\n    return jnp.asarray(history)\n"; mutations.append(bad)
        bad = deepcopy(original); bad["callable"]["bindings"]["jnp"]["name"] = "custom.numpy"; mutations.append(bad)
        for bad in mutations: self.assertFalse(_closed_native_constant(bad))

    def test_constant_route_does_not_erase_real_future_dependency(self):
        cbs = {"zero": callback("zero", [], "0"), "future": callback("future", ["history"], "history"),
               "route": {"arguments": [], "reads": [], "bindings": {}, "expression": None,
                         "status": "opaque", "identity": "original constant route", "original": native_constant()}}
        source = one_cell(cbs, {"utility": "zero"}, [{"kind": "reward", "callback": "zero", "phase": "both"}],
                          states={"history": "two"}, domains={"two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}})
        target = deepcopy(source["cells"][0]); target.update(id="future", age=21, period=1)
        target["functions"] = {"utility": "future"}
        target["consumers"] = [{"kind": "reward", "callback": "future", "phase": "both"}]
        source["cells"][0]["terminal"] = False; source["cells"].append(target)
        source["relations"] = [{"id": "edge", "kind": "transition", "source": "c", "targets": ["future"],
                                "phase": "both", "callbacks": ["route"], "state_updates": {"history": "history"}, "produces": ["history"]}]
        _, analysis, check = produce(source)
        self.assertEqual(analysis["dependency_closure"]["c"], ["history"])
        self.assertFalse(check["errors"], check)

    def test_native_gate_predicate_and_only_active_owned_routes_are_retained(self):
        for dormant in (False, True):
            for simulation in (False, True):
                source = captured_gate_graph(dormant_admitted=dormant, simulation_edge=simulation)
                declarations, analysis, check = produce(source)
                cells = {c["id"]: c for c in declarations["cells"]}
                expected_phases = {"solve", "simulation"} if simulation else {"solve"}
                gates = [r for r in declarations["relations"] if r["kind"] == "gate"]
                self.assertEqual({r["phase"] for r in gates}, expected_phases)
                self.assertTrue(all(cells[r["source"]]["regime"] == "landing" for r in gates))
                self.assertTrue(all(len(r["callbacks"]) == 1 for r in gates))
                fallbacks = [r for r in declarations["relations"] if r["kind"] == "fallback"]
                self.assertEqual({r["phase"] for r in fallbacks}, expected_phases)
                self.assertTrue(all(r["ownership"]["target_regime"] == "landing" for r in fallbacks))
                self.assertFalse(check["errors"], check)

    def test_native_gate_callback_or_owned_leg_mutation_is_independently_rejected(self):
        source = captured_gate_graph()
        declarations, _, _ = produce(source)
        for mutation in ("callback", "ownership", "phase"):
            bad = deepcopy(declarations)
            if mutation == "callback": next(r for r in bad["relations"] if r["kind"] == "gate")["callbacks"] = []
            elif mutation == "ownership": next(r for r in bad["relations"] if r["kind"] == "fallback")["ownership"]["leg"] = "other"
            else: bad["relations"] = [r for r in bad["relations"] if not (r["kind"] == "fallback" and r["phase"] == "simulation")]
            check = check_semantics(source, bad, MetadataAnalysis(bad).run())
            self.assertEqual(check["status"], "FAIL", check)

    def test_native_gate_reward_contrast_retains_the_decisive_history(self):
        source = captured_gate_graph()
        source["graph"]["fields"]["laws"]["source"]["fields"]["gated_edges"].pop("dormant")
        source["user_regimes"]["outside"]["fields"]["functions"]["utility"] = callback("outside_reward", [], "10")["original"]
        declarations, analysis, check = produce(source)
        # Literal oracle: history 0 fails the gate and receives outside value
        # 10; history 1 passes and receives landing value 0. This difference
        # cannot factor through a state-free source or landing interface.
        for cell in declarations["cells"]:
            expected = ["history"] if cell["regime"] in {"source", "landing"} else []
            self.assertEqual(analysis["dependency_closure"][cell["id"]], expected)
        self.assertEqual(check["status"], "PASS", check)

    def test_local_pass_has_scoped_diagnostics_and_no_completeness_claim(self):
        source = captured_gate_graph()
        declarations, analysis, _ = produce(source)
        local = check_semantics(source, declarations, analysis, check_coverage=False)
        self.assertFalse(local["projection_family_complete_verified"])
        self.assertTrue(any(item["scope"] == "family" for item in local["unresolved"]))
        for item in local["errors"] + local["unresolved"]: self.assertIn("scope", item)


if __name__ == "__main__":
    unittest.main()
