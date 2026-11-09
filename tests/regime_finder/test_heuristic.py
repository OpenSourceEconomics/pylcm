"""Check heuristic layouts against small independent exhaustive oracles."""
from copy import deepcopy
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import random
import unittest

from _lcm_regime_finder.frontier import Search, search_frontier
from _lcm_regime_finder.frontier_check import IndependentFamily
from _lcm_regime_finder.heuristic import search_heuristic
from _lcm_regime_finder.heuristic_check import check_heuristic_frontier, choose_candidate
from .round6_frontier_oracle import block as literal_block, brute_family, canonical


METRICS = ("work", "storage", "clarity")


def fixture(n=3, labels=("a",), *, edges=(), boundary_unit=1):
    """Small exact cost input with physical ages deliberately unlike periods."""
    cells, interfaces = [], {}
    for label in labels:
        for period in range(n):
            cid = f"{label}:{period}"
            cell = {
                "id": cid, "regime": label, "period": period, "age": 21 + 3 * period,
                "role": {"actor": "household"}, "ownership": "owner",
                "stage": "decision", "solver_identity": "same-solver",
                "states": {"x": "dx"}, "actions": {"save": "da"},
            }
            cells.append(cell)
            interfaces[cid] = {"options": [{
                "id": "keep", "states": dict(cell["states"]), "actions": dict(cell["actions"]),
                "state_projection": {"x": {"kind": "identity", "target": "x"}},
                "action_reconstruction": {"save": {"kind": "identity", "source": "save"}},
            }]}
    return {
        "declarations": {
            "cells": cells, "domains": {"dx": {"size": 3}, "da": {"size": 2}},
            "allocation": {"kind": "per_block", "status": "declared", "boundary_unit": boundary_unit},
            "relations": [{"source": source, "targets": [target], "support": support,
                           "phase": phase, "kind": kind}
                          for source, target, support, phase, kind in edges],
        },
        "interfaces": interfaces,
        "projection_policies": [{"id": "P0", "option_by_cell": {cell["id"]: "keep" for cell in cells}}],
        "discovery_complete": True,
    }


def score(cost, objective):
    return sum(Fraction(str(objective["weights"][name]))
               * Fraction(cost[name], objective["reference_scales"][name]) for name in METRICS)


def expected_cost(analysis, point):
    policy = next(row["option_by_cell"] for row in analysis["projection_policies"]
                  if row["id"] == point["projection_policy"])
    parts = [literal_block(analysis, part["cells"], policy) for part in point["blocks"]]
    if any(part is None for part in parts):
        raise AssertionError("literal oracle rejected heuristic plan")
    return {name: sum(part["cost"][name] for part in parts) for name in METRICS}


def pairwise_trap():
    """Every pair merge is worse; a three-way merge is worthwhile.

    Three singleton labels have distinct binary axes and all six directed
    same-period links.  The literal costs are original=(30,6,12), each pair
    merge=(34,10,9), all merged=(24,24,6).  At weights (1,.01,.01) and fixed
    unit scales, every pair merge is uphill but the three-way merge wins.
    """
    names = ("a", "b", "c")
    edges = [(f"{a}:0", f"{b}:0", "positive", "solve", "value_read")
             for a in names for b in names if a != b]
    analysis = fixture(1, names, edges=edges, boundary_unit=2)
    analysis["declarations"]["domains"] = {name: {"size": 2} for name in names}
    for cell in analysis["declarations"]["cells"]:
        name = cell["regime"]
        cell["states"], cell["actions"] = {name: name}, {}
        option = analysis["interfaces"][cell["id"]]["options"][0]
        option.update(states={name: name}, actions={}, action_reconstruction={},
                      state_projection={name: {"kind": "identity", "target": name}})
    return analysis


def declarative_coarsening_fixture():
    """Full source declarations: utility uses floor(x/2), x persists exactly."""
    declarations = fixture(2)["declarations"]
    declarations.update(
        schema="regime-declarations/v1", obligations=[],
        source_identity={"fixture": "coherent-categorical-identity-v1"},
        domains={"dx": {"kind": "categorical", "size": 3,
                         "values": [0, 1, 2], "codes": [0, 1, 2]}},
        callbacks={
            "phi": {"kind": "callable", "arguments": ["x"], "reads": ["x"],
                    "source": "def phi(x):\n    return x // 2\n", "expression": "x // 2",
                    "status": "proved", "identity": "fixture-phi-source"},
            "utility": {"kind": "callable", "arguments": ["feature"], "reads": ["feature"],
                        "source": "def utility(feature):\n    return feature\n", "expression": "feature",
                        "status": "proved", "identity": "fixture-utility-source"},
        },
        characteristics={"feature": {"state": "x", "domain": "dx", "callback": "phi", "name": "feature"}},
        relations=[{"id": "carry", "source": "a:0", "targets": ["a:1"],
                    "kind": "transition", "phase": "both", "support": "positive",
                    "state_updates": {"x": {"kind": "identity", "source": "x"}}}],
    )
    for cell in declarations["cells"]:
        cell.update(actions={}, functions={"feature": "phi", "utility": "utility"},
                    consumers=[{"kind": "utility", "callback": "utility", "phase": "both"}])
    return declarations


class HeuristicTests(unittest.TestCase):
    def checked(self, analysis, **kwargs):
        result = search_heuristic(analysis, **kwargs)
        receipt = check_heuristic_frontier(analysis, result)
        self.assertEqual(receipt["status"], "PASS", receipt)
        self.assertFalse(receipt["frontier_complete_verified"])
        self.assertEqual(receipt["source_semantics"], "NOT_CHECKED_HERE")
        self.assertEqual(result["status"], "HEURISTIC")
        self.assertIs(result["search"]["complete"], False)
        self.assertEqual(result["search"]["optimality_gap"], "unknown")
        for point in result["candidates"]:
            self.assertEqual(point["cost"], expected_cost(analysis, point))
        return result

    def test_fixed_group_scalar_interval_optimum_matches_tiny_exhaustive_oracle(self):
        for seed in range(8):
            rng = random.Random(seed)
            edges = [(f"a:{i}", f"a:{j}", rng.choice(("positive", "unknown", "zero")),
                      "both", "transition")
                     for i in range(4) for j in range(4) if rng.random() < .25]
            analysis = fixture(4, edges=edges)
            for period in range(4):
                if rng.random() < .5:
                    option = analysis["interfaces"][f"a:{period}"]["options"][0]
                    option["actions"] = {}
                    option["action_reconstruction"] = {"save": {"kind": "constant", "value": 0}}
            result = self.checked(analysis, recommendation={"allow_final_only_split": True})
            exact = brute_family(analysis)
            self.assertEqual(min(score(point["cost"], result["objective"]) for point in result["candidates"]),
                             min(score(point["cost"], result["objective"]) for point in exact))

    def test_tiny_multilabel_quality_reference_is_a_lower_bound_not_a_claim(self):
        for seed in range(6):
            rng = random.Random(seed)
            analysis = fixture(2, ("a", "b"), edges=[
                (f"{a}:0", f"{b}:{t}", "positive", "solve", "policy_read")
                for a in ("a", "b") for b in ("a", "b") for t in (0, 1)
                if rng.random() < .5])
            result = self.checked(analysis)
            optimum = min(score(point["cost"], result["objective"]) for point in brute_family(analysis))
            self.assertTrue(all(score(point["cost"], result["objective"]) >= optimum
                                for point in result["candidates"]))

    def test_same_period_branching_and_duplicate_incidence_costs(self):
        edges = [("a:0", "b:0", "positive", "solve", "value_read"),
                 ("a:0", "b:0", "positive", "simulation", "policy_read"),
                 ("a:0", "a:1", "unknown", "both", "transition"),
                 ("a:0", "b:1", "positive", "both", "joint_transition"),
                 ("a:0", "a:0", "positive", "solve", "self_read"),
                 ("b:0", "a:1", "zero", "both", "transition")]
        analysis = fixture(2, ("a", "b"), edges=edges, boundary_unit=2)
        policy = analysis["projection_policies"][0]["option_by_cell"]
        self.assertEqual(literal_block(analysis, ["a:0"], policy)["cost"]["work"], 42)
        before = canonical(analysis)
        self.checked(analysis)
        self.assertEqual(canonical(analysis), before)  # typed and phased metadata stays untouched

    def test_per_block_storage_and_original_baseline(self):
        analysis = fixture(1, ("a", "b"))
        analysis["declarations"]["domains"]["dy"] = {"size": 5}
        cell = analysis["declarations"]["cells"][1]
        cell["states"] = {"y": "dy"}
        option = analysis["interfaces"][cell["id"]]["options"][0]
        option["states"] = dict(cell["states"])
        option["state_projection"] = {"y": {"kind": "identity", "target": "y"}}
        result = self.checked(analysis)
        self.assertEqual(result["original"]["cost"]["storage"], 8)
        policy = analysis["projection_policies"][0]["option_by_cell"]
        self.assertEqual(literal_block(analysis, ["a:0", "b:0"], policy)["cost"]["storage"], 30)
        self.assertEqual(result["objective"]["reference_scales"]["storage"], 8)

    def test_original_baseline_is_not_the_only_offered_reduced_policy(self):
        analysis = fixture(2)
        analysis["declarations"]["domains"]["history"] = {"size": 100}
        for cell in analysis["declarations"]["cells"]:
            cell["states"]["history"] = "history"
            analysis["interfaces"][cell["id"]]["options"][0]["state_projection"]["history"] = {"kind": "drop"}
        result = self.checked(analysis)
        self.assertEqual(result["original"]["cost"], {"work": 1200, "storage": 600, "clarity": 4})
        self.assertEqual(result["objective"]["reference_scales"], result["original"]["cost"])
        self.assertEqual(min(point["cost"]["work"] for point in result["candidates"]), 12)
        self.assertNotEqual(result["recommendation"]["graph_id"], "retain-original")
        self.assertEqual(result["recommendation"]["authority"], "cost_only")

    def test_incompatible_grids_owners_roles_and_solvers_cannot_merge(self):
        for kind in ("domain", "ownership", "role", "solver_identity"):
            analysis = fixture(1, ("a", "b"))
            cell = analysis["declarations"]["cells"][1]
            if kind == "domain":
                analysis["declarations"]["domains"]["other"] = {"size": 3}
                cell["states"]["x"] = "other"
                analysis["interfaces"]["b:0"]["options"][0]["states"]["x"] = "other"
            else:
                cell[kind] = {"actor": "another"} if kind == "role" else "another"
            result = self.checked(analysis)
            self.assertTrue(all(len(part["cells"]) == 1
                                for point in result["candidates"] for part in point["blocks"]))

    def test_coherent_policy_moves_do_not_mix_cell_options(self):
        analysis = fixture(2)
        analysis["declarations"]["domains"]["coarse"] = {"size": 2}
        for cid, record in analysis["interfaces"].items():
            reduced = deepcopy(record["options"][0])
            reduced.update(id="coarse", states={"class": "coarse"}, state_projection={
                "x": {"kind": "characteristic", "callback": "known_map", "target": "class",
                      "domain": "coarse", "values": [0, 0, 1]}})
            record["options"].append(reduced)
        analysis["projection_policies"].append({"id": "P1", "option_by_cell": {
            cell["id"]: "coarse" for cell in analysis["declarations"]["cells"]}})
        result = self.checked(analysis)
        policies = {p["id"]: p["option_by_cell"] for p in analysis["projection_policies"]}
        for point in result["candidates"]:
            actual = {cid: option for part in point["blocks"] for cid, option in part["option_by_cell"].items()}
            self.assertEqual(actual, policies[point["projection_policy"]])
        bad = deepcopy(result)
        part = bad["candidates"][0]["blocks"][0]
        cid = part["cells"][0]
        part["option_by_cell"][cid] = "keep" if part["option_by_cell"][cid] == "coarse" else "coarse"
        self.assertEqual(check_heuristic_frontier(analysis, bad)["status"], "FAIL")

    def test_terminal_cosmetic_preference_has_explicit_override(self):
        analysis = fixture(3)
        analysis["declarations"]["domains"]["history"] = {"size": 20}
        for cell in analysis["declarations"]["cells"]:
            cell["states"]["history"] = "history"
            analysis["interfaces"][cell["id"]]["options"][0]["state_projection"]["history"] = {"kind": "drop"}
        last = analysis["interfaces"]["a:2"]["options"][0]
        last.update(actions={}, action_reconstruction={"save": {"kind": "constant", "value": 0}})
        configuration = {"weights": {"clarity": "1/1000000"}}
        ordinary = self.checked(analysis, recommendation=configuration)
        selected = ordinary["recommendation"]["graph_id"]
        self.assertNotIn(selected, ordinary["recommendation"]["endpoint_preference"]["deferred_graph_ids"])
        override = self.checked(analysis, recommendation=dict(configuration, allow_final_only_split=True))
        self.assertIn(override["recommendation"]["graph_id"],
                      override["recommendation"]["endpoint_preference"]["deferred_graph_ids"])

    def test_strict_improvement_and_source_eligible_filter_keep_original(self):
        analysis = fixture(2)
        result = self.checked(analysis)
        self.assertEqual(result["recommendation"]["graph_id"], "retain-original")
        self.assertEqual(choose_candidate(analysis, result, allowed_ids=set())["operation"], "retain_original")
        self.assertEqual(result["objective"]["reference_kind"], "original")

    def test_unpriceable_original_retains_identity_even_with_explicit_scales(self):
        analysis = fixture(2)
        # The original label crosses protected contexts.  Keeping the original
        # remains an identity action, but the block model cannot compare it as
        # one admissible regime.  No surrogate improvement is then asserted.
        analysis["declarations"]["cells"][1]["ownership"] = "second-owner"
        result = self.checked(analysis)
        self.assertIsNone(result["original"]["cost"])
        self.assertEqual(result["objective"]["reference_kind"], "unit_fallback")
        self.assertEqual(result["recommendation"]["operation"], "retain_original")
        explicit = self.checked(analysis, recommendation={"reference": dict.fromkeys(METRICS, 1)})
        self.assertEqual(explicit["recommendation"]["operation"], "retain_original")

    def test_per_policy_eligible_candidate_survives_a_cheaper_other_policy(self):
        analysis = fixture(2)
        analysis["declarations"]["domains"]["history"] = {"size": 100}
        for cell in analysis["declarations"]["cells"]:
            cell["states"]["history"] = "history"
            row = analysis["interfaces"][cell["id"]]
            row["options"][0]["state_projection"]["history"] = {"kind": "drop"}
            alternative = deepcopy(row["options"][0])
            alternative.update(id="drop-all", states={}, state_projection={
                "x": {"kind": "drop"}, "history": {"kind": "drop"}})
            row["options"].append(alternative)
        analysis["projection_policies"].append({"id": "P1", "option_by_cell": {
            cell["id"]: "drop-all" for cell in analysis["declarations"]["cells"]}})
        result = self.checked(analysis)
        self.assertTrue(all(point["projection_policy"] == "P1" for point in result["frontier"]))
        safe_policy_candidates = {point["id"] for point in result["candidates"] if point["projection_policy"] == "P0"}
        self.assertTrue(safe_policy_candidates)
        selected = choose_candidate(analysis, result, allowed_ids=safe_policy_candidates)
        self.assertIn(selected["graph_id"], safe_policy_candidates)
        self.assertEqual(selected["authority"], "cost_only")  # ranking never grants source authority

    def test_pair_merge_local_minimum_is_not_global_optimality(self):
        analysis = pairwise_trap()
        policy = analysis["projection_policies"][0]["option_by_cell"]
        singleton = [literal_block(analysis, [f"{name}:0"], policy) for name in ("a", "b", "c")]
        original = {name: sum(part["cost"][name] for part in singleton) for name in METRICS}
        objective = {"weights": {"work": "1", "storage": "1/100", "clarity": "1/100"},
                     "reference_scales": dict.fromkeys(METRICS, 1)}
        self.assertEqual(tuple(original[name] for name in METRICS), (30, 6, 12))
        for a, b, other in (("a", "b", "c"), ("a", "c", "b"), ("b", "c", "a")):
            parts = [literal_block(analysis, [f"{a}:0", f"{b}:0"], policy),
                     literal_block(analysis, [f"{other}:0"], policy)]
            cost = {name: sum(part["cost"][name] for part in parts) for name in METRICS}
            self.assertEqual(tuple(cost[name] for name in METRICS), (34, 10, 9))
            self.assertGreater(score(cost, objective), score(original, objective))
        together = literal_block(analysis, ["a:0", "b:0", "c:0"], policy)["cost"]
        self.assertEqual(tuple(together[name] for name in METRICS), (24, 24, 6))
        self.assertLess(score(together, objective), score(original, objective))
        result = self.checked(analysis, recommendation={"weights": objective["weights"],
                              "reference": objective["reference_scales"]}, effort={"rounds": 2})
        self.assertEqual(result["search"]["optimality_gap"], "unknown")
        self.assertEqual(result["recommendation"]["graph_id"], "retain-original")
        best_examined = min(score(point["cost"], objective) for point in result["candidates"])
        self.assertGreater(best_examined, min(score(point["cost"], objective)
                                             for point in brute_family(analysis)))

    def test_width_one_chain_never_enters_exponential_assignment_or_checker_cover(self):
        # n-1 A signatures can each be assigned to A or A-union-B; the final
        # A-union-B provider is forced.  The native assignment count is 2**(n-1)
        # although every directed link goes to the immediately next period.
        n = 5
        labels = tuple(f"r{i}" for i in range(n))
        analysis = fixture(1, labels)
        analysis["declarations"]["analysis_scope"] = "pylcm-structural-graph/v1"
        analysis["declarations"]["domains"]["dy"] = {"size": 2}
        for i, cell in enumerate(analysis["declarations"]["cells"]):
            cell["period"], cell["age"] = i, 20 + 2 * i
            cell["actions"] = {}
            option = analysis["interfaces"][cell["id"]]["options"][0]
            option.update(actions={}, action_reconstruction={})
            if i == n - 1:
                cell["states"]["y"] = "dy"
                option["states"]["y"] = "dy"
                option["state_projection"]["y"] = {"kind": "identity", "target": "y"}
        analysis["declarations"]["relations"] = [
            {"source": f"r{i}:0", "targets": [f"r{i+1}:0"], "support": "positive"}
            for i in range(n - 1)]
        # Instrumentation proves absence of expensive calls; it is not a
        # claimed defect under a monkeypatched scientific environment.
        with patch.object(Search, "run", side_effect=AssertionError("exact producer invoked")), \
             patch.object(Search, "signature_frontier", side_effect=AssertionError("assignment cover invoked")), \
             patch.object(IndependentFamily, "complete_frontier", side_effect=AssertionError("exact checker invoked")), \
             patch.object(IndependentFamily, "signature_points", side_effect=AssertionError("checker assignment cover invoked")):
            self.checked(analysis, effort={"rounds": 1})

    def test_input_permutations_keep_costs_and_selection(self):
        analysis = fixture(2, ("a", "b", "c"))
        before = self.checked(analysis)
        permuted = deepcopy(analysis)
        permuted["declarations"]["cells"].reverse()
        permuted["declarations"]["relations"].reverse()
        permuted["interfaces"] = dict(reversed(list(permuted["interfaces"].items())))
        after = self.checked(permuted)
        self.assertEqual([point["cost"] for point in before["frontier"]],
                         [point["cost"] for point in after["frontier"]])
        self.assertEqual(before["recommendation"], after["recommendation"])

    def test_candidate_and_source_binding_negative_controls(self):
        analysis = fixture(3, ("a", "b"), edges=[("a:0", "b:0", "positive", "solve", "value_read")])
        result = self.checked(analysis)
        for mutation in ("cost", "date", "age", "coverage", "axes", "detail", "complete", "baseline", "reference"):
            bad = deepcopy(result)
            first = bad["candidates"][0]
            part = first["blocks"][0]
            if mutation == "cost": first["cost"]["work"] += 1
            elif mutation == "date": part["periods"][0] += 1
            elif mutation == "age": part["ages"][0] += 1
            elif mutation == "coverage": part["cells"].pop()
            elif mutation == "axes": part["state_axes"] = {}
            elif mutation == "detail": part["cost_detail"]["boundary_incidences"] += 1
            elif mutation == "complete": bad["search"]["complete"] = True
            elif mutation == "baseline": bad["original"]["cost"]["storage"] += 1
            elif mutation == "reference": bad["objective"]["reference_scales"]["work"] += 1
            with self.subTest(mutation=mutation):
                self.assertEqual(check_heuristic_frontier(analysis, bad)["status"], "FAIL")
        for field, value in (("period", 91), ("ownership", "new-owner"), ("solver_identity", "another-solver")):
            altered = deepcopy(analysis)
            altered["declarations"]["cells"][0][field] = value
            self.assertEqual(check_heuristic_frontier(altered, result)["status"], "FAIL")
        changed = deepcopy(analysis)
        changed["declarations"]["relations"][0]["phase"] = "simulation"
        self.assertEqual(check_heuristic_frontier(changed, result)["status"], "FAIL")

    def test_recomputed_costs_do_not_authorize_noninterval_or_partial_date_blocks(self):
        for same_date in (False, True):
            analysis = fixture(3, ("a", "b") if same_date else ("a",))
            search = Search(analysis)
            policy = analysis["projection_policies"][0]["option_by_cell"]
            members = ([["a:0", "b:0", "a:1"], ["b:1", "a:2", "b:2"]] if same_date
                       else [["a:0", "a:2"], ["a:1"]])
            blocks = [search.block(ids, policy) for ids in members]
            bad_plan = {"projection_policy": "P0", "blocks": blocks,
                        "cost": {name: sum(part["cost"][name] for part in blocks) for name in METRICS}}
            errors = IndependentFamily(analysis).admitted_plan(bad_plan)
            self.assertIn("block is not an interval of complete exact date buckets", errors)

    def test_cooperative_stop_keeps_literal_original_without_claiming_optimum(self):
        analysis = fixture(4, ("a", "b"))
        result = self.checked(analysis, limits={"seconds": 0})
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["recommendation"]["operation"], "retain_original")
        self.assertEqual(result["search"]["evaluated_candidates"], 0)

    def test_exact_api_remains_an_explicit_tiny_oracle(self):
        analysis = fixture(2, ("a", "b"))
        exact = search_frontier(analysis)
        self.assertEqual(exact["status"], "COMPLETE")
        self.assertTrue(exact["certificate"]["complete"])
        heuristic = self.checked(analysis, effort={"rounds": 0})
        self.assertNotEqual(exact["schema"], heuristic["schema"])


class ProductionIntegrationTests(unittest.TestCase):
    def test_disconnected_coarsening_regions_do_not_enumerate_the_powerset(self):
        from _lcm_regime_finder import analyze_production, verify_production_result
        from _lcm_regime_finder.production import MetadataAnalysis
        source = declarative_coarsening_fixture()
        template = source['cells'][0]
        source['cells'] = [dict(deepcopy(template), id=f'a:{i}', period=i, age=21+i)
                           for i in range(32)]
        source['relations'] = []
        with patch.object(MetadataAnalysis, 'closed_regions', side_effect=AssertionError('exact region powerset invoked')):
            result = analyze_production(source)
            checked = verify_production_result(source, result)
        self.assertEqual(checked['status'], 'PASS', checked)
        self.assertFalse(result['analysis']['discovery_complete'])
        self.assertEqual(len(result['analysis']['projection_policies']), 2)
        self.assertTrue(any(row['authority'] == 'source_checked' for row in result['recommendations']))

    def test_heuristic_label_region_is_closed_under_original_identity_links(self):
        from _lcm_regime_finder.production import MetadataAnalysis
        source = declarative_coarsening_fixture()
        source['cells'][1]['regime'] = 'b'
        analysis = MetadataAnalysis(source, search_mode='heuristic').run()
        candidate = analysis['characteristic_candidates'][0]
        regions = MetadataAnalysis(source, search_mode='heuristic').heuristic_regions(candidate)[0]
        self.assertEqual({tuple(row['active_cells']) for row in regions},
                         {('a:0', 'a:1'), ('a:1',)})
        from _lcm_regime_finder.semantic_check import check_semantics
        receipt = check_semantics(source, source, analysis, check_coverage=False)
        self.assertFalse(receipt['errors'], receipt)

    def test_heuristic_coarsening_axes_do_not_form_a_cartesian_policy_family(self):
        from _lcm_regime_finder.production import MetadataAnalysis
        from _lcm_regime_finder.semantic_check import check_semantics
        source = declarative_coarsening_fixture()
        for axis in ('y', 'z'):
            name = 'feature_' + axis
            source['callbacks'][name] = dict(source['callbacks']['phi'], arguments=[axis], reads=[axis],
                source=f'def {name}({axis}):\n    return {axis} // 2\n', expression=f'{axis} // 2', identity=name)
            source['characteristics'][name] = {'state': axis, 'domain': 'dx', 'callback': name, 'name': name}
            for cell in source['cells']:
                cell['states'][axis] = 'dx'
                cell['functions'][name] = name
            source['relations'][0]['state_updates'][axis] = {'kind': 'identity', 'source': axis}
        source['callbacks']['utility'].update(arguments=['feature', 'feature_y', 'feature_z'],
            reads=['feature', 'feature_y', 'feature_z'], expression='feature + feature_y + feature_z',
            source='def utility(feature, feature_y, feature_z):\n    return feature + feature_y + feature_z\n')
        with patch('_lcm_regime_finder.production.product', side_effect=AssertionError('Cartesian policies invoked')):
            analysis = MetadataAnalysis(source, search_mode='heuristic').run()
        self.assertEqual(len(analysis['projection_policies']), 5)
        self.assertFalse(analysis['discovery_complete'])
        self.assertFalse(check_semantics(source, source, analysis, check_coverage=False)['errors'])

    def test_useful_change_is_source_checked_before_public_delivery(self):
        from _lcm_regime_finder import analyze_production, verify_production_result
        from _lcm_regime_finder.cli import candidate_boundary_maps
        path = Path(__file__).resolve().parents[2] / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
        source = json.loads(path.read_text())
        result = analyze_production(source, recommendation={
            "weights": {"work": 1, "storage": 1, "clarity": "1/10"}})
        selected = result["recommendations"][0]
        self.assertEqual(selected["authority"], "source_checked")
        self.assertEqual(selected["operation"], "representation_specification")
        self.assertEqual(verify_production_result(source, result)["status"], "PASS")
        rows = candidate_boundary_maps(result)
        self.assertTrue(rows)
        self.assertTrue(all(row["transformation_authorized"] for row in rows))
        changed = deepcopy(source)
        changed["cells"][0]["ownership"] = "different-owner"
        self.assertEqual(verify_production_result(changed, result)["status"], "FAIL")

    def test_cost_admission_never_substitutes_for_global_policy_semantics(self):
        from _lcm_regime_finder.production import MetadataAnalysis
        from _lcm_regime_finder.proposal_decision import build_decisions
        from _lcm_regime_finder.semantic_check import check_semantics
        original = declarative_coarsening_fixture()
        analysis = MetadataAnalysis(original).run()
        source_receipt = check_semantics(original, original, analysis, check_coverage=False)
        self.assertFalse(source_receipt["errors"], source_receipt)
        self.assertTrue(all(item["scope"] == "family" for item in source_receipt["unresolved"]))
        keep, coarse = {}, {}
        for cid, row in analysis["interfaces"].items():
            for option in row["options"]:
                target = coarse if option["state_projection"]["x"]["kind"] == "characteristic" else keep
                target[cid] = option["id"]
        changed = deepcopy(analysis)
        changed["projection_policies"] = [{"id": "invalid-offered-policy", "option_by_cell": {
            "a:0": coarse["a:0"], "a:1": keep["a:1"]}}]
        scenario = search_heuristic(changed, effort={"rounds": 0})
        self.assertEqual(check_heuristic_frontier(changed, scenario)["status"], "PASS")
        semantic = check_semantics(original, original, changed, check_coverage=False)
        self.assertTrue(any(error["kind"] == "future projected identity law loses history"
                            for error in semantic["errors"]), semantic)
        decisions = build_decisions(original, changed, [scenario])
        proposed = [row for row in decisions if row["operation"] == "representation_specification"]
        self.assertTrue(proposed)
        self.assertTrue(all(row["disposition"] == "reject" for row in proposed))

    def test_actual_entry_points_keep_exact_optional_and_replay_source_guards(self):
        from _lcm_regime_finder import analyze_production, verify_production_result
        from .test_model_adapter import snapshot
        source = snapshot()
        result = analyze_production(source)
        self.assertTrue(result["scenarios"])
        self.assertTrue(all(row["schema"] == "regime-heuristic-frontier/v1" for row in result["scenarios"]))
        self.assertNotEqual(result["status"], "COMPLETE")
        checked = verify_production_result(source, result)
        self.assertEqual(checked["status"], "PASS", checked)
        for row in result["recommendations"]:
            self.assertIn(row["authority"], ("source_checked", "retain_original"))
        exact = analyze_production(source, search_mode="exact")
        self.assertEqual(exact["status"], "COMPLETE")
        self.assertTrue(exact["scenarios"][0]["certificate"]["complete"])
        corrupted = deepcopy(result)
        callback = next(item for item in corrupted["analysis"]["declarations"]["callbacks"].values()
                        if item.get("arguments") == ["x", "history"])
        callback["arguments"] = []
        callback["reads"] = []
        self.assertEqual(verify_production_result(source, corrupted)["status"], "FAIL")
        # A producer flag cannot transform an unresolved/retained representation
        # into source authority; the independently replayed recommendation binds.
        corrupted = deepcopy(result)
        corrupted["recommendations"][0]["graph_id"] = "invented-graph"
        corrupted["recommendations"][0]["authority"] = "source_checked"
        self.assertEqual(verify_production_result(source, corrupted)["status"], "FAIL")
        corrupted = deepcopy(result)
        corrupted["status"] = "COMPLETE"
        self.assertEqual(verify_production_result(source, corrupted)["status"], "FAIL")
        expired = verify_production_result(source, result, limits={"seconds": 0})
        self.assertNotEqual(expired["status"], "PASS")
        self.assertTrue(all(row["operation"] == "retain_original" for row in expired.get("recommendations", [])))


if __name__ == "__main__":
    unittest.main()
