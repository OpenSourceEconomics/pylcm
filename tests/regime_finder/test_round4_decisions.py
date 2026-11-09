"""Independent decision replay and conservative local-obligation regressions."""
from copy import deepcopy
import unittest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.proposal_decision import build_decisions, verify_decisions, prepare_decision_context
from .tiny_metadata_family import original


def produce(source=None):
    source = original() if source is None else source
    normalized = normalize_declarations(source)
    analysis = MetadataAnalysis(normalized).run()
    return source, analysis, [search_frontier(analysis)]


class ProposalDecisionTests(unittest.TestCase):
    def test_source_and_candidate_anchors_and_local_adoption(self):
        source, analysis, scenarios = produce()
        records = build_decisions(source, analysis, scenarios)
        self.assertEqual(records[0]["operation"], "retain_original")
        self.assertEqual(records[0]["disposition"], "adopt")
        self.assertEqual(len(records), 1 + len(scenarios[0]["frontier"]))
        for row in records[1:]:
            self.assertEqual(row["disposition"], "adopt", row)
            self.assertEqual(row["scenario_index"], 0)
            self.assertEqual(len(row["source_snapshot_sha256"]), 64)
            self.assertEqual(len(row["candidate_sha256"]), 64)
            self.assertFalse(row["family_coverage"]["frontier_complete_verified"])
            self.assertIsNone(row["structural_cost_tradeoff"]["wall_time_saving"])
        self.assertEqual(verify_decisions(source, analysis, scenarios, records)["status"], "PASS")

    def test_false_saved_adopt_is_not_evidence(self):
        source, analysis, scenarios = produce()
        records = build_decisions(source, analysis, scenarios)
        records[1]["source_snapshot_sha256"] = "0" * 64
        self.assertEqual(verify_decisions(source, analysis, scenarios, records)["status"], "FAIL")
        records = build_decisions(source, analysis, scenarios, seconds=0)
        records[1]["disposition"] = "adopt"
        self.assertEqual(verify_decisions(source, analysis, scenarios, records)["status"], "FAIL")

    def test_altered_source_or_structural_cost_rejects(self):
        source, analysis, scenarios = produce()
        changed = deepcopy(source)
        changed["cells"][0]["role"]["actor"] = "different actor"
        records = build_decisions(changed, analysis, scenarios)
        self.assertTrue(all(row["disposition"] == "reject" for row in records[1:]))
        changed_scenarios = deepcopy(scenarios)
        changed_scenarios[0]["frontier"][0]["cost"]["work"] -= 1
        records = build_decisions(source, analysis, changed_scenarios)
        self.assertEqual(records[1]["disposition"], "reject")
        self.assertTrue(records[1]["semantic_checks"]["candidate_admission_errors"])

    def test_partial_frontier_can_have_locally_adoptable_candidate(self):
        source, analysis, _ = produce()
        scenario = search_frontier(analysis, limits={"segments": 0})
        self.assertEqual(scenario["status"], "INCOMPLETE")
        records = build_decisions(source, analysis, [scenario])
        self.assertTrue(all(row["disposition"] == "adopt" for row in records[1:]), records)
        self.assertTrue(all(not row["family_coverage"]["frontier_complete_verified"] for row in records))

    def test_conditional_allocation_is_investigation_not_measured_speed(self):
        source, analysis, scenarios = produce()
        scenario = deepcopy(scenarios[0])
        scenario["allocation_contract"]["status"] = "conditional_scenario"
        records = build_decisions(source, analysis, [scenario])
        for row in records[1:]:
            self.assertEqual(row["disposition"], "investigate", row)
            self.assertEqual(row["structural_cost_tradeoff"]["authority"], "conditional")
            self.assertEqual(row["semantic_checks"]["status"], "PASS")

    def test_scenario_cannot_promote_its_own_allocation_authority(self):
        source = original()
        source["allocation"]["status"] = "unresolved"
        source, analysis, scenarios = produce(source)
        scenarios[0]["allocation_contract"] = dict(scenarios[0]["allocation_contract"], status="known")
        records = build_decisions(source, analysis, scenarios)
        self.assertTrue(all(row["disposition"] == "investigate" for row in records[1:]), records)
        self.assertTrue(all(row["structural_cost_tradeoff"]["authority"] == "conditional" for row in records[1:]))

    def test_budget_exhaustion_preserves_original_and_no_new_adopt(self):
        source, analysis, scenarios = produce()
        records = build_decisions(source, analysis, scenarios, seconds=0)
        self.assertEqual(records[0]["disposition"], "adopt")
        self.assertTrue(all(row["disposition"] == "investigate" for row in records[1:]))
        self.assertTrue(all(row["family_coverage"]["candidate_admission"] == "NOT_RUN" for row in records[1:]))

    def test_opaque_unchanged_use_does_not_block_unrelated_safe_coarsening(self):
        source = original()
        cb = source["callbacks"]["u_early"]
        cb.update(status="opaque", expression=None,
                  source="def u_early(assets, family):\n    return unknown_pure(assets, family)\n")
        cb["original"]["source"] = cb["source"]
        source, analysis, scenarios = produce(source)
        records = build_decisions(source, analysis, scenarios)
        changed = [row for row in records[1:] if row["changed_interfaces"]]
        self.assertTrue(changed)
        for row in changed:
            self.assertEqual(row["disposition"], "adopt", row)
            self.assertTrue(row["discharged_unchanged_obligations"])
        self.assertEqual(verify_decisions(source, analysis, scenarios, records)["status"], "PASS")

    def test_missing_capture_contract_stays_explicit(self):
        source, analysis, scenarios = produce()
        analysis["obligations"].append({"kind": "capture_identity", "detail": "mutable global unresolved"})
        records = build_decisions(source, analysis, scenarios)
        self.assertTrue(all(row["disposition"] == "investigate" for row in records[1:]), records)
        self.assertTrue(all(any(o["kind"] == "capture_identity" for o in row["residual_obligations"]) for row in records[1:]))

    def test_shared_context_matches_fresh_replay_and_rejects_mutations(self):
        source, analysis, scenarios = produce()
        context = prepare_decision_context(source, analysis)
        self.assertEqual(context.semantic["status"], "PASS")
        fresh = build_decisions(source, analysis, scenarios)
        shared = build_decisions(source, analysis, scenarios, _context=context)
        self.assertEqual(fresh, shared)
        external = context.semantic
        external["status"] = "FAIL"
        self.assertEqual(context.semantic["status"], "PASS")
        self.assertEqual(verify_decisions(source, analysis, scenarios, shared, _context=context)["status"], "PASS")
        source["cells"][0]["role"]["actor"] = "changed after checking"
        with self.assertRaisesRegex(ValueError, "unchanged source"):
            build_decisions(source, analysis, scenarios, _context=context)

    def test_serialized_pass_cannot_substitute_for_replay_context(self):
        source, analysis, scenarios = produce()
        with self.assertRaises(TypeError):
            build_decisions(source, analysis, scenarios, _context={"semantic": {"status": "PASS"}})

    def test_honest_budget_limited_investigation_can_be_strengthened(self):
        source, analysis, scenarios = produce()
        saved = build_decisions(source, analysis, scenarios, seconds=0)
        replay = verify_decisions(source, analysis, scenarios, saved)
        self.assertEqual(replay["status"], "PASS", replay)
        self.assertEqual(len(replay["strengthened"]), len(scenarios[0]["frontier"]))
        self.assertTrue(all(row["replayed_disposition"] == "adopt" for row in replay["strengthened"]))

    def test_known_stale_anchor_fails_even_when_replay_budget_is_zero(self):
        source, analysis, scenarios = produce()
        saved = build_decisions(source, analysis, scenarios, seconds=0)
        saved[0]["source_snapshot_sha256"] = "0" * 64
        replay = verify_decisions(source, analysis, scenarios, saved, seconds=0)
        self.assertEqual(replay["status"], "FAIL", replay)
        self.assertTrue(any("source_snapshot_sha256" in error for error in replay["errors"]))


if __name__ == "__main__":
    unittest.main()
