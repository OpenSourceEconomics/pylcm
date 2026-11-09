"""Independent legacy materialization and exhaustive oracle checks of compact scores."""
from copy import deepcopy
import unittest

from _lcm_regime_finder import find_regimes, verify_certificate
from _lcm_regime_finder import core
from _lcm_regime_finder.compact import SegmentIndex
from . import fixtures


class CompactTests(unittest.TestCase):
    def problems(self):
        return list(fixtures.named_cases().values()) + [fixtures.generated(seed) for seed in range(12)]

    def test_every_interval_matches_original_materialized_segment(self):
        for problem in self.problems():
            model = core.prepare(problem)
            for lane in model["lanes"].values():
                index = SegmentIndex(problem, model, lane)
                for i in range(len(lane)):
                    for j, cost in index.from_start(i):
                        original, _ = core.segment(problem, model, lane, i, j)
                        self.assertEqual(cost, original["cost"] if original else None)

    def test_optimum_layout_and_ties_match_legacy_search(self):
        for original in self.problems():
            expected = find_regimes(original)
            problem = deepcopy(original)
            problem.setdefault("implementation", {})["compact_search"] = True
            result = find_regimes(problem)
            self.assertEqual(result["status"], expected["status"])
            self.assertEqual(result["recommended"], expected["recommended"])
            checked = verify_certificate(problem, result)
            self.assertEqual(checked["status"], "PASS", checked)

    def test_budget_and_forged_lower_bound(self):
        problem = fixtures.generated(4)
        problem.setdefault("implementation", {})["compact_search"] = True
        incomplete = find_regimes(problem, {"candidate_segments": 0})
        self.assertEqual(incomplete["status"], "INCOMPLETE")
        self.assertFalse(verify_certificate(problem, incomplete)["optimum_verified"])
        complete = find_regimes(problem)
        forged = deepcopy(complete)
        first = next(iter(forged["certificate"]["potentials"].values()))
        first[0][0] += 1
        self.assertEqual(verify_certificate(problem, forged)["status"], "FAIL")
        forged = deepcopy(complete)
        forged["certificate"]["family_sha256"] = "0" * 64
        self.assertEqual(verify_certificate(problem, forged)["status"], "FAIL")

    def test_sparse_admission_masks_do_not_create_new_dates(self):
        problem = {"schema_version": 1, "name": "alternating original stages", "ages": [0, 1, 2, 3, 4],
            "states": {"x": [0, 1]}, "actions": {}, "implementation": {"allow_sparse_admission": True},
            "programs": [
                {"id": "A", "channel": "choice", "when": "age in [0,2,4]", "reward": "x", "discount": 1,
                 "transitions": [{"target": "B", "when": "age in [0,2]", "probability": 1, "updates": {"x": "x"}}]},
                {"id": "B", "channel": "consume", "when": "age in [1,3]", "reward": "x", "discount": 1,
                 "transitions": [{"target": "A", "probability": 1, "updates": {"x": "x"}}]}]}
        expected = find_regimes(problem)
        problem["implementation"]["compact_search"] = True
        result = find_regimes(problem)
        self.assertEqual(result["recommended"], expected["recommended"])
        self.assertEqual(result["recommended"]["cost"]["work"], 18)
        self.assertEqual(result["recommended"]["cost"]["blocks"], 2)
        self.assertEqual(sorted(age for regime in result["recommended"]["regimes"] for age in regime["ages"]), [0,1,2,3,4])
        self.assertEqual(verify_certificate(problem, result)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
