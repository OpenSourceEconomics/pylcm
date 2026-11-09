"""Compare ordered-separator costs with independent finite enumeration."""
from copy import deepcopy
from itertools import product
import random
import unittest

from _lcm_regime_finder.separator_frontier import (
    Engine, METRICS, Problem, check, dominates, produce, validate_point,
)


def fixture(seed=0, *, gap=0, left_count=2, right_count=2):
    rng = random.Random(seed)
    domains = {"assets": {"size": 2}, "young": {"size": 3}, "old": {"size": 2},
               "extra": {"size": 2}, "choice": {"size": 2}}
    cells, interfaces, policy = [], {}, {}
    label_specs: list[tuple[str, tuple[int, ...], str | None, int]] = [(f"L{i}", (0, 1), "young", i) for i in range(left_count)]
    label_specs += [(f"R{i}", (2 + gap, 3 + gap), "old", i) for i in range(right_count)]
    label_specs += [("dead", tuple(range(4 + gap)), None, 0)]
    for label, dates, health, variant in label_specs:
        for date in dates:
            cid = f"{label}:{date}"
            states = {"assets": "assets"} | ({"health": health} if health else {})
            if health and variant % 2:
                states["extra"] = "extra"
            actions = {"choice": "choice"} if health and variant == 0 else {}
            cells.append({"id": cid, "regime": label, "period": date, "age": 50 + date,
                          "states": states, "actions": actions, "role": None, "ownership": None,
                          "stage": None, "terminal": health is None, "declaration_type": "regime",
                          "structural_roles": {}, "solver_identity": "same", "decision_objective": None})
            option = {"id": cid + "/identity", "states": states, "actions": actions,
                      "state_projection": {k: {"kind": "identity", "target": k} for k in states},
                      "action_reconstruction": {k: {"kind": "identity", "source": k} for k in actions}}
            interfaces[cid] = {"options": [option]}
            policy[cid] = option["id"]
    relations = []
    ids = [c["id"] for c in cells]
    for source in ids:
        for target in ids + ["external"]:
            if rng.randrange(4) == 0:
                relations.append({"source": source, "targets": [target],
                                  "support": rng.choice(("zero", "positive", "unknown"))})
    return {"declarations": {"analysis_scope": "pylcm-structural-graph/v1",
                "allocation": {"kind": "per_block", "boundary_unit": 1}, "domains": domains,
                "cells": cells, "relations": relations},
            "interfaces": interfaces, "discovery_complete": True,
            "projection_policies": [{"id": "identity", "option_by_cell": policy}]}


def partitions(labels):
    blocks = []
    def visit(i):
        if i == len(labels):
            yield [tuple(b) for b in blocks]
            return
        for block in blocks:
            block.append(labels[i])
            yield from visit(i + 1)
            block.pop()
        blocks.append([labels[i]])
        yield from visit(i + 1)
        blocks.pop()
    yield from visit(0)


def direct_cost(problem, ids):
    # Test oracle states the formula afresh and uses no search cost implementation.
    selected = set(ids)
    axes = []
    for kind in ("states", "actions"):
        present = {}
        for cell in ids:
            for key, domain in problem.options[cell][kind].items():
                present.setdefault(key, set()).add(domain)
        if any(len(v) != 1 for v in present.values()):
            return None
        rows = 1
        for values in present.values():
            domain = problem.domains[next(iter(values))]
            if domain.get("solve_axis") is not False:
                rows *= domain["size"]
        axes.append(rows)
    s, a = axes
    boundary = len({(x, y) for x, y in problem.pairs if x in selected and y not in selected})
    chars = set()
    for c in ids:
        chars.update(problem.chars[c])
    label_count = len({problem.cells[c]["regime"] for c in ids})
    w = problem.weights
    return ((len(ids) + problem.boundary * boundary) * s * a, len(ids) * s,
            w["block"] + w["dispatch"] * (label_count - 1) + w["characteristic"] * len(chars))


def group_by_literal_cuts(problem, labels):
    ids = [c for label in labels for c in problem.bylabel[label]]
    dates = sorted({problem.cells[c]["period"] for c in ids})
    buckets = [[c for c in ids if problem.cells[c]["period"] == date] for date in dates]
    vectors = []
    for cut_flags in product((False, True), repeat=len(dates) - 1):
        blocks, current = [], list(buckets[0])
        for cut, bucket in zip(cut_flags, buckets[1:]):
            if cut:
                blocks.append(current)
                current = []
            current += bucket
        blocks.append(current)
        costs = [direct_cost(problem, block) for block in blocks]
        if all(cost is not None for cost in costs):
            vectors.append(tuple(sum(cost[i] for cost in costs) for i in range(3)))
    return set(vectors)


def brute_family(problem):
    cache, vectors = {}, set()
    for partition in partitions(tuple(sorted(problem.bylabel))):
        offered = []
        for labels in partition:
            if labels not in cache:
                cache[labels] = group_by_literal_cuts(problem, labels)
            offered.append(cache[labels])
        for choices in product(*offered):
            vectors.add(tuple(sum(cost[i] for cost in choices) for i in range(3)))
    return sorted(v for v in vectors if not any(dominates(w, v) for w in vectors))


class OrderedSeparatorTests(unittest.TestCase):
    def costs(self, result):
        return [tuple(p["cost"][k] for k in METRICS) for p in result["frontier"]]

    def test_literal_global_groupings_all_cuts_arbitrary_directed_pairs(self):
        for seed in range(8):
            problem = Problem(fixture(seed))
            result = produce(problem)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(self.costs(result), brute_family(problem), seed)
            self.assertEqual(check(problem, result)["status"], "PASS")
            for point in result["frontier"]:
                validate_point(problem, point)

    def test_dead_only_gap_cuts_are_all_kept_in_the_family(self):
        problem = Problem(fixture(3, gap=3, left_count=1, right_count=1))
        self.assertEqual(problem.cuts, (2, 3, 4, 5))
        result = produce(problem)
        self.assertEqual(self.costs(result), brute_family(problem))
        self.assertEqual(check(problem, result)["status"], "PASS")

    def test_equal_cardinalities_are_still_incompatible_identities(self):
        analysis = fixture(4, left_count=1, right_count=1)
        analysis["declarations"]["domains"]["old"]["size"] = 3
        problem = Problem(analysis)
        self.assertNotEqual(problem.left_domain, problem.right_domain)
        self.assertEqual(self.costs(produce(problem)), brute_family(problem))

    def test_endpoint_count_table_matches_literal_directed_pair_costs(self):
        problem = Problem(fixture(5))
        checker = Engine(problem, checker=True)
        ids = sorted(problem.cells)
        dates = sorted({problem.cells[c]["period"] for c in ids})
        buckets = [[c for c in ids if problem.cells[c]["period"] == date] for date in dates]
        costs = checker._checker_table(buckets)
        for start in range(len(buckets)):
            for stop in range(start + 1, len(buckets) + 1):
                members = [c for bucket in buckets[start:stop] for c in bucket]
                self.assertEqual(costs.get((start, stop)), direct_cost(problem, members))

    def test_period_translation_and_input_permutation(self):
        analysis = fixture(6, left_count=1, right_count=1)
        baseline = self.costs(produce(Problem(analysis)))
        shifted = deepcopy(analysis)
        shifted["declarations"]["cells"].reverse()
        shifted["declarations"]["relations"].reverse()
        for c in shifted["declarations"]["cells"]:
            c["period"] += 37
        self.assertEqual(self.costs(produce(Problem(shifted))), baseline)

    def test_original_candidate_survives_budget_exhaustion(self):
        problem = Problem(fixture())
        result = produce(problem, seconds=0)
        self.assertEqual(result["status"], "INCOMPLETE")
        validate_point(problem, result["original_candidate"])
        self.assertEqual(check(problem, result)["status"], "PASS_PARTIAL")

    def test_guards_fail_closed(self):
        for mutation in ("allocation", "overlap", "bridge_axis", "policy_count", "missing_axis"):
            analysis = fixture()
            if mutation == "allocation":
                analysis["declarations"]["allocation"]["kind"] = "global_union"
            elif mutation == "overlap":
                next(c for c in analysis["declarations"]["cells"] if c["regime"] == "R0")["period"] = 0
            elif mutation == "bridge_axis":
                analysis["interfaces"]["dead:0"]["options"][0]["states"]["health"] = "young"
            elif mutation == "policy_count":
                analysis["projection_policies"] *= 2
            else:
                analysis["interfaces"]["L0:0"]["options"][0]["states"].pop("health")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Problem(analysis)

    def test_checker_rejects_cost_coverage_witness_and_binding_mutations(self):
        problem = Problem(fixture(1, left_count=1, right_count=1))
        original = produce(problem)
        for mutation in ("cost", "cuts", "trace", "coverage", "policy"):
            bad = deepcopy(original)
            if mutation == "cost":
                bad["frontier"][0]["cost"]["work"] += 1
            elif mutation == "cuts":
                bad["separator"]["cuts"] = ()
            elif mutation == "trace":
                bad["certificate"]["trace_sha256"] = "0" * 64
            elif mutation == "coverage":
                bad["frontier"] = []
            else:
                bad["policy_sha256"] = "0" * 64
            self.assertEqual(check(problem, bad)["status"], "FAIL", mutation)



if __name__ == "__main__":
    unittest.main()
