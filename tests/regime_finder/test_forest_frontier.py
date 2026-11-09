"""Receiver-run decisive tests; NOT executed in the reviewer container."""
from copy import deepcopy
import random
import unittest

from _lcm_regime_finder.forest_frontier import digest, optimize
from _lcm_regime_finder.forest_check import materialize_verified_plan, query, verify


def example(layouts, *, pairs=(), sizes=None, actions=None, beta=1, chars=None):
    sizes = sizes or {name: 2 for layout in layouts for name in layout}
    actions = actions or [{} for _ in layouts]
    chars = chars or [[] for _ in layouts]
    return {"cells": [{"id": str(i), "label": "L" + str(i), "context": "same owner",
                       "states": {name: name for name in layout}, "actions": actions[i],
                       "characteristics": chars[i], "period": i, "age": 20 + i}
                      for i, layout in enumerate(layouts)],
            "domains": {name: {"size": size, "solve_axis": True} for name, size in sizes.items()},
            "pairs": [list(p) for p in pairs], "allocation": "per_block", "boundary_unit": beta,
            "clarity_weights": {"block": 4, "dispatch": 1, "characteristic": 2},
            "bindings": {"analysis_sha256": "tiny independent test", "policy_sha256": "P0"}}


def partitions(ids):
    if not ids:
        yield []
        return
    first, *rest = ids
    for parts in partitions(rest):
        yield [[first], *parts]
        for at in range(len(parts)):
            yield [([first, *block] if i == at else block[:]) for i, block in enumerate(parts)]


def literal(instance):
    """Independent original-cell partition oracle; no union-bin optimization."""
    by_id = {c["id"]: c for c in instance["cells"]}
    values = set()
    pairs = set(map(tuple, instance["pairs"]))
    for partition in partitions(sorted(by_id)):
        total = [0, 0, 0]
        good = True
        for members in partition:
            axes = ({}, {})
            characteristics = set()
            for v in members:
                characteristics.update(by_id[v]["characteristics"])
                for dest, kind in zip(axes, ("states", "actions")):
                    for name, domain in by_id[v][kind].items():
                        if name in dest and dest[name] != domain:
                            good = False
                        dest[name] = domain
            if not good:
                break
            rows = []
            for axis in axes:
                n = 1
                for d in axis.values():
                    if instance["domains"][d].get("solve_axis") is not False:
                        n *= instance["domains"][d]["size"]
                rows.append(n)
            inside = set(members)
            crossing = sum(s in inside and t not in inside for s, t in pairs)
            total[0] += (len(members) + instance["boundary_unit"] * crossing) * rows[0] * rows[1]
            total[1] += len(members) * rows[0]
            w = instance["clarity_weights"]
            total[2] += w["block"] + w["dispatch"] * (len(members) - 1) + w["characteristic"] * len(characteristics)
        if good:
            values.add(tuple(total))
    return sorted(v for v in values if not any(u != v and all(a <= b for a, b in zip(u, v)) for u in values))


class ExactForestTests(unittest.TestCase):
    def agreement(self, instance):
        result = optimize(instance, seconds=None)
        self.assertEqual(result["status"], "COMPLETE")
        receipt = verify(instance, result)
        self.assertEqual(receipt["status"], "PASS", receipt)
        self.assertEqual(receipt["vectors"], literal(instance))
        return result

    def test_unsupported_pareto_vector_survives(self):
        instance = example([("x", "y"), ("x", "z"), ("y", "z")])
        result = self.agreement(instance)
        # (20,20,9) is above the segment joining the other two points. It is
        # nondominated but cannot minimize any strictly positive linear score.
        self.assertEqual([p["cost"] for p in result["frontier"]],
                         [[12, 12, 12], [20, 20, 9], [24, 24, 6]])
        self.assertTrue(query(instance, result, operation="threshold", target=[20, 20, 9])["answer"])
        self.assertFalse(query(instance, result, operation="dominance", target=[20, 20, 9])["answer"])
        self.assertTrue(query(instance, result, operation="dominance", target=[21, 21, 9])["answer"])

    def test_directions_self_pairs_external_targets_and_duplicate_edges(self):
        instance = example([("x",), ("y",), ()],
                           pairs=[("0", "1"), ("1", "0"), ("1", "1"),
                                  ("1", "2"), ("2", "outside"), ("0", "1")],
                           actions=[{}, {"a": "a"}, {}], sizes={"x": 2, "y": 3, "a": 2},
                           chars=[["h"], [], ["h"]], beta=2)
        self.agreement(instance)

    def test_deterministic_random_forest_neighborhood(self):
        for seed in range(18):
            rng = random.Random(seed)
            n = 1 + seed % 5
            layouts = [tuple(x for x in ("x", "y", "z") if rng.randrange(2)) for _ in range(n)]
            pairs = []
            for v in range(1, n):
                parent = str(rng.randrange(v))
                if rng.randrange(2):
                    pairs.append((str(v), parent))
                else:
                    pairs.append((parent, str(v)))
                if rng.randrange(3) == 0:
                    pairs.extend([(parent, str(v)), (str(v), parent)])
            pairs += [("0", "0"), (str(n - 1), "outside")]
            instance = example(layouts, pairs=pairs, sizes={"x": 1, "y": 2, "z": 3}, beta=seed % 3)
            if seed % 2:
                instance["domains"]["x"]["solve_axis"] = False
            with self.subTest(seed=seed):
                self.agreement(instance)

    def test_provider_tokens_are_actual_original_members(self):
        instance = example([("x",), ("y",)])
        result = self.agreement(instance)
        for index in range(len(result["frontier"])):
            plan = materialize_verified_plan(instance, result, frontier_index=index)
            self.assertEqual(plan["status"], "PASS")
            self.assertEqual(sorted(v for block in plan["blocks"] for v in block), ["0", "1"])
            for block in plan["blocks"]:
                k = plan["assignment"][block[0]]
                actual = {name: domain for v in block for name, domain in instance["cells"][int(v)]["states"].items()}
                self.assertEqual(dict(result["bins"][k][0]), actual)
        for mutation in ("mask", "binding", "rule"):
            bad = deepcopy(result)
            if mutation == "mask":
                row = next(table["rows"][0] for table in bad["tables"] if table["operation"]["kind"] == "cell")
                row["used_bins"] = (1 << len(bad["bins"])) - 1
            elif mutation == "binding":
                bad["frontier"][0]["normalized_witness_sha256"] = "missing actual token provider"
            else:
                bad["normalization_rule"] = "trust an active-bin flag"
            self.assertEqual(verify(instance, bad)["status"], "FAIL", mutation)

    def test_missing_recurrence_branch_cannot_be_a_complete_certificate(self):
        instance = example([("x",), ("y",)], pairs=[("0", "1")])
        result = self.agreement(instance)
        for mutation in ("row", "cost", "witness", "root", "bound"):
            bad = deepcopy(result)
            if mutation == "row":
                bad["tables"][bad["root_table"]]["rows"].pop()
            elif mutation == "cost":
                bad["tables"][bad["root_table"]]["rows"][0]["points"][0]["cost"][0] += 1
            elif mutation == "witness":
                bad["frontier"][0]["source"] = [-1, 0]
            elif mutation == "root":
                bad["root_table"] = 0
            else:
                bad["lower_bound"] = [0, 0, 0]
            with self.subTest(mutation=mutation):
                self.assertEqual(verify(instance, bad)["status"], "FAIL")

    def test_retained_original_is_independently_checked(self):
        instance = example([("x",), ("y",)], pairs=[("0", "1"), ("1", "outside")])
        result = self.agreement(instance)
        self.assertTrue(verify(instance, result)["original_representation_verified"])
        for mutation in ("blocks", "cost"):
            changed = deepcopy(result)
            if mutation == "blocks":
                changed["original_representation"]["blocks"].pop()
            else:
                changed["original_representation"]["cost"][0] += 1
            self.assertEqual(verify(instance, changed)["status"], "FAIL", mutation)

    def test_pair_and_domain_rebinding_do_not_bypass_cost_reconstruction(self):
        instance = example([("x",), ("y",)], pairs=[("0", "1")])
        result = self.agreement(instance)
        for mutation in ("external", "reverse", "domain", "owner"):
            source, changed = deepcopy(instance), deepcopy(result)
            if mutation == "external":
                source["pairs"].append(["0", "outside"])
            elif mutation == "reverse":
                source["pairs"].append(["1", "0"])
            elif mutation == "domain":
                source["domains"]["x"]["size"] = 3
            else:
                source["cells"][1]["context"] = "another owner"
            changed["instance_sha256"] = digest(source)
            self.assertEqual(verify(source, changed)["status"], "FAIL", mutation)

    def test_repeated_small_tree_family_uses_shared_union_states(self):
        instance = example([("x",) if i % 2 else ("y",) for i in range(40)],
                           pairs=[(str(i), str(i + 1)) for i in range(0, 40, 2)])
        result = optimize(instance, seconds=None, max_states=20_000, max_points=20_000)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertLess(result["resource"]["candidate_transitions"], 1_000_000)
        self.assertEqual(verify(instance, result)["status"], "PASS")

    def test_subset_mask_cover_preserves_costs_and_has_its_own_schema(self):
        instance = example([('x',), ('y',), ()], pairs=[('0', '1'), ('1', '2')])
        result = self.agreement(instance)
        self.assertEqual(result['schema'], 'regime-singleton-forest-mask-frontier/v2')
        self.assertGreater(result['resource']['mask_dominated_points'], 0)
        old_meaning = deepcopy(result)
        old_meaning['schema'] = 'regime-singleton-forest-mask-frontier/v1'
        old_meaning.pop('state_cover_rule')
        self.assertEqual(verify(instance, old_meaning)['status'], 'FAIL')
        bad_rule = deepcopy(result)
        bad_rule['state_cover_rule'] = 'any cheap state, irrespective of mask or boundary'
        self.assertEqual(verify(instance, bad_rule)['status'], 'FAIL')

    def test_legacy_exact_mask_certificate_keeps_its_meaning(self):
        instance = example([('x',)])
        result = self.agreement(instance)
        self.assertEqual(result['resource']['mask_dominated_points'], 0)
        result['schema'] = 'regime-singleton-forest-mask-frontier/v1'
        result.pop('state_cover_rule')
        self.assertEqual(verify(instance, result)['status'], 'PASS')

    def test_exhaustion_is_never_certified_and_original_is_retained(self):
        instance = example([("x",), ("y",)])
        result = optimize(instance, seconds=None, max_states=1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["lower_bound"], [0, 0, 0])
        self.assertFalse(verify(instance, result)["frontier_complete_verified"])
        self.assertEqual(result["original_representation"]["blocks"], [["0"], ["1"]])

    def test_cycle_is_outside_this_reference_contract(self):
        instance = example([("x",), ("y",), ()],
                           pairs=[("0", "1"), ("1", "2"), ("2", "0")])
        with self.assertRaisesRegex(ValueError, "not a forest"):
            optimize(instance)


if __name__ == "__main__":
    unittest.main()
