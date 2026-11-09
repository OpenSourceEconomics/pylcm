"""Exact symbolic cost-family acceptance and mutation tests."""
from copy import deepcopy
from itertools import product
import random
import unittest
from typing import TypedDict

from _lcm_regime_finder.symbolic_frontier import (Budget, Diagram, Exhausted, check, digest, dominates,
                             query, recover_witness, solve)


type AxisName = str


class Projection(TypedDict, total=False):
    kind: str
    callback: str
    target: AxisName
    source: AxisName
    domain: str
    values: list[int]


class LayoutOption(TypedDict):
    id: str
    states: dict[str, str]
    actions: dict[str, str]
    state_projection: dict[str, Projection]
    action_reconstruction: dict[str, Projection]
    constants: dict[str, int]


def toy(*, seed=0):
    rng = random.Random(seed)
    ids = ["a0", "a1", "b0", "b1"]
    chains = [["a0", "a1"], ["b0", "b1"]]
    candidate = {"id": "candidate", "callback": "callback", "state": "claim_age",
                 "domain": "raw", "target": "image", "target_domain": "image-domain",
                 "values": [0, 1, 2, 0, 1, 2], "target_values": [0, 1, 2],
                 "cells": ids, "requires": {"a0": ["a1", "b1"], "a1": [],
                                            "b0": ["b1"], "b1": []}}
    cells, interfaces = [], {}
    for cid in ids:
        axes = {"claim_age": "raw", "u" if cid[0] == "a" else "v": "u" if cid[0] == "a" else "v"}
        actions = {"choose": "action"}
        cell = {"id": cid, "regime": cid[0], "period": int(cid[1]),
                "age": 60 + int(cid[1]), "states": axes, "actions": actions,
                "solver_identity": "fixed", "role": None, "ownership": None,
                "stage": None, "terminal": False, "declaration_type": "native",
                "structural_roles": {}, "decision_objective": None}
        cells.append(cell)
        base: LayoutOption = {"id": "original:" + cid, "states": axes, "actions": actions,
                "state_projection": {a: {"kind": "identity", "target": a} for a in axes},
                "action_reconstruction": {"choose": {"kind": "identity", "source": "choose"}},
                "constants": {}}
        altered = deepcopy(base)
        altered["id"] = "offered-coarsened:" + cid
        del altered["states"]["claim_age"]
        altered["states"]["image"] = "image-domain"
        altered["state_projection"]["claim_age"] = {
            "kind": "characteristic", "callback": "callback", "target": "image",
            "domain": "image-domain", "values": candidate["values"]}
        # One absent local variant exercises reconstruction beyond offered policies.
        interfaces[cid] = {"options": [base] + ([] if cid == "b1" else [altered])}
    pairs = [(a, b) for a in ids for b in ids + ["external"] if rng.randrange(3) == 0]
    relations = [{"source": a, "targets": [b], "support": "positive"} for a, b in pairs]
    packet = {"schema": "exact-graph-cost-instance/v1", "citekey": "tiny-independent-oracle",
              "declarations": {"schema": "native", "analysis_scope": "pylcm-structural-graph/v1",
                  "allocation": {"kind": "per_block", "boundary_unit": seed % 3},
                  "cells": cells, "domains": {
                      "raw": {"size": 6, "kind": "categorical", "values": list(range(6))},
                      "u": {"size": 2}, "v": {"size": 3}, "action": {"size": 2}},
                  "relations": relations, "source_identity": "toy-source"},
              "derived_domains": {"image-domain": {"size": 3, "kind": "categorical",
                                                     "values": [0, 1, 2], "codes": [0, 1, 2]}},
              "interfaces": interfaces, "characteristic_candidates": [candidate],
              "discovery_complete": False,
              "clarity_weights": {"block": 2 + seed % 4, "dispatch": seed % 5,
                                  "characteristic": 1 + seed % 3},
              "policy_encoding": {"cells": ids, "basis": [0] * 4,
                                  "policies": [{"id": "identity-policy", "changes": []}]},
              "source_record": {"analysis_sha256": "toy-only-no-native-claim"}}
    packet["original_policy_selection_sha256"] = digest([
        ["identity-policy", {c: "original:" + c for c in ids}]])
    return packet, chains


def partitions(labels):
    """Literal restricted-growth set partitions, independent of the subset DP."""
    if not labels:
        yield []
        return
    first, *rest = labels
    for partition in partitions(rest):
        yield [[first]] + [list(x) for x in partition]
        for i in range(len(partition)):
            yield [list(x) + ([first] if j == i else []) for j, x in enumerate(partition)]


def literal_frontier(packet):
    cells = {c["id"]: c for c in packet["declarations"]["cells"]}
    ids = sorted(cells)
    candidate = packet["characteristic_candidates"][0]
    pairs = {(r["source"], t) for r in packet["declarations"]["relations"]
             if r.get("support") != "zero" for t in r["targets"]}
    domains = dict(packet["declarations"]["domains"]) | packet["derived_domains"]
    labels = sorted({c["regime"] for c in cells.values()})
    weights = packet["clarity_weights"]
    points = set()
    for bits in product((False, True), repeat=len(ids)):
        selected = {c for c, bit in zip(ids, bits) if bit}
        if any(c in selected and t not in selected for c, targets in candidate["requires"].items() for t in targets):
            continue
        for grouping in partitions(labels):
            choices = []
            for group in grouping:
                members = [c for c in ids if cells[c]["regime"] in group]
                dates = sorted({cells[c]["period"] for c in members})
                buckets = [[c for c in members if cells[c]["period"] == date] for date in dates]
                cuts = []
                for separators in product((False, True), repeat=max(0, len(buckets) - 1)):
                    blocks, block = [], []
                    for j, bucket in enumerate(buckets):
                        block.extend(bucket)
                        if j == len(buckets) - 1 or separators[j]:
                            blocks.append(block)
                            block = []
                    cuts.append(blocks)
                choices.append(cuts)
            for chosen in product(*choices):
                total, valid = [0, 0, 0], True
                for block in [block for blocks in chosen for block in blocks]:
                    unions = [{}, {}]
                    for c in block:
                        states = dict(packet["interfaces"][c]["options"][0]["states"])
                        if c in selected:
                            del states["claim_age"]
                            states["image"] = "image-domain"
                        for union, axes in zip(unions, (states, cells[c]["actions"])):
                            for axis, domain in axes.items():
                                if axis in union and union[axis] != domain:
                                    valid = False
                                union[axis] = domain
                    rows = []
                    for union in unions:
                        value = 1
                        for domain in union.values():
                            if domains[domain].get("solve_axis") is not False:
                                value *= domains[domain]["size"]
                        rows.append(value)
                    crossings = sum(a in block and b not in block for a, b in pairs)
                    total[0] += rows[0] * rows[1] * (len(block) +
                        packet["declarations"]["allocation"]["boundary_unit"] * crossings)
                    total[1] += len(block) * rows[0]
                    total[2] += weights["block"] + weights["dispatch"] * (
                        len({cells[c]["regime"] for c in block}) - 1)
                    total[2] += weights["characteristic"] * bool(set(block) & selected)
                if valid:
                    points.add(tuple(total))
    return {v for v in points if not any(u != v and all(u[i] <= v[i] for i in range(3)) for u in points)}


class SymbolicAcceptance(unittest.TestCase):
    def test_joint_frontier_against_literal_policies_groupings_and_all_cuts(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                packet, chains = toy(seed=seed)
                proof = solve(packet=packet, chains=chains, seconds=60, max_nodes=200000,
                              max_operations=2_000_000, max_frontier_points=1000)
                self.assertEqual(proof["status"], "COMPLETE_COST_FAMILY")
                self.assertEqual(check(packet=packet, certificate=proof)["status"], "PASS_COST_FAMILY")
                self.assertEqual({tuple(v) for v in proof["frontier"]}, literal_frontier(packet))
                self.assertEqual(proof["synthesized_cost_variants"], ["b1"])
                for cost in proof["frontier"]:
                    witness = recover_witness(packet=packet, certificate=proof, cost=cost)
                    self.assertEqual(tuple(witness["cost"]), tuple(cost))
                    self.assertEqual(sorted(c for b in witness["blocks"] for c in b["cells"]),
                                     ["a0", "a1", "b0", "b1"])

    def test_proof_omissions_input_changes_and_false_costs_rejected(self):
        packet, chains = toy(seed=2)
        proof = solve(packet=packet, chains=chains, seconds=60)
        mutations = []
        changed = deepcopy(proof); changed["intervals"].pop(); mutations.append((packet, changed))
        changed = deepcopy(proof); changed["frontier"] = []; mutations.append((packet, changed))
        changed = deepcopy(proof); changed["closure_root"] = changed["root"]; mutations.append((packet, changed))
        changed = deepcopy(proof); changed["source_semantics"] = "COMPLETE"; mutations.append((packet, changed))
        changed_packet = deepcopy(packet)
        changed_packet["declarations"]["domains"]["u"]["size"] += 1
        changed = deepcopy(proof); changed["packet_sha256"] = digest(changed_packet)
        mutations.append((changed_packet, changed))
        for value, cert in mutations:
            with self.assertRaises((ValueError, KeyError, IndexError)):
                check(packet=value, certificate=cert)

    def test_budget_exit_preserves_bound_valid_incumbent(self):
        packet, chains = toy()
        result = solve(packet=packet, chains=chains, max_operations=0)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["lower_bound"], [0, 0, 0])
        verdict = check(packet=packet, certificate=result)
        self.assertEqual(verdict["status"], "PASS_INCOMPLETE_RECORD")
        self.assertFalse(verdict["cost_frontier_complete"])
        changed = deepcopy(result)
        changed["incumbent"]["cost"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            check(packet=packet, certificate=changed)

    def test_leaf_product_budget_is_streamed_and_fails_explicitly(self):
        d = Diagram(lengths=[1], budget=Budget(seconds=60, nodes=1000,
                                               operations=10000, frontier_points=2))
        root = d.leaf(((0, 2, 0), (2, 0, 0)))
        with self.assertRaises(Exhausted):
            d.apply("sum", root, root)

    def test_components_must_share_the_same_policy_threshold(self):
        d = Diagram(lengths=[1], budget=Budget(seconds=60, nodes=1000,
                                               operations=10000, frontier_points=100))
        zero = d.leaf(((0, 0, 0),))
        ten = d.leaf(((10, 10, 0),))
        left = d.split(0, (zero, ten))
        right = d.split(0, (ten, zero))
        combined = d.apply("sum", left, right)
        self.assertEqual(d.frontier(combined), ((10, 10, 0),))
        self.assertNotIn((0, 0, 0), d.frontier(combined))

    def test_threshold_bijection_guards_reject_omissions_and_backward_arcs(self):
        packet, chains = toy()
        for changed in ([chains[0][:-1], chains[1]], [chains[0], chains[1] + ["a0"]]):
            with self.assertRaises(ValueError):
                solve(packet=packet, chains=changed)
        changed = deepcopy(packet)
        changed["characteristic_candidates"][0]["requires"]["a0"].remove("a1")
        with self.assertRaises(ValueError):
            solve(packet=changed, chains=chains)
        changed = deepcopy(packet)
        changed["characteristic_candidates"][0]["requires"]["a1"].append("a0")
        with self.assertRaises(ValueError):
            solve(packet=changed, chains=chains)

    def test_unsupported_point_remains_answerable(self):
        cert = {"frontier": [(0, 4, 3), (3, 2, 3), (4, 0, 3)]}
        self.assertEqual(query(certificate=cert, upper_bounds=(3, 2, 3)), [(3, 2, 3)])
        self.assertFalse(dominates(certificate=cert, vector=(3, 2, 3)))
        self.assertTrue(dominates(certificate=cert, vector=(3, 2, 3), strict=False))
        with self.assertRaises(ValueError):
            query(certificate=cert, weights=(-1, 1, 1), scales=(-1, 1, 1))


if __name__ == "__main__":
    unittest.main()
