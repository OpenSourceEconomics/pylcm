"""Witness recovery for a checked symbolic cost-family certificate.

Tests use it to read one feasible representative back out of a checked Pareto
vector; no production path recovers witnesses.
"""
from collections import defaultdict

from _lcm_regime_finder.symbolic_frontier import Instance, ProofChecker, digest


def recover_witness(*, packet, certificate, cost, max_seconds=900, max_equations=50_000_000,
                    max_leaf_candidates=2_000_000, max_frontier_points=5000):
    """Recover and directly validate one feasible representative of a Pareto vector.

    Equal-cost layout ties use ascending threshold/cut choices here. This does not
    claim identity with the older UI's hashed-policy tie representative.
    """
    instance = Instance(packet=packet, chains=certificate["chains"])
    verifier = ProofChecker(instance=instance, certificate=certificate,
                            max_seconds=max_seconds, max_equations=max_equations,
                            max_leaf_candidates=max_leaf_candidates, max_frontier_points=max_frontier_points)
    verifier.check()
    target = tuple(cost)
    if target not in {tuple(v) for v in certificate["frontier"]}:
        raise ValueError("Requested cost is not in the checked Pareto frontier")
    contains = {}

    def has(node):
        if node not in contains:
            value = verifier.nodes[node]
            contains[node] = (target in value[1] if value[0] == "leaf"
                              else any(has(c) for c in value[2]))
        return contains[node]

    thresholds = [0] * len(instance.chains)
    node = certificate["root"]
    if not has(node):
        raise ValueError("Pareto vector lacks a symbolic policy witness")
    while verifier.nodes[node][0] != "leaf":
        _, level, children = verifier.nodes[node]
        value = next(k for k, child in enumerate(children) if has(child))
        thresholds[level], node = value, children[value]

    def values(root):
        while verifier.nodes[root][0] != "leaf":
            _, level, children = verifier.nodes[root]
            root = children[thresholds[level]]
        return verifier.nodes[root][1]

    component_roots = [row["root"] for row in certificate["components"]]
    impossible = set()

    def decompose(index, remaining):
        key = (index, remaining)
        if key in impossible:
            return None
        if index == len(component_roots):
            return [] if remaining == (0, 0, 0) else None
        for head in values(component_roots[index]):
            tail = tuple(remaining[i] - head[i] for i in range(3))
            if min(tail) < 0:
                continue
            result = decompose(index + 1, tail)
            if result is not None:
                return [head] + result
        impossible.add(key)
        return None

    component_costs = decompose(0, target)
    if component_costs is None:
        raise ValueError("No component-cost decomposition")
    interval_records = {(r["component"], r["mask"]): r["suffix"]
                        for r in certificate["intervals"]}
    blocks = []

    def interval_witness(ci, group, target_cost):
        component = instance.components[ci]
        dates = defaultdict(list)
        for i, label in enumerate(component):
            if group & (1 << i):
                for c in component[label]:
                    dates[instance.cells[c]["period"]].append(c)
        buckets = [sorted(v) for _, v in sorted(dates.items())]
        suffix, start = interval_records[ci, group], 0
        while start < len(buckets):
            ids = []
            for stop in range(start + 1, len(buckets) + 1):
                ids.extend(buckets[stop - 1])
                possible = instance.literal_block(ids=ids, thresholds=thresholds)
                if not possible:
                    continue
                local = possible[0]
                rest = tuple(target_cost[i] - local[i] for i in range(3))
                if rest in values(suffix[stop]):
                    blocks.append({"component": ci, "label_group_mask": group,
                                   "cells": sorted(ids), "cost": local})
                    start, target_cost = stop, rest
                    break
            else:
                raise ValueError("No date-cut witness for the checked cost")
        if target_cost != (0, 0, 0):
            raise ValueError("Date witness did not consume its cost")

    for ci, (component, target_cost) in enumerate(zip(instance.components, component_costs)):
        states = {int(k): v for k, v in certificate["components"][ci]["states"].items()}
        mask = (1 << len(component)) - 1
        while mask:
            first, remaining = mask & -mask, mask ^ (mask & -mask)
            subset, found = remaining, None
            while True:
                group = subset | first
                for head in values(interval_records[ci, group][0]):
                    tail = tuple(target_cost[i] - head[i] for i in range(3))
                    if tail in values(states[mask ^ group]):
                        found = (group, head, tail)
                        break
                if found is not None or subset == 0:
                    break
                subset = (subset - 1) & remaining
            if found is None:
                raise ValueError("No label-partition witness for the checked cost")
            group, head, target_cost = found
            interval_witness(ci, group, head)
            mask ^= group
        if target_cost != (0, 0, 0):
            raise ValueError("Grouping witness did not consume its cost")

    selected = {c for c, (j, p) in instance.position.items() if thresholds[j] <= p}
    if any(c in selected and t not in selected for c, t in instance.requires):
        raise ValueError("Recovered policy is not forward closed")
    covered = [c for block in blocks for c in block["cells"]]
    if len(covered) != len(set(covered)) or set(covered) != set(instance.cells):
        raise ValueError("Recovered blocks do not cover original cells exactly")
    total = tuple(sum(block["cost"][i] for block in blocks) for i in range(3))
    if total != target:
        raise ValueError("Recovered witness cost is wrong")
    for block in blocks:
        if instance.literal_block(ids=block["cells"], thresholds=thresholds) != (tuple(block["cost"]),):
            raise ValueError("Recovered block failed direct cost validation")
        block["option_by_cell"] = {
            c: (instance.coarsened[c] if c in selected else instance.baseline[c])["id"]
            for c in block["cells"]}
    return {"cost": target, "thresholds": thresholds, "blocks": blocks,
            "selected_cells": sorted(selected), "packet_sha256": digest(packet),
            "candidate_sha256": digest(instance.candidate),
            "synthesized_cost_variants_used": sorted(selected & set(instance.synthesized_cost_variants)),
            "status": "VALID_COST_FAMILY_WITNESS", "native_source_semantics": "NOT_ESTABLISHED"}
