"""Exact symbolic cost-family reference for the shipped Borella instance.

REVIEWER STATUS: authored and statically inspected; execution NOT_RUN.
No project modules, callbacks, solver, or numerical Model are imported.
The result concerns the finite cost family. It does not certify native rewrites.
"""
from __future__ import annotations

import argparse
from .records import BaselineBlock
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import chain
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_packet(*, path, reads=None):
    """Only JSON decoding and integrity checks; never import the fixture loader."""
    path = Path(path)
    header = reads.document(path) if reads is not None else json.loads(path.read_text())
    if header.get("schema") == "exact-graph-cost-instance/v1":
        return header
    if header.get("schema") != "interned-json-cost-instance/v1":
        raise ValueError("Unsupported cost-instance serialization")
    rows = header.get("records")
    if rows is None:
        rows = []
        for index, part in enumerate(header["record_parts"]):
            if Path(part["path"]).name != part["path"]:
                raise ValueError("Record part must be beside its header")
            target = (reads.child(path, 'record_parts/' + str(index), part['path'],
                                  base='supplied_parent', confined=True) if reads is not None
                      else path.parent / part['path'])
            raw = reads.read_bytes(target) if reads is not None else target.read_bytes()
            if hashlib.sha256(raw).hexdigest() != part["sha256"]:
                raise ValueError("Part digest mismatch")
            part_rows = json.loads(raw)['records']
            if len(part_rows) != part['records']:
                raise ValueError('Record-part length mismatch')
            rows.extend(part_rows)
    values = []

    def deref(value):
        if not isinstance(value, dict):
            return value
        if (set(value) != {"ref"} or type(value["ref"]) is not int
                or not 0 <= value["ref"] < len(values)):
            raise ValueError("Reference is not a completed earlier record")
        return values[value["ref"]]

    for row in rows:
        if set(row) == {"mapping"}:
            if len({k for k, _ in row["mapping"]}) != len(row["mapping"]):
                raise ValueError("Duplicate map key")
            values.append({k: deref(v) for k, v in row["mapping"]})
        elif set(row) == {"sequence"}:
            values.append([deref(v) for v in row["sequence"]])
        else:
            raise ValueError("Unknown interned record")
    packet = deref(header["root"])
    if digest(packet) != header["decoded_sha256"]:
        raise ValueError("Expanded packet digest mismatch")
    if packet.get("schema") != "exact-graph-cost-instance/v1":
        raise ValueError("Wrong expanded schema")
    return packet


def pareto(costs):
    """All nondominated vectors; no scalar weights or truncation."""
    result = []
    for value in sorted(set(tuple(v) for v in costs)):
        if not any(all(a <= b for a, b in zip(other, value)) for other in result):
            result.append(value)
    return tuple(result)


def reference_pareto(costs):
    """Separate all-pairs definition used by the certificate verifier."""
    values = set(tuple(v) for v in costs)
    return tuple(sorted(v for v in values if not any(
        u != v and all(u[i] <= v[i] for i in range(3)) for u in values)))


class Exhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, *, seconds, nodes, operations, frontier_points=5000):
        self.start = time.perf_counter()
        self.seconds, self.nodes, self.operations = seconds, nodes, operations
        self.frontier_points = frontier_points
        self.calls = 0

    def tick(self):
        self.calls += 1
        if self.calls > self.operations:
            raise Exhausted("symbolic operation budget")
        if time.perf_counter() - self.start > self.seconds:
            raise Exhausted("metadata time budget")


class Diagram:
    """Canonical ordered multi-valued decision diagram of Pareto antichains.

    A split at j has one child for each threshold 0,...,length[j].
    Leaves are tuple[tuple[int,int,int],...]. Empty is infeasible; {(0,0,0)}
    is the multiplicative unit and also encodes Boolean true for masks.
    """
    def __init__(self, *, lengths, budget):
        self.lengths = tuple(lengths)
        self.budget = budget
        self.nodes, self.unique, self.operations = [], {}, {}
        self.empty = self.leaf(())
        self.zero = self.leaf(((0, 0, 0),))

    def intern(self, value):
        if value not in self.unique:
            self.budget.tick()
            if len(self.nodes) >= self.budget.nodes:
                raise Exhausted("symbolic node budget")
            self.unique[value] = len(self.nodes)
            self.nodes.append(value)
        return self.unique[value]

    def leaf(self, values):
        return self.intern(("leaf", self.bounded_frontier(values)))

    def bounded_frontier(self, values):
        """Stream candidate products; never allocate an unbounded Cartesian tuple."""
        result = []
        for raw in values:
            self.budget.tick()
            value = tuple(raw)
            if any(all(p[i] <= value[i] for i in range(3)) for p in result):
                continue
            result = [p for p in result if not all(value[i] <= p[i] for i in range(3))]
            result.append(value)
            if len(result) > self.budget.frontier_points:
                raise Exhausted("symbolic Pareto-antichain budget")
        return tuple(sorted(result))

    def split(self, level, children):
        children = tuple(children)
        if len(children) != self.lengths[level] + 1:
            raise ValueError("Wrong threshold arity")
        if len(set(children)) == 1:
            return children[0]
        if any(self.level(c) <= level for c in children):
            raise ValueError("Diagram variables are not ordered")
        return self.intern(("split", level, children))

    def level(self, node):
        value = self.nodes[node]
        return len(self.lengths) if value[0] == "leaf" else value[1]

    def branch(self, node, level, value):
        item = self.nodes[node]
        return item[2][value] if self.level(node) == level else node

    def predicate(self, level, allowed):
        allowed = set(allowed)
        return self.split(level, [self.zero if k in allowed else self.empty
                                  for k in range(self.lengths[level] + 1)])

    def apply(self, operation, left, right):
        if operation not in ("union", "sum"):
            raise ValueError("Unknown Pareto operation")
        left, right = sorted((left, right))
        key = (operation, left, right)
        if key in self.operations:
            return self.operations[key]
        self.budget.tick()
        a, b = self.nodes[left], self.nodes[right]
        if a[0] == b[0] == "leaf":
            values = (chain(a[1], b[1]) if operation == "union" else
                      (tuple(x[i] + y[i] for i in range(3)) for x in a[1] for y in b[1]))
            answer = self.leaf(values)
        else:
            level = min(self.level(left), self.level(right))
            answer = self.split(level, [self.apply(
                operation, self.branch(left, level, k), self.branch(right, level, k))
                for k in range(self.lengths[level] + 1)])
        self.operations[key] = answer
        return answer

    def evaluate(self, node, thresholds):
        while self.nodes[node][0] != "leaf":
            _, level, children = self.nodes[node]
            node = children[thresholds[level]]
        return self.nodes[node][1]

    def frontier(self, root):
        def points():
            todo, seen = [root], set()
            while todo:
                node = todo.pop()
                if node in seen:
                    continue
                seen.add(node)
                item = self.nodes[node]
                if item[0] == "leaf":
                    yield from item[1]
                else:
                    todo.extend(item[2])
        return self.bounded_frontier(points())


def merge_axes(items):
    result = {}
    for item in items:
        for axis, domain in item.items():
            if axis in result and result[axis] != domain:
                return None
            result[axis] = domain
    return result


class Instance:
    """Cost-only compilation, with an explicit native semantic boundary."""
    def __init__(self, *, packet, chains):
        self.packet = packet
        self.decl = packet["declarations"]
        if self.decl.get("analysis_scope") != "pylcm-structural-graph/v1":
            raise ValueError("This seam requires the native structural cost scope")
        self.allocation = packet.get("allocation_scenario", self.decl["allocation"])
        if self.allocation.get("kind") != "per_block":
            raise ValueError("This seam requires per-block allocation")
        candidates = packet["characteristic_candidates"]
        if len(candidates) != 1:
            raise ValueError("This seam is for the supplied one-candidate family")
        self.candidate = candidates[0]
        self.eligible = set(self.candidate["cells"])
        self.chains = tuple(tuple(c) for c in chains)
        flat = [c for chain in self.chains for c in chain]
        if len(flat) != len(set(flat)) or set(flat) != self.eligible:
            raise ValueError("Chains must cover every eligible cell exactly once")
        self.position = {c: (j, k) for j, chain in enumerate(self.chains)
                         for k, c in enumerate(chain)}
        self.requires = {(c, t) for c, ts in self.candidate["requires"].items()
                         for t in ts}
        if any(c not in self.eligible or t not in self.eligible for c, t in self.requires):
            raise ValueError("An implication endpoint is outside eligibility")
        if any((u, v) not in self.requires for chain in self.chains
               for u, v in zip(chain, chain[1:])):
            raise ValueError("A proposed chain successor lacks an original implication")
        if any(self.position[c][0] == self.position[t][0]
               and self.position[c][1] >= self.position[t][1]
               for c, t in self.requires if c != t):
            raise ValueError("Backward chain arc needs SCC compression before this seam")
        self.cells = {c["id"]: c for c in self.decl["cells"]}
        self.domains = dict(self.decl["domains"]) | packet.get("derived_domains", {})
        self.axis = self.candidate["state"]
        self.target = self.candidate["target"]
        self.target_domain = self.candidate["target_domain"]
        original_domain = self.domains[self.candidate["domain"]]
        if len(self.candidate["values"]) != original_domain["size"]:
            raise ValueError("Incomplete characteristic value map")
        images = {canonical(v) for v in self.candidate["values"]}
        if images != {canonical(v) for v in self.candidate["target_values"]}:
            raise ValueError("Characteristic image differs from target values")
        if len(images) != len(self.candidate["target_values"]):
            raise ValueError("Repeated target value")
        if self.target_domain not in self.domains:
            raise ValueError("Missing bound target domain")
        target_domain = self.domains[self.target_domain]
        if (target_domain["size"] != len(images)
                or target_domain.get("values") != self.candidate["target_values"]):
            raise ValueError("Target domain does not match the complete image")
        enc = packet["policy_encoding"]
        if len(enc["cells"]) != len(enc["basis"]) or len(set(enc["cells"])) != len(enc["cells"]):
            raise ValueError("Malformed policy basis")
        self.baseline = {c: deepcopy(packet["interfaces"][c]["options"][index])
                         for c, index in zip(enc["cells"], enc["basis"])}
        if set(self.baseline) != set(self.cells):
            raise ValueError("Basis does not select every original cost cell")
        if any(v["kind"] == "characteristic" for o in self.baseline.values()
               for v in o["state_projection"].values()):
            raise ValueError("Basis must be the uncoarsened local projection")
        self.coarsened, self.synthesized_cost_variants = {}, []
        fields = ("states", "actions", "state_projection", "action_reconstruction", "constants")
        for c in self.eligible:
            base = self.baseline[c]
            if base["states"].get(self.axis) != self.candidate["domain"]:
                raise ValueError("Eligible cell lacks the original characteristic axis")
            if self.target in base["states"]:
                raise ValueError("Characteristic target collides with an existing axis")
            option = {key: deepcopy(base.get(key, {})) for key in fields}
            del option["states"][self.axis]
            option["state_projection"][self.axis] = {
                "kind": "characteristic", "callback": self.candidate["callback"],
                "target": self.target, "domain": self.target_domain,
                "values": self.candidate["values"]}
            if len(images) == 1:
                option["constants"][self.target] = self.candidate["target_values"][0]
            else:
                option["states"][self.target] = self.target_domain
            offered = packet["interfaces"][c]["options"]
            matched = [o for o in offered if all(o.get(k, {}) == option[k] for k in fields)]
            if not matched:
                self.synthesized_cost_variants.append(c)
            if any(not all(o.get(k, {}) == base.get(k, {}) for k in fields)
                   and not all(o.get(k, {}) == option[k] for k in fields) for o in offered):
                raise ValueError("An offered interface is outside this binary local family")
            # Preserve every offered identity. A missing variant has a deliberately
            # distinct cost-only namespace; it is not a forged native interface id.
            option["id"] = (matched[0]["id"] if matched else
                            "cost-interface:" + digest([digest(self.candidate), c, option]))
            self.coarsened[c] = option
        for c in set(self.cells) - self.eligible:
            if any(any(o.get(k, {}) != self.baseline[c].get(k, {}) for k in fields)
                   for o in packet["interfaces"][c]["options"]):
                raise ValueError("Unmodelled noncandidate local choice")
        # Bind and qualify the offered subset without treating it as the whole family.
        selections = []
        for row in enc["policies"]:
            indices = list(enc["basis"])
            for index, value in row["changes"]:
                indices[index] = value
            chosen = {c: packet["interfaces"][c]["options"][indices[i]]
                      for i, c in enumerate(enc["cells"])}
            selected = {c for c in self.eligible
                        if chosen[c]["state_projection"][self.axis]["kind"] == "characteristic"}
            if any(c in selected and t not in selected for c, t in self.requires):
                raise ValueError("Offered policy violates an original closure implication")
            selections.append([row["id"], {c: o["id"] for c, o in chosen.items()}])
        if digest(selections) != packet["original_policy_selection_sha256"]:
            raise ValueError("Offered policy-selection digest mismatch")
        self.rules = {"block": 4, "dispatch": 1, "characteristic": 2} | packet.get("clarity_weights", {})
        self.boundary = self.allocation.get("boundary_unit", 1)
        if (any(type(v) is not int or v < 0 for v in self.rules.values())
                or self.rules["block"] == 0 or type(self.boundary) is not int
                or self.boundary < 0):
            raise ValueError("Invalid cost coefficients")
        self.pairs = {(r["source"], t) for r in self.decl["relations"]
                      if r.get("support") != "zero" for t in r.get("targets", [])}
        self.roles = {c: self.role(cell) for c, cell in self.cells.items()}
        groups = defaultdict(lambda: defaultdict(list))
        for c, cell in self.cells.items():
            groups[self.roles[c]][cell["regime"]].append(c)
        self.components = [{k: tuple(sorted(v[k])) for k in sorted(v)}
                           for _, v in sorted(groups.items())]

    @staticmethod
    def role(cell):
        role = cell.get("role")
        if role is None:
            role = {"original_contexts_preserved_by_dispatch": True,
                    "decision_objective": cell.get("decision_objective")}
        return canonical([role, cell.get("ownership"), cell.get("stage"), None,
                          cell.get("declaration_type"), cell.get("structural_roles", {}),
                          cell.get("solver_identity")])

    def rows(self, axes):
        product = 1
        for domain in axes.values():
            record = self.domains[domain]
            if record.get("solve_axis") is False:
                continue
            if type(record.get("size")) is not int or record["size"] <= 0:
                raise ValueError("Missing positive integer solve-axis cardinality")
            product *= record["size"]
        return product

    def cost_from_unions(self, ids, states, actions, characteristic_count):
        if states is None or actions is None or len({self.roles[c] for c in ids}) != 1:
            return ()
        ns, na = self.rows(states), self.rows(actions)
        members = set(ids)
        crossings = sum(c in members and t not in members for c, t in self.pairs)
        labels = len({self.cells[c]["regime"] for c in ids})
        return ((ns * na * (len(ids) + self.boundary * crossings),
                 ns * len(ids), self.rules["block"] + self.rules["dispatch"] * (labels - 1)
                 + self.rules["characteristic"] * characteristic_count),)

    def literal_block(self, *, ids, thresholds):
        """Checker/oracle cost: union the actual chosen local records cell by cell."""
        options = []
        for c in ids:
            selected = c in self.position and thresholds[self.position[c][0]] <= self.position[c][1]
            options.append(self.coarsened[c] if selected else self.baseline[c])
        states = merge_axes(o["states"] for o in options)
        actions = merge_axes(o["actions"] for o in options)
        chars = {canonical(v) for o in options for v in o["state_projection"].values()
                 if v["kind"] == "characteristic"}
        return self.cost_from_unions(ids, states, actions, len(chars))

    def symbolic_block(self, *, ids, diagram):
        """Producer cost: two presence predicates, not a policy-product loop."""
        fixed = []
        raw, projected = diagram.empty, diagram.empty
        for c in ids:
            states = dict(self.baseline[c]["states"])
            if c in self.position:
                del states[self.axis]
                j, p = self.position[c]
                selected = diagram.predicate(j, range(p + 1))
                unselected = diagram.predicate(j, range(p + 1, len(self.chains[j]) + 1))
                raw = diagram.apply("union", raw, unselected)
                projected = diagram.apply("union", projected, selected)
            fixed.append(states)
        axes = merge_axes(fixed)
        actions = merge_axes(self.baseline[c]["actions"] for c in ids)
        if axes is None or actions is None:
            return diagram.empty

        def complement(root):
            item = diagram.nodes[root]
            if item[0] == "leaf":
                if item[1] not in ((), ((0, 0, 0),)):
                    raise ValueError("Not a Boolean mask")
                return diagram.zero if not item[1] else diagram.empty
            return diagram.split(item[1], [complement(c) for c in item[2]])

        no_raw, no_projected = complement(raw), complement(projected)
        result = diagram.empty
        for use_raw, use_projected in ((0, 0), (0, 1), (1, 0), (1, 1)):
            state_parts = [axes]
            if use_raw:
                state_parts.append({self.axis: self.candidate["domain"]})
            if use_projected and len(self.candidate["target_values"]) > 1:
                state_parts.append({self.target: self.target_domain})
            cost = self.cost_from_unions(ids, merge_axes(state_parts), actions, use_projected)
            mask = diagram.apply("sum", raw if use_raw else no_raw,
                                 projected if use_projected else no_projected)
            term = diagram.apply("sum", mask, diagram.leaf(cost))
            result = diagram.apply("union", result, term)
        return result


def threshold_mask(*, instance, diagram):
    result = diagram.zero
    for c, t in sorted(instance.requires):
        i, p = instance.position[c]
        j, q = instance.position[t]
        not_selected = diagram.predicate(i, range(p + 1, len(instance.chains[i]) + 1))
        selected_target = diagram.predicate(j, range(q + 1))
        implication = diagram.apply("union", not_selected, selected_target)
        result = diagram.apply("sum", result, implication)
    return result


def baseline_incumbent(*, instance):
    """Keep a feasible uncoarsened cost plan before any full-family search.

    Use original-label blocks; if their domains vary incompatibly over dates,
    use one block per original date bucket within that same singleton label.
    The original declaration/option packet remains the authoritative input.
    """
    thresholds = [len(c) for c in instance.chains]
    blocks: list[BaselineBlock] = []
    for ci, component in enumerate(instance.components):
        for index, (_, members) in enumerate(component.items()):
            groups = [members]
            if not instance.literal_block(ids=members, thresholds=thresholds):
                dates = defaultdict(list)
                for c in members:
                    dates[instance.cells[c]["period"]].append(c)
                groups = [tuple(sorted(v)) for _, v in sorted(dates.items())]
            for ids in groups:
                costs = instance.literal_block(ids=ids, thresholds=thresholds)
                if not costs:
                    raise ValueError("No valid uncoarsened block even within one label/date bucket")
                blocks.append({"component": ci, "label_group_mask": 1 << index,
                               "cells": sorted(ids), "cost": costs[0],
                               "option_by_cell": {c: instance.baseline[c]["id"] for c in ids}})
    return {"kind": "uncoarsened offered cost policy with original-label grouping",
            "thresholds": thresholds, "blocks": blocks,
            "cost": tuple(sum(row["cost"][i] for row in blocks) for i in range(3)),
            "native_source_semantics": "NOT_ESTABLISHED_BY_COST_PACKET"}


def check_incumbent(*, instance, incumbent):
    thresholds = tuple(incumbent["thresholds"])
    if thresholds != tuple(map(len, instance.chains)):
        raise ValueError("Retained baseline must be uncoarsened")
    assigned, total = [], [0, 0, 0]
    for block in incumbent["blocks"]:
        component = instance.components[block["component"]]
        mask = block["label_group_mask"]
        if type(mask) is not int or mask <= 0 or mask & (mask - 1):
            raise ValueError("Baseline block is not an original singleton label group")
        index = mask.bit_length() - 1
        if index >= len(component):
            raise ValueError("Baseline label index is outside its component")
        members = component[tuple(component)[index]]
        ids = tuple(block["cells"])
        if ids != tuple(sorted(set(ids))) or not ids or not set(ids) <= set(members):
            raise ValueError("Invalid baseline membership")
        dates = sorted({instance.cells[c]["period"] for c in members})
        chosen = sorted({instance.cells[c]["period"] for c in ids})
        left, right = dates.index(chosen[0]), dates.index(chosen[-1])
        expected = {c for c in members if instance.cells[c]["period"] in dates[left:right + 1]}
        if set(ids) != expected:
            raise ValueError("Baseline block is not a whole date-bucket interval")
        if block["option_by_cell"] != {c: instance.baseline[c]["id"] for c in ids}:
            raise ValueError("Baseline changed a selected original cost interface")
        actual = instance.literal_block(ids=ids, thresholds=thresholds)
        if actual != (tuple(block["cost"]),):
            raise ValueError("Retained baseline cost is false")
        for i in range(3):
            total[i] += block["cost"][i]
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(instance.cells):
        raise ValueError("Retained baseline does not cover every original cell once")
    if tuple(total) != tuple(incumbent["cost"]):
        raise ValueError("Retained baseline total is false")


def solve(*, packet, chains, seconds=900, max_nodes=1_000_000,
          max_operations=50_000_000, max_frontier_points=5000):
    """Pointwise interval/subset DP, followed by one existential Pareto union.

    This is the implemented optimizer, not an external-solver/model-building seam.
    A budget exit is explicitly INCOMPLETE and yields no completeness certificate.
    """
    budget = Budget(seconds=seconds, nodes=max_nodes, operations=max_operations,
                    frontier_points=max_frontier_points)
    instance = Instance(packet=packet, chains=chains)
    incumbent = baseline_incumbent(instance=instance)
    d = None
    interval_records, component_records, block_roots = {}, [], {}

    def block(ids):
        if d is None:
            raise ValueError("Symbolic diagram is not initialized")
        key = tuple(sorted(ids))
        if key not in block_roots:
            block_roots[key] = instance.symbolic_block(ids=key, diagram=d)
        return block_roots[key]

    def interval(ci, mask):
        if d is None:
            raise ValueError("Symbolic diagram is not initialized")
        key = (ci, mask)
        if key in interval_records:
            return interval_records[key][0]
        component = instance.components[ci]
        labels = tuple(component)
        dates = defaultdict(list)
        for i, label in enumerate(labels):
            if mask & (1 << i):
                for c in component[label]:
                    dates[instance.cells[c]["period"]].append(c)
        buckets = [tuple(sorted(v)) for _, v in sorted(dates.items())]
        suffix = [d.empty] * len(buckets) + [d.zero]
        for start in range(len(buckets) - 1, -1, -1):
            ids, accumulator = [], d.empty
            for stop in range(start + 1, len(buckets) + 1):
                budget.tick()
                ids.extend(buckets[stop - 1])
                term = d.apply("sum", block(ids), suffix[stop])
                accumulator = d.apply("union", accumulator, term)
            suffix[start] = accumulator
        interval_records[key] = suffix
        return suffix[0]

    try:
        budget.tick()
        d = Diagram(lengths=list(map(len, chains)), budget=budget)
        combined = d.zero
        for ci, component in enumerate(instance.components):
            labels = tuple(component)
            states = {0: d.zero}

            def visit(mask):
                if mask in states:
                    return states[mask]
                pivot, accumulator = mask & -mask, d.empty
                remaining = mask ^ pivot
                sub = remaining
                while True:
                    budget.tick()
                    group = sub | pivot
                    term = d.apply("sum", interval(ci, group), visit(mask ^ group))
                    accumulator = d.apply("union", accumulator, term)
                    if sub == 0:
                        break
                    sub = (sub - 1) & remaining
                states[mask] = accumulator
                return accumulator

            root = visit((1 << len(labels)) - 1)
            component_records.append({"states": {str(k): v for k, v in sorted(states.items())},
                                      "root": root})
            combined = d.apply("sum", combined, root)
        closure = threshold_mask(instance=instance, diagram=d)
        root = d.apply("sum", combined, closure)
        frontier = d.frontier(root)
    except Exhausted as exc:
        return {"schema": "borella-symbolic-pareto-proof/v1", "status": "INCOMPLETE",
                "reason": str(exc), "cost_frontier_complete": False,
                "packet_sha256": digest(packet), "candidate_sha256": digest(instance.candidate),
                "chains": [list(c) for c in chains], "chains_sha256": digest(chains),
                "source_record": packet.get("source_record"), "incumbent": incumbent,
                "lower_bound": [0, 0, 0],
                "source_semantics": "NOT_ESTABLISHED_BY_COST_PACKET",
                "node_count": len(d.nodes) if d is not None else 0, "symbolic_calls": budget.calls,
                "elapsed_seconds": time.perf_counter() - budget.start,
                "scope": "Valid uncoarsened cost witness retained; full original packet unchanged; no complete full-family result established"}
    return {
        "schema": "borella-symbolic-pareto-proof/v1", "status": "COMPLETE_COST_FAMILY",
        "packet_sha256": digest(packet), "candidate_sha256": digest(instance.candidate),
        "chains": [list(c) for c in chains], "lengths": list(map(len, chains)),
        "chains_sha256": digest(chains), "source_record": packet.get("source_record"),
        "incumbent": incumbent, "lower_bound": [0, 0, 0],
        "nodes": d.nodes,
        "operations": [[op, a, b, answer] for (op, a, b), answer in sorted(d.operations.items())],
        "blocks": [{"cells": list(ids), "root": r} for ids, r in sorted(block_roots.items())],
        "intervals": [{"component": ci, "mask": mask, "suffix": suffix}
                      for (ci, mask), suffix in sorted(interval_records.items())],
        "components": component_records, "closure_root": closure, "root": root,
        "frontier": frontier, "elapsed_seconds": time.perf_counter() - budget.start,
        "symbolic_calls": budget.calls,
        "source_semantics": "NOT_ESTABLISHED_BY_COST_PACKET",
        "synthesized_cost_variants": sorted(instance.synthesized_cost_variants),
        "query_contract": "All integer Pareto vectors; exact threshold/dominance and positive-weight recommendation queries. Equal-cost policy/layout representative is selected separately; no claim of the old display tie representative.",
    }


class ProofChecker:
    """Independent equation replay; does not call producer Diagram.apply.

    Scalar costs use selected original local records. Each algebraic equation is
    checked by simultaneous descent through its three diagrams, with an all-pairs
    Pareto definition at leaves. Family coverage is reconstructed from input.
    """
    def __init__(self, *, instance, certificate, max_leaf_candidates=2_000_000,
                 max_equations=50_000_000, max_frontier_points=5000,
                 max_seconds=900):
        self.instance, self.certificate = instance, certificate
        self.max_leaf_candidates, self.max_equations = max_leaf_candidates, max_equations
        self.max_frontier_points, self.max_seconds = max_frontier_points, max_seconds
        self.started, self.steps = time.perf_counter(), 0
        self.nodes, self.lookup = [], {}
        self.lengths = tuple(map(len, instance.chains))
        for raw in certificate["nodes"]:
            self.guard()
            if raw[0] == "leaf" and len(raw) == 2:
                values = tuple(tuple(v) for v in raw[1])
                if len(values) > self.max_frontier_points:
                    raise Exhausted("independent leaf-antichain budget")
                if any(len(v) != 3 or any(type(x) is not int or x < 0 for x in v) for v in values):
                    raise ValueError("Invalid cost leaf")
                if self.exact_frontier(values) != values:
                    raise ValueError("Leaf is not an exact canonical antichain")
                node = ("leaf", values)
            elif raw[0] == "split" and len(raw) == 3:
                level, children = raw[1], tuple(raw[2])
                if type(level) is not int or not 0 <= level < len(self.lengths):
                    raise ValueError("Invalid split level")
                if len(children) != self.lengths[level] + 1 or len(set(children)) == 1:
                    raise ValueError("Invalid or unreduced split")
                if any(type(c) is not int or not 0 <= c < len(self.nodes)
                       or self.level(c) <= level for c in children):
                    raise ValueError("Unordered/forward diagram reference")
                node = ("split", level, children)
            else:
                raise ValueError("Unknown diagram node")
            if node in self.lookup:
                raise ValueError("Duplicate canonical node")
            self.lookup[node] = len(self.nodes)
            self.nodes.append(node)
        self.empty = self.lookup[("leaf", ())]
        self.zero = self.lookup[("leaf", ((0, 0, 0),))]
        self.ops = {}
        for op, left, right, answer in certificate["operations"]:
            self.guard()
            if op not in ("sum", "union") or left > right:
                raise ValueError("Invalid algebra record")
            if any(type(i) is not int or not 0 <= i < len(self.nodes) for i in (left, right, answer)):
                raise ValueError("Unknown algebra operand")
            key = (op, left, right)
            if key in self.ops:
                raise ValueError("Duplicate algebra record")
            self.ops[key] = answer
        self.checked = set()

    def guard(self):
        self.steps += 1
        if self.steps > self.max_equations:
            raise Exhausted("independent proof-step budget")
        if time.perf_counter() - self.started > self.max_seconds:
            raise Exhausted("independent metadata-time budget")

    def exact_frontier(self, values):
        """Bounded all-pairs definition, distinct from the producer's streaming prune."""
        values = set(tuple(v) for v in values)
        result = []
        for value in values:
            dominated = False
            for other in values:
                self.guard()
                if other != value and all(other[i] <= value[i] for i in range(3)):
                    dominated = True
                    break
            if not dominated:
                result.append(value)
        return tuple(sorted(result))

    def level(self, node):
        return len(self.lengths) if self.nodes[node][0] == "leaf" else self.nodes[node][1]

    def branch(self, node, level, value):
        return self.nodes[node][2][value] if self.level(node) == level else node

    def equation(self, op, left, right, answer):
        key = (op, left, right, answer)
        if key in self.checked:
            return
        self.guard()
        if len(self.checked) >= self.max_equations:
            raise Exhausted("independent symbolic-equation budget")
        level = min(self.level(left), self.level(right), self.level(answer))
        if level == len(self.lengths):
            a, b, actual = (self.nodes[i][1] for i in (left, right, answer))
            candidates = len(a) + len(b) if op == "union" else len(a) * len(b)
            if candidates > self.max_leaf_candidates:
                raise Exhausted("independent leaf-candidate budget")
            expected = self.exact_frontier(a + b if op == "union" else
                [tuple(x[j] + y[j] for j in range(3)) for x in a for y in b])
            if actual != expected:
                raise ValueError("False pointwise Pareto equation")
        else:
            for k in range(self.lengths[level] + 1):
                self.equation(op, self.branch(left, level, k), self.branch(right, level, k),
                              self.branch(answer, level, k))
        self.checked.add(key)

    def apply(self, op, left, right):
        left, right = sorted((left, right))
        answer = self.ops[(op, left, right)]
        self.equation(op, left, right, answer)
        return answer

    def predicate(self, level, allowed):
        allowed = set(allowed)
        children = tuple(self.zero if k in allowed else self.empty
                         for k in range(self.lengths[level] + 1))
        if len(set(children)) == 1:
            return children[0]
        return self.lookup[("split", level, children)]

    def check_block(self, *, ids, root):
        """At most three presence patterns per occupied chain, not all policies."""
        positions = defaultdict(list)
        for c in ids:
            if c in self.instance.position:
                j, k = self.instance.position[c]
                positions[j].append(k)
        thresholds = [0] * len(self.lengths)
        memo = set()

        def visit(level, node, modes):
            self.guard()
            key = (level, node, modes)
            if key in memo:
                return
            if level == len(self.lengths):
                expected = self.instance.literal_block(ids=ids, thresholds=thresholds)
                if self.nodes[node] != ("leaf", expected):
                    raise ValueError("Block cost disagrees with original selected interfaces/pairs")
            else:
                ps = positions.get(level, [])
                cuts = {0, self.lengths[level] + 1}
                if ps:
                    cuts.update((min(ps) + 1, max(ps) + 1))
                cuts = sorted(cuts)
                for index, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
                    thresholds[level] = lo
                    children = {self.branch(node, level, k) for k in range(lo, hi)}
                    for child in children:
                        visit(level + 1, child, modes + (index,))
            memo.add(key)

        visit(0, root, ())

    def check(self):
        cert, instance = self.certificate, self.instance
        if cert.get("schema") != "borella-symbolic-pareto-proof/v1" or cert.get("status") != "COMPLETE_COST_FAMILY":
            raise ValueError("No complete symbolic proof supplied")
        if cert["packet_sha256"] != digest(instance.packet) or cert["candidate_sha256"] != digest(instance.candidate):
            raise ValueError("Input identity mismatch")
        if cert.get("chains_sha256") != digest(cert["chains"]) or cert.get("source_record") != instance.packet.get("source_record"):
            raise ValueError("Chain/source binding changed")
        if cert.get("lower_bound") != [0, 0, 0]:
            raise ValueError("Only the universal nonnegative lower bound is asserted here")
        check_incumbent(instance=instance, incumbent=cert["incumbent"])
        if (cert.get("source_semantics") != "NOT_ESTABLISHED_BY_COST_PACKET"
                or cert.get("synthesized_cost_variants") != sorted(instance.synthesized_cost_variants)):
            raise ValueError("Source-semantic boundary was altered")
        if tuple(cert["lengths"]) != self.lengths:
            raise ValueError("Threshold-domain mismatch")
        blocks = {}
        for row in cert["blocks"]:
            self.guard()
            ids = tuple(row["cells"])
            if ids != tuple(sorted(set(ids))) or not ids or any(c not in instance.cells for c in ids):
                raise ValueError("Invalid block membership")
            if ids in blocks:
                raise ValueError("Duplicate block record")
            blocks[ids] = row["root"]
            self.check_block(ids=ids, root=row["root"])
        intervals = {}
        for row in cert["intervals"]:
            key = (row["component"], row["mask"])
            if key in intervals:
                raise ValueError("Duplicate interval record")
            intervals[key] = row["suffix"]
        if len(cert["components"]) != len(instance.components):
            raise ValueError("Missing/extra protected component")
        used_intervals = set()
        combined = self.zero
        for ci, component in enumerate(instance.components):
            labels = tuple(component)
            full = (1 << len(labels)) - 1
            states = {int(k): v for k, v in cert["components"][ci]["states"].items()}
            expected_masks = {0, full} | set(range(0, full + 1, 2))
            if set(states) != expected_masks or states[0] != self.zero:
                raise ValueError("Subset recurrence state coverage changed")

            def interval(group):
                key = (ci, group)
                suffix = intervals[key]
                if key in used_intervals:
                    return suffix[0]
                dates = defaultdict(list)
                for i, label in enumerate(labels):
                    if group & (1 << i):
                        for c in component[label]:
                            dates[instance.cells[c]["period"]].append(c)
                buckets = [sorted(v) for _, v in sorted(dates.items())]
                if len(suffix) != len(buckets) + 1 or suffix[-1] != self.zero:
                    raise ValueError("Date suffix coverage changed")
                for start in range(len(buckets)):
                    accumulator, ids = self.empty, []
                    for stop in range(start + 1, len(buckets) + 1):
                        self.guard()
                        ids.extend(buckets[stop - 1])
                        term = self.apply("sum", blocks[tuple(sorted(ids))], suffix[stop])
                        accumulator = self.apply("union", accumulator, term)
                    if accumulator != suffix[start]:
                        raise ValueError("A cut or interval cost is missing")
                used_intervals.add(key)
                return suffix[0]

            for mask in sorted(expected_masks - {0}):
                first = next(i for i in range(len(labels)) if mask & (1 << i))
                others = [i for i in range(first + 1, len(labels)) if mask & (1 << i)]
                accumulator = self.empty
                # Reverse binary enumeration reproduces the declared canonical union
                # chain; this verifies coverage of every subset, not a trusted count.
                for binary in range((1 << len(others)) - 1, -1, -1):
                    self.guard()
                    group = (1 << first) | sum(1 << i for j, i in enumerate(others) if binary & (1 << j))
                    term = self.apply("sum", interval(group), states[mask ^ group])
                    accumulator = self.apply("union", accumulator, term)
                if accumulator != states[mask]:
                    raise ValueError("A label-group branch is missing")
            if cert["components"][ci]["root"] != states[full]:
                raise ValueError("Wrong component root")
            combined = self.apply("sum", combined, states[full])
        if set(intervals) != used_intervals:
            raise ValueError("Unbound/extra interval-family records")
        closure = self.zero
        for c, t in sorted(instance.requires):
            i, p = instance.position[c]
            j, q = instance.position[t]
            a = self.predicate(i, range(p + 1, self.lengths[i] + 1))
            b = self.predicate(j, range(q + 1))
            closure = self.apply("sum", closure, self.apply("union", a, b))
        if closure != cert["closure_root"] or self.apply("sum", combined, closure) != cert["root"]:
            raise ValueError("Full-policy consistency or final root changed")
        claimed = tuple(tuple(v) for v in cert["frontier"])
        if len(claimed) > self.max_frontier_points:
            raise Exhausted("independent final-frontier budget")
        if self.exact_frontier(claimed) != claimed:
            raise ValueError("Displayed frontier is not a canonical antichain")
        todo, seen, witnessed = [cert["root"]], set(), set()
        while todo:
            self.guard()
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            if self.nodes[node][0] == "leaf":
                for value in self.nodes[node][1]:
                    self.guard()
                    if value in claimed:
                        witnessed.add(value)
                    if not any(all(p[i] <= value[i] for i in range(3)) for p in claimed):
                        raise ValueError("An attainable cost has no displayed dominator")
            else:
                todo.extend(self.nodes[node][2])
        if witnessed != set(claimed):
            raise ValueError("A displayed nondominated vector is not attainable")
        return {"status": "PASS_COST_FAMILY", "cost_frontier_complete": True,
                "native_source_semantics": "NOT_ESTABLISHED_BY_COST_PACKET",
                "verified_equations": len(self.checked), "full_policy_enumeration": False}


def check(*, packet, certificate, max_seconds=900, max_equations=50_000_000,
          max_leaf_candidates=2_000_000, max_frontier_points=5000):
    instance = Instance(packet=packet, chains=certificate["chains"])
    if certificate.get("status") == "INCOMPLETE":
        if (certificate.get("packet_sha256") != digest(packet)
                or certificate.get("candidate_sha256") != digest(instance.candidate)
                or certificate.get("chains_sha256") != digest(certificate["chains"])
                or certificate.get("source_record") != packet.get("source_record")
                or certificate.get("source_semantics") != "NOT_ESTABLISHED_BY_COST_PACKET"
                or certificate.get("lower_bound") != [0, 0, 0]
                or certificate.get("cost_frontier_complete") is not False):
            raise ValueError("Invalid incomplete-result binding or completeness claim")
        check_incumbent(instance=instance, incumbent=certificate["incumbent"])
        return {"status": "PASS_INCOMPLETE_RECORD", "cost_frontier_complete": False,
                "native_source_semantics": "NOT_ESTABLISHED_BY_COST_PACKET",
                "guarantee": "Retained feasible cost witness and zero lower bound only"}
    try:
        return ProofChecker(instance=instance, certificate=certificate,
                            max_seconds=max_seconds, max_equations=max_equations,
                            max_leaf_candidates=max_leaf_candidates,
                            max_frontier_points=max_frontier_points).check()
    except Exhausted as exc:
        return {"status": "UNRESOLVED_COST_FAMILY", "cost_frontier_complete": False,
                "reason": str(exc), "native_source_semantics": "NOT_ESTABLISHED"}


def query(*, certificate, upper_bounds=None, weights=None, scales=None):
    """Call only after check(); exact cost-vector queries including unsupported points."""
    points = [tuple(v) for v in certificate["frontier"]]
    if upper_bounds is not None:
        points = [v for v in points if all(v[i] <= upper_bounds[i] for i in range(3))]
    if weights is None:
        return points
    if scales is None or len(weights) != 3 or len(scales) != 3:
        raise ValueError("Three positive weights and scales are required")
    weights = [Fraction(str(w)) for w in weights]
    scales = [Fraction(str(s)) for s in scales]
    if any(x <= 0 for x in weights + scales):
        raise ValueError("Recommendation weights/scales must be positive")
    factors = [w / s for w, s in zip(weights, scales)]
    if not points:
        return None
    best = min(points, key=lambda v: (sum(factors[i] * v[i] for i in range(3)), v))
    return {"cost": best, "score": str(sum(factors[i] * best[i] for i in range(3))),
            "scope": "Cost-vector recommendation; native layout/endpoint preferences require separate witness selection"}


def dominates(*, certificate, vector, strict=True):
    """Exact checked-frontier dominance query; caller must first call check()."""
    vector = tuple(vector)
    if len(vector) != 3:
        raise ValueError("A cost vector has three coordinates")
    return any(all(v[i] <= vector[i] for i in range(3))
               and (not strict or any(v[i] < vector[i] for i in range(3)))
               for v in certificate["frontier"])


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("solve", "check"))
    parser.add_argument("--instance", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--chains", help="Hash-bound closure-chain-inspection.json; solve only")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-operations", type=int, default=50_000_000)
    parser.add_argument("--max-frontier-points", type=int, default=5000)
    args = parser.parse_args()
    packet = load_packet(path=args.instance)
    if args.command == "check":
        result = check(packet=packet, certificate=json.loads(Path(args.proof).read_text()),
                       max_seconds=args.seconds, max_equations=args.max_operations,
                       max_frontier_points=args.max_frontier_points)
    else:
        record = json.loads(Path(args.chains).read_text())
        if record["source"] != digest(packet) or record["candidate_sha256"] != digest(packet["characteristic_candidates"][0]):
            raise ValueError("Chain record is not bound to this packet/candidate")
        result = solve(packet=packet, chains=record["chains"], seconds=args.seconds,
                       max_nodes=args.max_nodes, max_operations=args.max_operations,
                       max_frontier_points=args.max_frontier_points)
        Path(args.proof).write_text(canonical(result) + "\n")
    print(canonical({k: v for k, v in result.items()
                     if k not in ("nodes", "operations", "blocks", "intervals", "components")}))


if __name__ == "__main__":
    main()
