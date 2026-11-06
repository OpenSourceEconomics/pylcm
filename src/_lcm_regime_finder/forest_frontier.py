"""Exact singleton-component Pareto optimizer for a forest of internal pairs.

Standard library only; never imports the audited implementation.  The output is
an explicit frontier and a local recurrence certificate, not a solver status.
The certificate checker lives in forest_verify.py and does not import this file.

This is a reviewer reference seam. Native/corpus execution is NOT claimed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import time

METRICS = ("work", "storage", "clarity")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def signature(cell):
    return (tuple(sorted(cell["states"].items())),
            tuple(sorted(cell["actions"].items())),
            tuple(sorted(cell["characteristics"])))


def join(a, b):
    axes = []
    for x, y in zip(a[:2], b[:2]):
        merged = dict(x)
        for name, domain in y:
            if name in merged and merged[name] != domain:
                return None
            merged[name] = domain
        axes.append(tuple(sorted(merged.items())))
    return (*axes, tuple(sorted(set(a[2]) | set(b[2]))))


def _bins(cells):
    found = {signature(c) for c in cells}
    todo = list(found)
    while todo:
        left = todo.pop()
        for right in tuple(found):
            value = join(left, right)
            if value is not None and value not in found:
                found.add(value)
                todo.append(value)
    return sorted(found)


def _prepare(instance):
    cells = sorted(instance["cells"], key=lambda c: c["id"])
    ids = [c["id"] for c in cells]
    if not cells or len(set(ids)) != len(ids):
        raise ValueError("Require nonempty unique original cells")
    if len({c["label"] for c in cells}) != len(cells):
        raise ValueError("Require exactly one cell per original label")
    if len({canonical(c["context"]) for c in cells}) != 1:
        raise ValueError("Protected contexts differ")
    if instance["allocation"] != "per_block":
        raise ValueError("Require per_block allocation")
    w = instance["clarity_weights"]
    if any(type(w[k]) is not int or w[k] < 0 for k in ("block", "dispatch", "characteristic")):
        raise ValueError("Require nonnegative integer clarity weights")
    if w["block"] <= w["dispatch"]:
        raise ValueError("Require strict block > dispatch guard")
    beta = instance["boundary_unit"]
    if type(beta) is not int or beta < 0:
        raise ValueError("Require a nonnegative integer boundary unit")
    for cell in cells:
        if len(set(cell["characteristics"])) != len(cell["characteristics"]):
            raise ValueError("Characteristic identifiers must be unique")
        for kind in ("states", "actions"):
            for domain in cell[kind].values():
                n = instance["domains"][domain]["size"]
                if type(n) is not int or n <= 0:
                    raise ValueError("Require positive integer captured domain sizes")
    by_id = {c["id"]: c for c in cells}
    pairs = set(map(tuple, instance["pairs"]))
    if any(s not in by_id for s, _ in pairs):
        raise ValueError("Only source-owned directed pairs belong to a component")
    edges = {tuple(sorted((s, t))) for s, t in pairs if t in by_id and s != t}
    parent = {v: v for v in ids}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    adjacency = {v: set() for v in ids}
    for s, t in sorted(edges):
        a, b = find(s), find(t)
        if a == b:
            raise ValueError("Internal non-self graph is not a forest")
        parent[a] = b
        adjacency[s].add(t)
        adjacency[t].add(s)
    bins = _bins(cells)
    lookup = {sig: i for i, sig in enumerate(bins)}
    joins = {(i, j): lookup[value] for i, a in enumerate(bins) for j, b in enumerate(bins)
             if (value := join(a, b)) is not None}
    choices = {c["id"]: tuple(k for k, b in enumerate(bins) if join(signature(c), b) == b)
               for c in cells}
    rows = [(math.prod(instance["domains"][d]["size"] for _, d in sig[0]
                       if instance["domains"][d].get("solve_axis") is not False),
             math.prod(instance["domains"][d]["size"] for _, d in sig[1]
                       if instance["domains"][d].get("solve_axis") is not False)) for sig in bins]
    outside = {v: sum(s == v and t not in by_id for s, t in pairs) for v in ids}
    return {"cells": cells, "by_id": by_id, "pairs": pairs, "adj": adjacency,
            "bins": bins, "lookup": lookup, "joins": joins, "choices": choices,
            "rows": rows, "outside": outside, "beta": beta, "weights": w}


class Exhausted(RuntimeError):
    pass


@dataclass
class Budget:
    seconds: float | None = 900.0
    max_states: int = 200_000
    max_points: int = 2_000_000
    started: float = 0.0
    states: int = 0
    points: int = 0
    candidates: int = 0
    mask_dominated_points: int = 0
    mask_dominated_states: int = 0

    def __post_init__(self):
        self.started = time.perf_counter()
        if self.seconds is not None and self.seconds < 0:
            raise ValueError("seconds must be nonnegative or None")
        if self.max_states < 1 or self.max_points < 1:
            raise ValueError("state/point limits must be positive")

    def check(self):
        if self.seconds is not None and time.perf_counter() - self.started > self.seconds:
            raise Exhausted("forest recurrence elapsed-time budget")
        if self.states > self.max_states:
            raise Exhausted("forest recurrence stored-state budget")
        if self.points > self.max_points:
            raise Exhausted("forest recurrence stored-point budget")

    def receipt(self):
        return {"stored_states": self.states, "stored_points": self.points,
                "mask_dominated_points": self.mask_dominated_points,
                "mask_dominated_states": self.mask_dominated_states,
                "candidate_transitions": self.candidates,
                "elapsed_seconds": time.perf_counter() - self.started}


def _literal_original(p):
    work = storage = clarity = 0
    w = p["weights"]
    for cell in p["cells"]:
        k = p["lookup"][signature(cell)]
        ns, na = p["rows"][k]
        crossing = sum(s == cell["id"] and t != s for s, t in p["pairs"])
        work += ns * na * (1 + p["beta"] * crossing)
        storage += ns
        clarity += w["block"] + w["characteristic"] * len(p["bins"][k][2])
    return {"blocks": [[c["id"]] for c in p["cells"]], "cost": [work, storage, clarity]}


def _normalize_and_cost(p, assignment):
    """Construct actual token providers by shrinking then coalescing bins."""
    members = defaultdict(list)
    for v, k in assignment.items():
        members[k].append(v)
    normalized = {}
    for group in members.values():
        actual = ((), (), ())
        for v in group:
            actual = join(actual, signature(p["by_id"][v]))
        k = p["lookup"][actual]
        normalized.update({v: k for v in group})
    grouped = defaultdict(set)
    for v, k in normalized.items():
        grouped[k].add(v)
    total = [0, 0, 0]
    w = p["weights"]
    for k, ids in grouped.items():
        ns, na = p["rows"][k]
        crossing = sum(s in ids and t not in ids for s, t in p["pairs"])
        total[0] += (len(ids) + p["beta"] * crossing) * ns * na
        total[1] += len(ids) * ns
        total[2] += w["block"] + w["dispatch"] * (len(ids) - 1) + w["characteristic"] * len(p["bins"][k][2])
    return normalized, tuple(total)


def optimize(instance, *, seconds=900.0, max_states=200_000, max_points=2_000_000, checkpoint=None):
    """Exact forest optimizer using boundary labels and a used-bin mask.

    The recurrence includes missing-provider assignments as an explicit
    relaxation. Every such assignment normalizes to an actual partition with
    no larger cost. Every true reduced partition is already in the relaxation.
    Thus their Pareto vector sets coincide. At each reported nondominated root
    vector, normalization MUST have equal cost and supply every active token.
    """
    budget = Budget(seconds, max_states, max_points)
    p = _prepare(instance)
    bins, rows, beta = p["bins"], p["rows"], p["beta"]
    nbin = len(bins)
    tables, scopes = [], []
    all_ids = set(p["by_id"])
    base = {"schema": "regime-singleton-forest-mask-frontier/v2", "instance_sha256": digest(instance),
            "scope": "exact singleton-label cost family; source semantics and endpoint preferences separate",
            "state_cover_rule": "same-boundary-subset-mask-weak-cost/v1",
            "normalization_rule": "shrink-to-actual-union-then-merge-equal-unions/v1",
            "bins": bins, "original_representation": _literal_original(p)}

    def save(operation, scope, candidates):
        by_state = {}
        for state, cost, witness in candidates:
            budget.candidates += 1
            budget.check()
            front = by_state.get(state)
            if front is None:
                front = by_state[state] = []
                budget.states += 1
            if any(q[0][0] <= cost[0] and q[0][1] <= cost[1] for q in front):
                continue
            kept = [q for q in front if not (cost[0] <= q[0][0] and cost[1] <= q[0][1])]
            budget.points += 1 + len(kept) - len(front)
            kept.append((cost, witness))
            by_state[state] = kept
            budget.check()
        # Future joins only union masks; a subset activates no additional bins.
        boundary_points = defaultdict(list)
        for (boundary, mask), points in by_state.items():
            boundary_points[boundary].extend((cost, mask.bit_count(), mask, witness) for cost, witness in points)
        reduced = {}
        for boundary, points in boundary_points.items():
            best_storage = {}
            for cost, _, mask, witness in sorted(points):
                budget.check()
                if (1 << mask.bit_count()) <= len(best_storage):
                    subset, covered = mask, False
                    while True:
                        if best_storage.get(subset, math.inf) <= cost[1]:
                            covered = True
                            break
                        if subset == 0:
                            break
                        subset = (subset - 1) & mask
                else:
                    covered = any(other & mask == other and storage <= cost[1]
                                  for other, storage in best_storage.items())
                if covered:
                    budget.mask_dominated_points += 1
                    continue
                reduced.setdefault((boundary, mask), []).append((cost, witness))
                best_storage[mask] = min(best_storage.get(mask, math.inf), cost[1])
        removed_states = len(by_state) - len(reduced)
        removed_points = sum(map(len, by_state.values())) - sum(map(len, reduced.values()))
        budget.states -= removed_states
        budget.points -= removed_points
        budget.mask_dominated_states += removed_states
        by_state = reduced
        table_rows = []
        for (b, mask), points in sorted(by_state.items()):
            table_rows.append({"boundary_bin": b, "used_bins": mask,
                               "points": [{"cost": list(cost), "from": source}
                                          for cost, source in sorted(points)]})
        tables.append({"operation": operation, "rows": table_rows})
        scopes.append(frozenset(scope))
        if checkpoint is not None:
            checkpoint({'schema': 'regime-forest-table-checkpoint/v1',
                'instance_sha256': base['instance_sha256'], 'table_index': len(tables) - 1,
                'table': tables[-1], 'complete_cost_family': False})
        return len(tables) - 1

    def entries(at):
        for ri, row in enumerate(tables[at]["rows"]):
            state = row["boundary_bin"], row["used_bins"]
            for pi, point in enumerate(row["points"]):
                yield state, tuple(point["cost"]), [ri, pi]

    def leaf(v):
        def candidates():
            for k in p["choices"][v]:
                ns, na = rows[k]
                yield (k, 1 << k), (ns * na * (1 + beta * p["outside"][v]), ns), []
        return save({"kind": "cell", "cell": v}, {v}, candidates())

    def combine(left, right, parent=None, child=None):
        scope = scopes[left] | scopes[right]
        is_edge = parent is not None
        operation = {"kind": "edge" if is_edge else "forest", "left": left, "right": right}
        if is_edge:
            operation.update(parent=parent, child=child)
        def candidates():
            for a, ac, ai in entries(left):
                for b, bc, bi in entries(right):
                    extra = 0
                    if is_edge and a[0] != b[0]:
                        extra = beta * ((math.prod(rows[a[0]]) if (parent, child) in p["pairs"] else 0)
                                        + (math.prod(rows[b[0]]) if (child, parent) in p["pairs"] else 0))
                    yield ((a[0] if is_edge else -1), a[1] | b[1]), (ac[0] + bc[0] + extra, ac[1] + bc[1]), [*ai, *bi]
        return save(operation, scope, candidates())

    def assignment_at(table_index, row_index, point_index):
        assigned = {}
        todo = [(table_index, row_index, point_index)]
        while todo:
            at, ri, pi = todo.pop()
            table = tables[at]
            row = table["rows"][ri]
            op = table["operation"]
            if op["kind"] == "cell":
                assigned[op["cell"]] = row["boundary_bin"]
            elif op["kind"] != "empty":
                a, b, c, d = row["points"][pi]["from"]
                todo.extend(((op["left"], a, b), (op["right"], c, d)))
        return assigned

    try:
        accumulator = save({"kind": "empty"}, set(), [((-1, 0), (0, 0), [])])
        unseen = set(all_ids)
        while unseen:
            root = min(unseen)
            parent = {root: None}
            order, stack = [], [root]
            while stack:
                v = stack.pop()
                order.append(v)
                for c in sorted(p["adj"][v] - ({parent[v]} if parent[v] is not None else set()), reverse=True):
                    parent[c] = v
                    stack.append(c)
            solved = {}
            for v in reversed(order):
                at = leaf(v)
                for c in sorted(w for w in p["adj"][v] if parent.get(w) == v):
                    at = combine(at, solved[c], v, c)
                solved[v] = at
            accumulator = combine(accumulator, solved[root])
            unseen.difference_update(order)
        candidates = []
        w = p["weights"]
        for ri, row in enumerate(tables[accumulator]["rows"]):
            used = [k for k in range(nbin) if row["used_bins"] & (1 << k)]
            clarity = w["dispatch"] * len(p["cells"]) + sum(
                w["block"] - w["dispatch"] + w["characteristic"] * len(bins[k][2]) for k in used)
            for pi, point in enumerate(row["points"]):
                candidates.append(((*point["cost"], clarity), [ri, pi]))
        best = {}
        for cost, source in sorted(candidates):
            budget.check()
            if any(all(x <= y for x, y in zip(q, cost)) for q in best):
                continue
            best = {q: s for q, s in best.items() if not all(x <= y for x, y in zip(cost, q))}
            best[cost] = source
        frontier = []
        for cost, (ri, pi) in sorted(best.items()):
            budget.check()
            normalized, true_cost = _normalize_and_cost(p, assignment_at(accumulator, ri, pi))
            if true_cost != cost:
                raise ValueError("A relaxed nondominated point failed equal-cost normalization")
            frontier.append({"cost": list(cost), "source": [ri, pi],
                             "normalized_witness_sha256": digest(normalized)})
        return {**base, "status": "COMPLETE", "tables": tables, "root_table": accumulator,
                "frontier": frontier, "lower_bound": None, "exhaustion_reason": None, "resource": budget.receipt()}
    except Exhausted as exc:
        return {**base, "status": "INCOMPLETE", "tables": tables, "root_table": None,
                "frontier": [], "lower_bound": [0, 0, 0], "exhaustion_reason": str(exc), "resource": budget.receipt()}
