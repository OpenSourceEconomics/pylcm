"""Exact cost-frontier reference for an ordered incompatible-domain separator.

Standard library only. No Model, callback, solver, or project module is imported.
This reference was written but NOT executed by the reviewer (native preflight
failed). See ACA-SEPARATOR.md for the theorem, checked guards, and tie scope.

CLI:
  python aca_separator_reference.py solve INSTANCE.json OUTPUT.json
  python aca_separator_reference.py check INSTANCE.json OUTPUT.json

INSTANCE may be the bundled interned ACA header or an ordinary analysis JSON.
The default metadata budget is 900 seconds and 50000 Pareto points per state.
Budget exhaustion yields INCOMPLETE and a checked original-label candidate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
import argparse
import hashlib
import json
import time


METRICS = ("work", "storage", "clarity")
ZERO = (0, 0, 0)
RULE = "ordered-incompatible-domains-one-bridge/v1"
TIE_SCOPE = (
    "Complete cost frontier with deterministic feasible representatives. "
    "The legacy global canonical-layout minimum and legacy endpoint-filtered "
    "recommendation are not certified by this reference."
)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def plus(a, b):
    return tuple(x + y for x, y in zip(a, b))


def dominates(a, b):
    return a != b and all(x <= y for x, y in zip(a, b))


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Point:
    cost: tuple
    # Each group is a sequence of date-interval blocks; each block is cell IDs.
    groups: tuple


@dataclass(frozen=True)
class Attachment:
    cost: tuple
    regular: tuple
    anchor: tuple


def load_analysis(path):
    """Decode data and verify its bindings; never import the native loader."""
    path = Path(path)
    packet = json.loads(path.read_text())
    transport = {"header_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if packet.get("schema") == "interned-json-cost-instance/v1":
        records = packet.get("records")
        if records is None:
            records = []
            for part in packet["record_parts"]:
                name = part["path"]
                if Path(name).name != name:
                    raise ValueError("Record parts must be beside the header")
                raw = (path.parent / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != part["sha256"]:
                    raise ValueError("Record-part hash mismatch")
                rows = json.loads(raw)["records"]
                if len(rows) != part["records"]:
                    raise ValueError("Record-part length mismatch")
                records.extend(rows)
        values = []

        def decode(v):
            if not isinstance(v, dict):
                return v
            if set(v) != {"ref"} or type(v["ref"]) is not int or not 0 <= v["ref"] < len(values):
                raise ValueError("Reference is not a completed earlier record")
            return values[v["ref"]]

        for row in records:
            if set(row) == {"mapping"}:
                pairs = row["mapping"]
                if len({k for k, _ in pairs}) != len(pairs):
                    raise ValueError("Duplicate mapping key")
                values.append({k: decode(v) for k, v in pairs})
            elif set(row) == {"sequence"}:
                values.append([decode(v) for v in row["sequence"]])
            else:
                raise ValueError("Unknown interned row")
        expanded = decode(packet["root"])
        if digest(expanded) != packet["decoded_sha256"]:
            raise ValueError("Expanded packet hash mismatch")
        transport["expanded_packet_sha256"] = digest(expanded)
        packet = expanded
    if packet.get("schema") == "exact-graph-cost-instance/v1":
        enc = packet["policy_encoding"]
        policies = []
        for row in enc["policies"]:
            choices = list(enc["basis"])
            for index, value in row["changes"]:
                choices[index] = value
            selected = {c: packet["interfaces"][c]["options"][choices[i]]["id"]
                        for i, c in enumerate(enc["cells"])}
            policies.append({"id": row["id"], "option_by_cell": selected})
        if digest([[p["id"], p["option_by_cell"]] for p in policies]) != packet["original_policy_selection_sha256"]:
            raise ValueError("Selected-policy hash mismatch")
        analysis = {k: packet[k] for k in ("declarations", "interfaces", "derived_domains",
                    "discovery_complete", "clarity_weights", "allocation_scenario") if k in packet}
        analysis["projection_policies"] = policies
    else:
        analysis = packet
    return analysis, transport


class Problem:
    def __init__(self, analysis, *, bridge="dead", axis="health"):
        self.analysis, self.bridge, self.axis = analysis, bridge, axis
        decl = analysis["declarations"]
        if decl.get("analysis_scope") != "pylcm-structural-graph/v1":
            raise ValueError("Native structural graph scope is required")
        allocation = analysis.get("allocation_scenario", decl["allocation"])
        if allocation.get("kind") != "per_block":
            raise ValueError("Per-block allocation is required")
        self.boundary = allocation.get("boundary_unit", 1)
        self.weights = {"block": 4, "dispatch": 1, "characteristic": 2} | analysis.get("clarity_weights", {})
        if type(self.boundary) is not int or self.boundary < 0:
            raise ValueError("Nonnegative integer boundary coefficient required")
        if any(type(v) is not int or v < 0 for v in self.weights.values()) or self.weights["block"] <= 0:
            raise ValueError("Invalid clarity coefficients")
        if analysis.get("discovery_complete") is not True:
            raise ValueError("This seam requires a complete offered projection family")
        policies = analysis["projection_policies"]
        if len(policies) != 1:
            raise ValueError("This seam supports exactly one coherent policy")
        self.policy_record = policies[0]
        self.policy = self.policy_record["option_by_cell"]
        self.cells = {c["id"]: c for c in decl["cells"]}
        if len(self.cells) != len(decl["cells"]) or set(self.policy) != set(self.cells):
            raise ValueError("Cells or policy coverage are not exact")
        self.options = {}
        for cid in self.cells:
            matches = [o for o in analysis["interfaces"][cid]["options"] if o["id"] == self.policy[cid]]
            if len(matches) != 1:
                raise ValueError("Selected interface is not unique")
            self.options[cid] = matches[0]
        self.domains = decl["domains"] | analysis.get("derived_domains", {})
        self.roles = {}
        self.bylabel = defaultdict(list)
        for cid, c in self.cells.items():
            if type(c["period"]) is not int:
                raise ValueError("Integer dates required")
            role = c.get("role")
            if role is None:
                role = {"original_contexts_preserved_by_dispatch": True,
                        "decision_objective": c.get("decision_objective")}
            self.roles[cid] = canonical([role, c.get("ownership"), c.get("stage"), None,
                c.get("declaration_type"), c.get("structural_roles", {}), c.get("solver_identity")])
            self.bylabel[c["regime"]].append(cid)
        self.bylabel = {k: tuple(sorted(v)) for k, v in self.bylabel.items()}
        if len(set(self.roles.values())) != 1 or bridge not in self.bylabel:
            raise ValueError("One protected context and one existing bridge label required")
        self.pairs = set()
        for row in decl["relations"]:
            if row.get("support") != "zero":
                for target in row.get("targets", []):
                    self.pairs.add((row["source"], target))
        self.outgoing, self.incoming = defaultdict(set), defaultdict(set)
        for source, target in self.pairs:
            self.outgoing[source].add(target)
            self.incoming[target].add(source)
        self.chars = {c: frozenset(canonical(x) for x in o["state_projection"].values()
                                  if x.get("kind") == "characteristic") for c, o in self.options.items()}
        by_domain = defaultdict(list)
        for label, ids in self.bylabel.items():
            values = {self.options[c]["states"].get(axis) for c in ids}
            if label == bridge:
                if values != {None}:
                    raise ValueError("The bridge must omit the separating state axis")
            elif len(values) != 1 or None in values:
                raise ValueError("Every nonbridge label must have one fixed separating domain")
            else:
                by_domain[next(iter(values))].append(label)
        if len(by_domain) != 2:
            raise ValueError("Exactly two incompatible domain identities required")
        ordered = sorted(by_domain, key=lambda d: min(self.cells[c]["period"]
                         for l in by_domain[d] for c in self.bylabel[l]))
        self.left_domain, self.right_domain = ordered
        self.left = tuple(sorted(by_domain[ordered[0]]))
        self.right = tuple(sorted(by_domain[ordered[1]]))
        self.min_date = {l: min(self.cells[c]["period"] for c in ids) for l, ids in self.bylabel.items()}
        self.max_date = {l: max(self.cells[c]["period"] for c in ids) for l, ids in self.bylabel.items()}
        if max(self.max_date[l] for l in self.left) >= min(self.min_date[l] for l in self.right):
            raise ValueError("The incompatible sides are not strictly ordered in date")
        self.cuts = tuple(t for t in sorted({c["period"] for c in self.cells.values()})
                          if any(self.max_date[l] < t for l in self.left)
                          and any(self.min_date[l] >= t for l in self.right))

    def product(self, axes):
        result = 1
        for name, domain in axes.items():
            d = self.domains[domain]
            if d.get("solve_axis") is False:
                continue
            if type(d.get("size")) is not int or d["size"] <= 0:
                raise ValueError("Missing original domain cardinality: " + name)
            result *= d["size"]
        return result

    def literal_cost(self, ids):
        """Independent direct cost oracle, used for output-witness validation."""
        members = set(ids)
        if not members or len(members) != len(ids) or len({self.roles[c] for c in ids}) != 1:
            return None
        axes = [{}, {}]
        for c in ids:
            for i, kind in enumerate(("states", "actions")):
                for name, domain in self.options[c][kind].items():
                    if name in axes[i] and axes[i][name] != domain:
                        return None
                    axes[i][name] = domain
        s, a = map(self.product, axes)
        crossing = sum(source in members and target not in members for source, target in self.pairs)
        chars = set().union(*(self.chars[c] for c in ids))
        labels = {self.cells[c]["regime"] for c in ids}
        return ((len(ids) + self.boundary * crossing) * s * a, len(ids) * s,
                self.weights["block"] + self.weights["dispatch"] * (len(labels) - 1)
                + self.weights["characteristic"] * len(chars))

    def separator(self):
        return {"rule": RULE, "axis": self.axis, "bridge": self.bridge,
                "left_domain": self.left_domain, "right_domain": self.right_domain,
                "left_labels": self.left, "right_labels": self.right, "cuts": self.cuts}


class Engine:
    """Producer: incremental outgoing-boundary costs and suffix interval DP.

    Checker mode: endpoint-count cost tables and prefix interval DP, highest
    rather than lowest label pivots. Neither trusts a producer cost or rule flag.
    The common outer separator identity is the theorem stated in the appendix.
    """
    def __init__(self, problem, *, checker=False, seconds=900, max_points=50000):
        self.p, self.checker = problem, checker
        self.seconds, self.max_points, self.started = seconds, max_points, time.perf_counter()
        self.intervals, self.residual, self.trace = {}, {}, {}
        self.counts = Counter()

    def tick(self):
        if time.perf_counter() - self.started >= self.seconds:
            raise BudgetExhausted("seconds")

    def pf(self, items):
        unique = {}
        def prune_buffer():
            kept = {}
            for cost in sorted(unique):
                self.tick()
                if not any(dominates(other, cost) for other in kept):
                    kept[cost] = unique[cost]
            if len(kept) > self.max_points:
                raise BudgetExhausted("frontier_points")
            return kept
        for item in items:
            self.tick()
            if isinstance(item, Attachment):
                key = canonical([item.regular, item.anchor])
            else:
                key = canonical(item.groups)
            if item.cost not in unique or key < unique[item.cost][0]:
                unique[item.cost] = (key, item)
            if len(unique) > max(256, 2 * min(self.max_points, 5000)):
                unique = prune_buffer()
        # Direct dominance is deliberately transparent. Frontier cardinality,
        # rather than the old 19-label grouping recurrence, is the residual risk.
        unique = prune_buffer()
        return [unique[cost][1] for cost in sorted(unique)]

    def record(self, name, points):
        self.trace[name] = digest([p.cost for p in points])

    def cells_for(self, labels, mask):
        return tuple(sorted(c for i, label in enumerate(labels) if mask & (1 << i)
                            for c in self.p.bylabel[label]))

    def _checker_table(self, buckets):
        """Independent 2-D endpoint count, not incremental boundary updates."""
        p, n = self.p, len(buckets)
        index = {c: i for i, bucket in enumerate(buckets) for c in bucket}
        counts = [[0] * n for _ in range(n)]
        out = [0] * n
        for source, target in p.pairs:
            if source in index and source != target:
                out[index[source]] += 1
                if target in index:
                    a, b = sorted((index[source], index[target]))
                    counts[a][b] += 1
        inside = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            running = 0
            for j in range(i + 1, n + 1):
                running += counts[i][j - 1]
                inside[i][j] = inside[i + 1][j] + running
        features = defaultdict(lambda: [0] * (n + 1))
        for i, bucket in enumerate(buckets):
            features[("cells",)][i + 1] = len(bucket)
            features[("out",)][i + 1] = out[i]
            for c in bucket:
                features[("label", p.cells[c]["regime"])][i + 1] += 1
                for ch in p.chars[c]:
                    features[("char", ch)][i + 1] += 1
                for kind in ("states", "actions"):
                    for name, domain in p.options[c][kind].items():
                        features[(kind, name, domain)][i + 1] += 1
        for seq in features.values():
            for i in range(n):
                seq[i + 1] += seq[i]
        costs = {}
        for i in range(n):
            for j in range(i + 1, n + 1):
                self.tick()
                axes = {"states": {}, "actions": {}}
                labels = chars = 0
                admissible = True
                for token, seq in features.items():
                    if seq[j] == seq[i]:
                        continue
                    if token[0] in axes:
                        kind, name, domain = token
                        if name in axes[kind] and axes[kind][name] != domain:
                            admissible = False
                        axes[kind][name] = domain
                    elif token[0] == "label":
                        labels += 1
                    elif token[0] == "char":
                        chars += 1
                if admissible:
                    s, a = p.product(axes["states"]), p.product(axes["actions"])
                    ncells = features[("cells",)][j] - features[("cells",)][i]
                    crossings = features[("out",)][j] - features[("out",)][i] - inside[i][j]
                    costs[i, j] = ((ncells + p.boundary * crossings) * s * a, ncells * s,
                        p.weights["block"] + p.weights["dispatch"] * (labels - 1)
                        + p.weights["characteristic"] * chars)
        return costs

    def group(self, ids):
        ids = tuple(sorted(ids))
        if not ids:
            return [Point(ZERO, ())]
        if ids in self.intervals:
            return self.intervals[ids]
        p = self.p
        dates = defaultdict(list)
        for c in ids:
            dates[p.cells[c]["period"]].append(c)
        buckets = [tuple(v) for _, v in sorted(dates.items())]
        n = len(buckets)
        if self.checker:
            costs = self._checker_table(buckets)
            dp = {0: [Point(ZERO, ())]}
            for stop in range(1, n + 1):
                candidates = []
                for start in range(stop):
                    if (start, stop) in costs:
                        candidates.extend(Point(plus(v.cost, costs[start, stop]), ()) for v in dp[start])
                dp[stop] = self.pf(candidates)
            result = dp[n]
        else:
            suffix = {n: [Point(ZERO, ((),))]}
            for start in range(n - 1, -1, -1):
                members, labels, chars = set(), set(), set()
                axes = {"states": {}, "actions": {}}
                crossings, compatible = 0, True
                candidates = []
                for stop in range(start + 1, n + 1):
                    self.tick()
                    for c in buckets[stop - 1]:
                        crossings -= len(p.incoming[c] & members)
                        crossings += sum(t != c and t not in members for t in p.outgoing[c])
                        members.add(c)
                        labels.add(p.cells[c]["regime"])
                        chars.update(p.chars[c])
                        for kind in axes:
                            for name, domain in p.options[c][kind].items():
                                if name in axes[kind] and axes[kind][name] != domain:
                                    compatible = False
                                axes[kind][name] = domain
                    if not compatible:
                        break
                    s, a = p.product(axes["states"]), p.product(axes["actions"])
                    cost = ((len(members) + p.boundary * crossings) * s * a, len(members) * s,
                        p.weights["block"] + p.weights["dispatch"] * (len(labels) - 1)
                        + p.weights["characteristic"] * len(chars))
                    block = tuple(sorted(members))
                    candidates.extend(Point(plus(cost, tail.cost), ((block,) + tail.groups[0],))
                                      for tail in suffix[stop])
                suffix[start] = self.pf(candidates)
            result = suffix[0]
        self.intervals[ids] = result
        self.counts["interval_groups"] += 1
        self.record("interval:" + digest(ids), result)
        return result

    def residuals(self, labels, name):
        count = 1 << len(labels)
        for mask in range(1, count):
            self.group(self.cells_for(labels, mask))
        dp = {0: [Point(ZERO, ())]}
        for mask in range(1, count):
            pivot = (1 << (mask.bit_length() - 1)) if self.checker else mask & -mask
            def candidates():
                rest, sub = mask ^ pivot, mask ^ pivot
                while True:
                    self.counts["residual_transitions"] += 1
                    group_mask = sub | pivot
                    head = self.group(self.cells_for(labels, group_mask))
                    for a in head:
                        for b in dp[mask ^ group_mask]:
                            yield Point(plus(a.cost, b.cost), a.groups + b.groups)
                    if sub == 0:
                        break
                    sub = (sub - 1) & rest
            dp[mask] = self.pf(candidates())
            self.record(f"residual:{name}:{mask}", dp[mask])
        self.residual[name] = dp
        return dp[count - 1]

    def attached(self, labels, name, bridge_ids, eligible, *, nonempty, tag):
        full = (1 << len(labels)) - 1
        eligible_mask = sum(1 << i for i, l in enumerate(labels) if l in eligible)
        def candidates():
            subset = eligible_mask
            while True:
                self.tick()
                if subset or not nonempty:
                    self.counts["attachment_subsets"] += 1
                    anchors = self.group(self.cells_for(labels, subset) + tuple(bridge_ids))
                    rest = self.residual[name][full ^ subset]
                    for a in anchors:
                        for b in rest:
                            anchor = () if self.checker else (a.groups[0] if a.groups else ())
                            yield Attachment(plus(a.cost, b.cost), b.groups, anchor)
                if subset == 0:
                    break
                subset = (subset - 1) & eligible_mask
        result = self.pf(candidates())
        self.record("attachment:" + tag, result)
        return result

    def solve(self):
        p = self.p
        left = self.residuals(p.left, "left")
        right = self.residuals(p.right, "right")
        bridge = p.bylabel[p.bridge]
        lf = self.attached(p.left, "left", bridge, p.left, nonempty=False, tag="left-full")
        rf = self.attached(p.right, "right", bridge, p.right, nonempty=False, tag="right-full")
        candidates = self.pf(chain(
            (Point(plus(a.cost, b.cost), a.regular + ((a.anchor,) if a.anchor else ()) + b.groups)
             for a in lf for b in right),
            (Point(plus(a.cost, b.cost), a.groups + b.regular + ((b.anchor,) if b.anchor else ()))
             for a in left for b in rf)))
        for cut in p.cuts:
            eligible_l = {l for l in p.left if p.max_date[l] < cut}
            eligible_r = {l for l in p.right if p.min_date[l] >= cut}
            before = [c for c in bridge if p.cells[c]["period"] < cut]
            after = [c for c in bridge if p.cells[c]["period"] >= cut]
            a = self.attached(p.left, "left", before, eligible_l, nonempty=True, tag=f"left:{cut}")
            b = self.attached(p.right, "right", after, eligible_r, nonempty=True, tag=f"right:{cut}")
            combined = (Point(plus(x.cost, y.cost), x.regular + y.regular
                        + (((x.anchor + y.anchor),) if x.anchor or y.anchor else ())) for x in a for y in b)
            candidates = self.pf(chain(candidates, combined))
        self.record("root", candidates)
        return candidates


def original_point(problem):
    groups, total = [], ZERO
    for label, ids in sorted(problem.bylabel.items()):
        cost = problem.literal_cost(ids)
        if cost is None:
            raise ValueError("Original-label candidate is not one admissible block per label")
        groups.append((ids,))
        total = plus(total, cost)
    return Point(total, tuple(groups))


def dump_point(point):
    return {"cost": dict(zip(METRICS, point.cost)), "groups": point.groups}


def produce(problem, *, seconds=900, max_points=50000):
    seed = original_point(problem)
    engine = Engine(problem, seconds=seconds, max_points=max_points)
    complete, reason = True, None
    try:
        points = engine.solve()
    except BudgetExhausted as exc:
        complete, reason, points = False, str(exc), [seed]
    return {"schema": "aca-ordered-separator/v1", "status": "COMPLETE" if complete else "INCOMPLETE",
            "analysis_sha256": digest(problem.analysis), "policy_sha256": digest(problem.policy_record),
            "separator": problem.separator(), "frontier": [dump_point(x) for x in points],
            "original_candidate": dump_point(seed), "tie_scope": TIE_SCOPE,
            "certificate": {"complete": complete, "trace_sha256": digest(engine.trace) if complete else None,
                            "trace_records": len(engine.trace), "counts": dict(engine.counts)},
            "elapsed_seconds": time.perf_counter() - engine.started, "exhaustion_reason": reason,
            "lower_bound": None if complete else dict(zip(METRICS, ZERO))}


def validate_point(problem, point):
    """Validate the full original static grouping and every date-bucket cut."""
    seen, owners, total = set(), {}, ZERO
    for gi, group in enumerate(point["groups"]):
        if not group:
            raise ValueError("Empty group")
        ids = [c for block in group for c in block]
        if len(ids) != len(set(ids)) or seen.intersection(ids) or any(c not in problem.cells for c in ids):
            raise ValueError("Cell duplication, overlap, or unknown cell")
        seen.update(ids)
        labels = {problem.cells[c]["regime"] for c in ids}
        for label in labels:
            if label in owners:
                raise ValueError("One original label belongs to multiple groups")
            owners[label] = gi
        expected = set(c for label in labels for c in problem.bylabel[label])
        if set(ids) != expected:
            raise ValueError("Static group does not contain all dates of its labels")
        dates = sorted({problem.cells[c]["period"] for c in ids})
        position = 0
        for block in group:
            if not block:
                raise ValueError("Empty block")
            block_dates = sorted({problem.cells[c]["period"] for c in block})
            if block_dates != dates[position:position + len(block_dates)]:
                raise ValueError("Blocks are not consecutive date-bucket intervals")
            expected_block = {c for c in ids if problem.cells[c]["period"] in set(block_dates)}
            if set(block) != expected_block:
                raise ValueError("A date bucket was split")
            position += len(block_dates)
            cost = problem.literal_cost(block)
            if cost is None:
                raise ValueError("Inadmissible block")
            total = plus(total, cost)
        if position != len(dates):
            raise ValueError("Incomplete group interval coverage")
    if seen != set(problem.cells) or total != tuple(point["cost"][k] for k in METRICS):
        raise ValueError("Incomplete cell coverage or false cost")


def check(problem, result, *, seconds=900, max_points=50000):
    try:
        if result.get("schema") != "aca-ordered-separator/v1":
            raise ValueError("Wrong result schema")
        if result.get("analysis_sha256") != digest(problem.analysis) or result.get("policy_sha256") != digest(problem.policy_record):
            raise ValueError("Input/policy binding mismatch")
        if canonical(result.get("separator")) != canonical(problem.separator()):
            raise ValueError("Separator guards/coverage mismatch")
        if result.get("tie_scope") != TIE_SCOPE:
            raise ValueError("Result claims a different witness/recommendation scope")
        validate_point(problem, result["original_candidate"])
        for point in result["frontier"]:
            validate_point(problem, point)
        vectors = [tuple(point["cost"][k] for k in METRICS) for point in result["frontier"]]
        if len(vectors) != len(set(vectors)) or any(dominates(a, b) for a in vectors for b in vectors):
            raise ValueError("Supplied frontier is not an antichain")
        if result["status"] == "INCOMPLETE":
            if result["certificate"].get("complete") is not False:
                raise ValueError("False complete flag")
            if result.get("lower_bound") != dict(zip(METRICS, ZERO)) or result["certificate"].get("trace_sha256") is not None:
                raise ValueError("Unsupported partial-result bound or complete recurrence claim")
            return {"status": "PASS_PARTIAL", "frontier_complete_verified": False}
        if result["status"] != "COMPLETE" or result["certificate"].get("complete") is not True:
            raise ValueError("Inconsistent status")
        engine = Engine(problem, checker=True, seconds=seconds, max_points=max_points)
        expected = engine.solve()
        if vectors != [x.cost for x in expected]:
            raise ValueError("Incomplete or incorrect cost frontier")
        if result["certificate"].get("trace_sha256") != digest(engine.trace):
            raise ValueError("Independent recurrence commitment differs")
        if result["certificate"].get("trace_records") != len(engine.trace) or result["certificate"].get("counts") != dict(engine.counts):
            raise ValueError("Independent coverage counts differ")
        return {"status": "PASS", "frontier_complete_verified": True, "counts": dict(engine.counts),
                "tie_scope": TIE_SCOPE, "elapsed_seconds": time.perf_counter() - engine.started}
    except BudgetExhausted as exc:
        return {"status": "UNVERIFIED", "frontier_complete_verified": False, "reason": str(exc)}
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"status": "FAIL", "frontier_complete_verified": False, "reason": str(exc)}


def cost_query(result, *, upper_bounds=None, positive_weights=None):
    """Exact upper-threshold and positive-rational cost recommendation queries.

    Call only after independent PASS. This query never applies the legacy
    endpoint preference to a different representative and never claims to do so.
    """
    if result["status"] != "COMPLETE":
        raise ValueError("A complete, independently checked result is required")
    points = result["frontier"]
    if upper_bounds is not None:
        points = [p for p in points if all(p["cost"][k] <= v for k, v in upper_bounds.items())]
    if positive_weights is None:
        return points
    weights = {k: Fraction(str(positive_weights[k])) for k in METRICS}
    if any(w <= 0 for w in weights.values()):
        raise ValueError("Strictly positive rational weights required")
    return min(points, key=lambda p: (sum(weights[k] * p["cost"][k] for k in METRICS),
               tuple(p["cost"][k] for k in METRICS))) if points else None


def strictly_dominates_cost(result, candidate):
    """After independent PASS, test strict cost domination of an arbitrary vector."""
    if result["status"] != "COMPLETE":
        raise ValueError("A complete, independently checked result is required")
    value = tuple(candidate[k] for k in METRICS)
    return any(dominates(tuple(point["cost"][k] for k in METRICS), value) for point in result["frontier"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("solve", "check"))
    parser.add_argument("instance")
    parser.add_argument("result")
    parser.add_argument("--bridge", default="dead")
    parser.add_argument("--axis", default="health")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()
    analysis, transport = load_analysis(args.instance)
    problem = Problem(analysis, bridge=args.bridge, axis=args.axis)
    if args.mode == "solve":
        result = produce(problem, seconds=args.seconds, max_points=args.max_points)
        result["transport"] = transport
        Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "exhaustion_reason")}))
    else:
        result = check(problem, json.loads(Path(args.result).read_text()), seconds=args.seconds, max_points=args.max_points)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
