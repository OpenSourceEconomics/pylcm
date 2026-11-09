"""Independent local cover checker for forest_frontier.py certificates.

No producer imports, no optimization backend, no floating-point inequalities.
The checker interprets original pairs and recomputes all local transitions. It
checks feasible representatives and coverage in both directions; a list of
feasible witnesses alone cannot pass as a complete recurrence.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math


def _json(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(x):
    return hashlib.sha256(_json(x).encode()).hexdigest()


def _signature(cell):
    return (tuple(sorted(cell["states"].items())), tuple(sorted(cell["actions"].items())),
            tuple(sorted(cell["characteristics"])))


def _union(signatures):
    states, actions, characteristics = {}, {}, set()
    for s, a, h in signatures:
        for target, axes in ((states, s), (actions, a)):
            for name, domain in axes:
                if name in target and target[name] != domain:
                    return None
                target[name] = domain
        characteristics.update(h)
    return tuple(sorted(states.items())), tuple(sorted(actions.items())), tuple(sorted(characteristics))


def _decode(instance):
    cells = {c["id"]: c for c in instance["cells"]}
    if not cells or len(cells) != len(instance["cells"]):
        raise ValueError("Invalid original cell identity")
    if len({c["label"] for c in cells.values()}) != len(cells):
        raise ValueError("Not a singleton-label family")
    if len({_json(c["context"]) for c in cells.values()}) != 1:
        raise ValueError("Protected contexts disagree")
    weights = instance["clarity_weights"]
    if any(type(weights[k]) is not int or weights[k] < 0 for k in ("block", "dispatch", "characteristic")):
        raise ValueError("Invalid clarity coefficients")
    if weights["block"] <= weights["dispatch"] or instance["allocation"] != "per_block":
        raise ValueError("Union-signature dominance guard fails")
    beta = instance["boundary_unit"]
    if type(beta) is not int or beta < 0:
        raise ValueError("Invalid directed-boundary coefficient")
    original = {v: _signature(c) for v, c in cells.items()}
    for v, sig in original.items():
        if len(set(cells[v]["characteristics"])) != len(cells[v]["characteristics"]):
            raise ValueError("Duplicate characteristic identifier")
        for axes in sig[:2]:
            for _, name in axes:
                size = instance["domains"][name]["size"]
                if type(size) is not int or size <= 0:
                    raise ValueError("Invalid domain cardinality")
    # A different construction from producer pairwise saturation: induct over
    # primitive types, adjoining each compatible subset-union exactly as a set.
    possibilities = set()
    for sig in sorted(set(original.values())):
        extension = {_union((prior, sig)) for prior in possibilities}
        possibilities |= {sig} | (extension - {None})
    bins = sorted(possibilities)
    by_signature = {sig: k for k, sig in enumerate(bins)}
    allowed = {v: tuple(k for k, sig in enumerate(bins) if _union((original[v], sig)) == sig)
               for v in cells}
    rows = []
    for sig in bins:
        dimensions = []
        for axes in sig[:2]:
            size = 1
            for _, name in axes:
                domain = instance["domains"][name]
                if domain.get("solve_axis") is not False:
                    size *= domain["size"]
            dimensions.append(size)
        rows.append(tuple(dimensions))
    pairs = set(tuple(pair) for pair in instance["pairs"])
    if any(len(pair) != 2 or pair[0] not in cells for pair in pairs):
        raise ValueError("A component pair is not source-owned")
    adjacency = {v: set() for v in cells}
    for s, t in pairs:
        if s != t and t in cells:
            adjacency[s].add(t)
            adjacency[t].add(s)
    visited = set()
    for root in sorted(cells):
        if root in visited:
            continue
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            if node in visited:
                raise ValueError("Internal non-self pair graph has a cycle")
            visited.add(node)
            stack.extend((v, node) for v in adjacency[node] if v != parent)
    return cells, original, bins, by_signature, allowed, rows, pairs, weights, beta


def _weak(a, b):
    return all(x <= y for x, y in zip(a, b))


def _antichain(points):
    return len(points) == len(set(points)) and all(
        not _weak(a, b) for i, a in enumerate(points) for j, b in enumerate(points) if i != j)


def _literal(instance, assignment, bins):
    """Independent witness interpretation, not a recurrence cost routine."""
    cells = {c["id"]: c for c in instance["cells"]}
    if set(assignment) != set(cells):
        raise ValueError("Final witness omits or duplicates original cells")
    groups = {}
    for v, k in assignment.items():
        if type(k) is not int or not 0 <= k < len(bins):
            raise ValueError("Invalid witness bin")
        groups.setdefault(k, []).append(v)
    result = [0, 0, 0]
    pairs = set(map(tuple, instance["pairs"]))
    weights = instance["clarity_weights"]
    for k, members in groups.items():
        actual = _union(_signature(cells[v]) for v in members)
        if actual != bins[k]:
            raise ValueError("Active bin lacks an actual token provider")
        dimensions = []
        for axes in actual[:2]:
            dimensions.append(math.prod(instance["domains"][domain]["size"] for _, domain in axes
                                        if instance["domains"][domain].get("solve_axis") is not False))
        inside = set(members)
        crossing = len({(s, t) for s, t in pairs if s in inside and t not in inside})
        result[0] += (len(members) + instance["boundary_unit"] * crossing) * math.prod(dimensions)
        result[1] += len(members) * dimensions[0]
        result[2] += weights["block"] + weights["dispatch"] * (len(members) - 1) + weights["characteristic"] * len(actual[2])
    return tuple(result)


def verify(instance, certificate, *, max_candidate_transitions=None):
    """Return PASS only after a complete local cover and witness check.

    A verifier limit produces UNRESOLVED, never PASS. The returned frontier
    guarantees exact integer cost queries; native source semantics, all-Model
    component coverage, and endpoint recommendation filters are separate.
    """
    try:
        return _verify(instance, certificate, max_candidate_transitions=max_candidate_transitions)
    except RuntimeError as exc:
        return {"status": "UNRESOLVED", "frontier_complete_verified": False, "reason": str(exc)}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "FAIL", "frontier_complete_verified": False, "reason": str(exc)}


def _recover_assignment(certificate, table_index, row_index, point_index):
    assignment = {}
    stack = [(table_index, row_index, point_index)]
    while stack:
        table_index, ri, pi = stack.pop()
        table = certificate["tables"][table_index]
        row = table["rows"][ri]
        point = row["points"][pi]
        op = table["operation"]
        if op["kind"] == "cell":
            if op["cell"] in assignment:
                raise ValueError("Repeated original cell in witness")
            assignment[op["cell"]] = row["boundary_bin"]
        elif op["kind"] != "empty":
            a, b, c, d = point["from"]
            stack.append((op["left"], a, b))
            stack.append((op["right"], c, d))
    return assignment


def _normalize(instance, assignment, bins):
    """Independently derive actual unions and coalesce identical unions."""
    cells = {c["id"]: c for c in instance["cells"]}
    if set(assignment) != set(cells):
        raise ValueError("Relaxed witness omits original cells")
    groups = {}
    for v, k in assignment.items():
        if type(k) is not int or not 0 <= k < len(bins):
            raise ValueError("Invalid relaxed label")
        if _union((_signature(cells[v]), bins[k])) != bins[k]:
            raise ValueError("Relaxed label cannot accommodate original axes/tokens")
        groups.setdefault(k, []).append(v)
    index = {sig: k for k, sig in enumerate(bins)}
    normalized = {}
    for old_bin, members in groups.items():
        actual_union = _union(_signature(cells[v]) for v in members)
        if actual_union is None or _union((actual_union, bins[old_bin])) != bins[old_bin]:
            raise ValueError("Actual bin union is incompatible")
        new_bin = index[actual_union]
        for v in members:
            normalized[v] = new_bin
    # Equal new_bin indices implicitly coalesce the old groups. Re-evaluate
    # literal costs afterwards: no cached activation or boundary flag survives.
    return normalized


def _verify(instance, certificate, *, max_candidate_transitions):
    cells, original, bins, bin_index, allowed, dimensions, pairs, weights, beta = _decode(instance)
    schema = certificate["schema"]
    if schema not in ("regime-singleton-forest-mask-frontier/v1", "regime-singleton-forest-mask-frontier/v2"):
        raise ValueError("Unknown certificate format")
    subset_cover = schema == "regime-singleton-forest-mask-frontier/v2"
    if subset_cover and certificate.get("state_cover_rule") != "same-boundary-subset-mask-weak-cost/v1":
        raise ValueError("Unknown recurrence state-cover rule")
    if certificate["normalization_rule"] != "shrink-to-actual-union-then-merge-equal-unions/v1":
        raise ValueError("Unknown provider-normalization theorem")
    if certificate["status"] != "COMPLETE" or certificate["exhaustion_reason"] is not None:
        if certificate["status"] == "INCOMPLETE" and certificate["lower_bound"] != [0, 0, 0]:
            raise ValueError("Unestablished incomplete-search lower bound")
        raise RuntimeError("Optimizer has not completed its recurrence; partial original witness not certified here")
    if certificate["lower_bound"] is not None:
        raise ValueError("Unexpected complete-search lower bound")
    if certificate["instance_sha256"] != _digest(instance):
        raise ValueError("Original instance/source binding changed")
    if _json(certificate["bins"]) != _json(bins):
        raise ValueError("Compatible union-signature family changed")
    original_cost = [0, 0, 0]
    for v in sorted(cells):
        sig = original[v]
        ns, na = dimensions[bin_index[sig]]
        crossing = sum(s == v and t != v for s, t in pairs)
        original_cost[0] += (1 + beta * crossing) * ns * na
        original_cost[1] += ns
        original_cost[2] += weights["block"] + weights["characteristic"] * len(sig[2])
    original_expected = {"blocks": [[v] for v in sorted(cells)], "cost": original_cost}
    if certificate["original_representation"] != original_expected:
        raise ValueError("Retained original representation or its literal cost changed")
    nbin = len(bins)
    scopes, boundary_cells, maps = [], [], []
    generated = 0

    def rows_from(table):
        mapping = {}
        for ri, row in enumerate(table["rows"]):
            b, mask = row["boundary_bin"], row["used_bins"]
            if type(b) is not int or not -1 <= b < nbin or type(mask) is not int or not 0 <= mask < 1 << nbin:
                raise ValueError("Invalid recurrence state")
            key = b, mask
            if key in mapping or not row["points"]:
                raise ValueError("Duplicate or empty recurrence row")
            costs = [tuple(p["cost"]) for p in row["points"]]
            if any(len(c) != 2 or any(type(x) is not int or x < 0 for x in c) for c in costs):
                raise ValueError("Invalid exact recurrence cost")
            if not _antichain(costs):
                raise ValueError("Recurrence row is not a unique antichain")
            mapping[key] = ri, row
        return mapping

    def entries(table_index):
        for state, (ri, row) in maps[table_index].items():
            for pi, point in enumerate(row["points"]):
                yield state, tuple(point["cost"]), [ri, pi]

    for at, table in enumerate(certificate["tables"]):
        op = table["operation"]
        kind = op["kind"]
        if kind in ("edge", "forest"):
            left, right = op["left"], op["right"]
            if any(type(x) is not int or not 0 <= x < at for x in (left, right)):
                raise ValueError("Recurrence dependency is not an earlier table")
            if scopes[left] & scopes[right]:
                raise ValueError("Recurrence double counts original cells")
            scope = scopes[left] | scopes[right]
            crossings = {(s, t) for s, t in pairs
                         if (s in scopes[left] and t in scopes[right]) or (s in scopes[right] and t in scopes[left])}
            if kind == "edge":
                parent, child = op["parent"], op["child"]
                if boundary_cells[left] != parent or boundary_cells[right] != child:
                    raise ValueError("Incorrect exposed boundary endpoints")
                if not crossings or any({s, t} != {parent, child} for s, t in crossings):
                    raise ValueError("An internal pair was omitted from the separator")
                boundary = parent
            else:
                if crossings:
                    raise ValueError("Forest join hides a directed pair")
                boundary = None
        elif kind == "cell":
            v = op["cell"]
            if v not in cells:
                raise ValueError("Unknown original cell")
            scope, boundary = frozenset((v,)), v
        elif kind == "empty":
            scope, boundary = frozenset(), None
        else:
            raise ValueError("Unknown recurrence operation")
        mapping = rows_from(table)
        if any((b == -1) != (boundary is None) for b, _ in mapping):
            raise ValueError("Recurrence has the wrong boundary dimension")

        def candidates():
            if kind == "empty":
                yield (-1, 0), (0, 0), []
            elif kind == "cell":
                v = op["cell"]
                outside = sum(s == v and t not in cells for s, t in pairs)
                for k in allowed[v]:
                    ns, na = dimensions[k]
                    yield (k, 1 << k), ((1 + beta * outside) * ns * na, ns), []
            else:
                for a, ca, ai in entries(op["left"]):
                    for b, cb, bi in entries(op["right"]):
                        extra = 0
                        if kind == "edge" and a[0] != b[0]:
                            for source, target in crossings:
                                label = a[0] if source == op["parent"] else b[0]
                                ns, na = dimensions[label]
                                extra += beta * ns * na
                        yield ((a[0] if kind == "edge" else -1), a[1] | b[1]), (ca[0] + cb[0] + extra, ca[1] + cb[1]), [*ai, *bi]

        unmatched = {(key, tuple(point["cost"]), tuple(point["from"]))
                     for key, (_, row) in mapping.items() for point in row["points"]}
        by_boundary = {}
        for (boundary_bin, mask), (_, row) in mapping.items():
            by_boundary.setdefault(boundary_bin, {})[mask] = [tuple(p['cost']) for p in row['points']]

        def covers(key, cost):
            if not subset_cover:
                return key in mapping and any(_weak(tuple(p['cost']), cost) for p in mapping[key][1]['points'])
            boundary_bin, mask = key
            rows = by_boundary.get(boundary_bin, {})
            if (1 << mask.bit_count()) <= len(rows):
                subset = mask
                while True:
                    if any(_weak(value, cost) for value in rows.get(subset, ())):
                        return True
                    if not subset:
                        return False
                    subset = (subset - 1) & mask
            return any(other & mask == other and any(_weak(value, cost) for value in points)
                       for other, points in rows.items())

        for key, cost, source in candidates():
            generated += 1
            if max_candidate_transitions is not None and generated > max_candidate_transitions:
                raise RuntimeError("Independent recurrence-cover transition budget")
            if not covers(key, cost):
                raise ValueError("A relaxed recurrence candidate/state is uncovered")
            unmatched.discard((key, cost, tuple(source)))
        if unmatched:
            raise ValueError("A recurrence representative has no feasible transition witness")
        scopes.append(scope)
        boundary_cells.append(boundary)
        maps.append(mapping)

    root = certificate["root_table"]
    if type(root) is not int or not 0 <= root < len(maps) or scopes[root] != set(cells) or boundary_cells[root] is not None:
        raise ValueError("Root is not a closed cover of every original cell")
    frontier = certificate["frontier"]
    vectors = [tuple(point["cost"]) for point in frontier]
    if any(len(p) != 3 or any(type(x) is not int or x < 0 for x in p) for p in vectors) or not _antichain(vectors):
        raise ValueError("Output is not an exact unique cost antichain")

    def root_cost(row, point):
        clarity = weights["dispatch"] * len(cells)
        for k in range(nbin):
            if row["used_bins"] & (1 << k):
                clarity += weights["block"] - weights["dispatch"] + weights["characteristic"] * len(bins[k][2])
        return (*point["cost"], clarity)

    for row in certificate["tables"][root]["rows"]:
        for point in row["points"]:
            candidate = root_cost(row, point)
            if not any(_weak(v, candidate) for v in vectors):
                raise ValueError("An attainable relaxed root vector is not covered")

    for point in frontier:
        ri, pi = point["source"]
        if type(ri) is not int or type(pi) is not int or ri < 0 or pi < 0:
            raise ValueError("Invalid root witness index")
        row = certificate["tables"][root]["rows"][ri]
        if root_cost(row, row["points"][pi]) != tuple(point["cost"]):
            raise ValueError("Output vector is not its root witness cost")
        relaxed = _recover_assignment(certificate, root, ri, pi)
        normalized = _normalize(instance, relaxed, bins)
        # This check enforces actual token providers, exact original directed
        # pairs and active-bin costs. A missing-provider relaxation by itself
        # can never be delivered as a true feasible frontier witness.
        if _literal(instance, normalized, bins) != tuple(point["cost"]):
            raise ValueError("Relaxed frontier point failed equal-cost provider normalization")
        if point["normalized_witness_sha256"] != _digest(normalized):
            raise ValueError("Normalized original-cell witness binding changed")
        # Do not retain every state-sized assignment in diagnostic output.
    if not frontier:
        raise ValueError("Valid original input has no claimed frontier witness")
    return {"status": "PASS", "frontier_complete_verified": True,
            "instance_sha256": _digest(instance), "certificate_sha256": _digest(certificate),
            "vectors": sorted(vectors), "checked_witnesses": len(frontier),
            "checked_tables": len(maps), "checked_candidate_transitions": generated,
            "coverage_domain": "compatible union-bin assignment relaxation, with every true partition embedded",
            "token_provider_coverage": "each reported witness normalized and literally checked at equal cost",
            "source_semantics": "NOT_CHECKED_HERE", "endpoint_preference": "NOT_APPLIED_HERE",
            "original_representation_verified": True, "original_representation": original_expected}




def query(instance, certificate, *, operation, target=None, weights=None, scales=None):
    """Verified exact threshold, strict dominance, and positive-weight queries."""
    receipt = verify(instance, certificate)
    if receipt["status"] != "PASS":
        return receipt
    points = receipt["vectors"]
    answer = {"status": "PASS", "instance_sha256": receipt["instance_sha256"],
              "certificate_sha256": receipt["certificate_sha256"]}
    if operation in ("threshold", "dominance"):
        if target is None or len(target) != 3 or any(type(x) is not int for x in target):
            raise ValueError("Require an exact integer target triple")
        hits = [p for p in points if _weak(p, target) and (operation == "threshold" or tuple(p) != tuple(target))]
        return {**answer, "answer": bool(hits), "witness_cost": min(hits) if hits else None}
    if operation != "recommend":
        raise ValueError("Unknown query operation")
    names = ("work", "storage", "clarity")
    weights = weights or dict.fromkeys(names, 1)
    scales = scales or dict.fromkeys(names, 1)
    coefficients = []
    for name in names:
        weight = Fraction(str(weights[name]))
        scale = scales[name]
        if weight <= 0 or type(scale) is not int or scale <= 0:
            raise ValueError("Require positive rational weights and positive integer scales")
        coefficients.append(weight / scale)
    score = lambda p: sum(a * b for a, b in zip(coefficients, p))
    best = min(points, key=lambda p: (score(p), p))
    return {**answer, "witness_cost": best, "score": str(score(best)),
            "scope": "all cost-frontier vectors before any endpoint or timing preference"}
