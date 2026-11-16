"""Independent check of a layout curve against the Supplement's definitions.

The checker shares only the fact layer with the search (`fibre_live`,
`linked_points`, `status_axes`). It re-derives the dense regimes, the kept
fibres, the claimed inert statuses (I1, I2) and the lanes, prices every
witness block literally (Defs. "Blocks" and "Counts"), and checks that every
layout partitions the lanes into groups cut along their own periods and that
the curve is the at-most-k envelope of the exact table under the tie rule.
It does not re-prove optimality; the literal oracle tests do that on small
models.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping
from itertools import product
from math import prod

from .facts import Facts, Record, fibre_live, linked_points, live_relations, status_axes

type Point = tuple[tuple[str, int], ...]
type Fibre = tuple[str, Point]


class _Model:
    def __init__(self, declarations: Record, facts: Facts) -> None:
        self.facts = facts
        self.cells = {c["id"]: c for c in facts.cells.values()}
        self.domains = facts.domains
        self.axes = {cid: dict(status_axes(facts, c)) for cid, c in self.cells.items()}
        self.memo: dict = {}

    def dense(self) -> list[tuple[str, ...]]:
        regimes: dict[str, Record] = {}
        for cell in self.cells.values():
            ops = cell.get("operators", {})
            r = regimes.setdefault(cell["regime"], {"terminal": True, "choice": False, "law": ops.get("taste_shocks"),
                                                    "kind": repr([cell.get("stage"), cell.get("solver_identity"),
                                                                  sorted((k, v) for k, v in ops.items() if k != "taste_shocks")])})
            r["terminal"] = r["terminal"] and cell["terminal"]
            r["choice"] = r["choice"] or any(self.domains[d].get("kind") == "categorical" for d in cell["actions"].values())
        out: list[tuple[str, ...]] = [(n,) for n, r in regimes.items() if r["terminal"]]
        for kind in {r["kind"] for r in regimes.values() if not r["terminal"]}:
            names = [n for n, r in regimes.items() if not r["terminal"] and r["kind"] == kind]
            plain = sorted(n for n in names if regimes[n]["law"] is None)
            shocked = sorted(n for n in names if regimes[n]["law"] is not None)
            if plain and len(shocked) == 1 and not any(regimes[n]["choice"] for n in plain):
                out.append(tuple(sorted(plain + shocked)))
            else:
                out += [tuple(plain)] if plain else []
                out += [(n,) for n in shocked]
        return sorted(out)

    def points(self, cid: str, skip: frozenset[str] = frozenset()) -> list[Point]:
        axes = sorted((n, c) for n, c in self.axes[cid].items() if n not in skip)
        return [tuple(zip([n for n, _ in axes], combo, strict=True)) for combo in product(*(c for _, c in axes))]

    def kept(self) -> set[Fibre]:
        todo = [(cid, p) for cid, c in self.cells.items() if c.get("admissible_start") for p in self.points(cid)]
        seen = set(todo)
        while todo:
            cid, point = todo.pop()
            for relation in live_relations(self.facts, cid):
                for target in relation.get("targets", []):
                    if target == cid:
                        continue
                    for tp in linked_points(self.facts, relation, self.cells[target], dict(point)):
                        key = (target, tuple(sorted(tp.items())))
                        if key not in seen:
                            seen.add(key)
                            todo.append(key)
        return seen

    def live(self, fibre: Fibre, ordinary: frozenset[str] = frozenset()) -> frozenset[str]:
        cid, point = fibre
        return frozenset(fibre_live(self.facts, self.cells[cid], dict(point), self.memo, ordinary))

    def actions(self, cid: str) -> set[str]:
        return set(self.cells[cid]["actions"]) - set(self.facts.forced.get(cid, {}))

    def value(self, members: tuple[str, ...], fibre: Fibre, name: str) -> object:
        cid, point = fibre
        cell = self.cells[cid]
        if name in cell["states"]:
            domain = self.domains[cell["states"][name]]
            if name in dict(point):
                return domain["values"][domain["codes"].index(dict(point)[name])]
            if len(domain.get("values", [])) == 1:
                return domain["values"][0]
        first = min((c for c in self.cells.values() if c["regime"] in members and name in c["states"]),
                    key=lambda c: (c["regime"], c["period"]))
        return self.domains[first["states"][name]]["values"][0]

    def size(self, domain_id: str) -> int:
        domain = self.domains[domain_id]
        return 1 if domain.get("solve_axis") is False else domain["size"]


def literal_price(model: _Model, members: tuple[str, ...], fibres: list[Fibre]) -> tuple[int, int]:
    """Price a block from Defs. "Blocks" and "Counts" (:516-566, :843-859)."""
    cells = [model.cells[cid] for cid, _ in fibres]
    periods = {c["period"] for c in cells}
    live = set().union(*(model.live(f) for f in fibres))
    statuses = {n for cid, _ in fibres for n in model.axes[cid]}
    rows = len({c["regime"] for c in cells}) if len(members) > 1 else 1
    for name in live:
        if name in statuses:
            rows *= len({repr(model.value(members, f, name)) for f in fibres})
        else:
            rows *= max(model.size(c["states"][name]) for c in cells if name in c["states"])
    actions = set().union(*(model.actions(cid) for cid, _ in fibres))
    columns = prod(max(model.size(c["actions"][a]) for c in cells if a in c["actions"]) for a in actions)
    return len(periods) * rows * columns, len(periods) * rows


def _inert_errors(model: _Model, kept: set[Fibre], inert: frozenset[str]) -> list[str]:
    errors = []
    for cid, point in sorted(kept):
        coarse = tuple((n, c) for n, c in point if n not in inert)
        for rest in model.points(cid, frozenset(model.axes[cid]) - inert):
            part = tuple(sorted(coarse + rest))
            if (cid, part) not in kept:
                errors.append(f"I1: part {part} of a kept fibre of {cid} is not kept")
        if model.live((cid, point)) != model.live((cid, coarse), inert):
            errors.append(f"I2: fibre {cid} {point} differs from its contracted fibre")
    return errors


def check_layouts(declarations: Record, facts: Facts, result: Record, guard: Callable[[], None] = lambda: None) -> Record:
    """Return PASS or FAIL with the errors for a `search_layouts` result."""
    model = _Model(declarations, facts)
    errors: list[str] = []
    dense = model.dense()
    if [tuple(d) for d in result["dense_regimes"]] != dense:
        errors.append("dense regimes differ from Def. 'Dense model'")
    kept = model.kept()
    inert = frozenset(result["inert_statuses"])
    errors += _inert_errors(model, kept, inert)
    contracted = inert | set(result["restricted_statuses"])
    if (result["status"] == "RESTRICTED") != bool(result["restricted_statuses"]):
        errors.append("status does not say whether the family is restricted")
    lanes: dict[int, dict[tuple, list[Fibre]]] = {i: defaultdict(list) for i in range(len(dense))}
    home = {r: i for i, members in enumerate(dense) for r in members}
    for cid, point in sorted(kept):
        regime = model.cells[cid]["regime"]
        lanes[home[regime]][(regime, tuple((n, c) for n, c in point if n not in contracted))].append((cid, point))
    total = sum(prod(model.size(d) for d in c["states"].values()) * prod(model.size(d) for d in c["actions"].values())
                for c in model.cells.values())
    if result["dense_count"] != total:
        errors.append("dense count differs from ev^0")

    def layout_cost(layout: list[Record], where: str) -> tuple[int, int, int]:
        used: dict[int, list[tuple]] = defaultdict(list)
        evaluations = storage = blocks = 0
        for group in layout:
            guard()
            members = tuple(group["dense_regime"])
            index = dense.index(members) if members in dense else None
            if index is None:
                errors.append(f"{where}: unknown dense regime {members}")
                continue
            keys = [(lane["regime"], tuple(sorted((n, c) for n, c in lane["statuses"].items()))) for lane in group["lanes"]]
            used[index] += keys
            owned = [f for key in keys for f in lanes[index].get(key, [])]
            active = sorted({model.cells[cid]["period"] for cid, _ in owned})
            cuts = group["cuts"]
            if active != group["periods"] or cuts[0] != 0 or cuts[-1] != len(active) or sorted(set(cuts)) != cuts:
                errors.append(f"{where}: cuts do not partition the group's own periods")
                continue
            for (start, stop), block in zip(zip(cuts, cuts[1:], strict=False), group["blocks"], strict=True):
                fibres = [f for f in owned if model.cells[f[0]]["period"] in active[start:stop]]
                ev, st = literal_price(model, members, fibres)
                if (ev, st) != (block["evaluations"], block["storage"]):
                    errors.append(f"{where}: block {members} periods {active[start:stop]} priced "
                                  f"{block['evaluations']}/{block['storage']}, literal {ev}/{st}")
                evaluations, storage, blocks = evaluations + ev, storage + st, blocks + 1
        for index in range(len(dense)):
            if sorted(used[index]) != sorted(lanes[index]):
                errors.append(f"{where}: groups do not partition the lanes of {dense[index]}")
        return evaluations, storage, blocks

    exact = {}
    for entry in result["exact"]:
        ev, st, n = layout_cost(entry["layout"], f"w({entry['blocks']})")
        if (ev, st, n) != (entry["evaluations"], entry["storage"], entry["blocks"]):
            errors.append(f"w({entry['blocks']}): totals differ from its blocks")
        exact[n] = (ev, st)
    previous = None
    for point in result["curve"]:
        k = point["max_blocks"]
        ev, st, n = layout_cost(point["layout"], f"ev*({k})")
        if (ev, st, n) != (point["evaluations"], point["storage"], point["blocks"]) or n > k:
            errors.append(f"ev*({k}): totals differ from its blocks or exceed the budget")
        envelope = min(((e, m, s) for m, (e, s) in exact.items() if m <= k), default=None)
        if envelope != (point["evaluations"], point["blocks"], point["storage"]):
            errors.append(f"ev*({k}) is not the envelope of w(n), n <= {k}, under the tie rule")
        if previous is not None and point["evaluations"] > previous:
            errors.append(f"ev*({k}) increases")
        previous = point["evaluations"]
    return {"status": "FAIL" if errors else "PASS", "errors": errors}
