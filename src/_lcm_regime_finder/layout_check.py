"""Independent check of a layout curve against the Supplement's definitions.

The checker shares only the fact layer with the search (`fibre_live`,
`linked_points`, `status_axes`). It re-derives the dense regimes, the kept
fibres, the claimed inert statuses (I1, I2) and the lanes, prices every
witness block literally (Defs. "Blocks" and "Counts"), and checks that every
layout partitions the lanes into groups cut along their own periods and that
the curve is the at-most-k envelope of the exact table under the tie rule.
What the result exports is checked as well: the lane listing, and each
block's periods, states, actions, excluded actions and fibres, whose order
alone is free.
It does not re-prove optimality; the literal oracle tests do that on small
models.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping
from itertools import product
from math import prod

from .facts import Facts, Record, fibre_live, linked_points, live_relations, status_axes
from .records import canonical

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
            # Members in their order, the whole exit and the renegotiated states
            # with the nodes of their grids are part of a regime's kind.
            own_lists = sorted((n, self.domains[cell["states"][n]].get("values", cell["states"][n]))
                               for n in cell.get("renegotiated_states", []))
            r = regimes.setdefault(cell["regime"], {"terminal": True, "choice": False, "law": ops.get("taste_shocks"), "grids": {},
                                                    "kind": repr([cell.get("stage"), cell.get("solver_identity"),
                                                                  sorted((k, v) for k, v in ops.items() if k != "taste_shocks"),
                                                                  cell.get("members"), cell.get("exit_identity"),
                                                                  own_lists])})
            r["terminal"] = r["terminal"] and cell["terminal"]
            r["choice"] = r["choice"] or any(self.domains[d].get("kind") == "categorical" for d in cell["actions"].values())
            for a, d in cell["actions"].items():
                r["grids"].setdefault(a, set()).add(repr((self.domains[d].get("kind"), self.domains[d].get("values", d))))
        # An action that two nonterminal regimes of one kind declare on unequal
        # grids is part of the kind of every regime that has it.
        living = [r for r in regimes.values() if not r["terminal"]]
        unequal = [tuple((a, tuple(sorted(g))) for a, g in sorted(r["grids"].items())
                         if any(o["kind"] == r["kind"] and o["grids"].get(a, g) != g for o in living)) for r in living]
        for r, own in zip(living, unequal, strict=True):
            r["kind"] = repr((r["kind"], own))
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
    # One stored value and one action value per member of a collective regime.
    rows *= max(len(c.get("members") or ()) for c in cells) or 1
    for name in live:
        if name in statuses:
            rows *= len({repr(model.value(members, f, name)) for f in fibres})
        else:
            rows *= max(model.size(c["states"][name]) for c in cells if name in c["states"])
    actions = set().union(*(model.actions(cid) for cid, _ in fibres))
    columns = prod(max(model.size(c["actions"][a]) for c in cells if a in c["actions"]) for a in actions)
    return len(periods) * rows * columns, len(periods) * rows


def literal_block_payload(*, model: _Model, members: tuple[str, ...], fibres: list[Fibre]) -> Record:
    """Derive the block fields the result exports from the kept fibres, with the checker's own code."""
    cells = [model.cells[cid] for cid, _ in fibres]
    periods = sorted({c["period"] for c in cells})
    present = {c["regime"] for c in cells}
    live = set().union(*(model.live(f) for f in fibres))
    statuses = {name for cid, _ in fibres for name in model.axes[cid]}
    states: Record = {"regime": sorted(present)} if len(members) > 1 else {}
    for name in sorted(live):
        if name in statuses:
            states[name] = sorted({model.value(members, f, name) for f in fibres}, key=repr)
        else:
            states[name] = max(model.size(c["states"][name]) for c in cells if name in c["states"])
    action_names = set().union(*(model.actions(cid) for cid, _ in fibres))
    actions = {name: max(model.size(c["actions"][name]) for c in cells if name in c["actions"])
               for name in sorted(action_names)}
    hull_actions = {name for c in model.cells.values()
                    if c["regime"] in present and c["period"] in periods for name in c["actions"]}
    excluded = {}
    for cell in sorted(model.cells.values(), key=lambda c: (c["regime"], c["period"])):
        if cell["regime"] in members:
            for name, did in cell["actions"].items():
                values = model.domains[did].get("values")
                if name not in action_names | hull_actions and isinstance(values, list) and len(values) > 1:
                    excluded.setdefault(name, values[1:])
    described_fibres = [
        {"period": model.cells[cid]["period"], "regime": model.cells[cid]["regime"],
         "statuses": {name: model.value(members, (cid, point), name) for name in sorted(statuses)}}
        for cid, point in fibres
    ]
    return {"periods": periods, "states": states, "actions": actions,
            "excluded_actions": excluded, "fibres": described_fibres}


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


def lowest_bound(model: _Model, dense: list[tuple[str, ...]], kept: set[Fibre]) -> tuple[int, int]:
    """lem:lowest (vi) and (iv) by breadth-first components: (LB, sum of e(phi))."""
    home = {r: members for members in dense for r in members}
    lower = finest = 0
    for cid in sorted({cid for cid, _ in kept}):
        fibres = sorted(f for f in kept if f[0] == cid)
        members = home[model.cells[cid]["regime"]]
        price = {f: literal_price(model, members, [f])[0] for f in fibres}
        finest += sum(price.values())
        unseen = set(fibres)
        while unseen:
            frontier = [unseen.pop()]
            largest = price[frontier[0]]
            while frontier:
                a = frontier.pop()
                for b in sorted(unseen):
                    shared = (model.live(a) | model.live(b)) & set(model.axes[cid])
                    if {n: c for n, c in a[1] if n in shared} == {n: c for n, c in b[1] if n in shared}:
                        unseen.discard(b)
                        frontier.append(b)
                        largest = max(largest, price[b])
            lower += largest
    return lower, finest


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
    for members in dense:
        for name in sorted(inert):
            seen = {repr(model.domains[c["states"][name]].get("values")) if name in c["states"] else None
                    for c in model.cells.values() if c["regime"] in members}
            if len(seen) > 1:
                errors.append(f"I1: {name} has different grids across the members of {list(members)}")
    contracted = inert | set(result["restricted_statuses"])
    if (result["status"] == "RESTRICTED") != bool(result["restricted_statuses"]):
        errors.append("status does not say whether the family is restricted")
    lanes: dict[int, dict[tuple, list[Fibre]]] = {i: defaultdict(list) for i in range(len(dense))}
    home = {r: i for i, members in enumerate(dense) for r in members}
    for cid, point in sorted(kept):
        regime = model.cells[cid]["regime"]
        lanes[home[regime]][(regime, tuple((n, c) for n, c in point if n not in contracted))].append((cid, point))
    expected_lanes = [[{"regime": regime, "statuses": dict(point)}
                       for regime, point in sorted(lanes[index])] for index in range(len(dense))]
    if result.get("lanes") != expected_lanes:
        errors.append("reported lanes differ from the independently derived lanes")
    if result["original_count"] != sum(prod(model.size(d) for d in c["states"].values())
                                       * prod(model.size(d) for d in c["actions"].values())
                                       * (len(c.get("members") or ()) or 1)
                                       for c in model.cells.values()):
        errors.append("original count differs from the model as written")
    ev0 = 0
    for members in dense:
        member_cells = [c for c in model.cells.values() if c["regime"] in members]
        values: dict[str, set[str]] = {}
        sizes: dict[tuple[str, str], int] = {}
        for c in member_cells:
            for kind in ("states", "actions"):
                for name, did in c[kind].items():
                    domain = model.domains[did]
                    if domain.get("solve_axis") is False:
                        continue
                    if kind == "states" and isinstance(domain.get("values"), list):
                        values.setdefault(name, set()).update(map(repr, domain["values"]))
                    else:
                        sizes[kind, name] = max(sizes.get((kind, name), 0), model.size(did))
        label = (len(members) if len(members) > 1 else 1) * (max(len(c.get("members") or ()) for c in member_cells) or 1)
        periods = len({c["period"] for c in member_cells})
        ev0 += periods * label * prod(len(v) for v in values.values()) * prod(sizes.values())
    if result["dense_count"] != ev0:
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
                payload = literal_block_payload(model=model, members=members, fibres=fibres)
                for field in ("periods", "states", "actions", "excluded_actions"):
                    if block.get(field) != payload[field]:
                        errors.append(f"{where}: block {members} periods {active[start:stop]} "
                                      f"{field} differ from the independently derived block")
                reported_fibres = block.get("fibres")
                if (not isinstance(reported_fibres, list)
                        or sorted(map(canonical, reported_fibres)) != sorted(map(canonical, payload["fibres"]))):
                    errors.append(f"{where}: block {members} periods {active[start:stop]} "
                                  "fibres differ from the independently kept fibres")
                if (ev, st) != (block["evaluations"], block["storage"]):
                    errors.append(f"{where}: block {members} periods {active[start:stop]} priced "
                                  f"{block['evaluations']}/{block['storage']}, literal {ev}/{st}")
                evaluations, storage, blocks = evaluations + ev, storage + st, blocks + 1
        for index in range(len(dense)):
            if sorted(used[index]) != sorted(lanes[index]):
                errors.append(f"{where}: groups do not partition the lanes of {dense[index]}")
        return evaluations, storage, blocks

    exact = {}
    lower, finest = lowest_bound(model, dense, kept)
    if (result["lower_bound"], result["finest_count"]) != (lower, finest):
        errors.append("lem:lowest (vi) bound or the finest count differs")
    restricted = result["status"] == "RESTRICTED"
    for index, point in enumerate(result["curve"]):
        if point["evaluations"] < lower:
            errors.append(f"curve point k={point['max_blocks']} lies below the lem:lowest (vi) bound")
        expected = not restricted or index == 0 or point["evaluations"] == lower
        if point["exact"] != expected:
            errors.append(f"curve point k={point['max_blocks']} misstates whether it is exact")
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
