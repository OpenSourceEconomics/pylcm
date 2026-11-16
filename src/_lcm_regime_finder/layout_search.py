"""The layout search of the Supplement (lcm-paper, paper/sections/supplement-choice.tex).

Line numbers refer to that file at lcm-paper d28955bd.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from math import prod

from .facts import Facts, FactsBackendName, Record, fibre_live, linked_points, live_relations, status_axes
from .records import canonical


def regime_kinds(declarations: Record) -> dict[str, Record]:
    """Return each original regime's pooling attributes.

    A regime is terminal if none of its cells has a successor. Its kind is its
    stage, solver and operators (aggregator, certainty equivalent, Pareto
    objective); its law is its taste-shock declaration. It has a discrete choice
    if it has a categorical action coordinate.
    """
    domains = declarations["domains"]
    kinds: dict[str, Record] = {}
    for cell in declarations["cells"]:
        operators = cell.get("operators", {})
        key = canonical([cell.get("stage"), cell.get("solver_identity"),
                         {k: v for k, v in operators.items() if k != "taste_shocks"}])
        record = kinds.setdefault(cell["regime"], {
            "terminal": True, "key": key, "taste_shocks": operators.get("taste_shocks"),
            "discrete_choice": False})
        if record["key"] != key:
            raise ValueError(f"Regime {cell['regime']} changes its kind between periods")
        record["terminal"] = record["terminal"] and cell["terminal"]
        record["discrete_choice"] = record["discrete_choice"] or any(
            domains[did].get("kind") == "categorical" for did in cell["actions"].values())
    return kinds


def dense_regimes(kinds: Mapping[str, Record]) -> list[tuple[str, ...]]:
    """Pool the nonterminal regimes of each kind by Def. "Dense model" (:126-146).

    (i) Regimes with equal laws pool. Only the degenerate law (no taste shocks)
    is shown equal across regimes: a shocked regime's scale is a parameter of its
    own. (ii) A shocked regime always has a discrete action, so no regime changes
    its law. (iii) If the degenerate regimes are without discrete choice and
    exactly one shocked regime of the kind exists, the two pool. Terminal regimes
    stay regimes of their own (:128, :131).
    """
    groups: list[tuple[str, ...]] = []
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name, kind in kinds.items():
        if kind["terminal"]:
            groups.append((name,))
        else:
            by_kind[kind["key"]].append(name)
    for names in by_kind.values():
        degenerate = sorted(n for n in names if kinds[n]["taste_shocks"] is None)
        shocked = sorted(n for n in names if kinds[n]["taste_shocks"] is not None)
        if degenerate and len(shocked) == 1 and not any(kinds[n]["discrete_choice"] for n in degenerate):
            groups.append(tuple(sorted([*degenerate, *shocked])))
            continue
        if degenerate:
            groups.append(tuple(degenerate))
        groups.extend((n,) for n in shocked)
    return sorted(groups)


type Point = tuple[tuple[str, int], ...]
type FibreKey = tuple[str, Point]
type StatusValue = int | float | str | bool


@dataclass(frozen=True)
class Choice:
    """A partial layout: its counts and its groups as (dense index, lane mask, cuts)."""

    evaluations: int
    storage: int
    groups: tuple[tuple[int, int, tuple[int, ...]], ...] = ()

    @property
    def blocks(self) -> int:
        return sum(len(cuts) - 1 for _, _, cuts in self.groups)

    @property
    def key(self) -> tuple:
        return tuple(sorted(self.groups))


def _plus(left: Choice, right: Choice) -> Choice:
    return Choice(left.evaluations + right.evaluations, left.storage + right.storage,
                  left.groups + right.groups)


def _lexmin(items: Iterable[Choice | None], key: Callable[[Choice], tuple]) -> Choice | None:
    best = None
    for item in items:
        if item is not None and (best is None or key(item) < key(best)):
            best = item
    return best


def _exact_best(items: Iterable[Choice | None]) -> Choice | None:
    """Lexicographic minimum of (evaluations, storage) at one block count (:923-959)."""
    return _lexmin(items, lambda c: (c.evaluations, c.storage, c.key))


def _at_most_best(items: Iterable[Choice | None]) -> Choice | None:
    """The tie rule: evaluations, then blocks, then storage (Prop. (ii))."""
    return _lexmin(items, lambda c: (c.evaluations, c.blocks, c.storage, c.key))


class LayoutProblem:
    """The kept fibres, lanes and block prices of a model's dense regimes.

    Fibres are (cell, full status point); a cell is an original regime-period.
    Kept fibres are those reachable from the initial cells' points along links
    (Def. "Statuses, fibres, lanes, links", :181-229). Each fibre's coordinates
    are those of the least profile (`fibre_live`). Lanes contract the statuses
    that lem:inert certifies (:1737-1889); if a dense regime still has more lanes
    than `limits["lanes_per_regime"]`, further statuses are contracted and the
    layouts form a restricted family whose counts are upper bounds
    (Remark "Restricted and heuristic searches", :2005-2021).
    """

    def __init__(self, declarations: Record, facts: Facts, limits: Mapping, guard: Callable[[], None]) -> None:
        self.facts, self.guard = facts, guard
        self.cells = {c["id"]: c for c in facts.cells.values()}
        self.domains = facts.domains
        self.kinds = regime_kinds(declarations)
        self.dense = dense_regimes(self.kinds)
        self.axes = {cid: dict(status_axes(facts, cell)) for cid, cell in self.cells.items()}
        self._check_marked_links()
        self.kept = self._kept_fibres()
        self.memo: dict = {}
        member_test = self._inert_statuses()
        # (I1) takes the parts of a fibre over the dense regime's grid (:1743-1747).
        # The test above takes them over each member's own grid, so a status
        # whose grid differs across members is contracted as a restriction only:
        # a layout of the contracted lanes is a layout of the full family, so its
        # count is an upper bound there.
        mixed = {name for name in member_test if self._grid_differs(name)}
        self.inert = member_test - mixed
        limit = limits.get("lanes_per_regime", 12)
        self.contracted = set(member_test)
        self.restricted: list[str] = sorted(mixed)
        while True:
            self.lanes = self._lanes(frozenset(self.contracted))
            crowded = [i for i, lanes in enumerate(self.lanes) if len(lanes) > limit]
            if not crowded:
                break
            candidates = sorted({name for i in crowded for (regime, point) in self.lanes[i] for name, _ in point},
                                key=lambda n: (-max(len(axes.get(n, ())) for axes in self.axes.values()), n))
            self.contracted.add(candidates[0])
            self.restricted.append(candidates[0])
        self.notes: set[str] = set()
        self.fill = self._fill_values()
        self._lane_fibres = [self._fibres_by_lane(i) for i in range(len(self.dense))]
        self._prices: dict = {}

    def _check_marked_links(self) -> None:
        """Refuse a within-date link inside one dense regime: (A1) would bind (:540)."""
        home = {r: i for i, members in enumerate(self.dense) for r in members}
        for relation in live_relations(self.facts):
            source = self.cells[relation["source"]]
            for target in relation.get("targets", []):
                other = self.cells[target]
                if target != source["id"] and other["period"] == source["period"] and home[other["regime"]] == home[source["regime"]]:
                    msg = (f"Relation {relation['id']} links two fibres of one date inside one dense regime; "
                           "condition (A1) of the Supplement is not implemented.")
                    raise NotImplementedError(msg)

    def points(self, cid: str, ordinary: frozenset[str] = frozenset()) -> list[dict[str, int]]:
        axes = [(n, c) for n, c in self.axes[cid].items() if n not in ordinary]
        return [dict(zip([n for n, _ in axes], combo, strict=True)) for combo in product(*(c for _, c in axes))]

    def _kept_fibres(self) -> set[FibreKey]:
        """Fibres reachable from every point of the initial cells along links."""
        frontier = [(cid, tuple(sorted(p.items()))) for cid, cell in self.cells.items()
                    if cell.get("admissible_start") for p in self.points(cid)]
        kept = set(frontier)
        while frontier:
            self.guard()
            cid, point = frontier.pop()
            for relation in live_relations(self.facts, cid):
                for target in relation.get("targets", []):
                    if target == cid:
                        continue
                    for target_point in linked_points(self.facts, relation, self.cells[target], dict(point)):
                        key = (target, tuple(sorted(target_point.items())))
                        if key not in kept:
                            kept.add(key)
                            frontier.append(key)
        return kept

    def live(self, fibre: FibreKey, ordinary: frozenset[str] = frozenset()) -> frozenset[str]:
        cid, point = fibre
        return frozenset(fibre_live(self.facts, self.cells[cid], dict(point), self.memo, ordinary))

    def kept_actions(self, cid: str) -> frozenset[str]:
        """The least action coordinates: every action that is not forced (:416-417)."""
        return frozenset(self.cells[cid]["actions"]) - set(self.facts.forced.get(cid, {}))

    def _grid_differs(self, name: str) -> bool:
        """Whether two members of a dense regime give `name` different grids, or one lacks it."""
        for members in self.dense:
            grids = {repr(self.domains[c["states"][name]].get("values")) if name in c["states"] else None
                     for c in self.cells.values() if c["regime"] in members}
            if len(grids) > 1:
                return True
        return False

    def _inert_statuses(self) -> frozenset[str]:
        """Return the largest status set this test certifies inert (lem:inert).

        (I2) every part of a contracted fibre has the coarse fibre's live set, and
        (I1) every part of a kept contracted fibre is kept. (I3) holds because a
        link fixes only the statuses it carries, and (I4) because no dense regime
        holds a within-date link. (I5): pylcm's taste shocks are degenerate or
        iid, so (A2) holds on every block.
        """
        names = {n for axes in self.axes.values() for n in axes}
        candidate = set(names)
        while candidate:
            self.guard()
            ordinary = frozenset(candidate)
            implicated: set[str] = set()
            for cid, point in sorted(self.kept):
                coarse = tuple((n, c) for n, c in point if n not in ordinary)
                parts = [tuple(sorted({**dict(coarse), **v}.items()))
                         for v in self.points(cid, frozenset(self.axes[cid]) - ordinary)]
                if any((cid, part) not in self.kept for part in parts):
                    implicated |= {n for n in ordinary if n in self.axes[cid]}
                    continue
                if self.live((cid, point)) != self.live((cid, coarse), ordinary):
                    for name in ordinary & set(self.axes[cid]):
                        for code in self.axes[cid][name]:
                            other = tuple(sorted({**dict(point), name: code}.items()))
                            if self.live((cid, other)) != self.live((cid, point)):
                                implicated.add(name)
                    if not implicated & set(self.axes[cid]):
                        implicated.add(min(ordinary & set(self.axes[cid])))
            if not implicated:
                return frozenset(candidate)
            candidate -= implicated
        return frozenset()

    def _lanes(self, contracted: frozenset[str]) -> list[list[tuple[str, Point]]]:
        home = {r: i for i, members in enumerate(self.dense) for r in members}
        lanes: list[set] = [set() for _ in self.dense]
        for cid, point in self.kept:
            regime = self.cells[cid]["regime"]
            lanes[home[regime]].add((regime, tuple((n, c) for n, c in point if n not in contracted)))
        return [sorted(lane) for lane in lanes]

    def _fibres_by_lane(self, index: int) -> list[dict[int, list[FibreKey]]]:
        position = {lane: i for i, lane in enumerate(self.lanes[index])}
        result: list[dict[int, list[FibreKey]]] = [defaultdict(list) for _ in self.lanes[index]]
        for cid, point in sorted(self.kept):
            cell = self.cells[cid]
            lane = (cell["regime"], tuple((n, c) for n, c in point if n not in self.contracted))
            if lane in position:
                result[position[lane]][cell["period"]].append((cid, point))
        return result

    def _fill_values(self) -> dict[tuple[int, str], StatusValue]:
        """A member without a status holds it at the first label of its grid (fill rule)."""
        fill: dict[tuple[int, str], StatusValue] = {}
        for index, members in enumerate(self.dense):
            for cell in sorted((c for c in self.cells.values() if c["regime"] in members),
                               key=lambda c: (c["regime"], c["period"])):
                for name, did in cell["states"].items():
                    if self.domains[did].get("kind") == "categorical" and self.domains[did].get("values"):
                        first = self.domains[did]["values"][0]
                        fill.setdefault((index, name), first)
                        if fill[index, name] != first:
                            self.notes.add(f"Members of dense regime {list(members)} start the grid of {name} "
                                           f"with different labels; the fill value is {fill[index, name]!r}.")
        return fill

    def _value(self, index: int, fibre: FibreKey, name: str) -> StatusValue:
        cid, point = fibre
        cell = self.cells[cid]
        codes = dict(point)
        if name in cell["states"]:
            domain = self.domains[cell["states"][name]]
            if name in codes:
                return domain["values"][domain["codes"].index(codes[name])]
            if domain.get("kind") == "categorical" and len(domain.get("values", [])) == 1:
                return domain["values"][0]
        return self.fill[(index, name)]

    def price(self, index: int, fibres: list[FibreKey]) -> Record:
        """Price one block (Defs. "Blocks" :516-566 and "Counts" :843-859).

        The block keeps the union of its fibres' coordinates. A status is
        charged by the values it takes on the block's fibres in any period; an
        ordinary coordinate by its whole grid; the regime label of a pooled dense
        regime by the regimes present. No action value is excluded: pylcm
        establishes no infeasible values, so N is empty.
        """
        cells = [self.cells[cid] for cid, _ in fibres]
        periods = sorted({c["period"] for c in cells})
        keeps = set().union(*(self.live(f) for f in fibres))
        actions = set().union(*(self.kept_actions(cid) for cid, _ in fibres))
        statuses = {n for cid, _ in fibres for n in self.axes[cid]}
        grids: dict[str, list | int] = {}
        if len(self.dense[index]) > 1:
            grids["regime"] = sorted({c["regime"] for c in cells})
        for name in sorted(keeps):
            if name in statuses:
                grids[name] = sorted({self._value(index, f, name) for f in fibres}, key=repr)
            else:
                grids[name] = max(self._size(c["states"][name]) for c in cells if name in c["states"])
        action_grids = {name: max(self._size(c["actions"][name]) for c in cells if name in c["actions"])
                        for name in sorted(actions)}
        rows = prod([len(g) if isinstance(g, list) else g for g in grids.values()])
        storage = len(periods) * rows
        # Infeasible values (Supplement :289-294) under the fill rule: at a member
        # without action j every value but the first is infeasible. A block in
        # whose hull no member has j excludes them, so j is not among its actions.
        present = {c["regime"] for c in cells}
        excluded = {}
        for c in sorted(self.cells.values(), key=lambda c: (c["regime"], c["period"])):
            if c["regime"] in self.dense[index]:
                for name, did in c["actions"].items():
                    values = self.domains[did].get("values")
                    if name not in actions and isinstance(values, list) and len(values) > 1:
                        excluded.setdefault(name, values[1:])
        excluded = {n: v for n, v in excluded.items()
                    if not any(n in c["actions"] for c in self.cells.values()
                               if c["regime"] in present and c["period"] in periods)}
        return {"periods": periods, "states": grids, "actions": action_grids, "excluded_actions": excluded,
                "evaluations": storage * prod(action_grids.values()), "storage": storage}

    def select(self, spec: Record) -> tuple[int, list[FibreKey]]:
        """Return the dense regime and the kept fibres a block specification selects."""
        regimes, periods = set(spec["regimes"]), set(spec["periods"])
        allowed = spec.get("statuses", {})
        homes = {i for i, members in enumerate(self.dense) if regimes & set(members)}
        if len(homes) != 1 or not regimes <= set(self.dense[next(iter(homes))]):
            raise ValueError(f"A block must lie in one dense regime: {sorted(regimes)}")
        index = homes.pop()
        fibres = [(cid, point) for cid, point in sorted(self.kept)
                  if self.cells[cid]["regime"] in regimes and self.cells[cid]["period"] in periods
                  and all(self._value(index, (cid, point), name) in values for name, values in allowed.items())]
        if not fibres:
            raise ValueError(f"The block selects no kept fibre: {spec}")
        return index, fibres

    def _size(self, domain_id: str) -> int:
        domain = self.domains[domain_id]
        return 1 if domain.get("solve_axis") is False else domain["size"]

    def original_count(self) -> int:
        """The count of the model as written: each original regime on its own grids."""
        return sum(prod(self._size(d) for d in c["states"].values()) * prod(self._size(d) for d in c["actions"].values())
                   for c in self.cells.values())

    def dense_count(self) -> int:
        """ev^0 (:851-859): each dense regime charges its whole grid in every active period.

        The grid is the regime label over all members, each state's union of
        values (statuses) or largest grid (ordinary), and each action's largest grid.
        """
        total = 0
        for members in self.dense:
            cells = [c for c in self.cells.values() if c["regime"] in members]
            rows = len(members) if len(members) > 1 else 1
            for kind in ("states", "actions"):
                for name in sorted({n for c in cells for n in c[kind]}):
                    domains = [self.domains[c[kind][name]] for c in cells if name in c[kind]]
                    if all(d.get("solve_axis") is False for d in domains):
                        continue
                    if kind == "states" and all(isinstance(d.get("values"), list) for d in domains):
                        rows *= len({repr(v) for d in domains for v in d["values"]})
                    else:
                        rows *= max(self._size(c[kind][name]) for c in cells if name in c[kind])
            total += len({c["period"] for c in cells}) * rows
        return total


class _RegimeSearch:
    """The recurrences f and h of Def. "The search" (:923-959) for one dense regime."""

    def __init__(self, problem: LayoutProblem, index: int) -> None:
        self.problem, self.index = problem, index
        self.fibres = problem._lane_fibres[index]
        self.size = len(self.fibres)
        self._active: dict[int, tuple[int, ...]] = {}
        self._cost: dict[tuple[int, int, int], tuple[int, int]] = {}
        self._interval: dict[tuple[int, int], Choice | None] = {}
        self._subset: dict[tuple[int, int], Choice | None] = {}

    def active(self, mask: int) -> tuple[int, ...]:
        if mask not in self._active:
            self._active[mask] = tuple(sorted({t for i in range(self.size) if mask >> i & 1 for t in self.fibres[i]}))
        return self._active[mask]

    def block(self, mask: int, start: int, stop: int) -> list[FibreKey]:
        periods = self.active(mask)[start:stop]
        return [f for i in range(self.size) if mask >> i & 1 for t in periods for f in self.fibres[i].get(t, ())]

    def cost(self, mask: int, start: int, stop: int) -> tuple[int, int]:
        key = (mask, start, stop)
        if key not in self._cost:
            self.problem.guard()
            priced = self.problem.price(self.index, self.block(mask, start, stop))
            self._cost[key] = (priced["evaluations"], priced["storage"])
        return self._cost[key]

    def interval(self, mask: int, blocks: int) -> Choice | None:
        """f: the best cut of one group's periods into `blocks` intervals."""
        key = (mask, blocks)
        if key not in self._interval:
            length = len(self.active(mask))
            suffix: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, ...]] | None] = {}
            for start in range(length, -1, -1):
                for remaining in range(blocks + 1):
                    if start == length:
                        suffix[start, remaining] = ((0, 0), (length,)) if remaining == 0 else None
                        continue
                    if remaining == 0 or remaining > length - start:
                        suffix[start, remaining] = None
                        continue
                    best = None
                    for stop in range(start + 1, length - remaining + 2):
                        tail = suffix[stop, remaining - 1]
                        if tail is None:
                            continue
                        head = self.cost(mask, start, stop)
                        candidate = ((head[0] + tail[0][0], head[1] + tail[0][1]), (start, *tail[1]))
                        if best is None or candidate < best:
                            best = candidate
                    suffix[start, remaining] = best
            found = suffix[0, blocks]
            self._interval[key] = None if found is None else Choice(*found[0], ((self.index, mask, found[1]),))
        return self._interval[key]

    def subset(self, mask: int, blocks: int) -> Choice | None:
        """h: the best partition of the lanes in `mask` into groups with `blocks` blocks."""
        key = (mask, blocks)
        if key in self._subset:
            return self._subset[key]
        if mask == 0:
            return Choice(0, 0) if blocks == 0 else None
        if blocks <= 0:
            return None
        anchor = mask & -mask
        best = None
        group = mask
        while group:
            if group & anchor:
                self.problem.guard()
                rest = mask ^ group
                for count in ((blocks,) if rest == 0 else range(1, blocks)):
                    head = self.interval(group, count)
                    tail = self.subset(rest, blocks - count)
                    if head is not None and tail is not None:
                        best = _exact_best([best, _plus(head, tail)])
            group = (group - 1) & mask
        self._subset[key] = best
        return best

    def floor(self) -> int | None:
        """lem:lowest(iv): the least count of the family, every group cut at every period."""
        if self.size > 8:
            return None
        best = None
        for grouping in _partitions(self.size):
            total = 0
            for group in grouping:
                mask = sum(1 << i for i in group)
                total += sum(self.cost(mask, i, i + 1)[0] for i in range(len(self.active(mask))))
            best = total if best is None else min(best, total)
        return best


def _partitions(n: int) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Every set partition of range(n) once (restricted growth)."""
    groups: list[list[int]] = []

    def visit(item: int) -> Iterator[tuple[tuple[int, ...], ...]]:
        if item == n:
            yield tuple(tuple(g) for g in groups)
            return
        for g in groups:
            g.append(item)
            yield from visit(item + 1)
            g.pop()
        groups.append([item])
        yield from visit(item + 1)
        groups.pop()

    yield from visit(0)


DEVIATIONS = (
    "Taste-shock laws are not compared across regimes: shocked regimes stay dense regimes of their own.",
    "The regime label is live at every fibre of a pooled dense regime.",
)


def search_layouts(
    declarations: Record,
    facts: Facts,
    limits: Mapping,
    guard: Callable[[], None],
    price: Iterable[Record] = (),
) -> Record:
    """Return the curve of the Supplement's family: w(n), ev*(k) and their layouts.

    `price` lists further blocks to price, each as original `regimes`, `periods`
    and optionally `statuses` (name -> allowed values); a block takes the kept
    fibres they select.
    """
    problem = LayoutProblem(declarations, facts, limits, guard)
    priced = [problem.price(*problem.select(spec)) for spec in price]
    searches = [_RegimeSearch(problem, i) for i in range(len(problem.dense))]
    first = len(searches)
    largest = limits.get("max_blocks", first + 12)
    table: dict[int, Choice] = {0: Choice(0, 0)}
    for search in searches:
        full = (1 << search.size) - 1
        cap = min(largest - first + 1, sum(len(f) for lane in search.fibres for f in [lane]))
        entries = {n: search.subset(full, n) for n in range(1, cap + 1)}
        merged: dict[int, Choice] = {}
        for left_n, left in table.items():
            for right_n, right in entries.items():
                if right is None or left_n + right_n > largest:
                    continue
                candidate = _plus(left, right)
                incumbent = merged.get(left_n + right_n)
                if incumbent is None or _exact_best([incumbent, candidate]) is candidate:
                    merged[left_n + right_n] = candidate
        table = merged
    floors = [s.floor() for s in searches]
    floor = None if None in floors else sum(f for f in floors if f is not None)
    exact = [{"blocks": n, "evaluations": c.evaluations, "storage": c.storage, "layout": _layout(problem, searches, c)}
             for n, c in sorted(table.items())]
    curve = []
    for k in range(first, largest + 1):
        best = _at_most_best(c for n, c in table.items() if n <= k)
        if best is not None:
            curve.append({"max_blocks": k, "blocks": best.blocks, "evaluations": best.evaluations,
                          "storage": best.storage, "layout": _layout(problem, searches, best)})
    flat = next((p["max_blocks"] for p in curve if floor is not None and p["evaluations"] == floor), None)
    return {
        "schema": "supplement-layout-curve/v1",
        "status": "RESTRICTED" if problem.restricted else "COMPLETE",
        "dense_regimes": [list(members) for members in problem.dense],
        "inert_statuses": sorted(problem.inert),
        "restricted_statuses": problem.restricted,
        "lanes": [[{"regime": r, "statuses": dict(p)} for r, p in lanes] for lanes in problem.lanes],
        "dense_count": problem.dense_count(), "original_count": problem.original_count(),
        "exact": exact,
        "curve": curve,
        "floor": floor,
        "flat_from": flat,
        "deviations": list(DEVIATIONS),
        "notes": sorted(problem.notes),
        "priced": priced,
    }


def analyze_layouts(
    model: object,
    *,
    facts_backend: FactsBackendName = "dense",
    limits: Mapping | None = None,
    price: Iterable[Record] = (),
) -> Record:
    """Read a model, compile its facts and return its layout curve with timings."""
    from time import monotonic

    from .declarations import normalize_declarations
    from .facts import make_facts
    from .graph_scope import apply_scope
    from .model_snapshot import read_model

    limits = {"metadata_categories": 100_000} | dict(limits or {})
    started = monotonic()
    seconds = limits.get("seconds", float("inf"))

    def guard() -> None:
        if monotonic() - started > seconds:
            raise TimeoutError("The layout analysis exceeded its time limit")

    original = model if isinstance(model, dict) else read_model(model)
    declarations = apply_scope(original, normalize_declarations(original))
    facts = make_facts(facts_backend, declarations, limits, guard)
    compiled = monotonic()
    result = search_layouts(declarations, facts, limits, guard, price)
    result["phase_timings"] = {"facts": compiled - started, "search": monotonic() - compiled}
    return result


def _layout(problem: LayoutProblem, searches: list[_RegimeSearch], choice: Choice) -> list[Record]:
    groups = []
    for index, mask, cuts in sorted(choice.groups):
        search = searches[index]
        lanes = [problem.lanes[index][i] for i in range(search.size) if mask >> i & 1]
        blocks = []
        for start, stop in zip(cuts, cuts[1:], strict=False):
            fibres = search.block(mask, start, stop)
            priced = problem.price(index, fibres)
            blocks.append({**priced, "fibres": [
                {"period": problem.cells[cid]["period"], "regime": problem.cells[cid]["regime"],
                 "statuses": {name: problem._value(index, (cid, point), name)
                              for name in sorted({n for c, _ in fibres for n in problem.axes[c]})}}
                for cid, point in fibres]})
        groups.append({"dense_regime": list(problem.dense[index]),
                       "lanes": [{"regime": r, "statuses": dict(p)} for r, p in lanes],
                       "periods": list(search.active(mask)), "cuts": list(cuts), "blocks": blocks})
    return groups
