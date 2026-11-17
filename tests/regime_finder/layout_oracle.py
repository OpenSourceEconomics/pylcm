"""A literal oracle for the Supplement's family on tiny models.

It enumerates every partition of each dense regime's lanes and every cut of
each group's own periods, prices each block from the definitions with its own
code, and convolves over the dense regimes. It shares only the problem's facts
(kept fibres, live sets, kept actions) with the search, never its recurrences or
its price. Adapted from the 099 transfer's literal_curve.py.
"""

from itertools import product
from math import prod

from _lcm_regime_finder.layout_search import FibreKey, LayoutProblem, Point, StatusValue


def _partitions(items: list[int]) -> list[list[list[int]]]:
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    result = []
    for partition in _partitions(rest):
        result.append([[first], *partition])
        for i in range(len(partition)):
            result.append([*partition[:i], [first, *partition[i]], *partition[i + 1 :]])
    return result


def _value(
    problem: LayoutProblem, members: tuple[str, ...], cid: str, point: Point, name: str
) -> StatusValue:
    cell = problem.cells[cid]
    if name in cell["states"]:
        domain = problem.domains[cell["states"][name]]
        if name in dict(point):
            return domain["values"][domain["codes"].index(dict(point)[name])]
        if len(domain.get("values", [])) == 1:
            return domain["values"][0]
    holders = sorted(
        (
            c
            for c in problem.cells.values()
            if c["regime"] in members and name in c["states"]
        ),
        key=lambda c: (c["regime"], c["period"]),
    )
    return problem.domains[holders[0]["states"][name]]["values"][0]


def literal_price(
    problem: LayoutProblem, members: tuple[str, ...], fibres: list[FibreKey]
) -> tuple[int, int]:
    cells = [problem.cells[cid] for cid, _ in fibres]
    periods = {c["period"] for c in cells}
    live = set().union(*(problem.live(f) for f in fibres))
    statuses = {n for cid, _ in fibres for n in problem.axes[cid]}
    rows = len({c["regime"] for c in cells}) if len(members) > 1 else 1
    for name in live:
        if name in statuses:
            rows *= len(
                {
                    repr(_value(problem, members, cid, point, name))
                    for cid, point in fibres
                }
            )
        else:
            rows *= max(
                1
                if problem.domains[c["states"][name]].get("solve_axis") is False
                else problem.domains[c["states"][name]]["size"]
                for c in cells
                if name in c["states"]
            )
    actions = set().union(*(problem.kept_actions(cid) for cid, _ in fibres))
    columns = prod(
        max(
            problem.domains[c["actions"][a]]["size"] for c in cells if a in c["actions"]
        )
        for a in actions
    )
    return len(periods) * rows * columns, len(periods) * rows


def literal_tables(
    problem: LayoutProblem,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int, int]]]:
    """Return w(n) as (ev, st) and ev*(k) as (ev, blocks, st), by enumeration."""
    per_regime = []
    for members in problem.dense:
        lanes = sorted(
            {
                (
                    problem.cells[cid]["regime"],
                    tuple((n, c) for n, c in point if n not in problem.contracted),
                )
                for cid, point in problem.kept
                if problem.cells[cid]["regime"] in members
            }
        )
        fibres = {
            lane: [
                (cid, point)
                for cid, point in sorted(problem.kept)
                if (
                    problem.cells[cid]["regime"],
                    tuple((n, c) for n, c in point if n not in problem.contracted),
                )
                == lane
            ]
            for lane in lanes
        }
        best: dict[int, tuple[int, int]] = {}
        for grouping in _partitions(list(range(len(lanes)))):
            options = []
            for group in grouping:
                owned = [f for i in group for f in fibres[lanes[i]]]
                active = sorted({problem.cells[cid]["period"] for cid, _ in owned})
                cuts = []
                for mask in range(1 << (len(active) - 1)):
                    bounds = [
                        0,
                        *(i for i in range(1, len(active)) if mask >> (i - 1) & 1),
                        len(active),
                    ]
                    total = (0, 0)
                    for a, b in zip(bounds, bounds[1:], strict=False):
                        block = [
                            f
                            for f in owned
                            if problem.cells[f[0]]["period"] in active[a:b]
                        ]
                        ev, st = literal_price(problem, members, block)
                        total = (total[0] + ev, total[1] + st)
                    cuts.append((len(bounds) - 1, total))
                options.append(cuts)
            for chosen in product(*options):
                n = sum(c[0] for c in chosen)
                cost = (sum(c[1][0] for c in chosen), sum(c[1][1] for c in chosen))
                best[n] = min(best.get(n, cost), cost)
        per_regime.append(best)
    exact: dict[int, tuple[int, int]] = {0: (0, 0)}
    for table in per_regime:
        exact = {
            n: min(
                (a[0] + b[0], a[1] + b[1])
                for m, a in exact.items()
                for l, b in table.items()
                if m + l == n
            )
            for n in {m + l for m in exact for l in table}
        }
    at_most = {
        k: min((exact[n][0], n, exact[n][1]) for n in exact if n <= k)
        for k in range(min(exact), max(exact) + 1)
    }
    return exact, at_most
