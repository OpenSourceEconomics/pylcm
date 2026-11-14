"""Independent, tiny-only literal interpreter of the finite screen statement.

No producer/compiler/box-algebra functions are imported.  This materializes tiny
original fibres and links, and its liveness oracle uses dependency-graph traversal.
It is not intended to run on corpus captures; bounds are oracle host safeguards.
"""

from collections.abc import Mapping
from typing import TypedDict
from _lcm_regime_finder.implicit.compact_screen import Package

from collections import deque
from itertools import product
from math import prod


def literal(
    package: Package,
    removed: frozenset[tuple[str, str]],
    *,
    kept_override: Mapping[str, set[tuple[int, ...]]] | None = None,
) -> LiteralResult:
    cells = {c.identifier: c for c in package.cells}
    point_count = sum(
        prod(len(codes) for _, codes in c.status_domains) for c in cells.values()
    )
    if point_count > 4096 or point_count * point_count > 65536:
        raise ValueError("Tiny literal oracle safety limit; no corpus claim follows")
    points = {
        name: tuple(product(*(codes for _, codes in c.status_domains)))
        for name, c in cells.items()
    }
    rows = {}
    for cell, values in points.items():
        for point in values:
            matches = [
                f
                for f in package.facts
                if f.cell == cell
                and all(value in codes for value, codes in zip(point, f.box))
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Literal full rectangle needs exactly one fact per point"
                )
            rows[cell, point] = matches[0]
    links = {key: set() for key in rows}
    updates = {}
    for edge in package.supports:
        n = len(cells[edge.source].status_domains)
        keys = (
            dict(edge.update_keys)
            if edge.update_keys
            else {
                coordinate: "update:" + ":".join(edge.incidences) + ":" + coordinate
                for coordinate in cells[edge.target].coordinates
            }
        )
        for left in points[edge.source]:
            for right in points[edge.target]:
                if all(value in codes for value, codes in zip(left + right, edge.box)):
                    key = edge.source, left
                    link = (
                        edge.target,
                        right,
                        edge.phase,
                        edge.marked,
                        edge.incidences,
                    )
                    links[key].add(link)
                    binding_key = key, link
                    if updates.setdefault(binding_key, keys) != keys:
                        raise ValueError("Competing update producer binding")
    initial = {
        (name, point)
        for name, cell in cells.items()
        if cell.admissible_start
        for point in points[name]
    }
    if kept_override is None:
        kept = set(initial)
        queue = deque(initial)
        while queue:
            source = queue.popleft()
            for target, point, *_ in links[source]:
                if (target, point) not in kept:
                    kept.add((target, point))
                    queue.append((target, point))
    else:
        kept = {
            (cell, point) for cell, values in kept_override.items() for point in values
        }
    projection = {}
    groups = {}
    for cell, values in points.items():
        for point in values:
            named = tuple(
                (name, value)
                for (name, _), value in zip(cells[cell].status_domains, point)
            )
            q = (
                cell,
                tuple(
                    (name, value)
                    for name, value in named
                    if (cells[cell].regime, name) not in removed
                ),
            )
            projection[cell, point] = q
            groups.setdefault(q, []).append((cell, point))
    status = {name: "proved" for name in ("I1", "I2", "I3", "I4", "I5", "a", "b", "c")}
    for group in groups.values():
        live = [p for p in group if p in kept]
        if live and len(live) != len(group):
            status["I1"] = "refuted"
        live_values = [{key: value.live for key, value in rows[p].reads} for p in live]
        if any(value != live_values[0] for value in live_values) or any(
            rows[p].forced != rows[live[0]].forced for p in live
        ):
            status["I2"] = "refuted"
        excluded = [dict(rows[p].excluded) for p in group]
        if any(value != excluded[0] for value in excluded):
            status["I2"] = "refuted"
        neighbors = [
            {
                (projection[t, pt], phase, mark, incidences)
                for t, pt, phase, mark, incidences in links[p]
            }
            for p in live
        ]
        if any(value != neighbors[0] for value in neighbors):
            status["I3"] = "refuted"
    status["b"] = status["I2"]
    coarse_marked = set()
    fine_marked = set()
    for source, outgoing in links.items():
        for target, point, _phase, marked, _incidences in outgoing:
            if marked:
                fine_marked.add((source, (target, point)))
                if (
                    source in kept
                    and (target, point) in kept
                    and cells[source[0]].regime == cells[target].regime
                ):
                    coarse_marked.add((projection[source], projection[target, point]))

    def slice_assignment(address: tuple[str, tuple[int, ...]]) -> dict[str, int]:
        name, point = address
        cell = cells[name]
        return dict(
            (axis, value)
            for (axis, _), value in zip(cell.status_domains, point)
            if (cell.regime, axis) in removed
        )

    for source_group, target_group in coarse_marked:
        for left in groups[source_group]:
            for right in groups[target_group]:
                if (
                    slice_assignment(left) == slice_assignment(right)
                    and (left, right) not in fine_marked
                ):
                    status["I4"] = "refuted"
    if any(rows[p].forced for p in kept):
        for name in ("I4", "I5", "c"):
            if status[name] != "refuted":
                status[name] = "unknown"
    # Independent least liveness: reverse graph reachability from direct consumers.
    dependencies = {}
    seed = set()
    for address, fact in rows.items():
        for coordinate in dict(fact.reads)["direct"].live:
            seed.add((address, coordinate))
        for link in links[address]:
            target, point, *_ = link
            for coordinate in cells[target].coordinates:
                key = updates[address, link][coordinate]
                if key is not None and key not in dict(fact.reads):
                    raise ValueError("Missing original update read key")
                origin = ((target, point), coordinate)
                dependencies.setdefault(origin, set()).update(
                    (address, source_coordinate)
                    for source_coordinate in (
                        cells[address[0]].coordinates
                        if key is None
                        else dict(fact.reads)[key].live
                    )
                )
    live = set(seed)
    queue = deque(seed)
    while queue:
        node = queue.popleft()
        for parent in dependencies.get(node, ()):
            if parent not in live:
                live.add(parent)
                queue.append(parent)
    return {
        "premises": status,
        "kept": kept,
        "liveness": live,
        "projection": projection,
        "links": links,
    }


type Address = tuple[str, tuple[int, ...]]
type Projected = tuple[str, tuple[tuple[str, int], ...]]
type Link = tuple[str, tuple[int, ...], str, bool, tuple[str, ...]]


class LiteralResult(TypedDict):
    premises: dict[str, str]
    kept: set[Address]
    liveness: set[tuple[Address, str]]
    projection: dict[Address, Projected]
    links: dict[Address, set[Link]]
