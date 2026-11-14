"""NOT RUN: original Python >=3.14 checker versus compact finite predicates.

Run in the original project environment with this directory on PYTHONPATH:
    python -m unittest -v native_finite_differential
Do not import this file in the portable reviewer controls.
"""

from typing import Literal, cast
from _lcm_regime_finder.fibre_source import _mapping
from _lcm_regime_finder.implicit.compact_screen import Package

from itertools import product
import unittest

from _lcm_regime_finder.fibre_facts import (
    JointReadSet,
    ScreenFibre,
    ScreenLink,
    check_inert_projection,
)

from _lcm_regime_finder.implicit.compact_screen import check_screen, possible_closure
from tests.regime_finder.test_implicit_screen import all_removed, simple


def original_finite_rows(
    package: Package,
) -> tuple[tuple[ScreenFibre, ...], frozenset[str]]:
    """Tiny-only literal expansion, with no producer source translation reused."""
    cells = {c.identifier: c for c in package.cells}
    points = {
        key: tuple(product(*(codes for _, codes in c.status_domains)))
        for key, c in cells.items()
    }
    count = sum(len(values) for values in points.values())
    if count * count > 65536:
        raise ValueError("Tiny native differential fixture exceeds the oracle cap")
    rows = []
    for name, cell in cells.items():
        for point in points[name]:
            match = [
                f
                for f in package.facts
                if f.cell == name
                and all(code in values for code, values in zip(point, f.box))
            ]
            if len(match) != 1:
                raise ValueError("Missing or overlapping full-hull fact")
            fact = match[0]
            links = []
            for edge in package.supports:
                if edge.source != name:
                    continue
                for target in points[edge.target]:
                    if all(
                        code in values for code, values in zip(point + target, edge.box)
                    ):
                        links.append(
                            ScreenLink(
                                cells[edge.target].address(target),
                                cast(
                                    Literal["solve", "simulation", "both"], edge.phase
                                ),
                                edge.marked,
                                edge.incidences,
                            )
                        )
            rows.append(
                ScreenFibre(
                    identifier=cell.address(point),
                    cell_id=name,
                    regime=cell.regime,
                    status_domains=cell.status_domains,
                    statuses=tuple(
                        (axis, code)
                        for (axis, _), code in zip(cell.status_domains, point)
                    ),
                    reads=tuple(
                        (
                            key,
                            JointReadSet(
                                value.live, value.removal_order, value.removed_statuses
                            ),
                        )
                        for key, value in fact.reads
                    ),
                    forced=fact.forced,
                    excluded=fact.excluded,
                    links=tuple(links),
                )
            )
    closure = possible_closure(package)
    kept = frozenset(
        cells[name].address(point)
        for name, values in points.items()
        for point in values
        if closure[name].contains(point)
    )
    return tuple(rows), kept


class NativeFiniteDifferential(unittest.TestCase):
    def compare(self, package: Package, removed: frozenset[tuple[str, str]]) -> None:
        fibres, kept = original_finite_rows(package)
        original = check_inert_projection(fibres=fibres, kept=kept, removed=removed)
        compact = check_screen(package, removed)
        self.assertEqual(
            {name: value.status for name, value in original.premises.items()},
            {
                name: _mapping(value)["status"]
                for name, value in _mapping(compact["premises"]).items()
            },
        )
        self.assertEqual(original.conditions_pass, compact["conditions_pass"])
        self.assertEqual(original.scope, compact["proof_scope"])

    def test_all_binary_support_relations(self) -> None:
        pairs = tuple(product((0, 1), repeat=2))
        for mask in range(16):
            package = simple(
                edges=tuple(pair for i, pair in enumerate(pairs) if mask & (1 << i))
            )
            axes = tuple(sorted(all_removed(package)))
            for subset in range(4):
                removed = frozenset(
                    axis for i, axis in enumerate(axes) if subset & (1 << i)
                )
                with self.subTest(mask=mask, subset=subset):
                    self.compare(package, removed)

    def test_marked_cross_slice_negative_control(self) -> None:
        package = simple(edges=((0, 1), (1, 0)), same_regime=True, marked=True)
        self.compare(package, all_removed(package))


if __name__ == "__main__":
    unittest.main(verbosity=2)
