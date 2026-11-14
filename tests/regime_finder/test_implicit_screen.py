"""Portable controls; no pylcm, Model, JAX, callback, or source translator import."""

from collections.abc import Iterable, Mapping, Sequence
from typing import cast
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.fibre_source import FibreSource, Wire, _mapping
from _lcm_regime_finder.implicit.compact_screen import (
    Box,
    Package,
    ReadCertificate,
    Region,
    ScreenResult,
)

from dataclasses import replace
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace
import unittest

from _lcm_regime_finder.implicit.compact_screen import (
    Cell,
    FactRegion,
    SupportRegion,
    check_screen,
    least_liveness,
    possible_closure,
    validate,
    verdict_passes,
)
from tests.regime_finder.implicit_literal_oracle import literal
from _lcm_regime_finder.implicit.factored_bridge import (
    as_box_package,
    build_factored_table,
)
from _lcm_regime_finder.implicit.source_versions import resolve_source_candidate

REFERENCE_VERSION = resolve_source_candidate(source_version="source587")


def package_from_wire(data: Mapping[str, JSONValue]) -> Package:
    """Normalize explicit representation records, never callable constants."""
    if data.get("schema") != "compact-original-fibres/v1":
        raise ValueError("Unknown compact representation schema")

    def sequence(value: JSONValue) -> tuple[JSONValue, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected a representation list or tuple")
        return tuple(value)

    def record(value: JSONValue) -> Mapping[str, JSONValue]:
        if not isinstance(value, Mapping):
            raise ValueError("Expected a representation record")
        return value

    def text(value: JSONValue) -> str:
        if not isinstance(value, str):
            raise ValueError("Expected a named original coordinate")
        return value

    def boolean(value: JSONValue) -> bool:
        if type(value) is not bool:
            raise ValueError("Expected a Boolean flag")
        return value

    def strings(value: JSONValue) -> tuple[str, ...]:
        return tuple(text(item) for item in sequence(value))

    def nominal_codes(value: JSONValue) -> tuple[int, ...]:
        codes = sequence(value)
        if any(type(code) is not int for code in codes) or len(set(codes)) != len(
            codes
        ):
            raise ValueError("Code lists must contain distinct nominal integer codes")
        return cast(tuple[int, ...], codes)

    def names_and_codes(value: JSONValue) -> tuple[tuple[str, tuple[int, ...]], ...]:
        rows = []
        for item in sequence(value):
            name, codes = sequence(item)
            rows.append((text(name), nominal_codes(codes)))
        return tuple(rows)

    def box(value: JSONValue) -> Box:
        return tuple(frozenset(nominal_codes(codes)) for codes in sequence(value))

    cells = []
    for item in sequence(data["cells"]):
        row = record(item)
        cells.append(
            Cell(
                text(row["identifier"]),
                text(row["regime"]),
                names_and_codes(row["status_domains"]),
                frozenset(strings(row["coordinates"])),
                strings(row["read_keys"]),
                names_and_codes(row["action_domains"]),
                boolean(row["admissible_start"]),
            )
        )
    facts = []
    for item in sequence(data["facts"]):
        row = record(item)
        reads = []
        for pair in sequence(row["reads"]):
            key, raw = sequence(pair)
            value = record(raw)
            reads.append(
                (
                    text(key),
                    ReadCertificate(
                        frozenset(strings(value["live"])),
                        strings(value["removal_order"]),
                        strings(value["removed_statuses"]),
                    ),
                )
            )
        excluded = tuple(
            (name, frozenset(codes)) for name, codes in names_and_codes(row["excluded"])
        )
        facts.append(
            FactRegion(
                text(row["cell"]),
                box(row["box"]),
                tuple(reads),
                frozenset(strings(row["forced"])),
                excluded,
            )
        )
    supports = []
    for item in sequence(data["supports"]):
        row = record(item)
        updates = []
        for pair in sequence(row.get("update_keys", ())):
            coordinate, key = sequence(pair)
            updates.append((text(coordinate), text(key) if key is not None else None))
        supports.append(
            SupportRegion(
                text(row["source"]),
                text(row["target"]),
                box(row["box"]),
                text(row["phase"]),
                boolean(row["marked"]),
                strings(row["incidences"]),
                tuple(updates),
            )
        )
    package = Package(
        tuple(cells),
        tuple(facts),
        tuple(supports),
        sequence(data.get("source_obligations", ())),
    )
    validate(package)
    return package


def fixture_record(value: JSONValue) -> Wire:
    assert isinstance(value, dict)
    return value


def fixture_sequence(value: JSONValue) -> Sequence[JSONValue]:
    assert isinstance(value, (list, tuple))
    return value


def fixture_list(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def fixed(*names: str) -> ReadCertificate:
    return ReadCertificate(frozenset(names))


def simple(
    *,
    edges: Iterable[tuple[int, ...]] = ((0, 0), (1, 1)),
    same_regime: bool = False,
    marked: bool = False,
) -> Package:
    domain = (("i", (0, 1)),)
    sc = Cell(
        "source",
        "r" if same_regime else "source",
        domain,
        frozenset({"i"}),
        ("direct", "update:law:i"),
        (),
        True,
    )
    tc = Cell(
        "target",
        "r" if same_regime else "target",
        domain,
        frozenset({"i"}),
        ("direct",),
        (),
        False,
    )
    facts = (
        FactRegion(
            "source", sc.domains, (("direct", fixed()), ("update:law:i", fixed()))
        ),
        FactRegion("target", tc.domains, (("direct", fixed()),)),
    )
    supports = tuple(
        SupportRegion(
            "source",
            "target",
            (frozenset({i}), frozenset({j})),
            "solve",
            marked,
            ("law",),
        )
        for i, j in edges
    )
    return Package((sc, tc), facts, supports, ("retained query-owner qualifier",))


def all_removed(package: Package) -> frozenset[tuple[str, str]]:
    return frozenset(
        (cell.regime, name) for cell in package.cells for name, _ in cell.status_domains
    )


def wire(package: Package) -> Wire:
    return {
        "schema": "compact-original-fibres/v1",
        "cells": tuple(
            {
                "identifier": c.identifier,
                "regime": c.regime,
                "status_domains": c.status_domains,
                "coordinates": tuple(sorted(c.coordinates)),
                "read_keys": c.read_keys,
                "action_domains": c.action_domains,
                "admissible_start": c.admissible_start,
            }
            for c in package.cells
        ),
        "facts": tuple(
            {
                "cell": f.cell,
                "box": tuple(tuple(sorted(v)) for v in f.box),
                "reads": tuple(
                    (
                        key,
                        {
                            "live": tuple(sorted(v.live)),
                            "removal_order": v.removal_order,
                            "removed_statuses": v.removed_statuses,
                        },
                    )
                    for key, v in f.reads
                ),
                "forced": tuple(sorted(f.forced)),
                "excluded": tuple((key, tuple(sorted(v))) for key, v in f.excluded),
            }
            for f in package.facts
        ),
        "supports": tuple(
            {
                "source": e.source,
                "target": e.target,
                "box": tuple(tuple(sorted(v)) for v in e.box),
                "phase": e.phase,
                "marked": e.marked,
                "incidences": e.incidences,
                "update_keys": e.update_keys,
            }
            for e in package.supports
        ),
        "source_obligations": package.source_obligations,
    }


def factor_fixture() -> tuple[FibreSource, Wire, dict[str, tuple[str, ...]]]:
    # Independently hand-encoded typed facts, without producer pack/unpack.
    def strings(kind: str, values: Sequence[str]) -> JSONValue:
        return [kind, [["str", value] for value in values]]

    def read(live: Sequence[str], removed: Sequence[str]) -> JSONValue:
        return [
            "ReadCertificate",
            [
                strings("frozenset", live),
                strings("tuple", ("i",)),
                strings("tuple", removed),
            ],
        ]

    def support(code: int) -> JSONValue:
        return ["Support", [["frozenset", [["int", code]]], ["bool", True]]]

    cells = {
        "source": {
            "id": "source",
            "regime": "source",
            "period": 0,
            "states": {"i": "binary"},
            "actions": {},
            "admissible_start": True,
        },
        "target": {
            "id": "target",
            "regime": "target",
            "period": 1,
            "states": {"i": "binary"},
            "actions": {},
            "admissible_start": False,
        },
    }
    relation = {
        "id": "law",
        "source": "source",
        "targets": ("target",),
        "kind": "transition",
        "phase": "solve",
        "state_updates": {"i": "original_callback"},
    }
    source = SimpleNamespace(
        _cells=cells,
        _relations={"law": relation},
        _domains={"binary": {"kind": "categorical", "identity": "binary"}},
        _category=lambda domain: SimpleNamespace(by_code={0: 0, 1: 1}),
        source_obligations=("original query qualifier",),
    )
    source_diagram = {
        "domains": [["i", [0, 1]]],
        "nodes": [
            {"leaf": read((), ("i",))},
            {"leaf": read(("i",), ())},
            {"leaf": support(0)},
            {"leaf": support(1)},
            {"axis": 0, "children": [2, 3]},
        ],
        "roots": {"direct": 0, "read:law::i": 1, "support:law::i": 4},
    }
    target_diagram = {
        "domains": [["i", [0, 1]]],
        "nodes": [{"leaf": read((), ("i",))}],
        "roots": {"direct": 0},
    }
    record = {
        "schema": "exact-implicit-fibre-table/v1",
        "reference_contract": "compile_primitive_fibres.py@587478274ff55f1d873ae9d056bb9bf3e222b873",
        "source_obligations": list(source.source_obligations),
        "cells": [
            {
                "cell_id": "source",
                "regime": "source",
                "period": 0,
                "forced": [],
                "relations": [relation],
                "diagram": source_diagram,
            },
            {
                "cell_id": "target",
                "regime": "target",
                "period": 1,
                "forced": [],
                "relations": [],
                "diagram": target_diagram,
            },
        ],
    }
    return cast(FibreSource, source), record, {"source": ("i",), "target": ("i",)}


class TestCompactScreenControls(unittest.TestCase):
    def assert_matches_literal(
        self,
        package: Package,
        removed: frozenset[tuple[str, str]],
        *,
        kept_override: Mapping[str, Region] | None = None,
    ) -> ScreenResult:
        ordinary_kept = None
        if kept_override is not None:
            ordinary_kept = {
                c.identifier: {
                    point
                    for point in product(*(codes for _, codes in c.status_domains))
                    if kept_override[c.identifier].contains(point)
                }
                for c in package.cells
            }
        expected = literal(package, removed, kept_override=ordinary_kept)
        observed = check_screen(package, removed, kept_override=kept_override)
        self.assertEqual(
            {
                key: _mapping(value)["status"]
                for key, value in _mapping(observed["premises"]).items()
            },
            expected["premises"],
        )
        if kept_override is None:
            closure = possible_closure(package)
            actual_kept = {
                (c.identifier, point)
                for c in package.cells
                for point in product(*(codes for _, codes in c.status_domains))
                if closure[c.identifier].contains(point)
            }
            self.assertEqual(actual_kept, expected["kept"])
        live = least_liveness(package)
        actual_live = {
            ((c.identifier, point), coordinate)
            for c in package.cells
            for point in product(*(codes for _, codes in c.status_domains))
            for coordinate in c.coordinates
            if live[c.identifier, coordinate].contains(point)
        }
        self.assertEqual(actual_live, expected["liveness"])
        self.assertEqual(
            observed["source_obligations"], list(package.source_obligations)
        )
        self.assertFalse(observed["transformation_authorized"])
        return observed

    def test_box_algebra_exhaustive_two_binary_axes(self) -> None:
        domain = (frozenset({0, 1}), frozenset({0, 1}))
        points = tuple(product((0, 1), repeat=2))
        subsets = [
            {p for i, p in enumerate(points) if mask & (1 << i)} for mask in range(16)
        ]
        for left in subsets:
            a = Region.make(domain, (tuple(frozenset({v}) for v in p) for p in left))
            for right in subsets:
                b = Region.make(
                    domain, (tuple(frozenset({v}) for v in p) for p in right)
                )
                with self.subTest(left=sorted(left), right=sorted(right)):
                    for actual, expected in (
                        (a.union(b), left | right),
                        (a.intersection(b), left & right),
                        (a.difference(b), left - right),
                    ):
                        self.assertEqual(
                            {p for p in points if actual.contains(p)}, expected
                        )
                        self.assertEqual(actual.cardinality, len(expected))

    def test_zero_dimensional_full_and_empty(self) -> None:
        empty = Region.make(())
        full = Region.full(())
        self.assertEqual(full.cardinality, 1)
        self.assertEqual(empty.cardinality, 0)
        self.assertTrue(full.contains(()))
        self.assertFalse(full.difference(full).boxes)
        self.assertTrue(empty.union(full).equal(full))

    def test_generated_all_binary_relations_and_removal_sets(self) -> None:
        pairs = tuple(product((0, 1), repeat=2))
        for mask in range(16):
            package = simple(
                edges=tuple(pair for i, pair in enumerate(pairs) if mask & (1 << i))
            )
            axes = tuple(sorted(all_removed(package)))
            for removed_mask in range(4):
                removed = frozenset(
                    axis for i, axis in enumerate(axes) if removed_mask & (1 << i)
                )
                with self.subTest(mask=mask, removed_mask=removed_mask):
                    self.assert_matches_literal(package, removed)

    def test_simultaneous_source_and_target_projection(self) -> None:
        package = simple()
        joint = self.assert_matches_literal(package, all_removed(package))
        unilateral = self.assert_matches_literal(package, frozenset({("source", "i")}))
        self.assertTrue(joint["conditions_pass"])
        self.assertEqual(unilateral["premises"]["I3"]["status"], "refuted")

    def test_I1_requires_every_original_part(self) -> None:
        package = simple(edges=((0, 0), (1, 0)))
        result = self.assert_matches_literal(package, all_removed(package))
        self.assertEqual(result["premises"]["I1"]["status"], "refuted")
        self.assertFalse(result["conditions_pass"])

    def test_unkept_full_hull_exclusion_is_checked(self) -> None:
        package = simple(edges=())
        source, target = package.cells
        target = replace(target, action_domains=(("choice", (0, 1)),))
        facts = (
            package.facts[0],
            FactRegion(
                "target",
                (frozenset({0}),),
                (("direct", fixed()),),
                excluded=(("choice", frozenset()),),
            ),
            FactRegion(
                "target",
                (frozenset({1}),),
                (("direct", fixed()),),
                excluded=(("choice", frozenset({1})),),
            ),
        )
        package = replace(package, cells=(source, target), facts=facts)
        result = self.assert_matches_literal(package, all_removed(package))
        self.assertEqual(result["premises"]["I2"]["status"], "refuted")

    def test_marked_cross_slice_edges_fail(self) -> None:
        package = simple(edges=((0, 1), (1, 0)), same_regime=True, marked=True)
        result = self.assert_matches_literal(package, all_removed(package))
        self.assertEqual(result["premises"]["I3"]["status"], "proved")
        self.assertEqual(result["premises"]["I4"]["status"], "refuted")

    def test_named_slice_alignment_with_reversed_axis_order(self) -> None:
        for transpose in (False, True):
            source = Cell(
                "a",
                "r",
                (("i", (0, 1)), ("j", (0, 1))),
                frozenset({"i", "j"}),
                ("direct", "u_i", "u_j"),
                (),
                True,
            )
            target = Cell(
                "b", "r", (("j", (0, 1)), ("i", (0, 1))), frozenset({"i", "j"})
            )
            facts = (
                FactRegion(
                    "a",
                    source.domains,
                    (("direct", fixed()), ("u_i", fixed()), ("u_j", fixed())),
                ),
                FactRegion("b", target.domains, (("direct", fixed()),)),
            )
            edges = tuple(
                SupportRegion(
                    "a",
                    "b",
                    tuple(
                        frozenset({v})
                        for v in ((i, j) + ((i, j) if transpose else (j, i)))
                    ),
                    "solve",
                    True,
                    ("law",),
                    (("i", "u_i"), ("j", "u_j")),
                )
                for i, j in product((0, 1), repeat=2)
            )
            package = Package((source, target), facts, edges)
            with self.subTest(transpose=transpose):
                result = self.assert_matches_literal(package, all_removed(package))
                self.assertEqual(
                    result["premises"]["I4"]["status"],
                    "refuted" if transpose else "proved",
                )

    def test_phase_and_incidence_identity_survive(self) -> None:
        for change in ("phase", "incidences"):
            package = simple()
            altered = replace(
                package.supports[1],
                **(
                    {"phase": "simulation"}
                    if change == "phase"
                    else {"incidences": ("other",)}
                ),
            )
            if change == "incidences":
                altered = replace(altered, update_keys=(("i", "update:law:i"),))
            package = replace(package, supports=(package.supports[0], altered))
            with self.subTest(change=change):
                result = self.assert_matches_literal(package, all_removed(package))
                self.assertEqual(result["premises"]["I3"]["status"], "refuted")

    def test_omitted_and_unknown_premises_cannot_pass(self) -> None:
        result = check_screen(simple(), all_removed(simple()))
        self.assertTrue(verdict_passes(result["premises"]))
        for name in tuple(result["premises"]):
            missing = dict(result["premises"])
            missing.pop(name)
            self.assertFalse(verdict_passes(missing))
            unknown = dict(result["premises"])
            unknown[name] = {"status": "unknown"}
            self.assertFalse(verdict_passes(unknown))

    def test_nonempty_forcing_remains_unknown(self) -> None:
        package = simple()
        package = replace(
            package,
            facts=(
                replace(package.facts[0], forced=frozenset({"choice"})),
                package.facts[1],
            ),
        )
        result = self.assert_matches_literal(package, all_removed(package))
        self.assertEqual(result["premises"]["I5"]["status"], "unknown")

    def test_coverage_overlap_and_unrecognised_facts_are_rejected(self) -> None:
        package = simple()
        for facts in (
            package.facts[:1],
            package.facts + (package.facts[0],),
            (replace(package.facts[0], reads=(("direct", fixed()),)), package.facts[1]),
        ):
            with self.subTest(facts=repr(facts)):
                with self.assertRaises(ValueError):
                    validate(replace(package, facts=facts))

    def test_tuple_and_json_list_records_have_identical_meaning(self) -> None:
        original = simple()
        tuple_package = package_from_wire(wire(original))
        json_package = package_from_wire(json.loads(json.dumps(wire(original))))
        self.assertEqual(tuple_package, json_package)
        self.assertEqual(tuple_package, original)
        self.assertEqual(
            check_screen(tuple_package, all_removed(tuple_package)),
            check_screen(json_package, all_removed(json_package)),
        )

    def test_noninteger_or_duplicate_wire_codes_rejected(self) -> None:
        for invalid in ([0, False], [0, 0], [0.0], [True]):
            data = json.loads(json.dumps(wire(simple())))
            data["facts"][0]["box"][0] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    package_from_wire(data)

    def test_nominal_codes_remain_distinct_addresses(self) -> None:
        # An external categorical payload may assign 7.0 to both codes.  Addresses
        # and support still use the original 11 and 27; values are never consulted.
        cell = Cell(
            "c", "r", (("i", (11, 27)),), frozenset({"i"}), admissible_start=True
        )
        package = Package(
            (cell,), (FactRegion("c", cell.domains, (("direct", fixed()),)),), ()
        )
        self.assertEqual(cell.address((11,)), "c:0")
        self.assertEqual(cell.address((27,)), "c:1")
        self.assertEqual(possible_closure(package)["c"].cardinality, 2)
        self.assert_matches_literal(package, all_removed(package))

    def test_full_rectangle_uses_one_box_without_point_expansion(self) -> None:
        domains = tuple(
            (f"s{i}", tuple(range(n))) for i, n in enumerate((2, 3, 5, 7, 11, 13, 17))
        )
        cell = Cell(
            "large",
            "r",
            domains,
            frozenset(name for name, _ in domains),
            admissible_start=True,
        )
        package = Package(
            (cell,), (FactRegion("large", cell.domains, (("direct", fixed()),)),), ()
        )
        closure = possible_closure(package)
        self.assertEqual(closure["large"].cardinality, 510510)
        self.assertEqual(len(closure["large"].boxes), 1)
        self.assertTrue(check_screen(package, all_removed(package))["conditions_pass"])

    def test_fixed_liveness_updates_and_named_explicit_bindings(self) -> None:
        a = Cell(
            "a", "a", (), frozenset({"x", "unused"}), ("direct", "law_a::b"), (), True
        )
        b = Cell("b", "b", (), frozenset({"b"}), ("direct", "law_b::z"))
        c = Cell("c", "c", (), frozenset({"z"}))
        facts = (
            FactRegion("a", (), (("direct", fixed()), ("law_a::b", fixed("x")))),
            FactRegion("b", (), (("direct", fixed()), ("law_b::z", fixed("b")))),
            FactRegion("c", (), (("direct", fixed("z")),)),
        )
        edges = (
            SupportRegion(
                "a", "b", (), "solve", False, ("law_a",), (("b", "law_a::b"),)
            ),
            SupportRegion(
                "b", "c", (), "simulation", True, ("law_b",), (("z", "law_b::z"),)
            ),
        )
        package = Package((a, b, c), facts, edges)
        self.assert_matches_literal(package, frozenset())
        live = least_liveness(package)
        self.assertEqual(live["a", "x"].cardinality, 1)
        self.assertEqual(live["a", "unused"].cardinality, 0)

    def test_competing_update_bindings_fail_loudly(self) -> None:
        package = simple()
        extra = replace(package.supports[0], update_keys=(("i", "direct"),))
        with self.assertRaises(ValueError):
            least_liveness(replace(package, supports=package.supports + (extra,)))

    def test_explicit_missing_producer_keeps_every_source_coordinate(self) -> None:
        source = Cell("a", "a", (), frozenset({"x", "y"}), admissible_start=True)
        target = Cell("b", "b", (), frozenset({"z"}))
        facts = (
            FactRegion("a", (), (("direct", fixed()),)),
            FactRegion("b", (), (("direct", fixed("z")),)),
        )
        edge = SupportRegion("a", "b", (), "solve", False, ("law",), (("z", None),))
        package = Package((source, target), facts, (edge,))
        self.assert_matches_literal(package, frozenset())
        self.assertEqual(least_liveness(package)["a", "x"].cardinality, 1)
        self.assertEqual(least_liveness(package)["a", "y"].cardinality, 1)
        self.assertEqual(set(dict(facts[0].reads)), {"direct"})

    def test_cycles_have_least_not_arbitrary_fixed_point(self) -> None:
        cell = Cell(
            "c", "r", (("i", (0, 1)),), frozenset({"i"}), ("direct", "u"), (), True
        )
        edge = SupportRegion(
            "c", "c", cell.domains + cell.domains, "both", True, ("law",), (("i", "u"),)
        )
        for seed in (False, True):
            fact = FactRegion(
                "c",
                cell.domains,
                (("direct", fixed("i") if seed else fixed()), ("u", fixed("i"))),
            )
            package = Package((cell,), (fact,), (edge,))
            with self.subTest(seed=seed):
                self.assert_matches_literal(package, all_removed(package))
                self.assertEqual(
                    least_liveness(package)["c", "i"].cardinality, 2 if seed else 0
                )

    def test_factor_bridge_preserves_identity_support_and_table_read_keys(self) -> None:
        source, record, axes = factor_fixture()
        table = build_factored_table(
            source=source, record=record, status_axes=axes, version=REFERENCE_VERSION
        )
        self.assertEqual(len(table.relation_factors), 1)
        self.assertEqual(
            table.relation_factors[0].support_roots, (("i", "support:law::i"),)
        )
        package = as_box_package(table)
        self.assertEqual(package.cells[0].read_keys, ("direct", "law::i"))
        self.assertEqual(package.supports[0].update_keys, (("i", "law::i"),))
        result = self.assert_matches_literal(package, all_removed(package))
        self.assertTrue(result["conditions_pass"])

    def test_factor_bridge_native_tuple_and_json_list_metadata_agree(self) -> None:
        source, record, axes = factor_fixture()
        native = as_box_package(
            build_factored_table(
                source=source,
                record=record,
                status_axes=axes,
                version=REFERENCE_VERSION,
            )
        )
        json_record = json.loads(json.dumps(record))
        decoded = as_box_package(
            build_factored_table(
                source=source,
                record=json_record,
                status_axes=axes,
                version=REFERENCE_VERSION,
            )
        )
        self.assertEqual(native, decoded)

    def test_factor_bridge_rejects_incomplete_coverage_and_lost_guards(self) -> None:
        for mutation in (
            "missing_root",
            "missing_child",
            "cycle",
            "missing_qualification",
            "wrong_incidence",
        ):
            source, record, axes = factor_fixture()
            altered = deepcopy(record)
            diagram = fixture_record(
                fixture_record(fixture_sequence(altered["cells"])[0])["diagram"]
            )
            if mutation == "missing_root":
                fixture_record(diagram["roots"]).pop("support:law::i")
            elif mutation == "missing_child":
                fixture_record(fixture_sequence(diagram["nodes"])[4])["children"] = [2]
            elif mutation == "cycle":
                fixture_record(fixture_sequence(diagram["nodes"])[4])["children"] = [
                    4,
                    3,
                ]
            elif mutation == "missing_qualification":
                altered["source_obligations"] = []
            elif mutation == "wrong_incidence":
                fixture_record(
                    fixture_record(
                        fixture_sequence(
                            fixture_record(fixture_sequence(altered["cells"])[0])[
                                "relations"
                            ]
                        )[0]
                    )["state_updates"]
                )["i"] = "invented_identity"
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    build_factored_table(
                        source=source,
                        record=altered,
                        status_axes=axes,
                        version=REFERENCE_VERSION,
                    )

    def test_exclusion_code_types_are_nominal_in_memory_and_on_wire(self) -> None:
        package = simple()
        cell = replace(package.cells[0], action_domains=(("choice", (0, 1)),))
        fact = replace(package.facts[0], excluded=(("choice", frozenset()),))
        package = replace(
            package, cells=(cell, package.cells[1]), facts=(fact, package.facts[1])
        )
        for invalid in ([False], [0.0], [0, False], [0, 0]):
            with self.subTest(invalid=invalid):
                if len(invalid) == 1:
                    malformed = replace(
                        fact, excluded=(("choice", frozenset(invalid)),)
                    )
                    with self.assertRaises(ValueError):
                        validate(replace(package, facts=(malformed, package.facts[1])))
                data = json.loads(json.dumps(wire(package)))
                data["facts"][0]["excluded"][0][1] = invalid
                with self.assertRaises(ValueError):
                    package_from_wire(data)

    def test_factor_exclusion_terminal_cannot_use_false_as_code_zero(self) -> None:
        source, record, axes = factor_fixture()
        source._cells["source"]["actions"] = {"choice": "binary"}
        diagram = fixture_record(
            fixture_record(fixture_sequence(record["cells"])[0])["diagram"]
        )
        fixture_record(diagram["roots"])["exclude:choice"] = len(
            fixture_sequence(diagram["nodes"])
        )
        fixture_list(diagram["nodes"]).append(
            {"leaf": ["frozenset", [["bool", False]]]}
        )
        table = build_factored_table(
            source=source, record=record, status_axes=axes, version=REFERENCE_VERSION
        )
        with self.assertRaises(ValueError):
            as_box_package(table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
