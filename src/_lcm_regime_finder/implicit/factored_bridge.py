"""Consume the compact table contract without importing its producer translator.

build_factored_table keeps each read/exclusion/support field diagram separate.
as_box_package is an explicit exact diagnostic adapter for the box verifier.  Its
common refinement can be exponential (e.g. many independent identity updates);
the factor record, not this expanded adapter, is the persistent representation.
No original fibre-pair table is ever materialized.  Source authentication still
requires verify_native.py to finish all original point queries and source guards.
"""

from dataclasses import dataclass
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import cast
from ..records import JSONValue
from ..fibre_source import FibreSource, Wire, _mapping, _rows, _strings, _text
from .verify_native import _integer, _object, _sequence

from .compact_screen import (
    Box,
    Cell,
    FactRegion,
    Package,
    ReadCertificate,
    SupportRegion,
)


@dataclass(frozen=True)
class SupportValue:
    possible_codes: frozenset[int]
    proved: bool


def decode_leaf(encoded: object) -> object:
    """Independent closed decoder of facts; no arbitrary callable constants."""
    if not isinstance(encoded, (list, tuple)) or len(encoded) != 2:
        raise ValueError("Invalid typed terminal")
    kind, value = encoded
    if kind in {"str", "int", "bool"}:
        if type(value).__name__ != kind:
            raise ValueError("Terminal scalar type changed")
        return value
    if kind in {"tuple", "frozenset"}:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Invalid typed collection")
        values = tuple(decode_leaf(v) for v in value)
        if kind == "frozenset":
            if len(set(values)) != len(values):
                raise ValueError("Duplicate set terminal member")
            return frozenset(values)
        return values
    if kind == "ReadCertificate":
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError("Incomplete read certificate")
        live, order, removed = (decode_leaf(v) for v in value)
        if (
            type(live) is not frozenset
            or type(order) is not tuple
            or type(removed) is not tuple
        ):
            raise ValueError("Read certificate field type changed")
        if any(type(n) is not str for n in tuple(live) + order + removed):
            raise ValueError("Read coordinate must be named")
        return ReadCertificate(
            cast(frozenset[str], live),
            cast(tuple[str, ...], order),
            cast(tuple[str, ...], removed),
        )
    if kind == "Support":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Incomplete support certificate")
        codes, proved = (decode_leaf(v) for v in value)
        if (
            type(codes) is not frozenset
            or type(proved) is not bool
            or any(type(c) is not int for c in codes)
        ):
            raise ValueError("Support certificate type changed")
        return SupportValue(cast(frozenset[int], codes), proved)
    raise ValueError("Unrecognised or unresolved fact terminal: " + str(kind))


class Diagram:
    """Independent structural verifier and lazy terminal-box decoder."""

    def __init__(
        self,
        wire: Mapping[str, object],
        expected_domains: Sequence[tuple[str, Sequence[int]]],
    ) -> None:
        domains_list = []
        for item in _sequence(wire["domains"]):
            name, codes = _sequence(item)
            if not isinstance(name, str):
                raise ValueError("Original axis name must be text")
            domains_list.append(
                (name, tuple(_integer(code) for code in _sequence(codes)))
            )
        domains = tuple(domains_list)
        if domains != expected_domains:
            raise ValueError(
                "Diagram does not cover the original ordered nominal domains"
            )
        for _, codes in domains:
            if (
                not codes
                or len(set(codes)) != len(codes)
                or any(type(c) is not int for c in codes)
            ):
                raise ValueError("Malformed original domain")
        if len({name for name, _ in domains}) != len(domains):
            raise ValueError("Repeated original axis")
        self.domains = domains
        self.universe = tuple(frozenset(codes) for _, codes in domains)
        self.nodes = tuple(_object(node) for node in _sequence(wire["nodes"]))
        self.roots = {
            name: _integer(root) for name, root in _object(wire["roots"]).items()
        }
        if not self.nodes:
            raise ValueError("A diagram must contain a terminal")
        self.values: dict[int, object] = {}
        for i, node in enumerate(self.nodes):
            if set(node) == {"leaf"}:
                self.values[i] = decode_leaf(node["leaf"])
            elif set(node) == {"axis", "children"}:
                axis = _integer(node["axis"])
                if type(axis) is not int or not 0 <= axis < len(domains):
                    raise ValueError("Diagram axis is outside original schema")
                if len(_sequence(node["children"])) != len(domains[axis][1]):
                    raise ValueError("Diagram branch omits original codes")
                for raw_child in _sequence(node["children"]):
                    child = _integer(raw_child)
                    if type(child) is not int or not 0 <= child < len(self.nodes):
                        raise ValueError("Invalid diagram child")
            else:
                raise ValueError("Invalid diagram node shape")
        for node in self.nodes:
            if "axis" in node:
                for raw_child in _sequence(node["children"]):
                    child = _integer(raw_child)
                    nxt = self.nodes[child]
                    if "axis" in nxt and _integer(nxt["axis"]) <= _integer(
                        node["axis"]
                    ):
                        raise ValueError(
                            "Diagram order/cycle violates complete decision semantics"
                        )
        if any(
            type(root) is not int or not 0 <= root < len(self.nodes)
            for root in self.roots.values()
        ):
            raise ValueError("Invalid root")

    def cubes(self, root_name: str) -> Iterator[tuple[Box, object]]:
        def visit(root: int, box: Box) -> Iterator[tuple[Box, object]]:
            node = self.nodes[root]
            if "leaf" in node:
                yield box, self.values[root]
                return
            groups: dict[int, set[int]] = {}
            axis = _integer(node["axis"])
            for code, raw_child in zip(
                self.domains[axis][1], _sequence(node["children"])
            ):
                child = _integer(raw_child)
                groups.setdefault(child, set()).add(code)
            for child, codes in groups.items():
                current = list(box)
                current[axis] = frozenset(codes)
                yield from visit(child, tuple(current))

        yield from visit(self.roots[root_name], self.universe)


@dataclass(frozen=True)
class RelationFactors:
    source: str
    target: str
    phase: str
    marked: bool
    incidences: tuple[str, ...]
    update_keys: tuple[tuple[str, str | None], ...]
    # Every target status coordinate has an original support-root name, or None
    # for the explicit conservative full-domain missing-producer rule.
    support_roots: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class FactoredTable:
    cells: tuple[Cell, ...]
    diagrams: Mapping[str, Diagram]
    relation_factors: tuple[RelationFactors, ...]
    source_obligations: tuple[str, ...]
    source_authentication: str = "requires_independent_full_source_replay"


def build_factored_table(
    *, source: FibreSource, record: Wire, status_axes: Mapping[str, Sequence[str]]
) -> FactoredTable:
    """Bind the representation inventory to an already authenticated FibreSource.

    This inventory check is not a substitute for full-domain leaf verification.
    Neither the source guard nor a qualification is cleared here.
    """
    if record.get("schema") != "exact-implicit-fibre-table/v1":
        raise ValueError("Unknown fact table contract")
    if (
        record.get("reference_contract")
        != "compile_primitive_fibres.py@587478274ff55f1d873ae9d056bb9bf3e222b873"
    ):
        raise ValueError("Wrong reference program contract")
    original_cells = source._cells
    original_relations = source._relations
    records = {_text(row["cell_id"]): row for row in _rows(record["cells"])}
    if len(records) != len(_rows(record["cells"])) or tuple(records) != tuple(
        original_cells
    ):
        raise ValueError("Original cell inventory or order changed")
    if set(status_axes) != {_text(cell["regime"]) for cell in original_cells.values()}:
        raise ValueError("Explicit classification must cover every original regime")
    if _strings(record["source_obligations"]) != tuple(source.source_obligations):
        raise ValueError("Source qualification was dropped or changed")
    value_kinds = {"transition", "joint", "value_read", "fallback", "exit"}
    outgoing = {
        key: tuple(
            row
            for row in original_relations.values()
            if row["source"] == key and row["kind"] in value_kinds and row["targets"]
        )
        for key in original_cells
    }
    cells, diagrams = [], {}
    for cell_id, original in original_cells.items():
        row = records[cell_id]
        if (
            row["regime"] != original["regime"]
            or row["period"] != original["period"]
            or row["forced"] != []
        ):
            raise ValueError("Original cell owner, date or empty forcing changed")
        requested = tuple(status_axes[_text(original["regime"])])
        if (
            len(set(requested)) != len(requested)
            or not set(requested) <= _mapping(original["states"]).keys()
        ):
            raise ValueError("Invalid original status classification")
        axes = tuple(name for name in _mapping(original["states"]) if name in requested)
        domains = tuple(
            (
                name,
                tuple(
                    source._category(
                        source._domains[_text(_mapping(original["states"])[name])]
                    ).by_code
                ),
            )
            for name in axes
        )
        action_domains = tuple(
            (name, tuple(source._category(source._domains[_text(identity)]).by_code))
            for name, identity in _mapping(original["actions"]).items()
            if source._domains[_text(identity)].get("kind") == "categorical"
        )
        read_keys = ["direct"]
        root_names = {"direct"} | {"exclude:" + name for name, _ in action_domains}
        expected_meta = []
        for relation in outgoing[cell_id]:
            target = original_cells[_strings(relation["targets"])[0]]
            for coordinate in _mapping(target["states"]):
                key = _text(relation["id"]) + "::" + coordinate
                read_keys.append(key)
                root_names.add("read:" + key)
            root_names.update(
                "support:" + _text(relation["id"]) + "::" + name
                for name in status_axes[_text(target["regime"])]
            )
            expected_meta.append(
                {
                    k: relation[k]
                    for k in (
                        "id",
                        "source",
                        "targets",
                        "phase",
                        "kind",
                        "state_updates",
                    )
                    if k in relation
                }
            )

        # Only representation sequences are normalized; callable constants and
        # structural payloads remain behind the original authenticated source.
        def metadata(value: JSONValue) -> JSONValue:
            if isinstance(value, Mapping):
                return {key: metadata(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return tuple(metadata(item) for item in value)
            return value

        if metadata(row["relations"]) != metadata(expected_meta):
            raise ValueError("Original relation binding/phase/owner inventory changed")
        diagram = Diagram(_mapping(row["diagram"]), domains)
        if diagram.roots.keys() != root_names:
            raise ValueError(
                "Fact root coverage differs from the original point program"
            )
        admissible = original["admissible_start"]
        if type(admissible) is not bool:
            raise ValueError("Original initial-node declaration must be Boolean")
        cells.append(
            Cell(
                cell_id,
                _text(original["regime"]),
                domains,
                frozenset(_mapping(original["states"])),
                tuple(read_keys),
                action_domains,
                admissible,
            )
        )
        diagrams[cell_id] = diagram
    groups = {}
    for relation in original_relations.values():
        if relation["kind"] not in value_kinds:
            continue
        for target in _strings(relation["targets"]):
            group = (
                "transition"
                if relation["kind"] in {"transition", "joint"}
                else _text(relation["id"])
            )
            key = (
                _text(relation["source"]),
                target,
                _text(relation.get("phase", "both")),
                group,
            )
            groups.setdefault(key, []).append(_text(relation["id"]))
    factors = []
    for (source_id, target, phase, _group), incidences in groups.items():
        update_keys, support_roots = [], []
        for coordinate in _mapping(original_cells[target]["states"]):
            owners = [
                rid
                for rid in incidences
                if coordinate
                in _mapping(original_relations[rid].get("state_updates", {}))
            ]
            if len(owners) > 1:
                raise ValueError("Multiple original update producers")
            read_key = owners[0] + "::" + coordinate if owners else None
            update_keys.append((coordinate, read_key))
            if coordinate in status_axes[_text(original_cells[target]["regime"])]:
                support_roots.append(
                    (
                        coordinate,
                        "support:" + read_key if read_key is not None else None,
                    )
                )
        factors.append(
            RelationFactors(
                source_id,
                target,
                phase,
                original_cells[source_id]["period"] == original_cells[target]["period"],
                tuple(incidences),
                tuple(update_keys),
                tuple(support_roots),
            )
        )
    return FactoredTable(
        tuple(cells),
        MappingProxyType(diagrams),
        tuple(factors),
        tuple(source.source_obligations),
    )


def _refine(
    parts: Sequence[tuple[Box, tuple[object, ...]]], diagram: Diagram, root: str
) -> list[tuple[Box, tuple[object, ...]]]:
    """Common refinement only when the caller explicitly asks for a box adapter."""
    answer = []
    for box, values in parts:
        for extra, value in diagram.cubes(root):
            intersection = tuple(a & b for a, b in zip(box, extra))
            if all(intersection):
                answer.append((intersection, values + (value,)))
    return answer


def as_box_package(table: FactoredTable) -> Package:
    """Exact optional adapter, potentially exponential in independent factors.

    Never use adapter completion/failure to infer source certification, economic
    feasibility, maximum removability, or an impossibility result.  Full source
    replay must independently verify every field even if this adapter finishes.
    """
    cells = {cell.identifier: cell for cell in table.cells}
    facts, supports = [], []
    for cell in table.cells:
        diagram = table.diagrams[cell.identifier]
        roots = tuple(
            "direct" if key == "direct" else "read:" + key for key in cell.read_keys
        )
        roots += tuple("exclude:" + name for name, _ in cell.action_domains)
        parts = [(cell.domains, ())]
        for root in roots:
            parts = _refine(parts, diagram, root)
        for box, values in parts:
            reads = tuple(zip(cell.read_keys, values[: len(cell.read_keys)]))
            exclusions = tuple(
                (name, value)
                for (name, _), value in zip(
                    cell.action_domains, values[len(cell.read_keys) :]
                )
            )
            if any(type(value) is not ReadCertificate for _, value in reads) or any(
                type(value) is not frozenset
                or any(type(code) is not int for code in value)
                for _, value in exclusions
            ):
                raise ValueError(
                    "Read/exclusion leaf type violates original query contract"
                )
            facts.append(
                FactRegion(
                    cell.identifier,
                    box,
                    cast(tuple[tuple[str, ReadCertificate], ...], reads),
                    frozenset(),
                    cast(tuple[tuple[str, frozenset[int]], ...], exclusions),
                )
            )
    for relation in table.relation_factors:
        source, target = cells[relation.source], cells[relation.target]
        diagram = table.diagrams[relation.source]
        parts = [(source.domains, ())]
        actual = [
            (name, root) for name, root in relation.support_roots if root is not None
        ]
        for _, root in actual:
            parts = _refine(parts, diagram, root)
        for box, values in parts:
            target_box = list(target.domains)
            for (coordinate, _), value in zip(actual, values):
                if type(value) is not SupportValue:
                    raise ValueError(
                        "Support leaf type violates original query contract"
                    )
                index = target.names.index(coordinate)
                if not value.possible_codes <= target.domains[index]:
                    raise ValueError("Support names an unoriginal target code")
                if value.proved:
                    target_box[index] = value.possible_codes
                elif value.possible_codes != target.domains[index]:
                    raise ValueError(
                        "Unknown support must keep the full original domain"
                    )
            if all(target_box):
                supports.append(
                    SupportRegion(
                        source.identifier,
                        target.identifier,
                        box + tuple(target_box),
                        relation.phase,
                        relation.marked,
                        relation.incidences,
                        relation.update_keys,
                    )
                )
    package = Package(
        table.cells, tuple(facts), tuple(supports), table.source_obligations
    )
    # Independent structural checker validates exact partition coverage.
    from .compact_screen import validate

    validate(package)
    return package
