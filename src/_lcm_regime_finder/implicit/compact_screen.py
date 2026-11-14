"""Portable exact box algebra for original fibres and the finite pre-screen.

This module imports no pylcm code and never calls a Model or a callback.  It checks
supplied source-bound facts.  Full-domain coverage is an independent structural
claim; source authenticity is deliberately not inferred from structural coverage.

The declared order and nominal integer codes identify original points.  Equal fact
payloads share storage, not addresses.  No operation below loops over the Cartesian
product of original point pairs.  A difficult relation may nevertheless require
exponentially many boxes; this is a representation limitation, not an impossibility
claim about a captured input.
"""

from dataclasses import dataclass
from math import prod
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict, cast
from ..records import JSONValue


type Codes = frozenset[int]
type Box = tuple[Codes, ...]
PREMISES = frozenset({"I1", "I2", "I3", "I4", "I5", "a", "b", "c"})


def _intersect_box(left: Box, right: Box) -> Box | None:
    if len(left) != len(right):
        raise ValueError("Box dimensions differ")
    answer = tuple(a & b for a, b in zip(left, right))
    return answer if all(answer) else None


def _subtract_box(left: Box, right: Box) -> tuple[Box, ...]:
    """Disjoint axis-by-axis partition of left minus right, including 0-D."""
    overlap = _intersect_box(left, right)
    if overlap is None:
        return (left,)
    inside = list(left)
    answer = []
    for index, intersection in enumerate(overlap):
        outside = inside[index] - intersection
        if outside:
            piece = list(inside)
            piece[index] = outside
            answer.append(tuple(piece))
        inside[index] = intersection
    return tuple(answer)


@dataclass(frozen=True)
class Region:
    """A disjoint union of boxes over one explicitly declared finite universe."""

    domains: Box
    boxes: tuple[Box, ...] = ()

    @classmethod
    def make(cls, domains: Box, boxes: Iterable[Box] = ()) -> Region:
        if any(
            not domain or any(type(code) is not int for code in domain)
            for domain in domains
        ):
            raise ValueError("Universes need nonempty distinct integer-code sets")
        output: list[Box] = []
        for box in boxes:
            if len(box) != len(domains):
                raise ValueError("Box dimension does not match universe")
            if any(
                not codes
                or not codes <= domain
                or any(type(code) is not int for code in codes)
                for codes, domain in zip(box, domains)
            ):
                raise ValueError("Box contains an empty or out-of-domain code set")
            pieces = [box]
            for old in output:
                pieces = [
                    piece
                    for candidate in pieces
                    for piece in _subtract_box(candidate, old)
                ]
                if not pieces:
                    break
            output.extend(pieces)
        return cls(domains, tuple(output))

    @classmethod
    def full(cls, domains: Box) -> Region:
        return cls.make(domains, (domains,))

    def _same(self, other: Region) -> None:
        if self.domains != other.domains:
            raise ValueError("Region universes differ")

    def union(self, other: Region) -> Region:
        self._same(other)
        return Region.make(self.domains, self.boxes + other.boxes)

    def intersection(self, other: Region) -> Region:
        self._same(other)
        return Region.make(
            self.domains,
            (
                intersection
                for a in self.boxes
                for b in other.boxes
                if (intersection := _intersect_box(a, b)) is not None
            ),
        )

    def difference(self, other: Region) -> Region:
        self._same(other)
        pieces = list(self.boxes)
        for right in other.boxes:
            pieces = [piece for left in pieces for piece in _subtract_box(left, right)]
        return Region.make(self.domains, pieces)

    def equal(self, other: Region) -> bool:
        self._same(other)
        return not self.difference(other).boxes and not other.difference(self).boxes

    def project(self, indices: Sequence[int]) -> Region:
        if len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < len(self.domains) for i in indices
        ):
            raise ValueError("Projection indices must be distinct original axes")
        return Region.make(
            tuple(self.domains[i] for i in indices),
            (tuple(box[i] for i in indices) for box in self.boxes),
        )

    def lift(self, domains: Box, positions: Sequence[int]) -> Region:
        if len(positions) != len(self.domains) or len(set(positions)) != len(positions):
            raise ValueError("Lift must place each axis exactly once")
        if any(type(p) is not int or not 0 <= p < len(domains) for p in positions):
            raise ValueError("Lift position is outside universe")
        if any(self.domains[i] != domains[p] for i, p in enumerate(positions)):
            raise ValueError("Lift changes an original code domain")
        answer = []
        for box in self.boxes:
            lifted = list(domains)
            for p, codes in zip(positions, box):
                lifted[p] = codes
            answer.append(tuple(lifted))
        return Region.make(domains, answer)

    def saturate(self, retained: Sequence[int]) -> Region:
        return self.project(retained).lift(self.domains, retained)

    def contains(self, point: Sequence[int]) -> bool:
        if len(point) != len(self.domains):
            return False
        return any(
            all(code in values for code, values in zip(point, box))
            for box in self.boxes
        )

    @property
    def cardinality(self) -> int:
        return sum(prod(len(codes) for codes in box) for box in self.boxes)

    def witness(self) -> tuple[int, ...] | None:
        return tuple(min(codes) for codes in self.boxes[0]) if self.boxes else None


@dataclass(frozen=True)
class ReadCertificate:
    live: frozenset[str]
    removal_order: tuple[str, ...] = ()
    removed_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cell:
    identifier: str
    regime: str
    status_domains: tuple[tuple[str, tuple[int, ...]], ...]
    coordinates: frozenset[str]
    read_keys: tuple[str, ...] = ("direct",)
    action_domains: tuple[tuple[str, tuple[int, ...]], ...] = ()
    admissible_start: bool = False

    @property
    def domains(self) -> Box:
        return tuple(frozenset(codes) for _, codes in self.status_domains)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.status_domains)

    def address(self, point: Sequence[int]) -> str:
        """Exactly the cell:index convention of build_source_screen."""
        if len(point) != len(self.status_domains):
            raise ValueError("Wrong point dimension")
        index = 0
        for code, (_, codes) in zip(point, self.status_domains):
            if type(code) is not int or code not in codes:
                raise ValueError("Unknown nominal code")
            index = index * len(codes) + codes.index(code)
        return f"{self.identifier}:{index}"


@dataclass(frozen=True)
class FactRegion:
    cell: str
    box: Box
    reads: tuple[tuple[str, ReadCertificate], ...]
    forced: frozenset[str] = frozenset()
    excluded: tuple[tuple[str, frozenset[int]], ...] = ()

    @property
    def read_values(self) -> dict[str, frozenset[str]]:
        return {key: value.live for key, value in self.reads}


@dataclass(frozen=True)
class SupportRegion:
    source: str
    target: str
    box: Box
    phase: str
    marked: bool
    incidences: tuple[str, ...]
    # None is an explicit, source-verified missing producer, whose established
    # conservative update reads every source coordinate. It is not an identity.
    update_keys: tuple[tuple[str, str | None], ...] = ()

    @property
    def label(self) -> tuple[str, str, str, bool, tuple[str, ...]]:
        return self.source, self.target, self.phase, self.marked, self.incidences


@dataclass(frozen=True)
class Package:
    cells: tuple[Cell, ...]
    facts: tuple[FactRegion, ...]
    supports: tuple[SupportRegion, ...]
    source_obligations: tuple[JSONValue, ...] = ()


def validate(package: Package) -> dict[str, Cell]:
    """Check exact coverage, inventory, domains, and incidence labels independently.

    Cell schemas must be bound to authenticated original declarations by the source
    verifier.  This routine never declares them authenticated merely because they
    are internally consistent.
    """
    cells = {cell.identifier: cell for cell in package.cells}
    if len(cells) != len(package.cells) or any(not key for key in cells):
        raise ValueError("Cell identifiers must be unique and nonempty")
    regime_domains: dict[str, dict[str, Codes]] = {}
    for cell in package.cells:
        if not cell.regime or len(set(cell.names)) != len(cell.names):
            raise ValueError("Regime/name inventory malformed")
        if not set(cell.names) <= cell.coordinates:
            raise ValueError("Status axis is not an original state coordinate")
        for _, codes in cell.status_domains + cell.action_domains:
            if (
                not codes
                or len(set(codes)) != len(codes)
                or any(type(c) is not int for c in codes)
            ):
                raise ValueError("Domains need distinct nominal integer codes")
        mapping = {name: frozenset(codes) for name, codes in cell.status_domains}
        if regime_domains.setdefault(cell.regime, mapping) != mapping:
            raise ValueError("Original status domains differ across regime cells")
        if (
            len(set(cell.read_keys)) != len(cell.read_keys)
            or "direct" not in cell.read_keys
        ):
            raise ValueError("Read keys need one direct entry and no duplicates")
        if len(dict(cell.action_domains)) != len(cell.action_domains):
            raise ValueError("Action names must be unique")
    covered = {key: Region.make(cell.domains) for key, cell in cells.items()}
    for fact in package.facts:
        if fact.cell not in cells:
            raise ValueError("Unknown fact cell")
        cell = cells[fact.cell]
        area = Region.make(cell.domains, (fact.box,))
        if covered[fact.cell].intersection(area).boxes:
            raise ValueError("Fact boxes must partition, not overlap")
        covered[fact.cell] = covered[fact.cell].union(area)
        if len(dict(fact.reads)) != len(fact.reads) or set(dict(fact.reads)) != set(
            cell.read_keys
        ):
            raise ValueError("Every original primitive/update read key is required")
        for _, certificate in fact.reads:
            if not certificate.live <= cell.coordinates:
                raise ValueError("Read certificate exceeds original coordinates")
            if len(set(certificate.removal_order)) != len(certificate.removal_order):
                raise ValueError("Duplicate removal-order coordinate")
            if not set(certificate.removal_order) <= set(cell.names):
                raise ValueError("Removal order contains a nonstatus coordinate")
            if (
                tuple(
                    n
                    for n in certificate.removal_order
                    if n in certificate.removed_statuses
                )
                != certificate.removed_statuses
            ):
                raise ValueError("Removed-status certificate changes selection order")
        if len(dict(fact.excluded)) != len(fact.excluded) or set(
            dict(fact.excluded)
        ) != set(dict(cell.action_domains)):
            raise ValueError("Every categorical action exclusion is required")
        if any(
            any(type(code) is not int for code in codes)
            or not codes <= frozenset(dict(cell.action_domains)[name])
            for name, codes in fact.excluded
        ):
            raise ValueError("Excluded value is outside original action codes")
    for key, cell in cells.items():
        if not covered[key].equal(Region.full(cell.domains)):
            raise ValueError("Fact partition omits original full-hull fibres")
    for support in package.supports:
        if support.source not in cells or support.target not in cells:
            raise ValueError("Support references an unknown original cell")
        Region.make(
            cells[support.source].domains + cells[support.target].domains,
            (support.box,),
        )
        if (
            support.phase not in {"solve", "simulation", "both"}
            or type(support.marked) is not bool
        ):
            raise ValueError("Invalid phase/mark")
        if not isinstance(support.incidences, tuple) or any(
            not isinstance(r, str) or not r for r in support.incidences
        ):
            raise ValueError("Incidences must retain their original ordered tuple")
        if support.update_keys and (
            len(dict(support.update_keys)) != len(support.update_keys)
            or set(dict(support.update_keys)) != cells[support.target].coordinates
            or not {
                key for key in dict(support.update_keys).values() if key is not None
            }
            <= set(cells[support.source].read_keys)
        ):
            raise ValueError(
                "Explicit updates must bind every target coordinate to one original read key"
            )
    return cells


def relations(
    package: Package, cells: Mapping[str, Cell]
) -> dict[tuple[str, str, str, bool, tuple[str, ...]], Region]:
    answer: dict[tuple, Region] = {}
    for edge in package.supports:
        domains = cells[edge.source].domains + cells[edge.target].domains
        old = answer.get(edge.label, Region.make(domains))
        answer[edge.label] = old.union(Region.make(domains, (edge.box,)))
    return answer


def possible_closure(package: Package) -> dict[str, Region]:
    """Original initial overapproximations and least possible-support closure."""
    cells = validate(package)
    edges = relations(package, cells)
    kept = {
        key: Region.full(cell.domains)
        if cell.admissible_start
        else Region.make(cell.domains)
        for key, cell in cells.items()
    }
    while True:
        updated = dict(kept)
        for (source, target, *_), relation in edges.items():
            ns = len(cells[source].domains)
            reached = relation.intersection(
                kept[source].lift(relation.domains, range(ns))
            )
            image = reached.project(range(ns, len(relation.domains)))
            updated[target] = updated[target].union(image)
        if all(updated[key].equal(kept[key]) for key in cells):
            return updated
        kept = updated


def _member_facts(
    package: Package, cells: Mapping[str, Cell], key: str, coordinate: str
) -> dict[str, Region]:
    answer = {name: Region.make(cell.domains) for name, cell in cells.items()}
    for fact in package.facts:
        read = dict(fact.reads).get(key)
        if read is not None and coordinate in read.live:
            answer[fact.cell] = answer[fact.cell].union(
                Region.make(cells[fact.cell].domains, (fact.box,))
            )
    return answer


def least_liveness(package: Package) -> dict[tuple[str, str], Region]:
    """Close the fixed pointwise certificates; never choose certificates anew."""
    cells = validate(package)
    edges = relations(package, cells)
    update_keys = {}
    for support in package.supports:
        keys = (
            dict(support.update_keys)
            if support.update_keys
            else {
                coordinate: "update:" + ":".join(support.incidences) + ":" + coordinate
                for coordinate in cells[support.target].coordinates
            }
        )
        if update_keys.setdefault(support.label, keys) != keys:
            raise ValueError(
                "Competing update-read bindings for one original support incidence"
            )
    live = {
        (key, coordinate): Region.make(cell.domains)
        for key, cell in cells.items()
        for coordinate in cell.coordinates
    }
    for fact in package.facts:
        for coordinate in dict(fact.reads)["direct"].live:
            key = fact.cell, coordinate
            live[key] = live[key].union(
                Region.make(cells[fact.cell].domains, (fact.box,))
            )
    while True:
        updated = dict(live)
        for label, relation in edges.items():
            source, target, _phase, _mark, incidences = label
            ns = len(cells[source].domains)
            target_positions = tuple(range(ns, len(relation.domains)))
            for target_coordinate in cells[target].coordinates:
                read_key = update_keys[label][target_coordinate]
                if read_key is not None and read_key not in cells[source].read_keys:
                    raise ValueError("Missing original coordinate update certificate")
                needed = relation.intersection(
                    live[target, target_coordinate].lift(
                        relation.domains, target_positions
                    )
                ).project(range(ns))
                for fact in package.facts:
                    if fact.cell != source:
                        continue
                    relevant = needed.intersection(
                        Region.make(cells[source].domains, (fact.box,))
                    )
                    source_reads = (
                        cells[source].coordinates
                        if read_key is None
                        else dict(fact.reads)[read_key].live
                    )
                    for coordinate in source_reads:
                        updated[source, coordinate] = updated[source, coordinate].union(
                            relevant
                        )
        if all(updated[key].equal(live[key]) for key in live):
            return updated
        live = updated


def _diagonal_preimage(
    relation: Region, source: Cell, target: Cell, removed_names: frozenset[str]
) -> Region:
    """Pull back equality of removed *named* coordinates without a dense diagonal.

    Output axes are all source axes then retained target axes.  The removed target
    code sets intersect their namesake source code sets.  Domain values are never
    compared or merged; only original nominal codes enter this operation.
    """
    ns = len(source.names)
    target_kept = tuple(
        i for i, name in enumerate(target.names) if name not in removed_names
    )
    domains = source.domains + tuple(target.domains[i] for i in target_kept)
    boxes = []
    for box in relation.boxes:
        left = list(box[:ns])
        for j, name in enumerate(target.names):
            if name in removed_names:
                i = source.names.index(name)
                left[i] = left[i] & box[ns + j]
        result = tuple(left) + tuple(box[ns + j] for j in target_kept)
        if all(result):
            boxes.append(result)
    return Region.make(domains, boxes)


class ScreenResult(TypedDict):
    schema: str
    proof_scope: str
    premises: dict[str, dict[str, JSONValue]]
    conditions_pass: bool
    source_authentication: str
    source_obligations: list[JSONValue]
    transformation_authorized: bool
    forcing: str
    forced_action_savings: str
    kept_cardinality_by_cell: dict[str, int]
    projection: dict[str, dict[str, JSONValue]]


def check_screen(
    package: Package,
    removed: frozenset[tuple[str, str]],
    *,
    kept_override: Mapping[str, Region] | None = None,
) -> ScreenResult:
    """Exact finite pre-screen on supplied facts, keeping source guards separate.

    kept_override exists for independent tests of the finite supplied-fact checker.
    The source-bound integration must omit it and use possible_closure unchanged.
    """
    cells = validate(package)
    local_axes = {(cell.regime, name) for cell in cells.values() for name in cell.names}
    if not removed <= local_axes:
        raise ValueError("Removed axes are not original regime status axes")
    kept = (
        dict(kept_override) if kept_override is not None else possible_closure(package)
    )
    if kept.keys() != cells.keys() or any(
        kept[key].domains != cell.domains for key, cell in cells.items()
    ):
        raise ValueError("Kept regions must cover the exact original cell inventory")
    retain = {
        key: tuple(
            i for i, name in enumerate(cell.names) if (cell.regime, name) not in removed
        )
        for key, cell in cells.items()
    }
    statuses = {name: "proved" for name in PREMISES}
    witnesses: dict[str, dict[str, JSONValue]] = {}

    def refute(name: str, witness: dict[str, JSONValue]) -> None:
        if statuses[name] == "proved":
            statuses[name] = "refuted"
            witnesses[name] = witness

    for key, cell in cells.items():
        missing = kept[key].saturate(retain[key]).difference(kept[key])
        if missing.boxes:
            witness = missing.witness()
            assert witness is not None
            refute("I1", {"cell": key, "unkept_part": cell.address(witness)})
        facts = tuple(f for f in package.facts if f.cell == key)
        for i, left in enumerate(facts):
            l = Region.make(cell.domains, (left.box,))
            for right in facts[i + 1 :]:
                r = Region.make(cell.domains, (right.box,))
                if left.read_values != right.read_values or left.forced != right.forced:
                    clash = (
                        l.intersection(kept[key])
                        .project(retain[key])
                        .intersection(r.intersection(kept[key]).project(retain[key]))
                    )
                    if clash.boxes:
                        refute(
                            "I2",
                            {
                                "cell": key,
                                "kind": "kept_reads_or_forcing",
                                "projected_codes": clash.witness(),
                            },
                        )
                if dict(left.excluded) != dict(right.excluded):
                    clash = l.project(retain[key]).intersection(r.project(retain[key]))
                    if clash.boxes:
                        refute(
                            "I2",
                            {
                                "cell": key,
                                "kind": "full_hull_exclusion",
                                "projected_codes": clash.witness(),
                            },
                        )
    statuses["b"] = statuses["I2"]
    if "I2" in witnesses:
        witnesses["b"] = witnesses["I2"]
    edges = relations(package, cells)
    marked: dict[tuple[str, str], Region] = {}
    for (source, target, phase, mark, incidences), relation in edges.items():
        ns = len(cells[source].domains)
        selected = tuple(range(ns)) + tuple(ns + j for j in retain[target])
        projected_target = relation.project(selected)
        source_kept = kept[source].lift(projected_target.domains, range(ns))
        restricted = projected_target.intersection(source_kept)
        retained_both = retain[source] + tuple(range(ns, len(selected)))
        missing = (
            restricted.saturate(retained_both)
            .intersection(source_kept)
            .difference(restricted)
        )
        if missing.boxes:
            refute(
                "I3",
                {
                    "source": source,
                    "target": target,
                    "phase": phase,
                    "marked": mark,
                    "incidences": incidences,
                    "codes": missing.witness(),
                },
            )
        if mark and cells[source].regime == cells[target].regime:
            key = source, target
            marked[key] = marked.get(key, Region.make(relation.domains)).union(relation)
    for (source, target), relation in marked.items():
        sc, tc = cells[source], cells[target]
        ns = len(sc.names)
        live_relation = relation.intersection(
            kept[source].lift(relation.domains, range(ns))
        )
        live_relation = live_relation.intersection(
            kept[target].lift(relation.domains, range(ns, len(relation.domains)))
        )
        coarse = live_relation.project(
            retain[source] + tuple(ns + j for j in retain[target])
        )
        removed_names = frozenset(
            name for regime, name in removed if regime == sc.regime
        )
        diagonal = _diagonal_preimage(relation, sc, tc, removed_names)
        positions = retain[source] + tuple(range(ns, len(diagonal.domains)))
        expected = coarse.lift(diagonal.domains, positions)
        missing = expected.difference(diagonal)
        if missing.boxes:
            refute(
                "I4",
                {
                    "source": source,
                    "target": target,
                    "equal_slice_codes": missing.witness(),
                },
            )
    if any(
        fact.forced
        and Region.make(cells[fact.cell].domains, (fact.box,))
        .intersection(kept[fact.cell])
        .boxes
        for fact in package.facts
    ):
        for name in ("I4", "I5", "c"):
            if statuses[name] != "refuted":
                statuses[name] = "unknown"
    regimes = {cell.regime for cell in cells.values()}
    names = {name for _, name in removed}
    literal = (
        len(names) == 1
        and removed == frozenset((r, n) for r in regimes for n in names)
        and all(names <= set(cell.names) for cell in cells.values())
    )
    scope = "identity" if not removed else "inert-literal" if literal else "inert-joint"
    premises: dict[str, dict[str, JSONValue]] = {
        name: {"status": statuses[name], "witness": witnesses.get(name)}
        for name in sorted(PREMISES)
    }
    return {
        "schema": "independent-compact-screen/v1",
        "proof_scope": scope,
        "premises": premises,
        "conditions_pass": verdict_passes(premises),
        "source_authentication": "not_established_by_this_checker",
        "source_obligations": list(package.source_obligations),
        "transformation_authorized": False,
        "forcing": "no forced-action facts"
        if not any(f.forced for f in package.facts)
        else "nonempty forcing needs separate evidence",
        "forced_action_savings": "not measured",
        "kept_cardinality_by_cell": {
            key: region.cardinality for key, region in kept.items()
        },
        "projection": {
            key: {
                "retained_axes": [cells[key].names[i] for i in retain[key]],
                "representative_rule": "original first declared code in every removed axis",
            }
            for key in cells
        },
    }


def verdict_passes(premises: Mapping[str, JSONValue]) -> bool:
    return premises.keys() == PREMISES and all(
        isinstance(value, Mapping) and value.get("status") == "proved"
        for value in premises.values()
    )


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
