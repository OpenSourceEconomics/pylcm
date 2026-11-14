"""Independent native all-point checker for exact-implicit-fibre-table/v1.

Does NOT import implicit_core, implicit_native, their translator, or their decoder.
It directly calls the authoritative FibreSource APIs and independently walks the
serialized decision graphs at every original fibre.  It stores no point table or
point-pair graph. Completion may be expensive; interruption is never COMPLETE.
"""

from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast
from ..fibre_facts import JointReadSet
from ..fibre_source import _mapping, _rows, _strings, _text, Wire
from itertools import product
import argparse
import hashlib
import json
import sys
from .binding_purity import certify_binding_cache


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_key(value: object) -> object:
    """Python container equality alone aliases bool codes with integer codes."""
    if type(value) in (tuple, list):
        return (
            type(value).__name__,
            tuple(exact_key(x) for x in cast(Sequence[object], value)),
        )
    if type(value) in (set, frozenset):
        return (
            type(value).__name__,
            frozenset(
                exact_key(x) for x in cast(set[object] | frozenset[object], value)
            ),
        )
    if type(value) is dict:
        return (
            "dict",
            frozenset((exact_key(k), exact_key(v)) for k, v in value.items()),
        )
    return (type(value).__name__, value)


def same(left: object, right: object) -> bool:
    return exact_key(left) == exact_key(right)


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected a named record")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a sequence")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected an integer reference")
    return value


def decode(value: object) -> object:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Malformed leaf encoding")
    tag, payload = value
    if tag in ("str", "int", "bool"):
        if type(payload).__name__ != tag:
            raise ValueError("Typed leaf mismatch")
        return payload
    if tag in ("tuple", "list", "frozenset", "set"):
        decoded = [decode(item) for item in _sequence(payload)]
        if tag == "tuple":
            return tuple(decoded)
        if tag == "list":
            return decoded
        if len(set(decoded)) != len(decoded):
            raise ValueError("Duplicate set terminal member")
        if tag == "frozenset":
            return frozenset(decoded)
        return set(decoded)
    if tag == "ReadCertificate":
        items = _sequence(payload)
        if len(items) != 3:
            raise ValueError("Malformed read certificate")
        return ("read", *(decode(item) for item in items))
    if tag == "Support":
        items = _sequence(payload)
        if len(items) != 2:
            raise ValueError("Malformed support")
        return ("support", *(decode(item) for item in items))
    raise ValueError("Unexpected published terminal type: " + str(tag))


def check_graph(
    diagram: Mapping[str, object],
    domains: Sequence[tuple[str, Sequence[int]]],
    root_names: Iterable[str],
) -> Callable[[str, Sequence[int]], object]:
    if not same(diagram["domains"], [[name, list(codes)] for name, codes in domains]):
        raise ValueError("Original named domain/order/code identity changed")
    roots = {name: _integer(root) for name, root in _object(diagram["roots"]).items()}
    if set(roots) != set(root_names):
        raise ValueError("Missing or invented fact roots")
    nodes = tuple(_object(node) for node in _sequence(diagram["nodes"]))
    decoded: dict[int, object] = {}
    active: set[int] = set()
    completed: set[int] = set()

    def visit(node_id: int) -> None:
        if not 0 <= node_id < len(nodes):
            raise ValueError("Bad node reference")
        if node_id in active:
            raise ValueError("Cyclic decision graph")
        if node_id in completed:
            return
        active.add(node_id)
        node = nodes[node_id]
        if set(node) == {"leaf"}:
            decoded[node_id] = decode(node["leaf"])
        elif set(node) == {"axis", "children"}:
            axis = _integer(node["axis"])
            if not 0 <= axis < len(domains):
                raise ValueError("Bad axis index")
            children = tuple(_integer(child) for child in _sequence(node["children"]))
            if len(children) != len(domains[axis][1]):
                raise ValueError(
                    "Decision node does not cover entire original code domain"
                )
            for child in children:
                visit(child)
                if "axis" in nodes[child] and _integer(nodes[child]["axis"]) <= axis:
                    raise ValueError("Decision order not strict")
        else:
            raise ValueError("Unknown node shape")
        active.remove(node_id)
        completed.add(node_id)

    for root in roots.values():
        visit(root)
    if completed != set(range(len(nodes))):
        raise ValueError("Unreachable extra certificate nodes")

    def lookup(name: str, codes: Sequence[int]) -> object:
        root = roots[name]
        while "axis" in nodes[root]:
            node = nodes[root]
            axis = _integer(node["axis"])
            children = _sequence(node["children"])
            root = _integer(children[domains[axis][1].index(codes[axis])])
        return decoded[root]

    return lookup


def native_read(
    value: JointReadSet,
) -> tuple[str, frozenset[str], tuple[str, ...], tuple[str, ...]]:
    return ("read", value.live_coordinates, value.removal_order, value.removed_statuses)


def verify(*, snapshot: Path, axes_file: Path, artifact: Path) -> Wire:
    from _lcm_regime_finder.cli import load_problem
    from _lcm_regime_finder.fibre_source import FibreSource, _mapping, _text, _strings
    from _lcm_regime_finder.structural_store import final_input_check
    import _lcm_regime_finder

    package = Path(_lcm_regime_finder.__file__).parent
    packet = _mapping(json.loads(artifact.read_text()))
    if (
        packet.get("schema") != "exact-implicit-fibre-table/v1"
        or packet.get("reference_contract")
        != "compile_primitive_fibres.py@587478274ff55f1d873ae9d056bb9bf3e222b873"
    ):
        raise ValueError("Wrong schema or pointwise reference contract")
    guards = {path.name: sha(path) for path in sorted(package.glob("*.py"))}
    here = Path(__file__).resolve().parent
    pinned = json.loads((here / "reference_sources.json").read_text())
    if any(guards.get(name) != digest for name, digest in pinned.items()):
        raise ValueError("Independently pinned authoritative source bytes changed")
    if packet["reference_sources"] != guards:
        raise ValueError("Pointwise source bytes differ from producer")
    algorithm = {
        name: sha(here / name)
        for name in (
            "implicit_core.py",
            "implicit_native.py",
            "binding_purity.py",
            "reference_sources.json",
        )
    }
    if packet["algorithm_sources"] != algorithm:
        raise ValueError("Producer implementation bytes changed")
    snapshot_sha, axes_sha, artifact_sha = sha(snapshot), sha(axes_file), sha(artifact)
    if (
        packet["snapshot_sha256"] != snapshot_sha
        or packet["classification_sha256"] != axes_sha
    ):
        raise ValueError("Input identity mismatch")
    original = load_problem(snapshot)
    source = FibreSource(original)
    by_reference = {
        "snapshot_sha256": snapshot_sha,
        "cells": list(source._cells),
        "relations": list(source._relations),
        "domains": list(source._domains),
        "scope": "All original declaration fields, owners, laws, ages, phases and structural payloads remain in the pinned unchanged source; no field is projected away.",
    }
    if packet["metadata_by_reference"] != by_reference:
        raise ValueError("Whole original metadata reference is incomplete")
    status_axes = {
        name: _strings(value)
        for name, value in _mapping(json.loads(axes_file.read_text())).items()
    }
    regimes = {cell["regime"] for cell in source._cells.values()}
    if set(status_axes) != regimes:
        raise ValueError("Classification does not cover original regimes")
    if packet["source_obligations"] != list(source.source_obligations):
        raise ValueError("Source qualification lost or changed")
    cells = _rows(packet["cells"])
    if [x["cell_id"] for x in cells] != list(source.get_cell_ids()):
        raise ValueError("Original cell identity/order/coverage changed")
    counts = {}
    for row in cells:
        cell_id = _text(row["cell_id"])
        cell = source._cells[cell_id]
        states = _mapping(cell["states"])
        actions_map = _mapping(cell["actions"])
        regime = _text(cell["regime"])
        purity = certify_binding_cache(
            callbacks=source._callbacks,
            records=source._captures.records,
            resolve=source._captures.resolve,
        )
        expected_mode = (
            "pure-demand-diagram" if purity["proved"] else "original-order-stream"
        )
        if row["construction_mode"] != expected_mode:
            raise ValueError(
                "Cache-sensitive baseline was evaluated out of reference order"
            )
        if row["binding_cache_purity"] != purity:
            raise ValueError(
                "Binding cache purity claim differs from original source check"
            )
        if row["source_reference"] != {
            "cell_id": cell_id,
            "source_path": cell["source_path"],
        }:
            raise ValueError("Original cell source pointer changed")
        if (
            row["regime"] != cell["regime"]
            or not same(row["period"], cell["period"])
            or row["forced"] != []
        ):
            raise ValueError("Original cell or empty-forcing contract changed")
        declared = status_axes[regime]
        if len(set(declared)) != len(declared) or not set(declared) <= states.keys():
            raise ValueError("Bad original status classification")
        axes = tuple(name for name in states if name in declared)
        domains = tuple(
            (
                name,
                tuple(source._category(source._domains[_text(states[name])]).by_code),
            )
            for name in axes
        )
        actions = tuple(
            name
            for name, identity in actions_map.items()
            if source._domains[_text(identity)].get("kind") == "categorical"
        )
        outgoing = [
            (rid, relation)
            for rid, relation in source._relations.items()
            if relation["source"] == cell_id
            and relation["kind"]
            in {"transition", "joint", "value_read", "fallback", "exit"}
            and relation["targets"]
        ]
        metadata = [
            {
                key: relation[key]
                for key in ("id", "source", "targets", "phase", "kind", "state_updates")
                if key in relation
            }
            for _, relation in outgoing
        ]
        if row["relations"] != metadata:
            raise ValueError("Original incidence/phase/producer identity changed")
        roots = {"direct", *("exclude:" + action for action in actions)}
        for rid, relation in outgoing:
            target = source._cells[_strings(relation["targets"])[0]]
            roots.update(
                "read:" + rid + "::" + coordinate
                for coordinate in _mapping(target["states"])
            )
            roots.update(
                "support:" + rid + "::" + coordinate
                for coordinate in status_axes[_text(target["regime"])]
            )
        lookup = check_graph(_object(row["diagram"]), domains, roots)
        count = 0
        for codes in product(*(codes for _, codes in domains)):
            statuses = dict(zip(axes, codes))
            expected = native_read(
                source.get_direct_reads(
                    cell_id=cell_id, statuses=statuses, removal_order=axes
                )
            )
            if not same(lookup("direct", codes), expected):
                raise ValueError(("direct mismatch", cell_id, statuses))
            for action in actions:
                expected = source.get_excluded_action_values(
                    cell_id=cell_id,
                    action=action,
                    status_values={name: (code,) for name, code in statuses.items()},
                )
                if not same(lookup("exclude:" + action, codes), expected):
                    raise ValueError(
                        ("full-hull exclusion mismatch", cell_id, statuses, action)
                    )
            for rid, relation in outgoing:
                target = source._cells[_strings(relation["targets"])[0]]
                for coordinate in _mapping(target["states"]):
                    expected = native_read(
                        source.get_update_reads(
                            relation_id=rid,
                            target_coordinate=coordinate,
                            statuses=statuses,
                            removal_order=axes,
                        )
                    )
                    if not same(
                        lookup("read:" + rid + "::" + coordinate, codes), expected
                    ):
                        raise ValueError(
                            ("update mismatch", cell_id, statuses, rid, coordinate)
                        )
                for coordinate in status_axes[_text(target["regime"])]:
                    expected = source.get_target_support(
                        relation_id=rid, target_coordinate=coordinate, statuses=statuses
                    )
                    if not same(
                        lookup("support:" + rid + "::" + coordinate, codes),
                        ("support", expected.possible_codes, expected.proved),
                    ):
                        raise ValueError(
                            ("support mismatch", cell_id, statuses, rid, coordinate)
                        )
            count += 1
        counts[cell_id] = count
    final_input_check(original)
    if (
        sha(snapshot) != snapshot_sha
        or sha(axes_file) != axes_sha
        or sha(artifact) != artifact_sha
        or any(sha(package / name) != digest for name, digest in guards.items())
        or any(sha(here / name) != digest for name, digest in algorithm.items())
    ):
        raise ValueError("Source/certificate transaction changed")
    return {
        "schema": "exact-implicit-fibre-verification/v1",
        "status": "COMPLETE_FACT_EQUALITY",
        "source_obligations": list(source.source_obligations),
        "all_point_counts": counts,
        "artifact_sha256": artifact_sha,
        "snapshot_sha256": snapshot_sha,
        "reference_sources": guards,
        "qualification": "All original point facts compared. Support factor expansion, closure, finite projection premises and native adoption remain separate checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("axes", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if sys.version_info < (3, 14):
        raise SystemExit("Native verifier requires declared Python >=3.14; no shim")
    # A receipt is written only after full-domain replay and final guards finish.
    receipt = verify(
        snapshot=args.snapshot, axes_file=args.axes, artifact=args.artifact
    )
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
