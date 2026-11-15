"""Authenticated finite data for the pinned analyzer's cache-order boundary.

Record generation is portable JSON/hash work. native_observations is deliberately
not called during this audit: it is a metadata API control for Python >=3.14.
"""

from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.fibre_source import FibreSource, Wire
import hashlib
import json


def digest(value: JSONValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def nested_records(length: int = 33) -> tuple[dict[str, Wire], Wire, Wire]:
    records: dict[str, Wire] = {
        "B" + str(i): {
            "kind": "binding-record",
            "capture_ref": "B" + str(i),
            "capture_status": "COMPLETE",
            "binding_type": "builtins.tuple",
            "value": [{"kind": "callable-reference", "capture_ref": "B" + str(i + 1)}]
            if i + 1 < length
            else [0],
        }
        for i in range(length)
    }
    hashes = {name: digest(value) for name, value in records.items()}
    graph = digest(hashes)
    for name, record in records.items():
        record["record_sha256"] = hashes[name]
        record["capture_graph_sha256"] = graph
        record["identity_sha256"] = digest(
            {
                "capture_ref": name,
                "record_sha256": hashes[name],
                "capture_graph_sha256": graph,
            }
        )

    def reference(name: str) -> Wire:
        return {
            "kind": "callable-reference",
            "capture_ref": name,
            "identity_sha256": records[name]["identity_sha256"],
            "capture_graph_sha256": graph,
        }

    for index in range(length - 1):
        records["B" + str(index)]["value"] = [reference("B" + str(index + 1))]
    return records, reference("B0"), reference("B1")


def native_fixture() -> Wire:
    from tests.regime_finder.test_fibre_source import _original, _fields
    from tests.regime_finder.test_model_adapter import function

    original = _original()
    records, long, short = nested_records()
    original["callable_records"] = records
    transitions = _fields(original)["state_transitions"]
    assert isinstance(transitions, dict)
    transitions["x"] = function(
        "def next_x():\n    return B" + "[0]" * 33 + "\n", [], {"B": long}
    )
    transitions["history"] = function(
        "def next_history():\n    return B" + "[0]" * 32 + "\n", [], {"B": short}
    )
    return original


def native_observations() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    from _lcm_regime_finder.fibre_source import FibreSource

    original = native_fixture()
    source = FibreSource(original)
    first = source.get_cell_ids()[0]
    rid = next(
        rid
        for rid, record in source._relations.items()
        if record["source"] == first
        and record["kind"] == "transition"
        and record.get("phase") == "solve"
    )

    def read(coordinate: str) -> frozenset[str]:
        return source.get_update_reads(
            relation_id=rid, target_coordinate=coordinate, statuses={}, removal_order=()
        ).live_coordinates

    cold = read("x")
    warming = read("history")
    warm = read("x")
    return cold, warming, warm
