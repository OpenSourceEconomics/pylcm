"""Compact-capture regressions: metadata only, never Model or Bellman execution."""

from tests.regime_finder._callable_records import validate_callable_records

import pytest
from copy import deepcopy
from dataclasses import dataclass
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import patch

from _lcm_regime_finder.callable_identity import (
    CaptureIncomplete,
    CaptureSession,
    capture_session,
    resolve_callable,
    resolve_binding,
)
from _lcm_regime_finder.model_snapshot import (
    read_declarations,
    read_model,
    source_function,
)


def leaf(history):
    return history


def alternate(history):
    return -history


def shared_layer(child):
    left = right = child

    def callback(history):
        return left(history) + right(history)

    return callback


def callback_with_binding(offset):
    def callback(history):
        return history + offset

    return callback


def recursive_left(history):
    return recursive_right(history)


def recursive_right(history):
    return recursive_left(history)


_MUTABLE = [1, 2]
_NAMESPACE = SimpleNamespace(operation=leaf)


def reads_mutable(history):
    return history + _MUTABLE[0]


def reads_module_attribute(history):
    return _NAMESPACE.operation(history)


@dataclass
class Receiver:
    coefficient: int

    def callback(self, history):
        return history + self.coefficient


class FakeModel:
    description = "explicit metadata fixture, not an economic Model"
    n_periods = 2
    ages = None
    initial_nodes = [(0, "original")]
    regime_names_to_ids = {"original": 0}
    fixed_params = {}
    reachability = {"nodes": [(0, "original"), (1, "original")]}
    graph = {"initial_nodes": initial_nodes, "reachability": reachability}

    def __init__(self):
        self.user_regimes = {
            "original": {
                "functions": {"utility": shared_layer(leaf)},
                "states": {"history": {"n_points": 10**18}},
            }
        }


@dataclass
class SharedTables:
    offset: int
    values: list[int]


def bound_table_callback(tables, shift):
    def callback(history):
        return history + tables.offset + tables.values[0] + shift

    return callback


def binding_payload_callback(payload):
    def callback(history):
        return history if payload is None else history + 1

    return callback


def test_missing_process_tree_telemetry_uses_labelled_self_peak() -> None:
    with (
        patch("_lcm_regime_finder.resources._tree_rss", return_value=(0, 1)),
        patch(
            "_lcm_regime_finder.callable_identity.resource.getrusage",
            return_value=SimpleNamespace(
                ru_maxrss=32 * 1024**2 if sys.platform == "darwin" else 32 * 1024
            ),
        ),
    ):
        session = CaptureSession({"host_gib": 8})
        session.check(force=True)
    report = session.report()
    assert not (report["process_tree_rss_available"])
    assert (report["peak_process_tree_gib"]) is None
    assert (report["peak_observed_host_gib"]) == (32 / 1024)
    assert (report["resource_status"]) == ("UNVERIFIED_PROCESS_TREE")


def test_self_peak_enforces_memory_guard_when_proc_is_unavailable() -> None:
    with (
        patch("_lcm_regime_finder.resources._tree_rss", return_value=(0, 1)),
        patch(
            "_lcm_regime_finder.callable_identity.resource.getrusage",
            return_value=SimpleNamespace(
                ru_maxrss=9 * 1024**3 if sys.platform == "darwin" else 9 * 1024**2
            ),
        ),
    ):
        session = CaptureSession({"host_gib": 8})
        with pytest.raises(CaptureIncomplete):
            session.check(force=True)
    assert (session.exhaustion_reason) == ("process_tree_memory")
    assert (session.report()["status"]) == ("INCOMPLETE")


def test_shared_dag_grows_with_unique_nodes_not_paths() -> None:
    sizes = []
    for depth in (8, 32, 80):
        callback = leaf
        for _ in range(depth):
            callback = shared_layer(callback)
        captured = source_function(callback)
        records = captured["callable_records"]
        assert (len(records)) == (depth + 1)
        assert validate_callable_records(records)
        assert all(
            record["capture_status"] == "COMPLETE" for record in records.values()
        )
        assert all(
            "depth-limited" not in str(record.get("obligations"))
            for record in records.values()
        )
        sizes.append(len(json.dumps(captured)))
        assert (sizes[-1]) < (8000 * (depth + 1))
    assert (sizes[-1]) < (sizes[0] * 12)


def test_recursive_callables_are_finite_shared_graph_not_truncated_tree() -> None:
    captured = source_function(recursive_left)
    records = captured["callable_records"]
    assert (len(records)) == (2)
    assert validate_callable_records(records)
    child = resolve_callable(captured["call_targets"]["recursive_right"], records)
    back = resolve_callable(child["call_targets"]["recursive_left"], records)
    assert (back["capture_ref"]) == (captured["capture_ref"])
    assert ("source") in (child)
    assert ("recursive/depth-limited") not in (str(records).lower())
    from _lcm_regime_finder.semantic_check import SourceProof

    checked = SourceProof({}).callback(captured)
    assert not (checked.proved)


def test_distinct_partial_bindings_share_only_actual_function() -> None:
    captured = read_declarations(
        {
            "first": functools.partial(leaf, history=1),
            "second": functools.partial(leaf, history=2),
        }
    )
    first, second = captured["first"], captured["second"]
    assert (first["capture_ref"]) != (second["capture_ref"])
    assert (first["binding_sha256"]) != (second["binding_sha256"])
    assert (
        (first["partial_target"]["capture_ref"])
        == (second["partial_target"]["capture_ref"])
    )
    assert (first["partial_bindings"]["history"]) == (1)
    assert (second["partial_bindings"]["history"]) == (2)
    assert validate_callable_records(captured["callable_records"])


def test_repeated_bound_method_identity_interns_function_and_receiver() -> None:
    receiver = Receiver(4)
    captured = read_declarations(
        {"first": receiver.callback, "second": receiver.callback}
    )
    assert (captured["first"]) is (captured["second"])
    assert (len(captured["callable_records"])) == (1)
    other = source_function(Receiver(5).callback)
    assert (captured["first"]["binding_sha256"]) != (other["binding_sha256"])


def test_shared_structured_binding_is_serialized_once_across_callbacks() -> None:
    tables = SharedTables(3, list(range(8192)))
    captured = read_declarations(
        {"callbacks": [bound_table_callback(tables, shift) for shift in range(80)]}
    )
    records = captured["callable_records"]
    bindings = {
        ref: record
        for ref, record in records.items()
        if record["kind"] == "binding-record"
    }
    assert (len(bindings)) == (2)  # Tables and its one actual list.
    assert (len(records)) == (82)
    roots = captured["callbacks"]
    refs = {root["bindings"]["tables"]["capture_ref"] for root in roots}
    assert (len(refs)) == (1)
    decoded = resolve_binding(roots[0]["bindings"]["tables"], records)
    assert (decoded["fields"]["values"]) == (list(range(8192)))
    assert validate_callable_records(records)
    assert (len(json.dumps(captured))) < (600000)


def test_shared_data_dag_is_linear_and_cycle_is_explicit() -> None:
    payload = [1]
    for _ in range(80):
        payload = [payload, payload]
    captured = source_function(binding_payload_callback(payload))
    records = captured["callable_records"]
    assert (sum(record["kind"] == "binding-record" for record in records.values())) == (
        81
    )
    assert (len(json.dumps(captured))) < (100000)
    assert validate_callable_records(records)
    cyclic = []
    cyclic.append(cyclic)
    captured = source_function(binding_payload_callback(cyclic))
    assert validate_callable_records(captured["callable_records"])
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_binding(captured["bindings"]["payload"], captured["callable_records"])


def test_nested_binding_mutation_invalidates_dependent_callbacks() -> None:
    tables = SharedTables(3, [1, 2])
    with capture_session() as session:
        captured = source_function(bound_table_callback(tables, 4))
        tables.values[0] = 999
    assert (captured["capture_status"]) == ("UNRESOLVED_MUTATION")
    assert (session.report()["status"]) == ("INCOMPLETE")
    assert validate_callable_records(session.records)


def test_binding_container_and_array_shapes_are_preserved() -> None:
    captured = source_function(binding_payload_callback((1, 2)))
    resolved = resolve_binding(
        captured["bindings"]["payload"], captured["callable_records"]
    )
    assert isinstance(resolved, tuple)
    assert (resolved) == ((1, 2))
    try:
        import numpy as np
    except ImportError:
        return
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    captured = source_function(binding_payload_callback(array))
    resolved = resolve_binding(
        captured["bindings"]["payload"], captured["callable_records"]
    )
    assert (resolved["shape"]) == ([2, 3])
    assert (resolved["dtype"]) == ("int32")
    assert (resolved["values"]) == ([[0, 1, 2], [3, 4, 5]])
    with capture_session():
        changed = source_function(binding_payload_callback(array))
        array[0, 1] = 999
    assert (changed["capture_status"]) == ("UNRESOLVED_MUTATION")


def test_unsupported_library_is_small_explicit_boundary() -> None:
    captured = source_function(math.sqrt)
    assert (captured["kind"]) == ("callable")
    assert captured["source_unavailable"]
    assert ("opaque_boundary") in (captured)
    assert captured["obligations"]
    assert ("operation") not in (captured)
    assert (len(json.dumps(captured))) < (16000)
    assert (len(captured["callable_records"])) == (1)


def test_numpy_sqrt_and_clip_are_not_recursive_library_trees(
    *, subtests: pytest.Subtests
) -> None:
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy optional; original JAX native probes remain separate")
    for target in (np.sqrt, np.clip):
        with subtests.test(target=target.__name__):
            captured = source_function(target)
            assert ("opaque_boundary") in (captured)
            assert (len(json.dumps(captured))) < (32000)
            assert (len(captured["callable_records"])) <= (4)


def test_wrapper_is_not_unwrapped_to_advertised_body() -> None:
    @functools.wraps(leaf)
    def wrapper(history):
        return -history

    captured = source_function(wrapper)
    assert ("return -history") in (captured["source"])
    assert (captured["code_sha256"]) != (source_function(leaf)["code_sha256"])
    from _lcm_regime_finder.declarations import ScalarSource

    expression = ScalarSource(captured).expression()
    assert ("-history") in (str(expression))


def test_wrapper_defaults_come_from_actual_wrapper() -> None:
    def original(history=1):
        return history

    @functools.wraps(original)
    def wrapper(history=7):
        return -history

    captured = source_function(wrapper)
    assert (captured["defaults"]["history"]) == (7)
    partial = source_function(functools.partial(wrapper))
    assert (partial["defaults"]["history"]) == (7)
    assert ("return -history") in (captured["source"])


def test_rebound_library_export_does_not_assign_primitive_by_spelling() -> None:
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy optional")
    original = np.where
    try:
        setattr(np, "where", np.clip)
        captured = source_function(np.where)
        assert (captured.get("kind")) != ("primitive")

        @functools.wraps(original)
        def wrapper(*arguments, **keywords):
            return 0

        setattr(np, "where", wrapper)
        captured = source_function(wrapper)
        assert (captured.get("kind")) != ("primitive")
        assert ("return 0") in (captured["source"])
    finally:
        setattr(np, "where", original)


def test_export_model_keeps_original_graph_and_hash_bound_receipt() -> None:
    from _lcm_regime_finder.pytask_api import export_model

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        receipt = export_model(
            FakeModel(),
            path,
            production_metadata={
                "status": "complete",
                "scope": "explicit metadata fixture",
            },
            seconds=10,
            host_gib=8,
        )
        assert (receipt["status"]) == ("COMPLETE")
        assert (receipt["snapshot_sha256"]) == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
        saved = json.loads(path.read_text())
        assert (saved["user_regimes"]["original"]["states"]["history"]["n_points"]) == (
            10**18
        )
        assert validate_callable_records(saved["callable_records"])
        checkpoint = json.loads(path.with_name("model.original-graph.json").read_text())
        assert (checkpoint["stage"]) == ("original_graph")
        assert (checkpoint["snapshot"]["initial_nodes"]) == ([[0, "original"]])
        assert (
            json.loads(path.with_name("model.metadata-capture.json").read_text())
        ) == (receipt)


def test_mutation_during_capture_invalidates_callback() -> None:
    try:
        with capture_session():
            captured = source_function(reads_mutable)
            _MUTABLE[0] = 999
        assert (captured["capture_status"]) == ("UNRESOLVED_MUTATION")
        assert any("mutation" in item.lower() for item in captured["obligations"])
        from _lcm_regime_finder.declarations import DeclarationError, ScalarSource

        with pytest.raises(DeclarationError):
            ScalarSource(captured).expression()
    finally:
        _MUTABLE[0] = 1


def test_module_attribute_rebinding_during_capture_is_not_cached_away() -> None:
    try:
        with capture_session():
            captured = source_function(reads_module_attribute)
            _NAMESPACE.operation = alternate
        assert (captured["capture_status"]) == ("UNRESOLVED_MUTATION")
        assert any("rebinding" in item.lower() for item in captured["obligations"])
    finally:
        _NAMESPACE.operation = leaf


def test_source_digest_is_session_cached_and_file_change_invalidates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "capture_source.py"
        path.write_text(
            "def first(x):\n    return x\n\ndef second(x):\n    return -x\n"
        )
        spec = importlib.util.spec_from_file_location("capture_source_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # definitions only, no callback evaluation
        with capture_session() as session:
            first = source_function(module.first)
            second = source_function(module.second)
            assert (len(session.files)) == (1)
            assert (session.source_bytes) == (path.stat().st_size)
            path.write_text(path.read_text() + "# changed after inspection\n")
        assert (session.report()["status"]) == ("INCOMPLETE")
        assert (first["capture_status"]) == ("INCOMPLETE")
        assert (second["capture_status"]) == ("INCOMPLETE")


def test_tampered_helper_record_breaks_identity() -> None:
    captured = source_function(shared_layer(leaf))
    records = deepcopy(captured["callable_records"])
    target = captured["call_targets"]["left"]
    records[target["capture_ref"]]["source"] = "def leaf(history):\n    return 0\n"
    assert not (validate_callable_records(records))
    assert (resolve_callable(target, records)["kind"]) == ("unresolved")


def test_original_graph_checkpoint_precedes_budgeted_expansion() -> None:
    checkpoints = []
    captured = read_model(
        FakeModel(),
        checkpoint=checkpoints.append,
        capture_limits={"seconds": 10, "host_gib": 8, "max_records": 1},
    )
    assert (checkpoints[0]["stage"]) == ("original_graph")
    assert checkpoints[0]["topology_complete"]
    assert (checkpoints[0]["snapshot"]["n_periods"]) == (2)
    assert (captured["capture_report"]["status"]) == ("INCOMPLETE")
    assert (checkpoints[-1]["stage"]) == ("capture_incomplete")
    assert ("graph") in (captured)
    assert (len(captured["callable_records"])) <= (1)
    assert ("recommendation") not in (captured)


def test_complete_capture_preserves_production_domain_without_rows() -> None:
    checkpoints = []
    captured = read_model(
        FakeModel(),
        checkpoint=checkpoints.append,
        capture_limits={"seconds": 10, "host_gib": 8},
    )
    assert (captured["capture_report"]["status"]) == ("COMPLETE")
    assert (captured["user_regimes"]["original"]["states"]["history"]["n_points"]) == (
        10**18
    )
    assert validate_callable_records(captured["callable_records"])
    assert (checkpoints[-1]["stage"]) == ("capture_complete")
