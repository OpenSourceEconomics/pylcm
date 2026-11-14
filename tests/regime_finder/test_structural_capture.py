"""End-to-end capture and replay fixtures; no economic Model or solver runs."""

import pytest
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType
import tempfile

from _lcm_regime_finder.structural_store import (
    CaptureStore,
    load_capture_structure,
    reference_info,
)
from _lcm_regime_finder.pytask_api import export_model, analyse_snapshot
from _lcm_regime_finder.cli import load_problem, save_json
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_events import source_anchor, compare_source
from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.callable_identity import validate_callable_records
from tests.regime_finder.test_round6_capture import FakeModel, callback_with_binding


@dataclass(frozen=True)
class ReachabilityFixture:
    members: tuple[str, ...]
    by_node: MappingProxyType[tuple[int, str], frozenset[tuple[int, ...]]]
    union: tuple[tuple[int, ...], ...]


def test_large_mapping_is_retained_without_expanding_capture_items() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = FakeModel()
        model.fixed_params = {"labels": {f"label{i}": i for i in range(10000)}}
        path = Path(tmp) / "model.json"
        receipt = export_model(
            model, path, production_metadata={"fixture": True}, seconds=20
        )
        assert (receipt["status"]) == ("COMPLETE")
        loaded = load_problem(path)
        assert (reference_info(loaded["fixed_params"]["labels"])) == (
            {"type": "dict", "length": 10000}
        )
        assert (receipt["capture_report"]["estimated_metadata_payload_bytes"]) < (
            100000
        )


def test_categorical_cardinality_comes_from_checked_payload() -> None:
    from _lcm_regime_finder.declarations import Normalizer

    with tempfile.TemporaryDirectory() as tmp:
        store = CaptureStore(
            Path(tmp) / "input.json",
            check=lambda: None,
            source_context={"fixture": True},
        )
        ref = store.reference(list(range(9000)))
        store.finish()
        grid = {
            "kind": "object",
            "type": "lcm.grids.DiscreteGrid",
            "fields": {"codes": ref, "values": ref},
        }
        normal = Normalizer({"grid": grid}, {})
        result = normal.domains[normal.domain(grid, "/grid")]
        assert (result["size"]) == (9000)
        assert (result["structural_values"]) == (ref)
        assert ("values") not in (result)
        assert any(
            o["kind"] == "structural_domain_evaluation" for o in normal.obligations
        )


def test_complete_typed_reachability_includes_small_sets_and_mapping_keys() -> None:
    first = tuple([0, 1])
    equal = tuple([0, 1])
    source = ReachabilityFixture(
        ("first", "second"),
        MappingProxyType(
            {
                (0, "a"): frozenset({first}),
                (1, "a"): frozenset({equal}),
                (2, "a"): frozenset(),
            }
        ),
        (first,),
    )
    anchor = source_anchor(
        source, source_context={"fixture": True}, transaction={"kind": "test-local"}
    )
    bundle = NativeWriter().finish(source, anchor)
    reader = Reader(bundle)
    assert (compare_source(reader, anchor)) == (anchor["logical_sha256"])
    assert (reader.row_cache) == ({})


def test_export_and_cold_load_retain_original_large_field() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = FakeModel()
        model.fixed_params = {"large_declared_data": list(range(10000))}
        path = Path(tmp) / "model.json"
        receipt = export_model(
            model,
            path,
            production_metadata={"fixture": True},
            seconds=20,
            source_transaction={
                "kind": "exclusive-owner",
                "authority": "this test owns the generated fake model",
            },
        )
        assert (receipt["status"]) == ("COMPLETE")
        captured = load_problem(path)
        assert (reference_info(captured["fixed_params"]["large_declared_data"])) == (
            {"type": "list", "length": 10000}
        )
        assert validate_callable_records(captured["callable_records"])


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="actual NumPy required"
)
def test_bound_arrays_are_not_omitted_or_one_record_per_element() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        data = np.arange(20000, dtype="<i4")
        model = FakeModel()
        model.fixed_params = {"array": data}
        model.user_regimes["original"]["functions"]["utility"] = callback_with_binding(
            data
        )
        path = Path(tmp) / "model.json"
        receipt = export_model(
            model, path, production_metadata={"fixture": True}, seconds=20
        )
        assert (receipt["status"]) == ("COMPLETE")
        snapshot = load_problem(path)
        assert ("UNRESOLVED_LARGE_BINDING") not in (json.dumps(snapshot))
        assert validate_callable_records(snapshot["callable_records"])
        assert (snapshot["structural_store"]["manifest"]["source_consistency"]) == (
            "UNRESOLVED_QUIESCENCE"
        )


def test_replay_export_moves_store_once_and_loads_normalized_nested_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = CaptureStore(
            root / "input.json",
            check=lambda: None,
            source_context={"fixture": True},
            transaction={"kind": "test-local"},
        )
        ref = store.reference([tuple([i, i + 1]) for i in range(9000)])
        source = {"original": {"structural_store": store.finish(), "data": ref}}
        path = root / "input.json"
        path.write_text(json.dumps(source))
        loaded = load_capture_structure(source, path)
        output = root / "result"
        save_json(output / "normalized.json", loaded)
        save_json(output / "second.json", loaded)
        replay = load_problem(output / "normalized.json")
        assert (reference_info(replay["original"]["data"])["length"]) == (9000)
        assert (
            output / replay["original"]["structural_store"]["path"] / "manifest.json"
        ).is_file()
        assert (
            (replay["original"]["structural_store"]["path"])
            == (source["original"]["structural_store"]["path"])
        )


def test_public_pipeline_retains_source_bound_store_with_a_generated_declaration() -> (
    None
):
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = CaptureStore(
            root / "input.json", check=lambda: None, source_context={"fixture": True}
        )
        ref = store.reference([tuple([3, 4])])
        source = json.loads(fixture.read_text())
        source["structural_store"] = store.finish()
        source["additional_fixture_metadata"] = ref
        path = root / "input.json"
        path.write_text(json.dumps(source))
        result = analyse_snapshot(path, root / "analysis", seconds=30)
        assert (result["status"]) not in (("INVALID", "FAILED", "INCOMPLETE"))
        normalized = load_problem(root / "analysis/normalized.json")
        assert (
            reference_info(normalized["additional_fixture_metadata"])["length"]
        ) == (1)
