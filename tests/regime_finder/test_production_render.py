"""Report behavior, escaping, exact records, PNG validity and bounded previews."""

import binascii
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch
import zlib

from _lcm_regime_finder import production_render as renderer


def fixture(count=3):
    cells = [
        {
            "id": f"c{i}",
            "regime": "life" if i < count - 1 else "terminal",
            "period": i,
            "age": 20 + 2 * i,
            "stage": "choice",
            "states": {"history": "h"},
            "actions": {},
            "terminal": i == count - 1,
            "admissible_start": i == 0,
        }
        for i in range(count)
    ]
    option = {
        "id": "identity",
        "states": {"history": "h"},
        "actions": {},
        "state_projection": {"history": {"kind": "identity", "target": "history"}},
        "action_reconstruction": {},
        "witnesses": [],
    }
    blocks = [
        {
            "cells": [c["id"]],
            "option_by_cell": {c["id"]: "identity"},
            "original_regimes": [c["regime"]],
            "ages": [c["age"]],
            "periods": [c["period"]],
            "state_axes": {"history": "h"},
            "action_axes": {},
            "allocation_kind": "per_block",
            "cost": {"work": 4, "storage": 2, "clarity": 4},
        }
        for c in cells
    ]
    scenario = {
        "status": "COMPLETE",
        "allocation_contract": {
            "status": "declared",
            "kind": "per_block",
            "boundary_unit": 1,
        },
        "clarity_weights": {"block": 4, "characteristic": 2, "dispatch": 1},
        "frontier": [
            {
                "id": "G0000",
                "blocks": blocks,
                "cost": {"work": 4 * count, "storage": 2 * count, "clarity": 4 * count},
            }
        ],
        "recommendation": {
            "graph_id": "G0000",
            "score": "3",
            "weights": dict.fromkeys(renderer.METRICS, "1"),
            "reference_scales": {
                "work": 4 * count,
                "storage": 2 * count,
                "clarity": 4 * count,
            },
        },
        "certificate": {
            "complete": True,
            "family": "Explicit renderer fixture family",
            "frontier_scope": "All fixture candidates",
        },
        "independent_check": {"status": "PASS", "errors": [], "unresolved": []},
    }
    return {
        "schema": "regime-production-result/v1",
        "status": "COMPLETE",
        "source_identity": {"name": "Renderer test fixture"},
        "analysis": {
            "declarations": {
                "cells": cells,
                "domains": {"h": {"kind": "categorical", "size": 2, "values": [0, 1]}},
                "relations": [
                    {
                        "id": f"r{i}",
                        "kind": "transition",
                        "phase": "solve",
                        "source": f"c{i}",
                        "targets": [f"c{i + 1}"],
                    }
                    for i in range(count - 1)
                ],
                "production_metadata": {"status": "synthetic renderer fixture"},
            },
            "interfaces": {c["id"]: {"options": [deepcopy(option)]} for c in cells},
            "obligations": [],
            "production_execution": {
                "callbacks_called": 0,
                "dense_cartesian_cells_enumerated": 0,
            },
        },
        "semantic_check": {"status": "PASS", "errors": [], "unresolved": []},
        "scenarios": [scenario],
        "layers": {
            "source_semantics": "PASS",
            "discovery": "COMPLETE",
            "allocation": "DECLARED",
            "frontier": "VERIFIED",
            "native_numerical_execution": "NOT_RUN",
        },
    }


def read_png(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("PNG signature missing")
    cursor, data, size = 8, [], None
    while cursor < len(raw):
        length = struct.unpack("!I", raw[cursor : cursor + 4])[0]
        kind = raw[cursor + 4 : cursor + 8]
        payload = raw[cursor + 8 : cursor + 8 + length]
        crc = struct.unpack("!I", raw[cursor + 8 + length : cursor + 12 + length])[0]
        if crc != binascii.crc32(kind + payload) & 0xFFFFFFFF:
            raise AssertionError("PNG chunk checksum mismatch")
        if kind == b"IHDR":
            size = struct.unpack("!IIBBBBB", payload)
        if kind == b"IDAT":
            data.append(payload)
        cursor += 12 + length
    return size, zlib.decompress(b"".join(data))


def test_exact_saved_result_and_all_alternatives() -> None:
    result = fixture()
    alternate = deepcopy(result["scenarios"][0]["frontier"][0])
    alternate["id"] = "G0001"
    alternate["cost"] = {"work": 15, "storage": 5, "clarity": 14}
    result["scenarios"][0]["frontier"].append(alternate)
    with tempfile.TemporaryDirectory() as directory:
        manifest = renderer.write_production_report(result, directory)
        assert (json.loads((Path(directory) / "report.json").read_text())) == (result)
        data = json.loads((Path(directory) / "report-data.json").read_text())
        assert (len(data["scenarios"][0]["alternatives"])) == (2)
        assert (data["scenarios"][0]["alternatives"][1]["plan"]) == (alternate)
        assert (manifest["frontier_alternative_count"]) == (2)
        assert ("Option 2") in ((Path(directory) / "report.html").read_text())
        assert not ((Path(directory) / "result.json").exists())


def test_html_and_mermaid_injection_are_data() -> None:
    result = fixture()
    attack = '"><script>alert("x")</script>\nclick N0 "javascript:x"'
    result["source_identity"]["name"] = attack
    result["analysis"]["declarations"]["cells"][0]["regime"] = attack
    rendered = renderer.render_html(result)
    assert ("<script>") not in (rendered)
    assert ("&lt;script&gt;") in (rendered)
    graph = renderer.graph_mermaid(renderer._view(result)["original_graph"])
    assert not (any(line.lstrip().startswith("click ") for line in graph.splitlines()))
    assert ('javascript:x"\n') not in (graph)


def test_png_is_real_bounded_rgb_and_deterministic() -> None:
    result = fixture()
    with tempfile.TemporaryDirectory() as directory:
        first = renderer.write_production_report(result, directory)
        image = next(row for row in first["files"] if row["path"].endswith(".png"))
        size, pixels = read_png(Path(directory) / image["path"])
        width, height, depth, color, _, _, _ = size
        assert ((width, depth, color)) == ((1000, 8, 2))
        assert (height) <= (first["max_png_height"])
        assert (len(pixels)) == (height * (1 + width * 3))
        assert (len(set(pixels))) > (10)
        second = renderer.write_production_report(result, directory)
        second_image = next(
            row for row in second["files"] if row["path"] == image["path"]
        )
        assert (image["sha256"]) == (second_image["sha256"])


def test_joint_relation_remains_reified_and_phase_labeled() -> None:
    result = fixture()
    result["analysis"]["declarations"]["relations"] = [
        {
            "id": "lottery",
            "kind": "joint",
            "phase": "simulation",
            "source": "c0",
            "targets": ["c1", "c2"],
            "joint_outputs": ["health", "income"],
        }
    ]
    graph = renderer._view(result)["original_graph"]
    assert (sum(n["kind"] == "joint" for n in graph["nodes"])) == (1)
    assert (len(graph["edges"])) == (3)
    assert all(e["phase"] == "simulation" for e in graph["edges"])
    assert ("Joint lottery; simulation; lottery") in (renderer.graph_mermaid(graph))


def test_preview_budget_preserves_full_graph_and_result() -> None:
    result = fixture(25)
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(renderer, "MAX_PNG_PAGES", 2),
    ):
        manifest = renderer.write_production_report(result, directory)
        assert (manifest["png_page_count"]) == (2)
        assert (manifest["omitted_preview_page_count"]) > (0)
        assert ("PNG preview limit") in ((Path(directory) / "report.html").read_text())
        assert ("PNG preview limit") in ((Path(directory) / "report.md").read_text())
        assert ("N24[") in ((Path(directory) / "graphs/original.mmd").read_text())
        assert (json.loads((Path(directory) / "report.json").read_text())) == (result)
        pngs = [
            row["path"] for row in manifest["files"] if row["path"].endswith(".png")
        ]
        assert all("scenario-001-option-0001" in name for name in pngs)


def test_incomplete_missing_analysis_and_conditional_scope() -> None:
    incomplete = {
        "status": "INCOMPLETE",
        "error": "Resource budget reached",
        "resources": {"wall_seconds": 900},
    }
    with tempfile.TemporaryDirectory() as directory:
        manifest = renderer.write_production_report(incomplete, directory)
        assert (manifest["png_page_count"]) == (0)
        text = (Path(directory) / "report.html").read_text()
        assert ("No frontier scenario was supplied") in (text)
        assert ("Resource budget reached") in (text)
    result = fixture()
    result["scenarios"][0]["allocation_contract"]["status"] = "conditional_scenario"
    result["semantic_check"]["status"] = "UNRESOLVED"
    text = renderer.render_html(result)
    assert ("actual backend allocation contract is unresolved") in (text)
    assert (
        "candidate changes, not an established source-equivalent transformation"
    ) in (text)


def test_heuristic_local_discovery_is_not_labeled_graph_family_coverage() -> None:
    result = fixture()
    result["status"] = "HEURISTIC"
    result["scenarios"][0]["status"] = "HEURISTIC"
    for text in (renderer.render_html(result), renderer.render_markdown(result)):
        assert "Dependency and local-option discovery" in text
        assert "Candidate-family coverage" not in text
        assert "unknown optimality gap" in text
