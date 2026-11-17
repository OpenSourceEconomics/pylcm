"""Report behavior, escaping, exact records, PNG validity and bounded previews."""

import binascii
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch
import zlib

from _lcm_regime_finder import production_render as renderer
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_declaration_ingestion import (
    json_list,
    json_member,
    json_object,
    json_rows,
)


def _point(
    cells: Sequence[Mapping[str, JSONValue]], max_blocks: int
) -> dict[str, JSONValue]:
    blocks = [
        {
            "fibres": [{"period": c["period"], "regime": c["regime"], "statuses": {}}],
            "periods": [c["period"]],
            "states": {"history": "h"},
            "actions": {},
            "evaluations": 4,
            "storage": 2,
        }
        for c in cells
    ]
    return {
        "max_blocks": max_blocks,
        "blocks": len(cells),
        "evaluations": 4 * len(cells),
        "storage": 2 * len(cells),
        "exact": True,
        "layout": [{"lanes": [], "blocks": blocks}],
    }


def fixture(count: int = 3) -> dict[str, JSONValue]:
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
    return {
        "schema": "regime-production-result/v2",
        "status": "COMPLETE",
        "facts_backend": "metadata",
        "source_identity": {"name": "Renderer test fixture"},
        "recommendation": {
            "operation": "layout",
            "curve_index": 0,
            "max_blocks": count,
            "score": 3,
            "original_score": 3,
        },
        "original_cost": {"work": 4 * count, "storage": 2 * count, "clarity": 2},
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
            "interfaces": {
                str(c["id"]): {"options": [deepcopy(option)]} for c in cells
            },
            "obligations": [],
            "production_execution": {
                "callbacks_called": 0,
                "dense_cartesian_cells_enumerated": 0,
            },
        },
        "semantic_check": {"status": "PASS", "errors": [], "unresolved": []},
        "layouts": {
            "schema": "supplement-layout-curve/v1",
            "status": "COMPLETE",
            "curve": [_point(cells, count)],
        },
        "layout_check": {"status": "PASS", "errors": []},
        "layers": {
            "facts": "metadata",
            "source_semantics": "PASS",
            "layouts": "COMPLETE",
            "layout_check": "PASS",
            "native_numerical_execution": "NOT_RUN",
        },
    }


def read_png(path: Path) -> tuple[tuple[int, ...], bytes]:
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
    if size is None:
        raise AssertionError("PNG header missing")
    return size, zlib.decompress(b"".join(data))


def test_exact_saved_result_and_all_curve_points() -> None:
    result = fixture()
    cells = json_rows(result, "analysis", "declarations", "cells")
    json_list(result, "layouts", "curve").append(_point(cells, 4))
    with tempfile.TemporaryDirectory() as directory:
        manifest = renderer.write_production_report(result, directory)
        assert (json.loads((Path(directory) / "report.json").read_text())) == (result)
        data = json.loads((Path(directory) / "report-data.json").read_text())
        assert (data["schema"]) == ("regime-render-data/v2")
        assert ([row["max_blocks"] for row in data["layouts"]]) == ([3, 4])
        assert (manifest["curve_points"]) == (2)
        assert ("At most 4 regimes") in ((Path(directory) / "report.html").read_text())
        assert not ((Path(directory) / "result.json").exists())


def test_html_and_mermaid_injection_are_data() -> None:
    result = fixture()
    attack = '"><script>alert("x")</script>\nclick N0 "javascript:x"'
    json_member(result, "source_identity")["name"] = attack
    json_rows(result, "analysis", "declarations", "cells")[0]["regime"] = attack
    point = json_rows(result, "layouts", "curve")[0]
    block = json_rows(json_rows(point, "layout")[0], "blocks")[0]
    json_rows(block, "fibres")[0]["regime"] = attack
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
    json_member(result, "analysis", "declarations")["relations"] = [
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
        assert ("N24[") in ((Path(directory) / "graphs/original.mmd").read_text())
        assert (json.loads((Path(directory) / "report.json").read_text())) == (result)
        pngs = [
            row["path"] for row in manifest["files"] if row["path"].endswith(".png")
        ]
        assert all("layout-k025" in name for name in pngs)


def test_incomplete_result_states_why_no_curve_exists() -> None:
    incomplete = {
        "status": "INCOMPLETE",
        "error": "Resource budget reached",
        "resources": {"wall_seconds": 900},
    }
    with tempfile.TemporaryDirectory() as directory:
        manifest = renderer.write_production_report(incomplete, directory)
        assert (manifest["png_page_count"]) == (0)
        text = (Path(directory) / "report.html").read_text()
        assert ("No layout curve was produced") in (text)
        assert ("Resource budget reached") in (text)


def test_period_only_blocks_list_their_periods_in_both_reports() -> None:
    result = fixture()
    for cell in json_rows(result, "analysis", "declarations", "cells"):
        cell["age"] = None
    with tempfile.TemporaryDirectory() as directory:
        renderer.write_production_report(result, directory)
        markdown = (Path(directory) / "report.md").read_text()
        page = (Path(directory) / "report.html").read_text()
    header = "| regimes | periods | states | actions | evaluations | stored values |"
    table = markdown.split(header, 1)[1].splitlines()[2:5]
    assert [row.split(" | ")[1] for row in table] == ["0", "1", "2"]
    assert "<td>Not supplied</td>" not in page
