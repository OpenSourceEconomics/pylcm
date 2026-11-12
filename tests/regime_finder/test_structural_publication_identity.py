"""Publication preserves exact committed JSON and cold-replayable store locators."""

import pytest
from copy import deepcopy
import contextlib
import errno
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from _lcm_regime_finder import cli, verify_production_result
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.structural_store import (
    CaptureStore,
    LoadedCapture,
    final_input_check,
    relocate_captures,
)


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
)


def _create_source(root):
    path = root / "input.json"
    source = json.loads(FIXTURE.read_text())
    store = CaptureStore(
        path,
        check=lambda: None,
        source_context={"fixture": "generated original declarations"},
        transaction={"kind": "exclusive-owner", "authority": "serial test"},
    )
    source["extra_fixture_data"] = store.reference([tuple([1, 2, 3])])
    source["structural_store"] = store.finish()
    path.write_text(json.dumps(source))
    return path, store


def test_actual_saved_report_passes_public_checker_on_unchanged_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        before = source.read_bytes()
        output = root / "result"
        check = root / "check.json"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                ["analyze-model", str(source), "--output", str(output), "--worker"]
            )
            checked = cli.main(
                [
                    "check-model-result",
                    str(source),
                    str(output / "report.json"),
                    "--output",
                    str(check),
                    "--worker",
                ]
            )
        assert (code) == (0)
        assert (checked) == (0)
        assert (json.loads(check.read_text())["status"]) == ("PASS")
        assert (source.read_bytes()) == (before)


def test_publication_does_not_change_any_committed_report_field() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        problem = cli.load_problem(source)
        report = analyze_production(problem)
        expected = deepcopy(report)
        assert (verify_production_result(problem, report)["status"]) == ("PASS")
        for folder in (root / "first", root / "second/nested"):
            path = folder / "report.json"
            cli.save_json(path, report)
            saved = json.loads(path.read_text())
            assert (saved) == (expected)
            assert (report) == (expected)
            assert (verify_production_result(problem, saved)["status"]) == ("PASS")
            cli.load_problem(path)  # every preserved relative locator resolves cold


def test_same_generation_at_two_locators_keeps_both_exact_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        value = cli.load_problem(source)
        alternate = deepcopy(value["structural_store"])
        alternate["path"] = "second/" + alternate["manifest"]["root_digest"]
        value["other"] = {"structural_store": alternate}
        expected = deepcopy(value)
        output = root / "result"
        cli.save_json(output / "snapshot.json", value)
        loaded = cli.load_problem(output / "snapshot.json")
        assert (dict(loaded)) == (dict(expected))
        assert (output / value["structural_store"]["path"] / "manifest.json").is_file()
        assert (output / alternate["path"] / "manifest.json").is_file()


def test_source_within_output_does_not_replace_required_locator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        value = cli.load_problem(source)
        value["structural_store"]["path"] = "mirror/" + owner.reader.doc["root_digest"]
        expected = deepcopy(value)
        relocated = relocate_captures(value, root)
        assert (relocated) == (expected)
        assert (root / value["structural_store"]["path"] / "manifest.json").is_file()


def test_escape_and_initial_store_collision_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        original = cli.load_problem(source)
        output = root / "result"
        for locator in ("../escaped", "/outside", "."):
            value = deepcopy(original)
            value["structural_store"]["path"] = locator
            with pytest.raises(ValueError, match="escapes"):
                relocate_captures(value, output)
        target = output / original["structural_store"]["path"]
        target.mkdir(parents=True)
        (target / "manifest.json").write_text("{}")
        with pytest.raises(ValueError, match="differs"):
            relocate_captures(original, output)


def test_final_validation_covers_copied_artifacts_at_preserved_locator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        value = cli.load_problem(source)
        output = root / "result"
        with patch(
            "_lcm_regime_finder.structural_store.os.link",
            side_effect=OSError(errno.EXDEV, "generated different filesystem"),
        ):
            relocate_captures(value, output)
        reader = value.structure
        key = next(iter(reader.payloads))
        original = reader.payloads[key]
        chunk = output / value["structural_store"]["path"] / (key + ".bin")
        chunk.write_bytes(b"changed")
        with pytest.raises(ValueError, match="output artifact changed"):
            final_input_check(value)
        assert (reader.payloads[key]) == (original)
