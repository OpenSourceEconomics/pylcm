"""Public outer-budget metadata runner, not a solver smoke fixture."""

import pytest
import hashlib
import json
import tempfile
from pathlib import Path
from _lcm_regime_finder.pytask_api import analyse_snapshot

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"


def test_actual_bounded_runner_produces_a_readable_source_bound_summary() -> None:
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "result"
        summary = analyse_snapshot(SOURCE, output, facts_backend="metadata")
        assert (summary["status"]) == ("COMPLETE")
        assert summary["checker_modules_unchanged"]
        assert (summary["source_snapshot_sha256"]) == (before)
        assert (output / "next-actions.html").is_file()
        assert (output / "next-actions.md").is_file()
        assert (summary["layout_check"]) == ("PASS")
        assert summary["curve"]
        assert (summary["recommendation"]["operation"]) == ("retain_original")
        assert json.loads((output / "outer-resource/resource.json").read_text())[
            "within_declared_budget"
        ]
        assert ("No recommendation is applied automatically") in (
            (output / "next-actions.md").read_text()
        )
    assert (hashlib.sha256(SOURCE.read_bytes()).hexdigest()) == (before)


def test_exhaustion_has_no_adoption_and_preserves_original() -> None:
    before = SOURCE.read_bytes()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "result"
        with pytest.raises(RuntimeError, match="INCOMPLETE"):
            analyse_snapshot(SOURCE, output, seconds=0, facts_backend="metadata")
        summary = json.loads((output / "summary.json").read_text())
        assert (summary["status"]) == ("INCOMPLETE")
        assert ("decisions") not in (summary)
    assert (SOURCE.read_bytes()) == (before)


def test_output_namespace_cannot_overwrite_original_summary_named_input() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        source = folder / "summary.json"
        source.write_bytes(SOURCE.read_bytes())
        before = source.read_bytes()
        with pytest.raises(ValueError, match="original input"):
            analyse_snapshot(source, folder, facts_backend="metadata")
        assert (source.read_bytes()) == (before)
