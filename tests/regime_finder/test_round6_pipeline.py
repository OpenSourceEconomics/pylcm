"""Whole metadata accounting and failure output, without native construction."""

import pytest
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from _lcm_regime_finder.pytask_api import analyse_snapshot
from _lcm_regime_finder.resources import run_bounded, _tree_rss


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
)


def test_separate_phase_receipt_includes_checks_and_delivery() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "result"
        analyse_snapshot(SOURCE, output, seconds=10)
        receipt = json.loads((output / "delivery-timings.json").read_text())
        assert all(
            receipt["phase_timings"][name] >= 0
            for name in (
                "normalization_seconds",
                "metadata_analysis_seconds",
                "source_check_seconds",
                "frontier_search_seconds",
                "frontier_check_seconds",
                "proposal_check_seconds",
            )
        )
        assert all(
            receipt["delivery_timings"][name] >= 0
            for name in (
                "checkpoint_write_seconds",
                "canonical_report_write_seconds",
                "graph_and_boundary_artifacts_seconds",
                "readable_report_seconds",
            )
        )
        progress = json.loads((output / "metadata-progress.json").read_text())
        assert ("source_check_seconds") in (progress["phase_timings"])
        assert (progress["elapsed_seconds"]) > (0)


def test_existing_capture_consumes_the_same_budget() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "snapshot.json"
        source.write_bytes(SOURCE.read_bytes())
        receipt = {
            "schema": "regime-metadata-capture-receipt/v1",
            "snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "status": "COMPLETE",
            "elapsed_seconds": 2.0,
        }
        (root / "snapshot.metadata-capture.json").write_text(json.dumps(receipt))
        with pytest.raises(RuntimeError, match="INCOMPLETE"):
            analyse_snapshot(source, root / "result", seconds=1)
        summary = json.loads((root / "result/summary.json").read_text())
        assert (summary["metadata_budget"]["capture_seconds_charged"]) == (2.0)
        assert (summary["metadata_budget"]["remaining_worker_seconds"]) == (0)
        assert not (summary["metadata_budget"]["within_time_budget"])
        assert ("decisions") not in (summary)


def test_stale_success_is_replaced_on_early_exhaustion() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "result"
        output.mkdir()
        (output / "summary.json").write_text(
            '{"status":"COMPLETE","decisions":["stale"]}'
        )
        (output / "next-actions.html").write_text("stale complete optimum")
        (output / "analysis.json").write_text('{"old_source":true}')
        (output / "family-boundary.json").write_text('{"old_family":true}')
        (output / "semantic-check.json.writing").write_text("incomplete old check")
        with pytest.raises(RuntimeError, match="INCOMPLETE"):
            analyse_snapshot(SOURCE, output, seconds=0)
        summary = json.loads((output / "summary.json").read_text())
        assert (summary["status"]) == ("INCOMPLETE")
        assert ("decisions") not in (summary)
        assert ("stale complete optimum") not in (
            (output / "next-actions.html").read_text()
        )
        assert ("No optimum") in ((output / "next-actions.md").read_text())
        assert not ((output / "analysis.json").exists())
        assert not ((output / "family-boundary.json").exists())
        assert not ((output / "semantic-check.json.writing").exists())


def test_capture_receipt_is_bound_to_exact_snapshot() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "snapshot.json"
        source.write_bytes(SOURCE.read_bytes())
        (root / "snapshot.metadata-capture.json").write_text(
            json.dumps(
                {
                    "schema": "regime-metadata-capture-receipt/v1",
                    "snapshot_sha256": "old snapshot identity",
                    "status": "COMPLETE",
                    "elapsed_seconds": 0.1,
                }
            )
        )
        with pytest.raises(ValueError, match="original snapshot bytes"):
            analyse_snapshot(source, root / "result")


def test_outer_supervisor_references_the_workers_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        checkpoint = root / "partial-frontier.json"
        checkpoint.write_text('{"candidate":"unverified"}')
        code = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            output=root / "outer-resource",
            seconds=0.03,
            host_gib=0.5,
            checkpoint_path=checkpoint,
        )
        report = json.loads((root / "outer-resource/report.json").read_text())
        assert (code) == (2)
        reference = report["partial_frontier_reference"]
        assert ((root / "outer-resource" / reference["path"]).resolve()) == (checkpoint)
        assert (reference["candidate_safety"]) == ("unverified")
        assert (report["recommendation"]) is None


def test_late_budget_exit_cannot_publish_worker_success(
    *, subtests: pytest.Subtests
) -> None:
    for report_published in (False, True):
        with (
            subtests.test(report_published=report_published),
            tempfile.TemporaryDirectory() as folder,
        ):
            output = Path(folder) / "result"

            def exhausted_worker(_command, **_options):
                assert _options["output"] == output / "outer-resource"
                # Fix the supervisor outcome, not the child startup speed.
                # Real process termination/checkpoints are tested separately.
                if report_published:
                    (output / "summary.json").write_text(
                        json.dumps({"status": "COMPLETE", "decisions": ["old success"]})
                    )
                    (output / "report.json").write_text(
                        json.dumps(
                            {"status": "COMPLETE", "recommendation": "old success"}
                        )
                    )
                return 2

            with patch(
                "_lcm_regime_finder.resources.run_bounded", side_effect=exhausted_worker
            ):
                with pytest.raises(RuntimeError, match="INCOMPLETE"):
                    analyse_snapshot(SOURCE, output, seconds=10)
            summary = json.loads((output / "summary.json").read_text())
            report = json.loads((output / "report.json").read_text())
            assert (summary["status"]) == ("INCOMPLETE")
            assert ("decisions") not in (summary)
            assert (report["status"]) == ("INCOMPLETE")
            assert (report["recommendation"]) is None
            if report_published:
                retained = json.loads(
                    (output / report["completed_worker_evidence"]).read_text()
                )
                assert (retained["recommendation"]) == ("old success")
            else:
                assert (report.get("completed_worker_evidence")) is None
                assert not ((output / "completed-before-budget-check.json").exists())


def test_direct_cli_guards_the_entire_output_namespace(
    *, subtests: pytest.Subtests
) -> None:
    from _lcm_regime_finder.cli import main

    for name in ("metadata-progress.json", "resource.json", "report.json.writing"):
        with subtests.test(name=name), tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            original = output / name
            original.write_bytes(SOURCE.read_bytes())
            before = original.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["analyze-model", str(original), "--output", str(output)])
            assert (code) == (1)
            assert (original.read_bytes()) == (before)


@pytest.mark.skipif(
    not (Path("/proc/self/status").is_file()), reason="Linux /proc is unavailable"
)
def test_namespace_aware_rss_and_descendants_are_observed() -> None:
    import subprocess

    rss, count = _tree_rss(os.getpid())
    assert (rss) > (0)
    assert (count) >= (1)
    # This subprocess allocates only a small bytearray. It exercises actual
    # descendant telemetry, including an ancestor-mounted /proc namespace;
    # it performs no model construction or economic calculation.
    command = [sys.executable, str(Path(__file__).with_name("resource_probe.py"))]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    measured = json.loads(result.stdout)
    record = measured["resource"]
    assert (measured["exit_code"]) == (0)
    assert record["process_tree_rss_available"]
    assert (record["maximum_observed_processes"]) >= (2)
    assert (record["peak_host_gib"]) > (16 / 1024)
    assert record["within_declared_budget"]
