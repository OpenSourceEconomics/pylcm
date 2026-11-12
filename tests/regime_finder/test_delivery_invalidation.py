"""Late failure revokes every public authorization route; generated metadata only."""

import pytest
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from _lcm_regime_finder import cli, pipeline
from _lcm_regime_finder.pytask_api import analyse_snapshot
from _lcm_regime_finder.structural_store import CaptureStore, final_input_check


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
)
# Independent inventory of public result routes, not imported from the repair.
PUBLIC_DECISION_FILES = (
    "candidate-boundaries.json",
    "recommended-boundaries.json",
    "proposal-decisions.json",
    "report-data.json",
    "report-files.json",
    "report-manifest.json",
    "family-boundary.json",
    "certificate.json",
    "certificate-check.json",
    "transport-check.json",
    "recommended.mmd",
)


def _create_source(root):
    path = root / "input.json"
    source = json.loads(FIXTURE.read_text())
    store = CaptureStore(
        path,
        check=lambda: None,
        source_context={"fixture": "generated"},
        transaction={"kind": "exclusive-owner", "authority": "serial test"},
    )
    source["extra_fixture_data"] = store.reference([tuple([1, 2, 3])])
    source["structural_store"] = store.finish()
    path.write_text(json.dumps(source))
    return path, store


def _assert_revoked(output, status):
    for name in PUBLIC_DECISION_FILES:
        assert not ((output / name).exists()), name
        assert not ((output / (name + ".writing")).exists()), name
    for name in ("report.json", "summary.json"):
        record = json.loads((output / name).read_text())
        assert (record["status"]) == (status)
        assert not (record.get("transformation_authorized", False))
        assert (record.get("recommendation")) is None
        assert not (record.get("decisions"))
        assert not (record.get("proposal_decisions"))
    for name in ("report.md", "report.html", "next-actions.md", "next-actions.html"):
        assert ("No optimum or transformation is authorized") in (
            (output / name).read_text()
        )


def test_real_cli_late_chunk_failure_revokes_all_derived_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        # Reuse an actual successful output namespace before the failed attempt.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            assert (
                cli.main(
                    [
                        "analyze-model",
                        "--search-mode",
                        "exact",
                        str(source),
                        "--output",
                        str(output),
                        "--worker",
                    ]
                )
            ) == (0)

        def corrupt(snapshot):
            reader = snapshot.structure
            chunk = reader.payloads.path(next(iter(reader.payloads)))
            data = bytearray(chunk.read_bytes())
            data[0] ^= 1
            chunk.write_bytes(data)
            final_input_check(snapshot)

        with (
            patch("_lcm_regime_finder.structural_store.final_input_check", corrupt),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "analyze-model",
                    "--search-mode",
                    "exact",
                    str(source),
                    "--output",
                    str(output),
                    "--worker",
                ]
            )
        assert (code) == (1)
        _assert_revoked(output, "INVALID")
        assert (output / "original.json").exists()


def test_semantic_checker_exception_revokes_standalone_stale_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        output.mkdir()
        for name in PUBLIC_DECISION_FILES:
            (output / name).write_text('{"transformation_authorized": true}')
        with (
            patch(
                "_lcm_regime_finder.production.analyze_production",
                side_effect=RuntimeError("checker failed"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "analyze-model",
                    "--search-mode",
                    "exact",
                    str(source),
                    "--output",
                    str(output),
                    "--worker",
                ]
            )
        assert (code) == (1)
        _assert_revoked(output, "INVALID")


def test_positive_delivery_keeps_checked_authorizations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "analyze-model",
                    "--search-mode",
                    "exact",
                    str(source),
                    "--output",
                    str(output),
                    "--worker",
                ]
            )
        assert (code) == (0)
        rows = json.loads((output / "recommended-boundaries.json").read_text())
        assert any(row["transformation_authorized"] for row in rows)


def _create_stand_in(output, tail):
    # A supported command_prefix exercises the public parent boundary.
    return (
        "import json,time; from pathlib import Path; "
        f"p=Path({str(output)!r}); "
        'record={"status":"COMPLETE","recommendation":"old success","decisions":["adopt"]}; '
        '(p/"summary.json").write_text(json.dumps(record)); '
        '(p/"report.json").write_text(json.dumps(record)); '
        f"names={PUBLIC_DECISION_FILES!r}; "
        '[ (p/name).write_text(json.dumps({"transformation_authorized":True})) for name in names ]; '
        + tail
    )


def test_public_parent_late_source_failure_revokes_all_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, store = _create_source(root)
        output = root / "result"
        chunk = store.reader.payloads.path(next(iter(store.reader.payloads)))
        script = _create_stand_in(
            output, f'Path({str(chunk)!r}).write_bytes(b"corrupt")'
        )
        with pytest.raises(RuntimeError, match="INVALID"):
            analyse_snapshot(
                source,
                output,
                seconds=10,
                command_prefix=[sys.executable, "-c", script],
                search_mode="exact",
            )
        _assert_revoked(output, "INVALID")


def test_public_parent_late_budget_failure_revokes_all_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, store = _create_source(root)
        output = root / "result"
        script = _create_stand_in(output, "time.sleep(2)")
        with pytest.raises(RuntimeError, match="INCOMPLETE"):
            analyse_snapshot(
                source,
                output,
                seconds=0.15,
                command_prefix=[sys.executable, "-c", script],
                search_mode="exact",
            )
        _assert_revoked(output, "INCOMPLETE")


def test_pipeline_changed_checker_revokes_existing_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        original = pipeline._digest
        seen = 0

        def changing_digest(path):
            nonlocal seen
            if Path(path).name == "cli.py":
                seen += 1
                if seen > 1:
                    return "0" * 64
            return original(path)

        with (
            patch.object(pipeline, "_digest", side_effect=changing_digest),
            patch.object(
                sys,
                "argv",
                ["pipeline", str(source), "--output", str(output), "--worker"],
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = pipeline.main()
        assert (code) == (1)
        _assert_revoked(output, "INVALID")


def test_pipeline_rejects_overlapping_output_before_hash_or_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        source = output / "report.json"
        source.write_bytes(FIXTURE.read_bytes())
        before = source.read_bytes()
        with (
            patch.object(
                pipeline, "_digest", side_effect=RuntimeError("early hashing failed")
            ),
            patch.object(
                sys,
                "argv",
                ["pipeline", str(source), "--output", str(output), "--worker"],
            ),
        ):
            with pytest.raises(ValueError, match="original input"):
                pipeline.main()
        assert (source.read_bytes()) == (before)
        assert (list(output.iterdir())) == ([source])


def test_pipeline_does_not_restore_adoption_advice_after_cli_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        with (
            patch(
                "_lcm_regime_finder.production.analyze_production",
                side_effect=RuntimeError("checker failed"),
            ),
            patch.object(
                sys,
                "argv",
                ["pipeline", str(source), "--output", str(output), "--worker"],
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = pipeline.main()
        assert (code) == (1)
        _assert_revoked(output, "INVALID")


def test_independent_check_cannot_publish_pass_after_late_chunk_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        # Keep the report beside its input so this final-integrity test is
        # independent of the separate locator-relocation defect.
        from _lcm_regime_finder.production import analyze_production

        report_path = root / "bound-report.json"
        cli.save_json(
            report_path,
            analyze_production(cli.load_problem(source), search_mode="exact"),
        )
        check_path = root / "check.json"
        original = cli.verify_production_result
        called = []

        def changed_after_check(problem, report, **kwargs):
            result = original(problem, report, **kwargs)
            assert (result["status"]) == ("PASS")
            called.append(True)
            reader = problem.structure
            chunk = reader.payloads.path(next(iter(reader.payloads)))
            data = bytearray(chunk.read_bytes())
            data[0] ^= 1
            chunk.write_bytes(data)
            return result

        with (
            patch.object(
                cli, "verify_production_result", side_effect=changed_after_check
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "check-model-result",
                    str(source),
                    str(report_path),
                    "--output",
                    str(check_path),
                    "--worker",
                ]
            )
        assert (called) == ([True])
        assert (code) == (1)
        assert (json.loads(check_path.read_text())["status"]) == ("INVALID")


def test_checker_parent_revokes_pass_when_worker_fails_after_writing_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            assert (
                cli.main(
                    [
                        "analyze-model",
                        "--search-mode",
                        "exact",
                        str(source),
                        "--output",
                        str(output),
                        "--worker",
                    ]
                )
            ) == (0)
        check_path = root / "check.json"

        def failed_worker(*args, **kwargs):
            check_path.write_text('{"status":"PASS"}')
            return 2

        with (
            patch(
                "_lcm_regime_finder.resources.run_bounded", side_effect=failed_worker
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "check-model-result",
                    str(source),
                    str(output / "report.json"),
                    "--output",
                    str(check_path),
                ]
            )
        assert (code) == (2)
        assert (json.loads(check_path.read_text())["status"]) == ("INCOMPLETE")


def test_task_api_bad_capture_receipt_revokes_old_certificates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        output.mkdir()
        for name in PUBLIC_DECISION_FILES:
            (output / name).write_text('{"transformation_authorized":true}')
        receipt = source.with_name(source.stem + ".metadata-capture.json")
        receipt.write_text('{"schema":"wrong","snapshot_sha256":"old"}')
        with pytest.raises(ValueError, match="Capture receipt"):
            analyse_snapshot(source, output, search_mode="exact")
        _assert_revoked(output, "INVALID")


def test_task_api_malformed_worker_result_revokes_old_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        script = _create_stand_in(
            output, '(p/"summary.json").write_text("partial json")'
        )
        with pytest.raises(ValueError):
            analyse_snapshot(
                source,
                output,
                seconds=10,
                command_prefix=[sys.executable, "-c", script],
                search_mode="exact",
            )
        _assert_revoked(output, "INVALID")


def test_direct_worker_incomplete_analysis_revokes_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        from _lcm_regime_finder.production import analyze_production

        def incomplete(*args, **kwargs):
            result = analyze_production(*args, **kwargs)
            result["status"] = "INCOMPLETE"
            return result

        with (
            patch(
                "_lcm_regime_finder.production.analyze_production",
                side_effect=incomplete,
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "analyze-model",
                    "--search-mode",
                    "exact",
                    str(source),
                    "--output",
                    str(output),
                    "--worker",
                ]
            )
        assert (code) == (2)
        _assert_revoked(output, "INCOMPLETE")


def test_conditional_delivery_keeps_independently_valid_local_authority() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, owner = _create_source(root)
        output = root / "result"
        from _lcm_regime_finder.production import analyze_production

        def conditional(*args, **kwargs):
            result = analyze_production(*args, **kwargs)
            result["status"] = "UNRESOLVED"
            return result

        with (
            patch(
                "_lcm_regime_finder.production.analyze_production",
                side_effect=conditional,
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = cli.main(
                [
                    "analyze-model",
                    "--search-mode",
                    "exact",
                    str(source),
                    "--output",
                    str(output),
                    "--worker",
                ]
            )
        assert (code) == (3)
        rows = json.loads((output / "recommended-boundaries.json").read_text())
        assert any(row["transformation_authorized"] for row in rows)
