"""Tests for the benchmark PR-comment formatter."""

# ruff: noqa: SLF001

import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import pytest

from benchmarks import pr_comment


def test_grouped_table_uses_canonical_family_and_numeric_parameter_order():
    """CPU/GPU statistics stay together and parameter values sort numerically."""
    rows = [
        pr_comment._BenchmarkRow(
            "ReferenceChainSolve",
            "time_execution",
            "8",
            "1.0 s",
            "1.1 s",
            1.1,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulate",
            "time_execution",
            "10000",
            "1.0 s",
            "1.1 s",
            1.1,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulateGpuPeakMem",
            "track_gpu_peak_mem",
            "100000",
            "1 GB",
            "2 GB",
            2.0,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulate",
            "time_execution",
            "1000",
            "100 ms",
            "110 ms",
            1.1,
        ),
        pr_comment._BenchmarkRow(
            "ReferenceChainSolve",
            "time_execution",
            "2",
            "1.0 s",
            "1.1 s",
            1.1,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulate",
            "time_execution",
            "100000",
            "10 s",
            "11 s",
            1.1,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulateGpuPeakMem",
            "track_gpu_peak_mem",
            "1000",
            "1 GB",
            "2 GB",
            2.0,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulateGpuPeakMem",
            "track_gpu_peak_mem",
            "10000",
            "1 GB",
            "2 GB",
            2.0,
        ),
    ]

    table = pr_comment._build_grouped_table(rows)

    expected_labels = [
        "Collective Household - Simulate (1000)",
        "Collective Household - Simulate (10000)",
        "Collective Household - Simulate (100000)",
        "Reference Chain - Solve (2)",
        "Reference Chain - Solve (8)",
    ]
    assert [table.index(label) for label in expected_labels] == sorted(
        table.index(label) for label in expected_labels
    )
    assert table.count("Collective Household - Simulate (1000)") == 1
    assert table.count("Collective Household - Simulate (10000)") == 1
    assert table.count("Collective Household - Simulate (100000)") == 1


def test_grouped_table_labels_fixed_parameter_of_gpu_wrapper():
    """A no-param GPU wrapper joins the concrete case it actually measures."""
    rows = [
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulateGpuPeakMem",
            "track_gpu_peak_mem",
            "",
            "1 GB",
            "2 GB",
            2.0,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSimulate",
            "time_execution",
            "100000",
            "10 s",
            "11 s",
            1.1,
        ),
    ]

    table = pr_comment._build_grouped_table(rows)

    assert table.count("Collective Household - Simulate (100000)") == 1
    assert "|  | peak GPU mem |" in table


def test_baseline_fetch_retries_and_preserves_http_error(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """A failed site request is observable instead of masquerading as no baseline."""
    calls: list[str] = []

    def _failed_urlopen(request, **_kwargs):
        calls.append(request.full_url)
        raise URLError("unable to access benchmark site")

    monkeypatch.setattr(pr_comment, "urlopen", _failed_urlopen, raising=False)
    monkeypatch.setattr(
        pr_comment.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("baseline fetch must not invoke git"),
    )

    with pytest.raises(pr_comment._BaselineFetchError) as exc_info:
        pr_comment._fetch_baseline_from_site(
            machine_dir=tmp_path / "gpu-01",
            base_sha="98be11fb",
        )

    assert len(calls) == 3
    assert "unable to access benchmark site" in str(exc_info.value)
    assert "unable to access benchmark site" in capsys.readouterr().out


def test_baseline_fetch_downloads_matching_public_result(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The baseline is discovered and downloaded without cross-repository Git auth."""
    result_name = "98be11fb-existing-python.json"
    listing = json.dumps(
        [
            {"name": "98be11fb-compare.json", "download_url": "ignored"},
            {"name": result_name, "download_url": "https://example.test/result"},
        ]
    ).encode()
    responses = iter((listing, b'{"commit_hash": "98be11fb"}'))
    calls: list[str] = []

    def _successful_urlopen(request, **_kwargs):
        calls.append(request.full_url)
        return BytesIO(next(responses))

    monkeypatch.setattr(pr_comment, "urlopen", _successful_urlopen, raising=False)
    monkeypatch.setattr(
        pr_comment.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("baseline fetch must not invoke git"),
    )
    machine_dir = tmp_path / "gpu-01"
    machine_dir.mkdir()

    result = pr_comment._fetch_baseline_from_site(
        machine_dir=machine_dir,
        base_sha="98be11fb",
    )

    assert result == machine_dir / result_name
    assert result.read_bytes() == b'{"commit_hash": "98be11fb"}'
    assert calls[1] == "https://example.test/result"


def test_raw_comment_distinguishes_retrieval_failure_from_missing_results():
    """The fallback comment tells readers when infrastructure retrieval failed."""
    body = pr_comment._format_raw_comment(
        head_sha="77ebdd72",
        raw_md="| result |",
        baseline_note=(
            "Baseline retrieval failed for merge-base `98be11fb`; see the job log."
        ),
    )

    assert "HEAD only — baseline retrieval failed" in body
    assert "Baseline retrieval failed for merge-base `98be11fb`" in body
    assert "Run benchmarks on main" not in body
