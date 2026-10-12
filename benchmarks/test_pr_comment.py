"""Tests for the benchmark PR-comment formatter."""

# ruff: noqa: SLF001

import json
import re
import tomllib
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest

from benchmarks import pr_comment

if TYPE_CHECKING:
    from _lcm.typing import JSONValue


@pytest.mark.parametrize(
    ("bench_name", "value", "expected"),
    [
        (
            (
                "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
                "track_peak_gpu_mem_automatic_solve_simulate"
            ),
            2_500_000_000.0,
            "2.50 GB",
        ),
        (
            (
                "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
                "track_peak_gpu_mem_solve_save_all_persistable"
            ),
            32_000_000.0,
            "32 MB",
        ),
        (
            (
                "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
                "track_peak_gpu_mem_load_supplied_solution_simulate"
            ),
            1_000_000_000.0,
            "1.00 GB",
        ),
        ("bench_example.Example.peakmem_execution", 32_000_000.0, "32 MB"),
        ("bench_example.Example.track_gpu_peak_mem", 2_500_000_000.0, "2.50 GB"),
        ("bench_example.Example.track_compilation_time", 2.5, "2.50 s"),
        ("bench_mahler_yum.MahlerYumBudgetedGpu.time_execution", 0.125, "125.0 ms"),
    ],
)
def test_format_value_uses_memory_and_timing_units(
    *, bench_name: str, value: float, expected: str
) -> None:
    """GPU memory phases display bytes in MB or GB; timing metrics keep time units."""
    assert pr_comment._format_value(bench_name=bench_name, value=value) == expected


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


def test_comparison_matches_combined_aca_metrics_to_legacy_names(tmp_path: Path):
    """The combined measurement retains comparison continuity with main."""
    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_aca_baseline.AcaBaseline.time_execution": [[2.0], []],
                    "bench_aca_baseline.AcaBaseline.peakmem_execution": [
                        [200.0],
                        [],
                    ],
                    "bench_aca_baseline.AcaBaseline.track_compilation_time": [
                        [6.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    head_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_aca_baseline.AcaBaseline.track_execution_time": [
                        [1.0],
                        [],
                    ],
                    "bench_aca_baseline.AcaBaseline.track_peak_cpu_mem": [
                        [100.0],
                        [],
                    ],
                    "bench_aca_baseline.AcaBaseline.track_compilation_time": [
                        [3.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    rows = pr_comment._build_comparison_rows(
        base_file=base_file,
        head_file=head_file,
    )

    assert {row.method_name for row in rows} == {
        "time_execution",
        "peakmem_execution",
        "track_compilation_time",
    }
    assert {row.ratio for row in rows} == {0.5}


def test_comparison_omits_benchmarks_that_head_did_not_run(tmp_path: Path):
    """A benchmark with a main result but none on HEAD gets no row and no alert."""
    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_aca_baseline.AcaBaseline.track_execution_time": [
                        [2.0],
                        [],
                    ],
                    "bench_aca_baseline.AcaBaselineDebugLog.track_execution_time": [
                        [5.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    head_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_aca_baseline.AcaBaseline.track_execution_time": [
                        [3.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    rows = pr_comment._build_comparison_rows(base_file=base_file, head_file=head_file)

    assert rows == [
        pr_comment._BenchmarkRow(
            class_name="AcaBaseline",
            method_name="time_execution",
            params="",
            before_value="2.000 s",
            after_value="3.000 s",
            ratio=1.5,
        )
    ]


def test_benchmark_report_labels_are_specific_to_verified_workloads(
    tmp_path: Path,
) -> None:
    """Only verified automatic solve-and-simulate workloads get phase labels."""
    rows = [
        pr_comment._BenchmarkRow(
            "AcaBaseline",
            "track_compilation_time",
            "",
            "9.0 s",
            "10.0 s",
            1.11,
        ),
        pr_comment._BenchmarkRow(
            "AcaBaseline",
            "time_execution",
            "",
            "1.5 s",
            "1.6 s",
            1.07,
        ),
        pr_comment._BenchmarkRow(
            "MahlerYumBudgetedGpu",
            "track_compilation_time",
            "",
            "9.0 s",
            "10.0 s",
            1.11,
        ),
        pr_comment._BenchmarkRow(
            "MahlerYumBudgetedGpu",
            "time_execution",
            "",
            "1.5 s",
            "1.6 s",
            1.07,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdConstruct",
            "time_execution",
            "",
            "1.5 s",
            "1.6 s",
            1.07,
        ),
        pr_comment._BenchmarkRow(
            "PrecautionarySavingsSimulate",
            "time_execution",
            "",
            "10 ms",
            "11 ms",
            1.07,
        ),
        pr_comment._BenchmarkRow(
            "CollectiveHouseholdSolve",
            "track_compilation_time",
            "",
            "1.5 s",
            "1.6 s",
            1.07,
        ),
    ]

    table = pr_comment._build_grouped_table(rows)

    assert "| ACA (reduced) |" in table
    assert "tiny continuous grids" not in table
    assert table.count("cold solve + simulate (first run, includes compilation)") == 2
    assert table.count("warm solve + simulate (reuses compiled code)") == 2
    assert table.count("execution time") == 2
    assert table.count("first call (including compilation)") == 1

    result_file = tmp_path / "head.json"
    result_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_aca_baseline.AcaBaseline.track_compilation_time": [
                        [10.0],
                        [],
                    ],
                    "bench_aca_baseline.AcaBaseline.time_execution": [[1.6], []],
                    ("bench_mahler_yum.MahlerYumBudgetedGpu.track_compilation_time"): [
                        [10.0],
                        [],
                    ],
                    "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution": [
                        [1.6],
                        [],
                    ],
                    (
                        "bench_collective_household.CollectiveHouseholdConstruct."
                        "time_execution"
                    ): [[1.6], []],
                    (
                        "bench_precautionary_savings.PrecautionarySavingsSimulate."
                        "time_execution"
                    ): [[0.011], []],
                    (
                        "bench_collective_household.CollectiveHouseholdSolve."
                        "track_compilation_time"
                    ): [[1.6], []],
                }
            }
        ),
        encoding="utf-8",
    )
    raw_table = pr_comment._format_raw_results(
        result_file=result_file, head_sha="77ebdd72"
    )

    assert (
        raw_table.count("cold solve + simulate (first run, includes compilation)") == 2
    )
    assert raw_table.count("warm solve + simulate (reuses compiled code)") == 2
    assert raw_table.count("execution time") == 2
    assert raw_table.count("first call (including compilation)") == 1


def test_mahler_policy_identities_do_not_compare_to_legacy_default(tmp_path: Path):
    """Matching configured identities compare only within their own series."""
    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_mahler_yum.MahlerYum.time_execution": [[10.0], []],
                    "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution": [
                        [20.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    head_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution": [
                        [25.0],
                        [],
                    ],
                    (
                        "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
                        "track_peak_gpu_mem_automatic_solve_simulate"
                    ): [
                        [100.0],
                        [],
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    rows = pr_comment._build_comparison_rows(base_file=base_file, head_file=head_file)

    timing = next(row for row in rows if row.class_name == "MahlerYumBudgetedGpu")
    memory = next(
        row for row in rows if row.class_name == "MahlerYumBudgetedGpuPeakMem"
    )
    assert timing.ratio == 1.25
    assert memory.ratio is None
    table = pr_comment._build_grouped_table(rows)
    assert table.count("| Mahler-Yum |") == 1
    assert "capacity-half" not in table


def test_mahler_configured_identity_has_no_ratio_against_legacy_default(
    tmp_path: Path,
) -> None:
    """The raw parser ignores ASV version when it joins comparison keys."""
    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(
        json.dumps(
            {"results": {"bench_mahler_yum.MahlerYum.time_execution": [[10.0], []]}}
        ),
        encoding="utf-8",
    )
    head_file.write_text(
        json.dumps(
            {
                "results": {
                    "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution": [
                        [20.0],
                        [],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    rows = pr_comment._build_comparison_rows(base_file=base_file, head_file=head_file)

    assert len(rows) == 1
    assert rows[0].class_name == "MahlerYumBudgetedGpu"
    assert rows[0].ratio is None


@pytest.mark.parametrize(
    "prefix", ["", "bench_simulation_dispatch.SimulationDispatch."]
)
@pytest.mark.parametrize(
    ("method", "value", "expected"),
    [
        ("track_host_ms_per_period_regime", 2.5, "2.50 ms"),
        ("track_host_ms_per_period_regime", 0, "0.00 ms"),
        ("track_second_call_compiles", 0, "0"),
        ("track_second_call_compiles", 12, "12"),
        ("track_compilation_time", 2.5, "2.50 s"),
    ],
)
def test_simulation_dispatch_metric_units(
    *, prefix: str, method: str, value: float, expected: str
) -> None:
    """Bare and fully qualified metric names retain their declared units."""
    assert pr_comment._format_value(bench_name=prefix + method, value=value) == expected


def test_simulation_dispatch_units_preserve_raw_values_and_ratios(
    tmp_path: Path,
) -> None:
    """Display units change neither stored metrics nor zero-baseline policy."""
    prefix = "bench_simulation_dispatch.SimulationDispatch."
    methods = ("track_host_ms_per_period_regime", "track_second_call_compiles")
    base_file, head_file = tmp_path / "base.json", tmp_path / "head.json"
    for path, values in ((base_file, (5.0, 0)), (head_file, (2.5, 0))):
        path.write_text(
            json.dumps(
                {
                    "results": {
                        prefix + method: [[value], []]
                        for method, value in zip(methods, values, strict=True)
                    }
                }
            )
        )
    before = head_file.read_bytes()
    rows = pr_comment._build_comparison_rows(base_file=base_file, head_file=head_file)
    by_method = {row.method_name: row for row in rows}
    timing, count = (by_method[method] for method in methods)
    assert (timing.before_value, timing.after_value, timing.ratio) == (
        "5.00 ms",
        "2.50 ms",
        0.5,
    )
    assert (count.before_value, count.after_value, count.ratio) == ("", "0", None)
    raw = pr_comment._parse_raw_values(head_file)
    assert raw == {
        ("SimulationDispatch", methods[0], ""): 2.5,
        ("SimulationDispatch", methods[1], ""): 0,
    }
    table = pr_comment._format_raw_results(result_file=head_file, head_sha="head")
    assert "| 2.50 ms |" in table
    assert "| 0 |" in table
    assert "2.50 s" not in table
    assert "0.00 s" not in table
    assert head_file.read_bytes() == before


@pytest.mark.parametrize(
    ("benchmark_name", "expected"),
    [
        (
            "bench_aca_baseline.AcaBaseline.track_execution_time",
            ("ACA (reduced)", "warm solve + simulate (reuses compiled code)"),
        ),
        (
            (
                "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
                "track_peak_gpu_mem_load_supplied_solution_simulate"
            ),
            ("Mahler-Yum", "peak GPU mem: load saved solution + simulate"),
        ),
        (
            "bench_collective_household.ReferenceChainSolve.track_execution_time",
            ("Reference Chain - Solve", "execution time"),
        ),
        (
            "bench_simulation_dispatch.SimulationDispatch.track_second_call_compiles",
            (
                "SimulationDispatch",
                "backend compile requests on second simulation call",
            ),
        ),
    ],
)
def test_display_names_match_the_comparison_table(
    *, benchmark_name: str, expected: tuple[str, str]
) -> None:
    """A dashboard benchmark gets the table's benchmark and statistic labels."""
    assert pr_comment.display_names(benchmark_name) == expected


def test_display_sort_key_orders_families_then_execution_time_first() -> None:
    """Families follow the table order; each starts with its execution time."""
    names = [
        "bench_mahler_yum.MahlerYumBudgetedGpu.track_compilation_time",
        "bench_mahler_yum.MahlerYumBudgetedGpu.track_execution_time",
        "bench_aca_baseline.AcaBaseline.track_peak_cpu_mem",
        "bench_aca_baseline.AcaBaseline.track_compilation_time",
        "bench_aca_baseline.AcaBaseline.track_execution_time",
    ]

    assert sorted(names, key=pr_comment.display_sort_key) == [
        "bench_aca_baseline.AcaBaseline.track_execution_time",
        "bench_aca_baseline.AcaBaseline.track_compilation_time",
        "bench_aca_baseline.AcaBaseline.track_peak_cpu_mem",
        "bench_mahler_yum.MahlerYumBudgetedGpu.track_execution_time",
        "bench_mahler_yum.MahlerYumBudgetedGpu.track_compilation_time",
    ]


def _pixi_tasks() -> dict[str, JSONValue]:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pixi"]["tasks"]


@pytest.mark.parametrize(
    ("benchmark_name", "selected"),
    [
        ("bench_aca_baseline.AcaBaseline.track_execution_time", True),
        ("bench_aca_baseline.AcaBaseline.track_peak_cpu_mem", True),
        ("bench_mahler_yum.MahlerYumBudgetedGpu.track_execution_time", True),
        ("bench_aca_baseline.AcaBaselineDebugLog.track_execution_time", False),
        ("bench_aca_baseline.AcaBaselineDebugLog.track_compilation_time", False),
        ("bench_aca_baseline.AcaBaselineDebugLog.track_peak_cpu_mem", False),
    ],
)
def test_pr_benchmark_run_skips_only_the_aca_debug_log_timing(
    *, benchmark_name: str, selected: bool
) -> None:
    """The PR run's ASV selection, matched as ASV matches it, skips debug logging."""
    (run_step, _) = _pixi_tasks()["asv-run-and-pr-comment"]["depends-on"]
    assert run_step["task"] == "asv-run"
    (bench_regex,) = run_step["args"]
    assert (re.search(bench_regex, benchmark_name) is not None) is selected


def test_main_benchmark_run_selects_every_benchmark() -> None:
    """The main-branch run uses the default selection, which matches any name."""
    tasks = _pixi_tasks()
    assert tasks["asv-run-and-publish-main"]["depends-on"] == [
        "asv-run",
        "asv-publish",
    ]
    (bench_arg,) = tasks["asv-run"]["args"]
    assert re.search(
        bench_arg["default"],
        "bench_aca_baseline.AcaBaselineDebugLog.track_execution_time",
    )
