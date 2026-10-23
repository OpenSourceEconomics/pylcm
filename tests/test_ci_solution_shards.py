"""The slow solution battery is partitioned exhaustively across CI jobs."""

from pathlib import Path

import pytest
import yaml

from tests.ci.shard_test_files import (
    assign_test_files,
    assign_weighted_test_files,
    weighted_shard_report,
)


def test_solution_shards_cover_each_test_file_once() -> None:
    """Every slow solution module belongs to exactly one non-empty CI shard."""
    files = tuple(sorted(Path("tests/solution").glob("test_*.py")))

    groups = assign_test_files(files=files, n_shards=3)

    assert all(groups)
    assert sorted(path for group in groups for path in group) == list(files)


def test_solution_shards_are_independent_of_input_order() -> None:
    """Repository traversal order does not change a file's assigned shard."""
    files = tuple(Path(f"tests/solution/test_case_{i}.py") for i in range(12))

    forward = assign_test_files(files=files, n_shards=3)
    reverse = assign_test_files(files=tuple(reversed(files)), n_shards=3)

    assert forward == reverse


@pytest.mark.parametrize("n_shards", [0, -1])
def test_solution_shards_require_a_positive_shard_count(n_shards: int) -> None:
    """A non-positive shard count is rejected instead of dropping every file."""
    with pytest.raises(ValueError, match="positive"):
        assign_test_files(
            files=(Path("tests/solution/test_case.py"),),
            n_shards=n_shards,
        )


def test_codecov_waits_for_the_complete_cpu_python_report_set() -> None:
    """Coverage statuses use the one combined report the coverage job uploads."""
    config = yaml.safe_load(Path("codecov.yml").read_text(encoding="utf-8"))

    assert config["codecov"]["require_ci_to_pass"] is True
    assert config["comment"]["after_n_builds"] == 1
    assert config["codecov"]["notify"]["after_n_builds"] == 1
    assert config["flags"]["cpu-python"] == {
        "carryforward": False,
        "after_n_builds": 1,
    }


def test_every_cpu_coverage_upload_uses_the_cpu_python_flag() -> None:
    """The one combined report identifies itself as CPU Python coverage."""
    workflow = yaml.safe_load(
        Path(".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    uploads = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("codecov/codecov-action@")
    ]

    assert len(uploads) == 1
    assert all(step["with"]["flags"] == "cpu-python" for step in uploads)


_MEASURED = {f"tests/solution/test_measured_{i}.py": 10.0 for i in range(3)}
_UNMEASURED = tuple(f"tests/solution/test_new_{i}.py" for i in range(6))


def test_weighted_shards_spread_files_with_no_measured_weight() -> None:
    """Files with no measured weight are spread over the shards, not stacked in one."""
    groups = assign_weighted_test_files(
        files=(*_MEASURED, *_UNMEASURED), n_shards=3, weights=_MEASURED
    )

    unmeasured_per_shard = [len(set(group) & set(_UNMEASURED)) for group in groups]

    assert unmeasured_per_shard == [2, 2, 2]


def test_weighted_shards_spread_files_when_no_weight_is_measured() -> None:
    """A leg with no measured weights still spreads its files evenly."""
    groups = assign_weighted_test_files(files=_UNMEASURED, n_shards=3, weights={})

    assert [len(group) for group in groups] == [2, 2, 2]


def test_weighted_shard_report_names_every_file_with_no_measured_weight() -> None:
    """The report lists unmeasured files as unweighted wherever they are placed."""
    reports = weighted_shard_report(
        files=(*_MEASURED, *_UNMEASURED), n_shards=3, weights=_MEASURED
    )

    named = sorted(file_ for report in reports for file_ in report.unweighted)

    assert named == sorted(_UNMEASURED)


def test_weighted_shard_report_counts_only_measured_seconds() -> None:
    """A placement weight is never reported as measured shard seconds."""
    reports = weighted_shard_report(
        files=(*_MEASURED, *_UNMEASURED), n_shards=3, weights=_MEASURED
    )

    assert [report.weighted_seconds for report in reports] == [10.0, 10.0, 10.0]
