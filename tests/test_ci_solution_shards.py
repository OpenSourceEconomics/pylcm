"""The slow solution battery is partitioned exhaustively across CI jobs."""

from pathlib import Path

import pytest
import yaml

from tests.ci.shard_test_files import assign_test_files


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
    """Coverage statuses use the base leg and all three fp64 solution shards."""
    config = yaml.safe_load(Path("codecov.yml").read_text(encoding="utf-8"))

    assert config["codecov"]["require_ci_to_pass"] is True
    assert config["comment"]["after_n_builds"] == 4
    assert config["codecov"]["notify"]["after_n_builds"] == 4
    assert config["flags"]["cpu-python"] == {
        "carryforward": False,
        "after_n_builds": 4,
    }


def test_every_cpu_coverage_upload_uses_the_cpu_python_flag() -> None:
    """The four fp64 reports identify themselves as CPU Python coverage."""
    workflow = yaml.safe_load(
        Path(".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    uploads = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", ())
        if step.get("uses") == "codecov/codecov-action@v7.0.0"
    ]

    assert len(uploads) == 2
    assert all(step["with"]["flags"] == "cpu-python" for step in uploads)
