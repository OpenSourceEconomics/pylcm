"""Shard weights come from CI only, and CI fails while a shard file lacks one."""

import json
from pathlib import Path
from typing import NoReturn

import pytest

from tests.ci import ci_workloads
from tests.ci.workload_weights import (
    LEG_ARTIFACTS,
    CompletedRun,
    Skip,
    gate_message,
    refresh_manifest,
    unmeasured_files,
)

_ROOT = Path(__file__).resolve().parents[2]
_RUN = CompletedRun(run_id=123, attempt=1, head_sha="abc123")
_LEG = "fp64-solution"
_MEASURED_FILE = "tests/solution/test_solution_result.py"
_ZERO_FILE = "tests/solution/test_nbegm_axes.py"


def test_unmeasured_files_names_a_shard_file_with_no_weight() -> None:
    """A file a leg shards over without a measured weight fails the gate."""
    missing = unmeasured_files(
        leg_universes={"fp64-linux": ("tests/test_a.py", "tests/test_b.py")},
        leg_weights={"fp64-linux": {"tests/test_a.py": 3.0}},
    )

    assert missing == {"fp64-linux": ("tests/test_b.py",)}


def test_unmeasured_files_passes_when_every_shard_file_is_weighted() -> None:
    """A measured weight of zero is a measurement, not a gap."""
    missing = unmeasured_files(
        leg_universes={"fp64-linux": ("tests/test_a.py", "tests/test_b.py")},
        leg_weights={"fp64-linux": {"tests/test_a.py": 3.0, "tests/test_b.py": 0.0}},
    )

    assert missing == {}


def test_gate_message_names_the_files_and_the_commit_that_fills_them_in() -> None:
    """The failure says which files lack a weight and what will supply it."""
    message = gate_message({"fp64-linux": ("tests/test_b.py",)})

    assert all(
        fragment in message
        for fragment in ("fp64-linux: tests/test_b.py", "next commit", "this run")
    )


def test_legs_have_a_junit_artifact_pattern_each() -> None:
    """Every leg the manifest weighs names the JUnit artifacts it is refreshed from."""
    assert set(LEG_ARTIFACTS) == set(ci_workloads.leg_names())


def test_refresh_manifest_writes_the_run_weights_deterministically(
    tmp_path: Path,
) -> None:
    """The same run gives byte-identical manifests, carrying its measurements."""
    text = (_ROOT / "tests/ci/ci-workloads.json").read_text(encoding="utf-8")
    reports = _solution_reports(directory=tmp_path, seconds=7.25)

    first = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_run=lambda: _RUN,
        read_reports=lambda _run: reports,
        selected_at_run=lambda _run: {_LEG: frozenset({_MEASURED_FILE, _ZERO_FILE})},
    )
    second = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_run=lambda: _RUN,
        read_reports=lambda _run: reports,
        selected_at_run=lambda _run: {_LEG: frozenset({_MEASURED_FILE, _ZERO_FILE})},
    )
    seconds = json.loads(first.text)["leg_weights"][_LEG]["seconds"]

    assert (first.text == second.text, seconds) == (
        True,
        {_MEASURED_FILE: 7.25 * _solution_shards(), _ZERO_FILE: 0.0},
    )


def test_refresh_manifest_leaves_an_incomplete_leg_alone(tmp_path: Path) -> None:
    """A leg missing a shard's report keeps its weights, and the reason is printed."""
    text = (_ROOT / "tests/ci/ci-workloads.json").read_text(encoding="utf-8")
    reports = _solution_reports(directory=tmp_path, seconds=7.25)
    reports = {_LEG: reports[_LEG][1:]}

    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_run=lambda: _RUN,
        read_reports=lambda _run: reports,
        selected_at_run=lambda _run: {_LEG: frozenset({_MEASURED_FILE})},
    )

    assert (refreshed.text == text, f"{_LEG}: " in refreshed.message) == (True, True)


@pytest.mark.parametrize(
    "reason",
    ["no open pull request for branch topic", "`gh` unavailable: offline"],
)
def test_refresh_manifest_leaves_the_file_unchanged_without_a_run(reason: str) -> None:
    """No PR, no completed run or no network leaves the file and prints why."""
    text = (_ROOT / "tests/ci/ci-workloads.json").read_text(encoding="utf-8")

    def unreachable(_run: CompletedRun) -> NoReturn:
        raise AssertionError("no artifact may be read without a run")

    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_run=lambda: Skip(reason=reason),
        read_reports=unreachable,
        selected_at_run=unreachable,
    )

    assert (refreshed.text == text, reason in refreshed.message) == (True, True)


def _solution_shards() -> int:
    return ci_workloads.shard_layout()["solution"][_LEG]["shards"]


def _solution_reports(*, directory: Path, seconds: float) -> dict[str, list[Path]]:
    classname = _MEASURED_FILE.removesuffix(".py").replace("/", ".")
    paths = []
    for shard in range(1, _solution_shards() + 1):
        path = directory / f"junit-64-{shard}.xml"
        path.write_text(
            f'<testsuites><testsuite><testcase classname="{classname}" '
            f'name="test_case" time="{seconds}"/></testsuite></testsuites>',
            encoding="utf-8",
        )
        paths.append(path)
    return {_LEG: paths}
