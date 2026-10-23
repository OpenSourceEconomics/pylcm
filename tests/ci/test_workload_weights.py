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
_MISSING_FILE = "tests/solution/test_egm_passive_asset_row.py"
_SECONDS = 7.25


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


def test_refresh_manifest_leaves_every_measured_weight_byte_identical(
    tmp_path: Path,
) -> None:
    """Filling one missing file rewrites no weight a leg already has."""
    before = _manifest_without(file_=_MISSING_FILE)

    refreshed = refresh_manifest(
        manifest_text=json.dumps(before),
        repo_root=_ROOT,
        find_runs=lambda: (_RUN,),
        read_reports=lambda _run: _solution_reports(
            directory=tmp_path, files=(_MISSING_FILE, _MEASURED_FILE)
        ),
        run_manifest=lambda _run: _run_manifest(files={_MISSING_FILE, _MEASURED_FILE}),
    )
    after = json.loads(refreshed.text)["leg_weights"]

    assert {
        leg: {file_: after[leg]["seconds"][file_] for file_ in entry["seconds"]}
        for leg, entry in before["leg_weights"].items()
    } == {leg: entry["seconds"] for leg, entry in before["leg_weights"].items()}


def test_refresh_manifest_fills_a_missing_file_from_a_run_of_another_layout(
    tmp_path: Path,
) -> None:
    """A run of an earlier shard count measures a file once it has all its reports.

    Per-file weights do not depend on the shard count, so completeness is
    judged against the run's own layout, not the current manifest's.
    """
    shards = _solution_shards() - 2

    refreshed = refresh_manifest(
        manifest_text=json.dumps(_manifest_without(file_=_MISSING_FILE)),
        repo_root=_ROOT,
        find_runs=lambda: (_RUN,),
        read_reports=lambda _run: _solution_reports(
            directory=tmp_path, files=(_MISSING_FILE,), shards=shards
        ),
        run_manifest=lambda _run: _run_manifest(files={_MISSING_FILE}, shards=shards),
    )
    seconds = json.loads(refreshed.text)["leg_weights"][_LEG]["seconds"]

    assert seconds[_MISSING_FILE] == _SECONDS * shards


def test_refresh_manifest_fills_from_the_latest_run_that_measured_the_file(
    tmp_path: Path,
) -> None:
    """A newer run with an incomplete leg defers to the older complete one."""
    newer = CompletedRun(run_id=456, attempt=1, head_sha="def456")
    complete = _solution_reports(directory=tmp_path, files=(_MISSING_FILE,))

    refreshed = refresh_manifest(
        manifest_text=json.dumps(_manifest_without(file_=_MISSING_FILE)),
        repo_root=_ROOT,
        find_runs=lambda: (newer, _RUN),
        read_reports=lambda run: (
            {_LEG: complete[_LEG][1:]} if run == newer else complete
        ),
        run_manifest=lambda _run: _run_manifest(files={_MISSING_FILE}),
    )
    entry = json.loads(refreshed.text)["leg_weights"][_LEG]

    assert (entry["seconds"][_MISSING_FILE], entry["filled_from"][_MISSING_FILE]) == (
        _SECONDS * _solution_shards(),
        "run 123 attempt 1 (head abc123)",
    )


def test_refresh_manifest_measures_a_selected_file_without_testcases_at_zero(
    tmp_path: Path,
) -> None:
    """The run's leg selected the file and it recorded no testcase there."""
    refreshed = refresh_manifest(
        manifest_text=json.dumps(_manifest_without(file_=_MISSING_FILE)),
        repo_root=_ROOT,
        find_runs=lambda: (_RUN,),
        read_reports=lambda _run: _solution_reports(
            directory=tmp_path, files=(_MEASURED_FILE,)
        ),
        run_manifest=lambda _run: _run_manifest(files={_MISSING_FILE, _MEASURED_FILE}),
    )

    assert json.loads(refreshed.text)["leg_weights"][_LEG]["seconds"][
        _MISSING_FILE
    ] == pytest.approx(0.0)


def test_refresh_manifest_leaves_a_file_the_run_did_not_contain_unmeasured(
    tmp_path: Path,
) -> None:
    """A file absent from the run's manifest stays unmeasured for the gate."""
    text = json.dumps(_manifest_without(file_=_MISSING_FILE))

    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_runs=lambda: (_RUN,),
        read_reports=lambda _run: _solution_reports(
            directory=tmp_path, files=(_MEASURED_FILE,)
        ),
        run_manifest=lambda _run: _run_manifest(files={_MEASURED_FILE}),
    )

    assert (refreshed.text == text, _MISSING_FILE in refreshed.message) == (
        True,
        True,
    )


def test_refresh_manifest_downloads_nothing_when_no_file_is_missing() -> None:
    """With every shard file weighted, no run is looked up and nothing changes."""
    text = (_ROOT / "tests/ci/ci-workloads.json").read_text(encoding="utf-8")

    def unreachable(*_args: CompletedRun) -> NoReturn:
        raise AssertionError("no run may be looked up or downloaded")

    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_runs=unreachable,
        read_reports=unreachable,
        run_manifest=unreachable,
    )

    assert refreshed.text == text


@pytest.mark.parametrize(
    "reason",
    ["no open pull request for branch topic", "`gh` unavailable: offline"],
)
def test_refresh_manifest_leaves_the_file_unchanged_without_a_run(reason: str) -> None:
    """No PR, no completed run or no network leaves the file and prints why."""
    text = json.dumps(_manifest_without(file_=_MISSING_FILE))

    def unreachable(_run: CompletedRun) -> NoReturn:
        raise AssertionError("no artifact may be read without a run")

    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=_ROOT,
        find_runs=lambda: Skip(reason=reason),
        read_reports=unreachable,
        run_manifest=unreachable,
    )

    assert (refreshed.text == text, reason in refreshed.message) == (True, True)


def _solution_shards() -> int:
    return ci_workloads.shard_layout()["solution"][_LEG]["shards"]


def _manifest_without(*, file_: str) -> ci_workloads.WorkloadManifest:
    """Return the committed manifest with `file_` unweighted on `_LEG`."""
    manifest = json.loads((_ROOT / "tests/ci/ci-workloads.json").read_text())
    for key in ("seconds", "counts"):
        manifest["leg_weights"][_LEG][key].pop(file_)
    return manifest


def _run_manifest(
    *, files: set[str], shards: int | None = None
) -> ci_workloads.WorkloadManifest:
    """Return a run's manifest whose only leg is `_LEG`, sharding `files`."""
    manifest = json.loads((_ROOT / "tests/ci/ci-workloads.json").read_text())
    count = _solution_shards() if shards is None else shards
    manifest["shard_layout"] = {
        "general": {},
        "solution": {_LEG: {"shards": count, "workers": 2}},
    }
    manifest["invocations"] = [
        {"id": "solution-shard-1-fp64-linux", "files": sorted(files)}
    ]
    return manifest


def _solution_reports(
    *, directory: Path, files: tuple[str, ...], shards: int | None = None
) -> dict[str, list[Path]]:
    """Write one report per shard, each giving every file `_SECONDS`."""
    cases = "".join(
        f'<testcase classname="{file_.removesuffix(".py").replace("/", ".")}" '
        f'name="test_case" time="{_SECONDS}"/>'
        for file_ in files
    )
    paths = []
    for shard in range(1, (_solution_shards() if shards is None else shards) + 1):
        path = directory / f"junit-64-{shard}.xml"
        path.write_text(
            f"<testsuites><testsuite>{cases}</testsuite></testsuites>",
            encoding="utf-8",
        )
        paths.append(path)
    return {_LEG: paths}
