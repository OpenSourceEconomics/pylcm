"""The source-only certificate campaign runs exactly when its verdict can change."""

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

from tests.candidate_certificate import direct_flow
from tests.ci import certificate_campaign_gate, check_coverage_manifest, ci_workloads
from tests.ci._workflow_types import Workflow
from tests.ci.cpu_suite_invocations import cpu_suite_invocation_argvs

_REPO_ROOT = Path(__file__).parents[2]
_WORKFLOW_PATH = ".github/workflows/cpu.yml"
_GATE_PATH = Path(certificate_campaign_gate.__file__).relative_to(_REPO_ROOT).as_posix()
_JOB = "tests-source-contract"
_RUNS = "steps.campaign.outputs.run == 'true'"
_NOT_SCHEDULED = "github.event_name != 'schedule'"


def _workflow() -> Workflow:
    return yaml.safe_load((_REPO_ROOT / _WORKFLOW_PATH).read_text(encoding="utf-8"))


def _sealed_sources() -> list[str]:
    inventory = json.loads(
        (_REPO_ROOT / "tests/candidate_certificate/sources.json").read_text(
            encoding="utf-8"
        )
    )
    return [entry["path"] for entry in inventory["sources"]]


def _campaign_test_files() -> list[str]:
    """Return the test files the campaign step names, read from `cpu.yml`."""
    step = next(
        step
        for step in _workflow()["jobs"][_JOB]["steps"]
        if step.get("name") == "Run the source-only certificate campaign"
    )
    (argv,) = cpu_suite_invocation_argvs(step["run"])
    return [arg for arg in argv if arg.startswith("tests/") and arg.endswith(".py")]


def _decision(
    *,
    event: str,
    changed_files: list[str] | None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Return the gate's `run=` line for one event and changed-file list."""
    argv = ["--event", event]
    if changed_files is not None:
        listing = tmp_path / "changed-files"
        listing.write_text("".join(f"{name}\n" for name in changed_files))
        argv += ["--changed-files", str(listing)]
    assert certificate_campaign_gate.main(argv) == 0
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize(
    "changed_file",
    [
        "tests/candidate_certificate/verify.py",
        "tests/candidate_certificate/sources.json",
        _WORKFLOW_PATH,
        _GATE_PATH,
        "pixi.lock",
    ],
)
def test_a_pull_request_changing_a_campaign_input_runs_it(
    *, changed_file: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Certificate code, the workflow, the gate and the lock file each trigger."""
    assert (
        _decision(
            event="pull_request",
            changed_files=["docs/index.md", changed_file],
            tmp_path=tmp_path,
            capsys=capsys,
        )
        == "run=true"
    )


def test_every_campaign_test_file_runs_it(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each test file the campaign step names triggers the campaign on its own."""
    files = _campaign_test_files()
    assert len(files) == 6
    assert [
        _decision(
            event="pull_request",
            changed_files=[name],
            tmp_path=tmp_path,
            capsys=capsys,
        )
        for name in files
    ] == ["run=true"] * len(files)


def test_every_sealed_and_corridor_source_runs_it(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source sealed in `sources.json` or in the corridor tuple triggers alone."""
    sources = sorted({*_sealed_sources(), *direct_flow._CERTIFIED_CORRIDOR_SOURCES})
    assert sources
    assert [
        _decision(
            event="pull_request",
            changed_files=[name],
            tmp_path=tmp_path,
            capsys=capsys,
        )
        for name in sources
    ] == ["run=true"] * len(sources)


@pytest.mark.parametrize(
    "changed_files",
    [
        [],
        ["src/lcm/__init__.py", "tests/test_model.py", "docs/index.md"],
    ],
)
def test_a_pull_request_changing_no_campaign_input_skips_it(
    *, changed_files: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Uncertified sources, other tests and documentation leave the verdict alone."""
    assert not set(changed_files) & set(_sealed_sources())
    assert (
        _decision(
            event="pull_request",
            changed_files=changed_files,
            tmp_path=tmp_path,
            capsys=capsys,
        )
        == "run=false"
    )


def test_a_pull_request_with_unknown_changed_files_runs_it(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a changed-file list the gate fails open."""
    assert (
        _decision(
            event="pull_request", changed_files=None, tmp_path=tmp_path, capsys=capsys
        )
        == "run=true"
    )


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
@pytest.mark.parametrize("changed_files", [None, [], ["docs/index.md"]])
def test_every_other_event_runs_it(
    *,
    event: str,
    changed_files: list[str] | None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pushes, scheduled and manual runs ignore the changed files."""
    assert (
        _decision(
            event=event, changed_files=changed_files, tmp_path=tmp_path, capsys=capsys
        )
        == "run=true"
    )


def test_the_gate_runs_on_the_runner_python_before_any_install() -> None:
    """The gate parses as Python 3.12 and imports only the standard library."""
    module = ast.parse(
        (_REPO_ROOT / _GATE_PATH).read_text(encoding="utf-8"), feature_version=(3, 12)
    )
    imported = {
        alias.name.partition(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").partition(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported
    assert imported <= set(sys.stdlib_module_names)


def test_the_campaign_installs_and_runs_only_when_the_gate_says_so() -> None:
    """Every step after the gate is conditioned on its `run` output."""
    job = _workflow()["jobs"][_JOB]
    steps = job["steps"]
    (gate,) = [
        index for index, step in enumerate(steps) if step.get("id") == "campaign"
    ]
    assert _GATE_PATH in steps[gate]["run"]
    assert all(step["uses"].startswith("actions/checkout@") for step in steps[:gate])
    assert [step.get("if") for step in steps[gate + 1 :]] == [
        _RUNS,
        _RUNS,
        _RUNS,
        f"always() && {_RUNS}",
        f"always() && {_RUNS}",
    ]
    assert job["outputs"] == {"run": "${{ steps.campaign.outputs.run }}"}


def test_the_aggregate_still_requires_the_campaign_lane_to_succeed() -> None:
    """A skipped campaign is a succeeding lane, never an accepted `skipped`."""
    aggregate = _workflow()["jobs"]["cpu"]
    assert _JOB in aggregate["needs"]
    gate_runs = [
        step["run"] for step in aggregate["steps"] if "test " in step.get("run", "")
    ]
    assert len(gate_runs) == 1
    assert f'test "${{{{needs.{_JOB}.result}}}}" = "success"' in gate_runs[0]


def test_coverage_excuses_the_campaign_lane_only_when_it_did_not_run() -> None:
    """The combine stage names the lane as not run from the gate's output."""
    (check,) = [
        step
        for step in _workflow()["jobs"]["coverage"]["steps"]
        if "check_coverage_manifest" in step.get("run", "")
    ]
    assert check["env"] == {"CAMPAIGN_RUN": f"${{{{ needs.{_JOB}.outputs.run }}}}"}
    assert 'if test "$CAMPAIGN_RUN" = false; then' in check["run"]
    assert "--not-run coverage-source-contract" in check["run"]


def test_a_scheduled_run_starts_only_the_campaign() -> None:
    """Every other independent job and the aggregate stay off on a schedule."""
    workflow = _workflow()
    triggers = yaml.safe_load((_REPO_ROOT / _WORKFLOW_PATH).read_text())[True]
    assert {"schedule", "workflow_dispatch"} <= set(triggers)
    conditions = {
        name: job.get("if")
        for name, job in workflow["jobs"].items()
        if name != _JOB and "needs" not in job
    }
    assert conditions
    assert conditions == dict.fromkeys(conditions, _NOT_SCHEDULED)
    assert workflow["jobs"]["cpu"]["if"] == f"always() && {_NOT_SCHEDULED}"


@pytest.mark.parametrize(
    ("present", "not_run", "expected_exit"),
    [
        (("coverage-general-1",), ["coverage-source-contract"], 0),
        (("coverage-general-1", "coverage-source-contract"), [], 0),
        (
            ("coverage-general-1", "coverage-source-contract"),
            ["coverage-source-contract"],
            1,
        ),
        (("coverage-general-1",), [], 1),
        (("coverage-general-1", "coverage-source-contract"), ["coverage-unknown"], 1),
    ],
)
def test_coverage_check_excuses_only_a_named_recorded_lane_that_delivered_nothing(
    *,
    present: tuple[str, ...],
    not_run: list[str],
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane named as not run must be recorded and must have uploaded nothing."""
    for name in present:
        (tmp_path / name).mkdir()
        (tmp_path / name / "coverage.xml").write_text("<coverage/>\n")
    monkeypatch.setattr(
        ci_workloads,
        "coverage_contributors",
        lambda: ("coverage-general-1", "coverage-source-contract"),
    )
    argv = [str(tmp_path), *(f"--not-run={name}" for name in not_run)]
    assert check_coverage_manifest.main(argv) == expected_exit
