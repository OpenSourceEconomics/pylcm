"""Required CI gates pass only on success or on genuine supersession.

The `superseded` action runs its real Bash and jq against canned Actions API
responses (a stub `gh` on `PATH`); the gate steps run their real shell with the
`needs` results substituted.
"""

import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).parents[2]
_BASH = shutil.which("bash") or "/bin/bash"
_GATES = (
    ("cpu.yml", "cpu"),
    ("gpu32.yml", "gpu32"),
    ("gpu32.yml", "gpu64"),
    ("gpu64.yml", "gpu64"),
    ("notebooks.yml", "notebooks"),
    ("ty.yml", "ty"),
)
_GATE_CONDITION = (
    "steps.superseded.outputs.superseded != 'true' || "
    "contains(needs.*.result, 'failure')"
)
_CURRENT = {
    "id": 100,
    "workflow_id": 7,
    "event": "pull_request",
    "head_branch": "fix",
    "head_sha": "a" * 40,
    "head_repository": {"full_name": "alice/pylcm"},
}
_STUB_GH = r"""#!/usr/bin/env bash
set -euo pipefail
case "$2" in
  */workflows/*/runs\?*) source="$STUB_DATA/runs.json" ;;
  */runs/*) source="$STUB_DATA/current.json" ;;
  *) echo "unexpected API URL: $2" >&2; exit 64 ;;
esac
jq -r "$4" "$source"
"""


def _run_action(
    *, tmp_path: Path, runs: list[dict[str, Any]], event: str = "pull_request"
) -> str:
    assert shutil.which("jq") is not None
    action = yaml.safe_load(
        (_REPO_ROOT / ".github/actions/superseded/action.yml").read_text()
    )
    (tmp_path / "current.json").write_text(json.dumps(_CURRENT))
    (tmp_path / "runs.json").write_text(json.dumps({"workflow_runs": runs}))
    gh = tmp_path / "gh"
    gh.write_text(_STUB_GH)
    gh.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        STUB_DATA=str(tmp_path),
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_RUN_ID=str(_CURRENT["id"]),
        GITHUB_EVENT_NAME=event,
        GITHUB_OUTPUT=str(tmp_path / "output"),
        BRANCH="fix",
    )
    subprocess.run(  # noqa: S603 - repository-owned action script
        [_BASH, "-e", "-o", "pipefail", "-c", action["runs"]["steps"][0]["run"]],
        env=env,
        check=True,
        capture_output=True,
    )
    return (tmp_path / "output").read_text().strip()


@pytest.mark.parametrize(
    ("newer", "event", "expected"),
    [
        ({}, "pull_request", "superseded=true"),
        ({"head_sha": "b" * 40}, "pull_request", "superseded=true"),
        (
            {"head_repository": {"full_name": "bob/pylcm"}, "head_sha": "b" * 40},
            "pull_request",
            "superseded=false",
        ),
        ({"event": "workflow_dispatch"}, "pull_request", "superseded=false"),
        ({"id": 99}, "pull_request", "superseded=false"),
        ({}, "push", "superseded=false"),
    ],
)
def test_superseded_action_counts_only_newer_runs_of_the_same_head(
    *, tmp_path: Path, newer: dict[str, Any], event: str, expected: str
) -> None:
    """Only a newer pull-request run from the same head repository supersedes."""
    run = {**_CURRENT, "id": 101, **newer}
    assert _run_action(tmp_path=tmp_path, runs=[run], event=event) == expected


def _gate_step(*, workflow: str, job: str) -> dict[str, Any]:
    jobs = yaml.safe_load((_REPO_ROOT / ".github/workflows" / workflow).read_text())
    (step,) = [s for s in jobs["jobs"][job]["steps"] if "test " in s.get("run", "")]
    return step


def _gate_passes(
    *, step: dict[str, Any], needs: dict[str, str], superseded: bool
) -> bool:
    if superseded and "failure" not in needs.values():
        return True
    body = re.sub(
        r"\$\{\{\s*needs\.([\w-]+)\.result\s*\}\}",
        lambda match: needs[match.group(1)],
        step["run"],
    )
    completed = subprocess.run(  # noqa: S603 - repository-owned gate script
        [_BASH, "-e", "-c", body], capture_output=True, check=False
    )
    return completed.returncode == 0


@pytest.mark.parametrize(("workflow", "job"), _GATES)
def test_gate_step_skips_only_on_supersession_without_failure(
    *, workflow: str, job: str
) -> None:
    """The gate assertion runs unless superseded, and always on a failed lane."""
    assert _gate_step(workflow=workflow, job=job)["if"] == _GATE_CONDITION


@pytest.mark.parametrize(("workflow", "job"), _GATES)
def test_gate_passes_iff_every_lane_succeeded_or_the_run_was_superseded(
    *, workflow: str, job: str
) -> None:
    """Failure never passes; cancelled or skipped passes only when superseded."""
    step = _gate_step(workflow=workflow, job=job)
    jobs = yaml.safe_load((_REPO_ROOT / ".github/workflows" / workflow).read_text())
    names = jobs["jobs"][job]["needs"]
    names = [names] if isinstance(names, str) else names
    states = ("success", "failure", "cancelled", "skipped")
    mismatches = []
    for pattern in itertools.product(states, repeat=min(2, len(names))):
        needs = dict.fromkeys(names, "success") | dict(
            zip(names, pattern, strict=False)
        )
        for superseded in (False, True):
            all_success = set(needs.values()) == {"success"}
            expected = all_success or (superseded and "failure" not in needs.values())
            if _gate_passes(step=step, needs=needs, superseded=superseded) != expected:
                mismatches.append((needs, superseded))
    assert mismatches == []
