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
_STUB_GH = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlencode, urlsplit

args = iter(sys.argv[1:])
assert next(args) == "api"
endpoint = None
query = None
fields = {}
method = "GET"
for arg in args:
    if arg in ("--method", "-X"):
        method = next(args)
    elif arg in ("--raw-field", "-f"):
        key, value = next(args).split("=", 1)
        fields[key] = value
    elif arg in ("--jq", "-q"):
        query = next(args)
    elif endpoint is None:
        endpoint = arg
    else:
        raise AssertionError(f"Unexpected argument: {arg}")
assert endpoint is not None and method == "GET"
if fields:
    endpoint += ("&" if "?" in endpoint else "?") + urlencode(fields)
url = urlsplit(endpoint)
root = Path(os.environ["STUB_DATA"])
if "/workflows/" in url.path:
    filters = parse_qs(url.query)
    workflow_id = int(url.path.split("/workflows/", 1)[1].split("/", 1)[0])
    data = json.loads((root / "runs.json").read_text())
    data["workflow_runs"] = [
        row for row in data["workflow_runs"]
        if row["workflow_id"] == workflow_id
        and ("branch" not in filters or row["head_branch"] == filters["branch"][0])
        and ("event" not in filters or row["event"] == filters["event"][0])
    ]
else:
    data = json.loads((root / "current.json").read_text())
if query is None:
    print(json.dumps(data))
else:
    result = subprocess.run(
        ["jq", "-r", query], input=json.dumps(data), text=True, check=False,
    )
    sys.exit(result.returncode)
"""


def _run_action(
    *,
    tmp_path: Path,
    runs: list[dict[str, Any]],
    event: str = "pull_request",
    current: dict[str, Any] | None = None,
) -> str:
    current = _CURRENT if current is None else current
    assert shutil.which("jq") is not None
    action = yaml.safe_load(
        (_REPO_ROOT / ".github/actions/superseded/action.yml").read_text()
    )
    (tmp_path / "current.json").write_text(json.dumps(current))
    (tmp_path / "runs.json").write_text(json.dumps({"workflow_runs": runs}))
    gh = tmp_path / "gh"
    gh.write_text(_STUB_GH)
    gh.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        STUB_DATA=str(tmp_path),
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_RUN_ID=str(current["id"]),
        GITHUB_EVENT_NAME=event,
        GITHUB_OUTPUT=str(tmp_path / "output"),
        BRANCH=str(current["head_branch"]),
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


@pytest.mark.parametrize(
    ("branch", "other_branch"),
    [
        ("fix#topic", "fix"),
        ("fix&topic", "fix"),
        ("fix%2Ftopic", "fix/topic"),
        ("fix+topic", "different"),
        ('fix"topic', "different"),
        ("feature/ümlaut", "different"),
    ],
)
@pytest.mark.parametrize("same_branch", [False, True])
def test_supersession_preserves_literal_branch_identity(
    *,
    tmp_path: Path,
    branch: str,
    other_branch: str,
    same_branch: bool,
) -> None:
    """A query metacharacter must neither alias nor lose the PR's branch."""
    current = {**_CURRENT, "head_branch": branch}
    newer = {
        **current,
        "id": 101,
        "head_branch": branch if same_branch else other_branch,
    }
    observed = _run_action(tmp_path=tmp_path, current=current, runs=[newer])
    assert observed == f"superseded={str(same_branch).lower()}"
