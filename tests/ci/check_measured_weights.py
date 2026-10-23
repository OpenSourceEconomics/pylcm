"""Refuse a push while a test file the last CI run measured still has no weight.

The sharder places a file with no measured weight at the leg's median weight, so
new test files spread over the shards instead of all landing in one. That
placement stands in for a measurement nobody has taken yet. Once the branch's
open pull request has a completed CPU run, the run *has* measured every file it
executed, and the stand-in is no longer acceptable for those files.

This pre-push check reads the JUnit reports of that run, per leg, and compares
the files they cover with the leg's weights in `tests/ci/ci-workloads.json`:

- a file the run measured that the manifest does not weigh ⇒ refused, with the
  commands that refresh that leg from the run;
- a file added after the run, or deleted since ⇒ allowed;
- no open pull request, no completed run, or `gh` unavailable or offline ⇒
  allowed, with one line saying why. Lack of network never blocks a push.

Set `LCM_CI_WEIGHTS_OVERRIDE="<reason>"` to push anyway; the reason is printed,
so it lands in the push output instead of disappearing behind `--no-verify`.
"""

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.ci import ci_workloads

OVERRIDE_VARIABLE = "LCM_CI_WEIGHTS_OVERRIDE"
WORKFLOW = "cpu.yml"
# The JUnit artifact name pattern of every shard of each weighted leg.
LEG_ARTIFACTS: Mapping[str, str] = {
    "fp64-linux": "junit-cpu-fp64-Linux-general-*",
    "fp64-macos": "junit-cpu-fp64-macOS-general-*",
    "fp64-windows": "junit-cpu-fp64-Windows-general-*",
    "fp32-linux": "junit-cpu-fp32-general-*",
    "fp64-solution": "junit-cpu-fp64-solution-*",
    "fp32-solution": "junit-cpu-fp32-solution-*",
}

_GH_TIMEOUT_SECONDS = 60
_DOWNLOAD_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class CompletedRun:
    """The latest completed CPU workflow run of the branch's open pull request."""

    run_id: int
    """The GitHub Actions run id."""
    attempt: int
    """The run attempt whose artifacts are read."""
    head_sha: str
    """The commit the run tested."""


@dataclass(frozen=True)
class Skip:
    """No measurement to compare against, and why."""

    reason: str
    """One line naming what was missing."""


@dataclass(frozen=True)
class Verdict:
    """Whether the push may proceed, and what to print."""

    allowed: bool
    """False only when a measured file has no weight and no override is set."""
    message: str
    """The text the hook prints."""


def check_measured_weights(
    *,
    find_run: Callable[[], CompletedRun | Skip],
    read_measured: Callable[[CompletedRun], Mapping[str, frozenset[str]] | Skip],
    leg_weights: Mapping[str, Mapping[str, float]],
    present: frozenset[str],
    override: str | None,
) -> Verdict:
    """Decide whether the manifest weighs every file the last CI run measured.

    `read_measured` maps each leg to the test files its JUnit reports cover.
    Only files still in `present` are held to a weight.
    """
    if override:
        return Verdict(
            allowed=True,
            message=f"{OVERRIDE_VARIABLE} set, weights not checked: {override}",
        )
    run = find_run()
    if isinstance(run, Skip):
        return Verdict(allowed=True, message=f"weights not checked: {run.reason}")
    measured = read_measured(run)
    if isinstance(measured, Skip):
        return Verdict(allowed=True, message=f"weights not checked: {measured.reason}")

    missing = {
        leg: sorted((files & present) - set(leg_weights.get(leg, {})))
        for leg, files in sorted(measured.items())
    }
    missing = {leg: files for leg, files in missing.items() if files}
    if not missing:
        return Verdict(
            allowed=True,
            message=(
                f"every test file run {run.run_id} measured has a weight in "
                "tests/ci/ci-workloads.json"
            ),
        )
    return Verdict(allowed=False, message=_refusal(run=run, missing=missing))


def main() -> int:
    """Run the check against the working tree and the branch's CI run."""
    repo_root = Path.cwd()
    manifest = ci_workloads.load_manifest(path=repo_root / "tests/ci/ci-workloads.json")
    verdict = check_measured_weights(
        find_run=_find_run,
        read_measured=lambda run: _read_measured(run=run, repo_root=repo_root),
        leg_weights={
            leg: entry["seconds"] for leg, entry in manifest["leg_weights"].items()
        },
        present=frozenset(
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "tests").rglob("test_*.py")
        ),
        override=os.environ.get(OVERRIDE_VARIABLE, "").strip() or None,
    )
    sys.stdout.write(verdict.message + "\n")
    return 0 if verdict.allowed else 1


def _refusal(*, run: CompletedRun, missing: Mapping[str, list[str]]) -> str:
    lines = [
        (
            f"CI run {run.run_id} (head {run.head_sha}) measured these test "
            "files, but tests/ci/ci-workloads.json has no weight for them:"
        ),
        *(f"  {leg}: {file_}" for leg, files in missing.items() for file_ in files),
        "",
        "Refresh those legs from that run, then commit the manifest:",
        "  JUNIT_DIR=$(mktemp -d)",
    ]
    for leg in missing:
        pattern = LEG_ARTIFACTS[leg]
        lines += [
            f"  gh run download {run.run_id} -p '{pattern}' -D \"$JUNIT_DIR\"",
            (
                "  pixi run -e tests-cpu python -m tests.ci.generate_ci_workloads "
                f'--refresh-leg {leg} --junit "$JUNIT_DIR/{pattern}/*.xml" '
                f"--junit-source 'run {run.run_id} attempt {run.attempt} "
                f"(head {run.head_sha}); artifacts {pattern}; per file the "
                "summed JUnit testcase seconds of that run'"
            ),
        ]
    lines += ["", f'To push anyway, set {OVERRIDE_VARIABLE}="<reason>".']
    return "\n".join(lines)


def _find_run() -> CompletedRun | Skip:
    branch = _current_branch()
    if isinstance(branch, Skip):
        return branch
    pr = _open_pull_request(branch=branch)
    if isinstance(pr, Skip):
        return pr
    run = _run_text(
        "gh", "run", "list", "--workflow", WORKFLOW, "--branch", branch,
        "--status", "completed", "--limit", "1", "--json",
        "databaseId,attempt,headSha",
        "--jq", '.[0] // empty | "\\(.databaseId) \\(.attempt) \\(.headSha)"',
    )  # fmt: skip
    if isinstance(run, Skip):
        return run
    if not run:
        return Skip(reason=f"pull request {pr} has no completed {WORKFLOW} run")
    run_id, attempt, head_sha = run.split()
    return CompletedRun(run_id=int(run_id), attempt=int(attempt), head_sha=head_sha)


def _current_branch() -> str | Skip:
    branch = _run_text("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        return Skip(reason="detached HEAD has no pull request")
    return branch


def _open_pull_request(*, branch: str) -> str | Skip:
    pr = _run_text(
        "gh", "pr", "list", "--head", branch, "--state", "open", "--limit", "1",
        "--json", "number", "--jq", ".[0].number // empty",
    )  # fmt: skip
    if pr == "":
        return Skip(reason=f"no open pull request for branch {branch}")
    return pr


def _read_measured(
    *, run: CompletedRun, repo_root: Path
) -> Mapping[str, frozenset[str]] | Skip:
    with tempfile.TemporaryDirectory() as directory:
        patterns = [arg for p in LEG_ARTIFACTS.values() for arg in ("-p", p)]
        downloaded = _run_text(
            "gh", "run", "download", str(run.run_id), *patterns, "-D", directory,
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        )  # fmt: skip
        if isinstance(downloaded, Skip):
            return downloaded
        return {
            leg: frozenset(
                file_
                for report in sorted(Path(directory).glob(f"{pattern}/*.xml"))
                for file_ in _files_in_report(report=report, repo_root=repo_root)
            )
            for leg, pattern in LEG_ARTIFACTS.items()
        }


def _files_in_report(*, report: Path, repo_root: Path) -> frozenset[str]:
    """Return the existing test files a JUnit report has testcases for."""
    root = ET.parse(report).getroot()  # noqa: S314 - CI-produced JUnit artifacts
    files: set[str] = set()
    for classname in {case.get("classname", "") for case in root.iter("testcase")}:
        parts = classname.split(".")
        for end in range(len(parts), 0, -1):
            candidate = "/".join(parts[:end]) + ".py"
            if (repo_root / candidate).is_file():
                files.add(candidate)
                break
    return frozenset(files)


def _run_text(*args: str, timeout: int = _GH_TIMEOUT_SECONDS) -> str | Skip:
    """Return a command's stripped stdout, or a `Skip` naming why it failed."""
    try:
        # S603: every argument is a constant of this module or a value `git`
        # and `gh` themselves returned; nothing comes from user input.
        result = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Skip(reason=f"`{args[0]}` unavailable: {error}")
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["no message"]
        return Skip(reason=f"`{' '.join(args[:3])}` failed: {detail[0]}")
    return result.stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
