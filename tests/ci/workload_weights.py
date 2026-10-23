"""Keep every shard file's weight measured by CI, and fail CI while one is not.

The sharder places a file with no measured weight at its leg's median weight,
so new test files spread over the shards instead of all landing in one. That
placement is a stand-in for one CI run only, and two commands keep it so:

- `check` is the `workload-weights` CI gate. It fails while any file a leg
  shards over has no measured weight on that leg, naming the files. The shards
  themselves still run at median placement, so the same run measures them.
- `refresh` is the `ci-workload-weights` pre-commit hook. It replaces every
  leg's weights with the JUnit testcase seconds of the latest completed CPU run
  of the branch's open pull request, re-derives the shards, and stages
  `tests/ci/ci-workloads.json`. A file that run selected but that recorded no
  testcase on a leg is measured at 0 seconds there. A leg whose shard reports
  are incomplete keeps its weights.

Weights come only from CI, never from local timing. A new test file therefore
fails the gate once, on its first CI run, and the next commit fills its weight
in from that run. With no open pull request, no completed run, or `gh`
unavailable or offline, `refresh` leaves the file unchanged and prints why; it
never blocks a commit. Downloaded reports are cached per run under
`.cache/ci-junit/`, so the same run always gives the same file.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.ci import ci_workloads
from tests.ci.generate_ci_workloads import (
    MeasuredLeg,
    leg_weights_from_junit,
    refresh_leg,
    regenerate,
    render,
    unaccounted_test_files,
)

MANIFEST = Path("tests/ci/ci-workloads.json")
WORKFLOW = "cpu.yml"
CACHE = Path(".cache/ci-junit")
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
_DOWNLOAD_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class CompletedRun:
    """The latest completed CPU workflow run of the branch's open pull request."""

    run_id: int
    """The GitHub Actions run id."""
    attempt: int
    """The run attempt the reports belong to."""
    head_sha: str
    """The commit the run tested."""


@dataclass(frozen=True)
class Skip:
    """No run to refresh from, and why."""

    reason: str
    """One line naming what was missing."""


@dataclass(frozen=True)
class Refresh:
    """The manifest text after a refresh, and what to print."""

    text: str
    """The rendered manifest; the input text unchanged when nothing refreshed."""
    message: str
    """The lines the hook prints."""


def unmeasured_files(
    *,
    leg_universes: Mapping[str, Iterable[str]],
    leg_weights: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[str, ...]]:
    """Return, per leg, the files it shards over that have no measured weight."""
    missing = {
        leg: tuple(sorted(set(files) - set(leg_weights.get(leg, {}))))
        for leg, files in sorted(leg_universes.items())
    }
    return {leg: files for leg, files in missing.items() if files}


def gate_message(missing: Mapping[str, Sequence[str]]) -> str:
    """Name the unmeasured files and what fills their weights in."""
    count = sum(len(files) for files in missing.values())
    return "\n".join(
        (
            (
                f"{count} test file(s) have no measured weight in "
                f"{MANIFEST.as_posix()}; the shards placed them at their leg's "
                "median weight:"
            ),
            *(f"  {leg}: {file_}" for leg, files in missing.items() for file_ in files),
            "",
            (
                "This run measures them. On the next commit the "
                "ci-workload-weights pre-commit hook fills their weights in from "
                "this run."
            ),
        )
    )


def refresh_manifest(
    *,
    manifest_text: str,
    repo_root: Path,
    find_run: Callable[[], CompletedRun | Skip],
    read_reports: Callable[[CompletedRun], Mapping[str, Sequence[Path]] | Skip],
    selected_at_run: Callable[[CompletedRun], Mapping[str, frozenset[str]] | Skip],
) -> Refresh:
    """Replace every complete leg's weights with one CI run's measurements.

    `read_reports` maps a leg to the JUnit reports of its shards on the run;
    `selected_at_run` maps it to the files the run's own manifest had it shard
    over. A leg is refreshed only when it has one report per shard.
    """
    run = find_run()
    if isinstance(run, Skip):
        return Refresh(text=manifest_text, message=_unchanged(run.reason))
    reports = read_reports(run)
    if isinstance(reports, Skip):
        return Refresh(text=manifest_text, message=_unchanged(reports.reason))
    selected = selected_at_run(run)
    if isinstance(selected, Skip):
        return Refresh(text=manifest_text, message=_unchanged(selected.reason))

    manifest: ci_workloads.WorkloadManifest = json.loads(manifest_text)
    shards = {
        leg: config["shards"]
        for family in manifest["shard_layout"].values()
        for leg, config in family.items()
    }
    refreshed: list[str] = []
    kept: list[str] = []
    for leg in sorted(manifest["leg_weights"]):
        leg_reports = sorted(reports.get(leg, ()))
        if len(leg_reports) != shards[leg]:
            kept.append(
                f"  {leg}: {len(leg_reports)} of {shards[leg]} shard reports, "
                "weights kept"
            )
            continue
        manifest = refresh_leg(
            manifest=manifest,
            leg=leg,
            measured=_measure_leg(
                repo_root=repo_root,
                reports=leg_reports,
                selected=selected.get(leg, frozenset()),
            ),
            junit_source=(
                f"run {run.run_id} attempt {run.attempt} (head {run.head_sha}); "
                f"artifacts {LEG_ARTIFACTS[leg]}, one per shard; per file the "
                "summed JUnit testcase seconds of that run, 0 for a file the leg "
                "selected that recorded no testcase"
            ),
        )
        refreshed.append(leg)
    if not refreshed:
        return Refresh(
            text=manifest_text,
            message=_unchanged(f"run {run.run_id} has no complete leg", *kept),
        )
    manifest = regenerate(
        repo_root=repo_root,
        manifest=manifest,
        additions=unaccounted_test_files(repo_root=repo_root, manifest=manifest),
    )
    return Refresh(
        text=render(manifest),
        message="\n".join(
            (
                f"workload weights from run {run.run_id}: {', '.join(refreshed)}",
                *kept,
            )
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run `check` (the CI gate) or `refresh` (the pre-commit hook)."""
    command = (sys.argv[1:] if argv is None else list(argv))[:1]
    if command == ["check"]:
        return _check()
    if command == ["refresh"]:
        return _refresh(repo_root=Path.cwd())
    sys.stdout.write("usage: python -m tests.ci.workload_weights check|refresh\n")
    return 2


def _check() -> int:
    manifest = ci_workloads.load_manifest()
    universes = _general_universes(manifest=manifest) | _solution_universes(
        manifest=manifest
    )
    missing = unmeasured_files(
        leg_universes=universes,
        leg_weights={leg: ci_workloads.leg_weights(leg=leg) for leg in universes},
    )
    if missing:
        sys.stdout.write(gate_message(missing) + "\n")
        return 1
    sys.stdout.write(f"every shard file has a measured weight in {MANIFEST}\n")
    return 0


def _refresh(*, repo_root: Path) -> int:
    path = repo_root / MANIFEST
    text = path.read_text(encoding="utf-8")
    refreshed = refresh_manifest(
        manifest_text=text,
        repo_root=repo_root,
        find_run=_find_run,
        read_reports=lambda run: _read_reports(run=run, repo_root=repo_root),
        selected_at_run=_selected_at_run,
    )
    if refreshed.text != text:
        path.write_text(refreshed.text, encoding="utf-8")
        _run_text("git", "add", MANIFEST.as_posix())
    sys.stdout.write(refreshed.message + "\n")
    return 0


def _measure_leg(
    *, repo_root: Path, reports: Sequence[Path], selected: frozenset[str]
) -> MeasuredLeg:
    measured = leg_weights_from_junit(
        repo_root=repo_root, junit_paths=reports, drop_deleted_files=True
    )
    ran_nothing = {
        file_
        for file_ in selected - set(measured.seconds)
        if (repo_root / file_).is_file()
    }
    seconds = {**dict.fromkeys(ran_nothing, 0.0), **measured.seconds}
    counts = {**dict.fromkeys(ran_nothing, 0), **measured.counts}
    return MeasuredLeg(
        seconds={file_: seconds[file_] for file_ in sorted(seconds)},
        counts={file_: counts[file_] for file_ in sorted(counts)},
    )


def _general_universes(
    *, manifest: ci_workloads.WorkloadManifest
) -> dict[str, frozenset[str]]:
    universe = frozenset(manifest["general_shard_universe"])
    linux_only = frozenset(manifest.get("linux_only_files", ()))
    return {
        leg: universe if leg.endswith("-linux") else universe - linux_only
        for leg in manifest["shard_layout"]["general"]
    }


def _solution_universes(
    *, manifest: ci_workloads.WorkloadManifest
) -> dict[str, frozenset[str]]:
    return {
        leg: frozenset(
            file_
            for invocation in manifest["invocations"]
            if invocation["id"].startswith("solution-shard-")
            and invocation["id"].endswith(f"-{leg.removesuffix('-solution')}-linux")
            for file_ in invocation["files"]
        )
        for leg in manifest["shard_layout"]["solution"]
    }


def _selected_at_run(run: CompletedRun) -> Mapping[str, frozenset[str]] | Skip:
    text = _run_text("git", "show", f"{run.head_sha}:{MANIFEST.as_posix()}")
    if isinstance(text, Skip):
        return Skip(reason=f"head {run.head_sha} of run {run.run_id} is not local")
    manifest: ci_workloads.WorkloadManifest = json.loads(text)
    return _general_universes(manifest=manifest) | _solution_universes(
        manifest=manifest
    )


def _find_run() -> CompletedRun | Skip:
    branch = _run_text("git", "rev-parse", "--abbrev-ref", "HEAD")
    if isinstance(branch, Skip) or branch == "HEAD":
        return Skip(reason="detached HEAD has no pull request")
    pr = _run_text(
        "gh", "pr", "list", "--head", branch, "--state", "open", "--limit", "1",
        "--json", "number", "--jq", ".[0].number // empty",
    )  # fmt: skip
    if isinstance(pr, Skip):
        return pr
    if not pr:
        return Skip(reason=f"no open pull request for branch {branch}")
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


def _read_reports(
    *, run: CompletedRun, repo_root: Path
) -> Mapping[str, Sequence[Path]] | Skip:
    cache = repo_root / CACHE / f"{run.run_id}-{run.attempt}"
    if not cache.is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=cache.parent))
        patterns = [arg for p in LEG_ARTIFACTS.values() for arg in ("-p", p)]
        downloaded = _run_text(
            "gh", "run", "download", str(run.run_id), *patterns, "-D", str(staging),
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        )  # fmt: skip
        if isinstance(downloaded, Skip):
            shutil.rmtree(staging, ignore_errors=True)
            return downloaded
        staging.rename(cache)
    return {
        leg: tuple(sorted(cache.glob(f"{pattern}/*.xml")))
        for leg, pattern in LEG_ARTIFACTS.items()
    }


def _unchanged(reason: str, *details: str) -> str:
    return "\n".join((f"workload weights unchanged: {reason}", *details))


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
