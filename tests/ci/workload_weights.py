"""Keep every shard file's weight measured by CI, and fail CI while one is not.

The sharder places a file with no measured weight at its leg's median weight,
so new test files spread over the shards instead of all landing in one. That
placement is a stand-in for one CI run only, and two commands keep it so:

- `check` is the `workload-weights` CI gate. It fails while any file a leg
  shards over has no measured weight on that leg, naming the files. The shards
  themselves still run at median placement, so the same run measures them.
- `refresh` is the `ci-workload-weights` pre-commit hook. It fills in only
  the weights `check` names as missing and leaves every existing weight as it
  is. Each missing file takes its testcase seconds from the latest completed
  CPU run of the branch's open pull request that measured it; the run's leg
  counts when it has a report for every shard of the run's own layout, since
  per-file weights do not depend on the shard count. A file the run's manifest
  gave to the leg that recorded no testcase there is measured at 0 seconds; a
  file the run did not contain stays missing. Each filled weight names its run
  under the leg's `filled_from`. The hook then re-derives the shards and stages
  `tests/ci/ci-workloads.json`. With nothing missing it looks nothing up.

Re-measuring a whole leg is the manual
`generate_ci_workloads --refresh-leg` step, which replaces the leg's weights
and clears its `filled_from`.

Weights come only from CI, never from local timing. A new test file therefore
fails the gate once, on its first CI run, and the next commit fills its weight
in from that run. With no open pull request, no completed run, or `gh`
unavailable or offline, `refresh` leaves the file unchanged and prints why; it
never fails a commit. Downloaded reports are cached per run under
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
    leg_weights_from_junit,
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
_DOWNLOAD_TIMEOUT_SECONDS = 300
_RUNS_CONSIDERED = 5


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
    find_runs: Callable[[], Sequence[CompletedRun] | Skip],
    read_reports: Callable[[CompletedRun], Mapping[str, Sequence[Path]] | Skip],
    run_manifest: Callable[[CompletedRun], ci_workloads.WorkloadManifest | Skip],
) -> Refresh:
    """Fill in the weights `check` names as missing, from the latest run measuring each.

    Every weight the manifest already has is left exactly as it is; re-measuring
    a whole leg is the manual `generate_ci_workloads --refresh-leg` step. With
    nothing missing, no run is looked up.

    `find_runs` returns completed runs newest first. For each, `read_reports`
    maps a leg to the JUnit reports of its shards and `run_manifest` returns the
    workload manifest the run ran under. A leg of a run is used only when it has
    one report per shard of *that run's* layout: per-file weights do not depend
    on the shard count. A missing file is then filled with its summed testcase
    seconds, or with 0 when the run's manifest gave it to the leg and it
    recorded no testcase there. A file the run did not contain stays missing.
    """
    manifest: ci_workloads.WorkloadManifest = json.loads(manifest_text)
    missing = unmeasured_files(
        leg_universes=_shard_universes(manifest=manifest),
        leg_weights={
            leg: entry["seconds"] for leg, entry in manifest["leg_weights"].items()
        },
    )
    if not missing:
        return Refresh(text=manifest_text, message="no shard file lacks a weight")
    runs = find_runs()
    if isinstance(runs, Skip):
        return Refresh(
            text=manifest_text,
            message=_unchanged(reason=runs.reason, missing=missing),
        )

    filled: list[str] = []
    notes: list[str] = []
    for run in runs:
        if not missing:
            break
        found = _fill_from_run(
            manifest=manifest,
            missing=missing,
            run=run,
            repo_root=repo_root,
            read_reports=read_reports,
            run_manifest=run_manifest,
        )
        if isinstance(found, Skip):
            notes.append(f"  run {run.run_id}: {found.reason}")
            continue
        filled += found
        missing = unmeasured_files(
            leg_universes=missing,
            leg_weights={
                leg: entry["seconds"] for leg, entry in manifest["leg_weights"].items()
            },
        )
    if not filled:
        return Refresh(
            text=manifest_text,
            message=_unchanged(
                reason="no completed run measured them",
                missing=missing,
                details=notes,
            ),
        )
    manifest = regenerate(
        repo_root=repo_root,
        manifest=manifest,
        additions=unaccounted_test_files(repo_root=repo_root, manifest=manifest),
    )
    still = [f"  still missing, {leg}: {f}" for leg, fs in missing.items() for f in fs]
    return Refresh(
        text=render(manifest),
        message="\n".join(("filled workload weights:", *filled, *still, *notes)),
    )


def _fill_from_run(
    *,
    manifest: ci_workloads.WorkloadManifest,
    missing: Mapping[str, Sequence[str]],
    run: CompletedRun,
    repo_root: Path,
    read_reports: Callable[[CompletedRun], Mapping[str, Sequence[Path]] | Skip],
    run_manifest: Callable[[CompletedRun], ci_workloads.WorkloadManifest | Skip],
) -> list[str] | Skip:
    """Write into `manifest` the missing weights one run measured; name them."""
    ran_under = run_manifest(run)
    if isinstance(ran_under, Skip):
        return ran_under
    reports = read_reports(run)
    if isinstance(reports, Skip):
        return reports
    selected = _shard_universes(manifest=ran_under)
    shards = {
        leg: config["shards"]
        for family in ran_under["shard_layout"].values()
        for leg, config in family.items()
    }
    source = f"run {run.run_id} attempt {run.attempt} (head {run.head_sha})"
    filled: list[str] = []
    for leg, files in missing.items():
        leg_reports = sorted(reports.get(leg, ()))
        if len(leg_reports) != shards.get(leg):
            continue
        measured = leg_weights_from_junit(
            repo_root=repo_root, junit_paths=leg_reports, drop_deleted_files=True
        )
        entry = manifest["leg_weights"][leg]
        for file_ in files:
            if file_ in measured.seconds:
                seconds, count = measured.seconds[file_], measured.counts[file_]
            elif file_ in selected.get(leg, ()):
                seconds, count = 0.0, 0
            else:
                continue
            entry["seconds"] = _with(
                mapping=entry["seconds"], file_=file_, value=seconds
            )
            entry["counts"] = _with(mapping=entry["counts"], file_=file_, value=count)
            entry["filled_from"] = _with(
                mapping=entry.get("filled_from", {}), file_=file_, value=source
            )
            filled.append(f"  {leg}: {file_} = {seconds} s from {source}")
    return filled


def _with[Value](
    *, mapping: Mapping[str, Value], file_: str, value: Value
) -> dict[str, Value]:
    """Return `mapping` with `file_` set to `value`, keys in sorted order."""
    updated = {**mapping, file_: value}
    return {key: updated[key] for key in sorted(updated)}


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
    universes = _shard_universes(manifest=manifest)
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
        find_runs=_find_runs,
        read_reports=lambda run: _read_reports(run=run, repo_root=repo_root),
        run_manifest=_run_manifest,
    )
    if refreshed.text != text:
        path.write_text(refreshed.text, encoding="utf-8")
        _run_text("git", "add", MANIFEST.as_posix())
    sys.stdout.write(refreshed.message + "\n")
    return 0


def _shard_universes(
    *, manifest: ci_workloads.WorkloadManifest
) -> dict[str, frozenset[str]]:
    """Return, per leg, the files that leg shards over under `manifest`."""
    return _general_universes(manifest=manifest) | _solution_universes(
        manifest=manifest
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


def _run_manifest(run: CompletedRun) -> ci_workloads.WorkloadManifest | Skip:
    text = _run_text("git", "show", f"{run.head_sha}:{MANIFEST.as_posix()}")
    if isinstance(text, Skip):
        return Skip(reason=f"head {run.head_sha} of run {run.run_id} is not local")
    return json.loads(text)


def _find_runs() -> Sequence[CompletedRun] | Skip:
    """Return the open pull request's latest completed CPU runs, newest first."""
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
    listed = _run_text(
        "gh", "run", "list", "--workflow", WORKFLOW, "--branch", branch,
        "--status", "completed", "--limit", str(_RUNS_CONSIDERED), "--json",
        "databaseId,attempt,headSha",
        "--jq", '.[] | "\\(.databaseId) \\(.attempt) \\(.headSha)"',
    )  # fmt: skip
    if isinstance(listed, Skip):
        return listed
    if not listed:
        return Skip(reason=f"pull request {pr} has no completed {WORKFLOW} run")
    return tuple(
        CompletedRun(run_id=int(run_id), attempt=int(attempt), head_sha=head_sha)
        for run_id, attempt, head_sha in (line.split() for line in listed.splitlines())
    )


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


def _unchanged(
    *,
    reason: str,
    missing: Mapping[str, Sequence[str]],
    details: Sequence[str] = (),
) -> str:
    return "\n".join(
        (
            f"workload weights unchanged: {reason}",
            *(
                f"  missing, {leg}: {file_}"
                for leg, files in missing.items()
                for file_ in files
            ),
            *details,
        )
    )


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
