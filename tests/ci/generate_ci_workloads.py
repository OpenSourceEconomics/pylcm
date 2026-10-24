"""Register new test files in `ci-workloads.json` and re-derive the shard split.

Adding a test file under `tests/` without recording it here fails
`tests/ci/test_ci_workloads_manifest.py` in CI and nowhere earlier: the file is
selected by no invocation, so it runs on no lane. This tool is the deliberate,
reviewed step that the manifest's own module docstring points at.

What it changes, and nothing else:

- every file under `tests/` that no invocation selects and that the exclusion
  list does not name is added to `general_shard_universe` and to
  `unweighted_files` — never to `file_weights`, because a file with no observed
  weight is not a file that costs nothing;
- every file some invocation selects that has neither a weight nor an
  unweighted entry is added to `unweighted_files`;
- the `files` list of every `general-shard-<n>-<leg>` invocation is recomputed
  from that universe -- without `linux_only_files` on the macOS and Windows
  legs -- with the manifest's own weighted partition, so the recorded split is
  the one CI computes at run time, and its `predicted_payload_minutes` is
  recomputed from the same partition;
- the `files` list of every `solution-shard-<n>-fp<p>-linux` invocation is
  recomputed the way `cpu.yml` selects it at run time
  (`shard_test_files tests/solution --weights-leg fp<p>-solution`): a new
  solution file joins one shard, and the weighted partition may move others.
  Its `predicted_payload_minutes` is recomputed from the same partition.

With `--refresh-leg <leg> --junit <glob> --junit-source <text>` it first
replaces that one leg's `leg_weights` with the summed JUnit testcase seconds and
testcase counts of the given reports -- the shards of that leg on one run -- and
records `--junit-source` as their provenance.

What it leaves alone: `frozen_head`, the root `file_weights`, every other leg's
observed weights, the exclusion list, `linux_only_files`, the guardrail budgets,
the shard *counts*, and every other lane.
`frozen_head` binds the root cross-leg CSV weights; per-leg `junit_source`
records each leg's measurement provenance. Registering files or refreshing one
leg does not advance the root freeze. Re-measuring the root CSV weights does.

Exit codes:

- `0` ⇒ the manifest already accounts for every test file (or, without
  `--check`, it was rewritten to)
- `1` ⇒ under `--check`, regeneration would change the committed manifest
"""

import argparse
import glob
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.ci import ci_workloads
from tests.ci.shard_test_files import (
    general_shard_files,
    weighted_shard_report,
)

MANIFEST_PATH = Path(__file__).with_name("ci-workloads.json")
_INDENT = 1


def render[ManifestValue](manifest: Mapping[str, ManifestValue]) -> str:
    """Render the manifest exactly as the committed file is written.

    The committed file round-trips through `indent=1, ensure_ascii=False`, so a
    regeneration that changes nothing produces a byte-identical file and
    `--check` reports drift only when the content really moved.
    """
    return json.dumps(dict(manifest), indent=_INDENT, ensure_ascii=False) + "\n"


def unaccounted_test_files(
    *, repo_root: Path, manifest: ci_workloads.WorkloadManifest
) -> tuple[str, ...]:
    """Return test files that no invocation selects and no exclusion names."""
    present = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "tests").rglob("test_*.py")
    }
    selected = {file_ for inv in manifest["invocations"] for file_ in inv["files"]}
    excluded = set(manifest["excluded_files"])
    return tuple(sorted(present - selected - excluded))


@dataclass(frozen=True)
class MeasuredLeg:
    """Per-file JUnit testcase seconds and testcase counts for one leg."""

    seconds: Mapping[str, float]
    """Summed testcase seconds per repository-relative test file."""
    counts: Mapping[str, int]
    """Number of testcases per repository-relative test file."""


def leg_weights_from_junit(
    *, repo_root: Path, junit_paths: Sequence[Path], drop_deleted_files: bool = False
) -> MeasuredLeg:
    """Sum each file's JUnit testcase seconds over the reports of one run.

    The reports are the shards of one leg on one run, so every file appears in
    exactly one of them. A testcase keeps its recorded seconds whatever its
    outcome; a file with no testcase gets no entry. A classname that resolves
    to no file under `repo_root` is refused rather than dropped, unless
    `drop_deleted_files` says the reports may predate a file's deletion.
    """
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for junit_path in junit_paths:
        root = ET.parse(junit_path).getroot()  # noqa: S314 - CI-produced JUnit artifacts
        for case in root.iter("testcase"):
            try:
                file_ = _test_file_of(
                    classname=case.get("classname", ""), root=repo_root
                )
            except ValueError:
                if drop_deleted_files:
                    continue
                raise
            seconds[file_] = seconds.get(file_, 0.0) + float(case.get("time", "0"))
            counts[file_] = counts.get(file_, 0) + 1
    return MeasuredLeg(
        seconds={file_: round(seconds[file_], 3) for file_ in sorted(seconds)},
        counts={file_: counts[file_] for file_ in sorted(counts)},
    )


def refresh_leg(
    *,
    manifest: ci_workloads.WorkloadManifest,
    leg: str,
    measured: MeasuredLeg,
    junit_source: ci_workloads.SourcePath,
) -> ci_workloads.WorkloadManifest:
    """Replace one leg's observed weights and their provenance, nothing else.

    `frozen_head`, the root `file_weights` and every other leg stay as they
    are: re-measuring one leg does not advance the root freeze.
    """
    updated: ci_workloads.WorkloadManifest = json.loads(json.dumps(dict(manifest)))
    if leg not in updated["leg_weights"]:
        legs = sorted(updated["leg_weights"])
        raise KeyError(f"{leg!r} is not a recorded leg: {legs}")
    entry = updated["leg_weights"][leg]
    entry["junit_source"] = junit_source
    entry["seconds"] = dict(measured.seconds)
    entry["counts"] = dict(measured.counts)
    # A full re-measurement supersedes every weight filled in one file at a time.
    entry.pop("filled_from", None)
    return updated


def _test_file_of(*, classname: str, root: Path) -> str:
    """Return the test file a JUnit classname names, dropping any class suffix."""
    parts = classname.split(".")
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if (root / candidate).is_file():
            return candidate
    raise ValueError(f"JUnit classname {classname!r} names no test file under {root}")


def regenerate(
    *,
    repo_root: Path,
    manifest: ci_workloads.WorkloadManifest,
    additions: Sequence[str],
) -> ci_workloads.WorkloadManifest:
    """Return the manifest with `additions` registered and the shards re-derived."""
    updated: ci_workloads.WorkloadManifest = json.loads(json.dumps(dict(manifest)))
    weighted = set(updated["file_weights"])
    for key in ("general_shard_universe", "unweighted_files"):
        if key == "unweighted_files":
            new = [name for name in additions if name not in weighted]
        else:
            new = list(additions)
        updated[key] = sorted({*updated[key], *new})

    universe = tuple(updated["general_shard_universe"])
    by_id = {inv["id"]: inv for inv in updated["invocations"]}
    for leg, config in updated["shard_layout"]["general"].items():
        weights = {
            file_: float(seconds)
            for file_, seconds in updated["leg_weights"][leg]["seconds"].items()
        }
        linux_only = set(updated.get("linux_only_files", ()))
        leg_universe = (
            universe
            if leg.endswith("-linux")
            else tuple(file_ for file_ in universe if file_ not in linux_only)
        )
        reports = weighted_shard_report(
            files=leg_universe, n_shards=config["shards"], weights=weights
        )
        for index, report in enumerate(reports, start=1):
            group = report.files
            invocation_id = f"general-shard-{index}-{leg}"
            if invocation_id not in by_id:
                raise KeyError(f"{invocation_id} is not a recorded invocation")
            by_id[invocation_id]["files"] = sorted(group)
            by_id[invocation_id]["predicted_payload_minutes"] = round(
                report.payload_minutes(workers=config["workers"]), 2
            )
    solution_files = tuple(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "tests/solution").rglob("test_*.py")
    )
    for leg, config in updated["shard_layout"]["solution"].items():
        weights = {
            file_: float(seconds)
            for file_, seconds in updated["leg_weights"][leg]["seconds"].items()
        }
        reports = weighted_shard_report(
            files=solution_files, n_shards=config["shards"], weights=weights
        )
        precision = leg.removesuffix("-solution")
        for index, report in enumerate(reports, start=1):
            invocation_id = f"solution-shard-{index}-{precision}-linux"
            if invocation_id not in by_id:
                raise KeyError(f"{invocation_id} is not a recorded invocation")
            by_id[invocation_id]["files"] = sorted(report.files)
            by_id[invocation_id]["predicted_payload_minutes"] = round(
                report.payload_minutes(workers=config["workers"]), 2
            )
    # A file a lane selects by hand still needs a weight or an unweighted entry.
    selected = {file_ for inv in updated["invocations"] for file_ in inv["files"]}
    updated["unweighted_files"] = sorted(
        {*updated["unweighted_files"], *(selected - weighted)}
    )
    _fail_if_missing_paths(repo_root=repo_root, universe=universe)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    """Register unaccounted test files, or report the drift under `--check`."""
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = repo_root / "tests/ci/ci-workloads.json"
    committed_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(committed_text)

    if args.refresh_leg is not None:
        reports = tuple(
            Path(path)
            for pattern in args.junit
            for path in sorted(glob.glob(pattern))  # noqa: PTH207 - absolute patterns
        )
        if not reports or not args.junit_source:
            sys.stdout.write(
                "--refresh-leg needs --junit matching at least one report and "
                "--junit-source naming the run\n"
            )
            return 1
        manifest = refresh_leg(
            manifest=manifest,
            leg=args.refresh_leg,
            measured=leg_weights_from_junit(repo_root=repo_root, junit_paths=reports),
            junit_source=args.junit_source,
        )
    additions = tuple(args.add) or unaccounted_test_files(
        repo_root=repo_root, manifest=manifest
    )
    missing = sorted(name for name in additions if not (repo_root / name).is_file())
    if missing:
        sys.stdout.write(f"no such test file under {repo_root}: {missing}\n")
        return 1

    updated_text = render(
        regenerate(repo_root=repo_root, manifest=manifest, additions=additions)
    )
    if updated_text == committed_text:
        sys.stdout.write("ci-workloads.json already accounts for every test file\n")
        return 0

    for name in additions:
        sys.stdout.write(f"registering {name}\n")
    if args.check:
        sys.stdout.write(
            "\nThe committed manifest differs from regeneration. Run this tool "
            "without --check and stage tests/ci/ci-workloads.json.\n"
        )
        return 1

    manifest_path.write_text(updated_text, encoding="utf-8")
    sys.stdout.write(f"rewrote {manifest_path.relative_to(repo_root)}\n")
    _fail_if_sharder_disagrees(manifest_path=manifest_path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: derived from this script's location)",
    )
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "repository-relative test file to register; repeat once per file. "
            "Omitted, every unaccounted test file under tests/ is registered."
        ),
    )
    parser.add_argument(
        "--refresh-leg",
        default=None,
        metavar="LEG",
        help=(
            "re-measure this leg's `leg_weights` from --junit reports of one run, "
            "recording --junit-source as their provenance"
        ),
    )
    parser.add_argument(
        "--junit",
        action="append",
        default=[],
        metavar="GLOB",
        help="JUnit report glob for --refresh-leg; repeat once per pattern",
    )
    parser.add_argument(
        "--junit-source",
        default=None,
        help="the run, attempt, head and artifacts the --junit reports come from",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the drift and exit non-zero without writing",
    )
    return parser


def _fail_if_missing_paths(*, repo_root: Path, universe: Sequence[str]) -> None:
    phantom = sorted(name for name in universe if not (repo_root / name).is_file())
    if phantom:
        raise FileNotFoundError(f"manifest names files that do not exist: {phantom}")


def _fail_if_sharder_disagrees(*, manifest_path: Path) -> None:
    """Check the written split against the sharder CI itself calls.

    `regenerate` partitions the universe directly so that `--check` never has to
    write first. Reading the result back through `general_shard_files` is a
    second route to the same assignment: if the two ever disagree, the manifest
    records a split no lane would run.
    """
    ci_workloads.load_manifest.cache_clear()
    for leg, config in ci_workloads.shard_layout()["general"].items():
        for index in range(1, config["shards"] + 1):
            recorded = sorted(
                ci_workloads.files_for(invocation_id=f"general-shard-{index}-{leg}")
            )
            computed = sorted(
                general_shard_files(leg=leg, n_shards=config["shards"], shard=index)
            )
            if recorded != computed:
                raise AssertionError(
                    f"{leg} shard {index} in {manifest_path} disagrees with "
                    "tests.ci.shard_test_files.general_shard_files"
                )


if __name__ == "__main__":
    sys.exit(main())
