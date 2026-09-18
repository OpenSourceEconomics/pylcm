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
- the `files` list of every `general-shard-<n>-<leg>` invocation is recomputed
  from that universe with the manifest's own weighted partition, so the
  recorded split is the one CI computes at run time.

What it leaves alone: `frozen_head`, the observed weights, the exclusion list,
the guardrail budgets, the shard *counts*, and every non-general lane.
`frozen_head` names the commit the per-file weights were measured at, not the
commit the manifest was last edited at, so registering a file does not advance
it. Re-measuring the weights does.

Exit codes:

- `0` ⇒ the manifest already accounts for every test file (or, without
  `--check`, it was rewritten to)
- `1` ⇒ under `--check`, regeneration would change the committed manifest
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.ci import ci_workloads
from tests.ci.shard_test_files import assign_weighted_test_files, general_shard_files

MANIFEST_PATH = Path(__file__).with_name("ci-workloads.json")
_INDENT = 1


def render(manifest: Mapping[str, Any]) -> str:
    """Render the manifest exactly as the committed file is written.

    The committed file round-trips through `indent=1, ensure_ascii=False`, so a
    regeneration that changes nothing produces a byte-identical file and
    `--check` reports drift only when the content really moved.
    """
    return json.dumps(dict(manifest), indent=_INDENT, ensure_ascii=False) + "\n"


def unaccounted_test_files(
    *, repo_root: Path, manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return test files that no invocation selects and no exclusion names."""
    present = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "tests").rglob("test_*.py")
    }
    selected = {file_ for inv in manifest["invocations"] for file_ in inv["files"]}
    excluded = set(manifest["excluded_files"])
    return tuple(sorted(present - selected - excluded))


def regenerate(
    *, repo_root: Path, manifest: Mapping[str, Any], additions: Sequence[str]
) -> dict[str, Any]:
    """Return the manifest with `additions` registered and the shards re-derived."""
    updated = json.loads(json.dumps(dict(manifest)))
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
        groups = assign_weighted_test_files(
            files=universe, n_shards=config["shards"], weights=weights
        )
        for index, group in enumerate(groups, start=1):
            invocation_id = f"general-shard-{index}-{leg}"
            if invocation_id not in by_id:
                raise KeyError(f"{invocation_id} is not a recorded invocation")
            by_id[invocation_id]["files"] = sorted(group)
    _fail_if_missing_paths(repo_root=repo_root, universe=universe)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    """Register unaccounted test files, or report the drift under `--check`."""
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = repo_root / "tests/ci/ci-workloads.json"
    committed_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(committed_text)

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
