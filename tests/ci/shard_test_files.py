"""Assign test modules to stable, dependency-free CI shards.

Two assignment modes live here.

`assign_test_files` is the original stable hash-modulo partition. It needs no
timing manifest and is kept for callers that have no observed weights.

`assign_weighted_test_files` is the duration-weighted partition the CPU-suite
rebalancing uses (implementation plan batch 5): sort the eligible files by
decreasing observed weight, breaking ties on the repository-relative path, then
put each file in the currently lightest bin. Under ``--dist loadfile`` a whole
file is an indivisible atom, so this is a longest-processing-time partition of
atoms, and the largest single file is a lower bound on its bin no matter how
many bins there are. Both properties are what the caller needs to check a bin
against a phase budget, so `weighted_shard_report` returns them alongside the
assignment rather than leaving the caller to recompute them.

Weights are per *leg* (one OS/precision combination), never a cross-leg sum: the
same file costs 37 minutes on the Windows general leg and 27 on the Linux fp64
one. `tests/ci/ci-workloads.json` records them under `leg_weights`, keyed by the
leg name. A file with no observed weight is never treated as free; it is
reported in `unweighted` so the caller can see the gap.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.ci import ci_workloads


def assign_test_files(
    *, files: Iterable[Path], n_shards: int
) -> tuple[tuple[Path, ...], ...]:
    """Partition files exactly once using a stable hash of their repository path."""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")

    groups: list[list[Path]] = [[] for _ in range(n_shards)]
    for path in sorted(files):
        digest = hashlib.sha256(path.as_posix().encode()).digest()
        shard = int.from_bytes(digest[:8], byteorder="big") % n_shards
        groups[shard].append(path)

    return tuple(tuple(group) for group in groups)


def assign_weighted_test_files(
    *, files: Iterable[str], n_shards: int, weights: Mapping[str, float]
) -> tuple[tuple[str, ...], ...]:
    """Partition files into `n_shards` bins by decreasing observed weight.

    The order is fully determined by `(-weight, path)`, and the chosen bin is
    the lowest-indexed one among those with the smallest current load, so the
    result depends only on the inputs -- two machines computing it agree.

    A file absent from `weights` sorts as weight 0.0 *for ordering only*; it is
    still assigned, and `weighted_shard_report` names it as unweighted rather
    than claiming its bin cost nothing.
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")

    groups: list[list[str]] = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    ordered = sorted(files, key=lambda path: (-float(weights.get(path, 0.0)), path))
    for path in ordered:
        lightest = min(range(n_shards), key=lambda index: (loads[index], index))
        groups[lightest].append(path)
        loads[lightest] += float(weights.get(path, 0.0))

    return tuple(tuple(sorted(group)) for group in groups)


@dataclass(frozen=True)
class ShardReport:
    """One bin of a weighted partition and the numbers a budget check needs."""

    shard: int
    files: tuple[str, ...]
    weighted_seconds: float
    largest_file: str | None
    largest_file_seconds: float
    unweighted: tuple[str, ...]

    def payload_minutes(self, *, workers: int) -> float:
        """Predicted payload minutes at `workers` parallel workers.

        `workers` is the xdist worker count; 0 means a single serial process.
        The result is `max(total / workers, largest file)` because ``loadfile``
        cannot split one file across workers -- more workers never bring a bin
        below its largest atom.
        """
        effective = max(workers, 1)
        return max(self.weighted_seconds / effective, self.largest_file_seconds) / 60.0


def weighted_shard_report(
    *, files: Iterable[str], n_shards: int, weights: Mapping[str, float]
) -> tuple[ShardReport, ...]:
    """Return one `ShardReport` per bin of the weighted partition."""
    groups = assign_weighted_test_files(files=files, n_shards=n_shards, weights=weights)
    reports: list[ShardReport] = []
    for index, group in enumerate(groups, start=1):
        weighted = sum(float(weights[f]) for f in group if f in weights)
        unweighted = tuple(f for f in group if f not in weights)
        weighed = [(f, float(weights[f])) for f in group if f in weights]
        largest_file, largest_seconds = (
            max(weighed, key=lambda item: (item[1], item[0]))
            if weighed
            else (None, 0.0)
        )
        reports.append(
            ShardReport(
                shard=index,
                files=group,
                weighted_seconds=weighted,
                largest_file=largest_file,
                largest_file_seconds=largest_seconds,
                unweighted=unweighted,
            )
        )
    return tuple(reports)


def leg_weights(*, leg: str) -> Mapping[str, float]:
    """Return the manifest's observed per-file seconds for one leg."""
    return ci_workloads.leg_weights(leg=leg)


def general_shard_universe() -> tuple[str, ...]:
    """Return the files a general (`notslow`) leg shards over.

    This is the manifest's recorded general-lane selection, which already
    excludes the environment-configured eight-device witnesses (they are kept
    out of shared collection by `ignore_implicit_eight_device_collection` and
    run in their own lanes) and, since the source-contract lane exists, the two
    source-only certificate modules.
    """
    return ci_workloads.general_shard_universe()


def general_shard_files(*, leg: str, n_shards: int, shard: int) -> tuple[str, ...]:
    """Return the general-lane files one shard of one leg selects."""
    if not 1 <= shard <= n_shards:
        raise ValueError("shard must be between 1 and n_shards")
    groups = assign_weighted_test_files(
        files=general_shard_universe(),
        n_shards=n_shards,
        weights=leg_weights(leg=leg),
    )
    return groups[shard - 1]


def ignore_out_of_shard_collection(
    *, collection_path: Path, root: Path, shard_files: frozenset[str]
) -> bool:
    """Return True when this collection candidate belongs to another shard.

    Only *test modules* are judged. A DIRECTORY is never ignored here, even one
    whose name starts with `test_` (`tests/test_models/` is such a directory):
    ignoring it would drop every module beneath it from every shard at once,
    which no shard's own file list would reveal. That is a silent loss of
    coverage in which each shard still passes, so the `.py` check is the
    load-bearing half of this predicate, not a tidiness detail.
    """
    if collection_path.suffix != ".py":
        return False
    if not collection_path.name.startswith("test_"):
        return False
    try:
        relative = collection_path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    name = relative.as_posix()
    return name.startswith("tests/") and name not in shard_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one stable shard of test files, one path per line."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument(
        "--weights-leg",
        default=None,
        help=(
            "Manifest leg whose observed per-file seconds weight the partition "
            "(for example fp64-solution). Omitted, the stable hash partition is "
            "used instead."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the selected one-based shard for consumption by a CI shell."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.shard <= args.shards:
        parser.error("--shard must be between 1 and --shards")

    files = tuple(args.root.rglob("test_*.py"))
    if args.weights_leg is None:
        selected: tuple[str, ...] = tuple(
            path.as_posix()
            for path in assign_test_files(files=files, n_shards=args.shards)[
                args.shard - 1
            ]
        )
    else:
        selected = assign_weighted_test_files(
            files=(path.as_posix() for path in files),
            n_shards=args.shards,
            weights=leg_weights(leg=args.weights_leg),
        )[args.shard - 1]
    if not selected:
        parser.error("selected shard contains no test files")

    for path in selected:
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
