"""Assign test modules to stable, dependency-free CI shards."""

import argparse
import hashlib
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one stable shard of test files, one path per line."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the selected one-based shard for consumption by a CI shell."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.shard <= args.shards:
        parser.error("--shard must be between 1 and --shards")

    files = tuple(args.root.rglob("test_*.py"))
    groups = assign_test_files(files=files, n_shards=args.shards)
    selected = groups[args.shard - 1]
    if not selected:
        parser.error("selected shard contains no test files")

    for path in selected:
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
