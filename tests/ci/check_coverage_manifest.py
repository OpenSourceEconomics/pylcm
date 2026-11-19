"""Fail unless every coverage-contributing lane delivered its report.

The CPU suite used to accumulate coverage with `--cov-append` along one `&&`
chain and publish the XML from the chain's last link, so "complete" meant "the
last invocation ran". Splitting the chain into independent jobs destroys that
property: `--cov-append` appends to a file on one runner's disk, and a lane that
never started simply contributes nothing. A partial upload then reads as a
coverage *drop* -- a number that looks like a regression in the code under test
rather than like a missing lane.

This checker replaces that contract with an explicit one. `ci-workloads.json`
records the complete set of contributing lanes; the combine stage downloads one
artifact per lane and this module refuses to let the upload proceed unless every
recorded lane is present and non-empty. An unexpected extra artifact is also an
error: it means a lane was added without being recorded, so nobody decided
whether its coverage belongs in the published number.

`--not-run` names a recorded lane that legitimately ran no tests this run, such
as the source-only certificate campaign on a pull request that cannot change its
verdict. That lane must then deliver nothing: an artifact from it is unrecorded.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tests.ci import ci_workloads


def missing_and_unexpected(
    *, root: Path, expected: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return the missing, empty and unexpected contributor names under `root`.

    A contributor is present when `root/<name>` holds at least one non-empty
    `.xml`. An existing but empty report counts as missing evidence, not as a
    lane that legitimately covered nothing.
    """
    present = {path.name for path in root.iterdir() if path.is_dir()}
    missing = tuple(sorted(set(expected) - present))
    unexpected = tuple(sorted(present - set(expected)))
    empty = tuple(
        sorted(
            name
            for name in sorted(set(expected) & present)
            if not [
                xml for xml in (root / name).rglob("*.xml") if xml.stat().st_size > 0
            ]
        )
    )
    return missing, empty, unexpected


def main(argv: Sequence[str] | None = None) -> int:
    """Check a downloaded coverage-artifact tree against the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--not-run",
        action="append",
        default=[],
        help="A recorded contributor whose lane ran no tests this run.",
    )
    args = parser.parse_args(argv)

    recorded = tuple(ci_workloads.coverage_contributors())
    unknown = sorted(set(args.not_run) - set(recorded))
    if unknown:
        sys.stderr.write(f"--not-run names unrecorded contributors: {unknown}\n")
        return 1
    expected = tuple(name for name in recorded if name not in args.not_run)
    if not args.root.is_dir():
        sys.stderr.write(f"no coverage artifact directory at {args.root}\n")
        return 1

    missing, empty, unexpected = missing_and_unexpected(
        root=args.root, expected=expected
    )
    for label, names in (
        ("missing", missing),
        ("empty", empty),
        ("unrecorded", unexpected),
    ):
        if names:
            sys.stderr.write(f"{label} coverage contributors: {list(names)}\n")
    if missing or empty or unexpected:
        sys.stderr.write(
            f"expected exactly these {len(expected)} contributors: {list(expected)}\n"
        )
        return 1

    sys.stdout.write(f"all {len(expected)} coverage contributors present\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
