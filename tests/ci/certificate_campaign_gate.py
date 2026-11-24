"""Decide whether the source-only certificate campaign runs.

The campaign (`cpu.yml`, job `tests-source-contract`) proves properties of the
certified sources, so its verdict can change only when one of its inputs does:

- a certified source, as `tests/candidate_certificate/sources.json` lists them;
- anything under `tests/candidate_certificate/`;
- a test file the campaign runs, as the workload manifest records them;
- `pixi.lock`, which pins the Python whose `ast` the certificate digests;
- this gate and the workflow that calls it.

A pull request that changes none of them skips the campaign. Every other event
runs it, and so does a pull request whose changed files are unknown.

The job runs this file with the runner's own Python before it installs any
environment, so it imports the standard library only and keeps to Python 3.12
syntax.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_INVOCATION = "source-contract-fp64-linux"
_CERTIFICATE_DIRECTORY = "tests/candidate_certificate/"
_GATE_FILES = (
    ".github/workflows/cpu.yml",
    "tests/ci/certificate_campaign_gate.py",
    "pixi.lock",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Print `run=true` or `run=false` for `$GITHUB_OUTPUT`, the reason to stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="File listing the pull request's changed paths, one per line; "
        "omitted when they could not be computed.",
    )
    args = parser.parse_args(argv)

    if args.event != "pull_request":
        run, reason = True, f"a {args.event} event always runs the campaign"
    elif args.changed_files is None:
        run, reason = True, "the changed files are unknown, so the campaign runs"
    else:
        changed = args.changed_files.read_text(encoding="utf-8").splitlines()
        inputs = _campaign_inputs()
        triggers = sorted(
            name
            for name in changed
            if name in inputs or name.startswith(_CERTIFICATE_DIRECTORY)
        )
        run = bool(triggers)
        reason = (
            f"campaign inputs changed: {' '.join(triggers)}"
            if run
            else f"none of the {len(changed)} changed files is a campaign input"
        )
    sys.stderr.write(f"certificate campaign: {reason}\n")
    sys.stdout.write(f"run={str(run).lower()}\n")
    return 0


def _campaign_inputs() -> frozenset[str]:
    """Return the named files whose change can change the campaign's verdict."""
    inventory = json.loads(
        (_REPO_ROOT / "tests/candidate_certificate/sources.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (_REPO_ROOT / "tests/ci/ci-workloads.json").read_text(encoding="utf-8")
    )
    (invocation,) = [
        entry
        for entry in manifest["invocations"]
        if entry["id"] == _CAMPAIGN_INVOCATION
    ]
    return frozenset(
        [
            *(entry["path"] for entry in inventory["sources"]),
            *invocation["files"],
            *_GATE_FILES,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
