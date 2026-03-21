"""Check whether ASV benchmark results exist for the current PR commits.

Usage:
    python benchmarks/check_results.py [--org-site-dir DIR] [--pr-commits SHA,SHA,...]

Exit codes:
    0 — results are current, stale, or check is skipped
    1 — results are missing for all PR commits on at least one required machine
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_ORG_REPO = "https://github.com/OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_SUBDIR = "pylcm-benchmarks"


def _get_pr_commits() -> list[str]:
    """Return commit SHAs from origin/main..HEAD, newest first."""
    result = subprocess.run(
        ["git", "log", "--format=%H", "origin/main..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.strip().splitlines() if line]


def _clone_org_site(target: Path) -> None:
    """Shallow-clone the org site repo into *target*."""
    subprocess.run(
        ["git", "clone", "--depth", "1", _ORG_REPO, str(target)],
        check=True,
        capture_output=True,
    )


def _get_machine_dirs(results_root: Path) -> list[Path]:
    """Return machine directories under the results root."""
    if not results_root.is_dir():
        return []
    return [p for p in results_root.iterdir() if p.is_dir()]


def _find_result_for_commits(machine_dir: Path, pr_commits: list[str]) -> str | None:
    """Return the first PR commit SHA that has a result file in machine_dir."""
    for sha in pr_commits:
        # ASV stores results as {commit_hash[:8]}-*.json or {commit_hash}.json
        # Check both full and short hash patterns
        matches = list(machine_dir.glob(f"{sha[:8]}*.json"))
        if matches:
            return sha
    return None


def _find_comparison_for_commits(
    machine_dir: Path,
    pr_commits: list[str],
) -> dict[str, Any] | None:
    """Return comparison data for the first PR commit that has a compare file."""
    for sha in pr_commits:
        compare_file = machine_dir / f"{sha[:8]}-compare.json"
        if compare_file.exists():
            return json.loads(compare_file.read_text(encoding="utf-8"))
    return None


def _check_machine(
    machine_dir: Path,
    pr_commits: list[str],
    head_sha: str,
) -> str:
    """Check one machine's results. Return "current", "stale", or "missing"."""
    machine_name = machine_dir.name

    found_sha = _find_result_for_commits(machine_dir, pr_commits)

    if found_sha is None:
        print(f"Machine `{machine_name}`: no results found for any PR commit.")
        return "missing"

    if found_sha == head_sha:
        print(f"Machine `{machine_name}`: results are current (HEAD).")
        return "current"

    short = found_sha[:12]
    print(f"Machine `{machine_name}`: results for `{short}`, not HEAD.")
    result = subprocess.run(
        ["git", "log", "--format=- %h %s", f"{found_sha}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        print(f"  Commits since:\n{result.stdout.rstrip()}")
    return "stale"


def check(
    org_site_dir: Path,
    pr_commits: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """Check benchmark status and print details.

    Returns:
        Tuple of (status, comparison_data) where status is one of "current",
        "stale", "missing", or "skip", and comparison_data is the parsed
        content of a `-compare.json` file (or None if not found).

    """
    results_root = org_site_dir / _SUBDIR / "results"

    if not pr_commits:
        print("No PR commits found — skipping check.")
        return "skip", None

    machine_dirs = _get_machine_dirs(results_root)
    if not machine_dirs:
        print("No machine results found on org site.")
        return "missing", None

    head_sha = pr_commits[0]
    statuses = [_check_machine(d, pr_commits, head_sha) for d in machine_dirs]

    # Find comparison data from any machine
    comparison = None
    for machine_dir in machine_dirs:
        comparison = _find_comparison_for_commits(machine_dir, pr_commits)
        if comparison is not None:
            break

    if "missing" in statuses:
        return "missing", comparison
    if "stale" in statuses:
        return "stale", comparison
    return "current", comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--org-site-dir",
        type=Path,
        default=None,
        help="Path to org site clone (default: temp clone).",
    )
    parser.add_argument(
        "--pr-commits",
        default=None,
        help="Comma-separated SHAs (default: read from git via "
        "`git log origin/main..HEAD`).",
    )
    args = parser.parse_args()

    pr_commits = args.pr_commits.split(",") if args.pr_commits else _get_pr_commits()

    if args.org_site_dir is not None:
        status, comparison = check(args.org_site_dir, pr_commits)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            org_site_dir = Path(tmpdir) / "org-site"
            print(f"Cloning org site to {org_site_dir} ...")
            _clone_org_site(org_site_dir)
            status, comparison = check(org_site_dir, pr_commits)

    print(f"\nStatus: {status}")
    if comparison is not None:
        print(f"Comparison: {json.dumps(comparison)}")
    sys.exit(1 if status == "missing" else 0)


if __name__ == "__main__":
    main()
